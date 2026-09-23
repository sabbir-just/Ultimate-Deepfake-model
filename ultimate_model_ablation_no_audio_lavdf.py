#!/usr/bin/env python3
"""
================================================================================
 quadkd_offline_pipeline.py

  EfficientNet-B2-based multimodal deepfake detector.
================================================================================

WHAT CHANGED FROM THE ORIGINAL deepfake_kd_pipeline.py
--------------------------------------------------------
This is the redesign described in the accompanying patch guide
(quadkd_offline_kd_patch_guide.md). The teacher checkpoints, teacher
architectures, manifest path, dataset labels, and evaluation metric
definitions are UNCHANGED. What changed:

  1. Three explicit pipeline MODEs -- PRECOMPUTE_TEACHERS / TRAIN_STUDENT /
     EVALUATE -- run as separate processes/sessions. Only one executes per run.
  2. Teachers are run ONCE, offline, via a single infer_all() forward per
     sample, and their outputs are stored in large sharded .pt files
     (ShardedTeacherArtifactStore) instead of one .pt file per sample or a
     repeated live forward every batch.
  3. The audio teacher (audio_lavdf / WavLM+HGAT) now uses a deterministic
     center-crop during precomputation, so it is cacheable for the first time.
  4. The collapsing learned TeacherReliabilityRouter is replaced by a
     deterministic DomainPriorRouter (fixed per-domain weights, normalized
     over present teachers only).
  5. The student's frame path is EfficientNet-B2 (not EfficientFormer-L3 run
     twice); the HF branch is a small dedicated CNN; the temporal branch is a
     small dedicated conv stem + short Transformer over 4 sampled frames (not
     the heavy frame backbone reused over 8 frames).
  6. The curriculum is 3 stages / ~6 epochs (A=warmup, B=multimodal KD,
     C=finetune) instead of 6 stages / 16 epochs.
  7. Sampling is domain-and-class balanced (dfdc/diffusionface/ffpp/lavdf),
     not label-only.
  8. TRAIN_STUDENT never runs evaluation; EVALUATE never trains; a pre-training
     feasibility gate measures real timing (without mutating model state)
     before any long run starts.

Fill in CKPT_PATHS / MANIFEST_PATH below, set MODE, and run the whole file as
one cell/script.
"""

from __future__ import annotations

import subprocess, sys, importlib.util

# Preflight for decord: install it HERE, before torch is ever touched. A
# missing decord doesn't crash anything -- it makes EVERY video-decode call
# silently fail and mark that row's temporal modality "absent."
if importlib.util.find_spec("decord") is None:
    print("[startup] `decord` not found; installing now (no kernel restart needed for this one).")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "decord", "--break-system-packages"])
import decord  # noqa: F401
print(f"[startup] decord {decord.__version__} available.")

# CRITICAL: do NOT wrap `import torch` in try/except and retry in-process.
# torch._C registers pybind11 types as GLOBAL PROCESS STATE the instant it
# loads; a first failed import poisons this kernel process permanently.
if "torch" in sys.modules:
    raise RuntimeError(
        "[startup] `torch` is already present in sys.modules for this kernel process, "
        "which means an earlier `import torch` in this same session already failed "
        "partway. RESTART THE KAGGLE KERNEL (Run > Restart Session / Factory Reset), "
        "then run this cell FIRST, before anything else, in the fresh session."
    )

import torch
print(f"[startup] bare `import torch` OK — torch {torch.__version__}")

# Kaggle/Jupyter containers ship a small /dev/shm. The default 'file_descriptor'
# sharing strategy passes worker tensors through shared memory; with several
# persistent DataLoader workers (frame/temporal/audio precompute) this can fill
# /dev/shm and kill a worker in a way that takes the whole kernel process down
# WITHOUT raising a catchable Python exception -- exactly the untrapped
# "Kernel died while waiting for execute reply" failure seen mid-frame_diff.
# 'file_system' avoids /dev/shm entirely.
import torch.multiprocessing as _mp
try:
    _mp.set_sharing_strategy("file_system")
    print("[startup] multiprocessing sharing strategy set to 'file_system'.")
except RuntimeError as e:
    print(f"[startup] could not set sharing strategy ({e}); continuing with default.")

import timm

try:
    import ctypes, ctypes.util
    _libav = ctypes.CDLL(ctypes.util.find_library("avutil") or "libavutil.so")
    _libav.av_log_set_level(8)   # AV_LOG_FATAL
    print("[startup] ffmpeg/decord log level lowered to FATAL.")
except Exception as e:
    print(f"[startup] could not lower ffmpeg log level ({e}); decoder warnings will still print.")

have_ef = "efficientnet_b2" in timm.list_models()
if not have_ef:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "timm",
                            "--break-system-packages"])
    raise RuntimeError(
        "[startup] timm was just upgraded to add efficientnet_b2 support. "
        "RESTART THE KAGGLE KERNEL now, then re-run this cell in the fresh session."
    )
print(f"[startup] timm {timm.__version__} ready.")

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import io
import gc
import copy
import json
import hashlib
import math as _math
import time
import random
import zipfile
import tempfile
import warnings
import traceback
import contextlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Callable, Any, Set

import numpy as np

warnings.filterwarnings("ignore")


def _free_gpu_memory(tag: str = ""):
    gc.collect()
    if TORCH_OK and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if tag:
            free_b, total_b = torch.cuda.mem_get_info()
            host_str = ""
            try:
                import psutil
                vm = psutil.virtual_memory()
                # GPU memory alone gave no warning before the kernel died mid-frame_diff --
                # it stayed flat at ~14.06 GiB free right up to the crash. Host RAM/shared
                # memory is the more likely exhaustion point for multi-worker DataLoaders,
                # so log it every time so a slow leak is visible in the logs before it takes
                # the kernel down.
                host_str = (f" | host RAM: {vm.available/1e9:.2f} GiB free / "
                            f"{vm.total/1e9:.2f} GiB total ({vm.percent:.1f}% used)")
            except Exception:
                pass
            print(f"[mem] after {tag}: {free_b/1e9:.2f} GiB free / {total_b/1e9:.2f} GiB total{host_str}")


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    TORCH_OK = True
except Exception as e:  # pragma: no cover
    TORCH_OK = False
    print(f"[env] PyTorch not available ({e}); only inspection/manifest-audit "
          f"utilities that don't require torch will run.")

if TORCH_OK:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BF16_OK = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
else:
    DEVICE = "cpu"
    BF16_OK = False

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
if TORCH_OK:
    torch.manual_seed(SEED)


# ══════════════════════════════════════════════════════════════════════════
# 0.  TOP-LEVEL CONFIG
# ══════════════════════════════════════════════════════════════════════════

CKPT_PATHS = dict(
    video_ff    = "/kaggle/input/models/syedazmulhasansabbir/ff-video-teacher-epoch-19/tensorflow2/default/1/checkpoint_epoch_019",
    frame_dfdc  = "/kaggle/input/models/syedazmulhasansabbir/dfdc-kd/tensorflow2/default/1/student_epoch15_auc0.9988.pth",
    frame_diff  = "/kaggle/input/models/syedazmulhasansabbir/kd-diffusion/tensorflow2/default/1/epoch_014.pth",
    audio_lavdf = "/kaggle/input/models/syedazmulhasansabbir/audio-teacher/tensorflow2/default/1/ckpt_epoch001_auc0.9126",
)

TEACHER_ORDER = ["frame_dfdc", "frame_diff", "video_ff"]

TEACHER_BRANCH = {
    "frame_dfdc": "frame",
    "frame_diff": "frame",
    "video_ff": "temporal",
    "audio_lavdf": "audio",
}
BRANCH_ORDER = ["frame", "temporal", "audio"]

# RUN_TAG identifies this specific configuration -- the full model, or a
# leave-one-teacher-out ablation. Every output path below (OOD manifest,
# checkpoints, figures, CSVs, artifact dir) is namespaced under it, so an
# ablation session can NEVER silently overwrite the full model's (or another
# ablation's) checkpoints/results just by reusing this same script. Change
# this -- and restrict TEACHER_ORDER above -- before every ablation session,
# e.g. RUN_TAG = "no_video_ff".
RUN_TAG = "no_audio_lavdf"

MANIFEST_PATH = "/kaggle/input/datasets/syedazmulhasansabbir/ultimate-model-datasset-jsonl-files/manifest.jsonl"

# ---- cross-dataset (OOD) generalization sources (Q1-paper requirement) ----
# Raw dataset roots -- build_ood_manifest() below walks these directly, so
# there's no separate ood_manifest.jsonl to upload as its own Kaggle dataset.
CELEBDF_ROOT = "/kaggle/input/datasets/reubensuju/celeb-df-v2"           # video dataset
WILDDEEPFAKE_ROOT = "/kaggle/input/datasets/maysuni/wild-deepfake/test"  # frame dataset
# Regenerated each EVALUATE session by default (it's a directory walk, not a
# decode, so it's cheap) -- lives under OUT_DIR rather than a Kaggle input.
OOD_MANIFEST_PATH = f"/kaggle/working/kd_{RUN_TAG}/ood_manifest.jsonl"
REBUILD_OOD_MANIFEST_IF_PRESENT = False  # True forces a fresh rebuild even if the file exists
ALLOW_PARTIAL_MULTIMODAL_TRAINING = True
MIN_CALIBRATION_SAMPLES = 200
SMOKE_TEST_MODE = False

# ---- PIPELINE MODE -----------------------------------------------------
# Exactly one of these executes per process/session. TRAIN_STUDENT never
# instantiates a teacher nn.Module and never runs evaluation. EVALUATE never
# trains. PRECOMPUTE_TEACHERS never touches the student.
#   "PRECOMPUTE_TEACHERS" -> run/resume offline teacher inference into shards
#   "TRAIN_STUDENT"       -> train the student against precomputed shards only
#   "EVALUATE"            -> full comprehensive evaluation of a trained student
MODE = "EVALUATE" 
OUT_DIR = Path(f"/kaggle/working/kd_{RUN_TAG}")
CKPT_DIR = OUT_DIR / "checkpoints"
# NOTE for ablation sessions: these three point at the FULL model's saved
# checkpoints. For any RUN_TAG other than the full model's, clear all three
# to "" so this session trains from scratch under its own TEACHER_ORDER --
# the teacher_order guard below rejects a mismatched resume anyway.
RESUME_CKPT_PATH = "/kaggle/input/models/syedazmulhasansabbir/ultimate-model-last-checkpoint-no-audio-lavdf/tensorflow2/default/1/last_checkpoint.pth"
FORCE_RESTART_CURRENT_EPOCH = True
BEST_CKPT_RESUME_INPUT_PATH = "/kaggle/input/models/syedazmulhasansabbir/ultimate-model-best-auc-no-audio-lavdf/tensorflow2/default/1/best_auc.pth"
BEST_EMA_CKPT_RESUME_INPUT_PATH = "/kaggle/input/models/syedazmulhasansabbir/ultimate-model-best-auc-no-audio-lavdf/tensorflow2/default/1/best_auc.pth"
LOAD_EMA_WEIGHTS_FOR_EVAL = True
EMERGENCY_DIR = OUT_DIR / "emergency_ckpts"
FIG_DIR = OUT_DIR / "figures"
CSV_DIR = OUT_DIR / "csv"
for d in (OUT_DIR, CKPT_DIR, EMERGENCY_DIR, FIG_DIR, CSV_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---- offline teacher-artifact store (Part 1/2/3/23) ----
ARTIFACT_DIR = OUT_DIR / "teacher_artifacts"
# /kaggle/working is wiped every fresh session -- exactly the same problem
# RESUME_CKPT_PATH solves for the student checkpoint. Set this to a previously
# saved teacher_artifacts/ directory, mounted as a Kaggle input dataset (after
# a PRECOMPUTE_TEACHERS session, save its teacher_artifacts/ folder as a
# dataset, then point this at the mounted path next session; leave "" to start
# with an empty store). For ablation sessions: leave this pointed at the SAME
# precomputed artifacts every time -- artifacts are stored per TEACHER, not
# per RUN_TAG, so a leave-one-teacher-out ablation reuses the other teachers'
# existing shards without recomputing anything (the excluded teacher is simply
# left unused). Note this DOES mean every RUN_TAG's own ARTIFACT_DIR gets its
# own full copy of all teachers' shards on first restore -- budget disk/time
# for that on each new ablation session.
ARTIFACT_RESUME_INPUT_DIR = "/kaggle/input/notebooks/hafsatalukder/ultimate-model-10/kd_full_kd_ffvideo_v3/teacher_artifacts"
ARTIFACT_SHARD_SIZE = 2048
ARTIFACT_DTYPE_EMBED = "float16"
ARTIFACT_DTYPE_LOGIT = "float32"
PREPROCESSING_VERSION = "v1"
AUDIO_CROP_POLICY = "center_fixed_v1"
PRECOMPUTE_MAX_ROWS_PER_SESSION = None  # int to cap rows/teacher/session; None = no cap

# ---- KD objective config (Part 9/10/14) ----
KD_TEMPERATURE = 4.0
LOSS_WEIGHTS = dict(hard=0.65, logit_kd=0.10, frame_feat=0.15, temporal=0.05, audio=0.20)

# ---- shortened curriculum (Part 11) ----
STAGE_ORDER = ["A", "B", "C"]
STAGE_NAMES_V2 = {"A": "warmup", "B": "multimodal_kd", "C": "student_finetune"}
STAGE_EPOCHS_V2 = {"A": 1, "B": 3, "C": 3}
STAGE_TRAIN_POOL_MAX_V2: Dict[str, int] = {"A": 40000, "B": 60000, "C": 30000}
STAGE_BATCH_SIZE_V2: Dict[str, int] = {"A": 48, "B": 24, "C": 32}
ACTIVE_BRANCHES_BY_STAGE_V2: Dict[str, Set[str]] = {
    "A": {"frame"},
    "B": {"frame", "temporal", "audio"},
    "C": {"frame", "temporal", "audio"},
}

BATCH_SIZE = 24        # fallback default; per-stage overrides above take priority
NUM_WORKERS = 4
BASE_LR = 2e-4

# ---- precompute batch/timeout/worker tuning, PER BRANCH (Part 1/23) ----
# Video decode (temporal) is far heavier per-sample than frame/audio decode -- reusing
# a single fixed batch_size/timeout (e.g. the student's Stage-A batch size of 48, with a
# blanket 180s DataLoader timeout) for every teacher is what causes
# "RuntimeError: DataLoader timed out after 180 seconds" once precompute reaches video_ff.
# These are precompute-only settings; they do not affect STAGE_BATCH_SIZE_V2 (student
# training) at all.
PRECOMPUTE_BATCH_SIZE_BY_BRANCH: Dict[str, int] = {"frame": 64, "temporal": 8, "audio": 16}
PRECOMPUTE_TIMEOUT_SECONDS_BY_BRANCH: Dict[str, int] = {"frame": 180, "temporal": 600, "audio": 300}
PRECOMPUTE_NUM_WORKERS_BY_BRANCH: Dict[str, int] = {"frame": NUM_WORKERS, "temporal": max(1, min(NUM_WORKERS, 2)),
                                                      "audio": max(1, min(NUM_WORKERS, 3))}
PRECOMPUTE_PREFETCH_BY_BRANCH: Dict[str, int] = {"frame": 4, "temporal": 2, "audio": 2}
# On any batch-fetch/forward failure (timeout, CUDA OOM, a corrupt file, a worker crash),
# precompute backs off: halves the batch size, drops a worker, raises the timeout, and
# retries the remaining rows -- down to a floor of batch_size=1, at which point the single
# offending row is skipped permanently (logged) rather than losing the rest of the teacher.

# ---- validation / feasibility (Part 17, Part 22) ----
VAL_SUBSET_SIZE = 6000
FEASIBILITY_MAX_PROJECTED_HOURS = 10.0

# ---- run flags ----
RUN_INSPECTION = True
RUN_ABLATIONS = False     # opt-in, expensive -- Part 20
RUN_EFFICIENCY_TABLE = True # opt-in: param counts + inference latency, student vs. teachers

# Paired ablation significance (Issue 4/7): map ablation name -> path to a .npz
# file (ids/labels/probs) saved by save_raw_predictions_npz() at the end of
# THAT ablation's own EVALUATE run. Leave empty until you have those runs.
# e.g. {"no_video_ff": "/kaggle/input/.../raw_predictions_no_video_ff.npz"}
ABLATION_RAW_PREDICTIONS_PATHS: Dict[str, str] = {
    "no_video_ff": "/kaggle/input/<dataset-holding-the-npz>/raw_predictions_no_video_ff.npz"
}
KAGGLE_BUDGET_HOURS = 12.0
BUDGET_RESERVE_MINUTES = 40.0
RUN_START_TIME = time.time()
HARD_KILL_SAFETY_MINUTES = 10.0
MID_EPOCH_CKPT_SECONDS = 20 * 60.0

COVERAGE_ERROR_BELOW = 0.95
COVERAGE_WARN_BELOW = 0.99


def past_hard_deadline() -> bool:
    return time.time() >= (RUN_START_TIME + KAGGLE_BUDGET_HOURS * 3600.0
                            - HARD_KILL_SAFETY_MINUTES * 60.0)


# ══════════════════════════════════════════════════════════════════════════
# 1.  BUDGET MANAGER
# ══════════════════════════════════════════════════════════════════════════

class BudgetManager:
    """Tracks wall-clock budget so training/precompute never starts a unit of
    work it can't finish, and always leaves time for a clean checkpoint."""

    def __init__(self, total_hours: float = KAGGLE_BUDGET_HOURS,
                 reserve_minutes: float = BUDGET_RESERVE_MINUTES,
                 start_time: float = RUN_START_TIME):
        self.start_time = start_time
        self.total_seconds = total_hours * 3600.0
        self.reserve_seconds = reserve_minutes * 60.0
        self.epoch_durations: List[float] = []

    def elapsed(self) -> float:
        return time.time() - self.start_time

    def remaining(self) -> float:
        return self.total_seconds - self.elapsed()

    def remaining_for_training(self) -> float:
        return self.remaining() - self.reserve_seconds

    def record_epoch(self, duration_s: float):
        self.epoch_durations.append(duration_s)

    def eta_next_epoch(self) -> float:
        if not self.epoch_durations:
            return 0.0
        recent = self.epoch_durations[-3:]
        return float(np.mean(recent))

    def can_start_epoch(self, safety_factor: float = 1.15) -> bool:
        eta = self.eta_next_epoch()
        if eta <= 0:
            return self.remaining_for_training() > 300
        return self.remaining_for_training() > eta * safety_factor

    def should_stop_now(self) -> bool:
        return self.remaining_for_training() <= 0

    def status(self) -> str:
        return (f"[budget] elapsed={self.elapsed()/3600:.2f}h "
                f"remaining={self.remaining()/3600:.2f}h "
                f"remaining_for_training={self.remaining_for_training()/3600:.2f}h "
                f"eta_next_epoch={self.eta_next_epoch()/60:.1f}min")


BUDGET = BudgetManager()

# ══════════════════════════════════════════════════════════════════════════
# 2.  CHECKPOINT INSPECTION / FORENSICS MODE (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════

ZIP_ARCHIVE_MARKERS = {"data.pkl", "version", "byteorder"}


def _rezip_extracted_checkpoint(extracted_dir: Path) -> str:
    archive_name = extracted_dir.name.replace(" ", "_") or "archive"
    fd, out_path = tempfile.mkstemp(suffix=".pth")
    os.close(fd)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for file_path in sorted(extracted_dir.rglob("*")):
            if file_path.is_file():
                arcname = f"{archive_name}/{file_path.relative_to(extracted_dir)}"
                zf.write(file_path, arcname)
    return out_path


def resolve_ckpt_file(path: str) -> str:
    p = Path(path)
    if p.is_file():
        return str(p)
    if not p.exists():
        raise FileNotFoundError(f"[resolve_ckpt_file] path does not exist: {path}")
    if not p.is_dir():
        raise FileNotFoundError(f"[resolve_ckpt_file] not a file or dir: {path}")

    top_files = {c.name for c in p.iterdir() if c.is_file()}
    if ZIP_ARCHIVE_MARKERS.issubset(top_files) and (p / "data").is_dir():
        rezipped = _rezip_extracted_checkpoint(p)
        print(f"[resolve_ckpt_file] {path} is an extracted zip-archive checkpoint; "
              f"re-zipped to {rezipped}")
        return rezipped

    files = [c for c in p.iterdir() if c.is_file()]
    if not files:
        subdirs = [c for c in p.iterdir() if c.is_dir()]
        if len(subdirs) == 1:
            print(f"[resolve_ckpt_file] {path} has no files at this level, only one "
                  f"subdirectory ({subdirs[0].name}); descending into it (Kaggle nested "
                  f"the extracted archive one level deeper than usual).")
            return resolve_ckpt_file(str(subdirs[0]))

    if len(files) == 1:
        print(f"[resolve_ckpt_file] {path} is a directory; resolved to {files[0].name}")
        return str(files[0])

    preferred = [c for c in files if c.suffix in (".pth", ".pt", ".bin", ".ckpt")]
    if len(preferred) == 1:
        print(f"[resolve_ckpt_file] {path} is a directory; resolved to {preferred[0].name}")
        return str(preferred[0])

    same_name = [c for c in files if c.name == p.name]
    if len(same_name) == 1:
        return str(same_name[0])

    raise RuntimeError(
        f"[resolve_ckpt_file] {path} is a directory containing {[c.name for c in files]} "
        f"— ambiguous which file is the real checkpoint. Point CKPT_PATHS at the exact file."
    )


@dataclass
class CheckpointInspectionReport:
    name: str
    path: str
    exists: bool
    checkpoint_type: str = "unknown"
    top_level_keys: List[str] = field(default_factory=list)
    state_dict_key_used: str = ""
    num_tensors: int = 0
    total_param_count: int = 0
    tensor_shapes_sample: Dict[str, Tuple[int, ...]] = field(default_factory=dict)
    key_prefixes_seen: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tensor_shapes_sample"] = {k: list(v) for k, v in self.tensor_shapes_sample.items()}
        return d


def inspect_checkpoint_raw(name: str, path: str) -> CheckpointInspectionReport:
    rep = CheckpointInspectionReport(name=name, path=path, exists=Path(path).exists())
    if not rep.exists:
        rep.error = "file/dir does not exist"
        return rep
    try:
        resolved = resolve_ckpt_file(path)
    except Exception as e:
        rep.error = f"could not resolve to a single file: {e}"
        return rep

    if not TORCH_OK:
        rep.error = "torch not available in this environment"
        return rep

    try:
        raw = torch.load(resolved, map_location="cpu", weights_only=False)
    except Exception as e:
        rep.error = f"torch.load failed: {e}"
        return rep

    if isinstance(raw, dict):
        rep.top_level_keys = list(raw.keys())
        sd = None
        for k in ("model_state_dict", "ema_state", "model_state", "state_dict", "model"):
            if k in raw and isinstance(raw[k], dict):
                sd = raw[k]
                rep.state_dict_key_used = k
                break
        if sd is None:
            if all(hasattr(v, "shape") for v in raw.values()):
                sd = raw
                rep.state_dict_key_used = "<root>"
                rep.checkpoint_type = "raw_state_dict"
            else:
                rep.checkpoint_type = "wrapped_dict_no_state_dict_found"
                rep.notes.append("Top-level dict found but no recognizable state_dict key; "
                                  "manual inspection of top_level_keys required.")
                return rep
        else:
            rep.checkpoint_type = "wrapped_dict"
    elif hasattr(raw, "state_dict"):
        sd = raw.state_dict()
        rep.checkpoint_type = "full_module_pickle"
        rep.notes.append("Checkpoint is a pickled nn.Module, not a state_dict.")
    else:
        rep.error = f"unrecognized checkpoint container type: {type(raw)}"
        return rep

    rep.num_tensors = len(sd)
    total_params = 0
    prefixes = set()
    for i, (k, v) in enumerate(sd.items()):
        try:
            shape = tuple(v.shape)
        except Exception:
            continue
        total_params += int(np.prod(shape)) if shape else 1
        if i < 40:
            rep.tensor_shapes_sample[k] = shape
        prefixes.add(k.split(".")[0])
    rep.total_param_count = total_params
    rep.key_prefixes_seen = sorted(prefixes)

    any_key = next(iter(sd.keys()), "")
    if any_key.startswith("module."):
        rep.notes.append("Keys are prefixed with 'module.' -> saved from DataParallel/DDP; "
                          "strip this prefix before loading into a bare model.")
    if any_key.startswith("_orig_mod."):
        rep.notes.append("Keys are prefixed with '_orig_mod.' -> saved from torch.compile(); "
                          "strip this prefix before loading.")
    if isinstance(raw, dict):
        for meta_key in ("epoch", "val_auc", "auc", "best_auc", "step", "global_step"):
            if meta_key in raw:
                rep.notes.append(f"metadata found: {meta_key}={raw[meta_key]!r}")
    return rep


def print_inspection_report(rep: CheckpointInspectionReport):
    print("=" * 78)
    print(f"CHECKPOINT INSPECTION: {rep.name}")
    print(f"  path:            {rep.path}")
    print(f"  exists:          {rep.exists}")
    if rep.error:
        print(f"  ERROR:           {rep.error}")
        print("=" * 78)
        return
    print(f"  checkpoint_type: {rep.checkpoint_type}")
    print(f"  top_level_keys:  {rep.top_level_keys}")
    print(f"  state_dict_key:  {rep.state_dict_key_used}")
    print(f"  num_tensors:     {rep.num_tensors}")
    print(f"  total_params:    {rep.total_param_count:,}")
    print(f"  key_prefixes:    {rep.key_prefixes_seen[:15]}")
    print(f"  sample shapes (first 40 tensors):")
    for k, shp in list(rep.tensor_shapes_sample.items())[:40]:
        print(f"     {k:60s} {shp}")
    for n in rep.notes:
        print(f"  NOTE: {n}")
    print("=" * 78)


def run_all_checkpoint_inspections() -> Dict[str, CheckpointInspectionReport]:
    print("\n" + "#" * 78)
    print("# STEP 0 — RAW CHECKPOINT FORENSICS (architecture-agnostic)")
    print("#" * 78)
    reports = {}
    for name, path in CKPT_PATHS.items():
        rep = inspect_checkpoint_raw(name, path)
        print_inspection_report(rep)
        reports[name] = rep
        _free_gpu_memory(f"inspecting {name}")
    return reports

# ══════════════════════════════════════════════════════════════════════════
# 3.  TEACHER ARCHITECTURES (frozen, unchanged from original, + infer_all)
# ══════════════════════════════════════════════════════════════════════════

if TORCH_OK:

    TEACHER_EXPECTED = {
        "frame_dfdc":  dict(modality="RGB frame", input_shape="(B,3,224,224)",
                             output_shape="(B,) fake-logit", feature_dim=704),
        "frame_diff":  dict(modality="RGB frame", input_shape="(B,3,224,224)",
                             output_shape="(B,) fake-logit", feature_dim=2816),
        "video_ff":    dict(modality="RGB video clip", input_shape="(B,3,T=16,224,224)",
                             output_shape="(B,) fake-logit", feature_dim=768),
        "audio_lavdf": dict(modality="raw waveform", input_shape="(B, 64000) @16kHz",
                             output_shape="(B,) fake-logit", feature_dim=384),
    }

    # ---- 3.1 FF++ Video Teacher: Video Swin Transformer Tiny ----
    def _build_swin3d_backbone():
        from torchvision.models.video import swin3d_t
        return swin3d_t(weights=None)

    class VideoSwinFF(nn.Module):
        def __init__(self, num_classes: int = 2, dropout: float = 0.45):
            super().__init__()
            self.backbone = _build_swin3d_backbone()
            in_features = self.backbone.head.in_features
            self.backbone.head = nn.Sequential(
                nn.LayerNorm(in_features), nn.Dropout(dropout),
                nn.Linear(in_features, 512), nn.GELU(),
                nn.Dropout(dropout * 0.6),
                nn.Linear(512, 256), nn.GELU(),
                nn.Dropout(dropout * 0.4),
                nn.Linear(256, num_classes),
            )
            self.embed_dim = in_features  # 768

        def _backbone_embedding(self, x: torch.Tensor) -> torch.Tensor:
            m = self.backbone
            x = m.patch_embed(x); x = m.pos_drop(x); x = m.features(x); x = m.norm(x)
            x = x.permute(0, 4, 1, 2, 3); x = m.avgpool(x)
            return torch.flatten(x, 1)

        def embed(self, clip: torch.Tensor) -> torch.Tensor:
            return self._backbone_embedding(clip)

        def embed_sequence(self, clip: torch.Tensor) -> torch.Tensor:
            m = self.backbone
            x = m.patch_embed(clip); x = m.pos_drop(x); x = m.features(x); x = m.norm(x)
            b, t, h, w, c = x.shape
            x = x.mean(dim=(2, 3))
            return x

        def logit(self, clip: torch.Tensor) -> torch.Tensor:
            probs = torch.softmax(self.backbone(clip), dim=1)
            return torch.logit(probs[:, 1].clamp(1e-6, 1 - 1e-6))

        def infer_all(self, clip: torch.Tensor) -> Dict[str, torch.Tensor]:
            """Single-forward-pass API used ONLY by offline precomputation
            (Part 1). Computes logit + embed + seq from ONE backbone pass
            instead of calling .logit()/.embed()/.embed_sequence() separately."""
            m = self.backbone
            x = m.patch_embed(clip); x = m.pos_drop(x); x = m.features(x); x = m.norm(x)
            b, t, h, w, c = x.shape
            seq = x.mean(dim=(2, 3))                      # (B, T', C)
            embed = seq.mean(dim=1)                         # (B, C)
            head_in = x.permute(0, 4, 1, 2, 3)
            head_in = m.avgpool(head_in)
            head_in = torch.flatten(head_in, 1)
            logits2 = m.head(head_in)
            probs = torch.softmax(logits2, dim=1)
            logit = torch.logit(probs[:, 1].clamp(1e-6, 1 - 1e-6))
            return dict(logit=logit, embed=embed, seq=seq)

    # ---- 3.2 DFDC Frame Teacher: dual-path EfficientNet-B2 (RGB+HF) ----
    class LaplacianHF(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            kernel = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]
                                   ).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
            self.register_buffer("kernel", kernel)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            hf = F.conv2d(x, self.kernel, padding=1, groups=3).abs()
            b = hf.size(0)
            lo = hf.view(b, -1).min(1).values.view(b, 1, 1, 1)
            hi = hf.view(b, -1).max(1).values.view(b, 1, 1, 1)
            hf = (hf - lo) / (hi - lo + 1e-6)
            mean = torch.tensor([0.485, 0.456, 0.406], device=hf.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=hf.device).view(1, 3, 1, 1)
            return (hf - mean) / std

    _B2_CH = [24, 48, 120, 352]

    class DualPathB2Student(nn.Module):
        def __init__(self, dropout: float = 0.35):
            import timm
            super().__init__()
            self.rgb_backbone = timm.create_model("efficientnet_b2", pretrained=False,
                                                    features_only=True, out_indices=(1, 2, 3, 4))
            self.hf_backbone = timm.create_model("efficientnet_b2", pretrained=False,
                                                  features_only=True, out_indices=(1, 2, 3, 4))
            feat_dim = _B2_CH[-1]
            fused_dim = feat_dim * 2
            self.hf_gate_logit = nn.Parameter(torch.tensor(-4.6))
            self.fusion = nn.Sequential(
                nn.LayerNorm(fused_dim), nn.Dropout(dropout),
                nn.Linear(fused_dim, 512), nn.GELU(),
                nn.Dropout(dropout / 2), nn.Linear(512, 1),
            )
            self.proj_head = nn.Sequential(nn.LayerNorm(fused_dim), nn.Linear(fused_dim, 1536))
            self.inter_projs = nn.ModuleList([
                nn.Sequential(nn.LayerNorm(ch), nn.Linear(ch, 768)) for ch in _B2_CH
            ])
            self.attn_proj = nn.Conv2d(_B2_CH[2], 1, kernel_size=1, bias=False)
            self.laplacian = LaplacianHF()
            self.embed_dim = fused_dim  # 704

        @staticmethod
        def _gap(feat_map: torch.Tensor) -> torch.Tensor:
            return feat_map.mean(dim=[2, 3])

        def _fuse(self, rgb_pool, hf_pool):
            gate = torch.sigmoid(self.hf_gate_logit)
            return torch.cat([rgb_pool, gate * hf_pool], dim=1)

        def embed(self, rgb: torch.Tensor) -> torch.Tensor:
            hf = self.laplacian(rgb)
            rgb_pool = self._gap(self.rgb_backbone(rgb)[-1])
            hf_pool = self._gap(self.hf_backbone(hf)[-1])
            return self._fuse(rgb_pool, hf_pool)

        def logit(self, rgb: torch.Tensor) -> torch.Tensor:
            return self.fusion(self.embed(rgb)).squeeze(-1)

        def infer_all(self, rgb: torch.Tensor) -> Dict[str, torch.Tensor]:
            hf = self.laplacian(rgb)
            rgb_pool = self._gap(self.rgb_backbone(rgb)[-1])
            hf_pool = self._gap(self.hf_backbone(hf)[-1])
            embed = self._fuse(rgb_pool, hf_pool)
            logit = self.fusion(embed).squeeze(-1)
            return dict(logit=logit, embed=embed)

    # ---- 3.3 DiffusionFace Frame Teacher: EfficientNet-B2 dual path (light) ----
    class LightFusionHead(nn.Module):
        def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.3):
            super().__init__()
            fused_dim = 2 * in_dim
            self.proj = nn.Sequential(
                nn.LayerNorm(fused_dim), nn.Dropout(dropout),
                nn.Linear(fused_dim, hidden), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(hidden, 1),
            )

        def forward(self, f_rgb, f_hf):
            return self.proj(torch.cat([f_rgb, f_hf], dim=1))

    class StudentEfficientNetB2(nn.Module):
        def __init__(self, backbone: str = "efficientnet_b2"):
            import timm
            super().__init__()
            self.rgb_enc = timm.create_model(backbone, pretrained=False, num_classes=0)
            self.hf_enc = timm.create_model(backbone, pretrained=False, num_classes=0)
            feat_dim = self.rgb_enc.num_features  # 1408
            self.head = LightFusionHead(in_dim=feat_dim)
            self.laplacian = LaplacianHF()
            self.embed_dim = feat_dim * 2  # 2816

        def embed(self, rgb: torch.Tensor) -> torch.Tensor:
            hf = self.laplacian(rgb)
            f_rgb = self.rgb_enc(rgb); f_hf = self.hf_enc(hf)
            return torch.cat([f_rgb, f_hf], dim=1)

        def logit(self, rgb: torch.Tensor) -> torch.Tensor:
            hf = self.laplacian(rgb)
            return self.head(self.rgb_enc(rgb), self.hf_enc(hf)).squeeze(-1)

        def infer_all(self, rgb: torch.Tensor) -> Dict[str, torch.Tensor]:
            hf = self.laplacian(rgb)
            f_rgb = self.rgb_enc(rgb)
            f_hf = self.hf_enc(hf)
            embed = torch.cat([f_rgb, f_hf], dim=1)
            logit = self.head(f_rgb, f_hf).squeeze(-1)
            return dict(logit=logit, embed=embed)

    # ---- 3.4 Audio-Visual Teacher: WavLM + HGAT/HSTGAT head ----
    @dataclass
    class AudioCfg:
        wavlm_name: str = "microsoft/wavlm-base-plus"
        hidden_size: int = 768
        graph_hidden: int = 128
        num_graph_heads: int = 4
        num_spectral_bands: int = 16
        dropout: float = 0.2
        use_hst_graph: bool = True

    class AttentiveStatPool(nn.Module):
        def __init__(self, in_dim, hidden_dim):
            super().__init__()
            self.attn = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))

        def forward(self, x):
            w = torch.softmax(self.attn(x).float(), dim=1).to(x.dtype)
            mean = (w * x).sum(dim=1)
            var = (w * (x - mean.unsqueeze(1)).pow(2)).sum(dim=1).clamp(min=1e-6)
            return torch.cat([mean, var.sqrt().clamp(max=1e3)], dim=-1)

    class GraphAttentionLayer(nn.Module):
        def __init__(self, in_dim, out_dim, num_heads, dropout):
            super().__init__()
            self.num_heads = num_heads; self.out_dim = out_dim
            self.W = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.a = nn.Parameter(torch.empty(num_heads, 2 * out_dim))
            nn.init.xavier_uniform_(self.a)
            self.leaky = nn.LeakyReLU(0.2); self.dropout = nn.Dropout(dropout)
            self.norm = nn.LayerNorm(out_dim * num_heads)
            out_total = out_dim * num_heads
            self.residual_proj = None if in_dim == out_total else nn.Linear(in_dim, out_total)

        def forward(self, H):
            B, N, _ = H.shape
            Wh = self.W(H).view(B, N, self.num_heads, self.out_dim)
            src = Wh.unsqueeze(2).expand(-1, -1, N, -1, -1)
            tgt = Wh.unsqueeze(1).expand(-1, N, -1, -1, -1)
            e = self.leaky(torch.einsum("bmnkd,kd->bmnk", torch.cat([src, tgt], dim=-1), self.a)).clamp(-20., 20.)
            alpha = self.dropout(torch.softmax(e.float(), dim=2).to(H.dtype))
            out = F.elu(torch.einsum("bmnk,bnkd->bmkd", alpha, Wh).reshape(B, N, -1))
            out = out + (self.residual_proj(H) if self.residual_proj else H)
            return self.norm(out)

    def _band_energy_features(wav: torch.Tensor, n_bands: int, n_fft: int = 400, hop: int = 160) -> torch.Tensor:
        window = torch.hann_window(n_fft, device=wav.device)
        spec = torch.stft(wav, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
        mag = torch.log1p(spec.abs())
        F_bins = mag.shape[1]
        edges = torch.linspace(0, F_bins, n_bands + 1, device=wav.device).long()
        bands = []
        for i in range(n_bands):
            lo, hi = edges[i].item(), max(edges[i].item() + 1, edges[i + 1].item())
            bands.append(mag[:, lo:hi, :].mean(dim=(1, 2)))
        return torch.stack(bands, dim=1)

    class HSTGATHead(nn.Module):
        NODE_DIM = 128; NUM_HEADS = 4; OUT_PER_HEAD = 32

        def __init__(self, cfg: AudioCfg):
            super().__init__()
            D, G = cfg.hidden_size, self.NODE_DIM
            self.proj_temporal = nn.Linear(D, G)
            self.proj_spectral = nn.Linear(1, G)
            self.proj_global = nn.Linear(D, G)

            def _gat():
                return GraphAttentionLayer(G, self.OUT_PER_HEAD, self.NUM_HEADS, cfg.dropout)

            self.gat1_t, self.gat2_t = _gat(), _gat()
            self.gat1_s, self.gat2_s = _gat(), _gat()
            self.gat1_cross, self.gat2_cross = _gat(), _gat()
            self.stat_pool = AttentiveStatPool(G, G // 2)
            self.num_spectral_bands = cfg.num_spectral_bands

        def forward(self, hidden: torch.Tensor, wav: torch.Tensor) -> torch.Tensor:
            B, T, D = hidden.shape
            global_node = self.proj_global(hidden.mean(dim=1, keepdim=True))
            temporal = self.proj_temporal(hidden[:, ::max(1, T // 64), :])
            bands = _band_energy_features(wav, self.num_spectral_bands)
            spectral = self.proj_spectral(bands.unsqueeze(-1))
            nodes_t = self.gat2_t(self.gat1_t(torch.cat([global_node, temporal], dim=1)))
            nodes_s = self.gat2_s(self.gat1_s(torch.cat([global_node, spectral], dim=1)))
            nodes_cross = self.gat2_cross(self.gat1_cross(torch.cat([nodes_t, nodes_s], dim=1)))
            pooled = self.stat_pool(nodes_cross)
            return torch.cat([pooled, global_node.squeeze(1)], dim=-1)  # (B, 384)

    class HGATHead(nn.Module):
        def __init__(self, cfg: AudioCfg):
            super().__init__()
            D, G, K = cfg.hidden_size, cfg.graph_hidden, cfg.num_graph_heads
            head_dim = G // K
            self.proj_s = nn.Linear(D, G); self.proj_t = nn.Linear(D, G)
            self.gat1 = GraphAttentionLayer(G, head_dim, K, cfg.dropout)
            self.gat2 = GraphAttentionLayer(G, head_dim, K, cfg.dropout)
            self.stat_pool = AttentiveStatPool(G, G // 2)

        def forward(self, hidden, wav):
            B, T, D = hidden.shape
            s = self.proj_s(hidden.mean(dim=1, keepdim=True))
            t = self.proj_t(hidden[:, ::max(1, T // 64), :])
            nodes = self.gat2(self.gat1(torch.cat([s, t], dim=1)))
            return self.stat_pool(nodes)

    class WavLMAAIST(nn.Module):
        def __init__(self, cfg: AudioCfg):
            from transformers import WavLMModel
            super().__init__()
            self.cfg = cfg
            self.wavlm = WavLMModel.from_pretrained(cfg.wavlm_name)
            num_layers = self.wavlm.config.num_hidden_layers + 1
            self.layer_weights = nn.Parameter(torch.ones(num_layers))
            self.head = HSTGATHead(cfg) if cfg.use_hst_graph else HGATHead(cfg)
            total_dim = HSTGATHead.NODE_DIM * 3 if cfg.use_hst_graph else 2 * cfg.graph_hidden
            self.classifier = nn.Sequential(
                nn.Linear(total_dim, total_dim // 2), nn.GELU(),
                nn.Dropout(cfg.dropout), nn.Linear(total_dim // 2, 1),
            )
            self.embed_dim = total_dim

        def _fuse_hidden(self, wav):
            out = self.wavlm(wav, output_hidden_states=True)
            stacked = torch.stack(out.hidden_states, dim=1)
            weights = torch.softmax(self.layer_weights, dim=0)
            return (stacked * weights.view(1, -1, 1, 1)).sum(1)

        def embed(self, wav: torch.Tensor) -> torch.Tensor:
            fused = self._fuse_hidden(wav)
            return self.head(fused, wav)

        def logit(self, wav: torch.Tensor) -> torch.Tensor:
            return self.classifier(self.embed(wav)).squeeze(-1)

        def infer_all(self, wav: torch.Tensor) -> Dict[str, torch.Tensor]:
            fused = self._fuse_hidden(wav)
            embed = self.head(fused, wav)
            logit = self.classifier(embed).squeeze(-1)
            return dict(logit=logit, embed=embed)

    TEACHER_CLASSES = dict(
        frame_dfdc=lambda: DualPathB2Student(),
        frame_diff=lambda: StudentEfficientNetB2(),
        video_ff=lambda: VideoSwinFF(),
        audio_lavdf=lambda: WavLMAAIST(AudioCfg(use_hst_graph=True)),
    )

    # ══════════════════════════════════════════════════════════════════════
    # 4.  ARCHITECTURE-AWARE CHECKPOINT LOADING + COVERAGE VERIFICATION
    #     (unchanged from original)
    # ══════════════════════════════════════════════════════════════════════

    @dataclass
    class LoadReport:
        name: str
        total_tensors: int
        matched: int
        shape_mismatch: int
        missing: int
        unexpected: int
        expected_modality: str = ""
        expected_input_shape: str = ""
        expected_output_shape: str = ""
        feature_dim: Any = None

        @property
        def coverage(self) -> float:
            return self.matched / max(1, self.total_tensors)

        @property
        def status(self) -> str:
            if self.coverage < COVERAGE_ERROR_BELOW:
                return "ERROR"
            if self.coverage < COVERAGE_WARN_BELOW:
                return "WARNING"
            return "OK"

        def __str__(self) -> str:
            return (f"{self.name}\n"
                    f"  expected modality:     {self.expected_modality}\n"
                    f"  expected input shape:  {self.expected_input_shape}\n"
                    f"  expected output shape: {self.expected_output_shape}\n"
                    f"  feature dim:           {self.feature_dim}\n"
                    f"  total tensors (model): {self.total_tensors}\n"
                    f"  matched:                {self.matched}\n"
                    f"  shape mismatch:         {self.shape_mismatch}\n"
                    f"  missing:                {self.missing}\n"
                    f"  unexpected (in ckpt only): {self.unexpected}\n"
                    f"  coverage:               {self.coverage*100:.2f}%  [{self.status}]")

    def _normalize_state_dict_keys(sd: dict, name: str) -> dict:
        out = {}
        for k, v in sd.items():
            nk = k
            nk = nk.replace("_orig_mod.", "")
            if nk.startswith("module."):
                nk = nk[len("module."):]
            if name == "video_ff":
                if nk.startswith("swin."):
                    nk = "backbone." + nk[len("swin."):]
                elif not nk.startswith("backbone."):
                    nk = "backbone." + nk
            out[nk] = v
        return out

    def load_state_verified(model: "nn.Module", ckpt_path: str, name: str,
                             abort_on_error: bool = True) -> LoadReport:
        resolved = resolve_ckpt_file(ckpt_path)
        raw = torch.load(resolved, map_location="cpu", weights_only=False)
        sd = raw
        if isinstance(raw, dict):
            for k in ("model_state_dict", "ema_state", "model_state", "state_dict", "model"):
                if k in raw and isinstance(raw[k], dict):
                    sd = raw[k]
                    break
        sd = _normalize_state_dict_keys(sd, name)

        model_sd = model.state_dict()
        total = len(model_sd)
        matched = shape_mismatch = 0
        shape_mismatch_keys, missing_keys, unexpected_keys = [], [], []
        for k, v in model_sd.items():
            if k in sd:
                if tuple(sd[k].shape) == tuple(v.shape):
                    matched += 1
                else:
                    shape_mismatch += 1
                    shape_mismatch_keys.append((k, tuple(v.shape), tuple(sd[k].shape)))
            else:
                missing_keys.append(k)

        sd_loadable = {k: v for k, v in sd.items()
                       if k not in model_sd or tuple(v.shape) == tuple(model_sd[k].shape)}
        missing, unexpected = model.load_state_dict(sd_loadable, strict=False)

        meta = TEACHER_EXPECTED.get(name, {})
        report = LoadReport(
            name=name, total_tensors=total, matched=matched, shape_mismatch=shape_mismatch,
            missing=len(missing), unexpected=len(unexpected),
            expected_modality=meta.get("modality", "?"),
            expected_input_shape=meta.get("input_shape", "?"),
            expected_output_shape=meta.get("output_shape", "?"),
            feature_dim=meta.get("feature_dim", getattr(model, "embed_dim", None)),
        )
        print(report)
        if report.status in ("ERROR", "WARNING"):
            if shape_mismatch_keys:
                print(f"  [diag] shape mismatches ({len(shape_mismatch_keys)}):")
                for k, ms, cs in shape_mismatch_keys[:20]:
                    print(f"    {k}: model={ms} ckpt={cs}")
            if missing:
                print(f"  [diag] missing (model expects, checkpoint lacks) ({len(missing)}):")
                for k in list(missing)[:20]:
                    print(f"    {k}")
            if unexpected:
                print(f"  [diag] unexpected (checkpoint has, model doesn't use) ({len(unexpected)}):")
                for k in list(unexpected)[:20]:
                    print(f"    {k}")
        if abort_on_error and report.status == "ERROR":
            raise RuntimeError(
                f"[load_state_verified] {name}: coverage {report.coverage*100:.1f}% is below "
                f"the {COVERAGE_ERROR_BELOW*100:.0f}% safety threshold."
            )
        return report

    def load_all_teachers(abort_on_error: bool = True) -> Tuple[Dict[str, "nn.Module"], Dict[str, LoadReport]]:
        print("\n" + "#" * 78)
        print("# LOAD & VERIFY FOUR FROZEN TEACHERS (used only by PRECOMPUTE_TEACHERS mode)")
        print("#" * 78)
        teachers, reports = {}, {}
        for name in TEACHER_ORDER:
            model = TEACHER_CLASSES[name]()
            path = CKPT_PATHS[name]
            if Path(path).exists():
                reports[name] = load_state_verified(model, path, name, abort_on_error=abort_on_error)
            else:
                print(f"[load_all_teachers] WARNING: checkpoint not found for {name} ({path}); "
                      f"model left RANDOMLY INITIALIZED.")
                reports[name] = LoadReport(name=name, total_tensors=len(model.state_dict()),
                                            matched=0, shape_mismatch=0,
                                            missing=len(model.state_dict()), unexpected=0)
            model.to(DEVICE).eval()
            for p in model.parameters():
                p.requires_grad_(False)
            teachers[name] = model
            _free_gpu_memory(f"loading {name}")
        return teachers, reports

    def teachers_are_usable(reports: Dict[str, LoadReport]) -> Dict[str, bool]:
        usable = {}
        for name, rep in reports.items():
            ok = Path(CKPT_PATHS[name]).exists() and rep.status in ("OK", "WARNING")
            usable[name] = ok
            if not ok:
                print(f"[teachers_are_usable] {name}: NOT usable for precomputation "
                      f"(exists={Path(CKPT_PATHS[name]).exists()}, status={rep.status})")
        return usable

    # ══════════════════════════════════════════════════════════════════════
    # 5.  MANIFEST-ROW DATASET (Part 2, Part 13.1: deterministic audio crop +
    #     transforms built once per Dataset instance)
    # ══════════════════════════════════════════════════════════════════════

    class JointDeepfakeDataset(Dataset):
        """
        Expected manifest row schema (one JSON object per line for .jsonl):
          {
            "id": str, "dataset": str, "label": 0|1,
            "frame_path": Optional[str], "video_path": Optional[str],
            "audio_path": Optional[str], "manipulation": Optional[str],
            "split": "train"|"val"|"test"|"ood",
          }
        Any modality whose path is missing/empty is treated as ABSENT.
        """
        def __init__(self, rows: List[dict], split: str = "train", frame_size: int = 224,
                     clip_len: int = 16, audio_len: int = 64000, train: bool = True,
                     active_modalities: Optional[Set[str]] = None,
                     deterministic_audio: bool = False):
            self.rows = rows
            self.frame_size = frame_size
            self.clip_len = clip_len
            self.audio_len = audio_len
            self.train = train
            self.active_modalities = active_modalities
            # True ONLY for the precomputation loader -- forces the fixed-center-crop
            # audio path so audio_lavdf artifacts are reproducible (Part 2).
            self.deterministic_audio = deterministic_audio
            # Build torchvision transforms ONCE per dataset instance (Part 13.1)
            # instead of inside every _load_frame()/_load_middle_video_frame() call.
            import torchvision.transforms as T
            self._frame_tf = T.Compose([
                T.Resize((self.frame_size, self.frame_size)), T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

        def __len__(self):
            return len(self.rows)

        def _load_frame(self, path):
            if path is None or str(path).startswith("synthetic://"):
                return torch.randn(3, self.frame_size, self.frame_size)
            from PIL import Image
            img = Image.open(path).convert("RGB")
            if self.train:
                # Re-encode through JPEG at a random quality (35-95) so any single
                # source-specific compression fingerprint (real vs fake frames saved
                # via different pipelines) can no longer be a reliable, exploitable
                # shortcut -- the model is forced to see the SAME frame across many
                # different compression signatures during training.
                import io as _io
                quality = random.randint(35, 95)
                buf = _io.BytesIO()
                img.save(buf, format="JPEG", quality=quality)
                buf.seek(0)
                img = Image.open(buf).convert("RGB")
            return self._frame_tf(img)

        def _load_clip(self, path):
            if path is None or str(path).startswith("synthetic://"):
                return torch.randn(3, self.clip_len, self.frame_size, self.frame_size), True
            try:
                import decord
                vr = decord.VideoReader(path)
                n = len(vr)
                idx = np.linspace(0, max(0, n - 1), self.clip_len).astype(int)
                frames = vr.get_batch(idx).asnumpy()
                frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0
                frames = F.interpolate(frames.unsqueeze(0), size=(self.clip_len, self.frame_size, self.frame_size),
                                        mode="trilinear", align_corners=False).squeeze(0)
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
                return (frames - mean) / std, True
            except Exception as e:
                print(f"[decode-error] video decode failed for {path}: {e!r}; "
                      f"treating this row's temporal modality as absent.")
                return torch.zeros(3, self.clip_len, self.frame_size, self.frame_size), False

        def _load_middle_video_frame(self, path):
            if path is None or str(path).startswith("synthetic://"):
                return torch.randn(3, self.frame_size, self.frame_size)
            try:
                import decord
                from PIL import Image
                vr = decord.VideoReader(path)
                mid = len(vr) // 2
                frame = vr[mid].asnumpy()
                img = Image.fromarray(frame)
                return self._frame_tf(img)
            except Exception as e:
                print(f"[decode-error] single-frame extraction failed for {path}: {e!r}; "
                      f"returning a zero frame for this row.")
                return torch.zeros(3, self.frame_size, self.frame_size)

        def _load_audio(self, path, deterministic: bool = False):
            """`deterministic=True` (used ONLY by teacher precomputation) always takes
            a fixed CENTER crop (AUDIO_CROP_POLICY = "center_fixed_v1"), so the SAME
            sample id always yields the SAME audio_lavdf embedding (Part 2). Regular
            student-training reads keep the random crop for the student's OWN light
            audio encoder input."""
            if path is None or str(path).startswith("synthetic://"):
                return torch.randn(self.audio_len), True
            try:
                import torchaudio
                wav, sr = torchaudio.load(path)
                wav = wav.mean(0)
                if sr != 16000:
                    wav = torchaudio.functional.resample(wav, sr, 16000)
                if wav.numel() < self.audio_len:
                    wav = F.pad(wav, (0, self.audio_len - wav.numel()))
                else:
                    if deterministic or not self.train:
                        start = max(0, (wav.numel() - self.audio_len) // 2)
                    else:
                        start = random.randint(0, wav.numel() - self.audio_len)
                    wav = wav[start:start + self.audio_len]
                return wav, True
            except Exception as e:
                print(f"[decode-error] audio decode failed for {path}: {e!r}; "
                      f"treating this row's audio modality as absent.")
                return torch.zeros(self.audio_len), False

        def __getitem__(self, i):
            row = self.rows[i]
            allow_temporal = self.active_modalities is None or "temporal" in self.active_modalities
            allow_audio = self.active_modalities is None or "audio" in self.active_modalities
            row_has_frame = _row_has_modality(row, "frame")
            row_has_video = _row_has_modality(row, "temporal")
            row_has_audio = _row_has_modality(row, "audio")

            has_temporal = allow_temporal and row_has_video
            has_audio = allow_audio and row_has_audio

            if has_temporal:
                clip, clip_ok = self._load_clip(row.get("video_path"))
                has_temporal = has_temporal and clip_ok
            else:
                clip = torch.zeros(3, self.clip_len, self.frame_size, self.frame_size)
            if has_audio:
                audio, audio_ok = self._load_audio(row.get("audio_path"),
                                                     deterministic=self.deterministic_audio)
                has_audio = has_audio and audio_ok
            else:
                audio = torch.zeros(self.audio_len)

            derived_frame = False
            if row_has_frame:
                frame = self._load_frame(row.get("frame_path"))
                has_frame = True
            elif has_temporal:
                frame = clip[:, clip.shape[1] // 2].clone()
                derived_frame = True
                has_frame = True
            elif row_has_video and not allow_temporal:
                frame = self._load_middle_video_frame(row.get("video_path"))
                derived_frame = True
                has_frame = True
            else:
                frame = torch.zeros(3, self.frame_size, self.frame_size)
                has_frame = False

            return dict(
                id=row.get("id", str(i)),
                dataset=row.get("dataset", "unknown"),
                label=torch.tensor(float(row.get("label", 0))),
                frame=frame, clip=clip, audio=audio,
                mask_frame=torch.tensor(has_frame),
                mask_temporal=torch.tensor(has_temporal),
                mask_audio=torch.tensor(has_audio),
                manipulation=row.get("manipulation", "unknown"),
            )

    # ══════════════════════════════════════════════════════════════════════
    # 6.  OFFLINE SHARDED TEACHER-ARTIFACT STORE (Part 1/2/3/23)
    #     Replaces the old per-sample TeacherEmbeddingCache entirely.
    # ══════════════════════════════════════════════════════════════════════

    class ShardedTeacherArtifactStore:
        """
        Layout:
            teacher_artifacts/
                <teacher_name>/shard_00000.pt, shard_00001.pt, ...
                index.json      # sample_id -> (teacher_name, shard_idx, row_idx)
                metadata.json   # per-teacher config/version/progress

        Each shard is a torch.save()'d dict:
            {"sample_ids": [...], "logit": FloatTensor[N], "embed": Tensor[N, dim],
             "seq": Tensor[N, T, dim] (video_ff only)}

        Resumable: metadata.json records completed_samples / next_shard_idx per
        teacher, and index.json records exactly which sample ids are already done.
        """
        def __init__(self, root: Path, shard_size: int = ARTIFACT_SHARD_SIZE):
            self.root = root
            self.shard_size = shard_size
            self.root.mkdir(parents=True, exist_ok=True)
            self.index: Dict[str, Dict[str, Any]] = self._load_json(self.root / "index.json", default={})
            self.metadata: Dict[str, Any] = self._load_json(self.root / "metadata.json", default={})
            self._buffers: Dict[str, Dict[str, list]] = {}

        @staticmethod
        def _load_json(path: Path, default):
            if path.exists():
                with open(path) as f:
                    return json.load(f)
            return default

        def _save_index_and_metadata(self):
            tmp_i = self.root / "index.json.tmp"
            with open(tmp_i, "w") as f:
                json.dump(self.index, f)
            os.replace(tmp_i, self.root / "index.json")
            tmp_m = self.root / "metadata.json.tmp"
            with open(tmp_m, "w") as f:
                json.dump(self.metadata, f, indent=2, default=str)
            os.replace(tmp_m, self.root / "metadata.json")

        def teacher_completed_count(self, teacher_name: str) -> int:
            return int(self.metadata.get(teacher_name, {}).get("completed_samples", 0))

        def init_teacher(self, teacher_name: str, feature_dim: int, has_seq: bool,
                          ckpt_path: str, split_seed: int):
            self.metadata.setdefault(teacher_name, {})
            self.metadata[teacher_name].update(dict(
                feature_dim=feature_dim, has_seq=has_seq, checkpoint_path=str(ckpt_path),
                dtype_embed=ARTIFACT_DTYPE_EMBED, dtype_logit=ARTIFACT_DTYPE_LOGIT,
                preprocessing_version=PREPROCESSING_VERSION,
                audio_crop_policy=(AUDIO_CROP_POLICY if teacher_name == "audio_lavdf" else "n/a"),
                seed=split_seed, artifact_version=1,
                completed_samples=self.metadata.get(teacher_name, {}).get("completed_samples", 0),
                next_shard_idx=self.metadata.get(teacher_name, {}).get("next_shard_idx", 0),
            ))
            (self.root / teacher_name).mkdir(parents=True, exist_ok=True)
            self.index.setdefault(teacher_name, {})
            self._buffers.setdefault(teacher_name, dict(sample_ids=[], logit=[], embed=[], seq=[]))

        def append(self, teacher_name: str, sample_id: str, logit: torch.Tensor,
                   embed: torch.Tensor, seq: Optional[torch.Tensor] = None):
            buf = self._buffers[teacher_name]
            buf["sample_ids"].append(sample_id)
            buf["logit"].append(logit.detach().cpu().float())
            embed_dtype = torch.float16 if ARTIFACT_DTYPE_EMBED == "float16" else torch.float32
            buf["embed"].append(embed.detach().cpu().to(embed_dtype))
            if seq is not None:
                buf["seq"].append(seq.detach().cpu().to(embed_dtype))
            if len(buf["sample_ids"]) >= self.shard_size:
                self.flush(teacher_name)

        def flush(self, teacher_name: str):
            buf = self._buffers.get(teacher_name)
            if not buf or not buf["sample_ids"]:
                return
            shard_idx = self.metadata[teacher_name]["next_shard_idx"]
            payload = dict(sample_ids=buf["sample_ids"], logit=torch.stack(buf["logit"]),
                            embed=torch.stack(buf["embed"]))
            if buf["seq"]:
                payload["seq"] = torch.stack(buf["seq"])
            shard_path = self.root / teacher_name / f"shard_{shard_idx:05d}.pt"
            tmp_path = self.root / teacher_name / f"shard_{shard_idx:05d}.pt.tmp"
            torch.save(payload, tmp_path)
            os.replace(tmp_path, shard_path)
            for row_i, sid in enumerate(buf["sample_ids"]):
                self.index[teacher_name][sid] = dict(shard=shard_idx, row=row_i)
            self.metadata[teacher_name]["completed_samples"] += len(buf["sample_ids"])
            self.metadata[teacher_name]["next_shard_idx"] = shard_idx + 1
            self._save_index_and_metadata()
            print(f"[artifacts] {teacher_name}: wrote {shard_path.name} "
                  f"({len(buf['sample_ids'])} samples, "
                  f"{self.metadata[teacher_name]['completed_samples']} total so far)")
            buf["sample_ids"], buf["logit"], buf["embed"], buf["seq"] = [], [], [], []


    class TeacherArtifactReader:
        """Read-side companion, used ONLY by TRAIN_STUDENT / EVALUATE modes.
        Never instantiates a teacher nn.Module."""
        def __init__(self, root: Path):
            self.root = root
            with open(root / "index.json") as f:
                self.index: Dict[str, Dict[str, Any]] = json.load(f)
            with open(root / "metadata.json") as f:
                self.metadata: Dict[str, Any] = json.load(f)
            self._shard_cache: Dict[Tuple[str, int], dict] = {}

        def available_teachers(self) -> List[str]:
            return [n for n in self.index if self.index[n]]

        def has(self, teacher_name: str, sample_id: str) -> bool:
            return sample_id in self.index.get(teacher_name, {})

        def _get_shard(self, teacher_name: str, shard_idx: int) -> dict:
            key = (teacher_name, shard_idx)
            if key not in self._shard_cache:
                path = self.root / teacher_name / f"shard_{shard_idx:05d}.pt"
                self._shard_cache[key] = torch.load(path, map_location="cpu", weights_only=False)
            return self._shard_cache[key]

        def get(self, teacher_name: str, sample_id: str) -> Optional[dict]:
            loc = self.index.get(teacher_name, {}).get(sample_id)
            if loc is None:
                return None
            shard = self._get_shard(teacher_name, loc["shard"])
            row = loc["row"]
            out = dict(logit=shard["logit"][row].float(), embed=shard["embed"][row].float())
            if "seq" in shard:
                out["seq"] = shard["seq"][row].float()
            return out

        def get_batch(self, teacher_name: str, sample_ids: List[str]) -> Optional[Dict[str, torch.Tensor]]:
            items = [self.get(teacher_name, sid) for sid in sample_ids]
            present = torch.tensor([it is not None for it in items], dtype=torch.bool)
            if not present.any():
                return None
            feature_dim = self.metadata[teacher_name]["feature_dim"]
            example = next(it for it in items if it is not None)
            logit = torch.stack([it["logit"] if it is not None else torch.zeros(())
                                  for it in items])
            embed = torch.stack([it["embed"] if it is not None else torch.zeros(feature_dim)
                                  for it in items])
            out = dict(logit=logit, embed=embed, present=present)
            if "seq" in example:
                out["seq"] = torch.stack([it["seq"] if it is not None else torch.zeros_like(example["seq"])
                                           for it in items])
            return out


    # ══════════════════════════════════════════════════════════════════════
    # 7.  BASELINES (unchanged from original -- Part 32: not part of redesign)
    # ══════════════════════════════════════════════════════════════════════

    class SimpleLateFusion(nn.Module):
        def forward(self, teacher_logits: Dict[str, torch.Tensor], present_mask: Dict[str, torch.Tensor]):
            names = list(teacher_logits.keys())
            stacked = torch.stack([teacher_logits[n] for n in names], dim=1)
            mask = torch.stack([present_mask[n].float() for n in names], dim=1)
            summed = (stacked * mask).sum(1)
            count = mask.sum(1).clamp(min=1)
            return summed / count

    class HierarchicalGatedFusion(nn.Module):
        def __init__(self, embed_dims: Dict[str, int], hidden: int = 512, dropout: float = 0.3):
            super().__init__()
            self.names = list(embed_dims.keys())
            self.proj = nn.ModuleDict({n: nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden))
                                        for n, d in embed_dims.items()})
            self.gate = nn.Sequential(nn.Linear(hidden * len(self.names), 256), nn.GELU(),
                                       nn.Linear(256, len(self.names)))
            self.classifier = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout), nn.Linear(hidden, 1))

        def forward(self, embeds: Dict[str, torch.Tensor], present_mask: Dict[str, torch.Tensor]):
            projected = {n: self.proj[n](embeds[n]) for n in self.names}
            stacked = torch.stack([projected[n] for n in self.names], dim=1)
            cat = torch.cat([projected[n] for n in self.names], dim=1)
            gate_logits = self.gate(cat)
            mask = torch.stack([present_mask[n].float() for n in self.names], dim=1)
            gate_logits = gate_logits.masked_fill(mask < 0.5, float("-inf"))
            weights = torch.softmax(gate_logits, dim=1)
            weights = torch.nan_to_num(weights, nan=1.0 / len(self.names))
            fused = (weights.unsqueeze(-1) * stacked).sum(1)
            logit = self.classifier(fused).squeeze(-1)
            return logit, weights

    # ══════════════════════════════════════════════════════════════════════
    # 8.  STUDENT — EfficientNet-B2 based multimodal detector (Part 4/5/6/7)
    # ══════════════════════════════════════════════════════════════════════

    def _build_student_rgb_backbone():
        """Student RGB backbone (Part 4). ONE EfficientNet-B2 pass -- was
        EfficientFormer-L3 run TWICE per sample (RGB + a second full pass on the
        HF map). pretrained=True: this backbone previously trained from random
        init in ~7 total epochs, an order of magnitude too little to learn
        general visual features from scratch -- starting from ImageNet weights
        gives it working low-level features immediately, leaving the short
        curriculum to specialize on forgery cues rather than basic vision."""
        import timm
        m = timm.create_model("efficientnet_b2", pretrained=True, num_classes=0)
        return m, "efficientnet_b2"

    class LightHFBranch(nn.Module):
        """Lightweight high-frequency/forensic branch (Part 4). Small
        depthwise-separable CNN instead of a second full backbone pass on the
        Laplacian-HF map."""
        def __init__(self, out_dim: int = 256):
            super().__init__()
            def dw_block(c_in, c_out, stride):
                return nn.Sequential(
                    nn.Conv2d(c_in, c_in, 3, stride=stride, padding=1, groups=c_in, bias=False),
                    nn.BatchNorm2d(c_in), nn.GELU(),
                    nn.Conv2d(c_in, c_out, 1, bias=False),
                    nn.BatchNorm2d(c_out), nn.GELU(),
                )
            self.net = nn.Sequential(
                dw_block(3, 32, 2), dw_block(32, 64, 2), dw_block(64, 128, 2), dw_block(128, 192, 2),
                nn.AdaptiveAvgPool2d(1),
            )
            self.proj = nn.Linear(192, out_dim)
            self.out_dim = out_dim

        def forward(self, hf: torch.Tensor) -> torch.Tensor:
            x = self.net(hf).flatten(1)
            return self.proj(x)

    class LightTemporalModule(nn.Module):
        """Lightweight temporal encoder (Part 5). SEPARATE tiny conv stem (not
        the RGB backbone) run on `n_sampled_frames` frames (default 4), followed
        by a small Transformer over that short sequence."""
        def __init__(self, hidden: int = 384, n_layers: int = 2, n_heads: int = 4,
                     clip_len: int = 16, n_sampled_frames: int = 4):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.GELU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.GELU(),
                nn.Conv2d(128, hidden, 3, stride=2, padding=1), nn.BatchNorm2d(hidden), nn.GELU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.pos_embed = nn.Parameter(torch.randn(1, clip_len, hidden) * 0.02)
            layer = nn.TransformerEncoderLayer(hidden, n_heads, hidden * 4, dropout=0.1, batch_first=True)
            self.temporal_encoder = nn.TransformerEncoder(layer, n_layers)
            self.out_dim = hidden
            self.clip_len = clip_len
            self.n_sampled_frames = min(n_sampled_frames, clip_len)
            idx = torch.linspace(0, clip_len - 1, self.n_sampled_frames).round().long()
            self.register_buffer("sample_idx", idx, persistent=False)

        def forward(self, clip: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            B, C, T, H, W = clip.shape
            idx = self.sample_idx.to(clip.device)
            clip_sub = clip.index_select(2, idx)               # (B, C, Tsub, H, W)
            Tsub = clip_sub.shape[2]
            frames = clip_sub.permute(0, 2, 1, 3, 4).reshape(B * Tsub, C, H, W)
            feats = self.stem(frames).flatten(1).view(B, Tsub, -1)
            feats = feats + self.pos_embed[:, idx, :]
            seq = self.temporal_encoder(feats)
            pooled = seq.mean(dim=1)
            return pooled, seq

    class LightAudioEncoder(nn.Module):
        """Compact log-mel CNN audio encoder (Part 6, unchanged idea)."""
        def __init__(self, out_dim: int = 512, n_mels: int = 64):
            super().__init__()
            import torchaudio
            self.melspec = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=400,
                                                                  hop_length=160, n_mels=n_mels)
            self.amp_to_db = torchaudio.transforms.AmplitudeToDB()
            self.net = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1, stride=2), nn.BatchNorm2d(32), nn.GELU(),
                nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.BatchNorm2d(64), nn.GELU(),
                nn.Conv2d(64, 128, 3, padding=1, stride=2), nn.BatchNorm2d(128), nn.GELU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.proj = nn.Linear(128, out_dim)
            self.out_dim = out_dim

        def forward(self, wav: torch.Tensor) -> torch.Tensor:
            mel = self.amp_to_db(self.melspec(wav)).unsqueeze(1)
            mel = (mel - mel.mean(dim=(2, 3), keepdim=True)) / (mel.std(dim=(2, 3), keepdim=True) + 1e-5)
            x = self.net(mel).flatten(1)
            return self.proj(x)

    class CrossModalFusion(nn.Module):
        """Lightweight cross-modal fusion (Part 7, unchanged). Absent branches
        masked OUT of attention rather than zero-filled-and-trusted."""
        def __init__(self, dim: int = 512, n_heads: int = 8, dropout: float = 0.1):
            super().__init__()
            self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
            self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
            self.norm = nn.LayerNorm(dim)
            self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim))
            self.norm2 = nn.LayerNorm(dim)

        def forward(self, tokens: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
            B = tokens.size(0)
            q = self.query.expand(B, -1, -1)
            out, _ = self.attn(q, tokens, tokens, key_padding_mask=key_padding_mask)
            out = self.norm(out + q)
            out = self.norm2(out + self.ffn(out))
            return out.squeeze(1)

    class MultimodalStudent(nn.Module):
        """
        THE FINAL DEPLOYABLE MODEL (Part 4/28). EfficientNet-B2 main RGB backbone;
        LightHFBranch for the forensic/HF cue; LightTemporalModule (own conv stem,
        4 sampled frames) for temporal; LightAudioEncoder (unchanged) for audio;
        CrossModalFusion (unchanged) to fuse. Still genuinely multimodal: frame /
        temporal / audio / joint representations, each independently KD-targetable.
        """
        def __init__(self, fusion_dim: int = 512, clip_len: int = 16, dropout: float = 0.2,
                     n_sampled_frames: int = 4):
            super().__init__()
            backbone, backbone_name = _build_student_rgb_backbone()
            self.backbone_name = backbone_name
            self.frame_backbone = backbone
            frame_feat_dim = getattr(backbone, "num_features", fusion_dim)

            self.laplacian = LaplacianHF()
            self.hf_branch = LightHFBranch(out_dim=256)
            self.frame_proj = nn.Sequential(nn.LayerNorm(frame_feat_dim + self.hf_branch.out_dim),
                                             nn.Linear(frame_feat_dim + self.hf_branch.out_dim, fusion_dim))

            self.temporal_module = LightTemporalModule(hidden=fusion_dim, clip_len=clip_len,
                                                         n_sampled_frames=n_sampled_frames)
            self.audio_encoder = LightAudioEncoder(out_dim=fusion_dim)

            self.fusion = CrossModalFusion(dim=fusion_dim)
            self.branch_type_embed = nn.Parameter(torch.randn(3, fusion_dim) * 0.02)

            self.classifier = nn.Sequential(nn.LayerNorm(fusion_dim), nn.Dropout(dropout), nn.Linear(fusion_dim, 1))
            self.frame_only_classifier = nn.Linear(fusion_dim, 1)
            self.embed_dim = fusion_dim

        def frame_repr(self, frame: torch.Tensor) -> torch.Tensor:
            rgb_feat = self.frame_backbone(frame)           # ONE EfficientNet-B2 pass
            hf = self.laplacian(frame)
            hf_feat = self.hf_branch(hf)                     # small dedicated CNN, not the backbone
            return self.frame_proj(torch.cat([rgb_feat, hf_feat], dim=1))

        def forward(self, batch: dict, active_branches: Optional[Set[str]] = None) -> dict:
            if active_branches is None:
                active_branches = {"frame", "temporal", "audio"}

            frame = batch["frame"]; clip = batch["clip"]; audio = batch["audio"]
            mask_frame = batch["mask_frame"].bool()
            mask_temporal = batch["mask_temporal"].bool()
            mask_audio = batch["mask_audio"].bool()
            B = frame.shape[0]
            device = frame.device

            frame_rep = self.frame_repr(frame)

            if "temporal" in active_branches and mask_temporal.any():
                idx = mask_temporal.nonzero(as_tuple=True)[0]
                pooled_sub, seq_sub = self.temporal_module(clip.index_select(0, idx))
                temporal_pooled = torch.zeros(B, pooled_sub.shape[-1], device=device, dtype=pooled_sub.dtype)
                temporal_seq = torch.zeros(B, seq_sub.shape[1], seq_sub.shape[2], device=device, dtype=seq_sub.dtype)
                temporal_pooled[idx] = pooled_sub
                temporal_seq[idx] = seq_sub
                temporal_active_mask = mask_temporal
            else:
                temporal_pooled = torch.zeros(B, self.temporal_module.out_dim, device=device, dtype=frame_rep.dtype)
                temporal_seq = torch.zeros(B, self.temporal_module.n_sampled_frames, self.temporal_module.out_dim,
                                            device=device, dtype=frame_rep.dtype)
                temporal_active_mask = torch.zeros(B, dtype=torch.bool, device=device)

            if "audio" in active_branches and mask_audio.any():
                idx = mask_audio.nonzero(as_tuple=True)[0]
                audio_sub = self.audio_encoder(audio.index_select(0, idx))
                audio_rep = torch.zeros(B, audio_sub.shape[-1], device=device, dtype=audio_sub.dtype)
                audio_rep[idx] = audio_sub
                audio_active_mask = mask_audio
            else:
                audio_rep = torch.zeros(B, self.audio_encoder.out_dim, device=device, dtype=frame_rep.dtype)
                audio_active_mask = torch.zeros(B, dtype=torch.bool, device=device)

            tokens = torch.stack([frame_rep, temporal_pooled, audio_rep], dim=1) + self.branch_type_embed.unsqueeze(0)
            present = torch.stack([mask_frame, temporal_active_mask, audio_active_mask], dim=1)
            present = present | (present.sum(1, keepdim=True) == 0)
            key_padding_mask = ~present

            joint_rep = self.fusion(tokens, key_padding_mask)
            joint_logit = self.classifier(joint_rep).squeeze(-1)

            return dict(
                frame_rep=frame_rep, temporal_rep=temporal_pooled, temporal_seq=temporal_seq,
                audio_rep=audio_rep, joint_rep=joint_rep, joint_logit=joint_logit,
                frame_logit=self.frame_only_classifier(frame_rep).squeeze(-1),
            )

    # ══════════════════════════════════════════════════════════════════════
    # 9.  TEACHER TEMPERATURE CALIBRATION (Part 19: reads offline artifacts)
    # ══════════════════════════════════════════════════════════════════════

    def _nullcontext():
        return contextlib.nullcontext()

    def _to_device(batch: dict) -> dict:
        out = {}
        for k, v in batch.items():
            out[k] = v.to(DEVICE, non_blocking=True) if isinstance(v, torch.Tensor) else v
        return out

    def _teacher_input_mask(name: str, batch: dict) -> torch.Tensor:
        branch = TEACHER_BRANCH[name]
        return {"frame": batch["mask_frame"], "temporal": batch["mask_temporal"],
                "audio": batch["mask_audio"]}[branch].bool()

    class TemperatureScaler:
        def __init__(self, name: str):
            self.name = name
            self.T = 1.0

        def fit(self, logits: np.ndarray, labels: np.ndarray, min_samples: int = MIN_CALIBRATION_SAMPLES) -> float:
            n = len(logits)
            if n < min_samples:
                raise RuntimeError(
                    f"[TemperatureScaler:{self.name}] calibration set has only {n} samples "
                    f"(< MIN_CALIBRATION_SAMPLES={min_samples})."
                )
            logits_t = torch.tensor(logits, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.float32)
            T = nn.Parameter(torch.ones(1))
            opt = torch.optim.LBFGS([T], lr=0.05, max_iter=100)

            def closure():
                opt.zero_grad()
                scaled = logits_t / T.clamp(min=0.05)
                loss = F.binary_cross_entropy_with_logits(scaled, labels_t)
                loss.backward()
                return loss

            opt.step(closure)
            self.T = float(T.clamp(min=0.05).item())
            print(f"[TemperatureScaler:{self.name}] fit on n={n} samples -> T={self.T:.4f}")
            return self.T

        def apply(self, logits: torch.Tensor) -> torch.Tensor:
            return logits / self.T

    def load_teacher_temperatures_from_artifacts(artifact_reader: "TeacherArtifactReader",
                                                  calib_loader: "DataLoader") -> Dict[str, TemperatureScaler]:
        """Replaces the old live-teacher calibrate_all_teachers() for TRAIN_STUDENT/
        EVALUATE (Part 19): fits each TemperatureScaler from precomputed artifact
        logits looked up by sample id -- no teacher nn.Module forward at all."""
        print("\n" + "#" * 78)
        print("# TEACHER TEMPERATURE CALIBRATION (from precomputed artifacts)")
        print("#" * 78)
        logits_accum = {n: [] for n in TEACHER_ORDER}
        labels_accum = {n: [] for n in TEACHER_ORDER}
        for batch in calib_loader:
            ids = batch["id"] if isinstance(batch["id"], list) else list(batch["id"])
            labels = batch["label"].numpy()
            for name in TEACHER_ORDER:
                if name not in artifact_reader.available_teachers():
                    continue
                hit = artifact_reader.get_batch(name, ids)
                if hit is None:
                    continue
                present = hit["present"].numpy()
                if present.sum() == 0:
                    continue
                logits_accum[name].append(hit["logit"].numpy()[present])
                labels_accum[name].append(labels[present])

        scalers: Dict[str, TemperatureScaler] = {}
        for name in TEACHER_ORDER:
            if not logits_accum[name]:
                print(f"[calibrate] {name}: skipped (no artifacts or no aligned samples).")
                continue
            logits = np.concatenate(logits_accum[name])
            labels = np.concatenate(labels_accum[name])
            scaler = TemperatureScaler(name)
            try:
                scaler.fit(logits, labels)
                scalers[name] = scaler
            except RuntimeError as e:
                print(str(e))
                if not IS_SYNTHETIC_MANIFEST:
                    raise
                print(f"[calibrate] {name}: SYNTHETIC manifest -- proceeding UNCALIBRATED (T=1.0).")
                scalers[name] = TemperatureScaler(name)
        save_json_static(OUT_DIR / "teacher_temperatures.json", {k: v.T for k, v in scalers.items()})
        return scalers


    # ══════════════════════════════════════════════════════════════════════
    # 10. DETERMINISTIC DOMAIN-PRIOR TEACHER WEIGHTING (Part 8)
    # ══════════════════════════════════════════════════════════════════════

    class DomainPriorRouter:
        """
        Deterministic replacement for the old learned/collapsing router. No
        trainable parameters. weight[i,k] = fixed per-domain prior for teacher k,
        normalized over PRESENT teachers only for sample i.
        """
        def __init__(self, teacher_names: List[str], domain_prior: Dict[str, Dict[str, float]]):
            self.teacher_names = teacher_names
            self.domain_prior = domain_prior
            self._weight_log: List[dict] = []

        def prior_matrix(self, datasets: List[str], device) -> torch.Tensor:
            rows = []
            for ds in datasets:
                prior = self.domain_prior.get(ds, {n: 1.0 for n in self.teacher_names})
                rows.append([prior.get(n, 1.0) for n in self.teacher_names])
            return torch.tensor(rows, dtype=torch.float32, device=device)

        def __call__(self, present_mask: torch.Tensor, datasets: List[str],
                      log_epoch: Optional[int] = None) -> torch.Tensor:
            device = present_mask.device
            prior = self.prior_matrix(datasets, device)
            weights = prior * present_mask.float()
            denom = weights.sum(1, keepdim=True).clamp(min=1e-8)
            weights = weights / denom
            if log_epoch is not None:
                with torch.no_grad():
                    for i, ds in enumerate(datasets):
                        row = {n: float(weights[i, k]) for k, n in enumerate(self.teacher_names)}
                        row.update(dataset=ds, epoch=log_epoch)
                        self._weight_log.append(row)
            return weights

        def flush_weight_log(self, path: Path):
            if not self._weight_log:
                return
            import pandas as pd
            df = pd.DataFrame(self._weight_log)
            grouped = df.groupby(["epoch", "dataset"])[self.teacher_names].mean().reset_index()
            grouped.to_csv(path, index=False)
            print(f"[router] wrote average per-dataset teacher weights to {path}")
            self._weight_log.clear()
    
    def build_default_domain_prior() -> Dict[str, Dict[str, float]]:
        """Reliability-weighted per-domain teacher affinity (Part 8, revised).
        Base weights come from each teacher's MEASURED test-set AUC from the
        EVALUATE-mode teacher_agreement.csv of a prior run (frame_dfdc=0.691,
        frame_diff=0.796, video_ff=0.568 [near-random], audio_lavdf=0.896), so a
        teacher that is empirically unreliable (video_ff) can no longer dominate
        its "native" domain (ffpp) the way a flat 3x domain boost allowed."""
        RELIABILITY = dict(frame_dfdc=0.7, frame_diff=1.0, video_ff=0.3, audio_lavdf=1.6)
    
        def boosted(primary, factor=1.5):
            d = dict(RELIABILITY)
            d[primary] = d[primary] * factor
            return d
    
        return {
            "dfdc": boosted("frame_dfdc"),
            "diffusionface": boosted("frame_diff"),
            "ffpp": boosted("video_ff"),
            "lavdf": boosted("audio_lavdf"),
            "unknown": dict(RELIABILITY),
        }


    # ══════════════════════════════════════════════════════════════════════
    # 11. KD LOSSES (Part 9/10: normalized frame-KD, standard temperature)
    # ══════════════════════════════════════════════════════════════════════

    def safe_bce_logits(logit: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        logit = torch.nan_to_num(logit, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20, 20)
        loss = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
        if weight is not None:
            loss = loss * weight
        return loss.mean() if loss.numel() > 0 else torch.tensor(0.0, device=logit.device)

    def kd_logit_loss(student_logit: torch.Tensor, teacher_logit: torch.Tensor,
                       mask: torch.Tensor, kd_temp: float = KD_TEMPERATURE,
                       sample_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Standard temperature-scaled binary KD (Part 10). kd_temp defaults to
        the named, configured KD_TEMPERATURE (4.0) instead of an undocumented
        magic constant."""
        if mask.sum() == 0:
            return torch.tensor(0.0, device=student_logit.device)
        s_logit = (student_logit[mask] / kd_temp).clamp(-20, 20)
        t = torch.sigmoid((teacher_logit[mask].detach() / kd_temp).clamp(-20, 20)).clamp(1e-6, 1 - 1e-6)
        per_sample = F.binary_cross_entropy_with_logits(s_logit, t, reduction="none")
        if sample_weight is None:
            return per_sample.mean()
        w = sample_weight[mask].clamp(min=0.0)
        denom = w.sum().clamp(min=1e-6)
        return (per_sample * w).sum() / denom

    def kd_feature_loss(student_feat: torch.Tensor, teacher_feat: torch.Tensor,
                         proj: nn.Module, mask: torch.Tensor) -> torch.Tensor:
        if mask.sum() == 0:
            return torch.tensor(0.0, device=student_feat.device)
        s = F.normalize(student_feat[mask], dim=-1)
        t = F.normalize(proj(teacher_feat[mask].detach()), dim=-1)
        return (1 - (s * t).sum(-1)).mean()

    def relational_kd_loss(student_feat: torch.Tensor, teacher_feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.sum() < 2:
            return torch.tensor(0.0, device=student_feat.device)
        s = F.normalize(student_feat[mask], dim=-1)
        t = F.normalize(teacher_feat[mask].detach(), dim=-1)
        s_dist = torch.cdist(s, s, p=2)
        t_dist = torch.cdist(t, t, p=2)
        s_mean = s_dist[s_dist > 0].mean().clamp(min=1e-6)
        t_mean = t_dist[t_dist > 0].mean().clamp(min=1e-6)
        return F.smooth_l1_loss(s_dist / s_mean, t_dist / t_mean)

    def temporal_kd_loss(student_seq: torch.Tensor, teacher_seq: torch.Tensor,
                          proj: nn.Module, mask: torch.Tensor) -> torch.Tensor:
        if mask.sum() == 0:
            return torch.tensor(0.0, device=student_seq.device)
        s = student_seq[mask]
        t = teacher_seq[mask].detach()
        Ts, Tt = s.shape[1], t.shape[1]
        idx_s = torch.linspace(0, Ts - 1, min(Ts, 8)).long()
        idx_t = torch.linspace(0, Tt - 1, min(Tt, 8)).long()
        s_tok = proj(s[:, idx_s, :])
        t_tok = t[:, idx_t, :]
        n = min(s_tok.shape[1], t_tok.shape[1])
        s_tok = F.normalize(s_tok[:, :n], dim=-1)
        t_tok = F.normalize(t_tok[:, :n], dim=-1)
        return (1 - (s_tok * t_tok).sum(-1)).mean()

    def gate_regularization_loss(weights: torch.Tensor, present_mask: torch.Tensor,
                                  target_entropy_frac: float = 0.7) -> torch.Tensor:
        n_present = present_mask.float().sum(1).clamp(min=1)
        max_entropy = torch.log(n_present)
        entropy = -(weights.clamp(min=1e-8) * weights.clamp(min=1e-8).log()).sum(1)
        target = target_entropy_frac * max_entropy
        return F.relu(target - entropy).mean()

    @dataclass
    class LossWeights:
        hard: float = 1.0
        logit_kd: float = 0.0
        frame_feat: float = 0.0
        temporal: float = 0.0
        audio: float = 0.0
        rkd: float = 0.0
        gate: float = 0.0

    # Simplified 3-stage curriculum (Part 10/11). rkd/gate/SWA/learned-routing are
    # removed from the PRIMARY objective per Part 10 -- available as opt-in extras
    # by setting rkd/gate > 0 in a custom LossWeights, but off by default.
    STAGE_SCHEDULE_V2: Dict[str, LossWeights] = {
        "A": LossWeights(hard=1.0),
        "B": LossWeights(hard=LOSS_WEIGHTS["hard"], logit_kd=LOSS_WEIGHTS["logit_kd"],
                          frame_feat=LOSS_WEIGHTS["frame_feat"], temporal=LOSS_WEIGHTS["temporal"],
                          audio=LOSS_WEIGHTS["audio"]),
        "C": LossWeights(hard=1.0),
    }

    # ══════════════════════════════════════════════════════════════════════
    # 12. EMA + CSV/metric helpers (unchanged from original)
    # ══════════════════════════════════════════════════════════════════════

    class EMA:
        def __init__(self, model: "nn.Module", decay: float = 0.999):
            self.decay = decay
            self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

        @torch.no_grad()
        def update(self, model: "nn.Module"):
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
                else:
                    self.shadow[k] = v.detach().clone()

        def state_dict(self):
            return self.shadow

        def load_state_dict(self, sd):
            self.shadow = {k: v.clone() for k, v in sd.items()}

        def copy_to(self, model: "nn.Module"):
            model.load_state_dict(self.shadow, strict=True)

    def pd_safe_write_csv(rows: List[dict], path: Path, append: bool = False):
        if not rows:
            return
        import pandas as pd
        df = pd.DataFrame(rows)
        if append and path.exists():
            df.to_csv(path, mode="a", header=False, index=False)
        else:
            df.to_csv(path, index=False)

    def save_json(path, obj):
        with open(path, "w") as f:
            json.dump(obj, f, indent=2, default=str)

    def compute_binary_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
        from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                                      balanced_accuracy_score, precision_score, recall_score,
                                      f1_score, matthews_corrcoef, brier_score_loss, log_loss,
                                      confusion_matrix)
        out: Dict[str, float] = {}
        labels = np.asarray(labels).astype(int)
        probs = np.clip(np.asarray(probs).astype(float), 1e-7, 1 - 1e-7)
        preds = (probs >= threshold).astype(int)
        try:
            out["auc"] = roc_auc_score(labels, probs) if len(set(labels)) > 1 else float("nan")
        except Exception:
            out["auc"] = float("nan")
        try:
            out["pr_auc"] = average_precision_score(labels, probs) if len(set(labels)) > 1 else float("nan")
        except Exception:
            out["pr_auc"] = float("nan")
        out["accuracy"] = accuracy_score(labels, preds)
        out["balanced_accuracy"] = balanced_accuracy_score(labels, preds)
        out["precision"] = precision_score(labels, preds, zero_division=0)
        out["recall_sensitivity"] = recall_score(labels, preds, zero_division=0)
        out["f1"] = f1_score(labels, preds, zero_division=0)
        out["mcc"] = matthews_corrcoef(labels, preds) if len(set(preds)) > 1 else 0.0
        try:
            tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
            out["specificity"] = tn / max(1, (tn + fp))
        except Exception:
            out["specificity"] = float("nan")
        out["eer"] = compute_eer(labels, probs)
        out["brier"] = brier_score_loss(labels, probs)
        out["log_loss"] = log_loss(labels, probs, labels=[0, 1])
        out["ece"] = compute_ece(labels, probs)
        return out

    def compute_eer(labels, probs, n_thresh: int = 200) -> float:
        labels = np.asarray(labels); probs = np.asarray(probs)
        if len(set(labels)) < 2:
            return float("nan")
        thresholds = np.linspace(0, 1, n_thresh)
        pos = probs[labels == 1]; neg = probs[labels == 0]
        fars, frrs = [], []
        for th in thresholds:
            far = (neg >= th).mean() if len(neg) else 0.0
            frr = (pos < th).mean() if len(pos) else 0.0
            fars.append(far); frrs.append(frr)
        fars = np.array(fars); frrs = np.array(frrs)
        idx = np.argmin(np.abs(fars - frrs))
        return float((fars[idx] + frrs[idx]) / 2)

    def compute_ece(labels, probs, n_bins: int = 15) -> float:
        labels = np.asarray(labels); probs = np.asarray(probs)
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            lo, hi = bins[i], bins[i + 1]
            mask = (probs >= lo) & (probs < hi if i < n_bins - 1 else probs <= hi)
            if mask.sum() == 0:
                continue
            conf = probs[mask].mean()
            acc = labels[mask].mean()
            ece += (mask.sum() / len(probs)) * abs(acc - conf)
        return float(ece)

    def bootstrap_ci(labels, probs, ids: Optional[List[str]] = None, metric_fn=None,
                      n_boot: int = 1000, alpha: float = 0.05, seed: int = SEED):
        metric_fn = metric_fn or (lambda l, p: compute_binary_metrics(l, p).get("auc", float("nan")))
        rng = np.random.RandomState(seed)
        labels = np.asarray(labels); probs = np.asarray(probs)
        stats = []
        if ids is not None:
            ids = np.asarray(ids)
            unique_ids = np.unique(ids)
            id_to_idx = {u: np.where(ids == u)[0] for u in unique_ids}
            for _ in range(n_boot):
                sampled = rng.choice(unique_ids, size=len(unique_ids), replace=True)
                idxs = np.concatenate([id_to_idx[u] for u in sampled])
                stats.append(metric_fn(labels[idxs], probs[idxs]))
        else:
            n = len(labels)
            for _ in range(n_boot):
                idxs = rng.randint(0, n, n)
                stats.append(metric_fn(labels[idxs], probs[idxs]))
        stats = np.array([s for s in stats if s == s])
        if len(stats) == 0:
            return float("nan"), float("nan"), float("nan")
        lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return float(np.mean(stats)), float(lo), float(hi)

    def delong_test(labels, probs_a, probs_b) -> float:
        labels = np.asarray(labels); probs_a = np.asarray(probs_a); probs_b = np.asarray(probs_b)
        n_pos = int(labels.sum()); n_neg = len(labels) - n_pos
        if n_pos == 0 or n_neg == 0:
            return float("nan")

        def midrank(x):
            J = np.argsort(x)
            Z = x[J]
            n = len(x)
            T = np.zeros(n)
            i = 0
            while i < n:
                j = i
                while j < n and Z[j] == Z[i]:
                    j += 1
                T[i:j] = 0.5 * (i + j - 1) + 1
                i = j
            T2 = np.empty(n)
            T2[J] = T
            return T2

        idx_sorted = np.argsort(-labels, kind="mergesort")
        labels_s = labels[idx_sorted]; a = probs_a[idx_sorted]; b = probs_b[idx_sorted]

        def structural_components(probs):
            pos = probs[:n_pos]; neg = probs[n_pos:]
            tx = midrank(pos); ty = midrank(neg); tz = midrank(probs)
            v01 = (tz[:n_pos] - tx) / n_neg
            v10 = 1.0 - (tz[n_pos:] - ty) / n_pos
            auc = tz[:n_pos].sum() / (n_pos * n_neg) - (n_pos + 1) / (2.0 * n_neg)
            return v01, v10, auc

        v01_a, v10_a, auc_a = structural_components(a)
        v01_b, v10_b, auc_b = structural_components(b)
        s01 = np.cov(np.vstack([v01_a, v01_b])) / n_pos
        s10 = np.cov(np.vstack([v10_a, v10_b])) / n_neg
        S = s01 + s10
        var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
        if var <= 0:
            return float("nan")
        z = (auc_a - auc_b) / np.sqrt(var)
        from scipy.stats import norm
        return float(2 * (1 - norm.cdf(abs(z))))

    def mcnemar_test(labels, preds_a, preds_b) -> Tuple[float, int, int]:
        from statsmodels.stats.contingency_tables import mcnemar
        labels = np.asarray(labels); preds_a = np.asarray(preds_a); preds_b = np.asarray(preds_b)
        correct_a = (preds_a == labels); correct_b = (preds_b == labels)
        b = int((correct_a & (~correct_b)).sum())
        c = int(((~correct_a) & correct_b).sum())
        table = [[0, b], [c, 0]]
        try:
            result = mcnemar(table, exact=(b + c < 25))
            return float(result.pvalue), b, c
        except Exception:
            return float("nan"), b, c

    # ══════════════════════════════════════════════════════════════════════
    # 13. KD TRAINER — 3-stage curriculum, artifact-backed teacher access,
    #     domain-prior routing, single end-of-run EMA eval (Part 11/16/17/18/19)
    # ══════════════════════════════════════════════════════════════════════

    class EpochAbortedForBudget(Exception):
        pass

    class KDTrainer:
        ARCHITECTURE_VERSION = "multimodal_student_v6_jpeg_augmented"  # bump on any student-shape change

        def __init__(self, student: "nn.Module", artifact_reader: "TeacherArtifactReader",
                     usable: Dict[str, bool], scalers: Dict[str, TemperatureScaler],
                     router: "DomainPriorRouter", feature_projections: "nn.ModuleDict",
                     temporal_proj: "nn.Module", train_rows: List[dict], val_loader: "DataLoader",
                     budget: BudgetManager):
            # No teacher nn.Module is held by the trainer at all (Part 3/16): every
            # teacher output comes from `artifact_reader`, a plain tensor lookup
            # against the offline shard store.
            self.student = student.to(DEVICE)
            self.artifact_reader = artifact_reader
            self.usable = usable
            self.scalers = scalers
            self.router = router  # DomainPriorRouter is NOT an nn.Module -- no .to(DEVICE)
            self.feature_projections = feature_projections
            self.temporal_proj = temporal_proj
            self.train_rows = train_rows
            self.train_loader: Optional["DataLoader"] = None
            self.val_loader = val_loader
            self.budget = budget

            params = (list(self.student.parameters()) +
                      list(self.feature_projections.parameters()) + list(self.temporal_proj.parameters()))
            self.optimizer = torch.optim.AdamW(params, lr=BASE_LR, weight_decay=0.05)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=max(sum(STAGE_EPOCHS_V2.values()), 5))
            self.scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
            self.ema = EMA(self.student, decay=0.999)

            self.global_epoch = 0
            self.stage = "A"
            self.epoch_in_stage = 0
            self.best_auc = -1.0
            self.best_ema_auc = -1.0
            self.history: List[dict] = []

            self.batch_in_epoch = 0
            self._last_ckpt_wall_time = time.time()
            self._force_restart_applied = False
            self._maybe_resume()
            # Fast-forward past any stage that already completed before the last save.
            while self.stage is not None and self.epoch_in_stage >= STAGE_EPOCHS_V2[self.stage]:
                print(f"[resume] stage {self.stage} ({STAGE_NAMES_V2[self.stage]}) was already "
                      f"complete ({self.epoch_in_stage}/{STAGE_EPOCHS_V2[self.stage]} epochs); "
                      f"advancing.")
                self.epoch_in_stage = 0
                idx = STAGE_ORDER.index(self.stage)
                self.stage = STAGE_ORDER[idx + 1] if idx + 1 < len(STAGE_ORDER) else None
            if self.stage is not None:
                self._rebuild_train_loader_for_stage()

        # ---------------- checkpoint / resume ----------------
        def _ckpt_payload(self) -> dict:
            return dict(
                architecture_version=self.ARCHITECTURE_VERSION,
                run_tag=RUN_TAG,
                teacher_order=list(TEACHER_ORDER),
                student=self.student.state_dict(),
                router_domain_prior=self.router.domain_prior,
                feature_projections=self.feature_projections.state_dict(),
                temporal_proj=self.temporal_proj.state_dict(),
                optimizer=self.optimizer.state_dict(),
                scheduler=self.scheduler.state_dict(),
                scaler=self.scaler.state_dict(),
                ema=self.ema.state_dict(),
                stage=self.stage, epoch_in_stage=self.epoch_in_stage, global_epoch=self.global_epoch,
                batch_in_epoch=self.batch_in_epoch,
                best_auc=self.best_auc, best_ema_auc=self.best_ema_auc,
                force_restart_applied=self._force_restart_applied,
                teacher_temperatures={k: v.T for k, v in self.scalers.items()},
                rng_state=dict(python=random.getstate(), numpy=np.random.get_state(),
                                torch=torch.get_rng_state(),
                                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
                config=dict(BATCH_SIZE=BATCH_SIZE, BASE_LR=BASE_LR, SEED=SEED,
                            KD_TEMPERATURE=KD_TEMPERATURE, LOSS_WEIGHTS=LOSS_WEIGHTS,
                            STAGE_EPOCHS_V2=STAGE_EPOCHS_V2,
                            STAGE_TRAIN_POOL_MAX_V2=STAGE_TRAIN_POOL_MAX_V2,
                            STAGE_BATCH_SIZE_V2=STAGE_BATCH_SIZE_V2,
                            artifact_version=1, preprocessing_version=PREPROCESSING_VERSION),
            )

        def save_checkpoint(self, tag: str, keep_tagged: bool = False, student_override: "nn.Module" = None):
            payload = self._ckpt_payload()
            if student_override is not None:
                payload["student"] = student_override.state_dict()
            tmp_path = CKPT_DIR / "last_checkpoint.pth.tmp"
            torch.save(payload, tmp_path)
            os.replace(tmp_path, CKPT_DIR / "last_checkpoint.pth")
            if keep_tagged:
                path = CKPT_DIR / f"{tag}.pth"
                tmp2 = CKPT_DIR / f"{tag}.pth.tmp"
                torch.save(payload, tmp2)
                os.replace(tmp2, path)
                print(f"[checkpoint] saved {path}")
            else:
                print(f"[checkpoint] updated last_checkpoint.pth "
                      f"(stage={self.stage} epoch_in_stage={self.epoch_in_stage} "
                      f"batch_in_epoch={self.batch_in_epoch}, tag={tag})")
            self._last_ckpt_wall_time = time.time()

        def save_emergency(self, reason: str):
            path = EMERGENCY_DIR / f"emergency_{int(time.time())}.pth"
            payload = self._ckpt_payload()
            payload["emergency_reason"] = reason
            torch.save(payload, path)
            tmp_path = CKPT_DIR / "last_checkpoint.pth.tmp"
            torch.save(payload, tmp_path)
            os.replace(tmp_path, CKPT_DIR / "last_checkpoint.pth")
            print(f"[checkpoint] EMERGENCY checkpoint saved: {path} (reason: {reason})")
            self._last_ckpt_wall_time = time.time()

        def _rebuild_train_loader_for_stage(self):
            self.train_loader = build_stage_train_loader(self.train_rows, self.stage)
            n_batches = len(self.train_loader)
            if self.batch_in_epoch >= n_batches:
                if self.batch_in_epoch > 0:
                    print(f"[resume] batch_in_epoch={self.batch_in_epoch} doesn't fit this "
                          f"stage's rebuilt loader ({n_batches} batches/epoch); resetting to 0.")
                self.batch_in_epoch = 0

        def _maybe_resume(self):
            last = CKPT_DIR / "last_checkpoint.pth"
            if not last.exists() and RESUME_CKPT_PATH and Path(RESUME_CKPT_PATH).exists():
                import shutil
                resolved_resume = resolve_ckpt_file(RESUME_CKPT_PATH)
                shutil.copy2(resolved_resume, last)
                print(f"[resume] copied external checkpoint {RESUME_CKPT_PATH} "
                      f"(resolved to {resolved_resume}) -> {last}")
            if not last.exists():
                print("[resume] no checkpoint found; starting fresh from Stage A.")
                return
            print(f"[resume] found {last}; resuming...")
            payload = torch.load(last, map_location=DEVICE, weights_only=False)
            ckpt_arch = payload.get("architecture_version")
            if ckpt_arch != self.ARCHITECTURE_VERSION:
                raise RuntimeError(
                    f"[resume] checkpoint architecture_version={ckpt_arch!r} does not match "
                    f"current {self.ARCHITECTURE_VERSION!r}. Refusing to resume from an "
                    f"incompatible (old) student checkpoint (Part 26). Move/rename "
                    f"{last} aside to start fresh."
                )
            ckpt_teachers = payload.get("teacher_order")
            if ckpt_teachers is not None and ckpt_teachers != list(TEACHER_ORDER):
                raise RuntimeError(
                    f"[resume] checkpoint was trained with teacher_order={ckpt_teachers} "
                    f"(run_tag={payload.get('run_tag')!r}) but the current session's "
                    f"TEACHER_ORDER={list(TEACHER_ORDER)} (RUN_TAG={RUN_TAG!r}). Resuming would "
                    f"silently mix two different KD configurations. Refusing to resume. Point "
                    f"RESUME_CKPT_PATH at a checkpoint from the SAME ablation, or clear it to "
                    f"start this configuration from scratch."
                )
            self.student.load_state_dict(payload["student"])
            self.feature_projections.load_state_dict(payload["feature_projections"])
            self.temporal_proj.load_state_dict(payload["temporal_proj"])
            self.optimizer.load_state_dict(payload["optimizer"])
            self.scheduler.load_state_dict(payload["scheduler"])
            self.scaler.load_state_dict(payload["scaler"])
            self.ema.load_state_dict(payload["ema"])
            self.stage = payload["stage"]
            self.epoch_in_stage = payload["epoch_in_stage"]
            self.global_epoch = payload["global_epoch"]
            self.batch_in_epoch = payload.get("batch_in_epoch", 0)
            already_applied = payload.get("force_restart_applied", False)
            if FORCE_RESTART_CURRENT_EPOCH and not already_applied and self.batch_in_epoch > 0:
                print(f"[resume] one-time reset: batch_in_epoch {self.batch_in_epoch} -> 0.")
                self.batch_in_epoch = 0
            self._force_restart_applied = True
            self.best_auc = payload.get("best_auc", -1.0)
            self.best_ema_auc = payload.get("best_ema_auc", -1.0)
            rng = payload.get("rng_state")
            if rng:
                random.setstate(rng["python"]); np.random.set_state(rng["numpy"])
                torch_rng = rng["torch"]
                torch.set_rng_state(torch_rng.cpu() if torch.is_tensor(torch_rng) else torch_rng)
                if rng.get("cuda") and torch.cuda.is_available():
                    cuda_rng_states = [s.cpu() if torch.is_tensor(s) else s for s in rng["cuda"]]
                    torch.cuda.set_rng_state_all(cuda_rng_states)
            print(f"[resume] resumed at stage={self.stage} ({STAGE_NAMES_V2.get(self.stage)}) "
                  f"epoch_in_stage={self.epoch_in_stage} global_epoch={self.global_epoch} "
                  f"batch_in_epoch={self.batch_in_epoch}")

        # ---------------- teacher access (Part 1/3/16/19) ----------------
        def _teacher_forward(self, batch: dict) -> dict:
            """Reads precomputed teacher outputs for this batch's sample ids from
            the offline artifact store. NO teacher nn.Module is ever instantiated
            or forwarded here."""
            out = {}
            ids = batch["id"] if isinstance(batch["id"], list) else list(batch["id"])
            for name in TEACHER_ORDER:
                if not self.usable.get(name, False):
                    continue
                if name not in self.artifact_reader.available_teachers():
                    continue
                hit = self.artifact_reader.get_batch(name, ids)
                if hit is None:
                    continue
                out[name] = dict(logit=hit["logit"].to(DEVICE), embed=hit["embed"].to(DEVICE),
                                  present=hit["present"].to(DEVICE))
                if "seq" in hit:
                    out[name]["seq"] = hit["seq"].to(DEVICE)
            return out

        # ---------------- one training/eval step ----------------
        def _step(self, batch: dict, loss_w: LossWeights) -> Tuple[Optional[torch.Tensor], dict]:
            batch = _to_device(batch)
            needs_teachers = any([loss_w.logit_kd > 0, loss_w.frame_feat > 0, loss_w.temporal > 0,
                                   loss_w.audio > 0, loss_w.rkd > 0, loss_w.gate > 0])
            teacher_out = {}
            if needs_teachers:
                teacher_out = self._teacher_forward(batch)

            amp_dtype = torch.bfloat16 if BF16_OK else torch.float16
            amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=torch.cuda.is_available())
            active_branches = ACTIVE_BRANCHES_BY_STAGE_V2.get(self.stage, {"frame", "temporal", "audio"})
            with amp_ctx:
                student_out = self.student(batch, active_branches=active_branches)
                label = batch["label"]

                loss_hard = safe_bce_logits(student_out["joint_logit"], label)
                components = dict(hard=float(loss_hard.detach().cpu()))
                total = loss_w.hard * loss_hard

                present_flags = []
                if needs_teachers:
                    for name in TEACHER_ORDER:
                        if name not in teacher_out:
                            present_flags.append(torch.zeros(label.shape[0], dtype=torch.bool, device=DEVICE))
                            continue
                        present_flags.append(teacher_out[name]["present"])

                if needs_teachers:
                    present_mat = torch.stack(present_flags, dim=1)
                    weights = self.router(present_mat, batch["dataset"], log_epoch=self.global_epoch)
                else:
                    weights = torch.zeros(label.shape[0], len(TEACHER_ORDER), device=DEVICE)
                    present_mat = torch.zeros(label.shape[0], len(TEACHER_ORDER), dtype=torch.bool, device=DEVICE)

                if loss_w.logit_kd > 0:
                    lk_num = torch.tensor(0.0, device=DEVICE)
                    lk_denom = torch.tensor(1e-6, device=DEVICE)
                    for i, name in enumerate(TEACHER_ORDER):
                        if name not in teacher_out:
                            continue
                        t = teacher_out[name]
                        if not t["present"].any():
                            continue
                        # Per-teacher T = its OWN calibration temperature (already fitted to
                        # make this teacher's probabilities honest). Previously this was
                        # divided by KD_TEMPERATURE a SECOND time on top of calibration,
                        # so how much signal each teacher contributed depended on how
                        # mis-calibrated its raw logit happened to be, not on LOSS_WEIGHTS.
                        T_i = self.scalers[name].T if name in self.scalers else KD_TEMPERATURE
                        s_logit = (student_out["joint_logit"][t["present"]] / T_i).clamp(-20, 20)
                        t_prob = torch.sigmoid((t["logit"][t["present"]].detach() / T_i).clamp(-20, 20)
                                                ).clamp(1e-6, 1 - 1e-6)
                        per_sample = F.binary_cross_entropy_with_logits(s_logit, t_prob, reduction="none")
                        # Hinton et al. 2015: soft-target gradients shrink by ~1/T^2 relative
                        # to hard-label gradients; multiply by T^2 so LOSS_WEIGHTS["logit_kd"]
                        # means what it says relative to LOSS_WEIGHTS["hard"].
                        per_sample = per_sample * (T_i ** 2)
                        w_i = weights[:, i][t["present"]].clamp(min=0.0)
                        lk_num = lk_num + (per_sample * w_i).sum()
                        lk_denom = lk_denom + w_i.sum()
                    lk = lk_num / lk_denom
                    total = total + loss_w.logit_kd * lk
                    components["logit_kd"] = float(lk.detach().cpu())

                if loss_w.frame_feat > 0:
                    # MEAN over present frame teachers (Part 9), not a raw sum -- avoids
                    # structurally ~2x-weighting the frame family relative to temporal/audio.
                    ff_terms = []
                    for name in ("frame_dfdc", "frame_diff"):
                        if name not in teacher_out or name not in self.feature_projections:
                            continue
                        t = teacher_out[name]
                        proj = self.feature_projections[name]
                        if t["present"].any():
                            ff_terms.append(kd_feature_loss(student_out["frame_rep"], t["embed"], proj, t["present"]))
                    ff = torch.stack(ff_terms).mean() if ff_terms else torch.tensor(0.0, device=DEVICE)
                    total = total + loss_w.frame_feat * ff
                    components["frame_feat"] = float(ff.detach().cpu())

                if loss_w.temporal > 0 and "video_ff" in teacher_out:
                    t = teacher_out["video_ff"]
                    tl = temporal_kd_loss(student_out["temporal_seq"], t["seq"], self.temporal_proj, t["present"])
                    total = total + loss_w.temporal * tl
                    components["temporal"] = float(tl.detach().cpu())

                if loss_w.audio > 0 and "audio_lavdf" in teacher_out and "audio_lavdf" in self.feature_projections:
                    t = teacher_out["audio_lavdf"]
                    proj = self.feature_projections["audio_lavdf"]
                    al = kd_feature_loss(student_out["audio_rep"], t["embed"], proj, t["present"])
                    total = total + loss_w.audio * al
                    components["audio"] = float(al.detach().cpu())

                if loss_w.rkd > 0 and teacher_out:
                    ens = torch.zeros(label.shape[0], self.student.embed_dim, device=DEVICE)
                    denom = torch.zeros(label.shape[0], 1, device=DEVICE)
                    for i, name in enumerate(TEACHER_ORDER):
                        if name not in teacher_out or name not in self.feature_projections:
                            continue
                        t = teacher_out[name]
                        proj = self.feature_projections[name]
                        contrib = proj(t["embed"]) * weights[:, i:i + 1]
                        ens = ens + contrib * t["present"].float().unsqueeze(1)
                        denom = denom + t["present"].float().unsqueeze(1) * weights[:, i:i + 1]
                    ens = ens / denom.clamp(min=1e-6)
                    any_present = present_mat.any(dim=1)
                    rkd = relational_kd_loss(student_out["joint_rep"], ens, any_present)
                    total = total + loss_w.rkd * rkd
                    components["rkd"] = float(rkd.detach().cpu())

                if loss_w.gate > 0:
                    gl = gate_regularization_loss(weights, present_mat)
                    total = total + loss_w.gate * gl
                    components["gate"] = float(gl.detach().cpu())

            if not torch.isfinite(total):
                print(f"[safety] non-finite total loss; skipping batch. components={components}")
                self.optimizer.zero_grad(set_to_none=True)
                return None, dict(components=components, weights=[0.0] * len(TEACHER_ORDER))

            return total, dict(components=components, weights=weights.detach().mean(0).cpu().tolist())

        def _optimizer_step(self, loss: torch.Tensor) -> float:
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            params = [p for g in self.optimizer.param_groups for p in g["params"] if p.grad is not None]
            grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            if not torch.isfinite(grad_norm):
                print(f"[safety] non-finite grad norm ({grad_norm}); skipping optimizer step.")
                self.optimizer.zero_grad(set_to_none=True)
                return float("nan")
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.ema.update(self.student)
            return float(grad_norm)

        # ---------------- stage / epoch loop ----------------
        def run(self):
            print("\n" + "#" * 78)
            print("# STAGED MULTIMODAL KNOWLEDGE DISTILLATION (offline-artifact-backed)")
            print("#" * 78)
            while self.stage is not None:
                stage_epochs = STAGE_EPOCHS_V2[self.stage]
                loss_w = STAGE_SCHEDULE_V2[self.stage]
                print(f"\n=== STAGE {self.stage} [{STAGE_NAMES_V2[self.stage]}] "
                      f"epochs {self.epoch_in_stage + 1}..{stage_epochs} "
                      f"(resuming mid-epoch at batch {self.batch_in_epoch}) ===")
                while self.epoch_in_stage < stage_epochs:
                    if self.budget.should_stop_now():
                        print("[budget] out of time; saving emergency checkpoint and stopping.")
                        self.save_emergency("budget_exhausted")
                        return

                    t0 = time.time()
                    try:
                        train_metrics = self._run_epoch(loss_w, train=True)
                    except EpochAbortedForBudget:
                        print("[budget] training epoch aborted mid-way; checkpoint already saved. "
                              f"Re-run to resume from batch {self.batch_in_epoch} of stage "
                              f"{self.stage} epoch_in_stage {self.epoch_in_stage}.")
                        return

                    if not self.budget.can_start_epoch():
                        print("[budget] insufficient remaining time for a safe val pass; "
                              "saving checkpoint and stopping.")
                        self.epoch_in_stage += 1
                        self.global_epoch += 1
                        self.save_checkpoint(f"epoch_{self.global_epoch:03d}", keep_tagged=False)
                        return

                    val_metrics = self._run_epoch(loss_w, train=False)
                    dt = time.time() - t0
                    self.budget.record_epoch(dt)

                    self.epoch_in_stage += 1
                    self.global_epoch += 1
                    row = dict(stage=self.stage, stage_name=STAGE_NAMES_V2[self.stage],
                               epoch_in_stage=self.epoch_in_stage, global_epoch=self.global_epoch,
                               duration_s=dt, **{f"train_{k}": v for k, v in train_metrics.items()},
                               **{f"val_{k}": v for k, v in val_metrics.items()})
                    self.history.append(row)
                    pd_safe_write_csv(self.history, CSV_DIR / "training_history.csv")
                    print(self.budget.status())

                    val_auc = val_metrics.get("auc", float("nan"))
                    if val_auc == val_auc and val_auc > self.best_auc:
                        self.best_auc = val_auc
                        self.save_checkpoint("best_auc", keep_tagged=True)

                    self.save_checkpoint(f"epoch_{self.global_epoch:03d}", keep_tagged=False)

                self.epoch_in_stage = 0
                idx = STAGE_ORDER.index(self.stage)
                if idx + 1 < len(STAGE_ORDER):
                    self.stage = STAGE_ORDER[idx + 1]
                    self._rebuild_train_loader_for_stage()
                else:
                    self.stage = None

            print("\n[trainer] all stages complete.")
            self.evaluate_ema_once()  # Part 18: EMA validated ONCE, not every epoch

        def evaluate_ema_once(self) -> float:
            ema_model = copy.deepcopy(self.student)
            self.ema.copy_to(ema_model)
            ema_model.eval()
            ema_probs, ema_labels = [], []
            with torch.no_grad():
                for batch in self.val_loader:
                    b = _to_device(batch)
                    out = ema_model(b)
                    ema_probs.append(torch.sigmoid(out["joint_logit"]).cpu().numpy())
                    ema_labels.append(b["label"].cpu().numpy())
            if not ema_probs:
                del ema_model
                return float("nan")
            auc = compute_binary_metrics(np.concatenate(ema_labels), np.concatenate(ema_probs)).get("auc", float("nan"))
            print(f"[ema] one-time end-of-training EMA AUC on the val subset: {auc:.4f} "
                  f"(compare against self.best_auc={self.best_auc:.4f})")
            self.best_ema_auc = auc
            if auc == auc and auc > self.best_auc:
                self.save_checkpoint("best_ema_auc", keep_tagged=True, student_override=ema_model)
            del ema_model
            return auc

        def _run_epoch(self, loss_w: LossWeights, train: bool) -> dict:
            self.student.train(train)
            loader = self.train_loader if train else self.val_loader
            all_probs, all_labels = [], []
            loss_sum, n_batches, grad_norms = 0.0, 0, []
            batch_weight_rows = []

            batch_sampler_offset = self.batch_in_epoch if train else 0
            for bi, batch in enumerate(loader):
                if train:
                    if bi < batch_sampler_offset:
                        # DataLoader iteration order is fixed by domain_balanced_sample()'s
                        # deterministic seed for this (stage, epoch_in_stage) -- see
                        # build_stage_train_loader(). Skipping by index is cheap bookkeeping
                        # now that NO teacher forward pass happens per batch (Part 14/19).
                        continue

                    loss, info = self._step(batch, loss_w)
                    if loss is None:
                        self.batch_in_epoch = bi + 1
                    else:
                        gn = self._optimizer_step(loss)
                        if gn == gn:
                            grad_norms.append(gn)
                        loss_sum += float(loss.detach().cpu()); n_batches += 1
                        batch_weight_rows.append(dict(stage=self.stage, epoch=self.global_epoch, batch=bi,
                                                       **{TEACHER_ORDER[i]: w for i, w in enumerate(info["weights"])}))
                        self.batch_in_epoch = bi + 1

                    if time.time() - self._last_ckpt_wall_time > MID_EPOCH_CKPT_SECONDS:
                        self.save_checkpoint(
                            f"midepoch_stage{self.stage}_e{self.epoch_in_stage}_b{self.batch_in_epoch}",
                            keep_tagged=False,
                        )

                    if self.budget.should_stop_now():
                        self.save_checkpoint(
                            f"midepoch_budget_stage{self.stage}_e{self.epoch_in_stage}_b{self.batch_in_epoch}",
                            keep_tagged=False,
                        )
                        raise EpochAbortedForBudget()
                else:
                    with torch.no_grad():
                        b = _to_device(batch)
                        out = self.student(b)
                        all_probs.append(torch.sigmoid(out["joint_logit"]).detach().cpu().numpy())
                        all_labels.append(b["label"].detach().cpu().numpy())

            if train:
                self.batch_in_epoch = 0
                self.scheduler.step()
            metrics: Dict[str, float] = {}
            if train:
                metrics["loss"] = loss_sum / max(1, n_batches)
                metrics["grad_norm_mean"] = float(np.mean(grad_norms)) if grad_norms else float("nan")
                if batch_weight_rows:
                    pd_safe_write_csv(batch_weight_rows, CSV_DIR / "teacher_weights.csv",
                                       append=(CSV_DIR / "teacher_weights.csv").exists())
            else:
                if all_labels:
                    probs = np.concatenate(all_probs); labels = np.concatenate(all_labels)
                    metrics.update(compute_binary_metrics(labels, probs))
                    # Part 18: NO per-epoch EMA deepcopy + second val pass here anymore --
                    # EMA is validated once at the very end via evaluate_ema_once().
            return metrics

    # ══════════════════════════════════════════════════════════════════════
    # 14. OFFLINE TEACHER PRECOMPUTATION (Part 1/2/23) — MODE=="PRECOMPUTE_TEACHERS"
    # ══════════════════════════════════════════════════════════════════════

    def _maybe_restore_artifact_dir_from_input():
        """/kaggle/working is wiped every fresh session, so ARTIFACT_DIR's shards
        would otherwise vanish between PRECOMPUTE_TEACHERS sessions -- the same
        problem RESUME_CKPT_PATH solves for the student checkpoint. If ARTIFACT_DIR
        has no index.json yet and ARTIFACT_RESUME_INPUT_DIR is set and exists, copy
        the previously-saved teacher_artifacts/ tree in before touching the store."""
        if (ARTIFACT_DIR / "index.json").exists():
            return  # already populated this session (e.g. a re-run without a kernel restart)
        if not ARTIFACT_RESUME_INPUT_DIR:
            print("[artifacts] ARTIFACT_RESUME_INPUT_DIR is empty; starting with a fresh, "
                  "empty artifact store. If you already have artifacts from a previous "
                  "session, save that teacher_artifacts/ folder as a Kaggle dataset and "
                  "point ARTIFACT_RESUME_INPUT_DIR at the mounted path, or this session "
                  "will recompute from scratch.")
            return
        src = Path(ARTIFACT_RESUME_INPUT_DIR)
        if not src.exists():
            print(f"[artifacts] ARTIFACT_RESUME_INPUT_DIR={src} does not exist; "
                  f"starting with an empty artifact store.")
            return
        import shutil
        print(f"[artifacts] restoring teacher_artifacts from {src} -> {ARTIFACT_DIR} "
              f"(resuming across a fresh Kaggle session)")
        shutil.copytree(src, ARTIFACT_DIR, dirs_exist_ok=True)


    def _maybe_restore_best_checkpoint_from_input():
        """Restores BOTH best_auc.pth and best_ema_auc.pth when their input paths
        are set, instead of copying one in under the other's name -- EVALUATE now
        picks whichever recorded AUC is actually higher (see main())."""
        def _restore_one(input_path: str, dest_name: str):
            dest_path = CKPT_DIR / dest_name
            if dest_path.exists():
                return
            if not input_path:
                return
            if not Path(input_path).exists():
                print(f"[checkpoint] resume path for {dest_name}={input_path} "
                      f"does not exist; skipping restore of {dest_name}.")
                return
            import shutil
            resolved = resolve_ckpt_file(input_path)
            CKPT_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(resolved, dest_path)
            print(f"[checkpoint] restored {dest_name} from {input_path} "
                  f"(resolved to {resolved}) -> {dest_path}")

        _restore_one(BEST_CKPT_RESUME_INPUT_PATH, "best_auc.pth")
        _restore_one(BEST_EMA_CKPT_RESUME_INPUT_PATH, "best_ema_auc.pth")
        if not (CKPT_DIR / "best_auc.pth").exists() and not (CKPT_DIR / "best_ema_auc.pth").exists():
            print("[checkpoint] neither BEST_CKPT_RESUME_INPUT_PATH nor "
                  "BEST_EMA_CKPT_RESUME_INPUT_PATH resolved to a present checkpoint; "
                  "EVALUATE will fail.")


    def _precompute_one_teacher(name: str, todo_rows: List[dict], example_model: "nn.Module",
                                 store: "ShardedTeacherArtifactStore", branch: str) -> str:
        """Runs precomputation for ONE teacher over `todo_rows`, with adaptive
        batch-size/timeout/worker backoff on failure (Part 1/23). Any exception
        during a batch pass (DataLoader timeout, CUDA OOM, a corrupt file, a dead
        worker) triggers: flush whatever was already buffered (never lost), halve
        the batch size, drop a worker, raise the timeout, and retry the remaining
        rows. At the batch_size=1 floor, a still-failing row is the specific
        offender -- it is skipped permanently (logged) and the rest continues,
        rather than aborting the whole teacher for one bad file.
        Returns one of: "done", "done_with_skips", "budget_exhausted".
        """
        input_key = {"frame": "frame", "temporal": "clip", "audio": "audio"}[branch]
        present_key = {"frame": "mask_frame", "temporal": "mask_temporal", "audio": "mask_audio"}[branch]
        bs = PRECOMPUTE_BATCH_SIZE_BY_BRANCH.get(branch, 32)
        workers = PRECOMPUTE_NUM_WORKERS_BY_BRANCH.get(branch, NUM_WORKERS)
        timeout_s = PRECOMPUTE_TIMEOUT_SECONDS_BY_BRANCH.get(branch, 180)
        prefetch = PRECOMPUTE_PREFETCH_BY_BRANCH.get(branch, 4)
        default_bs = bs
        remaining = list(todo_rows)
        skipped_ids: List[str] = []
        _last_partial_flush = [time.time()]  # mutable cell so the inner loop can update it

        while remaining:
            already_done_now = set(store.index.get(name, {}).keys())
            remaining = [r for r in remaining if r.get("id", "") not in already_done_now]
            if not remaining:
                break

            ds = JointDeepfakeDataset(remaining, train=False, active_modalities={branch},
                                       deterministic_audio=True)
            # persistent_workers=False here on purpose: this loader is rebuilt every pass
            # through the retry/backoff while-loop above ("del loader" happens on both the
            # success and exception paths). With persistent_workers=True, torn-down-but-not-
            # yet-garbage-collected worker pools can accumulate across retries, compounding
            # shared-memory/file-descriptor pressure over a long precompute run. This loader
            # is short-lived by design, so there's no steady-state benefit to keeping workers
            # alive across it.
            loader = DataLoader(ds, batch_size=min(bs, max(1, len(remaining))), shuffle=False,
                                 num_workers=workers, pin_memory=torch.cuda.is_available(),
                                 drop_last=False, persistent_workers=False,
                                 prefetch_factor=(prefetch if workers > 0 else None),
                                 timeout=(timeout_s if workers > 0 else 0))
            try:
                with torch.no_grad():
                    for bi, batch in enumerate(loader):
                        if BUDGET.should_stop_now():
                            store.flush(name)
                            print(f"[precompute] budget exhausted mid-{name} at batch {bi}; "
                                  f"flushed partial shard and stopping. Re-run to resume "
                                  f"{name} from sample {store.teacher_completed_count(name)}.")
                            del loader
                            return "budget_exhausted"
                        b = _to_device(batch)
                        present = b[present_key].bool()
                        if not present.any():
                            continue
                        idx = present.nonzero(as_tuple=True)[0]
                        x = b[input_key].index_select(0, idx)
                        out = example_model.infer_all(x)
                        ids_present = [batch["id"][i] for i in idx.tolist()]
                        for j, sid in enumerate(ids_present):
                            seq_j = out["seq"][j] if "seq" in out else None
                            store.append(name, sid, out["logit"][j], out["embed"][j], seq_j)
                        if bi % 50 == 0:
                            _free_gpu_memory(f"{name} batch {bi}")
                        # Time-based partial flush, independent of ARTIFACT_SHARD_SIZE. A
                        # kernel death like the one that hit frame_diff can happen at any
                        # point between shard boundaries; flushing on a wall-clock cadence
                        # bounds how much work an untrapped crash can lose, regardless of
                        # how many samples happen to be buffered at that moment.
                        if time.time() - _last_partial_flush[0] > 300:
                            store.flush(name)
                            _last_partial_flush[0] = time.time()
                store.flush(name)
                del loader
                return "done_with_skips" if skipped_ids else "done"
            except Exception as e:
                # Whatever was successfully appended before the failure is flushed NOW --
                # a crash here must never discard already-computed samples.
                store.flush(name)
                del loader
                _free_gpu_memory(f"{name} error-recovery")
                print(f"[precompute] {name}: batch iteration failed "
                      f"(batch_size={bs}, workers={workers}, timeout={timeout_s}s): {e!r}")
                if bs > 1:
                    bs = max(1, bs // 2)
                    workers = max(1, workers - 1) if workers > 1 else workers
                    timeout_s = int(timeout_s * 1.5)
                    print(f"[precompute] {name}: backing off to batch_size={bs}, workers={workers}, "
                          f"timeout={timeout_s}s and retrying the remaining {len(remaining)} rows.")
                    continue
                # Already at the minimum batch size (1) and still failing -- in this
                # deterministic (shuffle=False) order, `remaining[0]` is the specific
                # offending row. Skip it permanently and keep going with the rest.
                already_done_now = set(store.index.get(name, {}).keys())
                remaining = [r for r in remaining if r.get("id", "") not in already_done_now]
                if not remaining:
                    break
                bad_row = remaining.pop(0)
                skipped_ids.append(bad_row.get("id", "<unknown>"))
                print(f"[precompute] {name}: sample id={bad_row.get('id','<unknown>')} still fails "
                      f"at batch_size=1; skipping this one row permanently for {name} (it will "
                      f"simply be absent from this teacher's artifacts) and continuing with the "
                      f"remaining {len(remaining)} rows.")
                bs, workers, timeout_s = (default_bs, PRECOMPUTE_NUM_WORKERS_BY_BRANCH.get(branch, NUM_WORKERS),
                                           PRECOMPUTE_TIMEOUT_SECONDS_BY_BRANCH.get(branch, 180))

        if skipped_ids:
            print(f"[precompute] {name}: finished with {len(skipped_ids)} row(s) permanently "
                  f"skipped: {skipped_ids[:20]}{'...' if len(skipped_ids) > 20 else ''}")
        return "done_with_skips" if skipped_ids else "done"


    def _dir_size_bytes(path: Path) -> int:
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


    def precompute_teachers(all_rows: List[dict]):
        print("\n" + "#" * 78)
        print("# MODE=PRECOMPUTE_TEACHERS -- offline teacher artifact generation")
        print("#" * 78)
        store = ShardedTeacherArtifactStore(ARTIFACT_DIR, shard_size=ARTIFACT_SHARD_SIZE)

        # Issue 16: report precompute wall-clock time, samples/sec, and disk
        # footprint per teacher, accumulated ACROSS sessions.
        cost_path = OUT_DIR / "precompute_cost.json"
        cost_log: Dict[str, Any] = json.load(open(cost_path)) if cost_path.exists() else {}

        teachers, load_reports = load_all_teachers(abort_on_error=True)
        usable = teachers_are_usable(load_reports)

        branch_of = TEACHER_BRANCH

        for name in TEACHER_ORDER:
            if not usable.get(name, False):
                print(f"[precompute] {name}: teacher not usable (bad/missing checkpoint); skipping.")
                continue
            if BUDGET.should_stop_now():
                print(f"[precompute] budget exhausted before starting {name}; stopping run. "
                      f"Re-run PRECOMPUTE_TEACHERS to continue with the remaining teachers.")
                break

            branch = branch_of[name]
            eligible_rows = [r for r in all_rows if _row_has_modality_for_audit(r, branch)]
            already_done = set(store.index.get(name, {}).keys())

            todo_rows = [r for r in eligible_rows if r.get("id", "") not in already_done]

            print(f"[precompute] {name}: {len(eligible_rows)} eligible rows, "
                  f"{len(already_done)} already have artifacts, {len(todo_rows)} remaining.")
            if not todo_rows:
                continue
            if PRECOMPUTE_MAX_ROWS_PER_SESSION is not None:
                todo_rows = todo_rows[:PRECOMPUTE_MAX_ROWS_PER_SESSION]
                print(f"[precompute] {name}: capped to {len(todo_rows)} rows this session "
                      f"(PRECOMPUTE_MAX_ROWS_PER_SESSION={PRECOMPUTE_MAX_ROWS_PER_SESSION}).")

            example_model = teachers[name]
            feature_dim = int(TEACHER_EXPECTED.get(name, {}).get("feature_dim") or
                               getattr(example_model, "embed_dim", 0))
            store.init_teacher(name, feature_dim=feature_dim, has_seq=(name == "video_ff"),
                                ckpt_path=CKPT_PATHS[name], split_seed=SEED)

            n_before = store.teacher_completed_count(name)
            t0 = time.time()
            status = _precompute_one_teacher(name, todo_rows, example_model, store, branch)
            elapsed_this_session = time.time() - t0
            n_after = store.teacher_completed_count(name)

            entry = cost_log.setdefault(name, dict(total_elapsed_seconds=0.0, total_samples_computed=0,
                                                     sessions=0))
            entry["total_elapsed_seconds"] += elapsed_this_session
            entry["total_samples_computed"] = n_after
            entry["sessions"] += 1
            entry["samples_per_second_this_session"] = (
                (n_after - n_before) / elapsed_this_session if elapsed_this_session > 0 else None)
            shard_dir = ARTIFACT_DIR / name
            entry["disk_usage_mb"] = round(_dir_size_bytes(shard_dir) / 1e6, 2) if shard_dir.exists() else 0.0

            save_json_static(cost_path, cost_log)

            if status == "budget_exhausted":
                return
            suffix = " -- some rows were permanently skipped, see log above" if status == "done_with_skips" else ""
            print(f"[precompute] {name}: DONE ({store.teacher_completed_count(name)} artifacts){suffix} "
                  f"[{elapsed_this_session:.1f}s this session, "
                  f"{entry['total_elapsed_seconds']:.1f}s total, {entry['disk_usage_mb']:.1f} MB on disk].")

        total_disk_mb = round(_dir_size_bytes(ARTIFACT_DIR) / 1e6, 2) if ARTIFACT_DIR.exists() else 0.0
        cost_log["_total_disk_usage_mb_all_teachers"] = total_disk_mb
        cost_log["_shard_layout"] = (
            "teacher_artifacts/<teacher_name>/shard_NNNNN.pt (torch.save dict of "
            "sample_ids/logit/embed[/seq]); index.json maps sample_id -> (shard, row); "
            f"metadata.json holds per-teacher config. ARTIFACT_SHARD_SIZE={ARTIFACT_SHARD_SIZE} "
            "samples per shard.")
        save_json_static(cost_path, cost_log)
        print(f"\n[precompute] wrote precompute_cost.json -> {cost_path} "
              f"(total disk usage: {total_disk_mb:.1f} MB)")
        print("[precompute] all teachers processed for this session. "
              "Re-run MODE=PRECOMPUTE_TEACHERS if any teacher above reported budget exhaustion.")


# ══════════════════════════════════════════════════════════════════════════
# 5a. MANIFEST LOADING + AUDIT — pure-Python (unchanged from original, works
#     even without torch so the audit-only fallback path can still run).
# ══════════════════════════════════════════════════════════════════════════

MODALITIES = ("frame", "temporal", "audio")

IS_SYNTHETIC_MANIFEST = False
def _ood_id(dataset: str, path: str) -> str:
    """Deterministic md5 id, namespaced by dataset, so it can never collide
    with an in-domain manifest id (main()'s id-collision check relies on this)."""
    h = hashlib.md5(path.encode("utf-8")).hexdigest()
    return f"{dataset}_{h}"


def build_ood_manifest(celebdf_root: str, wilddeepfake_root: str, out_path: str) -> List[dict]:
    """Walks the raw Celeb-DF v2 (video) and WildDeepfake (frame) dataset roots
    directly and writes a flat ood_manifest.jsonl -- no pre-built manifest needs
    to be uploaded as its own Kaggle dataset. Every row gets split='ood'."""
    rows: List[dict] = []

    # ---- Celeb-DF v2: video dataset, three label folders ----
    celebdf_p = Path(celebdf_root)
    celebdf_folders = {
        "Celeb-real": ("celeb_real", 0),
        "YouTube-real": ("youtube_real", 0),
        "Celeb-synthesis": ("celeb_synthesis", 1),
    }
    for folder_name, (manip, label) in celebdf_folders.items():
        folder = celebdf_p / folder_name
        if not folder.exists():
            print(f"[build_ood_manifest] WARNING: {folder} not found; skipping.")
            continue
        videos = sorted(folder.glob("*.mp4"))
        for v in videos:
            rows.append(dict(
                id=_ood_id("celebdf", str(v)), dataset="celebdf", label=label,
                video_path=str(v), manipulation=manip, split="ood",
            ))
        print(f"[build_ood_manifest] celebdf/{folder_name}: {len(videos)} videos "

              f"(label={label}, manipulation={manip})")

    # ---- WildDeepfake: frame dataset, real/ and fake/ (fake/ may nest subfolders) ----
    wdf_p = Path(wilddeepfake_root)
    wdf_folders = {"real": 0, "fake": 1}
    for folder_name, label in wdf_folders.items():
        folder = wdf_p / folder_name
        if not folder.exists():
            print(f"[build_ood_manifest] WARNING: {folder} not found; skipping.")
            continue
        frames = sorted(list(folder.rglob("*.png")) + list(folder.rglob("*.jpg")) +
                         list(folder.rglob("*.jpeg")))
        for f in frames:
            rows.append(dict(
                id=_ood_id("wilddeepfake", str(f)), dataset="wilddeepfake", label=label,
                frame_path=str(f), manipulation=f"wilddeepfake_{folder_name}", split="ood",
            ))
        print(f"[build_ood_manifest] wilddeepfake/{folder_name}: {len(frames)} frames (label={label})")

    if not rows:
        raise RuntimeError(f"[build_ood_manifest] found ZERO rows under {celebdf_root!r} / "
                            f"{wilddeepfake_root!r}; check the paths before trusting any OOD numbers.")

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n_celebdf = sum(1 for r in rows if r["dataset"] == "celebdf")
    n_wdf = sum(1 for r in rows if r["dataset"] == "wilddeepfake")
    print(f"[build_ood_manifest] wrote {len(rows)} rows ({n_celebdf} celebdf, {n_wdf} wilddeepfake) "
          f"-> {out_p}")

    return rows
def load_manifest(path: str) -> List[dict]:
    global IS_SYNTHETIC_MANIFEST
    p = Path(path)
    rows: List[dict] = []
    if not p.exists():
        if not SMOKE_TEST_MODE:
            raise RuntimeError(
                f"[load_manifest] MANIFEST_PATH not found: {path}\n"
                f"This is a hard error in research mode (SMOKE_TEST_MODE=False)."
            )
        IS_SYNTHETIC_MANIFEST = True
        print(f"[manifest] {path} not found; SMOKE_TEST_MODE=True -> generating synthetic "
              f"self-test manifest. ALL NUMBERS PRODUCED FROM THIS DATA ARE MEANINGLESS.")
        return _build_synthetic_manifest()
    if p.suffix == ".jsonl":
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        with open(p) as f:
            data = json.load(f)
        rows = data if isinstance(data, list) else data.get("samples", [])
    print(f"[manifest] loaded {len(rows)} rows from {path}")
    return rows

def _build_synthetic_manifest(n: int = 96) -> List[dict]:
    rows = []
    datasets_cycle = ["dfdc", "diffusionface", "ffpp", "lavdf"]
    for i in range(n):
        ds = datasets_cycle[i % len(datasets_cycle)]
        label = (i // 4) % 2
        split = "train" if i % 10 < 7 else ("val" if i % 10 < 9 else "test")
        row = dict(id=f"synthetic_{i:05d}", dataset=ds, label=label, split=split,
                   manipulation="synthetic", synthetic=True)
        if ds in ("dfdc", "diffusionface"):
            row["frame_path"] = f"synthetic://{ds}/frame_{i}.jpg"
        elif ds == "ffpp":
            row["video_path"] = f"synthetic://{ds}/clip_{i}.mp4"
        elif ds == "lavdf":
            row["audio_path"] = f"synthetic://{ds}/audio_{i}.wav"
        rows.append(row)
    return rows

def _row_has_modality(row: dict, modality: str) -> bool:
    key = {"frame": "frame_path", "temporal": "video_path", "audio": "audio_path"}[modality]
    v = row.get(key)
    return v is not None and str(v).strip() != ""


def _row_has_modality_for_audit(row: dict, modality: str) -> bool:
    if modality == "frame":
        return _row_has_modality(row, "frame") or _row_has_modality(row, "temporal")
    return _row_has_modality(row, modality)

@dataclass
class ManifestAuditReport:
    total: int = 0
    counts: Dict[str, int] = field(default_factory=dict)
    per_dataset: Dict[str, Dict[str, int]] = field(default_factory=dict)
    triple_count: int = 0

def _regime_of_row(row: dict) -> str:
    has = tuple(sorted(m for m in MODALITIES if _row_has_modality_for_audit(row, m)))
    return "+".join(has) if has else "none"

def audit_manifest(rows: List[dict]) -> ManifestAuditReport:
    print("\n" + "#" * 78)
    print("# MANIFEST MODALITY-COOCCURRENCE AUDIT")
    print("#" * 78)
    rep = ManifestAuditReport(total=len(rows))
    for row in rows:
        regime = _regime_of_row(row)
        rep.counts[regime] = rep.counts.get(regime, 0) + 1
        ds = row.get("dataset", "unknown")
        rep.per_dataset.setdefault(ds, {})
        rep.per_dataset[ds][regime] = rep.per_dataset[ds].get(regime, 0) + 1

    triple_key = "+".join(sorted(MODALITIES))
    rep.triple_count = rep.counts.get(triple_key, 0)

    print(f"total rows: {rep.total}")
    for regime, c in sorted(rep.counts.items(), key=lambda x: -x[1]):
        print(f"  {regime:30s} {c:6d}  ({100.0*c/max(1,rep.total):5.1f}%)")
    print(f"\ntriple-coverage (frame+temporal+audio simultaneously): {rep.triple_count}")

    if rep.triple_count == 0:
        msg = ("NO rows contain frame+temporal+audio simultaneously.")
        if not ALLOW_PARTIAL_MULTIMODAL_TRAINING:
            raise RuntimeError("[audit_manifest] " + msg + " Set ALLOW_PARTIAL_MULTIMODAL_TRAINING=True.")
        else:
            print(f"[audit_manifest] WARNING: {msg}")
    save_json_static(OUT_DIR / "manifest_audit.json",
                      dict(total=rep.total, counts=rep.counts, per_dataset=rep.per_dataset,
                           triple_count=rep.triple_count))
    return rep

def save_json_static(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def _stratified_split(rows: List[dict], val_frac: float, test_frac: float, seed: int
                       ) -> Tuple[List[dict], List[dict], List[dict]]:
    by_label: Dict[int, List[dict]] = {}
    for r in rows:
        by_label.setdefault(int(r.get("label", 0)), []).append(r)

    rng = random.Random(seed)
    train, val, test = [], [], []
    for label, group in by_label.items():
        g = list(group)
        rng.shuffle(g)
        n = len(g)
        n_val = max(1, int(round(val_frac * n))) if n >= 3 else 0
        n_test = max(1, int(round(test_frac * n))) if n >= 3 else 0
        val.extend(g[:n_val])
        test.extend(g[n_val:n_val + n_test])
        train.extend(g[n_val + n_test:])

    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)
    return train, val, test


def _assert_split_has_both_classes(rows: List[dict], split_name: str):
    labels = {int(r.get("label", 0)) for r in rows}
    if len(labels) < 2:
        msg = (f"[split-guard] split '{split_name}' contains only label(s) {labels} "
               f"({len(rows)} rows) -- at least one class is entirely absent.")
        if not SMOKE_TEST_MODE:
            raise RuntimeError(msg)
        print("[split-guard] WARNING: " + msg)

def _preflight_check_decoders(rows: List[dict], n_samples: int = 5):
    video_rows = [r for r in rows if _row_has_modality(r, "temporal")][:n_samples]
    audio_rows = [r for r in rows if _row_has_modality(r, "audio")][:n_samples]

    if video_rows:
        try:
            import decord  # noqa: F401
        except Exception as e:
            raise RuntimeError(f"[preflight] `decord` is not importable ({e!r}).")
        ds = JointDeepfakeDataset(video_rows, train=False, active_modalities={"temporal"})
        n_ok = sum(1 for i in range(len(video_rows)) if bool(ds[i]["mask_temporal"]))
        if n_ok == 0:
            raise RuntimeError(f"[preflight] decord imported but ALL sample video decodes failed.")
        print(f"[preflight] video decode OK: {n_ok}/{len(video_rows)} sample rows decoded.")

    if audio_rows:
        try:
            import torchaudio  # noqa: F401
        except Exception as e:
            raise RuntimeError(f"[preflight] `torchaudio` is not importable ({e!r}).")
        ds = JointDeepfakeDataset(audio_rows, train=False, active_modalities={"audio"})
        n_ok = sum(1 for i in range(len(audio_rows)) if bool(ds[i]["mask_audio"]))
        if n_ok == 0:
            raise RuntimeError(f"[preflight] torchaudio imported but ALL sample audio decodes failed.")
        print(f"[preflight] audio decode OK: {n_ok}/{len(audio_rows)} sample rows decoded.")

KNOWN_DOMAINS = ("dfdc", "diffusionface", "ffpp", "lavdf")


def _stratified_subsample(rows: List[dict], n: int, seed: int) -> List[dict]:
    """Label-only stratified subsample (kept for calib_pool/baseline_train_pool/
    val-fallback use). Part 12's domain-balanced sampling for STUDENT TRAINING
    lives in domain_balanced_sample() below."""
    if n >= len(rows):
        return list(rows)
    by_label: Dict[int, List[dict]] = {}
    for r in rows:
        by_label.setdefault(int(r.get("label", 0)), []).append(r)
    rng = random.Random(seed)
    out = []
    total = len(rows)
    for label, group in by_label.items():
        share = max(1, round(n * len(group) / total))
        share = min(share, len(group))
        out.extend(rng.sample(group, share))
    rng.shuffle(out)
    return out[:n]


def domain_balanced_sample(rows: List[dict], n: int, seed: int,
                            log_path: Optional[Path] = None) -> List[dict]:
    """Domain-balanced AND class-balanced deterministic sample (Part 12)."""
    by_domain: Dict[str, List[dict]] = {d: [] for d in KNOWN_DOMAINS}
    for r in rows:
        ds = r.get("dataset", "unknown")
        by_domain.setdefault(ds, []).append(r)

    present_domains = [d for d, g in by_domain.items() if g]
    if not present_domains:
        return []
    target_per_domain = max(1, n // len(present_domains))

    rng = random.Random(seed)
    out: List[dict] = []
    count_log: List[dict] = []
    for ds in present_domains:
        group = by_domain[ds]
        by_label: Dict[int, List[dict]] = {}
        for r in group:
            by_label.setdefault(int(r.get("label", 0)), []).append(r)
        take = min(target_per_domain, len(group))
        domain_out: List[dict] = []
        labels_present = list(by_label.keys())
        per_label_target = max(1, take // max(1, len(labels_present)))
        for label, lg in by_label.items():
            k = min(per_label_target, len(lg))
            domain_out.extend(rng.sample(lg, k))
        if len(domain_out) < take:
            remaining = [r for r in group if r not in domain_out]
            domain_out.extend(rng.sample(remaining, min(take - len(domain_out), len(remaining))))
        domain_out = domain_out[:take]
        out.extend(domain_out)
        for label, lg in by_label.items():
            n_taken = sum(1 for r in domain_out if int(r.get("label", 0)) == label)
            count_log.append(dict(dataset=ds, label=label, n=n_taken))

    rng.shuffle(out)
    if log_path is not None and TORCH_OK:
        pd_safe_write_csv(count_log, log_path, append=log_path.exists())
    return out[:n]


if TORCH_OK:

    def verify_artifacts_before_training(all_rows: List[dict]):
        """Part 25: hard-stop checks run once before TRAIN_STUDENT/EVALUATE build
        any loader or model."""
        print("\n" + "#" * 78)
        print("# ARTIFACT CONSISTENCY CHECK (Part 25)")
        print("#" * 78)
        if not (ARTIFACT_DIR / "index.json").exists():
            raise RuntimeError(f"[verify_artifacts] {ARTIFACT_DIR}/index.json not found. Run "
                                f"MODE='PRECOMPUTE_TEACHERS' to completion before TRAIN_STUDENT/EVALUATE.")
        reader = TeacherArtifactReader(ARTIFACT_DIR)
        available = reader.available_teachers()
        if not available:
            raise RuntimeError("[verify_artifacts] artifact store exists but contains ZERO usable teachers.")
        print(f"[verify_artifacts] teachers with artifacts: {available}")

        seen_ids: Dict[str, Set[str]] = {}
        for name in available:
            ids = set(reader.index[name].keys())
            seen_ids[name] = ids
            print(f"[verify_artifacts] {name}: {len(ids)} artifacts, "
                  f"feature_dim={reader.metadata[name]['feature_dim']}, "
                  f"dtype_embed={reader.metadata[name].get('dtype_embed')}, "
                  f"preprocessing_version={reader.metadata[name].get('preprocessing_version')}")
            if reader.metadata[name].get("preprocessing_version") != PREPROCESSING_VERSION:
                raise RuntimeError(f"[verify_artifacts] {name}: artifact preprocessing_version "
                                    f"{reader.metadata[name].get('preprocessing_version')!r} does not match "
                                    f"current PREPROCESSING_VERSION={PREPROCESSING_VERSION!r}. Re-run "
                                    f"PRECOMPUTE_TEACHERS for this teacher.")
            dup_check = list(reader.index[name].keys())
            if len(dup_check) != len(set(dup_check)):
                raise RuntimeError(f"[verify_artifacts] {name}: duplicate sample IDs found in index.json.")

        all_row_ids = [r.get("id", "") for r in all_rows]
        if len(all_row_ids) != len(set(all_row_ids)):
            dupes = len(all_row_ids) - len(set(all_row_ids))
            raise RuntimeError(f"[verify_artifacts] manifest contains {dupes} duplicate row ids.")

        id_to_split = {r.get("id", ""): r.get("split", "train") for r in all_rows}
        train_ids = {i for i, s in id_to_split.items() if s == "train"}
        val_ids = {i for i, s in id_to_split.items() if s == "val"}
        test_ids = {i for i, s in id_to_split.items() if s == "test"}
        overlap_tv = train_ids & val_ids
        overlap_tt = train_ids & test_ids
        if overlap_tv or overlap_tt:
            raise RuntimeError(f"[verify_artifacts] split leakage detected: "
                                f"{len(overlap_tv)} ids in both train/val, "
                                f"{len(overlap_tt)} ids in both train/test (Part 24).")

        for name in available:
            ids = seen_ids[name]
            labels_present = {int(r.get("label", 0)) for r in all_rows if r.get("id", "") in ids}
            if len(labels_present) < 2:
                print(f"[verify_artifacts] WARNING: {name} artifacts cover only label(s) "
                      f"{labels_present} -- KD from this teacher will be one-sided.")

        print(f"[verify_artifacts] manifest rows: {len(all_rows)}, all checks passed.")


    class IDLabelBatchLoader:
        """Yields (id, label, dataset) batches directly from row dicts, with NO
        frame/video/audio decoding at all. Calibration and baseline-fusion training
        only ever look up precomputed teacher artifacts by sample id -- wrapping
        those steps in the full JointDeepfakeDataset + DataLoader pipeline paid the
        full media-decode cost of every row just to discard the decoded tensors,
        which is what previously forced CALIB_POOL_MAX/BASELINE_TRAIN_POOL_MAX down
        to a few thousand rows for speed. This loader makes decode cost zero, so
        these steps can safely run over the WHOLE split.
        """
        def __init__(self, rows: List[dict], batch_size: int = 1024, shuffle: bool = False, seed: int = SEED):
            self.rows = rows
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.seed = seed

        def __len__(self):
            return max(1, (len(self.rows) + self.batch_size - 1) // self.batch_size)

        def __iter__(self):
            order = list(range(len(self.rows)))
            if self.shuffle:
                random.Random(self.seed).shuffle(order)
            for i in range(0, len(order), self.batch_size):
                idx = order[i:i + self.batch_size]
                chunk = [self.rows[j] for j in idx]
                yield dict(
                    id=[r.get("id", "") for r in chunk],
                    label=torch.tensor([float(r.get("label", 0)) for r in chunk]),
                    dataset=[r.get("dataset", "unknown") for r in chunk],
                )

    def build_data_loaders(rows: List[dict]):
        """`rows` is passed in (loaded once in main()) instead of reloading/
        re-auditing the manifest inside this function."""
        splits: Dict[str, List[dict]] = {"train": [], "val": [], "test": [], "ood": []}
        for r in rows:
            splits.setdefault(r.get("split", "train"), []).append(r)

        if not splits["train"] and not splits["val"] and not splits["test"]:
            splits["train"], splits["val"], splits["test"] = _stratified_split(
                rows, val_frac=0.15, test_frac=0.15, seed=SEED
            )
        if not splits["val"]:
            splits["val"] = splits["train"][: max(1, len(splits["train"]) // 10)]
        if not splits["test"]:
            splits["test"] = splits["val"]
        if not splits["train"]:
            splits["train"] = rows

        _assert_split_has_both_classes(splits["train"], "train")
        _assert_split_has_both_classes(splits["val"], "val")
        _assert_split_has_both_classes(splits["test"], "test")

        def _loader(split_rows: List[dict], train: bool) -> "DataLoader":
            ds = JointDeepfakeDataset(split_rows, train=train)
            bs = min(BATCH_SIZE, max(1, len(split_rows)))
            return DataLoader(ds, batch_size=bs, shuffle=train, num_workers=NUM_WORKERS,
                               pin_memory=torch.cuda.is_available(),
                               drop_last=(train and len(split_rows) > bs),
                               persistent_workers=False,
                               prefetch_factor=(4 if NUM_WORKERS > 0 else None),
                               timeout=(600 if NUM_WORKERS > 0 else 0))

        # Part 17: fixed, deterministic, domain-and-class-balanced VAL_SUBSET_SIZE-row
        # subset for per-epoch validation during TRAIN_STUDENT. Full test split is
        # still used untouched for EVALUATE mode.
        val_subset_rows = splits["val"]
        if len(val_subset_rows) > VAL_SUBSET_SIZE:
            val_subset_rows = domain_balanced_sample(splits["val"], VAL_SUBSET_SIZE, seed=SEED + 777)
            print(f"[build_data_loaders] val subset capped to {len(val_subset_rows)} rows "
                  f"(from {len(splits['val'])}) for per-epoch validation.")
        val_loader = _loader(val_subset_rows, False)
        test_loader = _loader(splits["test"], False)

        # Calibration/baseline-fusion steps only ever look up precomputed teacher
        # artifacts by id (see load_teacher_temperatures_from_artifacts /
        # train_and_eval_baselines_from_artifacts) -- they never need decoded
        # frame/clip/audio tensors. IDLabelBatchLoader removes that decode cost
        # entirely, so both pools can now cover the WHOLE split cheaply, giving
        # more robust temperature fits (esp. for video_ff/audio_lavdf, which are
        # only present on ~42% of rows) and a baseline trained on more data.
        CALIB_POOL_MAX = None  # None = use the entire val split (cheap: id/label only)
        calib_pool = splits["val"] if len(splits["val"]) >= 8 else splits["train"]
        if CALIB_POOL_MAX is not None and len(calib_pool) > CALIB_POOL_MAX:
            rng = random.Random(SEED)
            calib_pool = rng.sample(calib_pool, CALIB_POOL_MAX)
        calib_loader = IDLabelBatchLoader(calib_pool, batch_size=1024, shuffle=False)
        print(f"[build_data_loaders] calibration pool: {len(calib_pool)} rows "
              f"(id/label-only loader, no media decode).")

        BASELINE_TRAIN_POOL_MAX = None  # None = use the entire train split
        baseline_train_pool = splits["train"]
        if BASELINE_TRAIN_POOL_MAX is not None and len(baseline_train_pool) > BASELINE_TRAIN_POOL_MAX:
            rng = random.Random(SEED + 1)
            baseline_train_pool = rng.sample(baseline_train_pool, BASELINE_TRAIN_POOL_MAX)
        baseline_train_loader = IDLabelBatchLoader(baseline_train_pool, batch_size=1024,
                                                     shuffle=True, seed=SEED + 1)
        print(f"[build_data_loaders] baseline-fusion training pool: {len(baseline_train_pool)} rows "
              f"(id/label-only loader, no media decode).")

        # Lightweight mirror of the test split for the baseline-eval loops in
        # train_and_eval_baselines_from_artifacts() -- those loops only look up
        # precomputed logits/embeds by id, so re-running them through the full
        # decoding test_loader (as before) meant decoding every test
        # video/clip/waveform three extra times for nothing. The STUDENT's own
        # forward pass in run_comprehensive_evaluation_from_artifacts still uses
        # the real test_loader, since it genuinely needs decoded tensors.
        test_id_loader = IDLabelBatchLoader(splits["test"], batch_size=1024, shuffle=False)

        ood_by_name: Dict[str, List[dict]] = {}
        for r in splits["ood"]:
            ood_by_name.setdefault(r.get("dataset", "ood"), []).append(r)
        ood_loaders = {name: _loader(rows_, False) for name, rows_ in ood_by_name.items()}

        return (splits["train"], val_loader, test_loader, calib_loader, baseline_train_loader,
                test_id_loader, ood_loaders)


    def build_stage_train_loader(all_train_rows: List[dict], stage: str) -> "DataLoader":
        """Domain-and-class-balanced (Part 12), letter-keyed-stage (Part 11) loader.
        persistent_workers=False (Part 33): this loader lives for an entire stage's
        worth of epochs (thousands of batches with decord video decode active in
        Stages B/C) -- exactly the long-lived-worker-pool condition
        _precompute_one_teacher()'s persistent_workers=False was deliberately chosen
        to avoid, but this training loader had never been given the same treatment.
        A short per-epoch worker-startup cost is a better trade than an untrapped
        mid-epoch kernel death that loses partial epoch progress."""
        pool_max = STAGE_TRAIN_POOL_MAX_V2.get(stage, len(all_train_rows))
        log_path = CSV_DIR / "domain_class_sampling.csv"
        rows = domain_balanced_sample(all_train_rows, pool_max, seed=SEED + 1000 + STAGE_ORDER.index(stage),
                                       log_path=log_path)
        active = ACTIVE_BRANCHES_BY_STAGE_V2.get(stage, {"frame", "temporal", "audio"})
        ds = JointDeepfakeDataset(rows, train=True, active_modalities=active)
        bs = STAGE_BATCH_SIZE_V2.get(stage, BATCH_SIZE)
        bs = min(bs, max(1, len(rows)))
        print(f"[build_stage_train_loader] stage {stage} ({STAGE_NAMES_V2.get(stage)}): "
              f"{len(rows)} rows (pool cap {pool_max}), active_modalities={sorted(active)}, "
              f"batch_size={bs} -> {max(1, len(rows)//bs)} batches/epoch")
        return DataLoader(ds, batch_size=bs, shuffle=True, num_workers=NUM_WORKERS,
                           pin_memory=torch.cuda.is_available(),
                           drop_last=(len(rows) > bs),
                           persistent_workers=False,
                           prefetch_factor=(4 if NUM_WORKERS > 0 else None),
                           timeout=(600 if NUM_WORKERS > 0 else 0))


    def _dry_run_timing_check(trainer: "KDTrainer", n_batches: int = 20) -> dict:
        """Measures real s/batch WITHOUT taking any optimizer step and WITHOUT
        leaving model/optimizer/RNG state changed (Part 15)."""
        student_state = copy.deepcopy(trainer.student.state_dict())
        opt_state = copy.deepcopy(trainer.optimizer.state_dict())
        scaler_state = copy.deepcopy(trainer.scaler.state_dict())
        rng_state = dict(python=random.getstate(), numpy=np.random.get_state(),
                          torch=torch.get_rng_state(),
                          cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)

        trainer.student.eval()
        loss_w = STAGE_SCHEDULE_V2[trainer.stage]
        load_times, fwd_times, bwd_times = [], [], []
        n = 0
        it = iter(trainer.train_loader)
        for _ in range(n_batches):
            t0 = time.time()
            try:
                batch = next(it)
            except StopIteration:
                break
            t1 = time.time()
            loss, _info = trainer._step(batch, loss_w)
            t2 = time.time()
            if loss is not None and loss.requires_grad:
                loss.backward()
            t3 = time.time()
            load_times.append(t1 - t0); fwd_times.append(t2 - t1); bwd_times.append(t3 - t2)
            n += 1

        trainer.student.load_state_dict(student_state)
        trainer.optimizer.load_state_dict(opt_state)
        trainer.optimizer.zero_grad(set_to_none=True)
        trainer.scaler.load_state_dict(scaler_state)
        random.setstate(rng_state["python"]); np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch"])
        if rng_state["cuda"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng_state["cuda"])
        trainer.student.train()

        per_batch = (sum(load_times) + sum(fwd_times) + sum(bwd_times)) / max(1, n)
        batches_per_epoch = len(trainer.train_loader)
        result = dict(stage=trainer.stage, n_probed=n, s_per_batch=per_batch,
                      load_s=float(np.mean(load_times)) if load_times else 0.0,
                      fwd_s=float(np.mean(fwd_times)) if fwd_times else 0.0,
                      bwd_s=float(np.mean(bwd_times)) if bwd_times else 0.0,
                      batches_per_epoch=batches_per_epoch,
                      projected_epoch_hours=(per_batch * batches_per_epoch) / 3600.0)
        print(f"[dry_run] stage {trainer.stage} ({STAGE_NAMES_V2[trainer.stage]}): "
              f"{per_batch:.2f}s/batch (load={result['load_s']:.2f}s fwd={result['fwd_s']:.2f}s "
              f"bwd={result['bwd_s']:.2f}s) over {n} batches -> "
              f"~{result['projected_epoch_hours']:.2f}h for this stage's epoch "
              f"({batches_per_epoch} batches/epoch). Model/optimizer/RNG state fully restored.")
        return result


    def run_feasibility_gate(trainer: "KDTrainer", timing: dict) -> bool:
        """Part 22: prints a full feasibility report and returns False if the
        projected schedule cannot fit within a reasonable number of sessions."""
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        free_b, total_b = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0)
        remaining_stage_epochs = {s: (STAGE_EPOCHS_V2[s] - (trainer.epoch_in_stage if s == trainer.stage else 0))
                                   for s in STAGE_ORDER[STAGE_ORDER.index(trainer.stage):]}
        projected_total_hours = sum(timing["projected_epoch_hours"] * n for n in remaining_stage_epochs.values())
        sessions_needed = max(1, int(np.ceil(projected_total_hours / FEASIBILITY_MAX_PROJECTED_HOURS)))

        print("\n" + "#" * 78)
        print("# PRE-TRAINING FEASIBILITY GATE")
        print("#" * 78)
        print(f"  GPU:                       {gpu_name}")
        print(f"  GPU memory free/total:     {free_b/1e9:.2f} / {total_b/1e9:.2f} GiB")
        print(f"  training pool (this stage): {len(trainer.train_loader.dataset)} rows")
        print(f"  batch size (this stage):    {trainer.train_loader.batch_size}")
        print(f"  batches/epoch:              {timing['batches_per_epoch']}")
        print(f"  measured load s/batch:      {timing['load_s']:.3f}")
        print(f"  measured forward s/batch:   {timing['fwd_s']:.3f}")
        print(f"  measured backward s/batch:  {timing['bwd_s']:.3f}")
        print(f"  measured total s/batch:     {timing['s_per_batch']:.3f}")
        print(f"  projected time/epoch:       {timing['projected_epoch_hours']:.2f}h")
        print(f"  remaining epochs by stage:  {remaining_stage_epochs}")
        print(f"  projected TOTAL train time: {projected_total_hours:.2f}h")
        print(f"  estimated Kaggle sessions:  {sessions_needed} (at "
              f"{FEASIBILITY_MAX_PROJECTED_HOURS:.1f}h/session budget)")

        if projected_total_hours > FEASIBILITY_MAX_PROJECTED_HOURS * 6:
            print(f"\n[feasibility] STOP: projected total training time ({projected_total_hours:.2f}h) "
                  f"is unreasonably large (>6 sessions). Reduce STAGE_TRAIN_POOL_MAX_V2 / raise "
                  f"STAGE_BATCH_SIZE_V2 before starting a long run.")
            return False
        print(f"\n[feasibility] OK: proceeding. Expect ~{sessions_needed} session(s) to finish.")
        return True

    # ══════════════════════════════════════════════════════════════════════
    # 15. EVALUATION SUITE (Part 19/27: reads offline teacher artifacts, not
    #     live teacher forwards) — used ONLY by MODE=="EVALUATE"
    # ══════════════════════════════════════════════════════════════════════

    def _assert_checkpoint_matches_current_architecture(payload: dict, student: "nn.Module"):
        """Part 26: refuses to load an incompatible (e.g. old EfficientFormer-
        based) student checkpoint. Also refuses to evaluate a checkpoint trained
        under a DIFFERENT teacher set than the current TEACHER_ORDER -- the
        student's weight SHAPE doesn't change when a teacher is dropped, so
        without this check a mismatched checkpoint would load "successfully"
        and silently produce numbers for the wrong ablation."""
        ckpt_version = payload.get("architecture_version")
        current_version = KDTrainer.ARCHITECTURE_VERSION
        if ckpt_version != current_version:
            raise RuntimeError(
                f"[checkpoint-guard] checkpoint architecture_version={ckpt_version!r} does not "
                f"match the current student's architecture_version={current_version!r}. Refusing "
                f"to load an incompatible checkpoint."
            )
        ckpt_teachers = payload.get("teacher_order")
        if ckpt_teachers is not None and ckpt_teachers != list(TEACHER_ORDER):
            raise RuntimeError(
                f"[checkpoint-guard] checkpoint was trained with teacher_order={ckpt_teachers} "
                f"(run_tag={payload.get('run_tag')!r}) but the current session's "
                f"TEACHER_ORDER={list(TEACHER_ORDER)} (RUN_TAG={RUN_TAG!r}). This checkpoint "
                f"belongs to a DIFFERENT ablation/configuration -- evaluating it here would "
                f"produce numbers that don't match this run's config. Point "
                f"BEST_CKPT_RESUME_INPUT_PATH at the correct run's checkpoint."
            )
        model_keys = set(student.state_dict().keys())
        ckpt_keys = set(payload["student"].keys())
        if model_keys != ckpt_keys:
            missing = model_keys - ckpt_keys
            unexpected = ckpt_keys - model_keys
            raise RuntimeError(
                f"[checkpoint-guard] student state_dict key mismatch despite matching "
                f"architecture_version. missing={list(missing)[:10]} "
                f"unexpected={list(unexpected)[:10]}. Refusing to load."
            )

    def _regime_str(mf, mt, ma) -> str:
        parts = []
        if mf: parts.append("frame")
        if mt: parts.append("temporal")
        if ma: parts.append("audio")
        return "+".join(parts) if parts else "none"

    def _grouped_metrics(labels, probs, groups: List[str], ids: List[str]) -> Dict[str, dict]:
        groups = np.asarray(groups)
        ids_arr = np.asarray(ids)
        out = {}
        for g in sorted(set(groups)):
            mask = groups == g
            if mask.sum() < 2 or len(set(labels[mask])) < 2:
                out[g] = dict(n=int(mask.sum()), auc=float("nan"))
                continue
            m = compute_binary_metrics(labels[mask], probs[mask])
            m["n"] = int(mask.sum())
            _, ci_lo, ci_hi = bootstrap_ci(labels[mask], probs[mask], ids=ids_arr[mask])
            m["auc_ci_lo"], m["auc_ci_hi"] = ci_lo, ci_hi
            out[g] = m
        return out

    def _per_manipulation_vs_real(labels, probs, manip, ids) -> List[dict]:
        labels = np.asarray(labels); manip = np.asarray(manip); ids_arr = np.asarray(ids)
        # Real rows are identified by label==0, not a specific manipulation string --
        # OOD sets like CelebDF have more than one real-manipulation name
        # (celeb_real/youtube_real), which a single real_key would silently miss.
        real_mask = labels == 0
        if real_mask.sum() == 0:
            print("[error-analysis] no real (label==0) rows found; skipping per-manipulation-vs-real AUCs.")
            return []
        rows = []
        for m in sorted(set(manip[labels == 1].tolist())):
            mask = (manip == m) | real_mask
            sub_labels, sub_probs = labels[mask], probs[mask]
            if len(set(sub_labels.tolist())) < 2:
                continue
            metrics = compute_binary_metrics(sub_labels, sub_probs)
            _, lo, hi = bootstrap_ci(sub_labels, sub_probs, ids=ids_arr[mask])
            metrics["auc_ci_lo"], metrics["auc_ci_hi"] = lo, hi
            metrics["manipulation"] = m
            metrics["n_manip"] = int((manip == m).sum())
            metrics["n_real_paired"] = int(real_mask.sum())
            rows.append(metrics)
        return rows        

    def _plot_all_figures(labels, probs, per_dataset: dict):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
        plt.rcParams["figure.facecolor"] = "white"
        plt.rcParams["axes.facecolor"] = "white"

        def _save(fig, name):
            fig.savefig(FIG_DIR / f"{name}.png", dpi=150, bbox_inches="tight", facecolor="white")
            fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight", facecolor="white")
            plt.close(fig)

        if len(set(labels)) > 1:
            fpr, tpr, _ = roc_curve(labels, probs)
            fig, ax = plt.subplots(figsize=(5, 5)); ax.plot(fpr, tpr); ax.plot([0, 1], [0, 1], "--", color="gray")
            ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC Curve"); _save(fig, "roc_curve")

            prec, rec, _ = precision_recall_curve(labels, probs)
            fig, ax = plt.subplots(figsize=(5, 5)); ax.plot(rec, prec)
            ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("PR Curve"); _save(fig, "pr_curve")

        preds = (probs >= 0.5).astype(int)
        cm = confusion_matrix(labels, preds, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4, 4)); ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_title("Confusion Matrix")
        _save(fig, "confusion_matrix")

        bins = np.linspace(0, 1, 11)
        bin_idx = np.clip(np.digitize(probs, bins) - 1, 0, 9)
        bin_acc = [labels[bin_idx == i].mean() if (bin_idx == i).sum() > 0 else np.nan for i in range(10)]
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(bins[:-1] + 0.05, bin_acc, "o-"); ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set_xlabel("Confidence"); ax.set_ylabel("Empirical accuracy"); ax.set_title("Reliability Diagram")
        _save(fig, "reliability_diagram")

        names = [k for k in per_dataset if per_dataset[k].get("auc") == per_dataset[k].get("auc")]
        if names:
            fig, ax = plt.subplots(figsize=(max(5, len(names)), 4))
            ax.bar(names, [per_dataset[n]["auc"] for n in names]); ax.set_ylabel("AUC")
            ax.set_title("Per-domain AUC"); plt.xticks(rotation=30, ha="right")
            _save(fig, "per_domain_auc")

        # Real-vs-fake score separation -- shows how cleanly the classifier
        # separates the two classes beyond a single AUC number.
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(probs[labels == 0], bins=40, alpha=0.6, label="real", density=True)
        ax.hist(probs[labels == 1], bins=40, alpha=0.6, label="fake", density=True)
        ax.set_xlabel("Predicted P(fake)"); ax.set_ylabel("Density")
        ax.set_title("Score Distribution (Real vs. Fake)"); ax.legend()
        _save(fig, "score_distribution")

    def _plot_ood_diagnostic_figures(labels: np.ndarray, probs: np.ndarray, name: str):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
        plt.rcParams["figure.facecolor"] = "white"; plt.rcParams["axes.facecolor"] = "white"

        # ROC / PR curves -- previously only produced for the in-domain test set
        # (_plot_all_figures); OOD sets need the same pair for paper-figure parity.
        if len(set(labels)) > 1:
            fpr, tpr, _ = roc_curve(labels, probs)
            fig, ax = plt.subplots(figsize=(5, 5)); ax.plot(fpr, tpr); ax.plot([0, 1], [0, 1], "--", color="gray")
            ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title(f"ROC Curve — OOD:{name}")
            fig.savefig(FIG_DIR / f"roc_curve_ood_{name}.png", dpi=150, bbox_inches="tight", facecolor="white")
            fig.savefig(FIG_DIR / f"roc_curve_ood_{name}.pdf", bbox_inches="tight", facecolor="white")
            plt.close(fig)

            prec, rec, _ = precision_recall_curve(labels, probs)
            fig, ax = plt.subplots(figsize=(5, 5)); ax.plot(rec, prec)
            ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title(f"PR Curve — OOD:{name}")
            fig.savefig(FIG_DIR / f"pr_curve_ood_{name}.png", dpi=150, bbox_inches="tight", facecolor="white")
            fig.savefig(FIG_DIR / f"pr_curve_ood_{name}.pdf", bbox_inches="tight", facecolor="white")
            plt.close(fig)

        preds = (probs >= 0.5).astype(int)
        cm = confusion_matrix(labels, preds, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4, 4)); ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_title(f"Confusion Matrix — OOD:{name}")
        fig.savefig(FIG_DIR / f"confusion_matrix_ood_{name}.png", dpi=150, bbox_inches="tight", facecolor="white")
        fig.savefig(FIG_DIR / f"confusion_matrix_ood_{name}.pdf", bbox_inches="tight", facecolor="white")
        plt.close(fig)

        bins = np.linspace(0, 1, 11)
        bin_idx = np.clip(np.digitize(probs, bins) - 1, 0, 9)
        bin_acc = [labels[bin_idx == i].mean() if (bin_idx == i).sum() > 0 else np.nan for i in range(10)]
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(bins[:-1] + 0.05, bin_acc, "o-"); ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set_xlabel("Confidence"); ax.set_ylabel("Empirical accuracy")
        ax.set_title(f"Reliability Diagram — OOD:{name}")
        fig.savefig(FIG_DIR / f"reliability_diagram_ood_{name}.png", dpi=150, bbox_inches="tight", facecolor="white")
        fig.savefig(FIG_DIR / f"reliability_diagram_ood_{name}.pdf", bbox_inches="tight", facecolor="white")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(probs[labels == 0], bins=40, alpha=0.6, label="real", density=True)
        ax.hist(probs[labels == 1], bins=40, alpha=0.6, label="fake", density=True)
        ax.set_xlabel("Predicted P(fake)"); ax.set_ylabel("Density")
        ax.set_title(f"Score Distribution — OOD:{name}"); ax.legend()
        fig.savefig(FIG_DIR / f"score_distribution_ood_{name}.png", dpi=150, bbox_inches="tight", facecolor="white")
        fig.savefig(FIG_DIR / f"score_distribution_ood_{name}.pdf", bbox_inches="tight", facecolor="white")
        plt.close(fig)

    def _plot_cross_dataset_bar(ood_rows: List[dict], in_domain_auc: float, in_domain_ci: Tuple[float, float]):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["figure.facecolor"] = "white"; plt.rcParams["axes.facecolor"] = "white"
        labels_ = ["in-domain test"] + [r["ood_dataset"] for r in ood_rows]
        aucs = [in_domain_auc] + [r["auc"] for r in ood_rows]
        los = [in_domain_ci[0]] + [r["auc_ci_lo"] for r in ood_rows]
        his = [in_domain_ci[1]] + [r["auc_ci_hi"] for r in ood_rows]
        yerr = [[a - l for a, l in zip(aucs, los)], [h - a for a, h in zip(aucs, his)]]
        fig, ax = plt.subplots(figsize=(max(5, len(labels_) * 1.6), 4))
        ax.bar(labels_, aucs, yerr=yerr, capsize=4); ax.set_ylim(0, 1.02)
        ax.set_ylabel("AUC (95% bootstrap CI)"); ax.set_title("In-domain vs. Cross-dataset (OOD) Generalization")
        plt.xticks(rotation=20, ha="right")
        fig.savefig(FIG_DIR / "cross_dataset_auc_comparison.png", dpi=150, bbox_inches="tight", facecolor="white")
        fig.savefig(FIG_DIR / "cross_dataset_auc_comparison.pdf", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"[OOD] wrote cross_dataset_auc_comparison.png/pdf to {FIG_DIR}")

    def _find_best_threshold(labels: np.ndarray, probs: np.ndarray,
                              metric: str = "balanced_accuracy") -> Tuple[float, dict]:
        """Sweeps thresholds on VALIDATION data only. Never touches test labels --
        the whole point of a tuned threshold is that it's chosen without looking
        at the population it will be scored on."""
        best_t, best_score, best_metrics = 0.5, -1.0, None
        for t in np.linspace(0.01, 0.99, 99):
            m = compute_binary_metrics(labels, probs, threshold=t)
            score = m.get(metric, float("nan"))
            if score == score and score > best_score:
                best_score, best_t, best_metrics = score, float(t), m
        return best_t, (best_metrics or compute_binary_metrics(labels, probs, threshold=0.5))


    def run_domain_threshold_tuning(student: "nn.Module", val_loader: "DataLoader",
                                     test_raw: dict) -> List[dict]:
        """Issue 9: report each domain's performance at a threshold tuned on that
        domain's OWN validation split, not the blanket 0.5 default -- so the
        reader can tell whether FF++ has real discriminative power a bad
        threshold is hiding, or is genuinely at chance."""
        print("\n" + "#" * 78)
        print("# PER-DOMAIN THRESHOLD TUNING (tuned on validation split only)")
        print("#" * 78)
        student.eval()
        val_probs, val_labels, val_datasets = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                b = _to_device(batch)
                out = student(b)
                val_probs.append(torch.sigmoid(out["joint_logit"]).cpu().numpy())
                val_labels.append(b["label"].cpu().numpy())
                val_datasets.extend(b["dataset"])
        val_probs = np.concatenate(val_probs) if val_probs else np.array([])
        val_labels = np.concatenate(val_labels) if val_labels else np.array([])
        val_datasets = np.asarray(val_datasets)

        test_labels = np.asarray(test_raw["labels"]); test_probs = np.asarray(test_raw["probs"])
        test_datasets = np.asarray(test_raw["datasets"])

        rows = []
        for ds in sorted(set(val_datasets.tolist()) | set(test_datasets.tolist())):
            v_mask = val_datasets == ds
            t_mask = test_datasets == ds
            if v_mask.sum() < 10 or len(set(val_labels[v_mask].tolist())) < 2:
                print(f"[threshold-tuning] {ds}: insufficient/single-class validation rows "
                      f"(n={int(v_mask.sum())}); skipping.")
                continue
            if t_mask.sum() < 2 or len(set(test_labels[t_mask].tolist())) < 2:
                print(f"[threshold-tuning] {ds}: insufficient/single-class test rows; skipping.")
                continue
            best_t, _ = _find_best_threshold(val_labels[v_mask], val_probs[v_mask])
            at_tuned = compute_binary_metrics(test_labels[t_mask], test_probs[t_mask], threshold=best_t)
            at_default = compute_binary_metrics(test_labels[t_mask], test_probs[t_mask], threshold=0.5)
            rows.append(dict(
                dataset=ds, tuned_threshold=best_t, n_val=int(v_mask.sum()), n_test=int(t_mask.sum()),
                test_balanced_accuracy_at_0p5=at_default.get("balanced_accuracy"),
                test_balanced_accuracy_at_tuned=at_tuned.get("balanced_accuracy"),
                test_specificity_at_0p5=at_default.get("specificity"),
                test_specificity_at_tuned=at_tuned.get("specificity"),
                test_f1_at_0p5=at_default.get("f1"),
                test_f1_at_tuned=at_tuned.get("f1"),
                test_auc=at_tuned.get("auc"),  # AUC is threshold-independent
            ))
            print(f"[threshold-tuning] {ds}: tuned_threshold={best_t:.2f} -> balanced_accuracy "
                  f"{at_default.get('balanced_accuracy'):.4f} (@0.5) -> "
                  f"{at_tuned.get('balanced_accuracy'):.4f} (@tuned)")
        pd_safe_write_csv(rows, CSV_DIR / "domain_threshold_tuning.csv")
        return rows
        
    def run_embedding_tsne_visualization(student: "nn.Module", test_loader: "DataLoader",
                                          ood_loaders: Dict[str, "DataLoader"],
                                          max_points_per_set: int = 1500):
        """t-SNE of the student's joint fused representation, colored by
        real/fake and by source dataset -- shows whether OOD samples land
        inside or outside the in-domain decision regions, which is the usual
        Q1-reviewer request behind a bare cross-dataset AUC number."""
        print("\n" + "#" * 78)
        print("# EMBEDDING VISUALIZATION (t-SNE of joint_rep, in-domain + OOD)")
        print("#" * 78)
        try:
            from sklearn.manifold import TSNE
        except Exception as e:
            print(f"[tsne] scikit-learn TSNE unavailable ({e!r}); skipping.")
            return

        student.eval()
        all_embeds, all_labels, all_sources = [], [], []

        def _collect(loader, source_name, cap):
            n = 0
            with torch.no_grad():
                for batch in loader:
                    if n >= cap:
                        break
                    b = _to_device(batch)
                    out = student(b)
                    take = min(cap - n, out["joint_rep"].shape[0])
                    all_embeds.append(out["joint_rep"][:take].cpu().numpy())
                    all_labels.append(b["label"][:take].cpu().numpy())
                    all_sources.extend([source_name] * take)
                    n += take


        _collect(test_loader, "in-domain test", max_points_per_set)
        for name, loader in ood_loaders.items():
            _collect(loader, name, max_points_per_set)

        if not all_embeds:
            print("[tsne] no embeddings collected; skipping.")
            return

        X = np.concatenate(all_embeds, axis=0)
        y = np.concatenate(all_labels, axis=0)
        sources = np.array(all_sources)

        perplexity = min(30, max(5, X.shape[0] // 20))
        Z = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=SEED).fit_transform(X)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["figure.facecolor"] = "white"; plt.rcParams["axes.facecolor"] = "white"

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for label_val, marker, name in [(0, "o", "real"), (1, "^", "fake")]:
            mask = y == label_val
            axes[0].scatter(Z[mask, 0], Z[mask, 1], s=8, alpha=0.6, marker=marker, label=name)
        axes[0].set_title("t-SNE of joint representation (real vs. fake)"); axes[0].legend()

        for src in sorted(set(sources.tolist())):
            mask = sources == src
            axes[1].scatter(Z[mask, 0], Z[mask, 1], s=8, alpha=0.6, label=src)
        axes[1].set_title("t-SNE of joint representation (by source dataset)"); axes[1].legend()


        fig.savefig(FIG_DIR / "embedding_tsne.png", dpi=150, bbox_inches="tight", facecolor="white")
        fig.savefig(FIG_DIR / "embedding_tsne.pdf", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"[tsne] wrote embedding_tsne.png/pdf to {FIG_DIR} (n={X.shape[0]} points)")
    def run_comprehensive_evaluation_from_artifacts(student: "nn.Module",
                                                      artifact_reader: "TeacherArtifactReader",
                                                      usable: Dict[str, bool], scalers: Dict[str, TemperatureScaler],
                                                      test_loader: "DataLoader", ood_loaders: Dict[str, "DataLoader"]) -> dict:
        """Same metrics/CSVs/figures as before (pooled, per-domain, per-manip,
        per-regime, teacher_agreement.csv, OOD, ROC/PR/reliability-diagram figures).
        Student inference is unchanged. The per-teacher `teacher_agreement.csv` loop
        (Part 19/27) reads from `artifact_reader` instead of running a live teacher."""
        print("\n" + "#" * 78)
        print("# COMPREHENSIVE EVALUATION (student live, teachers from offline artifacts)")
        print("#" * 78)
        student.eval()
        all_probs, all_labels, all_ids, all_datasets, all_manip, all_regime = [], [], [], [], [], []
        teacher_probs = {n: [] for n in TEACHER_ORDER}
        teacher_labels = {n: [] for n in TEACHER_ORDER}
        teacher_ids = {n: [] for n in TEACHER_ORDER}
        with torch.no_grad():
            for bi, batch in enumerate(test_loader):
                if past_hard_deadline():
                    print(f"[eval] hard deadline reached at test batch {bi}; stopping early.")
                    break
                b = _to_device(batch)
                out = student(b)
                probs = torch.sigmoid(out["joint_logit"]).cpu().numpy()
                all_probs.append(probs); all_labels.append(b["label"].cpu().numpy())
                ids = batch["id"] if isinstance(batch["id"], list) else list(batch["id"])
                all_ids.extend(ids); all_datasets.extend(b["dataset"]); all_manip.extend(b["manipulation"])
                regime = [_regime_str(mf, mt, ma) for mf, mt, ma in
                          zip(b["mask_frame"].cpu().numpy(), b["mask_temporal"].cpu().numpy(),
                              b["mask_audio"].cpu().numpy())]
                all_regime.extend(regime)
                for name in TEACHER_ORDER:
                    if name not in artifact_reader.available_teachers():
                        continue
                    hit = artifact_reader.get_batch(name, ids)
                    if hit is None:
                        continue
                    present = hit["present"]
                    if present.sum() == 0:
                        continue
                    calibrated = scalers[name].apply(hit["logit"]) if name in scalers else hit["logit"]
                    prob = torch.sigmoid(calibrated)[present]
                    teacher_probs[name].append(prob.numpy())
                    teacher_labels[name].append(b["label"].cpu().numpy()[present.numpy()])
                    present_idx = present.nonzero(as_tuple=True)[0].tolist()
                    teacher_ids[name].extend([ids[i] for i in present_idx])

        labels = np.concatenate(all_labels); probs = np.concatenate(all_probs)
        pooled = compute_binary_metrics(labels, probs)
        mean_auc, lo, hi = bootstrap_ci(labels, probs, ids=all_ids)
        pooled["auc_ci_lo"], pooled["auc_ci_hi"] = lo, hi

        per_dataset = _grouped_metrics(labels, probs, all_datasets, all_ids)
        pd_safe_write_csv([dict(dataset=k, **v) for k, v in per_dataset.items()], CSV_DIR / "domain_metrics.csv")

        # Macro-average AUC across domains (Issue 1): the pooled (sample-weighted)
        # AUC above is dominated by whichever domain has the most rows -- with
        # FF++ at ~1.9% of the manifest, a near-chance FF++ AUC barely moves it.
        # Macro-AUC weights every domain equally, so a weak domain can't hide
        # behind a large easy one. Report both as co-headline numbers.
        domain_aucs = [v["auc"] for v in per_dataset.values() if v.get("auc") == v.get("auc")]
        pooled["macro_auc"] = float(np.mean(domain_aucs)) if domain_aucs else float("nan")
        pooled["macro_auc_weakest_domain"] = (
            min(per_dataset.items(), key=lambda kv: kv[1].get("auc", 1.0))[0] if domain_aucs else None)
        print("POOLED (pooled AUC + macro-AUC across domains):", pooled)
        save_json(CSV_DIR / "pooled_metrics.json", pooled)

# After
        # error_analysis.csv (grouped strictly by manipulation) is dropped: every
        # manipulation label is single-class, so its AUC column was always NaN.
        # per_manipulation_vs_real.csv is the correct, already-working replacement.
        manip_vs_real_rows = _per_manipulation_vs_real(labels, probs, all_manip, all_ids)
        pd_safe_write_csv(manip_vs_real_rows, CSV_DIR / "per_manipulation_vs_real.csv")
        per_manip = {r["manipulation"]: r for r in manip_vs_real_rows}

        per_regime = _grouped_metrics(labels, probs, all_regime, all_ids)
        pd_safe_write_csv([dict(regime=k, **v) for k, v in per_regime.items()], CSV_DIR / "threshold_analysis.csv")

        per_teacher_rows = []
        teacher_raw: Dict[str, dict] = {}
        for name in TEACHER_ORDER:
            if teacher_probs[name]:
                tp = np.concatenate(teacher_probs[name])
                tl = np.concatenate(teacher_labels[name])
                m = compute_binary_metrics(tl, tp)
                m["teacher"] = name
                m["n_modality_present"] = int(len(tl))
                per_teacher_rows.append(m)
                teacher_raw[name] = dict(ids=teacher_ids[name], labels=tl, probs=tp)
            else:
                print(f"[evaluate] {name}: no test rows had this teacher's artifact present; skipped.")
        pd_safe_write_csv(per_teacher_rows, CSV_DIR / "teacher_agreement.csv")

        first = True
        ood_summary_rows: List[dict] = []
        for ood_name, loader in ood_loaders.items():
            ood_probs, ood_labels, ood_ids, ood_manip = [], [], [], []
            with torch.no_grad():
                for batch in loader:
                    b = _to_device(batch)
                    out = student(b)
                    ood_probs.append(torch.sigmoid(out["joint_logit"]).cpu().numpy())
                    ood_labels.append(b["label"].cpu().numpy())
                    ids = batch["id"] if isinstance(batch["id"], list) else list(batch["id"])
                    ood_ids.extend(ids); ood_manip.extend(batch["manipulation"])
            if not ood_probs:
                print(f"[OOD:{ood_name}] loader produced 0 batches; skipping.")
                continue
            ol = np.concatenate(ood_labels); op = np.concatenate(ood_probs)
            m = compute_binary_metrics(ol, op)
            _, ci_lo, ci_hi = bootstrap_ci(ol, op, ids=ood_ids)
            m["auc_ci_lo"], m["auc_ci_hi"] = ci_lo, ci_hi
            m["ood_dataset"] = ood_name
            m["n"] = int(len(ol))
            ood_summary_rows.append(m)
            pd_safe_write_csv([m], CSV_DIR / "cross_dataset_metrics.csv", append=not first)
            first = False
            print(f"[OOD:{ood_name}] auc={m['auc']:.4f} (95% CI [{ci_lo:.4f}, {ci_hi:.4f}]) n={m['n']}")

# After
            # Same fix as in-domain: per-manipulation-only grouping is single-class
            # per group (e.g. celebdf's celeb_synthesis is 100% fake) -> AUC always
            # NaN. Pair each manipulation against this OOD set's real rows instead.
            manip_vs_real_ood = _per_manipulation_vs_real(ol, op, ood_manip, ood_ids)
            pd_safe_write_csv(
                [dict(ood_dataset=ood_name, **r) for r in manip_vs_real_ood],
                CSV_DIR / "cross_dataset_manipulation.csv", append=(ood_name != next(iter(ood_loaders))))
            _plot_ood_diagnostic_figures(ol, op, ood_name)

        if ood_summary_rows:
            _plot_cross_dataset_bar(ood_summary_rows, in_domain_auc=pooled["auc"],
                                     in_domain_ci=(pooled["auc_ci_lo"], pooled["auc_ci_hi"]))

        _plot_all_figures(labels, probs, per_dataset)
        return dict(pooled=pooled, per_dataset=per_dataset, per_manip=per_manip, per_regime=per_regime,
                    raw=dict(ids=all_ids, labels=labels, probs=probs, datasets=all_datasets),
                    teacher_raw=teacher_raw, per_teacher_metrics=per_teacher_rows,
                    ood_summary=ood_summary_rows)
        
    def train_and_eval_baselines_from_artifacts(artifact_reader: "TeacherArtifactReader",
                                                 usable: Dict[str, bool],
                                                 scalers: Dict[str, TemperatureScaler],
                                                 train_loader: "IDLabelBatchLoader",
                                                 test_loader: "IDLabelBatchLoader") -> Tuple[List[dict], Dict[str, dict]]:
        """Part 19 (revised): HGF trained on stored embeddings; late-fusion/best-single-teacher
        use stored logits. No teacher nn.Module anywhere in this function.
        NOW reports and returns raw (ids, labels, probs) on test_loader -- the SAME
        population the student is scored on in run_comprehensive_evaluation_from_artifacts --
        instead of val_loader, so the headline comparison and any downstream significance
        test compare like-for-like populations rather than two different sample sets."""
        print("\n" + "#" * 78)
        print("# BASELINE COMPARISONS (from offline teacher artifacts)")
        print("#" * 78)
        embed_dims = {n: artifact_reader.metadata[n]["feature_dim"]
                      for n in TEACHER_ORDER if n in artifact_reader.available_teachers()}
        if not embed_dims:
            print("[baselines] no usable teacher artifacts; skipping.")
            return [], {}

        hgf = HierarchicalGatedFusion(embed_dims).to(DEVICE)
        opt = torch.optim.AdamW(hgf.parameters(), lr=1e-3, weight_decay=0.01)

        def _epoch(loader, train, collect_ids=False):
            hgf.train(train)
            probs_all, labels_all, ids_all = [], [], []
            for batch in loader:
                ids = batch["id"] if isinstance(batch["id"], list) else list(batch["id"])
                labels = batch["label"]
                embeds, present = {}, {}
                skip = False
                for n in embed_dims:
                    hit = artifact_reader.get_batch(n, ids)
                    if hit is None:
                        skip = True
                        break
                    embeds[n] = hit["embed"].to(DEVICE)
                    present[n] = hit["present"].to(DEVICE)
                if skip:
                    continue
                logit, _ = hgf(embeds, present)
                loss = safe_bce_logits(logit, labels.to(DEVICE))
                if train:
                    opt.zero_grad(); loss.backward(); opt.step()
                probs_all.append(torch.sigmoid(logit).detach().cpu().numpy())
                labels_all.append(labels.numpy())
                if collect_ids:
                    ids_all.extend(ids)
            if not labels_all:
                return (np.array([]), np.array([]), []) if collect_ids else (np.array([]), np.array([]))
            labels_cat, probs_cat = np.concatenate(labels_all), np.concatenate(probs_all)
            return (labels_cat, probs_cat, ids_all) if collect_ids else (labels_cat, probs_cat)

        for ep in range(3):
            if past_hard_deadline():
                print(f"[baselines] hard deadline reached before baseline epoch {ep+1}/3; stopping early.")
                break
            _epoch(train_loader, True)
        test_labels, test_probs, hgf_ids = _epoch(test_loader, False, collect_ids=True)
        hgf_metrics = compute_binary_metrics(test_labels, test_probs) if len(test_labels) else {}
        print("HierarchicalGatedFusion (baseline B, test set):", hgf_metrics)

        late = SimpleLateFusion()
        late_probs_all, late_labels_all, late_ids = [], [], []
        for batch in test_loader:
            ids = batch["id"] if isinstance(batch["id"], list) else list(batch["id"])
            logits, present = {}, {}
            skip = False
            for n in embed_dims:
                hit = artifact_reader.get_batch(n, ids)
                if hit is None:
                    skip = True
                    break
                calibrated = scalers[n].apply(hit["logit"]) if n in scalers else hit["logit"]
                logits[n] = calibrated.to(DEVICE)
                present[n] = hit["present"].to(DEVICE)
            if skip:
                continue
            fused = late(logits, present)
            late_probs_all.append(torch.sigmoid(fused).cpu().numpy())
            late_labels_all.append(batch["label"].numpy())
            late_ids.extend(ids)
        late_labels = np.concatenate(late_labels_all) if late_labels_all else np.array([])
        late_probs = np.concatenate(late_probs_all) if late_probs_all else np.array([])
        late_metrics = compute_binary_metrics(late_labels, late_probs) if len(late_labels) else {}
        print("Simple late fusion (baseline A, test set):", late_metrics)

        best_single = {"name": None, "auc": -1.0}
        best_single_raw: Dict[str, Any] = {}
        for n in embed_dims:
            probs_all, labels_all, ids_all = [], [], []
            for batch in test_loader:
                ids = batch["id"] if isinstance(batch["id"], list) else list(batch["id"])
                hit = artifact_reader.get_batch(n, ids)
                if hit is None:
                    continue
                calibrated = scalers[n].apply(hit["logit"]) if n in scalers else hit["logit"]
                present = hit["present"]
                probs_all.append(torch.sigmoid(calibrated)[present].numpy())
                labels_all.append(batch["label"].numpy()[present.numpy()])
                ids_all.extend([ids[i] for i in present.nonzero(as_tuple=True)[0].tolist()])
            if not labels_all:
                continue
            # keep every metric field (pr_auc, accuracy, precision, ...), not just auc,
            # so best_single_teacher's row in baseline_comparison.csv isn't mostly blank
            m = compute_binary_metrics(np.concatenate(labels_all), np.concatenate(probs_all))
            if m["auc"] == m["auc"] and m["auc"] > best_single["auc"]:
                best_single = dict(name=n, **m)
                best_single_raw = dict(ids=ids_all, labels=np.concatenate(labels_all),
                                        probs=np.concatenate(probs_all))
        print("Best individual teacher (baseline, test set):", best_single)

        rows = [dict(baseline="best_single_teacher", **best_single),
                dict(baseline="simple_late_fusion", **late_metrics),
                dict(baseline="hierarchical_gated_fusion", **hgf_metrics)]
        pd_safe_write_csv(rows, CSV_DIR / "baseline_comparison.csv")

        raw_by_baseline = {
            "hierarchical_gated_fusion": dict(ids=hgf_ids, labels=test_labels, probs=test_probs),
            "simple_late_fusion": dict(ids=late_ids, labels=late_labels, probs=late_probs),
            "best_single_teacher": best_single_raw,
        }
        return rows, raw_by_baseline


    def run_significance_tests(student_raw: dict, baseline_raw: Dict[str, dict]) -> List[dict]:
        """Q1-reviewer-standard statistical comparison: is the student's AUC
        significantly different from each baseline's, on the SAME aligned test
        samples? Uses the paired DeLong test. Also writes a bar chart with 95%
        bootstrap CIs for the paper."""
        print("\n" + "#" * 78)
        print("# STATISTICAL SIGNIFICANCE: student vs. each baseline (paired DeLong test)")
        print("#" * 78)
        student_auc, student_lo, student_hi = bootstrap_ci(
            student_raw["labels"], student_raw["probs"], ids=student_raw["ids"])
        s_id_to_idx = {sid: i for i, sid in enumerate(student_raw["ids"])}
        rows = []
        plot_labels = ["student (full test set)"]
        plot_aucs = [student_auc]
        plot_los = [student_lo]
        plot_his = [student_hi]
        for name, b in baseline_raw.items():
            if not b or not b.get("ids"):
                print(f"[significance] {name}: no raw predictions available; skipping.")
                continue
            common = [sid for sid in b["ids"] if sid in s_id_to_idx]
            if len(common) < MIN_CALIBRATION_SAMPLES:
                print(f"[significance] {name}: only {len(common)} ids in common with the "
                      f"student's test set (< {MIN_CALIBRATION_SAMPLES}); skipping.")
                continue
            b_id_to_idx = {sid: i for i, sid in enumerate(b["ids"])}
            s_idx = [s_id_to_idx[sid] for sid in common]
            b_idx = [b_id_to_idx[sid] for sid in common]
            labels = student_raw["labels"][s_idx]
            s_probs_aligned = student_raw["probs"][s_idx]
            b_probs_aligned = np.asarray(b["probs"])[b_idx]
            p_delong = delong_test(labels, s_probs_aligned, b_probs_aligned)
            student_auc_common = compute_binary_metrics(labels, s_probs_aligned).get("auc", float("nan"))
            baseline_auc, b_lo, b_hi = bootstrap_ci(labels, b_probs_aligned, ids=common)
            student_preds_aligned = (s_probs_aligned >= 0.5).astype(int)
            baseline_preds_aligned = (b_probs_aligned >= 0.5).astype(int)
            p_mcnemar, mcnemar_b, mcnemar_c = mcnemar_test(labels, student_preds_aligned, baseline_preds_aligned)
            # Effect sizes, not just the p-value (Issue 7): at n in the tens of
            # thousands, McNemar's p-value is essentially guaranteed to be tiny
            # even for a trivial difference. The odds ratio (b/c) and the
            # proportion-based effect size g = |b-c|/(b+c) say how LARGE the
            # discordance actually is.
            mcnemar_odds_ratio = (mcnemar_b / mcnemar_c) if mcnemar_c > 0 else float("inf")
            mcnemar_effect_g = (abs(mcnemar_b - mcnemar_c) / (mcnemar_b + mcnemar_c)
                                 if (mcnemar_b + mcnemar_c) > 0 else float("nan"))
            rows.append(dict(baseline=name, n_common=len(common),
                              student_auc_on_common=student_auc_common, baseline_auc=baseline_auc,
                              baseline_auc_ci_lo=b_lo, baseline_auc_ci_hi=b_hi,
                              auc_diff=student_auc_common - baseline_auc,
                              delong_p_value=p_delong, mcnemar_p_value=p_mcnemar,
                              mcnemar_discordant_student_only=mcnemar_b,
                              mcnemar_discordant_baseline_only=mcnemar_c,
                              mcnemar_odds_ratio=mcnemar_odds_ratio,
                              mcnemar_effect_g=mcnemar_effect_g,
                              significant_at_0p05=bool(p_delong == p_delong and p_delong < 0.05)))
            print(f"[significance] student vs {name}: n={len(common)} "
                  f"student_auc={student_auc_common:.4f} baseline_auc={baseline_auc:.4f} "
                  f"diff={student_auc_common-baseline_auc:+.4f} DeLong p={p_delong:.4g} "
                  f"McNemar p={p_mcnemar:.4g} odds_ratio={mcnemar_odds_ratio:.3g} effect_g={mcnemar_effect_g:.3g}")
            plot_labels.append(name); plot_aucs.append(baseline_auc)
            plot_los.append(b_lo); plot_his.append(b_hi)
        pd_safe_write_csv(rows, CSV_DIR / "significance_tests.csv")
        if len(plot_labels) > 1:
            _plot_auc_comparison_with_ci(plot_labels, plot_aucs, plot_los, plot_his)
        return rows
    def compare_two_prediction_sets(name: str, ids_a, labels, probs_a, ids_b, probs_b,
                                     min_common: int = MIN_CALIBRATION_SAMPLES) -> Optional[dict]:
        """Generic paired comparison (DeLong + McNemar + bootstrap CI) between any
        two prediction sets sharing sample ids/labels -- e.g. the full model vs. a
        leave-one-teacher-out ablation retrain, evaluated on the SAME test split.
        This is the paired test Issue 4/7 asks for instead of "the two 95% CIs
        don't overlap," which is sufficient but not necessary for significance."""
        id_to_idx_a = {sid: i for i, sid in enumerate(ids_a)}
        id_to_idx_b = {sid: i for i, sid in enumerate(ids_b)}
        common = [sid for sid in ids_a if sid in id_to_idx_b]
        if len(common) < min_common:
            print(f"[paired-comparison] {name}: only {len(common)} common ids (< {min_common}); skipping.")
            return None
        idx_a = [id_to_idx_a[sid] for sid in common]
        idx_b = [id_to_idx_b[sid] for sid in common]
        lbl = np.asarray(labels)[idx_a]
        pa = np.asarray(probs_a)[idx_a]
        pb = np.asarray(probs_b)[idx_b]
        p_delong = delong_test(lbl, pa, pb)
        auc_a, ci_lo_a, ci_hi_a = bootstrap_ci(lbl, pa, ids=common)
        auc_b, ci_lo_b, ci_hi_b = bootstrap_ci(lbl, pb, ids=common)
        preds_a = (pa >= 0.5).astype(int); preds_b = (pb >= 0.5).astype(int)
        p_mcnemar, b_count, c_count = mcnemar_test(lbl, preds_a, preds_b)
        return dict(comparison=name, n_common=len(common),
                    auc_a=auc_a, auc_a_ci_lo=ci_lo_a, auc_a_ci_hi=ci_hi_a,
                    auc_b=auc_b, auc_b_ci_lo=ci_lo_b, auc_b_ci_hi=ci_hi_b,
                    auc_diff=auc_a - auc_b, delong_p_value=p_delong,
                    mcnemar_p_value=p_mcnemar, mcnemar_discordant_a_only=b_count,
                    mcnemar_discordant_b_only=c_count,
                    mcnemar_odds_ratio=(b_count / c_count if c_count > 0 else float("inf")),
                    ci_overlap=not (ci_hi_b < ci_lo_a or ci_hi_a < ci_lo_b))



    def run_ablation_paired_significance(full_model_raw: dict) -> List[dict]:
        """Loads .npz raw predictions from separately-run ablation sessions (see
        save_raw_predictions_npz) and runs a proper paired test against this run's
        full-model predictions on the same test population. No-op if
        ABLATION_RAW_PREDICTIONS_PATHS is empty."""
        if not ABLATION_RAW_PREDICTIONS_PATHS:
            print("[ablation-significance] ABLATION_RAW_PREDICTIONS_PATHS is empty; skipping.")
            return []
        print("\n" + "#" * 78)
        print("# ABLATION PAIRED SIGNIFICANCE (DeLong + McNemar vs. saved ablation runs)")
        print("#" * 78)
        rows = []
        for name, npz_path in ABLATION_RAW_PREDICTIONS_PATHS.items():
            p = Path(npz_path)
            if not p.exists():
                print(f"[ablation-significance] {name}: {p} not found; skipping.")
                continue
            data = np.load(p, allow_pickle=True)
            result = compare_two_prediction_sets(
                name=f"ablation:{name}", ids_a=full_model_raw["ids"], labels=full_model_raw["labels"],
                probs_a=full_model_raw["probs"], ids_b=data["ids"].tolist(), probs_b=data["probs"])
            if result:
                rows.append(result)
                print(f"[ablation-significance] full model vs {name}: n={result['n_common']} "
                      f"auc_diff={result['auc_diff']:+.4f} DeLong p={result['delong_p_value']:.4g} "
                      f"McNemar p={result['mcnemar_p_value']:.4g} CIs overlap={result['ci_overlap']}")
        pd_safe_write_csv(rows, CSV_DIR / "ablation_paired_significance.csv")
        return rows


    def save_raw_predictions_npz(raw: dict, path: Path):

        """Saves this run's (ids, labels, probs) so a LATER ablation session's own
        EVALUATE run can be paired-compared against it via
        compare_two_prediction_sets(), instead of relying on CI non-overlap."""
        np.savez(path, ids=np.asarray(raw["ids"]), labels=np.asarray(raw["labels"]),
                 probs=np.asarray(raw["probs"]))
        print(f"[main] saved raw predictions for future paired ablation comparisons -> {path}")

    def _plot_auc_comparison_with_ci(labels: List[str], aucs: List[float],
                                      los: List[float], his: List[float]):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["figure.facecolor"] = "white"
        plt.rcParams["axes.facecolor"] = "white"
        yerr = [[a - l for a, l in zip(aucs, los)], [h - a for a, h in zip(aucs, his)]]
        fig, ax = plt.subplots(figsize=(max(5, len(labels) * 1.3), 4))
        ax.bar(labels, aucs, yerr=yerr, capsize=4)
        ax.set_ylabel("AUC (95% bootstrap CI)")
        ax.set_title("Student vs. Baselines: Test-Set AUC")
        plt.xticks(rotation=20, ha="right")
        fig.savefig(FIG_DIR / "auc_comparison_ci.png", dpi=150, bbox_inches="tight", facecolor="white")
        fig.savefig(FIG_DIR / "auc_comparison_ci.pdf", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"[significance] wrote auc_comparison_ci.png/pdf to {FIG_DIR}")    

    def run_all_ablations_from_artifacts(student: "nn.Module", artifact_reader: "TeacherArtifactReader",
                                          usable: Dict[str, bool], val_loader: "DataLoader"):
        """Part 20 (corrected): a real per-teacher-subset ablation requires
        SEPARATE TRAIN_STUDENT retrains, one per subset -- KD influence is baked
        into the trained weights at training time, not something the already-
        trained student can selectively "un-learn" at inference. The previous
        version of this function looped over configs and called the SAME
        forward pass every time with no parameter tying it to `active`,
        silently producing four identical, misleadingly-labeled rows. Disabled
        here rather than left to produce that trap the first time someone
        enables RUN_ABLATIONS and reports the CSV."""
        if not RUN_ABLATIONS:
            print("[ablations] RUN_ABLATIONS=False; skipping (opt-in, expensive).")
            return
        print("\n" + "#" * 78)
        print("# ABLATIONS -- NOT IMPLEMENTED AS A SINGLE-MODEL SWEEP")
        print("#" * 78)
        print("[ablations] A genuine per-teacher ablation requires one full TRAIN_STUDENT "
              "retrain per teacher subset (set `usable[name]=False` for excluded teachers "
              "before building KDTrainer), since KD influence is fixed into the weights at "
              "training time and cannot be toggled on an already-trained student at "
              "inference. Refusing to write a misleading ablation_results.csv. To run a "
              "real ablation: repeat MODE='TRAIN_STUDENT' with a restricted TEACHER_ORDER "
              "or `usable` dict for each configuration you want to compare, each as its "
              "own checkpoint, then compare their EVALUATE results.")
        return

    def run_efficiency_comparison(student: "nn.Module") -> List[dict]:
        print("\n" + "#" * 78)
        print("# MODEL EFFICIENCY / SIZE COMPARISON (Part 30, opt-in)")
        print("#" * 78)
        rows = []

        def _measure(name: str, model: "nn.Module", make_input, n_runs: int = 30, call_fn=None):
            model = model.to(DEVICE).eval()
            n_params = sum(p.numel() for p in model.parameters())
            # Teacher classes (DualPathB2Student / StudentEfficientNetB2 / VideoSwinFF /
            # WavLMAAIST) never define forward() -- they only expose .logit()/.embed()/
            # .infer_all(). Calling model(x) on them hits nn.Module's default forward and
            # raises NotImplementedError. call_fn lets callers route to the right method
            # per model type; it defaults to model(x) for models (like MultimodalStudent)
            # that do implement forward().
            call = call_fn if call_fn is not None else (lambda m, inp: m(inp))
            with torch.no_grad():
                x = make_input()
                for _ in range(5):
                    call(model, x)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(n_runs):
                    call(model, x)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                ms_per_batch = (time.time() - t0) / n_runs * 1000.0
            rows.append(dict(model=name, n_params=n_params, n_params_millions=n_params / 1e6,
                              ms_per_batch=ms_per_batch, device=str(DEVICE)))
            print(f"[efficiency] {name}: {n_params/1e6:.2f}M params, {ms_per_batch:.2f} ms/batch")
            del model
            _free_gpu_memory(f"efficiency:{name}")

        _measure("student_full_pipeline", student,
                  lambda: dict(frame=torch.randn(8, 3, 224, 224, device=DEVICE),
                               clip=torch.randn(8, 3, 16, 224, 224, device=DEVICE),
                               audio=torch.randn(8, 64000, device=DEVICE),
                               mask_frame=torch.ones(8, dtype=torch.bool, device=DEVICE),
                               mask_temporal=torch.ones(8, dtype=torch.bool, device=DEVICE),
                               mask_audio=torch.ones(8, dtype=torch.bool, device=DEVICE)))
        _measure("frame_dfdc_teacher", DualPathB2Student(),
                  lambda: torch.randn(8, 3, 224, 224, device=DEVICE),
                  call_fn=lambda m, x: m.logit(x))
        _measure("frame_diff_teacher", StudentEfficientNetB2(),
                  lambda: torch.randn(8, 3, 224, 224, device=DEVICE),
                  call_fn=lambda m, x: m.logit(x))
        _measure("video_ff_teacher", VideoSwinFF(),
                  lambda: torch.randn(8, 3, 16, 224, 224, device=DEVICE),
                  call_fn=lambda m, x: m.logit(x))
        _measure("audio_lavdf_teacher", WavLMAAIST(AudioCfg(use_hst_graph=True)),
                  lambda: torch.randn(8, 64000, device=DEVICE),
                  call_fn=lambda m, x: m.logit(x))

        teacher_rows = [r for r in rows if r["model"].endswith("_teacher")]
        rows.append(dict(model="all_4_teachers_combined (HGF-baseline inference cost)",
                          n_params=sum(r["n_params"] for r in teacher_rows),
                          n_params_millions=sum(r["n_params"] for r in teacher_rows) / 1e6,
                          ms_per_batch=sum(r["ms_per_batch"] for r in teacher_rows),
                          device=str(DEVICE)))
        pd_safe_write_csv(rows, CSV_DIR / "efficiency_comparison.csv")
        return rows    

    def write_efficiency_reduction_summary(efficiency_rows: List[dict],
                                            per_teacher_metrics: List[dict]) -> List[dict]:
        """Issue 11: the headline reduction ratio against the SUM of all four
        teachers assumes a deployment that runs every teacher on every sample,
        which the domain-aware router never does. Report both framings honestly:
        against the sum of all teachers, and against the single best teacher by
        measured in-domain AUC (teacher_agreement.csv)."""
        by_name = {r["model"]: r for r in efficiency_rows}
        student_row = by_name.get("student_full_pipeline")
        combined_row = next((r for r in efficiency_rows if r["model"].startswith("all_")), None)
        if not student_row or not combined_row:
            print("[efficiency] student or combined-teachers row missing; skipping reduction summary.")
            return []

        best_teacher_name = (max(per_teacher_metrics, key=lambda m: m.get("auc", -1)).get("teacher")
                              if per_teacher_metrics else None)
        best_teacher_row = by_name.get(f"{best_teacher_name}_teacher") if best_teacher_name else None
        best_teacher_auc = next((m["auc"] for m in per_teacher_metrics
                                  if m.get("teacher") == best_teacher_name), None)

        rows = [dict(
            comparison="student vs. sum of all teachers",
            student_params_m=student_row["n_params_millions"],
            baseline_params_m=combined_row["n_params_millions"],
            param_reduction_x=combined_row["n_params_millions"] / student_row["n_params_millions"],
            student_ms_per_batch=student_row["ms_per_batch"],
            baseline_ms_per_batch=combined_row["ms_per_batch"],
            latency_reduction_x=combined_row["ms_per_batch"] / student_row["ms_per_batch"],
            note="Assumes running all four teachers on every sample; the domain-aware "
                 "router never actually does this.",
        )]
        if best_teacher_row:

            rows.append(dict(
                comparison=f"student vs. best single teacher ({best_teacher_name}, AUC={best_teacher_auc})",
                student_params_m=student_row["n_params_millions"],
                baseline_params_m=best_teacher_row["n_params_millions"],
                param_reduction_x=best_teacher_row["n_params_millions"] / student_row["n_params_millions"],
                student_ms_per_batch=student_row["ms_per_batch"],
                baseline_ms_per_batch=best_teacher_row["ms_per_batch"],
                latency_reduction_x=best_teacher_row["ms_per_batch"] / student_row["ms_per_batch"],
                note="The fairer single-teacher comparison a router-based deployment would face.",
            ))
        else:
            print("[efficiency] could not identify the best single teacher's efficiency row; "
                  "only the sum-of-teachers framing was written.")
        pd_safe_write_csv(rows, CSV_DIR / "efficiency_reduction_summary.csv")
        return rows
        
    # ══════════════════════════════════════════════════════════════════════
    # 16. MAIN ORCHESTRATION — mode-separated (Part 16/21/29)
    # ══════════════════════════════════════════════════════════════════════

    def write_paper_summary_table(pooled: dict, per_dataset: dict, ood_rows: List[dict], out_path: Path):
        """One row per evaluation population (in-domain pooled, each in-domain
        domain, each OOD dataset) with the numbers a Q1 results table actually
        reports -- AUC (95% CI), accuracy, F1, EER -- so Table 2/3 numbers don't
        have to be hand-copied out of six separate files."""
        rows = []
        rows.append(dict(population="in-domain (pooled test)", n=None,
                          auc=pooled.get("auc"), auc_ci_lo=pooled.get("auc_ci_lo"),
                          auc_ci_hi=pooled.get("auc_ci_hi"), accuracy=pooled.get("accuracy"),
                          f1=pooled.get("f1"), eer=pooled.get("eer")))
        for name, m in per_dataset.items():
            rows.append(dict(population=f"in-domain / {name}", n=m.get("n"),
                              auc=m.get("auc"), auc_ci_lo=m.get("auc_ci_lo"),
                              auc_ci_hi=m.get("auc_ci_hi"), accuracy=m.get("accuracy"),
                              f1=m.get("f1"), eer=m.get("eer")))
        for m in ood_rows:
            rows.append(dict(population=f"OOD / {m.get('ood_dataset')}", n=m.get("n"),
                              auc=m.get("auc"), auc_ci_lo=m.get("auc_ci_lo"),
                              auc_ci_hi=m.get("auc_ci_hi"), accuracy=m.get("accuracy"),
                              f1=m.get("f1"), eer=m.get("eer")))
        pd_safe_write_csv(rows, out_path)
        print(f"[main] wrote paper_summary_table.csv -> {out_path}")

    def write_reproducibility_metadata(out_dir: Path):
        """Issue 17: record the exact seed, whether this is a single run or an
        average of several, hyperparameter provenance, and compute environment,
        so a reviewer or re-implementer isn't left guessing."""
        import platform
        meta: Dict[str, Any] = dict(
            seed=SEED, run_tag=RUN_TAG,
            n_runs="single-seed (SEED=42); no multi-seed averaging was performed for the "
                   "numbers in pooled_metrics.json / domain_metrics.csv / cross_dataset_metrics.csv. "
                   "State this explicitly in the paper rather than implying multiple runs.",
            hyperparameter_search="LOSS_WEIGHTS, KD_TEMPERATURE, BASE_LR, and the stage/epoch "
                   "schedule below were set manually from prior pipeline iterations, not "
                   "selected via a formal search on a held-out split. State this plainly in "
                   "the training-protocol section.",
            loss_weights=LOSS_WEIGHTS, kd_temperature=KD_TEMPERATURE, base_lr=BASE_LR,
            stage_epochs=STAGE_EPOCHS_V2, stage_batch_size=STAGE_BATCH_SIZE_V2,
            stage_train_pool_max=STAGE_TRAIN_POOL_MAX_V2, teacher_order=TEACHER_ORDER,
            compute=dict(
                device=str(DEVICE),
                gpu_name=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
                torch_version=torch.__version__,
                cuda_version=(torch.version.cuda if torch.cuda.is_available() else None),
                python_version=platform.python_version(),
            ),
        )
        training_history_path = CSV_DIR / "training_history.csv"
        if training_history_path.exists():
            import pandas as pd
            hist = pd.read_csv(training_history_path)
            if "duration_s" in hist.columns:
                meta["training_wall_clock_hours_this_history"] = round(float(hist["duration_s"].sum()) / 3600.0, 3)
            meta["training_epochs_recorded"] = int(len(hist))

        else:
            meta["training_wall_clock_hours_this_history"] = None
            print("[reproducibility] training_history.csv not found under CSV_DIR; copy it in "
                  "from the TRAIN_STUDENT session, or fill this field in by hand for the paper.")
        save_json(out_dir / "reproducibility_metadata.json", meta)
        print(f"[main] wrote reproducibility_metadata.json -> {out_dir / 'reproducibility_metadata.json'}")
        
    def write_final_report(out_dir: Path):
        """Q1-paper convenience: gather every EVALUATE-mode metric already written
        under CSV_DIR/OUT_DIR into ONE consolidated JSON, so numbers for the paper
        don't have to be re-derived by hand from six separate files."""
        import pandas as pd

        def _load_json_safe(p):
            return json.load(open(p)) if p.exists() else None

        def _load_csv_safe(p):
            return pd.read_csv(p).to_dict(orient="records") if p.exists() else None

        report: Dict[str, Any] = dict(
            pooled_metrics=_load_json_safe(CSV_DIR / "pooled_metrics.json"),
            domain_metrics=_load_csv_safe(CSV_DIR / "domain_metrics.csv"),
            # error_analysis.csv removed -- see per_manipulation_vs_real for the
            # correct (class-balanced) per-manipulation numbers.
            per_manipulation_vs_real=_load_csv_safe(CSV_DIR / "per_manipulation_vs_real.csv"),
            regime_metrics=_load_csv_safe(CSV_DIR / "threshold_analysis.csv"),
            teacher_agreement=_load_csv_safe(CSV_DIR / "teacher_agreement.csv"),
            cross_dataset_metrics=_load_csv_safe(CSV_DIR / "cross_dataset_metrics.csv"),
            cross_dataset_manipulation=_load_csv_safe(CSV_DIR / "cross_dataset_manipulation.csv"),
            baseline_comparison=_load_csv_safe(CSV_DIR / "baseline_comparison.csv"),
            significance_tests=_load_csv_safe(CSV_DIR / "significance_tests.csv"),
            efficiency_comparison=_load_csv_safe(CSV_DIR / "efficiency_comparison.csv"),
            efficiency_reduction_summary=_load_csv_safe(CSV_DIR / "efficiency_reduction_summary.csv"),
            domain_threshold_tuning=_load_csv_safe(CSV_DIR / "domain_threshold_tuning.csv"),
            ablation_paired_significance=_load_csv_safe(CSV_DIR / "ablation_paired_significance.csv"),
            paper_summary_table=_load_csv_safe(CSV_DIR / "paper_summary_table.csv"),
            manifest_audit=_load_json_safe(out_dir / "manifest_audit.json"),
            teacher_temperatures=_load_json_safe(out_dir / "teacher_temperatures.json"),
            precompute_cost=_load_json_safe(out_dir / "precompute_cost.json"),
            reproducibility_metadata=_load_json_safe(out_dir / "reproducibility_metadata.json"),
        )
        path = out_dir / "final_report.json"
        save_json(path, report)
        print(f"[main] wrote consolidated final_report.json -> {path}")


    
# AFTER
    def write_ablation_config():
        """Every CSV/figure/checkpoint this run produces should be traceable to
        exactly which teacher set produced it -- otherwise an ablation run's
        numbers are indistinguishable from the full model's once pulled out of
        OUT_DIR for the paper. Written unconditionally, for all three MODEs."""
        config = dict(
            run_tag=RUN_TAG, mode=MODE, teacher_order=list(TEACHER_ORDER),
            teacher_branch=dict(TEACHER_BRANCH), loss_weights=dict(LOSS_WEIGHTS),
            kd_temperature=KD_TEMPERATURE, stage_epochs=dict(STAGE_EPOCHS_V2),
            stage_batch_size=dict(STAGE_BATCH_SIZE_V2), seed=SEED,
            resume_ckpt_path=RESUME_CKPT_PATH or None,
            best_ckpt_resume_input_path=BEST_CKPT_RESUME_INPUT_PATH or None,
            artifact_resume_input_dir=ARTIFACT_RESUME_INPUT_DIR or None,
        )
        out_path = OUT_DIR / "ablation_config.json"
        save_json(out_path, config)
        print(f"[main] wrote ablation_config.json (run_tag={RUN_TAG!r}, "
              f"teachers={TEACHER_ORDER}) -> {out_path}")


    def main():
        print("\n" + "=" * 78)
        print(" QUADKD OFFLINE MULTI-TEACHER KD PIPELINE — MODE-SEPARATED REDESIGN")
        print(f" MODE = {MODE}  |  RUN_TAG = {RUN_TAG}  |  TEACHER_ORDER = {TEACHER_ORDER}")
        print("=" * 78)
        write_ablation_config()

        if RUN_INSPECTION:
            run_all_checkpoint_inspections()

        if not TORCH_OK:
            print("[main] torch unavailable; stopping after inspection/audit-only utilities.")
            rows = load_manifest(MANIFEST_PATH)
            audit_manifest(rows)
            return

        if MODE not in ("PRECOMPUTE_TEACHERS", "TRAIN_STUDENT", "EVALUATE"):
            raise ValueError(f"[main] MODE={MODE!r} is not one of "
                              f"'PRECOMPUTE_TEACHERS' / 'TRAIN_STUDENT' / 'EVALUATE'.")

        all_rows = load_manifest(MANIFEST_PATH)
        if MODE == "EVALUATE":
            ood_path = Path(OOD_MANIFEST_PATH)
            if REBUILD_OOD_MANIFEST_IF_PRESENT or not ood_path.exists():
                build_ood_manifest(CELEBDF_ROOT, WILDDEEPFAKE_ROOT, str(ood_path))
            if ood_path.exists():
                ood_rows = load_manifest(str(ood_path))
                print(f"[main] merged {len(ood_rows)} OOD rows from {ood_path} "
                      f"(datasets={sorted(set(r.get('dataset') for r in ood_rows))})")
                in_domain_ids = {r.get("id", "") for r in all_rows}
                ood_ids = {r.get("id", "") for r in ood_rows}
                id_collision = in_domain_ids & ood_ids
                if id_collision:
                    raise RuntimeError(
                        f"[main] {len(id_collision)} OOD row id(s) collide with in-domain manifest "
                        f"ids (e.g. {list(id_collision)[:5]}) -- build_ood_manifest.py's md5-based ids "
                        f"should never collide with the in-domain manifest; investigate before trusting "
                        f"any cross-dataset numbers from this run."
                    )
                non_ood_split = [r for r in ood_rows if r.get("split") != "ood"]
                if non_ood_split:
                    raise RuntimeError(
                        f"[main] {len(non_ood_split)} row(s) in {ood_path} do not have split=='ood' "
                        f"-- build_data_loaders() buckets rows by the 'split' field, so a wrong value "
                        f"here would silently leak these rows into train/val/test instead of the OOD "
                        f"loaders."
                    )
                all_rows = all_rows + ood_rows
            else:
                print(f"[main] {ood_path} not found; EVALUATE will run without cross-dataset "
                      f"OOD sets. Run build_ood_manifest.py first for celebdf/wilddeepfake numbers.")
        audit_manifest(all_rows)
        _preflight_check_decoders(all_rows)
        # Restore any previously-saved shards BEFORE either mode touches ARTIFACT_DIR --
        # needed for PRECOMPUTE_TEACHERS to resume, and for TRAIN_STUDENT/EVALUATE to
        # find artifacts at all once precompute has moved to a later session.
        _maybe_restore_artifact_dir_from_input()

        # ==================================================================
        # MODE 1: offline teacher precomputation (Part 1/16). Exits before
        # touching the student at all.
        # ==================================================================
        if MODE == "PRECOMPUTE_TEACHERS":
            precompute_teachers(all_rows)
            print("\n[main] DONE (PRECOMPUTE_TEACHERS). Re-run this mode until every teacher "
                  "reports 0 remaining rows, then switch MODE to 'TRAIN_STUDENT'.")
            return

        # verify offline artifacts exist and are consistent BEFORE building any
        # loaders that would otherwise silently substitute zeros (Part 25).
        verify_artifacts_before_training(all_rows)
        artifact_reader = TeacherArtifactReader(ARTIFACT_DIR)

        train_rows, val_loader, test_loader, calib_loader, baseline_train_loader, test_id_loader, ood_loaders = \
            build_data_loaders(all_rows)

        scalers = load_teacher_temperatures_from_artifacts(artifact_reader, calib_loader)

        fusion_dim = 512
        embed_dims = {n: artifact_reader.metadata[n]["feature_dim"]
                      for n in TEACHER_ORDER if n in artifact_reader.available_teachers()}
        usable = {n: (n in embed_dims) for n in TEACHER_ORDER}
        feature_projections = nn.ModuleDict({
            n: nn.Sequential(nn.LayerNorm(embed_dims[n]), nn.Linear(embed_dims[n], fusion_dim))
            for n in embed_dims
        }).to(DEVICE)
        temporal_proj = nn.Linear(fusion_dim, 768).to(DEVICE)

        student = MultimodalStudent(fusion_dim=fusion_dim, n_sampled_frames=4).to(DEVICE)
        router = DomainPriorRouter(TEACHER_ORDER, build_default_domain_prior())

        # ==================================================================
        # MODE 2: student training only (Part 16). NEVER runs comprehensive
        # evaluation, bootstrap, OOD, figures, baselines, or ablations.
        # ==================================================================
        if MODE == "TRAIN_STUDENT":
            trainer = KDTrainer(student, artifact_reader, usable, scalers, router, feature_projections,
                                 temporal_proj, train_rows, val_loader, BUDGET)
            if trainer.stage is None:
                print("[main] resumed checkpoint has all stages already complete; "
                      "skipping the dry-run timing probe and feasibility gate (nothing "
                      "left to train) and going straight to the final val pass / EMA eval.")
            else:
                timing = _dry_run_timing_check(trainer)
                if not run_feasibility_gate(trainer, timing):
                    print("[main] feasibility gate failed; stopping before any real training epoch.")
                    return
            trainer.run()
            router.flush_weight_log(CSV_DIR / "teacher_router_weights_by_domain_epoch.csv")
            print("\n[main] DONE (TRAIN_STUDENT). Checkpoints under:", CKPT_DIR,
                  "\n[main] Switch MODE to 'EVALUATE' in a dedicated session to run the full "
                  "test/OOD/baseline evaluation suite against the saved best_auc checkpoint.")
            return

        # ==================================================================
        # MODE 3: evaluation only (Part 16/21/27). Loads the best trained
        # student checkpoint; never trains.
        # ==================================================================
        if MODE == "EVALUATE":
            _maybe_restore_best_checkpoint_from_input()
            # best_ema_auc.pth can record a higher validated AUC than best_auc.pth
            # (self.best_auc is only ever updated by per-epoch val, never by the
            # one-time end-of-run EMA eval) -- load whichever file's recorded AUC
            # is actually higher instead of hardcoding best_auc.pth.
            candidates = []
            for path, key in ((CKPT_DIR / "best_auc.pth", "best_auc"),
                               (CKPT_DIR / "best_ema_auc.pth", "best_ema_auc")):
                if path.exists():
                    raw = torch.load(path, map_location="cpu", weights_only=False)
                    candidates.append((raw.get(key, -1.0), path, raw))
            if not candidates:
                raise RuntimeError(f"[main] neither best_auc.pth nor best_ema_auc.pth found under "
                                    f"{CKPT_DIR}; run MODE='TRAIN_STUDENT' to completion "
                                    f"(or at least one improving epoch) first.")
            candidates.sort(key=lambda c: c[0])
            chosen_auc, best_path, payload = candidates[-1]
            print(f"[main] checkpoint candidates: " +
                  ", ".join(f"{p.name}(recorded_auc={a})" for a, p, _ in candidates) +
                  f" -> choosing {best_path.name}")
            _assert_checkpoint_matches_current_architecture(payload, student)
            print(f"[main] LOAD_EMA_WEIGHTS_FOR_EVAL={LOAD_EMA_WEIGHTS_FOR_EVAL}; "
                  f"checkpoint records best_auc={payload.get('best_auc')} "
                  f"best_ema_auc={payload.get('best_ema_auc')}")
            if LOAD_EMA_WEIGHTS_FOR_EVAL and "ema" in payload:
                if payload.get("best_ema_auc", -1.0) == -1.0:
                    print("[main] WARNING: best_ema_auc is -1.0 (never validated) -- "
                          "these EMA weights have NO known score. Consider setting "
                          "LOAD_EMA_WEIGHTS_FOR_EVAL=False to load the raw student instead.")
                student.load_state_dict(payload["ema"])
                print(f"[main] loaded EMA weights from checkpoint for evaluation "
                      f"(recorded EMA val AUC={payload.get('best_ema_auc')})")
            else:
                student.load_state_dict(payload["student"])
                print(f"[main] loaded raw student checkpoint for evaluation (val AUC={payload.get('best_auc')})")

            if IS_SYNTHETIC_MANIFEST:
                print("\n" + "!" * 78)
                print("! WARNING: MANIFEST_PATH was not found and this run used the SYNTHETIC ")
                print("! self-test manifest. Every metric below is MEANINGLESS.")
                print("!" * 78 + "\n")

            if past_hard_deadline():
                print("[main] hard deadline already reached at EVALUATE mode start; "
                      "exiting without producing (unreliable) partial evaluation output. "
                      "Re-run MODE='EVALUATE' in a fresh session.")
                return

            results = run_comprehensive_evaluation_from_artifacts(
                student, artifact_reader, usable, scalers, test_loader, ood_loaders)
            baseline_rows, baseline_raw = train_and_eval_baselines_from_artifacts(
                artifact_reader, usable, scalers, baseline_train_loader, test_id_loader)
            # Compare the student against EVERY individual teacher (not just the
            # strongest one) so the paper can report "beats all four teachers,"
            # each backed by its own paired DeLong/McNemar test.
            comparison_raw = dict(baseline_raw)
            for teacher_name, raw in results.get("teacher_raw", {}).items():
                comparison_raw[f"teacher_{teacher_name}"] = raw
            run_significance_tests(results["raw"], comparison_raw)

            final_kd_auc = results["pooled"].get("auc", float("nan"))
            valid_baseline_aucs = [r.get("auc", -1) for r in baseline_rows if r.get("auc") == r.get("auc")]
            best_baseline_auc = max(valid_baseline_aucs) if valid_baseline_aucs else -1.0
            if final_kd_auc == final_kd_auc and final_kd_auc <= best_baseline_auc:
                print(f"[main] HONEST REPORT: final KD student AUC ({final_kd_auc:.4f}) did NOT beat "
                      f"the best baseline ({best_baseline_auc:.4f}).")
            elif final_kd_auc == final_kd_auc:
                print(f"[main] final KD student AUC ({final_kd_auc:.4f}) beats best baseline "
                      f"({best_baseline_auc:.4f}).")
                
            run_embedding_tsne_visualization(student, test_loader, ood_loaders)
            run_domain_threshold_tuning(student, val_loader, results["raw"])
            write_paper_summary_table(results["pooled"], results["per_dataset"],
                                       results["ood_summary"], CSV_DIR / "paper_summary_table.csv")
            run_all_ablations_from_artifacts(student, artifact_reader, usable, val_loader)
            run_ablation_paired_significance(results["raw"])
            save_raw_predictions_npz(results["raw"], OUT_DIR / f"raw_predictions_{RUN_TAG}.npz")
            if RUN_EFFICIENCY_TABLE:
                efficiency_rows = run_efficiency_comparison(student)
                write_efficiency_reduction_summary(efficiency_rows, results.get("per_teacher_metrics", []))
            else:
                print("[efficiency] RUN_EFFICIENCY_TABLE=False; skipping (opt-in).")
            write_reproducibility_metadata(OUT_DIR)
            write_final_report(OUT_DIR)
            print("\n[main] DONE (EVALUATE). All CSVs and figures are under:", OUT_DIR)
            return


if __name__ == "__main__":
    if TORCH_OK:
        main()
    else:
        print("[main] torch unavailable; stopping after inspection/audit-only utilities.")
        rows = load_manifest(MANIFEST_PATH)
        audit_manifest(rows)