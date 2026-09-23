# ============================================================
# LAV-DF Audio Deepfake Detector — Production Training Pipeline v9
# ============================================================
# Architecture : WavLM-Base+ (WavLM-Large now OPT-IN only, see v9 diagnosis)
#                → HSTGAT Head + LFCC Branch

import subprocess, sys

def pip_install_quiet(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *pkgs])

pip_install_quiet(
    "transformers>=4.40.0", "accelerate>=0.29.0", "soundfile>=0.12.1",
    "librosa>=0.10.1", "einops>=0.7.0", "wandb>=0.17.0",
    "torchmetrics>=1.3.0", "scipy>=1.11.0", "scikit-learn>=1.3.0",
    "matplotlib>=3.7.0", "seaborn>=0.12.0",
)
print("✓ Dependencies installed")

import os, gc, json, math, random, glob, warnings, time, shutil, hashlib, resource, sys as _sys
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Tuple, Any, Union, Callable

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, sosfilt, fftconvolve

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.swa_utils import AveragedModel, update_bn as swa_update_bn

torch.multiprocessing.set_sharing_strategy('file_system')
try:
    _soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, _hard), _hard))
    print(f"✓ file_system sharing strategy set | nofile ulimit: {_soft}->{min(65536,_hard)}")
except Exception as _e:
    print(f"  (nofile ulimit raise skipped: {_e})")

import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as AF

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve, f1_score, precision_score,
    recall_score, accuracy_score, confusion_matrix,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import WavLMModel, get_cosine_schedule_with_warmup

warnings.filterwarnings("ignore")

# ── AMP / dtype detection ────────────────────────────────────
def detect_amp_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16

AMP_DTYPE = detect_amp_dtype()
_TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])

if _TORCH_VERSION >= (2, 0):
    from torch.amp import GradScaler, autocast as _autocast
    def autocast(enabled=True, device_type="cuda"):
        if device_type == "cuda" and not torch.cuda.is_available():
            enabled = False
        if enabled:
            try:
                return _autocast(device_type=device_type, enabled=True, dtype=AMP_DTYPE)
            except TypeError:
                return _autocast(device_type=device_type, enabled=True)
        return _autocast(device_type=device_type, enabled=False)
    def make_scaler(enabled=True):
        if not torch.cuda.is_available(): enabled = False
        if AMP_DTYPE == torch.bfloat16: enabled = False
        return GradScaler("cuda", enabled=enabled)
else:
    from torch.cuda.amp import GradScaler as _GradScaler, autocast as _autocast_old
    def autocast(enabled=True, device_type="cuda"):
        if device_type == "cuda" and not torch.cuda.is_available(): enabled = False
        return _autocast_old(enabled=enabled)
    def make_scaler(enabled=True):
        if not torch.cuda.is_available(): enabled = False
        return _GradScaler(enabled=enabled)

print(f"✓ Imports OK | torch {torch.__version__} | cuda: {torch.cuda.is_available()} | AMP: {AMP_DTYPE}")

# ── Config ───────────────────────────────────────────────────
@dataclass
class Config:
    data_root:      str   = "/kaggle/input"
    audio_glob:     str   = "/kaggle/input/**/audio_cache/*.wav"
    metadata_path:  str   = "/kaggle/input/lav-df/metadata.json"
    output_dir:     str   = "/kaggle/working/checkpoints"
    log_dir:        str   = "/kaggle/working/logs"
    figures_dir:    str   = "/kaggle/working/figures"
    audio_cache_dir: str  = "/kaggle/working/audio_cache"
    audio_cache_root: str = "/kaggle/input/datasets/syedazmulhasansabbir/audio-cache"
    default_resume_checkpoint: str = (
        "/kaggle/input/models/sharminaktarmunni/lavdf-ep22"
        "/tensorflow2/default/1/ckpt_epoch022_auc0.9241.pt"
    )
    lav_df_video_root: str = (
        "/kaggle/input/datasets/elin75"
        "/localized-audio-visual-deepfake-dataset-lav-df/LAV-DF"
    )
    sample_rate:      int   = 16_000
    segment_sec:      float = 4.0
    max_sec:          float = 30.0
    min_sec:          float = 0.5
    segment_samples:  int   = field(init=False)
    wavlm_name:                str   = "microsoft/wavlm-base-plus"
    auto_batch_for_large:      bool  = True
    freeze_feature_extractor:  bool  = True
    freeze_encoder_layers:     int   = 0
    hidden_size:               int   = 768
    num_graph_heads:           int   = 4
    graph_hidden:              int   = 128
    dropout:                   float = 0.1
    gradient_checkpointing:    bool  = False
    use_compile:               bool  = False
    layerwise_lr_decay:        float = 0.9
    use_hst_graph:             bool  = True
    num_spectral_bands:        int   = 20
    use_lfcc_branch:           bool  = True
    seed:              int   = 42
    epochs:            int   = 15
    batch_size:        int   = 16
    grad_accum:        int   = 2
    lr:                float = 5e-5
    lr_wavlm:          float = 5e-6
    weight_decay:      float = 1e-2
    max_grad_norm:     float = 1.0
    warmup_ratio:      float = 0.1
    label_smoothing:   float = 0.05
    focal_gamma_pos:   float = 2.0
    focal_gamma_neg:   float = 1.0
    use_asymmetric_focal: bool = True
    cudnn_benchmark:   bool  = True
    ema_decay:         float = 0.9999
    ema_decay_start:   float = 0.990
    ema_decay_end:     float = 0.9999
    use_official_split: bool  = True
    val_ratio:          float = 0.1
    test_ratio:          float = 0.1
    num_workers:      int  = 2
    pin_memory:       bool = False
    prefetch_factor:  int  = 2
    sorted_val_batching: bool = False
    aug_prob:              float = 0.5
    speed_factors:         Tuple = (0.95, 1.05)
    snr_db_range:          Tuple = (15, 30)
    time_mask_max_samples: int   = 4000
    time_mask_n:           int   = 1
    codec_sim_prob:        float = 0.15
    codec_cutoff_min_hz:   float = 3500.0
    codec_cutoff_max_hz:   float = 7500.0
    freq_mask_param:       int   = 30
    mixup_alpha:           float = 0.2
    rir_prob:              float = 0.15
    rir_rt60_range:        Tuple = (0.1, 1.5)
    rir_n_synthetic:       int   = 20
    noise_prob:            float = 0.40
    use_cutmix:            bool  = True
    cutmix_prob:            float = 0.50
    cutmix_alpha:            float = 0.4
    polarity_inv_prob:     float = 0.30
    packet_loss_prob:      float = 0.20
    pitch_shift_prob:      float = 0.0
    pitch_shift_range:     Tuple = (-2.0, 2.0)
    phase_scramble_prob:   float = 0.0
    use_rdrop:             bool  = False
    rdrop_alpha:           float = 4.0
    use_supcon:            bool  = True
    supcon_weight:         float = 0.1
    supcon_temperature:    float = 0.10
    use_frame_loss:        bool  = True
    frame_loss_weight:     float = 0.3
    aux_loss_warmup_epochs: int  = 2
    _frame_loss_available: bool  = field(default=False, init=False, repr=False)
    stage1_epochs:           int   = 3
    unfreeze_every_n_epochs: int   = 2
    use_swa:                 bool  = True
    swa_start_epoch:         int   = 15
    swa_lr:                  float = 5e-6
    use_hard_mining:         bool  = True
    hard_mining_top_pct:     float = 0.20
    hard_mining_weight:      float = 2.0
    use_manifold_mixup:      bool  = True
    use_tta:                 bool  = True
    tta_n_crops:             int   = 5
    use_temperature_scaling: bool  = True
    use_model_soup:          bool  = True
    log_grad_norms:    bool  = True
    log_ema_vs_raw:    bool  = True
    log_layer_weights: bool  = True
    amp:              bool  = True
    amp_dtype:        str   = "auto"
    early_stop_patience:   int   = 10
    save_top_k:            int   = 3
    use_wandb:             bool  = False
    time_budget_hours:     float = 9.5
    eval_every_n_steps:    int   = 500

    auto_select_backbone:      bool  = False
    large_backbone_min_vram_gb: float = 15.0

    quick_val_subset_size:     int   = 768
    mid_eval_use_ema_only:     bool  = True
    hard_session_hours:        float = 11.5
    initial_reserve_eval_hours: float = 1.5
    min_reserve_eval_hours:    float = 0.6
    max_reserve_eval_hours:    float = 2.5
    best_model_filename:       str   = "best_model.pt"
    last_model_filename:       str   = "last_model.pt"

    def __post_init__(self):
        self.segment_samples = int(self.sample_rate * self.segment_sec)

        # IMP-1: auto backbone upgrade based on detected VRAM.
        if self.auto_select_backbone and torch.cuda.is_available():
            try:
                vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            except Exception:
                vram_gb = 0.0
            if vram_gb >= self.large_backbone_min_vram_gb and "large" not in self.wavlm_name.lower():
                print(f"  [Config] Detected {vram_gb:.1f} GB VRAM (>= {self.large_backbone_min_vram_gb} GB) "
                      f"-> auto-upgrading backbone to microsoft/wavlm-large for higher accuracy")
                self.wavlm_name = "microsoft/wavlm-large"
                if self.auto_batch_for_large:
                    self.batch_size = 4
                    self.grad_accum = 8
                    self.gradient_checkpointing = True
                    self.lr_wavlm = 3e-6
            elif vram_gb and vram_gb < self.large_backbone_min_vram_gb:
                print(f"  [Config] Detected {vram_gb:.1f} GB VRAM -> keeping {self.wavlm_name}")

        if "large" in self.wavlm_name.lower():
            if self.hidden_size != 1024:
                print("  [Config] wavlm-large detected -> hidden_size=1024")
            self.hidden_size = 1024
        if self.graph_hidden % self.num_graph_heads != 0:
            old = self.graph_hidden
            self.graph_hidden = (self.graph_hidden // self.num_graph_heads) * self.num_graph_heads
            print(f"  [Config] graph_hidden rounded {old}->{self.graph_hidden}")
        if self.use_rdrop and self.batch_size > 8:
            self.batch_size = self.batch_size // 2
            self.grad_accum = self.grad_accum * 2
            print(f"  [Config] use_rdrop=True: batch_size->{self.batch_size}, grad_accum->{self.grad_accum}")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.audio_cache_dir, exist_ok=True)

        # IMP-4: canonical best/last checkpoint paths
        self.best_model_path = os.path.join(self.output_dir, self.best_model_filename)
        self.last_model_path = os.path.join(self.output_dir, self.last_model_filename)

CFG = Config()
print(asdict(CFG))

# ── Reproducibility ──────────────────────────────────────────
def set_seed(seed, cudnn_benchmark=False, seed_cuda=True):

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if seed_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = not cudnn_benchmark
    torch.backends.cudnn.benchmark = cudnn_benchmark
    os.environ["PYTHONHASHSEED"] = str(seed)

def get_rng_state():
    s = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available(): s["cuda"] = torch.cuda.get_rng_state_all()
    return s

def set_rng_state(state):
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])

def worker_init_fn(worker_id):
    base = torch.initial_seed() % (2**32)
    np.random.seed(base + worker_id); random.seed(base + worker_id)

set_seed(CFG.seed, CFG.cudnn_benchmark, seed_cuda=False)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_DEVICE_TYPE = DEVICE.type
print("Device:", DEVICE)


# ── Helpers ──────────────────────────────────────────────────
def _to_python(obj):
    if isinstance(obj, dict): return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_to_python(v) for v in obj]
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return obj

def try_compile(model, cfg):
    if getattr(cfg, "use_compile", False) and hasattr(torch, "compile"):
        try: return torch.compile(model)
        except Exception as e:
            print(f"  [Model] torch.compile failed ({e}); using eager.")
            return model
    return model

def _safe_torch_load(path, map_location=None):
    return torch.load(path, map_location=map_location, weights_only=False)

# IMP-6: robust checkpoint loading — only load tensors whose name AND
# shape match the current model. Needed so auto backbone-upgrade (IMP-1)
# can't crash on a stale/smaller pretrained checkpoint.
def load_compatible_state_dict(model, state_dict, label="checkpoint"):
    model_sd = model.state_dict()
    matched = {}
    n_shape_mismatch = 0
    for k, v in state_dict.items():
        if k in model_sd:
            if model_sd[k].shape == v.shape:
                matched[k] = v
            else:
                n_shape_mismatch += 1
        # keys not present in model_sd are silently dropped
    missing = set(model_sd.keys()) - set(matched.keys())
    model.load_state_dict(matched, strict=False)
    print(f"  [{label}] loaded {len(matched)}/{len(model_sd)} tensors "
          f"({n_shape_mismatch} shape-mismatched, {len(missing)-n_shape_mismatch if len(missing)>=n_shape_mismatch else len(missing)} "
          f"missing-from-checkpoint) — mismatched/missing tensors keep their (pretrained/random) init")
    return len(matched)

# IMP-4: canonical best/last checkpoint writer (atomic)
def save_named_checkpoint(state, path):
    tmp = str(path) + ".tmp"
    torch.save(state, tmp)
    try:
        os.replace(tmp, path)
    except OSError:
        shutil.move(tmp, path)

# ── Metadata loading ─────────────────────────────────────────
def _parse_json_file(path):
    text = path.read_text(encoding="utf-8").strip()
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if line: rows.append(json.loads(line))
        return rows
    if isinstance(raw, list): return raw
    if isinstance(raw, dict):
        rows = []
        for split_name, items in raw.items():
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        item = dict(item); item.setdefault("split", split_name); rows.append(item)
                    else: rows.append({"value": item, "split": split_name})
            elif isinstance(items, dict):
                lengths = [len(v) for v in items.values() if isinstance(v, list)]
                if not lengths: continue
                n = lengths[0]
                for i in range(n):
                    row = {k: v[i] for k, v in items.items() if isinstance(v, list) and i < len(v)}
                    row.setdefault("split", split_name); rows.append(row)
        return rows
    raise ValueError(f"Unrecognised JSON top-level type: {type(raw)}")

def _infer_label_column(df):
    cols_lower = {c.lower(): c for c in df.columns}
    for alias in ("label", "fake", "is_fake", "target", "class", "manipulated", "is_deepfake"):
        if alias in cols_lower:
            col = df[cols_lower[alias]]
            if col.dtype == object:
                col = col.astype(str).str.strip().str.lower()
                mapping = {"fake": 1, "real": 0, "1": 1, "0": 0, "true": 1, "false": 0,
                           "yes": 1, "no": 0, "manipulated": 1, "genuine": 0}
                mapped = col.map(mapping)
                if mapped.notna().all(): return mapped.astype(int)
            else:
                vals = col.dropna().unique()
                if set(vals).issubset({0, 1, True, False, 0.0, 1.0}):
                    return col.fillna(0).astype(int)
    if "n_fakes" in cols_lower:
        col = df[cols_lower["n_fakes"]]
        try: return (pd.to_numeric(col, errors="coerce").fillna(0) > 0).astype(int)
        except: pass
    if "fake_periods" in cols_lower:
        col = df[cols_lower["fake_periods"]]
        def _is_fp(v):
            if v is None: return 0
            if isinstance(v, list): return int(len(v) > 0)
            s = str(v).strip(); return int(s not in ("", "[]", "null", "None"))
        return col.map(_is_fp).astype(int)
    has_ma = "modify_audio" in cols_lower; has_mv = "modify_video" in cols_lower
    if has_ma or has_mv:
        result = pd.Series(0, index=df.index)
        for alias in ("modify_audio", "modify_video"):
            if alias in cols_lower:
                col = df[cols_lower[alias]]
                if col.dtype == object:
                    col = col.astype(str).str.strip().str.lower()
                    col = col.map({"true":1,"false":0,"1":1,"0":0,"yes":1,"no":0}).fillna(0)
                result = np.maximum(result, col.fillna(0).astype(int))
        return result.astype(int)
    if "original" in cols_lower:
        col = df[cols_lower["original"]]
        if col.dtype == object:
            col = col.astype(str).str.strip().str.lower()
            col = col.map({"true":0,"false":1,"1":0,"0":1,"yes":0,"no":1}).fillna(0)
        else: col = (col.fillna(1) == 0).astype(int)
        return col.astype(int)
    sample = df.head(2).to_dict(orient="records")
    raise KeyError(f"Cannot infer 'label' from columns: {list(df.columns)}\nSample: {json.dumps(sample, indent=2, default=str)}")

def load_metadata(cfg):
    meta_path = Path(cfg.metadata_path)
    if not meta_path.exists():
        candidates = sorted(Path(cfg.data_root).rglob("metadata.json"))
        if candidates: meta_path = candidates[0]; print(f"  Found metadata at: {meta_path}")
        else: raise FileNotFoundError(f"metadata.json not found under: {cfg.data_root}")
    rows = _parse_json_file(meta_path)
    df = pd.DataFrame(rows)
    if df.empty: raise ValueError(f"Metadata empty from {meta_path}")
    df.columns = [c.strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ("file","filename","id","video_id","audio_id","file_id"): rename[c] = "file_id"
        elif lc in ("split","subset","partition"): rename[c] = "split"
    df = df.rename(columns=rename)
    if "label" not in df.columns:
        df["label"] = _infer_label_column(df); label_source = "inferred"
    else: label_source = "direct"
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    if "file_id" not in df.columns:
        raise KeyError(f"No file-id column. Available: {list(df.columns)}")
    df["file_id"] = df["file_id"].astype(str).str.strip()
    before = len(df)
    conflict = df.groupby("file_id")["label"].nunique()
    conflicted = conflict[conflict > 1].index.tolist()
    if conflicted:
        print(f"  WARNING: {len(conflicted)} file_ids with conflicting labels — dropping.")
        df = df[~df["file_id"].isin(conflicted)]
    df = df.drop_duplicates(subset="file_id").reset_index(drop=True)
    if before != len(df): print(f"  Dedup: {before}->{len(df)}")
    print(f"Metadata rows: {len(df)}  (label source: {label_source})")
    print(df["label"].value_counts().to_string())
    if "split" in df.columns: print(df.groupby("split")["label"].value_counts().to_string())
    return df

# ── Audio discovery ──────────────────────────────────────────
def discover_audio_files(cfg):
    all_paths = []
    for ext in ("*.wav","*.flac","*.mp3","*.ogg"):
        all_paths.extend(glob.glob(str(Path(cfg.data_root)/"**"/ext), recursive=True))
        all_paths.extend(glob.glob(str(Path(cfg.audio_cache_dir)/"**"/ext), recursive=True))
    if cfg.audio_cache_root and Path(cfg.audio_cache_root).exists():
        for ext in ("*.wav","*.flac"):
            all_paths.extend(glob.glob(str(Path(cfg.audio_cache_root)/"**"/ext), recursive=True))
        print(f"  [discover] Scanned audio_cache_root: {cfg.audio_cache_root} -> {len(all_paths)} path(s) before dedup")
    all_paths.extend(glob.glob(cfg.audio_glob, recursive=True))
    all_paths = sorted(set(all_paths))
    print(f"  Audio file discovery: found {len(all_paths)} files total")
    if not all_paths: print("  WARNING: No audio files found."); return {}
    wav_map = {}
    for p in all_paths:
        pp = Path(p); stem = pp.stem; name = pp.name; parts = pp.parts
        candidates = [stem, name]
        if len(parts) >= 2:
            candidates.append("/".join(parts[-2:])); candidates.append("/".join((*parts[-2:-1], stem)))
        if len(parts) >= 3:
            candidates.append("/".join(parts[-3:])); candidates.append("/".join((*parts[-3:-1], stem)))
        for key in candidates:
            if key not in wav_map: wav_map[key] = p
    print(f"  Lookup keys registered: {len(wav_map)}")
    return wav_map

def _resolve_audio(file_id, lav_df_video_root, wav_map, data_root):
    stem = Path(file_id).stem; name = Path(file_id).name
    if lav_df_video_root:
        vp = str((Path(lav_df_video_root)/file_id).resolve())
        uid = hashlib.md5(vp.encode()).hexdigest()[:16]
        if uid in wav_map: return wav_map[uid]
        vp_raw = str(Path(lav_df_video_root)/file_id)
        if vp_raw != vp:
            uid2 = hashlib.md5(vp_raw.encode()).hexdigest()[:16]
            if uid2 in wav_map: return wav_map[uid2]
    uid3 = hashlib.md5(file_id.encode()).hexdigest()[:16]
    if uid3 in wav_map: return wav_map[uid3]
    for key in (file_id, stem, name, stem+".wav", stem+".flac"):
        if key in wav_map: return wav_map[key]
    return None

def _diagnose_cache_mismatch(df_split, cfg, wav_map, split_name):
    sample_ids = df_split["file_id"].head(5).tolist()
    sample_keys = list(wav_map.keys())[:8]
    print(f"\n  [{split_name}] ══ CACHE MISMATCH ══")
    print(f"  Sample file_ids : {sample_ids}")
    print(f"  Sample wav_keys : {sample_keys}")
    print(f"  HINTS: (a) Fix cfg.lav_df_video_root  (b) Rebuild cache  (c) Set cfg.audio_glob")

# ── Audio preprocessing ──────────────────────────────────────
def vad_trim(wav, sr, top_db=30.0, min_sec=0.5):
    try:
        frame_length = int(sr*0.025); hop_length = int(sr*0.010)
        rms = np.array([np.sqrt(np.mean(wav[i:i+frame_length]**2))
                        for i in range(0, max(1, len(wav)-frame_length), hop_length)])
        if len(rms) == 0: return wav
        rms_db = 20*np.log10(rms+1e-9); threshold = rms_db.max()-top_db
        voiced = rms_db >= threshold
        if voiced.sum() == 0: return wav
        first = int(np.argmax(voiced)*hop_length)
        last  = int((len(voiced)-1-np.argmax(voiced[::-1]))*hop_length+frame_length)
        trimmed = wav[first:min(last,len(wav))]
        return trimmed if len(trimmed)/sr >= min_sec else wav
    except: return wav

def remove_dc_offset(wav): return wav - wav.mean()

def rms_normalise(wav, target_rms=0.1):
    rms = np.sqrt(np.mean(wav**2))
    if rms < 1e-9: return wav
    return (wav*(target_rms/rms)).astype(np.float32)

def _cache_path(src, cfg):
    uid = hashlib.md5(src.encode()).hexdigest()[:16]
    return Path(cfg.audio_cache_dir)/f"{uid}.wav"

def load_wav(path, cfg):
    cache = _cache_path(path, cfg)
    if cache.exists():
        try:
            waveform, sr = torchaudio.load(str(cache))
            wav = waveform.mean(dim=0).numpy()
            if len(wav) > 0 and len(wav)/sr >= cfg.min_sec:
                return rms_normalise(remove_dc_offset(wav))
            cache.unlink(missing_ok=True)
        except: cache.unlink(missing_ok=True)
    try: waveform, sr = torchaudio.load(path)
    except:
        try:
            arr, sr = __import__('soundfile').read(path, dtype="float32", always_2d=True)
            waveform = torch.from_numpy(arr.T)
        except Exception as e:
            print(f"  [load_wav] ERROR {path!r}: {e}", file=_sys.stderr); return None
    if waveform.shape[1] == 0: return None
    wav_t = waveform.mean(dim=0, keepdim=True)
    if sr != cfg.sample_rate:
        wav_t = T.Resample(orig_freq=sr, new_freq=cfg.sample_rate)(wav_t)
    wav = wav_t.squeeze(0).numpy().astype(np.float32)
    if len(wav) == 0 or len(wav)/cfg.sample_rate < cfg.min_sec: return None
    wav = vad_trim(wav, cfg.sample_rate)
    wav = remove_dc_offset(wav); wav = rms_normalise(wav)
    try: torchaudio.save(str(cache), torch.from_numpy(wav).unsqueeze(0), cfg.sample_rate)
    except: pass
    return wav

def pad_or_crop(wav, target, train, return_start=False):
    if len(wav) == 0:
        wav = np.zeros(target, dtype=np.float32)
        return (wav, 0) if return_start else wav
    L = len(wav)
    if L >= target:
        start = random.randint(0, L-target) if train else (L-target)//2
        out = wav[start:start+target]
    else:
        start = 0; out = np.pad(wav, (0, target-L), mode="constant")
    return (out, start) if return_start else out

# ── Augmentation ─────────────────────────────────────────────
def generate_synthetic_rir(rt60, sr, room_dim=None):
    length = max(1, int(rt60*sr)); t = np.arange(length)
    rir = np.random.randn(length).astype(np.float32)*np.exp(-6.908*t/length)
    peak = np.abs(rir).max()
    if peak > 1e-9: rir /= peak
    return rir

def _make_white_noise(length): return np.random.randn(length).astype(np.float32)

def _make_pink_noise(length):
    white = np.random.randn(length).astype(np.float32); pink = np.cumsum(white)
    pink -= pink.mean(); std = pink.std()
    if std > 1e-9: pink /= std
    return pink.astype(np.float32)

def _make_babble_noise(length, n_speakers=5):
    t = np.linspace(0, length/16000., length, dtype=np.float32)
    babble = np.zeros(length, dtype=np.float32)
    for _ in range(n_speakers):
        f0=random.uniform(100,400); fr=random.uniform(-50,50); ph=random.uniform(0,2*math.pi)
        babble += np.sin(2*math.pi*(f0*t+0.5*fr*t**2)+ph)
    std = babble.std()
    if std > 1e-9: babble /= std
    return babble

def packet_loss(wav, sr, n_packets=3, packet_ms_range=(10,50)):
    wav = wav.copy(); L = len(wav)
    for _ in range(n_packets):
        plen = min(random.randint(int(packet_ms_range[0]*sr/1000), int(packet_ms_range[1]*sr/1000)), L)
        wav[random.randint(0, L-plen):random.randint(0, L-plen)+plen] = 0.0
    return wav

def pitch_shift_aug(wav, sr, n_steps_range=(-2.,2.)):
    n_steps = random.uniform(*n_steps_range)
    if abs(n_steps) < 0.1: return wav
    try:
        shifted = AF.pitch_shift(torch.from_numpy(wav).unsqueeze(0), sr, n_steps)
        return shifted.squeeze(0).numpy().astype(np.float32)
    except: return wav

def phase_scramble(wav, fmin_hz=300., fmax_hz=4000., sr=16000):
    try:
        n_fft=512; wav_t=torch.from_numpy(wav)
        stft=torch.stft(wav_t,n_fft=n_fft,return_complex=True)
        freq_bins=stft.shape[0]; bin_min=int(fmin_hz*n_fft/sr); bin_max=min(int(fmax_hz*n_fft/sr),freq_bins-1)
        phase=stft.angle().clone()
        phase[bin_min:bin_max,:]=2*math.pi*torch.rand(bin_max-bin_min,phase.shape[1])-math.pi
        stft_out=stft.abs()*torch.exp(1j*phase)
        return torch.istft(stft_out,n_fft=n_fft,length=len(wav)).numpy().astype(np.float32)
    except: return wav

def codec_simulation(wav, sr, cutoff_hz):
    nyq=sr/2.; sos=butter(6,min(cutoff_hz/nyq,0.99),btype="low",output="sos")
    return sosfilt(sos,wav).astype(np.float32)

class AudioAugmenter:
    def __init__(self, cfg):
        self.cfg=cfg
        self._rirs=[generate_synthetic_rir(random.uniform(*cfg.rir_rt60_range),cfg.sample_rate)
                    for _ in range(cfg.rir_n_synthetic)]
        self._noise_types=["white","pink","babble"]; self._noise_probs=[0.4,0.3,0.3]
    def _apply_rir(self,wav):
        try:
            rir=random.choice(self._rirs); out=fftconvolve(wav,rir,mode="full")[:len(wav)]
            peak=np.abs(out).max()
            if peak>1e-9: out/=peak
            return out.astype(np.float32)
        except: return wav
    def _apply_noise(self,wav):
        try:
            snr_db=random.uniform(*self.cfg.snr_db_range)
            tp=random.choices(self._noise_types,weights=self._noise_probs)[0]; L=len(wav)
            noise=_make_white_noise(L) if tp=="white" else (_make_pink_noise(L) if tp=="pink" else _make_babble_noise(L))
            rms_w=np.sqrt(np.mean(wav**2))+1e-9; rms_n=np.sqrt(np.mean(noise**2))+1e-9
            return (wav+noise*rms_w/(rms_n*(10**(snr_db/20)))).astype(np.float32)
        except: return wav
    def speed_perturb(self,wav,sr):
        factor=random.uniform(*self.cfg.speed_factors)
        if abs(factor-1.)<0.005: return wav
        try:
            orig_sr=int(sr*factor)
            if orig_sr==sr: return wav
            resampled=T.Resample(orig_freq=orig_sr,new_freq=sr)(torch.from_numpy(wav).unsqueeze(0)).squeeze(0).numpy()
            return resampled.astype(np.float32) if len(resampled)>=1 else wav
        except: return wav
    def codec_sim(self,wav,sr):
        return codec_simulation(wav,sr,random.uniform(self.cfg.codec_cutoff_min_hz,self.cfg.codec_cutoff_max_hz))
    def time_mask(self,wav):
        wav=wav.copy(); L=len(wav)
        if L==0: return wav
        for _ in range(self.cfg.time_mask_n):
            ml=random.randint(1,min(self.cfg.time_mask_max_samples,L))
            wav[random.randint(0,L-ml):random.randint(0,L-ml)+ml]=0.
        return wav
    def freq_mask(self,wav,sr):
        try:
            n_fft=512; wav_t=torch.from_numpy(wav)
            stft=torch.stft(wav_t,n_fft=n_fft,return_complex=True)
            fb=stft.shape[0]; mw=random.randint(1,min(self.cfg.freq_mask_param,fb))
            stft[random.randint(0,fb-mw):random.randint(0,fb-mw)+mw,:]=0.
            return torch.istft(stft,n_fft=n_fft,length=len(wav)).numpy().astype(np.float32)
        except: return wav
    def random_gain(self,wav):
        return (wav*(10**(random.uniform(-3.,3.)/20))).astype(np.float32)
    def __call__(self,wav,sr):
        if random.random()<self.cfg.polarity_inv_prob: wav=-wav
        if random.random()>self.cfg.aug_prob: return wav
        if random.random()<self.cfg.rir_prob: wav=self._apply_rir(wav)
        if random.random()<self.cfg.noise_prob: wav=self._apply_noise(wav)
        if random.random()<0.40: wav=self.speed_perturb(wav,sr)
        if random.random()<self.cfg.codec_sim_prob: wav=self.codec_sim(wav,sr)
        if random.random()<0.40: wav=self.time_mask(wav)
        if random.random()<0.30: wav=self.freq_mask(wav,sr)
        if random.random()<0.30: wav=self.random_gain(wav)
        if random.random()<self.cfg.packet_loss_prob: wav=packet_loss(wav,sr)
        # PERF-3: pitch_shift_prob and phase_scramble_prob default to 0.0
        if random.random()<self.cfg.pitch_shift_prob: wav=pitch_shift_aug(wav,sr,self.cfg.pitch_shift_range)
        if random.random()<self.cfg.phase_scramble_prob: wav=phase_scramble(wav,sr=sr)
        return wav

# ── Dataset + DataLoaders ────────────────────────────────────
def _parse_fake_periods(row):
    try:
        fp=row.get("fake_periods",None)
        if fp is None: return None
        if isinstance(fp,float) and math.isnan(fp): return None
        if isinstance(fp,list): return fp if fp else []
        if isinstance(fp,str):
            fp=fp.strip()
            if fp in ("","[]","null","None"): return []
            return json.loads(fp)
        return None
    except: return None

def _build_frame_labels(fake_periods, segment_samples, sample_rate, crop_start_sample=0):
    wavlm_stride=320; num_frames=segment_samples//wavlm_stride
    labels=np.zeros(num_frames,dtype=np.float32)
    if not fake_periods: return labels
    crop_sec=crop_start_sample/sample_rate
    for period in fake_periods:
        try:
            sf_f=int((float(period[0])-crop_sec)*sample_rate/wavlm_stride)
            ef_f=int((float(period[1])-crop_sec)*sample_rate/wavlm_stride)
            sf_f=max(0,min(sf_f,num_frames)); ef_f=max(0,min(ef_f,num_frames))
            if ef_f>sf_f: labels[sf_f:ef_f]=1.
        except: pass
    return labels

class LAVDFDataset(Dataset):
    def __init__(self, df, wav_map, cfg, split="train"):
        self.cfg=cfg; self.split=split; self.is_train=(split=="train")
        self.augmenter=AudioAugmenter(cfg) if self.is_train else None
        df=df.copy()
        df["path"]=df["file_id"].apply(
            lambda fid: _resolve_audio(str(fid),cfg.lav_df_video_root,wav_map,cfg.data_root))
        total=len(df); missing=df["path"].isna().sum()
        print(f"  [{split}] {total-missing}/{total} matched ({100.*(total-missing)/max(total,1):.1f}%)")
        if missing==total and total>0: _diagnose_cache_mismatch(df,cfg,wav_map,split)
        df=df.dropna(subset=["path"]).reset_index(drop=True); self.df=df
        n_real=int((df.label==0).sum()); n_fake=int((df.label==1).sum())
        print(f"  [{split}] usable: {len(df)}  (real={n_real}, fake={n_fake})")
        self._frame_loss_ok=False
        if cfg.use_frame_loss and "fake_periods" in df.columns:
            sample_fp=_parse_fake_periods(df.iloc[0].to_dict() if len(df) else {})
            self._frame_loss_ok=(sample_fp is not None)
        cfg._frame_loss_available=self._frame_loss_ok
    def __len__(self): return len(self.df)
    def __getitem__(self,idx):
        row=self.df.iloc[idx]; wav=load_wav(row["path"],self.cfg)
        if wav is None: return None
        if self.is_train and self.augmenter: wav=self.augmenter(wav,self.cfg.sample_rate)
        wav,crop_start=pad_or_crop(wav,self.cfg.segment_samples,self.is_train,return_start=True)
        frame_labels=None
        if self._frame_loss_ok and self.cfg.use_frame_loss:
            try:
                fp=_parse_fake_periods(row.to_dict())
                if fp is not None:
                    frame_labels=_build_frame_labels(fp,self.cfg.segment_samples,self.cfg.sample_rate,crop_start)
            except: frame_labels=None
        fl_t=torch.tensor(frame_labels,dtype=torch.float32) if frame_labels is not None else None
        return (torch.tensor(wav,dtype=torch.float32),
                torch.tensor(int(row["label"]),dtype=torch.long),
                str(row["file_id"]), fl_t)

def collate_fn(batch):
    valid=[b for b in batch if b is not None]
    skip=len(batch)-len(valid)
    if skip>0 and len(batch)>0 and skip/len(batch)>0.2:
        print(f"  [collate_fn] WARNING: {skip}/{len(batch)} skipped",file=_sys.stderr)
    if not valid: return {"n":0}
    wavs,labels,fids,frame_labels=zip(*valid)
    has_fl=all(fl is not None for fl in frame_labels)
    return {"wav":torch.stack(wavs),"labels":torch.stack(labels),"file_id":list(fids),
            "frame_labels":torch.stack(list(frame_labels)) if has_fl else None,"n":len(valid)}

def _safe_train_test_split(df, test_size, seed):
    if len(df)<2: return df, df.iloc[:0]
    min_class=df["label"].value_counts().min()
    try:
        if min_class>=2: a,b=train_test_split(df,test_size=test_size,stratify=df["label"],random_state=seed)
        else: raise ValueError("Too few")
    except ValueError as exc:
        print(f"  WARNING: stratified split failed ({exc}). Falling back to random.")
        a,b=train_test_split(df,test_size=test_size,random_state=seed)
    return a.reset_index(drop=True), b.reset_index(drop=True)

def make_dataloaders(df, wav_map, cfg):
    t0=time.time()
    if cfg.use_official_split and "split" in df.columns:
        unique_splits=df["split"].unique().tolist()
        alias_map={"train":"train","dev":"val","validation":"val","valid":"val","val":"val","test":"test"}
        df=df.copy(); df["split"]=df["split"].map(lambda x: alias_map.get(str(x).lower(),x))
        raw_splits={k:df[df["split"]==k].reset_index(drop=True) for k in ("train","val","test") if len(df[df["split"]==k])>0}
        matched_splits={}
        for key,sub in raw_splits.items():
            tmp=sub["file_id"].apply(lambda fid: _resolve_audio(str(fid),cfg.lav_df_video_root,wav_map,cfg.data_root))
            n_matched=tmp.notna().sum(); print(f"  [split={key}] rows={len(sub)}, matched={n_matched}")
            if n_matched>0: matched_splits[key]=sub
            else: print(f"  [split={key}] 0 audio matches — will carve from train.")
        splits=dict(matched_splits)
        if "val" not in splits and "train" in splits:
            tr,va=_safe_train_test_split(splits["train"],cfg.val_ratio,cfg.seed)
            splits["train"]=tr; splits["val"]=va
        if "test" not in splits and "train" in splits:
            total_carve=min(cfg.test_ratio,cfg.test_ratio/(1.-cfg.val_ratio+1e-9))
            tr,te=_safe_train_test_split(splits["train"],total_carve,cfg.seed)
            splits["train"]=tr; splits["test"]=te
        if "train" not in splits: raise RuntimeError(f"No usable train split. Available: {unique_splits}")
    else:
        tr_df,tmp=_safe_train_test_split(df,cfg.val_ratio+cfg.test_ratio,cfg.seed)
        if len(tmp)>0:
            te_ratio=cfg.test_ratio/max(cfg.val_ratio+cfg.test_ratio,1e-9)
            va_df,te_df=_safe_train_test_split(tmp,te_ratio,cfg.seed)
        else: va_df=df.iloc[:0].copy(); te_df=df.iloc[:0].copy()
        splits={"train":tr_df,"val":va_df,"test":te_df}
    print("\nCreating datasets...")
    datasets={}
    for sname in ("train","val","test"):
        print(f"Building {sname.upper()} dataset..."); datasets[sname]=LAVDFDataset(splits[sname],wav_map,cfg,split=sname)
    print(f"Dataset sizes: train={len(datasets['train'])}, val={len(datasets['val'])}, test={len(datasets['test'])}")
    train_labels=datasets["train"].df["label"].values; n_train=len(train_labels)
    if n_train==0: raise RuntimeError("FATAL: Training dataset is empty.")
    class_counts=np.bincount(train_labels,minlength=2)
    class_weights=np.where(class_counts>0,1./class_counts.astype(float),0.)
    sample_weights=class_weights[train_labels]
    if sample_weights.sum()==0: sample_weights=np.ones(n_train,dtype=float)
    sampler=WeightedRandomSampler(torch.tensor(sample_weights,dtype=torch.float32),n_train,replacement=True)
    _wif=worker_init_fn if cfg.num_workers>0 else None
    _pin=cfg.pin_memory and torch.cuda.is_available()
    loaders={}
    for sname,ds in datasets.items():
        is_train=(sname=="train"); persistent=(cfg.num_workers>=1)
        loaders[sname]=DataLoader(ds,batch_size=cfg.batch_size,
            sampler=sampler if is_train else None,shuffle=False,
            num_workers=cfg.num_workers,pin_memory=_pin,collate_fn=collate_fn,
            worker_init_fn=_wif,
            prefetch_factor=cfg.prefetch_factor if cfg.num_workers>0 else None,
            persistent_workers=persistent,drop_last=is_train)
    print(f"\nDataLoader creation took {(time.time()-t0):.1f}s")
    return loaders, datasets

# ── IMP-10: fast mid-epoch validation subset ──────────────────
def build_quick_eval_loader(dataset, cfg, n=768, seed=123):

    n = min(n, len(dataset))
    rng = random.Random(seed)
    idx = rng.sample(range(len(dataset)), n) if n < len(dataset) else list(range(len(dataset)))
    subset = torch.utils.data.Subset(dataset, idx)
    return DataLoader(
        subset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn,
        num_workers=min(2, cfg.num_workers),
        pin_memory=cfg.pin_memory and torch.cuda.is_available(),
        drop_last=False,
    )

# ── IMP-12: pre-CUDA worker priming ────────────────────────────
def _prime_loader_workers(loader, label=""):

    if getattr(loader, "num_workers", 0) <= 0:
        return
    try:
        it = iter(loader)
        next(it)
        del it
        print(f"  [DataLoader] Primed persistent workers for {label} loader (pre-CUDA-init).")
    except StopIteration:
        pass
    except Exception as e:
        print(f"  [DataLoader] WARNING: priming {label} loader failed ({e}); continuing anyway.")

# ── IMP-13: safe train-loader rebuild for post-crash fallback ──
def _rebuild_train_loader(dataset, cfg, sample_weights, num_workers):
    """Rebuild the train DataLoader (fresh WeightedRandomSampler) with a
    given num_workers — used to fall back to num_workers=0 after a
    DataLoader worker crash without losing the rest of the 9+ hour run."""
    n = len(dataset)
    sampler = WeightedRandomSampler(torch.tensor(sample_weights, dtype=torch.float32), n, replacement=True)
    _wif = worker_init_fn if num_workers > 0 else None
    return DataLoader(
        dataset, batch_size=cfg.batch_size, sampler=sampler, shuffle=False,
        num_workers=num_workers, pin_memory=cfg.pin_memory and torch.cuda.is_available(),
        collate_fn=collate_fn, worker_init_fn=_wif,
        prefetch_factor=cfg.prefetch_factor if num_workers > 0 else None,
        persistent_workers=(num_workers > 0), drop_last=True,
    )

# ── LFCC Branch [A5] — FIX-8: fp32 for LFCC+BN ──────────────
class LFCCBranch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        n_lfcc=20
        self.lfcc_transform=T.LFCC(sample_rate=cfg.sample_rate,n_lfcc=n_lfcc,
                                    speckwargs={"n_fft":512,"win_length":400,"hop_length":160})
        self.cnn=nn.Sequential(
            nn.Conv1d(n_lfcc*3,128,kernel_size=3,padding=1), nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128,128,kernel_size=3,padding=1), nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128,128,kernel_size=3,padding=1), nn.BatchNorm1d(128), nn.GELU())
        self.pool=nn.AdaptiveAvgPool1d(1); self.project=nn.Linear(128,cfg.graph_hidden)
    def forward(self, wav):

        try:
            wav32=wav.float()
            lfcc=self.lfcc_transform(wav32); delta=AF.compute_deltas(lfcc); delta2=AF.compute_deltas(delta)
            feats=torch.cat([lfcc,delta,delta2],dim=1)
            out=self.cnn(feats); out=self.pool(out).squeeze(-1)
            return self.project(out).to(wav.dtype)   # cast back to input dtype
        except Exception as e:
            print(f"  [LFCCBranch] Warning: {e}; returning zeros",file=_sys.stderr)
            return torch.zeros(wav.shape[0],self.project.out_features,device=wav.device,dtype=wav.dtype)

# ── Graph layers [A2-A4] ─────────────────────────────────────
class AttentiveStatPool(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.attn=nn.Sequential(nn.Linear(in_dim,hidden_dim),nn.Tanh(),nn.Linear(hidden_dim,1))
    def forward(self, x):
        w=torch.softmax(self.attn(x).float(),dim=1).to(x.dtype)
        mean=(w*x).sum(dim=1)
        var=(w*(x-mean.unsqueeze(1)).pow(2)).sum(dim=1).clamp(min=1e-6)
        return torch.cat([mean,var.sqrt().clamp(max=1e3)],dim=-1)

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, dropout):
        super().__init__()
        self.num_heads=num_heads; self.out_dim=out_dim
        self.W=nn.Linear(in_dim,out_dim*num_heads,bias=False)
        self.a=nn.Parameter(torch.empty(num_heads,2*out_dim))
        nn.init.xavier_uniform_(self.a.view(1,-1).expand(1,-1).view(num_heads,2*out_dim))
        self.leaky=nn.LeakyReLU(0.2); self.dropout=nn.Dropout(dropout)
        self.norm=nn.LayerNorm(out_dim*num_heads)
        out_total=out_dim*num_heads
        self.residual_proj=None if in_dim==out_total else nn.Linear(in_dim,out_total)
    def forward(self, H):
        B,N,_=H.shape
        Wh=self.W(H).view(B,N,self.num_heads,self.out_dim)
        src=Wh.unsqueeze(2).expand(-1,-1,N,-1,-1); tgt=Wh.unsqueeze(1).expand(-1,N,-1,-1,-1)
        e=self.leaky(torch.einsum("bmnkd,kd->bmnk",torch.cat([src,tgt],dim=-1),self.a)).clamp(-20.,20.)
        alpha=self.dropout(torch.softmax(e.float(),dim=2).to(H.dtype))
        out=F.elu(torch.einsum("bmnk,bnkd->bmkd",alpha,Wh).reshape(B,N,-1))
        out=out+(self.residual_proj(H) if self.residual_proj else H)
        return self.norm(out)

class HSTGATHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        D=cfg.hidden_size; G=cfg.graph_hidden; K=cfg.num_graph_heads; head_dim=G//K
        self.num_spectral_bands=cfg.num_spectral_bands; self.graph_hidden=G
        self.proj_temporal=nn.Linear(D,G); self.proj_spectral=nn.Linear(1,G); self.proj_global=nn.Linear(D,G)
        self.gat1_t=GraphAttentionLayer(G,head_dim,K,cfg.dropout)
        self.gat1_s=GraphAttentionLayer(G,head_dim,K,cfg.dropout)
        self.gat1_cross=GraphAttentionLayer(G,head_dim,K,cfg.dropout)
        self.gat2_t=GraphAttentionLayer(G,head_dim,K,cfg.dropout)
        self.gat2_s=GraphAttentionLayer(G,head_dim,K,cfg.dropout)
        self.gat2_cross=GraphAttentionLayer(G,head_dim,K,cfg.dropout)
        self.stat_pool=AttentiveStatPool(G,G//2)
    def _temporal_nodes(self,hidden):
        stride=max(1,hidden.shape[1]//64); return self.proj_temporal(hidden[:,::stride,:])
    def _spectral_nodes(self,wav):
        B=wav.shape[0]; n_fft=512; S=self.num_spectral_bands
        try:
            window=torch.hann_window(n_fft,device=wav.device)
            stft=torch.stft(wav.float(),n_fft=n_fft,hop_length=160,win_length=n_fft,window=window,return_complex=True)
            power=torch.log1p(stft.abs().pow(2)).mean(dim=-1); F_bins=power.shape[1]
            bands=[power[:,i*F_bins//S:max(i*F_bins//S+1,min((i+1)*F_bins//S,F_bins))].mean(dim=1,keepdim=True) for i in range(S)]
            band_power=torch.cat(bands,dim=1).unsqueeze(-1)
            bp_max=band_power.abs().amax(dim=1,keepdim=True).clamp(min=1e-8)
            return self.proj_spectral((band_power/bp_max).to(wav.dtype))
        except: return torch.zeros(B,S,self.graph_hidden,device=wav.device,dtype=wav.dtype)
    def _global_node(self,hidden): return self.proj_global(hidden.mean(dim=1,keepdim=True))
    def forward(self,hidden,wav):
        t_nodes=self._temporal_nodes(hidden); s_nodes=self._spectral_nodes(wav); g_node=self._global_node(hidden)
        T_prime=t_nodes.shape[1]
        t_nodes=self.gat1_t(t_nodes); s_nodes=self.gat1_s(s_nodes)
        all_n=self.gat1_cross(torch.cat([g_node,t_nodes,s_nodes],dim=1))
        g_node=all_n[:,:1,:]; t_nodes=all_n[:,1:1+T_prime,:]; s_nodes=all_n[:,1+T_prime:,:]
        t_nodes=self.gat2_t(t_nodes); s_nodes=self.gat2_s(s_nodes)
        all_n=self.gat2_cross(torch.cat([g_node,t_nodes,s_nodes],dim=1))
        return self.stat_pool(all_n)

class HGATHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        D=cfg.hidden_size; G=cfg.graph_hidden; K=cfg.num_graph_heads; head_dim=G//K
        self.proj_s=nn.Linear(D,G); self.proj_t=nn.Linear(D,G)
        self.gat1=GraphAttentionLayer(G,head_dim,K,cfg.dropout)
        self.gat2=GraphAttentionLayer(G,head_dim,K,cfg.dropout)
        self.stat_pool=AttentiveStatPool(G,G//2)
    def forward(self,hidden,wav):
        B,T,D=hidden.shape; s=self.proj_s(hidden.mean(dim=1,keepdim=True))
        t=self.proj_t(hidden[:,::max(1,T//64),:])
        nodes=self.gat2(self.gat1(torch.cat([s,t],dim=1)))
        return self.stat_pool(nodes)

# ── Full model ───────────────────────────────────────────────
def manifold_mixup(hidden, labels_a, labels_b, lam, mix_prob=0.5):
    if random.random()>mix_prob or lam>=1.: return hidden,labels_a,labels_b,lam
    B=hidden.size(0); idx=torch.randperm(B,device=hidden.device)
    new_lam=float(np.random.beta(0.2,0.2))
    return new_lam*hidden+(1-new_lam)*hidden[idx], labels_a, labels_a[idx], new_lam

class FrameLevelHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__(); self.linear=nn.Linear(hidden_size,1)
    def forward(self,h): return self.linear(h).squeeze(-1)

class WavLMAAIST(nn.Module):
    def __init__(self, cfg):
        super().__init__(); self.cfg=cfg
        self.wavlm=WavLMModel.from_pretrained(cfg.wavlm_name)
        if cfg.freeze_feature_extractor:
            for p in self.wavlm.feature_extractor.parameters(): p.requires_grad_(False)
            for p in self.wavlm.feature_projection.parameters(): p.requires_grad_(False)
        for i,layer in enumerate(self.wavlm.encoder.layers):
            if i<cfg.freeze_encoder_layers:
                for p in layer.parameters(): p.requires_grad_(False)
        if cfg.gradient_checkpointing:
            self.wavlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
        num_layers=self.wavlm.config.num_hidden_layers+1
        self.num_layers=num_layers; self.layer_weights=nn.Parameter(torch.ones(num_layers))
        self.head=HSTGATHead(cfg) if cfg.use_hst_graph else HGATHead(cfg)
        self.lfcc_branch=LFCCBranch(cfg) if cfg.use_lfcc_branch else None
        self.frame_head=FrameLevelHead(cfg.hidden_size)
        G=cfg.graph_hidden; total_dim=2*G+(G if cfg.use_lfcc_branch else 0)
        self.classifier=nn.Sequential(nn.Linear(total_dim,total_dim//2),nn.GELU(),nn.Dropout(cfg.dropout),nn.Linear(total_dim//2,1))
    def _fuse_hidden(self,wav):
        out=self.wavlm(wav,output_hidden_states=True); all_hidden=out.hidden_states
        if len(all_hidden)!=self.num_layers: raise RuntimeError(f"Expected {self.num_layers} hidden states, got {len(all_hidden)}")
        stacked=torch.stack(all_hidden,dim=1)
        weights=torch.softmax(self.layer_weights,dim=0)
        return (stacked*weights.view(1,-1,1,1)).sum(1), all_hidden
    def forward(self,wav):
        fused,_=self._fuse_hidden(wav); graph_emb=self.head(fused,wav)
        combined=torch.cat([graph_emb,self.lfcc_branch(wav)],dim=-1) if self.lfcc_branch else graph_emb
        return self.classifier(combined).squeeze(-1)
    def forward_train(self,wav,labels=None,labels_b=None,lam=1.0):

        fused,_=self._fuse_hidden(wav)
        frame_logits=None
        if self.cfg._frame_loss_available and self.cfg.use_frame_loss:
            try: frame_logits=self.frame_head(fused)
            except: frame_logits=None
        la=labels
        lb=labels_b if labels_b is not None else labels
        lam_out=lam
        already_mixed = (labels is not None) and (lam_out < 1.0 - 1e-6)
        if self.training and self.cfg.use_manifold_mixup and labels is not None and not already_mixed:
            fused,la,lb,lam_out=manifold_mixup(fused,labels,labels,1.,mix_prob=0.5)
        graph_emb=self.head(fused,wav)
        combined=torch.cat([graph_emb,self.lfcc_branch(wav)],dim=-1) if self.lfcc_branch else graph_emb
        return {"logits":self.classifier(combined).squeeze(-1),"embedding":combined,
                "frame_logits":frame_logits,"la":la,"lb":lb,"lam":lam_out}

# ── Loss functions ───────────────────────────────────────────
class AsymmetricFocalLoss(nn.Module):
    def __init__(self, gamma_pos=2., gamma_neg=1., smoothing=0.05):
        super().__init__(); self.gamma_pos=gamma_pos; self.gamma_neg=gamma_neg; self.smoothing=smoothing
    def forward(self, logits, targets):
        y=targets.float(); y_s=y*(1-self.smoothing)+0.5*self.smoothing if self.smoothing>0 else y
        bce=F.binary_cross_entropy_with_logits(logits,y_s,reduction="none")
        p=torch.sigmoid(logits); p_t=p*y_s+(1-p)*(1-y_s)
        gamma=self.gamma_pos*y_s+self.gamma_neg*(1-y_s)
        return ((1-p_t).pow(gamma)*bce).mean()

class FocalBCELoss(nn.Module):
    def __init__(self, gamma=2., smoothing=0.05):
        super().__init__(); self.gamma=gamma; self.smoothing=smoothing
    def forward(self, logits, targets):
        y=targets.float(); y_s=y*(1-self.smoothing)+0.5*self.smoothing if self.smoothing>0 else y
        bce=F.binary_cross_entropy_with_logits(logits,y_s,reduction="none")
        if self.gamma>0:
            p=torch.sigmoid(logits); p_t=p*y_s+(1-p)*(1-y_s); bce=((1-p_t)**self.gamma)*bce
        return bce.mean()

class RDropLoss(nn.Module):
    def __init__(self, base_criterion, alpha=4.):
        super().__init__(); self.base_criterion=base_criterion; self.alpha=alpha
    def forward(self, logits1, logits2, targets):
        ce=(self.base_criterion(logits1,targets)+self.base_criterion(logits2,targets))*0.5
        l1=logits1.float(); l2=logits2.float()
        p1=torch.sigmoid(l1).clamp(1e-6,1-1e-6); p2=torch.sigmoid(l2).clamp(1e-6,1-1e-6)
        kl_12=(p1*torch.log(p1/p2)+(1-p1)*torch.log((1-p1)/(1-p2))).mean()
        kl_21=(p2*torch.log(p2/p1)+(1-p2)*torch.log((1-p2)/(1-p1))).mean()
        return ce+self.alpha*(kl_12+kl_21)*0.5

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.10):   # raised from 0.07 for fp16 stability
        super().__init__(); self.temperature=temperature
    def forward(self, embeddings, labels):
        B=embeddings.shape[0]
        if B<2: return embeddings.sum()*0.
        emb=F.normalize(embeddings.float(),p=2,dim=1)
        sim=torch.matmul(emb,emb.T)/self.temperature
        mask_self=torch.eye(B,device=embeddings.device)
        mask_pos=(labels.view(-1,1)==labels.view(1,-1)).float()-mask_self
        sim_max=sim.detach().max(dim=1,keepdim=True).values; sim=sim-sim_max
        exp_sim=torch.exp(sim)*(1-mask_self)
        log_prob=sim-torch.log(exp_sim.sum(dim=1,keepdim=True).clamp(min=1e-9))
        n_pos=mask_pos.sum(dim=1); has_pos=(n_pos>0)
        if not has_pos.any(): return embeddings.sum()*0.
        per_sample=-(mask_pos*log_prob).sum(dim=1)/(n_pos+1e-12)
        return per_sample[has_pos].mean()

def build_criterion(cfg):
    if cfg.use_asymmetric_focal:
        return AsymmetricFocalLoss(cfg.focal_gamma_pos,cfg.focal_gamma_neg,cfg.label_smoothing)
    return FocalBCELoss(cfg.focal_gamma_pos,cfg.label_smoothing)

# ── EMA ──────────────────────────────────────────────────────
def get_ema_decay(epoch, total_epochs, start=0.990, end=0.9999):
    return start+(end-start)*epoch/max(total_epochs,1)

class ModelEMA:
    def __init__(self, model, decay=0.9999):
        self.decay=decay
        self.shadow={n:p.data.clone().detach() for n,p in model.named_parameters()}
        self.backup={}
    @torch.no_grad()
    def update(self, model, decay=None):
        d=decay if decay is not None else self.decay
        for n,p in model.named_parameters():
            if n in self.shadow: self.shadow[n].mul_(d).add_(p.data,alpha=1.-d)
            else: self.shadow[n]=p.data.clone().detach()
    def apply_shadow(self, model):
        for n,p in model.named_parameters():
            if n in self.shadow: self.backup[n]=p.data.clone(); p.data.copy_(self.shadow[n])
    def restore(self, model):
        for n,p in model.named_parameters():
            if n in self.backup: p.data.copy_(self.backup[n])
        self.backup={}
    def to(self, device):
        self.shadow={k:v.to(device) for k,v in self.shadow.items()}; return self

# ── Metrics ──────────────────────────────────────────────────
def _is_single_class(labels): return len(np.unique(labels))<2

def _safe_defaults():
    return {"auc":0.5,"eer":0.5,"eer_threshold":0.5,"f1":0.,"precision":0.,"recall":0.,
            "accuracy":0.5,"tp":0,"tn":0,"fp":0,"fn":0}

def compute_eer(labels, scores):
    if _is_single_class(labels): return 0.5,0.5
    fpr,tpr,thresholds=roc_curve(labels,scores,drop_intermediate=True); fnr=1.-tpr
    diff=fpr-fnr; sign_change=np.where(np.diff(np.sign(diff)))[0]
    if len(sign_change)==0:
        idx=np.argmin(np.abs(diff)); return float((fpr[idx]+fnr[idx])/2),float(thresholds[idx])
    i=sign_change[0]; x1,y1=fpr[i],fnr[i]; x2,y2=fpr[i+1],fnr[i+1]
    denom=(y2-x2)-(y1-x1); t=(x1-y1)/denom if abs(denom)>1e-12 else 0.5
    t=float(np.clip(t,0,1)); eer_val=x1+t*(x2-x1)
    return float(eer_val), float(thresholds[i]+t*(thresholds[i+1]-thresholds[i]))

def compute_tDCF(labels,scores,Pspoof=0.05,Cmiss_asv=1.,Cfa_asv=10.,Cmiss_cm=1.,Cfa_cm=10.,asv_fnr=0.0028,asv_fpr=0.02):
    if _is_single_class(labels): return 0.,0.
    try:
        fpr,tpr,thresholds=roc_curve(labels,scores,drop_intermediate=False); fnr=1.-tpr
        Ptar=1.-Pspoof
        C1=Pspoof*(Cfa_cm+Cmiss_asv*asv_fnr+Cfa_asv*asv_fpr)
        C2=Cmiss_cm*Ptar*Cmiss_asv+Pspoof*Cfa_asv*asv_fpr
        tdcf=C1*fpr+C2*fnr; min_tdcf=float(tdcf.min())
        eer_val,eer_thresh=compute_eer(labels,scores)
        idx=np.argmin(np.abs(thresholds-eer_thresh))
        return min_tdcf,float(tdcf[min(idx,len(tdcf)-1)])
    except Exception as e: print(f"  [tDCF] Warning: {e}"); return 0.,0.

def compute_minDCF(labels,scores,p_target=0.05):
    if _is_single_class(labels): return 0.
    try:
        fpr,tpr,_=roc_curve(labels,scores,drop_intermediate=False); fnr=1.-tpr
        return float((p_target*fnr+(1-p_target)*fpr).min())
    except: return 0.

def compute_cllr(labels,scores):
    if _is_single_class(labels): return 1.
    try:
        scores=np.clip(scores,1e-7,1-1e-7)
        tar=scores[labels==1]; non=scores[labels==0]
        cllr_tar=np.mean(np.log2(1+(1-tar)/tar)) if len(tar)>0 else 1.
        cllr_non=np.mean(np.log2(1+non/(1-non))) if len(non)>0 else 1.
        return float((cllr_tar+cllr_non)/2)
    except: return 1.

def optimise_threshold(labels, scores, metric="f1"):
    if _is_single_class(labels): return 0.5
    if metric=="eer": _,thresh=compute_eer(labels,scores); return thresh
    sorted_idx=np.argsort(-scores); sorted_labels=labels[sorted_idx]
    tp=np.cumsum(sorted_labels); fp=np.cumsum(1-sorted_labels); fn=sorted_labels.sum()-tp
    with np.errstate(divide="ignore",invalid="ignore"):
        if metric=="f1":
            denom=2*tp+fp+fn; sv=np.where(denom>0,2*tp/denom,0.)
        else:
            tn=(1-sorted_labels).sum()-fp; sv=(tp+tn)/len(labels)
    return float(scores[sorted_idx[int(np.argmax(sv))]])

def compute_all_metrics(labels, scores, threshold=0.5):
    if len(labels)==0 or _is_single_class(labels): return _safe_defaults()
    preds=(scores>=threshold).astype(int)
    eer,eer_thresh=compute_eer(labels,scores)
    auc_val=roc_auc_score(labels,scores)
    tn,fp,fn,tp=confusion_matrix(labels,preds,labels=[0,1]).ravel()
    min_tdcf,_=compute_tDCF(labels,scores)
    return {"auc":float(auc_val),"eer":float(eer),"eer_threshold":float(eer_thresh),
            "f1":float(f1_score(labels,preds,zero_division=0)),
            "precision":float(precision_score(labels,preds,zero_division=0)),
            "recall":float(recall_score(labels,preds,zero_division=0)),
            "accuracy":float(accuracy_score(labels,preds)),
            "min_tDCF":float(min_tdcf),"min_DCF":float(compute_minDCF(labels,scores)),
            "cllr":float(compute_cllr(labels,scores)),
            "tp":int(tp),"tn":int(tn),"fp":int(fp),"fn":int(fn)}

# ── Temperature Scaler ───────────────────────────────────────
class TemperatureScaler:
    def __init__(self): self.temperature=1.
    def fit(self, logits, labels):
        from scipy.optimize import minimize_scalar
        def nll(T):
            p=np.clip(1./(1.+np.exp(-logits/max(T,1e-6))),1e-7,1-1e-7)
            return -(labels*np.log(p)+(1-labels)*np.log(1-p)).mean()
        try: self.temperature=float(minimize_scalar(nll,bounds=(0.1,10.),method="bounded").x)
        except Exception as e: print(f"  [TempScaler] {e}"); self.temperature=1.
        return self.temperature
    def scale(self, logits): return logits/max(self.temperature,1e-6)

# ── Optimizer + Scheduler — FIX-3: alias added ───────────────
def build_stage1_optimizer_scheduler(model, cfg, total_steps):
    def no_wd(n): return "bias" in n or "norm" in n.lower()
    wd,no=[],[]
    for attr in ("head","lfcc_branch","frame_head","classifier","layer_weights"):
        m=getattr(model,attr,None)
        if m is None: continue
        if isinstance(m,nn.Parameter): wd.append(m); continue
        for n,p in m.named_parameters():
            if p.requires_grad: (no if no_wd(n) else wd).append(p)
    param_groups=[]
    if wd: param_groups.append({"params":wd,"lr":cfg.lr,"weight_decay":cfg.weight_decay})
    if no: param_groups.append({"params":no,"lr":cfg.lr,"weight_decay":0.})
    if not param_groups:
        param_groups=[{"params":[p for p in model.parameters() if p.requires_grad],"lr":cfg.lr,"weight_decay":cfg.weight_decay}]
    optimizer=AdamW(param_groups,betas=(0.9,0.999),eps=1e-8)
    warmup=int(total_steps*cfg.warmup_ratio)
    scheduler=get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=warmup,num_training_steps=total_steps)
    print(f"  [Stage1] {len(param_groups)} param groups | head-only training")
    return optimizer,scheduler

def build_stage2_optimizer_scheduler(model, cfg, total_steps):
    def no_wd(n): return "bias" in n or "norm" in n.lower()
    num_enc=len(model.wavlm.encoder.layers); param_groups=[]
    proj_wd,proj_no=[],[]
    for n,p in model.wavlm.feature_projection.named_parameters():
        if p.requires_grad: (proj_no if no_wd(n) else proj_wd).append(p)
    lr_proj=cfg.lr_wavlm*(cfg.layerwise_lr_decay**num_enc)
    if proj_wd: param_groups.append({"params":proj_wd,"lr":lr_proj,"weight_decay":cfg.weight_decay,"name":"proj_wd"})
    if proj_no: param_groups.append({"params":proj_no,"lr":lr_proj,"weight_decay":0.,"name":"proj_no"})
    for li,layer in enumerate(model.wavlm.encoder.layers):
        llr=cfg.lr_wavlm*(cfg.layerwise_lr_decay**(num_enc-1-li))
        wd_ps,no_ps=[],[]
        for n,p in layer.named_parameters():
            if p.requires_grad: (no_ps if no_wd(n) else wd_ps).append(p)
        if wd_ps: param_groups.append({"params":wd_ps,"lr":llr,"weight_decay":cfg.weight_decay,"name":f"enc{li}_wd"})
        if no_ps: param_groups.append({"params":no_ps,"lr":llr,"weight_decay":0.,"name":f"enc{li}_no"})
    if model.layer_weights.requires_grad:
        param_groups.append({"params":[model.layer_weights],"lr":cfg.lr_wavlm,"weight_decay":0.,"name":"lw"})
    for attr in ("head","lfcc_branch","frame_head","classifier"):
        m=getattr(model,attr,None)
        if m is None: continue
        wd_ps,no_ps=[],[]
        for n,p in m.named_parameters():
            if p.requires_grad: (no_ps if no_wd(n) else wd_ps).append(p)
        if wd_ps: param_groups.append({"params":wd_ps,"lr":cfg.lr,"weight_decay":cfg.weight_decay,"name":f"{attr}_wd"})
        if no_ps: param_groups.append({"params":no_ps,"lr":cfg.lr,"weight_decay":0.,"name":f"{attr}_no"})
    param_groups=[g for g in param_groups if len(g["params"])>0]  # BUG-4: filter empty groups
    optimizer=AdamW(param_groups,betas=(0.9,0.999),eps=1e-8)
    warmup=int(total_steps*cfg.warmup_ratio)
    scheduler=get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=warmup,num_training_steps=total_steps)
    print(f"  [Stage2] {len(param_groups)} param groups | LLRD | warmup={warmup}")
    return optimizer,scheduler

# FIX-3: alias so Stage 1->2 transition code works without NameError
build_optimizer_scheduler = build_stage2_optimizer_scheduler

def _unfreeze_layer_and_extend_optimizer(model, optimizer, layer_idx, cfg):
    def no_wd(n): return "bias" in n or "norm" in n.lower()
    layer=model.wavlm.encoder.layers[layer_idx]
    num_enc=len(model.wavlm.encoder.layers)
    layer_lr=cfg.lr_wavlm*(cfg.layerwise_lr_decay**(num_enc-1-layer_idx))
    for p in layer.parameters(): p.requires_grad_(True)
    wd_ps,no_ps=[],[]
    for n,p in layer.named_parameters(): (no_ps if no_wd(n) else wd_ps).append(p)
    if wd_ps: optimizer.add_param_group({"params":wd_ps,"lr":layer_lr,"weight_decay":cfg.weight_decay})
    if no_ps: optimizer.add_param_group({"params":no_ps,"lr":layer_lr,"weight_decay":0.})
    print(f"  [Progressive Unfreeze] Layer {layer_idx} unfrozen, lr={layer_lr:.2e}")

def _move_optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for k,v in state.items():
            if isinstance(v,torch.Tensor): state[k]=v.to(device)

# ── Checkpoint manager ───────────────────────────────────────
class CheckpointManager:
    def __init__(self, output_dir, top_k=3):
        self.output_dir=Path(output_dir); self.top_k=top_k; self.heap=[]
    def save(self, state, epoch, score):
        ss=f"{score:.4f}" if math.isfinite(score) else "0.0000"
        path=str(self.output_dir/f"ckpt_epoch{epoch:03d}_auc{ss}.pt")
        torch.save(state,path); self.heap.append((score if math.isfinite(score) else 0.,path))
        self.heap.sort(key=lambda x:-x[0])
        while len(self.heap)>self.top_k:
            _,old=self.heap.pop()
            if os.path.exists(old): os.remove(old)
        return path
    def best_path(self): return self.heap[0][1] if self.heap else None
    def all_paths(self): return [p for _,p in self.heap]
    def save_resume(self, state):
        tmp=self.output_dir/"resume.pt.tmp"; final=self.output_dir/"resume.pt"
        torch.save(state,str(tmp))
        try: tmp.replace(final)
        except OSError: shutil.move(str(tmp),str(final))

# ── Mixup / CutMix ───────────────────────────────────────────
def mixup_batch(wav, labels, alpha):
    if alpha<=0: return wav,labels,labels,1.
    lam=float(np.random.beta(alpha,alpha)); idx=torch.randperm(wav.size(0),device=wav.device)
    return lam*wav+(1-lam)*wav[idx],labels,labels[idx],lam

def cutmix_batch(wav, labels, alpha):
    if alpha<=0: return wav,labels,labels,1.
    B,T=wav.shape; lam=float(np.random.beta(alpha,alpha)); cut_len=int(T*(1-lam))
    if cut_len==0: return wav,labels,labels,1.
    t1=random.randint(0,T-cut_len); t2=t1+cut_len; idx=torch.randperm(B,device=wav.device)
    wav_out=wav.clone(); wav_out[:,t1:t2]=wav[idx,t1:t2]
    return wav_out,labels,labels[idx],1.-(t2-t1)/T

def mixup_loss(criterion, logits, la, lb, lam):
    return lam*criterion(logits,la)+(1.-lam)*criterion(logits,lb)

# ── SWA — FIX-5: return swa_model ────────────────────────────
def init_swa_model(model): return AveragedModel(model)
def update_swa(swa_model, model): swa_model.update_parameters(model)

def finalise_swa(swa_model, train_loader, cfg):
    """FIX-5: returns swa_model (was silently returning None before)."""
    print("  [SWA] Updating BatchNorm statistics…")
    try: swa_update_bn(train_loader, swa_model, device=DEVICE); print("  [SWA] Done.")
    except Exception as e: print(f"  [SWA] BN update failed: {e}")
    return swa_model   # FIX-5

# FIX-4: _build_sampler helper (was NameError in hard-mining rebuild)
def _build_sampler(weights, n_samples):
    return WeightedRandomSampler(torch.tensor(weights,dtype=torch.float32),n_samples,replacement=True)

# ── Greedy model soup ─────────────────────────────────────────
def greedy_soup(checkpoint_paths, val_loader, cfg):
    if len(checkpoint_paths)<2: print("  [ModelSoup] Need ≥2 checkpoints; skipping."); return None
    def _load_state(path):
        ck=_safe_torch_load(path,map_location=DEVICE)
        return ck.get("ema_shadow",ck.get("model",{}))
    def _eval_state(state):
        m=WavLMAAIST(cfg).to(DEVICE); e=ModelEMA(m)
        e.shadow={k:v.to(DEVICE) for k,v in state.items()}; e.apply_shadow(m); m.eval()
        result=evaluate(m,val_loader,build_criterion(cfg),cfg,"val"); del m; torch.cuda.empty_cache()
        return result["auc"]
    print(f"  [ModelSoup] Starting greedy soup with {len(checkpoint_paths)} checkpoints…")
    soup_state=_load_state(checkpoint_paths[0]); soup_auc=_eval_state(soup_state)
    print(f"    Seed AUC ({Path(checkpoint_paths[0]).name}): {soup_auc:.4f}")
    for ckpt_path in checkpoint_paths[1:]:
        try:
            cand=_load_state(ckpt_path)
            avg={k:(soup_state[k]+cand[k])/2 for k in soup_state if k in cand}
            avg_auc=_eval_state(avg); print(f"    {Path(ckpt_path).name}: avg={avg_auc:.4f} vs soup={soup_auc:.4f}")
            if avg_auc>soup_auc: soup_state=avg; soup_auc=avg_auc; print(f"    → Accepted. New soup AUC: {soup_auc:.4f}")
            else: print("    → Rejected.")
        except Exception as e: print(f"    → Error {ckpt_path}: {e}")
    print(f"  [ModelSoup] Final soup AUC: {soup_auc:.4f}")
    soup_model=WavLMAAIST(cfg).to(DEVICE); ema_f=ModelEMA(soup_model)
    ema_f.shadow={k:v.to(DEVICE) for k,v in soup_state.items()}; ema_f.apply_shadow(soup_model)
    return soup_model

# ── MetricsCSVLogger — FIX-7: only one definition ────────────
class MetricsCSVLogger:
    def __init__(self, path):
        self.path=path
        if not Path(path).exists():
            with open(path,"w") as f:
                f.write("epoch,global_step,train_loss,val_loss,val_auc,val_eer,"
                        "val_f1,val_acc,val_min_tDCF,val_min_DCF,lr_backbone,lr_head\n")
    def log(self, epoch, global_step, train_loss, vm, lr_bb, lr_h):
        with open(self.path,"a") as f:
            f.write(f"{epoch},{global_step},{train_loss:.6f},"
                    f"{vm.get('loss',0):.6f},{vm.get('auc',0):.6f},{vm.get('eer',0):.6f},"
                    f"{vm.get('f1',0):.6f},{vm.get('accuracy',0):.6f},"
                    f"{vm.get('min_tDCF',1):.6f},{vm.get('min_DCF',1):.6f},"
                    f"{lr_bb:.2e},{lr_h:.2e}\n")

# ── train_one_epoch — FIX-9: returns (loss, stopped_early, steps_done, worker_crashed) ──
def train_one_epoch(model, loader, optimizer, scheduler, criterion, scaler, ema, cfg, epoch,
                    rdrop_loss_fn=None, supcon_loss_fn=None,
                    mid_eval_callback=None, global_step_start=0):

    model.train(); total_loss=0.; n_backward=0; accum_count=0
    accum_steps_done=0; stopped_early=False; n_skipped=0
    optimizer.zero_grad(); total_steps=len(loader); grad_norm_log={}
    worker_crashed=False

    aux_scale = float(min(1.0, (epoch-1) / max(cfg.aux_loss_warmup_epochs, 1))) \
                if cfg.aux_loss_warmup_epochs > 0 else 1.0

    loader_iter = iter(loader)
    step = -1
    while True:
        step += 1
        try:
            batch = next(loader_iter)
        except StopIteration:
            break
        except RuntimeError as e:
            msg = str(e)
            if "DataLoader worker" in msg or "worker" in msg.lower() or "signal" in msg.lower():
                print(f"  [DataLoader] WARNING: worker crash while fetching step {step}: {e}")
                worker_crashed = True
                break
            raise
        is_last=(step==total_steps-1)
        if batch.get("n",0)==0:
            n_skipped+=1
            if is_last and accum_count>0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(),cfg.max_grad_norm)
                scaler.step(optimizer); scaler.update(); scheduler.step()
                optimizer.zero_grad()
                ema.update(model,decay=get_ema_decay(epoch,cfg.epochs,cfg.ema_decay_start,cfg.ema_decay_end))
                accum_steps_done+=1; accum_count=0

            continue

        wav=batch["wav"].to(DEVICE,non_blocking=True)
        labels=batch["labels"].to(DEVICE,non_blocking=True)
        fl=batch.get("frame_labels")
        if fl is not None: fl=fl.to(DEVICE,non_blocking=True)

        used_cutmix = cfg.use_cutmix and random.random()<cfg.cutmix_prob
        if used_cutmix:
            wav_m,la,lb,lam=cutmix_batch(wav,labels,cfg.cutmix_alpha)
        else:
            wav_m,la,lb,lam=mixup_batch(wav,labels,cfg.mixup_alpha)

        try:
            with autocast(enabled=cfg.amp,device_type=_DEVICE_TYPE):
                if cfg.use_rdrop and rdrop_loss_fn is not None:

                    out1=model.forward_train(wav_m,labels=la,labels_b=lb,lam=lam)
                    out2=model.forward_train(wav_m,labels=la,labels_b=lb,lam=lam)
                    clip_loss=rdrop_loss_fn(out1["logits"],out2["logits"],la)
                    emb=out1["embedding"]; fl_logits=out1["frame_logits"]
                else:
.
                    out=model.forward_train(wav_m,labels=la,labels_b=lb,lam=lam)
                    clip_loss=mixup_loss(criterion,out["logits"],out["la"],out["lb"],out["lam"])
                    emb=out["embedding"]; fl_logits=out["frame_logits"]
                loss=clip_loss
                if cfg.use_supcon and supcon_loss_fn is not None and aux_scale>0:
                    try: loss=loss+cfg.supcon_weight*aux_scale*supcon_loss_fn(emb,la)
                    except: pass

                if (cfg.use_frame_loss and cfg._frame_loss_available and fl_logits is not None
                        and fl is not None and aux_scale>0 and not used_cutmix):
                    try:
                        T_l=fl_logits.shape[1]; T_f=fl.shape[1]
                        fl_r=F.interpolate(fl_logits.unsqueeze(1).float(),size=T_f,mode="linear",align_corners=False).squeeze(1) if T_l!=T_f else fl_logits
                        loss=loss+cfg.frame_loss_weight*aux_scale*F.binary_cross_entropy_with_logits(fl_r,fl.float())
                    except: pass
                loss=loss/cfg.grad_accum

            if not torch.isfinite(loss):
                print(f"  WARNING: Non-finite loss at step {step} — skipping batch")
                optimizer.zero_grad(); accum_count=0; continue

            scaler.scale(loss).backward()
            accum_count+=1; n_backward+=1; total_loss+=loss.item()*cfg.grad_accum

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n  OOM at step {step} — clearing cache")
                torch.cuda.empty_cache(); optimizer.zero_grad(); accum_count=0; continue
            raise

        should_step=(accum_count>=cfg.grad_accum) or is_last
        if should_step and accum_count>0:
            scaler.unscale_(optimizer)
            if cfg.log_grad_norms and step%100==0:
                for name,param in model.named_parameters():
                    if param.grad is not None:
                        if "wavlm.encoder.layers" in name:
                            try: li=int(name.split("wavlm.encoder.layers.")[1].split(".")[0]); key=f"grad/enc{li:02d}"
                            except: key="grad/backbone"
                        elif "head" in name: key="grad/head"
                        elif "lfcc" in name: key="grad/lfcc"
                        else: key="grad/other"
                        grad_norm_log.setdefault(key,[]).append(param.grad.norm().item())
            nn.utils.clip_grad_norm_(model.parameters(),cfg.max_grad_norm)
            if torch.cuda.is_available() and scaler.is_enabled() and hasattr(scaler,"get_scale"):
                if scaler.get_scale()<256: print(f"  WARNING: GradScaler scale={scaler.get_scale():.0f}")
            scaler.step(optimizer); scaler.update(); scheduler.step(); optimizer.zero_grad()
            ema.update(model,decay=get_ema_decay(epoch,cfg.epochs,cfg.ema_decay_start,cfg.ema_decay_end))
            accum_count=0; accum_steps_done+=1

            # FIX-2 / FIX-10: mid-epoch eval + checkpoint
            if (mid_eval_callback is not None and cfg.eval_every_n_steps>0
                    and accum_steps_done%cfg.eval_every_n_steps==0):
                should_stop=mid_eval_callback(epoch, global_step_start+accum_steps_done,
                                               total_loss/max(n_backward,1))
                model.train()    # restore training mode (evaluate() sets eval mode)
                if should_stop: stopped_early=True; break

        if step%50==0:
            lr=optimizer.param_groups[0]["lr"]
            mem=f" | GPU {torch.cuda.memory_allocated()/1e9:.1f}GB" if torch.cuda.is_available() else ""
            print(f"  Ep{epoch} | step {step}/{total_steps} | loss {total_loss/max(n_backward,1):.4f} | "
                  f"lr {lr:.2e} | aux_scale {aux_scale:.2f}{mem}")

    if n_skipped: print(f"  Epoch {epoch}: {n_skipped}/{total_steps} batches skipped")
    if cfg.log_grad_norms and grad_norm_log:
        for k,v in sorted(grad_norm_log.items()): print(f"    {k}: mean_norm={np.mean(v):.4f}")
    return total_loss/max(n_backward,1), stopped_early, accum_steps_done, worker_crashed  # FIX-9

# ── Evaluation ────────────────────────────────────────────────
@torch.inference_mode()
def evaluate(model, loader, criterion, cfg, split="val"):
    model.eval(); all_logits=[]; all_labels=[]; total_loss=0.; n_batches=n_empty=0
    for batch in loader:
        if batch.get("n",0)==0: n_empty+=1; continue
        wav=batch["wav"].to(DEVICE,non_blocking=True); labels=batch["labels"].to(DEVICE,non_blocking=True)
        with autocast(enabled=cfg.amp,device_type=_DEVICE_TYPE):
            logits=model(wav); loss=criterion(logits,labels)
        all_logits.append(logits.cpu().float()); all_labels.append(labels.cpu())
        total_loss+=loss.item(); n_batches+=1
    if n_empty>0: print(f"  [{split}] {n_empty} empty batches")
    if not all_logits: print(f"  [{split}] WARNING: Zero valid batches!"); return {**_safe_defaults(),"loss":0.,"best_threshold":0.5}
    logits_np=torch.cat(all_logits).numpy(); labels_np=torch.cat(all_labels).numpy()
    scores_np=1./(1.+np.exp(-logits_np))
    if _is_single_class(labels_np):
        m=_safe_defaults(); m["loss"]=total_loss/max(n_batches,1); m["best_threshold"]=0.5
        torch.cuda.empty_cache(); return m
    best_thresh=optimise_threshold(labels_np,scores_np,metric="f1")
    metrics=compute_all_metrics(labels_np,scores_np,best_thresh)
    metrics["loss"]=total_loss/max(n_batches,1); metrics["best_threshold"]=best_thresh
    torch.cuda.empty_cache(); return metrics

def evaluate_with_monitoring(model, loader, criterion, cfg, ema, split="val"):
    if cfg.log_ema_vs_raw:
        raw_auc=evaluate(model,loader,criterion,cfg,split+"_raw")["auc"]
    else: raw_auc=None
    ema.apply_shadow(model); ema_metrics=evaluate(model,loader,criterion,cfg,split); ema.restore(model)
    if raw_auc is not None:
        print(f"  Raw AUC={raw_auc:.4f} | EMA AUC={ema_metrics['auc']:.4f}")
        if ema_metrics["auc"]<raw_auc-0.002: print("  ⚠ EMA not helping — consider adjusting ema_decay_start.")
    if cfg.log_layer_weights:
        try:
            w=torch.softmax(model.layer_weights,dim=0).detach().cpu().numpy(); top3=np.argsort(-w)[:3]
            print(f"  WavLM top-3 layers: {list(top3)} (weights {w[top3].round(3).tolist()})")
            ema_metrics["layer_weights"]=w.tolist()
        except: pass
    return ema_metrics

# ── TTA evaluation ────────────────────────────────────────────
@torch.inference_mode()
def tta_evaluate(model, loader, cfg):
    model.eval(); all_scores=[]; all_labels=[]
    for batch in loader:
        if batch.get("n",0)==0: continue
        wav_b=batch["wav"]; labels_b=batch["labels"]; B,T=wav_b.shape
        tgt=cfg.segment_samples; acc=torch.zeros(B); n_views=0
        for _ in range(cfg.tta_n_crops):
            if T>tgt:
                offsets=torch.randint(0,T-tgt+1,(B,))
                crops=torch.stack([wav_b[i,offsets[i]:offsets[i]+tgt] for i in range(B)])
            else: crops=F.pad(wav_b,(0,max(0,tgt-T)))[:,:tgt]
            with autocast(enabled=cfg.amp,device_type=_DEVICE_TYPE):
                acc+=torch.sigmoid(model(crops.to(DEVICE))).cpu()
            n_views+=1
        for frac in [0.,0.5,1.]:
            start=int(frac*max(0,T-tgt)); chunk=wav_b[:,start:min(start+tgt,T)]
            if chunk.shape[1]<tgt: chunk=F.pad(chunk,(0,tgt-chunk.shape[1]))
            with autocast(enabled=cfg.amp,device_type=_DEVICE_TYPE):
                acc+=torch.sigmoid(model(chunk.to(DEVICE))).cpu()
            n_views+=1
        all_scores.append(acc/n_views); all_labels.append(labels_b)
    if not all_scores: return _safe_defaults()
    scores_np=torch.cat(all_scores).numpy(); labels_np=torch.cat(all_labels).numpy()
    if _is_single_class(labels_np): return _safe_defaults()
    auc=roc_auc_score(labels_np,scores_np); best_t=optimise_threshold(labels_np,scores_np,"f1")
    m=compute_all_metrics(labels_np,scores_np,best_t)
    print(f"\n  [TTA] AUC={auc:.4f}  F1={m['f1']:.4f}  EER={m['eer']:.4f}")
    return m

# ── Figures ───────────────────────────────────────────────────
def save_roc_curve(labels,scores,auc_val,path):
    fpr,tpr,_=roc_curve(labels,scores); fig,ax=plt.subplots(figsize=(7,6))
    ax.plot(fpr,tpr,lw=2,label=f"AUC={auc_val:.4f}"); ax.plot([0,1],[0,1],"k--",lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC Curve"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path,dpi=150); plt.close(fig); print(f"  ROC curve → {path}")

def save_score_distribution(labels,scores,path):
    fig,ax=plt.subplots(figsize=(8,5))
    ax.hist(scores[labels==0],bins=80,alpha=0.6,label="Real",color="steelblue")
    ax.hist(scores[labels==1],bins=80,alpha=0.6,label="Fake",color="tomato")
    ax.set_title("Score Distribution"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path,dpi=150); plt.close(fig); print(f"  Score dist → {path}")

def save_confusion_matrix(labels,preds,path):
    cm=confusion_matrix(labels,preds,labels=[0,1]); fig,ax=plt.subplots(figsize=(5,4))
    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",ax=ax,xticklabels=["Real","Fake"],yticklabels=["Real","Fake"])
    ax.set_title("Confusion Matrix"); fig.tight_layout(); fig.savefig(path,dpi=150); plt.close(fig)
    print(f"  Confusion matrix → {path}")

def save_training_curves(history,path):
    epochs=[r["epoch"] for r in history]; fig,axes=plt.subplots(1,2,figsize=(12,4))
    axes[0].plot(epochs,[r["train_loss"] for r in history],label="Train"); axes[0].plot(epochs,[r.get("loss",0) for r in history],label="Val")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(epochs,[r.get("auc",0) for r in history],marker="o",ms=3); axes[1].set_title("Val AUC"); axes[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path,dpi=150); plt.close(fig); print(f"  Training curves → {path}")

def save_reliability_diagram(labels,scores,path,n_bins=10):
    try:
        bins=np.linspace(0,1,n_bins+1); bc=[]; fp=[]
        for lo,hi in zip(bins[:-1],bins[1:]):
            mask=(scores>=lo)&(scores<hi)
            if mask.sum()>0: bc.append((lo+hi)/2); fp.append(labels[mask].mean())
        fig,ax=plt.subplots(figsize=(6,5)); ax.plot([0,1],[0,1],"k--",label="Ideal"); ax.plot(bc,fp,"o-",label="Model")
        ax.set_title("Reliability Diagram"); ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(path,dpi=150); plt.close(fig); print(f"  Reliability diagram → {path}")
    except Exception as e: print(f"  Reliability diagram skipped: {e}")

# ── Extended evaluation ───────────────────────────────────────
@torch.inference_mode()
def extended_evaluation(model, loader, cfg, ema=None, val_threshold=0.5, temp_scaler=None):
    if ema is not None: ema.apply_shadow(model)
    model.eval(); all_logits=[]; all_labels=[]; all_fids=[]
    for batch in loader:
        if batch.get("n",0)==0: continue
        wav=batch["wav"].to(DEVICE,non_blocking=True)
        with autocast(enabled=cfg.amp,device_type=_DEVICE_TYPE): logits=model(wav)
        all_logits.append(logits.cpu().float()); all_labels.append(batch["labels"].cpu())
        all_fids.extend(batch.get("file_id",[""]*(batch["n"])))
    if ema is not None: ema.restore(model)
    torch.cuda.empty_cache()
    if not all_logits:
        print("\nWARNING: Test loader yielded no valid batches.")
        return {**_safe_defaults(),"scores":np.array([]),"labels":np.array([]),"fpr":np.array([]),"tpr":np.array([])}
    logits_np=torch.cat(all_logits).numpy(); labels_np=torch.cat(all_labels).numpy()
    if temp_scaler is not None and temp_scaler.temperature!=1.: logits_np=temp_scaler.scale(logits_np)
    scores_np=1./(1.+np.exp(-logits_np))
    if _is_single_class(labels_np):
        print("\nWARNING: Test set one class — metrics undefined.")
        return {**_safe_defaults(),"scores":scores_np,"labels":labels_np,"fpr":np.array([]),"tpr":np.array([])}
    preds_np=(scores_np>=val_threshold).astype(int)
    pred_df=pd.DataFrame({"file_id":all_fids,"true_label":labels_np,"pred_score":scores_np,"pred_label":preds_np})
    pred_csv=Path(cfg.log_dir)/"predictions.csv"; pred_df.to_csv(str(pred_csv),index=False); print(f"  Predictions → {pred_csv}")
    fpr,tpr,thresholds=roc_curve(labels_np,scores_np); auc_val=roc_auc_score(labels_np,scores_np)
    eer,eer_thresh=compute_eer(labels_np,scores_np); min_tDCF,_=compute_tDCF(labels_np,scores_np)
    min_dcf=compute_minDCF(labels_np,scores_np)
    print("\n"+"="*60+"\nEXTENDED TEST EVALUATION\n"+"="*60)
    print(f"AUC={auc_val:.4f}  EER={eer:.4f}(thresh={eer_thresh:.4f})  min-tDCF={min_tDCF:.4f}  min-DCF={min_dcf:.4f}")
    if temp_scaler: print(f"Temperature: {temp_scaler.temperature:.4f}")
    print("ROC at key FPR points:")
    for tfpr in [0.001,0.01,0.05,0.1,0.2]:
        idx=np.searchsorted(fpr,tfpr)
        if 0<idx<len(thresholds): print(f"  FPR={fpr[idx]:.4f}  TPR={tpr[idx]:.4f}  T={thresholds[idx]:.4f}")
    r,f=scores_np[labels_np==0],scores_np[labels_np==1]
    print(f"Real: mean={r.mean():.4f} std={r.std():.4f} | Fake: mean={f.mean():.4f} std={f.std():.4f}")
    all_m=compute_all_metrics(labels_np,scores_np,val_threshold)
    print(f"F1={all_m['f1']:.4f} Acc={all_m['accuracy']:.4f} P={all_m['precision']:.4f} R={all_m['recall']:.4f}")
    print(f"TP={all_m['tp']} TN={all_m['tn']} FP={all_m['fp']} FN={all_m['fn']}")
    fd=cfg.figures_dir
    save_roc_curve(labels_np,scores_np,auc_val,os.path.join(fd,"roc_curve.png"))
    save_score_distribution(labels_np,scores_np,os.path.join(fd,"score_dist.png"))
    save_confusion_matrix(labels_np,preds_np,os.path.join(fd,"conf_matrix.png"))
    save_reliability_diagram(labels_np,scores_np,os.path.join(fd,"reliability.png"))
    return {"auc":float(auc_val),"eer":float(eer),"eer_threshold":float(eer_thresh),
            "min_tDCF":float(min_tDCF),"min_DCF":float(min_dcf),
            "scores":scores_np,"labels":labels_np,"fpr":fpr,"tpr":tpr,**all_m}

# ── Hard mining helper ────────────────────────────────────────
@torch.inference_mode()
def _compute_hard_mining_weights(model, ema, dataset, cfg):
    ema.apply_shadow(model); model.eval()
    orig_train=dataset.is_train; orig_aug=dataset.augmenter
    dataset.is_train=False; dataset.augmenter=None
    hm_loader=DataLoader(dataset,batch_size=cfg.batch_size,shuffle=False,collate_fn=collate_fn,num_workers=0,drop_last=False)
    all_fids=[]; all_losses=[]
    try:
        for batch in hm_loader:
            if batch.get("n",0)==0: continue
            wav=batch["wav"].to(DEVICE); labels=batch["labels"].float().to(DEVICE)
            with autocast(enabled=cfg.amp,device_type=_DEVICE_TYPE): logits=model(wav)
            losses=F.binary_cross_entropy_with_logits(logits,labels,reduction="none").cpu().numpy()
            all_fids.extend(batch.get("file_id",[])); all_losses.extend(losses.tolist())
    finally: dataset.is_train=orig_train; dataset.augmenter=orig_aug
    ema.restore(model)
    if not all_losses: return {}
    loss_arr=np.array(all_losses); thresh=np.percentile(loss_arr,100.*(1.-cfg.hard_mining_top_pct))
    print(f"  [HardMining] {(loss_arr>=thresh).sum()} hard samples (top {cfg.hard_mining_top_pct:.0%}, thresh={thresh:.4f})")
    return {fid:(cfg.hard_mining_weight if l>=thresh else 1.) for fid,l in zip(all_fids,all_losses)}

# ── Ensemble helper ───────────────────────────────────────────
def ensemble_predict(checkpoint_paths, loader, cfg):
    all_run_scores=[]
    for ckpt_path in checkpoint_paths:
        print(f"  [Ensemble] Loading: {ckpt_path}")
        m=WavLMAAIST(cfg).to(DEVICE); ck=_safe_torch_load(ckpt_path,map_location=DEVICE)
        if "ema_shadow" in ck:
            e=ModelEMA(m); e.shadow={k:v.to(DEVICE) for k,v in ck["ema_shadow"].items()}; e.apply_shadow(m)
        else: m.load_state_dict(ck.get("model",ck))
        m.eval(); run_logits=[]
        with torch.inference_mode():
            for batch in loader:
                if batch.get("n",0)==0: continue
                with autocast(enabled=cfg.amp,device_type=_DEVICE_TYPE):
                    run_logits.append(m(batch["wav"].to(DEVICE)).cpu().float())
        all_run_scores.append(1./(1.+np.exp(-torch.cat(run_logits).numpy())))
        del m; torch.cuda.empty_cache()
    print(f"  [Ensemble] {len(checkpoint_paths)} models combined.")
    return np.mean(all_run_scores,axis=0)

# ── IMP-3: adaptive time-budget recalibration ─────────────────
def recalibrate_time_budget(cfg, t_start, epoch_wall_h, eval_wall_sec):

    reserve_h = float(np.clip(eval_wall_sec * 3.5 / 3600.0,
                               cfg.min_reserve_eval_hours, cfg.max_reserve_eval_hours))
    new_budget = max(cfg.hard_session_hours - reserve_h, epoch_wall_h + 0.05)
    elapsed_h = (time.time() - t_start) / 3600.0
    remaining_h = max(new_budget - elapsed_h, 0.0)
    feasible_more_epochs = int(remaining_h // max(epoch_wall_h, 1e-6))
    print(f"  [TimeBudget] epoch_wall≈{epoch_wall_h:.2f}h | eval_wall≈{eval_wall_sec:.0f}s | "
          f"reserve={reserve_h:.2f}h | time_budget {cfg.time_budget_hours:.2f}h -> {new_budget:.2f}h")
    cfg.time_budget_hours = new_budget
    return new_budget, feasible_more_epochs

# ── Full training pipeline ─────────────────────────────────────
def train(cfg):
    set_seed(cfg.seed, cfg.cudnn_benchmark, seed_cuda=False)
    t_start = time.time()
    budget_recalibrated = False

    print("=" * 60)
    print("LAV-DF Audio Deepfake Detector v10 — Training")
    print("=" * 60)

    # ── [1] Metadata ──────────────────────────────────────────
    print("\n[1/7] Loading metadata...")
    df = load_metadata(cfg)

    # ── [2] Audio discovery ───────────────────────────────────
    print("\n[2/7] Discovering audio files...")
    wav_map = discover_audio_files(cfg)
    if not wav_map:
        raise FileNotFoundError("No audio files found. Check cfg.data_root / cfg.audio_glob.")

    print("\n  Spot-check (first 5 file_ids):")
    for fid in df["file_id"].head(5):
        p = _resolve_audio(str(fid), cfg.lav_df_video_root, wav_map, cfg.data_root)
        print(f"    {str(fid)[:40]:40s} → {'✓ ' + Path(p).name if p else '✗ NOT FOUND'}")

    # ── [3] DataLoaders ───────────────────────────────────────
    print("\n[3/7] Building dataloaders...")
    loaders, datasets = make_dataloaders(df, wav_map, cfg)

    quick_val_loader = build_quick_eval_loader(datasets["val"], cfg, n=cfg.quick_val_subset_size)
    print(f"  [QuickVal] mid-epoch checks will use {len(quick_val_loader.dataset)} samples "
          f"(full val = {len(datasets['val'])})")

    print("\n[3c/7] Priming DataLoader workers before CUDA init...")
    _prime_loader_workers(loaders["train"], "train")
    _prime_loader_workers(loaders["val"], "val")
    _prime_loader_workers(loaders["test"], "test")
    _prime_loader_workers(quick_val_loader, "quick_val")

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # ── [4] Model ─────────────────────────────────────────────
    print("\n[4/7] Building model...")
    model = WavLMAAIST(cfg).to(DEVICE)
    model = try_compile(model, cfg)

    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_p/1e6:.1f}M  Trainable: {train_p/1e6:.1f}M")

    ema          = ModelEMA(model, decay=cfg.ema_decay)
    criterion    = build_criterion(cfg)
    rdrop_crit   = RDropLoss(criterion, cfg.rdrop_alpha)  if cfg.use_rdrop  else None
    supcon_crit  = SupConLoss(cfg.supcon_temperature)     if cfg.use_supcon else None
    scaler       = make_scaler(enabled=cfg.amp)
    swa_model    = init_swa_model(model) if cfg.use_swa else None

    steps_per_epoch = len(loaders["train"])

    # ── Resume / pre-trained weights ──────────────────────────
    resume_path     = Path(cfg.output_dir) / "resume.pt"
    start_epoch     = 0
    best_auc        = 0.0
    no_improve      = 0
    history         = []
    best_val_thresh = 0.5
    current_frozen  = cfg.freeze_encoder_layers
    global_step     = 0

    if resume_path.exists():
        print(f"\n  Resuming from {resume_path}")
        ckpt = _safe_torch_load(str(resume_path), map_location=DEVICE)
        # Same run/same architecture -> exact load is expected to match.
        try:
            model.load_state_dict(ckpt["model"])
        except RuntimeError as e:
            print(f"  WARNING: exact resume load failed ({e}); falling back to compatible load.")
            load_compatible_state_dict(model, ckpt["model"], label="resume.pt")
        ema.shadow       = {k: v.to(DEVICE) for k, v in ckpt["ema_shadow"].items()}
        start_epoch      = ckpt["epoch"] + 1
        best_auc         = ckpt["best_auc"]
        no_improve       = ckpt.get("no_improve", 0)
        history          = ckpt.get("history", [])
        best_val_thresh  = ckpt.get("best_val_thresh", 0.5)
        current_frozen   = ckpt.get("current_frozen", cfg.freeze_encoder_layers)
        global_step      = ckpt.get("global_step", 0)
        if "rng_state" in ckpt:
            set_rng_state(ckpt["rng_state"])

        budget_recalibrated = True
        print(f"  Resumed: epoch={start_epoch}, best_auc={best_auc:.4f}, global_step={global_step}")

    elif cfg.default_resume_checkpoint and Path(cfg.default_resume_checkpoint).exists():
        print(f"\n  Loading pre-trained weights from {cfg.default_resume_checkpoint}")
        ckpt_pre = _safe_torch_load(cfg.default_resume_checkpoint, map_location=DEVICE)
        # IMP-6: shape-safe load — survives a backbone upgrade (IMP-1) gracefully.
        load_compatible_state_dict(model, ckpt_pre.get("model", ckpt_pre), label="default_resume_checkpoint")
        if "ema_shadow" in ckpt_pre:
            try:
                ema.shadow = {k: v.to(DEVICE) for k, v in ckpt_pre["ema_shadow"].items()
                              if k in ema.shadow and ema.shadow[k].shape == v.shape}
            except Exception as e:
                print(f"  [EMA] Could not restore EMA shadow from pretrained checkpoint: {e}")
        print("  Pre-trained weights loaded (compatible tensors only). Starting from epoch 0.")

    # ── Build initial optimizer ───────────────────────────────
    in_stage1 = start_epoch < cfg.stage1_epochs
    if in_stage1:
        for p in model.wavlm.parameters():
            p.requires_grad_(False)
        s1_rem   = cfg.stage1_epochs - start_epoch
        s1_steps = math.ceil(steps_per_epoch / cfg.grad_accum) * s1_rem
        optimizer, scheduler = build_stage1_optimizer_scheduler(model, cfg, s1_steps)
    else:
        for p in model.wavlm.feature_projection.parameters():
            p.requires_grad_(True)
        for i, layer in enumerate(model.wavlm.encoder.layers):
            for p in layer.parameters():
                p.requires_grad_(i >= current_frozen)
        for p in model.wavlm.feature_extractor.parameters():
            p.requires_grad_(False)
        s2_rem   = cfg.epochs - start_epoch
        s2_steps = math.ceil(steps_per_epoch / cfg.grad_accum) * s2_rem
        # FIX-3: build_optimizer_scheduler alias resolves to build_stage2_optimizer_scheduler
        optimizer, scheduler = build_optimizer_scheduler(model, cfg, s2_steps)

    if resume_path.exists():
        ckpt_r = _safe_torch_load(str(resume_path), map_location=DEVICE)
        try:
            optimizer.load_state_dict(ckpt_r["optimizer"])
            _move_optimizer_state_to_device(optimizer, DEVICE)
            scheduler.load_state_dict(ckpt_r["scheduler"])
            scaler.load_state_dict(ckpt_r["scaler"])
        except Exception as e:
            print(f"  WARNING: could not restore opt/sched/scaler state: {e}")

    ckpt_mgr   = CheckpointManager(cfg.output_dir, top_k=cfg.save_top_k)
    csv_logger = MetricsCSVLogger(str(Path(cfg.log_dir) / "metrics.csv"))

    train_labels_np = datasets["train"].df["label"].values
    class_counts    = np.bincount(train_labels_np, minlength=2)
    class_weights   = np.where(class_counts > 0, 1.0 / class_counts.astype(float), 0.0)
    base_sample_w   = class_weights[train_labels_np].astype(np.float32)
    hard_weights    = {}

    if cfg.use_wandb:
        try:
            import wandb
            wandb.init(project="lavdf-deepfake-v8", config=asdict(cfg),
                       settings=wandb.Settings(_disable_stats=True))
        except Exception as e:
            print(f"  W&B init failed: {e}"); cfg.use_wandb = False


    def _mid_epoch_eval(epoch, g_step, train_loss_so_far):
        nonlocal best_auc, best_val_thresh, no_improve

        elapsed_h = (time.time() - t_start) / 3600.0
        print(f"\n  ── Mid-epoch eval @ opt-step {g_step} "
              f"(epoch {epoch}, elapsed {elapsed_h:.2f}h/{cfg.time_budget_hours:.2f}h) ──")

        if cfg.mid_eval_use_ema_only:
            ema.apply_shadow(model)
            vm = evaluate(model, quick_val_loader, criterion, cfg, "val_quick")
            ema.restore(model)
        else:
            vm = evaluate_with_monitoring(model, quick_val_loader, criterion, cfg, ema, "val_quick")
        val_auc = vm["auc"]
        lr_bb   = optimizer.param_groups[0]["lr"]
        lr_h    = optimizer.param_groups[-1]["lr"]
        print(f"  Val AUC={val_auc:.4f}  EER={vm['eer']:.4f}  "
              f"F1={vm['f1']:.4f}  loss={vm['loss']:.4f}  "
              f"(quick subset, n={len(quick_val_loader.dataset)})")

        csv_logger.log(epoch, g_step, train_loss_so_far, vm, lr_bb, lr_h)

        state_now = {
            "epoch":           epoch - 1,
            "global_step":     g_step,
            "model":           model.state_dict(),
            "ema_shadow":      ema.shadow,
            "optimizer":       optimizer.state_dict(),
            "scheduler":       scheduler.state_dict(),
            "scaler":          scaler.state_dict(),
            "best_auc":        best_auc,
            "no_improve":      no_improve,
            "history":         _to_python(history),
            "best_val_thresh": best_val_thresh,
            "current_frozen":  current_frozen,
            "cfg":             asdict(cfg),
            "rng_state":       get_rng_state(),
        }
        ckpt_mgr.save_resume(state_now)
        save_named_checkpoint(state_now, cfg.last_model_path)   # IMP-4

        if math.isfinite(val_auc) and val_auc > best_auc:
            best_auc        = val_auc
            best_val_thresh = vm.get("best_threshold", 0.5)
            no_improve      = 0
            saved_path = ckpt_mgr.save(state_now, epoch, val_auc)
            save_named_checkpoint(state_now, cfg.best_model_path)   # IMP-4
            print(f"  ★ New best AUC: {best_auc:.4f}  → {Path(saved_path).name} / {cfg.best_model_path}")
        else:
            print(f"  No improvement (best={best_auc:.4f}, no_improve={no_improve})")

        elapsed_h = (time.time() - t_start) / 3600.0
        if elapsed_h >= cfg.time_budget_hours:
            print(f"\n  Time budget ({cfg.time_budget_hours:.2f}h) reached mid-epoch. "
                  "Stopping to allow final evaluation.")
            return True
        return False

    # ── [5] Training loop ─────────────────────────────────────
    print("\n[5/7] Training...")

    for epoch in range(start_epoch, cfg.epochs):

        elapsed_h = (time.time() - t_start) / 3600.0
        if elapsed_h >= cfg.time_budget_hours:
            print(f"\n  Time budget ({cfg.time_budget_hours:.2f}h) reached at epoch {epoch} start.")
            break

        t_epoch_begin = time.time()
        print(f"\n{'─'*55}")
        print(f"Epoch {epoch+1}/{cfg.epochs}  "
              f"[{elapsed_h:.2f}h / {cfg.time_budget_hours:.2f}h]")

        # ── Stage 1 → 2 transition [T1] ──────────────────────
        if epoch == cfg.stage1_epochs and in_stage1:
            print("\n  ══ Stage 2: unfreezing WavLM backbone ══")
            for p in model.wavlm.feature_projection.parameters():
                p.requires_grad_(True)
            for i, layer in enumerate(model.wavlm.encoder.layers):
                for p in layer.parameters():
                    p.requires_grad_(i >= current_frozen)
            for p in model.wavlm.feature_extractor.parameters():
                p.requires_grad_(False)
            in_stage1 = False
            s2_rem    = cfg.epochs - epoch
            s2_steps  = math.ceil(steps_per_epoch / cfg.grad_accum) * s2_rem
            # FIX-3: alias is build_stage2_optimizer_scheduler — no NameError
            optimizer, scheduler = build_optimizer_scheduler(model, cfg, s2_steps)

        # ── Progressive unfreezing [T1] ───────────────────────
        if not in_stage1 and current_frozen > 0:
            eps_in_s2    = epoch - cfg.stage1_epochs
            target_frozen = max(
                0,
                cfg.freeze_encoder_layers - (eps_in_s2 // max(cfg.unfreeze_every_n_epochs, 1)),
            )
            while current_frozen > target_frozen:
                _unfreeze_layer_and_extend_optimizer(model, optimizer, current_frozen - 1, cfg)
                current_frozen -= 1

        # ── Hard mining sampler rebuild [T3] ─────────────────
        if cfg.use_hard_mining and hard_weights and epoch > start_epoch:
            fids = datasets["train"].df["file_id"].values
            sw   = np.array([hard_weights.get(str(fid), 1.0) for fid in fids], dtype=np.float32)
            sw   = sw * base_sample_w
            if sw.sum() > 0:
                new_sampler = _build_sampler(sw / sw.sum() * len(sw), len(sw))
                _wif  = worker_init_fn if cfg.num_workers > 0 else None
                loaders["train"] = DataLoader(
                    datasets["train"],
                    batch_size=cfg.batch_size,
                    sampler=new_sampler,
                    shuffle=False,
                    num_workers=cfg.num_workers,
                    pin_memory=cfg.pin_memory and torch.cuda.is_available(),
                    collate_fn=collate_fn,
                    worker_init_fn=_wif,
                    prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
                    persistent_workers=(cfg.num_workers > 0),
                    drop_last=True,
                )

        train_loss, stopped_early, steps_done, worker_crashed = train_one_epoch(
            model, loaders["train"], optimizer, scheduler,
            criterion, scaler, ema, cfg, epoch + 1,
            rdrop_loss_fn=rdrop_crit,
            supcon_loss_fn=supcon_crit,
            mid_eval_callback=_mid_epoch_eval if cfg.eval_every_n_steps > 0 else None,
            global_step_start=global_step,
        )
        global_step += steps_done

        if worker_crashed:
            print("\n  [DataLoader] Worker crash detected mid-epoch — falling back to "
                  "num_workers=0 for stability and retrying this epoch once.")
            cfg.num_workers = 0
            cfg.prefetch_factor = None
            cfg.pin_memory = False
            loaders["train"] = _rebuild_train_loader(datasets["train"], cfg, base_sample_w, num_workers=0)
            train_loss, stopped_early, steps_done2, worker_crashed2 = train_one_epoch(
                model, loaders["train"], optimizer, scheduler,
                criterion, scaler, ema, cfg, epoch + 1,
                rdrop_loss_fn=rdrop_crit,
                supcon_loss_fn=supcon_crit,
                mid_eval_callback=_mid_epoch_eval if cfg.eval_every_n_steps > 0 else None,
                global_step_start=global_step,
            )
            global_step += steps_done2
            if worker_crashed2:
                raise RuntimeError(
                    "DataLoader crashed even with num_workers=0 (single-process, no forking "
                    "involved). This points to a deeper problem (e.g. genuine system-RAM "
                    "exhaustion or a corrupt/unreadable audio file) rather than a worker-fork "
                    "issue — check the traceback just above this message."
                )

        if stopped_early:
            print("\n  Epoch terminated early (time budget). "
                  "Latest checkpoint saved by mid-epoch eval. Exiting training loop.")
            break

        # ── End-of-epoch evaluation ───────────────────────────
        t_eval_begin = time.time()
        val_metrics = evaluate_with_monitoring(
            model, loaders["val"], criterion, cfg, ema, split="val"
        )
        eval_wall_sec = time.time() - t_eval_begin
        val_auc    = val_metrics["auc"]
        val_thresh = val_metrics.get("best_threshold", 0.5)
        lr_bb      = optimizer.param_groups[0]["lr"]
        lr_h       = optimizer.param_groups[-1]["lr"]

        print(f"\n  Train loss  : {train_loss:.4f}")
        print(f"  Val   loss  : {val_metrics['loss']:.4f}  AUC={val_auc:.4f}  "
              f"EER={val_metrics['eer']:.4f}  F1={val_metrics['f1']:.4f}")
        print(f"  min-tDCF    : {val_metrics.get('min_tDCF', 1):.4f}  "
              f"min-DCF={val_metrics.get('min_DCF', 1):.4f}")
        print(f"  Val thresh  : {val_thresh:.4f}  LR_head={lr_h:.2e}")

        if val_auc < 0.52 and epoch >= 1:
            print("  ⚠ Val AUC near random — check audio path resolution.")

        csv_logger.log(epoch + 1, global_step, train_loss, val_metrics, lr_bb, lr_h)

        row = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            **{k: v for k, v in val_metrics.items()
               if not isinstance(v, (np.ndarray, list))},
        }
        history.append(row)
        with open(Path(cfg.log_dir) / "history.json", "w") as f:
            json.dump(_to_python(history), f, indent=2)

        if cfg.use_wandb:
            import wandb
            wandb.log({"epoch": epoch + 1, "train_loss": train_loss,
                       **{k: v for k, v in val_metrics.items()
                          if not isinstance(v, (np.ndarray, list))}})

        # ── Checkpoint ───────────────────────────────────────
        state = {
            "epoch":           epoch,
            "global_step":     global_step,
            "model":           model.state_dict(),
            "ema_shadow":      ema.shadow,
            "optimizer":       optimizer.state_dict(),
            "scheduler":       scheduler.state_dict(),
            "scaler":          scaler.state_dict(),
            "best_auc":        best_auc,
            "no_improve":      no_improve,
            "history":         _to_python(history),
            "best_val_thresh": best_val_thresh,
            "current_frozen":  current_frozen,
            "cfg":             asdict(cfg),
            "rng_state":       get_rng_state(),
        }
        ckpt_mgr.save_resume(state)
        save_named_checkpoint(state, cfg.last_model_path)   # IMP-4: always keep "latest" up to date

        if math.isfinite(val_auc) and val_auc > best_auc:
            best_auc        = val_auc
            best_val_thresh = val_thresh
            no_improve      = 0
            saved = ckpt_mgr.save(state, epoch + 1, val_auc)
            save_named_checkpoint(state, cfg.best_model_path)   # IMP-4
            print(f"  ★ New best AUC: {best_auc:.4f}  → {Path(saved).name} / {cfg.best_model_path}")
        else:
            no_improve += 1
            print(f"  No improvement ({no_improve}/{cfg.early_stop_patience})")

        if not budget_recalibrated:
            epoch_wall_h = (time.time() - t_epoch_begin) / 3600.0
            new_budget, feasible_more = recalibrate_time_budget(cfg, t_start, epoch_wall_h, eval_wall_sec)
            projected_total_epochs = (epoch + 1) + feasible_more
            if projected_total_epochs < cfg.epochs:
                old_epochs = cfg.epochs
                cfg.epochs = max(projected_total_epochs, epoch + 2)
                if cfg.use_swa:
                    cfg.swa_start_epoch = max(epoch + 2, int(cfg.epochs * 0.75))
                print(f"  [TimeBudget] Projected only ~{projected_total_epochs} epochs fit in the "
                      f"session (was targeting {old_epochs}). cfg.epochs -> {cfg.epochs}, "
                      f"cfg.swa_start_epoch -> {cfg.swa_start_epoch}.")
                if not in_stage1:
                    remaining_epochs_for_sched = cfg.epochs - (epoch + 1)
                    if remaining_epochs_for_sched > 0:
                        new_total_steps = math.ceil(steps_per_epoch / cfg.grad_accum) * remaining_epochs_for_sched
                        optimizer, scheduler = build_optimizer_scheduler(model, cfg, new_total_steps)
                        print(f"  [TimeBudget] Scheduler rebuilt for {remaining_epochs_for_sched} "
                              "remaining epochs so LR anneals to ~0 by the real stopping point.")
                # else: stage1->2 transition later will build stage-2 schedule
                # using the already-updated cfg.epochs, so nothing else to do.
            else:
                print(f"  [TimeBudget] On pace: {projected_total_epochs} epochs projected to fit "
                      f"(target was {cfg.epochs}).")
            budget_recalibrated = True

        if no_improve >= cfg.early_stop_patience:
            print("\n  Early stopping triggered.")
            break

        # ── SWA update [T2] ──────────────────────────────────
        if cfg.use_swa and epoch + 1 >= cfg.swa_start_epoch and swa_model is not None:
            ema.apply_shadow(model)
            update_swa(swa_model, model)
            ema.restore(model)

        # ── Hard mining weight update [T3] ───────────────────
        if cfg.use_hard_mining:
            try:
                hard_weights = _compute_hard_mining_weights(model, ema, datasets["train"], cfg)
            except Exception as e:
                print(f"  [HardMining] skipped: {e}")

        gc.collect()
        torch.cuda.empty_cache()

    # ── Training curves ───────────────────────────────────────
    if history:
        save_training_curves(history, str(Path(cfg.figures_dir) / "training_curves.png"))

    # ── [6] Final test evaluation ──────────────────────────────
    print("\n[6/7] Final evaluation on test set...")
    best = ckpt_mgr.best_path()
    if best:
        print(f"  Loading best checkpoint: {Path(best).name}")
        ck = _safe_torch_load(best, map_location=DEVICE)
        model.load_state_dict(ck["model"])
        ema.shadow      = {k: v.to(DEVICE) for k, v in ck["ema_shadow"].items()}
        best_val_thresh = ck.get("best_val_thresh", best_val_thresh)
    else:
        print("  WARNING: no top-k checkpoint found; using current model weights.")

    # Temperature scaling [I2]
    temp_scaler = None
    if cfg.use_temperature_scaling:
        print("  Fitting temperature scaler on val set...")
        ema.apply_shadow(model)
        val_logits_list, val_lbls_list = [], []
        with torch.inference_mode():
            for batch in loaders["val"]:
                if batch.get("n", 0) == 0:
                    continue
                wav = batch["wav"].to(DEVICE)
                with autocast(enabled=cfg.amp, device_type=_DEVICE_TYPE):
                    val_logits_list.append(model(wav).cpu().float())
                val_lbls_list.append(batch["labels"].cpu())
        ema.restore(model)
        if val_logits_list:
            vl_np = torch.cat(val_logits_list).numpy()
            vb_np = torch.cat(val_lbls_list).numpy()
            if not _is_single_class(vb_np):
                temp_scaler = TemperatureScaler()
                T_fit = temp_scaler.fit(vl_np, vb_np)
                print(f"  Temperature fitted: T={T_fit:.4f}")

    ext = extended_evaluation(
        model, loaders["test"], cfg,
        ema=ema, val_threshold=best_val_thresh, temp_scaler=temp_scaler,
    )

    # TTA [I1]
    if cfg.use_tta:
        print("\n  Running TTA evaluation...")
        ema.apply_shadow(model)
        tta_m = tta_evaluate(model, loaders["test"], cfg)
        ema.restore(model)
        ext["tta_auc"] = tta_m.get("auc", 0.0)
        print(f"  TTA AUC: {ext['tta_auc']:.4f}")

    if swa_model is not None and cfg.use_swa:
        print("\n  Finalising SWA model...")
        # FIX-5: finalise_swa now returns swa_model (was silently returning None)
        swa_final = finalise_swa(swa_model, loaders["train"], cfg)
        if swa_final is not None:
            swa_ext = extended_evaluation(
                swa_final, loaders["test"], cfg, val_threshold=best_val_thresh
            )
            print(f"  SWA test AUC: {swa_ext['auc']:.4f}")
            ext["swa_auc"] = swa_ext["auc"]

    # ── [7] Greedy model soup [I4] ───────────────────────────
    print("\n[7/7] Greedy model soup...")
    if cfg.use_model_soup and len(ckpt_mgr.heap) > 1:
        soup_model = greedy_soup([p for _, p in ckpt_mgr.heap], loaders["val"], cfg)
        if soup_model is not None:
            soup_ext = extended_evaluation(
                soup_model, loaders["test"], cfg, val_threshold=best_val_thresh
            )
            print(f"  Soup test AUC: {soup_ext['auc']:.4f}")
            ext["soup_auc"] = soup_ext["auc"]
    else:
        print("  Skipped (need ≥2 top-k checkpoints or use_model_soup=False).")

    # ── Final summary ─────────────────────────────────────────
    wall_h = (time.time() - t_start) / 3600.0
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"  Best val AUC   : {best_auc:.4f}")
    print(f"  Test AUC       : {ext.get('auc', 0):.4f}")
    print(f"  Test EER       : {ext.get('eer', 0):.4f}")
    print(f"  Test min-tDCF  : {ext.get('min_tDCF', 'N/A')}")
    if "tta_auc"  in ext: print(f"  TTA  AUC       : {ext['tta_auc']:.4f}")
    if "swa_auc"  in ext: print(f"  SWA  AUC       : {ext['swa_auc']:.4f}")
    if "soup_auc" in ext: print(f"  Soup AUC       : {ext['soup_auc']:.4f}")
    print(f"  Val threshold  : {best_val_thresh:.4f}")
    print(f"  Global steps   : {global_step}")
    print(f"  Wall time      : {wall_h:.2f}h")
    print(f"  Checkpoints    : {cfg.output_dir}")
    print(f"    best_model.pt: {cfg.best_model_path}")
    print(f"    last_model.pt: {cfg.last_model_path}")
    print(f"  Figures        : {cfg.figures_dir}")
    print(f"  Metrics CSV    : {cfg.log_dir}/metrics.csv")

    serialisable = {k: v for k, v in ext.items()
                    if not isinstance(v, (np.ndarray, list))}
    with open(Path(cfg.log_dir) / "test_metrics.json", "w") as f:
        json.dump(_to_python(serialisable), f, indent=2)

    if cfg.use_wandb:
        import wandb
        wandb.log({"test/" + k: v for k, v in serialisable.items()})
        wandb.finish()

    print("\nTraining complete.")
    return model, ext, history

if __name__ == "__main__":


    CFG.amp = True

    CFG.use_rdrop = False

    CFG.max_grad_norm = 1.0

    CFG.pitch_shift_prob    = 0.0
    CFG.phase_scramble_prob = 0.0

    CFG.rir_prob = 0.15

    CFG.num_workers     = 2
    CFG.prefetch_factor = 2
    CFG.pin_memory      = False

    CFG.auto_select_backbone = False
    CFG.eval_every_n_steps = 500


    CFG.hard_session_hours = 11.5
    CFG.time_budget_hours  = 9.5

    CFG.supcon_temperature = 0.10

    CFG.aux_loss_warmup_epochs = 2


    print("\nEffective config:")
    for k, v in sorted(asdict(CFG).items()):
        print(f"  {k:35s}: {v}")
    print()

    model, test_metrics, history = train(CFG)

    print("\n" + "=" * 60)
    print("RUN COMPLETE")
    print("=" * 60)
    print(f"  Checkpoints : {CFG.output_dir}")
    print(f"    best_model.pt : {CFG.best_model_path}")
    print(f"    last_model.pt : {CFG.last_model_path}")
    print(f"  Logs        : {CFG.log_dir}")
    print(f"  Figures     : {CFG.figures_dir}")
    print(f"  Test AUC    : {test_metrics.get('auc', 0):.4f}")
    print(f"  Test EER    : {test_metrics.get('eer', 0):.4f}")
    print(f"  Test min-tDCF: {test_metrics.get('min_tDCF', 'N/A')}")