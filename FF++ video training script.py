# CELL 1
!pip install "numpy<2" "opencv-python-headless>=4.8" "scipy==1.11.4" "scikit-learn==1.4.2" "facenet-pytorch" "decord" --force-reinstall --quiet


# CELL 2
%%writefile train_teacher_v6.py

import sys
import os
import importlib as _imp
import copy
import re

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 0-C — PIL COMPAT PATCH
# ══════════════════════════════════════════════════════════════════════════════
try:
    _pil_util = _imp.import_module("PIL._util")
    if not hasattr(_pil_util, "is_path"):
        _pil_util.is_path = lambda f: isinstance(f, (str, os.PathLike))
        print("[FIX-PIL] Patched PIL._util.is_path", flush=True)
    if not hasattr(_pil_util, "is_directory"):
        _pil_util.is_directory = lambda f: os.path.isdir(f)
        print("[FIX-PIL] Patched PIL._util.is_directory", flush=True)
except Exception as _pil_exc:
    print(f"[FIX-PIL] PIL._util patch skipped (non-fatal): {_pil_exc}", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 0-E — SUPPRESS NOISY LOGGING
# ══════════════════════════════════════════════════════════════════════════════
os.environ["OPENCV_LOG_LEVEL"]       = "FATAL"
os.environ["OPENCV_VIDEOIO_DEBUG"]   = "0"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-1"

import matplotlib
matplotlib.use("Agg")

THRESHOLD_EMA = 0.85

try:
    from torchvision.models.video import swin3d_t, Swin3D_T_Weights
    _TORCHVISION_SWIN_OK = True
    print("[FIX-ARCH-1] torchvision Swin3D_T available.", flush=True)
except (ImportError, Exception) as _tv_exc:
    _TORCHVISION_SWIN_OK = False
    print(f"[FIX-ARCH-1] torchvision Swin3D_T not available ({_tv_exc}). "
          "Will use custom VideoSwinTiny.", flush=True)


CONFIG = dict(
    data_root      = "/kaggle/input/datasets/xdxd003/ff-c23/FaceForensics++_C23",
    output_dir     = "/kaggle/working/ff_video_teacher_v4",
    face_cache_dir = "/kaggle/working/face_cache",

    # ── Dataset folders ────────────────────────────────────────────────────────
    real_folder    = "original",
    fake_folders   = [
        "DeepFakeDetection", "Deepfakes", "Face2Face",
        "FaceShifter", "FaceSwap", "NeuralTextures",
    ],
    csv_dir        = "",

    # ── Clip sampling ──────────────────────────────────────────────────────────
    frames_per_clip       = 16,
    clips_per_video_train = 2,
    clips_per_video_val   = 3,
    clips_per_video_val_epoch = 1,
    frame_stride           = 2,
    input_size            = 224,

    # ── Face detection — ENABLED; FF++ C23 videos are full-frame, not pre-cropped
    face_det_backend = "mtcnn",
    face_margin      = 0.3,
    use_face_cache   = True,     # cache crops to disk so the cost is paid once
    max_cache_gb     = 5.0,

    # ── Training ───────────────────────────────────────────────────────────────
    num_epochs       = 26,
    max_sessions_budget_hours = 34.5,
    target_teacher_auc = 0.90,
    batch_size       = 4,
    num_workers      = min(8, os.cpu_count() or 4),
    learning_rate    = 5e-4,
    weight_decay     = 0.09,
    warmup_epochs    = 1,
    grad_clip        = 5.0,
    label_smoothing  = 0.05,
    grad_accum_steps = 8,

    # ── Focal Loss ─────────────────────────────────────────────────────────────
    focal_gamma = 2.0,
    focal_alpha = [1.02, 1.0],

    # ── Backbone ───────────────────────────────────────────────────────────────
    freeze_backbone       = False,
    stochastic_depth_rate = 0.20,

    # ── Mode-collapse guard ────────────────────────────────────────────────────
    collapse_patience = 3,

    # ── Early stopping ─────────────────────────────────────────────────────────
    early_stop_patience  = 10,
    early_stop_min_delta = 0.002,

    # ── Time budget ────────────────════════════════════════════════════════════
    total_budget_sec = 11.5 * 3600,
    train_budget_sec = 10.5 * 3600,
    eval_reserve_sec =  0.5 * 3600,

    # ── Misc ───────────────────────────────────────────────────────────────────
    seed               = 42,
    amp                = True,
    pin_memory         = True,
    num_classes        = 2,
    dropout            = 0.45,

    pretrained_weights = "kinetics",
    log_interval       = 20,
    use_grad_checkpoint = False,
    use_optimal_threshold = True,

    # ── Model backend ──────────────────────────────────────────────────────────
    model_backend = "torchvision",

    # ── MixUp / CutMix ─────────────────────────────────────────────────────────
    use_mixup         = True,
    mixup_alpha       = 0.4,
    use_cutmix        = False,
    cutmix_alpha      = 1.0,
    mixup_cutmix_prob = 0.15,


    # ── Test-Time Augmentation ─────────────────────────────────────────────────
    use_tta     = True,
    tta_n_views = 6,

    # ── Threshold optimisation metric ──────────────────────────────────────────
    threshold_metric = "balanced_acc",

    aug_any_corruption_prob = 0.45,
    aug_motion_blur_prob    = 0.15,
    aug_noise_prob          = 0.15,
    aug_gamma_prob          = 0.20,
    aug_erasing_prob        = 0.15,
    aug_temporal_rev_prob   = 0.10,
    aug_temporal_shift_prob = 0.10,

    # ── Checkpoint selection metric ────────────────────────────────────────────
    ckpt_metric = "roc_auc",

    # ── Stochastic Weight Averaging (SWA) ──────────────────────────────────────
    use_swa        = True,
    swa_start_frac = 0.60,
    swa_lr         = 2e-5,

    use_hf_aux    = False,
    hf_aux_weight = 0.20,
    hf_aux_warmup_epochs = 5,

    # ── Dynamic EMA decay ─────────────────────────────────────────────────────
    ema_decay_max           = 0.999,
    ema_decay_warmup_epochs = 5,
)


import torch, zipfile, os, sys


def _is_torch_zipfile(path: str) -> bool:
    """Mirrors torch.serialization._is_zipfile()'s actual check: file must
    begin with the zip local-file-header magic bytes at offset 0.
    zipfile.is_zipfile() is looser (only checks the End-Of-Central-Directory
    record near the end) and can pass files torch itself will still reject."""
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"PK\x03\x04"
    except Exception:
        return False


def rebuild_checkpoint_from_extracted_dir(extracted_dir: str, out_path: str) -> str:
    if not os.path.isdir(extracted_dir):
        parent = os.path.dirname(extracted_dir)
        hint = ""
        if os.path.isdir(parent):
            hint = f" Sibling entries in {parent}: {os.listdir(parent)}"
        raise FileNotFoundError(
            f"_EXTRACTED_DIR does not exist: {extracted_dir}\n"
            f"Kaggle Model version path may have changed, or it isn't "
            f"attached to this session.{hint}"
        )

    archive_name = os.path.basename(extracted_dir.rstrip("/"))

    # Locate the true archive root by finding data.pkl, rather than assuming
    # extracted_dir's immediate contents are it.
    source_dir = None
    for root, _, files in os.walk(extracted_dir):
        if "data.pkl" in files:
            source_dir = root
            break
    if source_dir is None:
        raise FileNotFoundError(
            f"No data.pkl found anywhere under {extracted_dir} — this isn't "
            f"extracted PyTorch archive internals."
        )

    # Reuse the cache only if it passes torch's actual zip check.
    if os.path.isfile(out_path):
        if _is_torch_zipfile(out_path):
            print(f"[REBUILD] Valid cached archive at {out_path} — reusing.", flush=True)
            return out_path
        print(f"[REBUILD] Cached file at {out_path} fails torch's zip check "
              f"(stale/corrupted) — rebuilding.", flush=True)
        os.remove(out_path)

    # Write to temp + atomic rename so a crash mid-write never leaves a
    # broken file at out_path for a future run to blindly reuse.
    tmp_path = out_path + f".tmp{os.getpid()}"
    try:
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_STORED) as zf:
            for root, _, files in os.walk(source_dir):
                for fn in files:
                    full = os.path.join(root, fn)
                    rel  = os.path.relpath(full, source_dir)
                    zf.write(full, os.path.join(archive_name, rel))
        if not _is_torch_zipfile(tmp_path):
            raise RuntimeError("Rebuilt archive failed torch zip validation.")
        os.replace(tmp_path, out_path)
    except Exception:
        if os.path.isfile(tmp_path):
            os.remove(tmp_path)
        raise

    print(f"[REBUILD] Rebuilt archive at {out_path} from {source_dir}", flush=True)
    return out_path


# [FRESH START] Not resuming — the input pipeline (face-crop vs. center-crop)
# and split logic have both changed since epoch 19 was produced, so that
# checkpoint's provenance no longer matches this run's data pipeline.
# Training from scratch under the corrected pipeline instead.
RESUME_FROM = None
print("[FRESH START] Skipping checkpoint resume — training from scratch "
      "under the corrected face-crop + identity-aware-split pipeline.")

WEIGHTS_ONLY_INIT_FROM = None

MODE     = "train"
NO_FACE_DET = False   # let CONFIG["face_det_backend"] take effect


import csv
import gc
import json
import logging
import math
import multiprocessing
import multiprocessing.synchronize
import pickle
import random
import time
import warnings
from collections import defaultdict
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.amp import GradScaler, autocast as _autocast
    _AMP_DEVICE = "cuda"
    def autocast(enabled: bool = True):
        return _autocast(_AMP_DEVICE, enabled=enabled)
    print("[FIX-AMP] Using torch.amp (new API).", flush=True)
except ImportError:
    from torch.cuda.amp import GradScaler, autocast  # type: ignore[assignment]
    _AMP_DEVICE = None
    print("[FIX-AMP] Using torch.cuda.amp (legacy API).", flush=True)

from torch.utils.checkpoint import checkpoint as gradient_checkpoint
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    import torch.utils.serialization  # noqa: F401 — real submodule on torch>=2.6, provides `config`
except ImportError:

    if "torch.utils.serialization" not in sys.modules:
        sys.modules["torch.utils.serialization"] = torch  # type: ignore[assignment]

try:
    from decord import VideoReader, cpu as decord_cpu
    DECORD_AVAILABLE = True
    print(f"[SPEED] decord available (NumPy {np.__version__}).", flush=True)
except ImportError:
    DECORD_AVAILABLE = False
    VideoReader = None   # type: ignore[assignment,misc]
    decord_cpu  = None   # type: ignore[assignment]
    print("[SPEED] decord not installed — OpenCV fallback.", flush=True)

try:
    from tqdm.auto import tqdm
    TQDM_OK = True
except ImportError:
    TQDM_OK = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False

try:
    from sklearn.metrics import (
        accuracy_score, average_precision_score, balanced_accuracy_score,
        confusion_matrix, f1_score, matthews_corrcoef, precision_score,
        recall_score, roc_auc_score, roc_curve, precision_recall_curve,
    )
    from sklearn.model_selection import train_test_split
    SKLEARN_OK = True
except Exception as _sk_exc:
    print(f"[WARN] sklearn import failed: {_sk_exc}", flush=True)
    SKLEARN_OK = False
    train_test_split = None   # type: ignore[assignment]


# ══════════════════════════════════════════════════════════════════════════════
# 1.  UTILITIES  (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════

@contextmanager
def suppress_c_stderr():
    try:
        null_fd = os.open(os.devnull, os.O_WRONLY)
        old_fd  = os.dup(2)
        os.dup2(null_fd, 2)
        yield
    finally:
        os.dup2(old_fd, 2)
        os.close(old_fd)
        os.close(null_fd)


def setup_logging(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "training_log.txt")
    fmt  = "%(asctime)s [%(levelname)s] %(message)s"
    root = logging.getLogger()
    root.handlers.clear()
    logging.basicConfig(
        level    = logging.INFO,
        format   = fmt,
        handlers = [
            logging.FileHandler(log_path, mode="a"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("ff_teacher")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark     = False
    torch.backends.cudnn.deterministic = True


def log_system_info(logger: logging.Logger) -> None:
    logger.info("=" * 70)
    logger.info("SYSTEM INFORMATION")
    logger.info(f"  Python  : {sys.version.split()[0]}")
    logger.info(f"  PyTorch : {torch.__version__}")
    logger.info(f"  CUDA    : {torch.version.cuda}")
    logger.info(f"  numpy   : {np.__version__}")
    logger.info(f"  Decord  : {'available' if DECORD_AVAILABLE else 'not installed — OpenCV fallback'}")
    logger.info(f"  sklearn : {'available' if SKLEARN_OK else 'unavailable'}")
    logger.info(f"  tv_swin : {'available' if _TORCHVISION_SWIN_OK else 'unavailable — custom fallback'}")
    if torch.cuda.is_available():
        logger.info(f"  cuDNN   : {torch.backends.cudnn.version()}")
        try:
            logger.info(f"  GPUs    : {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"    GPU {i}: {props.name}  {props.total_memory // (1024**3)} GB")
        except Exception as e:
            logger.warning(f"  Could not query GPUs: {e}")
    else:
        logger.info("  No GPU available — running on CPU.")
    logger.info("=" * 70)


def save_config(cfg: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2, default=str)


class Timer:
    def __init__(self) -> None:
        self._start = time.time()

    def elapsed(self) -> float:
        return time.time() - self._start

    def remaining(self, budget: float) -> float:
        return max(0.0, budget - self.elapsed())

    def eta_str(self, budget: float) -> str:
        return str(timedelta(seconds=int(self.remaining(budget))))


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.avg = self.sum = self.count = 0.0

    def update(self, val: float, n: int = 1) -> None:
        self.sum   += float(val) * n
        self.count += n
        self.avg    = self.sum / max(self.count, 1)


def get_head_lr(optimizer: torch.optim.Optimizer) -> float:
    for g in optimizer.param_groups:
        if g.get("name") in ("head_and_norm", "norm_and_head"):
            return g["lr"]
    for g in optimizer.param_groups:
        if g.get("name") != "hf_head":
            return g["lr"]
    return optimizer.param_groups[-1]["lr"]


# ══════════════════════════════════════════════════════════════════════════════
# 1b.  DEBUG VISUALISATION  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

_DBG_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_DBG_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def save_debug_batch(
    clips:      torch.Tensor,
    labels:     torch.Tensor,
    output_dir: str,
    filename:   str = "debug_batch_0.png",
) -> Optional[str]:
    if not MATPLOTLIB_OK:
        return None
    try:
        clips_cpu  = clips.detach().cpu().float()
        labels_cpu = labels.detach().cpu()
        B     = min(clips_cpu.shape[0], 4)
        T     = clips_cpu.shape[2]
        t_idx = [min(int(T * f / 4), T - 1) for f in range(5)]
        fig, axes = plt.subplots(
            B, len(t_idx), figsize=(len(t_idx) * 2.8, B * 2.8), facecolor="white",
        )
        if B == 1:
            axes = axes[np.newaxis, :]
        if len(t_idx) == 1:
            axes = axes[:, np.newaxis]
        for b in range(B):
            lbl_str = "Fake" if labels_cpu[b].item() == 1 else "Real"
            for col, t in enumerate(t_idx):
                frame = clips_cpu[b, :, t, :, :]
                frame = (frame * _DBG_STD + _DBG_MEAN).clamp(0.0, 1.0).permute(1, 2, 0).numpy()
                ax = axes[b, col]
                ax.imshow(frame)
                ax.axis("off")
                ax.set_title(f"B{b} [{lbl_str}]" if col == 0 else f"t={t}", fontsize=8, pad=2)
        fig.suptitle(
            "DEBUG — First training batch. Check: face visible? augments sane?",
            fontsize=9,
        )
        fig.tight_layout()
        save_path = os.path.join(output_dir, filename)
        fig.savefig(save_path, dpi=120, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        plt.close(fig)
        return save_path
    except Exception as exc:
        logging.getLogger("ff_teacher").warning(f"save_debug_batch failed: {exc}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 2.  VIDEO SWIN TRANSFORMER TINY  (unchanged — see FIX-6 in optimizer builders
#     below for the actual behavioural change: same architecture, softer LLRD)
# ══════════════════════════════════════════════════════════════════════════════

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob     = 1.0 - self.drop_prob
        shape         = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.floor(
            torch.rand(shape, dtype=x.dtype, device=x.device) + keep_prob
        )
        return x.div(keep_prob) * random_tensor


def window_partition_3d(x: torch.Tensor, ws: Tuple[int, int, int]) -> torch.Tensor:
    B, D, H, W, C = x.shape
    wd, wh, ww     = ws
    x = x.view(B, D // wd, wd, H // wh, wh, W // ww, ww, C)
    return x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, wd * wh * ww, C)


def window_reverse_3d(
    windows: torch.Tensor, ws: Tuple[int, int, int], D: int, H: int, W: int
) -> torch.Tensor:
    wd, wh, ww = ws
    B = int(windows.shape[0] / (D * H * W // wd // wh // ww))
    x = windows.view(B, D // wd, H // wh, W // ww, wd, wh, ww, -1)
    return x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)


def build_relative_position_index(ws: Tuple[int, int, int]) -> torch.Tensor:
    wd, wh, ww  = ws
    coords      = torch.stack(
        torch.meshgrid(
            torch.arange(wd), torch.arange(wh), torch.arange(ww), indexing="ij"
        )
    )
    coords_flat = coords.flatten(1)
    rel         = coords_flat[:, :, None] - coords_flat[:, None, :]
    rel         = rel.permute(1, 2, 0).contiguous()
    rel[:, :, 0] += wd - 1
    rel[:, :, 1] += wh - 1
    rel[:, :, 2] += ww - 1
    rel[:, :, 0] *= (2 * wh - 1) * (2 * ww - 1)
    rel[:, :, 1] *= (2 * ww - 1)
    return rel.sum(-1)


class WindowAttention3D(nn.Module):
    def __init__(
        self, dim: int, ws: Tuple, num_heads: int,
        qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.ws        = ws
        self.num_heads = num_heads
        self.scale     = (dim // num_heads) ** -0.5
        tbl_size       = (2 * ws[0] - 1) * (2 * ws[1] - 1) * (2 * ws[2] - 1)
        self.rpb       = nn.Parameter(torch.zeros(tbl_size, num_heads))
        nn.init.trunc_normal_(self.rpb, std=0.02)
        idx = build_relative_position_index(ws)
        self.register_buffer("rpi", idx)
        self.qkv       = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B_, N, C = x.shape
        H        = self.num_heads
        qkv  = self.qkv(x).reshape(B_, N, 3, H, C // H).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn    = (q * self.scale) @ k.transpose(-2, -1)
        win_n   = self.ws[0] * self.ws[1] * self.ws[2]
        bias    = self.rpb[self.rpi[:win_n, :win_n].reshape(-1)]
        bias    = bias.reshape(win_n, win_n, H).permute(2, 0, 1).contiguous()
        attn    = attn + bias.unsqueeze(0)
        if mask is not None:
            nW   = mask.shape[0]
            attn = attn.view(B_ // nW, nW, H, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, H, N, N)
        attn = self.attn_drop(attn.softmax(dim=-1))
        x    = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


class SwinBlock3D(nn.Module):
    def __init__(
        self, dim: int, num_heads: int, ws: Tuple, shift: Tuple,
        mlp_ratio: float = 4.0, qkv_bias: bool = True,
        drop: float = 0.0, attn_drop: float = 0.0, drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.ws    = ws
        self.shift = shift
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = WindowAttention3D(dim, ws, num_heads, qkv_bias, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        hidden     = int(dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, dim), nn.Dropout(drop),
        )

    def _make_mask(self, D: int, H: int, W: int, device: torch.device) -> torch.Tensor:
        img = torch.zeros(1, D, H, W, 1, device=device)
        cnt = 0
        slices_d = (slice(0, -self.ws[0]), slice(-self.ws[0], -self.shift[0]), slice(-self.shift[0], None))
        slices_h = (slice(0, -self.ws[1]), slice(-self.ws[1], -self.shift[1]), slice(-self.shift[1], None))
        slices_w = (slice(0, -self.ws[2]), slice(-self.ws[2], -self.shift[2]), slice(-self.shift[2], None))
        for d in slices_d:
            for h in slices_h:
                for w in slices_w:
                    img[:, d, h, w, :] = cnt
                    cnt += 1
        wins = window_partition_3d(img, self.ws).view(-1, self.ws[0] * self.ws[1] * self.ws[2])
        mask = wins.unsqueeze(1) - wins.unsqueeze(2)
        return mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)

    def forward(self, x: torch.Tensor, D: int, H: int, W: int) -> torch.Tensor:
        B, _, C  = x.shape
        shortcut = x
        x        = self.norm1(x).view(B, D, H, W, C)
        pd = (self.ws[0] - D % self.ws[0]) % self.ws[0]
        ph = (self.ws[1] - H % self.ws[1]) % self.ws[1]
        pw = (self.ws[2] - W % self.ws[2]) % self.ws[2]
        if pd or ph or pw:
            x = F.pad(x, (0, 0, 0, pw, 0, ph, 0, pd))
        _, Dp, Hp, Wp, _ = x.shape
        mask = None
        if any(s > 0 for s in self.shift):
            x    = torch.roll(x, (-self.shift[0], -self.shift[1], -self.shift[2]), (1, 2, 3))
            mask = self._make_mask(Dp, Hp, Wp, x.device)
        wins = window_partition_3d(x, self.ws)
        wins = self.attn(wins, mask)
        x    = window_reverse_3d(wins, self.ws, Dp, Hp, Wp)
        if any(s > 0 for s in self.shift):
            x = torch.roll(x, (self.shift[0], self.shift[1], self.shift[2]), (1, 2, 3))
        if pd or ph or pw:
            x = x[:, :D, :H, :W, :].contiguous()
        x = x.view(B, D * H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging3D(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm      = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(
        self, x: torch.Tensor, D: int, H: int, W: int
    ) -> Tuple[torch.Tensor, int, int, int]:
        B, _, C = x.shape
        x = x.view(B, D, H, W, C)
        x = torch.cat([x[:, :, 0::2, 0::2, :], x[:, :, 1::2, 0::2, :],
                        x[:, :, 0::2, 1::2, :], x[:, :, 1::2, 1::2, :]], -1)
        return self.reduction(self.norm(x.view(B, -1, 4 * C))), D, H // 2, W // 2


class PatchEmbed3D(nn.Module):
    def __init__(self, embed_dim: int = 96) -> None:
        super().__init__()
        self.proj = nn.Conv3d(3, embed_dim, kernel_size=(2, 4, 4), stride=(2, 4, 4))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int, int]:
        x = self.proj(x)
        B, E, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x), D, H, W


class SwinStage3D(nn.Module):
    def __init__(
        self, dim: int, depth: int, num_heads: int,
        ws: Tuple, drop_path_rates: List[float],
        mlp_ratio: float = 4.0, drop: float = 0.0,
        attn_drop: float = 0.0, downsample: bool = False,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.use_checkpoint = use_checkpoint
        shift = tuple(s // 2 for s in ws)
        self.blocks = nn.ModuleList([
            SwinBlock3D(
                dim=dim, num_heads=num_heads, ws=ws,
                shift=(0, 0, 0) if i % 2 == 0 else shift,
                mlp_ratio=mlp_ratio, drop=drop, attn_drop=attn_drop,
                drop_path=drop_path_rates[i],
            )
            for i in range(depth)
        ])
        self.downsample = PatchMerging3D(dim) if downsample else None

    def forward(
        self, x: torch.Tensor, D: int, H: int, W: int
    ) -> Tuple[torch.Tensor, int, int, int]:
        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                x = gradient_checkpoint(blk, x, D, H, W, use_reentrant=False)
            else:
                x = blk(x, D, H, W)
        if self.downsample is not None:
            x, D, H, W = self.downsample(x, D, H, W)
        return x, D, H, W


class VideoSwinTiny(nn.Module):
    def __init__(
        self,
        num_classes:          int   = 2,
        dropout:              float = 0.35,
        use_grad_checkpoint:  bool  = False,
        stochastic_depth_rate: float = 0.20,
    ) -> None:
        super().__init__()
        E      = 96
        depths = [2, 2, 6, 2]
        heads  = [3, 6, 12, 24]
        ws     = (2, 7, 7)
        dpr    = [x.item() for x in torch.linspace(0, stochastic_depth_rate, sum(depths))]

        self.patch_embed = PatchEmbed3D(embed_dim=E)
        self.pos_drop    = nn.Dropout(0.0)

        self.stages = nn.ModuleList()
        ptr = 0
        for i, (d, h) in enumerate(zip(depths, heads)):
            stage = SwinStage3D(
                dim             = int(E * 2 ** i),
                depth           = d,
                num_heads       = h,
                ws              = ws,
                drop_path_rates = dpr[ptr: ptr + d],
                downsample      = (i < len(depths) - 1),
                use_checkpoint  = use_grad_checkpoint,
            )
            self.stages.append(stage)
            ptr += d

        feat_dim  = int(E * 2 ** (len(depths) - 1))
        self.norm = nn.LayerNorm(feat_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout * 0.4),
            nn.Linear(256, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x, D, H, W = self.patch_embed(x)
        x = self.pos_drop(x)
        for stage in self.stages:
            x, D, H, W = stage(x, D, H, W)
        x = self.norm(x)
        return x.mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)


def load_pretrained_kinetics(model: VideoSwinTiny, logger: logging.Logger) -> bool:
    ckpt_path = "/tmp/swin_tiny_kinetics400.pth"
    if not os.path.exists(ckpt_path):
        url = ("https://github.com/SwinTransformer/storage/releases/download/"
               "v1.0.4/swin_tiny_patch244_window877_kinetics400_1k.pth")
        try:
            import urllib.request
            logger.info(f"Downloading Kinetics weights from:\n  {url}")
            urllib.request.urlretrieve(url, ckpt_path)
        except Exception as exc:
            logger.warning(f"Download failed: {exc}. Using random init.")
            return False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt.get("model", ckpt))
        state = {k.replace("backbone.", ""): v for k, v in state.items()}
        state = {k: v for k, v in state.items()
                 if not k.startswith("head") and not k.startswith("cls")}
        miss, unexp = model.load_state_dict(state, strict=False)
        logger.info(f"Pretrained weights loaded — missing: {len(miss)}, unexpected: {len(unexp)}")
        return True
    except Exception as exc:
        logger.warning(f"Failed to parse pretrained weights: {exc}. Using random init.")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# 2b.  LAPLACIAN HIGH-FREQUENCY AUXILIARY HEAD  (unchanged; disabled by default
#      via CONFIG.use_hf_aux — see FIX-5 above)
# ══════════════════════════════════════════════════════════════════════════════

class LaplacianHFHead(nn.Module):
    def __init__(
        self,
        num_classes: int   = 2,
        dropout:     float = 0.30,
    ) -> None:
        super().__init__()
        lap = torch.tensor(
            [[0.0, -1.0, 0.0],
             [-1.0, 4.0, -1.0],
             [0.0, -1.0, 0.0]], dtype=torch.float32
        )
        self.register_buffer("laplacian", lap.view(1, 1, 3, 3))

        self.features = nn.Sequential(
            nn.Conv2d(3,  16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64 * 16, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _extract_hf(self, frame: torch.Tensor) -> torch.Tensor:
        B, C, H, W = frame.shape
        hf = F.conv2d(frame.view(B * C, 1, H, W), self.laplacian, padding=1)
        return hf.view(B, C, H, W).clamp(-1.0, 1.0)

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = clips.shape
        frame_idx = [max(0, min(T - 1, int(T * f))) for f in (0.25, 0.50, 0.75)]

        feat_list: List[torch.Tensor] = []
        for t in frame_idx:
            frame = clips[:, :, t, :, :]
            hf    = self._extract_hf(frame)
            feat  = self.features(hf)
            feat_list.append(feat)

        feat = torch.stack(feat_list, dim=1).mean(dim=1)
        feat = feat.view(B, -1)
        return self.classifier(feat)


# ══════════════════════════════════════════════════════════════════════════════
# 2c.  COMBINED MODEL WRAPPER  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

class DeepfakeDetector(nn.Module):
    def __init__(
        self,
        swin_model: nn.Module,
        hf_head:    Optional[nn.Module] = None,
        hf_weight:  float               = 0.20,
    ) -> None:
        super().__init__()
        self.swin      = swin_model
        self.hf_head   = hf_head
        self.hf_weight = hf_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main_logits = self.swin(x)
        if self.hf_head is not None:
            hf_logits = self.hf_head(x)
            return (1.0 - self.hf_weight) * main_logits + self.hf_weight * hf_logits
        return main_logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.swin, "get_features"):
            return self.swin.get_features(x)
        if hasattr(self.swin, "forward_features"):
            return self.swin.forward_features(x)
        raise AttributeError("Wrapped model has no get_features / forward_features method.")


# ══════════════════════════════════════════════════════════════════════════════
# 2d.  LAYER-WISE LEARNING RATE DECAY (LLRD)  — [FIX-6] softened ratios
# ══════════════════════════════════════════════════════════════════════════════

def build_optimizer_with_llrd(
    detector: DeepfakeDetector,
    cfg:      dict,
    logger:   logging.Logger,
) -> torch.optim.AdamW:
    base_lr = cfg["learning_rate"]
    model   = detector.swin

    param_groups = [
        {"params": list(model.patch_embed.parameters()),  "lr": base_lr * 0.10, "name": "patch_embed"},
        {"params": list(model.stages[0].parameters()),    "lr": base_lr * 0.18, "name": "stages_0"},
        {"params": list(model.stages[1].parameters()),    "lr": base_lr * 0.30, "name": "stages_1"},
        {"params": list(model.stages[2].parameters()),    "lr": base_lr * 0.50, "name": "stages_2"},
        {"params": list(model.stages[3].parameters()),    "lr": base_lr * 0.75, "name": "stages_3"},
        {
            "params": list(model.norm.parameters()) + list(model.head.parameters()),
            "lr": base_lr, "name": "norm_and_head",
        },
    ]
    if detector.hf_head is not None:
        param_groups.append({
            "params": list(detector.hf_head.parameters()),
            "lr": base_lr * 0.50, "name": "hf_head",
        })

    logger.info("Layer-wise learning rate decay (custom VideoSwinTiny) [FIX-6 softened ratios]:")
    for g in param_groups:
        n = sum(p.numel() for p in g["params"])
        logger.info(f"  {g['name']:20s}  lr={g['lr']:.1e}  params={n:,}")
    return torch.optim.AdamW(param_groups, weight_decay=cfg["weight_decay"])


def build_optimizer_torchvision(
    detector: DeepfakeDetector,
    cfg:      dict,
    logger:   logging.Logger,
) -> torch.optim.AdamW:
    base_lr = cfg["learning_rate"]
    model   = detector.swin

    head_params, late_params, mid_params, early_params = [], [], [], []
    for name, param in model.named_parameters():
        if "head" in name or ("norm" in name and "features" not in name):
            head_params.append(param)
        elif "features.6" in name or "features.7" in name:
            late_params.append(param)
        elif "features.3" in name or "features.4" in name or "features.5" in name:
            mid_params.append(param)
        else:
            early_params.append(param)

    param_groups = [
        {"params": early_params, "lr": base_lr * 0.12, "name": "early_layers"},
        {"params": mid_params,   "lr": base_lr * 0.35, "name": "mid_layers"},
        {"params": late_params,  "lr": base_lr * 0.70, "name": "late_layers"},
        {"params": head_params,  "lr": base_lr,        "name": "head_and_norm"},
    ]
    param_groups = [g for g in param_groups if len(g["params"]) > 0]

    if detector.hf_head is not None:
        param_groups.append({
            "params": list(detector.hf_head.parameters()),
            "lr": base_lr * 0.50, "name": "hf_head",
        })

    logger.info("torchvision Swin3D_T LLRD optimizer groups [FIX-6 softened ratios]:")
    for g in param_groups:
        n = sum(p.numel() for p in g["params"])
        logger.info(f"  {g['name']:20s}  lr={g['lr']:.1e}  params={n:,}")
    return torch.optim.AdamW(param_groups, weight_decay=cfg["weight_decay"])


# ══════════════════════════════════════════════════════════════════════════════
# 3.  FACE DETECTION + CACHING  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

class FaceDetector:
    def __init__(self, backend: str = "mtcnn", logger: logging.Logger = None) -> None:
        self.logger  = logger or logging.getLogger("ff_teacher")
        self.backend = backend
        self._det    = None
        self._init()

    def _init(self) -> None:
        if self.backend == "mtcnn":
            try:
                from facenet_pytorch import MTCNN
                self._det = MTCNN(keep_all=True, device="cpu")
                self.logger.info("FaceDetector: MTCNN on CPU")
                return
            except Exception as e:
                self.logger.warning(f"MTCNN unavailable ({e}) — falling back to Haar.")
                self.backend = "haar"
        haar = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._det    = haar
        self.backend = "haar"
        self.logger.info("FaceDetector: OpenCV Haar (fallback)")

    def detect(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        try:
            if self.backend == "mtcnn":
                from PIL import Image as PILImage
                img      = PILImage.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                boxes, _ = self._det.detect(img)
                if boxes is None:
                    return []
                return [(int(b[0]), int(b[1]), int(b[2]), int(b[3])) for b in boxes]
            else:
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                dets = self._det.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
                return [(x, y, x + w, y + h) for (x, y, w, h) in dets] if len(dets) else []
        except Exception:
            return []


def crop_face_clip(
    frames_bgr: List[np.ndarray],
    detector:   FaceDetector,
    margin:     float = 0.3,
    size:       int   = 224,
) -> Optional[List[np.ndarray]]:
    H, W = frames_bgr[0].shape[:2]
    T    = len(frames_bgr)
    probe_indices = sorted(set([T // 2, 0, T - 1, T // 4, 3 * T // 4]))
    best_box = None
    for idx in probe_indices:
        idx   = min(max(idx, 0), T - 1)
        boxes = detector.detect(frames_bgr[idx])
        if boxes:
            best_box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            break
    if best_box is None:
        return None
    x1, y1, x2, y2 = best_box
    cx   = (x1 + x2) / 2.0
    cy   = (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * (1.0 + margin)
    rx1  = max(0, int(cx - side / 2))
    ry1  = max(0, int(cy - side / 2))
    rx2  = min(W, int(cx + side / 2))
    ry2  = min(H, int(cy + side / 2))
    crops: List[np.ndarray] = []
    for f in frames_bgr:
        c = f[ry1:ry2, rx1:rx2]
        if c.size == 0:
            c = f
        c = cv2.resize(c, (size, size), interpolation=cv2.INTER_CUBIC)
        crops.append(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
    return crops


def _center_crop_frames(frames_bgr: List[np.ndarray], size: int) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for f in frames_bgr:
        H, W = f.shape[:2]
        side = min(H, W)
        y1   = (H - side) // 2
        x1   = (W - side) // 2
        c    = f[y1: y1 + side, x1: x1 + side]
        c    = cv2.resize(c, (size, size), interpolation=cv2.INTER_CUBIC)
        out.append(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
    return out


def _cache_key(video_path: str, num_frames: int, frame_stride: int, size: int) -> str:
    import hashlib
    tag = f"{video_path}|{num_frames}|{frame_stride}|{size}"
    return hashlib.md5(tag.encode()).hexdigest()


def load_from_cache(cache_dir: str, key: str) -> Optional[List[np.ndarray]]:
    p = os.path.join(cache_dir, key[:2], f"{key}.pkl")
    if os.path.isfile(p):
        try:
            with open(p, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None


def save_to_cache(cache_dir: str, key: str, frames_rgb: List[np.ndarray]) -> None:
    try:
        stat = os.statvfs(cache_dir)
        free_gb = stat.f_bavail * stat.f_frsize / (1024 ** 3)
        if free_gb < 2.0:
            return
        subdir = os.path.join(cache_dir, key[:2])
        os.makedirs(subdir, exist_ok=True)
        p = os.path.join(subdir, f"{key}.pkl")
        with open(p, "wb") as f:
            pickle.dump(frames_rgb, f, protocol=4)
    except Exception:
        pass


_worker_detector = None


def _init_worker(face_backend: str) -> None:
    global _worker_detector
    try:
        null_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(null_fd, 2)
    except Exception:
        pass
    if face_backend != "none":
        _worker_detector = FaceDetector(face_backend)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  VIDEO READING  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

_video_logger    = logging.getLogger("ff_teacher.video")
_MIN_FRAME_RATIO = 0.80


def _deterministic_start(total: int, span: int, window_idx: int, num_windows: int) -> int:

    usable = max(0, total - span)
    if num_windows <= 1:
        return usable // 2
    frac = (window_idx + 1) / (num_windows + 1)
    return int(round(usable * frac))


def read_video_frames(
    path:          str,
    num_frames:    int,
    stride:        int  = 2,
    deterministic: bool = False,
    window_idx:    int  = 0,
    num_windows:   int  = 1,
) -> Optional[List[np.ndarray]]:
    span = num_frames * stride

    with suppress_c_stderr():
        if DECORD_AVAILABLE:
            try:
                vr    = VideoReader(str(path), ctx=decord_cpu(0))
                total = len(vr)
                if total <= 0:
                    raise ValueError("Decord reports 0 frames")
                if total >= span:
                    if deterministic:
                        start = _deterministic_start(total, span, window_idx, num_windows)
                    else:
                        start = random.randint(0, total - span)
                    indices = list(range(start, start + span, stride))
                else:
                    indices = np.linspace(0, total - 1, num_frames, dtype=int).tolist()
                frames_rgb = vr.get_batch(indices).asnumpy()
                if len(frames_rgb) < int(_MIN_FRAME_RATIO * num_frames):
                    raise ValueError(f"Decord returned only {len(frames_rgb)}/{num_frames}")
                frames_bgr = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in frames_rgb]
                while len(frames_bgr) < num_frames:
                    frames_bgr.append(frames_bgr[-1].copy())
                return frames_bgr[:num_frames]
            except Exception as exc:
                _video_logger.warning(f"Decord failed for '{path}': {exc}")

        try:
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                _video_logger.warning(f"OpenCV cannot open '{path}'")
                return None
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                cap.release()
                return None
            if total >= span:
                if deterministic:
                    start = _deterministic_start(total, span, window_idx, num_windows)
                else:
                    start = random.randint(0, total - span)
                indices = list(range(start, start + span, stride))
            else:
                indices = np.linspace(0, total - 1, num_frames, dtype=int).tolist()
            sorted_indices = sorted(set(int(i) for i in indices))
            wanted         = set(sorted_indices)
            max_idx        = sorted_indices[-1]
            collected: Dict[int, np.ndarray] = {}
            frame_id = 0
            while frame_id <= max_idx:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_id in wanted:
                    collected[frame_id] = frame
                frame_id += 1
            cap.release()
            if not collected:
                return None
            if len(collected) < int(_MIN_FRAME_RATIO * len(indices)):
                return None
            all_keys   = sorted(collected.keys())
            frames_bgr = []
            for idx in indices:
                idx = int(idx)
                if idx in collected:
                    frames_bgr.append(collected[idx])
                else:
                    nearest = min(all_keys, key=lambda k: abs(k - idx))
                    frames_bgr.append(collected[nearest].copy())
            while len(frames_bgr) < num_frames:
                frames_bgr.append(frames_bgr[-1].copy())
            return frames_bgr[:num_frames]
        except Exception as exc:
            _video_logger.warning(f"OpenCV sequential decode failed for '{path}': {exc}")
            return None


# ══════════════════════════════════════════════════════════════════════════════
# 5.  TRANSFORMS  — [FIX-4] mutually-exclusive per-frame corruption augment
# ══════════════════════════════════════════════════════════════════════════════

_TTA_SCALES = [1.00, 1.00, 0.90, 0.90, 0.85, 0.85]
_TTA_FLIPS  = [False, True, False, True, False, True]


class ClipTransform:
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(
        self,
        mode:     str = "train",
        size:     int = 224,
        cfg:      Optional[dict] = None,
        tta_view: int = 0,
    ) -> None:
        self.mode     = mode
        self.size     = size
        self.cfg      = cfg or {}
        self.tta_view = tta_view
        self._mean    = torch.tensor(self.MEAN, dtype=torch.float32).view(3, 1, 1)
        self._std     = torch.tensor(self.STD,  dtype=torch.float32).view(3, 1, 1)

    def _normalize(self, img: torch.Tensor) -> torch.Tensor:
        mean = self._mean.to(img.device)
        std  = self._std.to(img.device)
        return (img - mean) / std

    @staticmethod
    def _resize(img: torch.Tensor, size: int) -> torch.Tensor:
        x = img.unsqueeze(0)
        try:
            return F.interpolate(x, (size, size), mode="bilinear",
                                 align_corners=False, antialias=True).squeeze(0)
        except TypeError:
            return F.interpolate(x, (size, size), mode="bilinear",
                                 align_corners=False).squeeze(0)

    @staticmethod
    def _hflip(img: torch.Tensor) -> torch.Tensor:
        return img.flip(-1)

    @staticmethod
    def _adjust_brightness(img: torch.Tensor, factor: float) -> torch.Tensor:
        return (img * factor).clamp(0.0, 1.0)

    @staticmethod
    def _adjust_contrast(img: torch.Tensor, factor: float) -> torch.Tensor:
        mean = img.mean(dim=(-2, -1), keepdim=True)
        return ((img - mean) * factor + mean).clamp(0.0, 1.0)

    @staticmethod
    def _adjust_saturation(img: torch.Tensor, factor: float) -> torch.Tensor:
        gray = (0.2989 * img[0:1] + 0.5870 * img[1:2] + 0.1140 * img[2:3])
        return ((img - gray) * factor + gray).clamp(0.0, 1.0)

    @staticmethod
    def _gaussian_blur(img: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
        half   = kernel_size // 2
        coords = torch.arange(kernel_size, dtype=torch.float32, device=img.device) - half
        g      = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
        g      = g / g.sum()
        C      = img.shape[0]
        k_row  = g.view(1, 1, 1, kernel_size).expand(C, 1, 1, kernel_size).contiguous()
        k_col  = g.view(1, 1, kernel_size, 1).expand(C, 1, kernel_size, 1).contiguous()
        x      = img.unsqueeze(0)
        x      = F.conv2d(x, k_row, padding=(0, half),  groups=C)
        x      = F.conv2d(x, k_col, padding=(half, 0),  groups=C)
        return x.squeeze(0)

    @staticmethod
    def _motion_blur(img: torch.Tensor, kernel_size: int = 9, angle_deg: float = 0.0) -> torch.Tensor:
        try:
            C = img.shape[0]
            k = torch.zeros(kernel_size, kernel_size, dtype=torch.float32, device=img.device)
            k[kernel_size // 2, :] = 1.0 / kernel_size
            theta = math.radians(angle_deg)
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            grid_k = torch.zeros(1, 2, 3, device=img.device)
            grid_k[0, 0, 0] =  cos_t; grid_k[0, 0, 1] = sin_t
            grid_k[0, 1, 0] = -sin_t; grid_k[0, 1, 1] = cos_t
            k_4d  = k.unsqueeze(0).unsqueeze(0)
            grid  = F.affine_grid(grid_k, k_4d.size(), align_corners=False)
            k_rot = F.grid_sample(k_4d, grid, align_corners=False)
            k_rot = k_rot.squeeze(0).expand(C, 1, kernel_size, kernel_size).contiguous()
            pad   = kernel_size // 2
            x     = F.conv2d(img.unsqueeze(0), k_rot, padding=pad, groups=C)
            return x.squeeze(0).clamp(0.0, 1.0)
        except Exception:
            return img

    @staticmethod
    def _gaussian_noise(img: torch.Tensor, std: float = 0.02) -> torch.Tensor:
        noise = torch.randn_like(img) * std
        return (img + noise).clamp(0.0, 1.0)

    @staticmethod
    def _random_gamma(img: torch.Tensor, gamma: float) -> torch.Tensor:
        return img.clamp(1e-8, 1.0).pow(gamma)

    @staticmethod
    def _random_erasing(
        img: torch.Tensor, sl: float = 0.02, sh: float = 0.15, r1: float = 0.3,
    ) -> torch.Tensor:
        try:
            C, H, W = img.shape
            area    = H * W
            for _ in range(10):
                target_area = random.uniform(sl, sh) * area
                aspect      = random.uniform(r1, 1.0 / r1)
                h = int(round(math.sqrt(target_area * aspect)))
                w = int(round(math.sqrt(target_area / aspect)))
                if h < H and w < W:
                    y = random.randint(0, H - h)
                    x = random.randint(0, W - w)
                    img = img.clone()
                    img[:, y:y+h, x:x+w] = img.mean(dim=(1, 2), keepdim=True).expand(C, h, w)
                    break
            return img
        except Exception:
            return img

    @staticmethod
    def _jpeg_sim(img: torch.Tensor) -> torch.Tensor:
        try:
            arr     = (img.permute(1, 2, 0).detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            q       = random.randint(40, 90)
            _, buf  = cv2.imencode(".jpg", arr_bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
            out     = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            return torch.from_numpy(out_rgb).permute(2, 0, 1).contiguous()
        except Exception:
            return img

    def _apply_freq_mask(self, img: torch.Tensor) -> torch.Tensor:
        try:
            fft  = torch.fft.rfft2(img)
            mask = torch.ones_like(fft, dtype=torch.float32)
            h_f, w_f = fft.shape[-2], fft.shape[-1]
            bw   = random.randint(1, max(1, h_f // 8))
            r0   = random.randint(0, h_f - bw)
            mask[..., r0:r0 + bw, :] = 0.0
            return torch.fft.irfft2(fft * mask, s=(self.size, self.size)).clamp(0.0, 1.0)
        except Exception:
            return img

    def __call__(self, frames: List[np.ndarray]) -> torch.Tensor:
        H, W = frames[0].shape[:2]
        T    = len(frames)

        if self.mode == "train":
            do_flip     = random.random() < 0.50
            do_any_corrupt = random.random() < self.cfg.get("aug_any_corruption_prob", 0.45)
            corrupt_choice = None
            if do_any_corrupt:
                ops     = ["jpeg", "blur", "motion", "noise", "gamma", "erasing", "freq"]
                weights = [
                    0.20,  # jpeg
                    0.05,  # blur
                    self.cfg.get("aug_motion_blur_prob",   0.15),
                    self.cfg.get("aug_noise_prob",          0.15),
                    self.cfg.get("aug_gamma_prob",          0.20),
                    self.cfg.get("aug_erasing_prob",        0.15),
                    0.15,  # freq mask
                ]
                corrupt_choice = random.choices(ops, weights=weights, k=1)[0]

            do_temp_rev = random.random() < self.cfg.get("aug_temporal_rev_prob",   0.10)
            do_temp_sft = random.random() < self.cfg.get("aug_temporal_shift_prob", 0.10)

            blur_k     = random.choice([3, 5])
            blur_s     = random.uniform(0.3, 1.5)
            motion_k   = random.choice([5, 7, 9])
            motion_ang = random.uniform(0.0, 180.0)
            noise_std  = random.uniform(0.005, 0.035)
            gamma_val  = random.uniform(0.7, 1.5)
            bright_f   = random.uniform(0.75, 1.25) if random.random() < 0.45 else 1.0
            contrast_f = random.uniform(0.75, 1.25) if random.random() < 0.45 else 1.0
            sat_f      = random.uniform(0.75, 1.25) if random.random() < 0.35 else 1.0
            scale      = random.uniform(0.78, 1.00)
            crop_h     = min(int(H * scale), H)
            crop_w     = min(int(W * scale), W)
            top        = random.randint(0, max(0, H - crop_h))
            left       = random.randint(0, max(0, W - crop_w))
            temp_shift = random.randint(1, max(1, T // 4))
        else:
            view = self.tta_view
            n_views = len(_TTA_SCALES)
            view = min(view, n_views - 1)
            scale    = _TTA_SCALES[view]
            do_flip  = _TTA_FLIPS[view]
            crop_h   = min(int(H * scale), H)
            crop_w   = min(int(W * scale), W)
            top      = (H - crop_h) // 2
            left     = (W - crop_w) // 2
            bright_f = 1.0
            contrast_f = sat_f = 1.0
            do_temp_rev = do_temp_sft = False
            corrupt_choice = None
            blur_k = blur_s = motion_k = motion_ang = noise_std = gamma_val = 0
            temp_shift = 0

        if do_temp_rev:
            frames = frames[::-1]
        if do_temp_sft and T > 1:
            frames = frames[temp_shift:] + frames[:temp_shift]

        out: List[torch.Tensor] = []
        for frame in frames:
            img_np = np.ascontiguousarray(frame.astype(np.float32) / 255.0)
            img    = torch.from_numpy(img_np).permute(2, 0, 1)
            img    = img[:, top: top + crop_h, left: left + crop_w]
            img    = self._resize(img, self.size)

            if do_flip:
                img = self._hflip(img)

            if self.mode == "train":
                if bright_f != 1.0:   img = self._adjust_brightness(img, bright_f)
                if contrast_f != 1.0: img = self._adjust_contrast(img, contrast_f)
                if sat_f != 1.0:      img = self._adjust_saturation(img, sat_f)

                if corrupt_choice == "jpeg":
                    img = self._jpeg_sim(img)
                elif corrupt_choice == "blur":
                    img = self._gaussian_blur(img, blur_k, blur_s)
                elif corrupt_choice == "motion":
                    img = self._motion_blur(img, motion_k, motion_ang)
                elif corrupt_choice == "noise":
                    img = self._gaussian_noise(img, noise_std)
                elif corrupt_choice == "gamma":
                    img = self._random_gamma(img, gamma_val)
                elif corrupt_choice == "erasing":
                    img = self._random_erasing(img)
                elif corrupt_choice == "freq":
                    img = self._apply_freq_mask(img)

            img = img.clamp(0.0, 1.0)
            img = self._normalize(img)
            out.append(img)

        clip = torch.stack(out, dim=0)    # T C H W
        return clip.permute(1, 0, 2, 3)  # C T H W


# ══════════════════════════════════════════════════════════════════════════════
# 5b.  BATCH-LEVEL MIXUP  — [FIX-3] cutmix removed from the pipeline
# ══════════════════════════════════════════════════════════════════════════════

def mixup_data(
    clips:  torch.Tensor,
    labels: torch.Tensor,
    alpha:  float = 0.4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    lam   = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    B     = clips.size(0)
    idx   = torch.randperm(B, device=clips.device)
    mixed = lam * clips + (1.0 - lam) * clips[idx]
    return mixed, labels, labels[idx], lam


def mixup_criterion(
    criterion: nn.Module,
    logits:    torch.Tensor,
    labels_a:  torch.Tensor,
    labels_b:  torch.Tensor,
    lam:       float,
) -> torch.Tensor:
    return lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(logits, labels_b)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  DATASET  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

_EMPTY_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
_EMPTY_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)


class FFPPVideoDataset(Dataset):
    def __init__(
        self,
        video_list:      List[Tuple[str, int]],
        transform:       ClipTransform,
        num_frames:      int   = 32,
        frame_stride:    int   = 2,
        clips_per_video: int   = 1,
        face_backend:    str   = "none",
        face_margin:     float = 0.3,
        target_size:     int   = 224,
        fail_log:        Optional[str] = None,
        fail_log_lock:   Optional[Any] = None,
        face_cache_dir:  Optional[str] = None,
        use_face_cache:  bool  = False,
        deterministic:   bool  = False,
    ) -> None:
        self.transform       = transform
        self.num_frames      = num_frames
        self.frame_stride    = frame_stride
        self.clips_per_video = clips_per_video
        self.face_backend    = face_backend
        self.face_margin     = face_margin
        self.target_size     = target_size
        self.fail_log        = fail_log
        self.fail_log_lock   = fail_log_lock
        self.face_cache_dir  = face_cache_dir
        self.use_face_cache  = use_face_cache
        self.deterministic   = deterministic
        self.detector        = None

        self.samples         = [
            (p, l, w) for p, l in video_list for w in range(clips_per_video)
        ]
        self.clips_per_video_n = clips_per_video

    def __len__(self) -> int:
        return len(self.samples)

    def _log_fail(self, path: str, reason: str) -> None:
        if not self.fail_log:
            return
        entry = f"{path}\t{reason}\n"
        if self.fail_log_lock is not None:
            with self.fail_log_lock:
                with open(self.fail_log, "a") as f:
                    f.write(entry)
        else:
            with open(self.fail_log, "a") as f:
                f.write(entry)

    def _empty(self, label: int, name: str) -> Tuple[torch.Tensor, int, str]:
        clip = (
            torch.zeros(3, self.num_frames, self.target_size, self.target_size)
            .sub_(_EMPTY_MEAN)
            .div_(_EMPTY_STD)
        )
        return clip, label, name

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        path, label, window_idx = self.samples[idx]
        name        = Path(path).stem
        frames_rgb  = None

        if self.face_backend != "none":
            if self.use_face_cache and self.face_cache_dir:
                # [FIX-9] cache key now includes window_idx so distinct
                # deterministic windows of the same video don't collide.
                key        = _cache_key(
                    f"{path}#w{window_idx}", self.num_frames, self.frame_stride, self.target_size
                )
                frames_rgb = load_from_cache(self.face_cache_dir, key)

        if frames_rgb is None:
            frames_bgr = read_video_frames(
                path, self.num_frames, self.frame_stride,
                deterministic=self.deterministic,
                window_idx=window_idx,
                num_windows=self.clips_per_video_n,
            )
            if frames_bgr is None:
                self._log_fail(path, "read_failed")
                return self._empty(label, name)

            if self.face_backend != "none":
                if self.detector is None:
                    self.detector = FaceDetector(self.face_backend)
                result = crop_face_clip(frames_bgr, self.detector, self.face_margin, self.target_size)
                if result is not None:
                    frames_rgb = result
                else:
                    frames_rgb = _center_crop_frames(frames_bgr, self.target_size)
                    self._log_fail(path, "face_det_failed_center_crop_used")
                if self.use_face_cache and self.face_cache_dir:
                    key = _cache_key(
                        f"{path}#w{window_idx}", self.num_frames, self.frame_stride, self.target_size
                    )
                    save_to_cache(self.face_cache_dir, key, frames_rgb)
            else:
                frames_rgb = _center_crop_frames(frames_bgr, self.target_size)

        clip = self.transform(frames_rgb)
        return clip, label, name


# ══════════════════════════════════════════════════════════════════════════════
# 7.  DATA LOADING UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def mute_worker_stderr(worker_id: int) -> None:
    try:
        null_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(null_fd, 2)
    except Exception:
        pass


class UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: str, y: str) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[rx] = ry


def _extract_ff_source_ids(video_path: str) -> List[str]:
    stem = Path(video_path).stem
    m    = re.match(r'^(\d+)(?:_(\d+))?', stem)
    if m:
        return [g for g in m.groups() if g is not None]
    return [stem]


def group_aware_split(
    all_vids:   List[Tuple[str, int]],
    seed:       int,
    val_ratio:  float = 0.15,
    test_ratio: float = 0.15,
    logger:     Optional[logging.Logger] = None,
) -> Tuple[List, List, List]:
    log = logger or logging.getLogger("ff_teacher")

    uf       = UnionFind()
    vid_ids: Dict[str, List[str]] = {}

    for path, label in all_vids:
        ids = _extract_ff_source_ids(path)
        vid_ids[path] = ids
        for i in range(1, len(ids)):
            uf.union(ids[0], ids[i])

    group_to_vids: Dict[str, List] = defaultdict(list)
    for path, label in all_vids:
        ids   = vid_ids[path]
        group = uf.find(ids[0])
        group_to_vids[group].append((path, label))

    total_vids   = len(all_vids)
    largest_size = max(len(v) for v in group_to_vids.values()) if group_to_vids else 0
    connectivity = largest_size / max(total_vids, 1)

    if connectivity > 0.80:
        log.warning(
            f"[GROUP-SPLIT] Identity graph too connected "
            f"(largest group = {largest_size}/{total_vids} = {connectivity:.1%}). "
            "Falling back to SOURCE-ID-ONLY grouping (ignores reenactment target "
            "overlap) instead of a raw random split, to avoid identity leakage "
            "across train/val/test."
        )
        # Group only by each video's *source* id (ids[0]) rather than the full
        # transitive Union-Find closure. This still keeps every video derived
        # from the same source face in one split, it just tolerates some
        # target-identity overlap — far safer than a pure random shuffle.
        source_groups: Dict[str, List] = defaultdict(list)
        for path, label in all_vids:
            source_groups[vid_ids[path][0]].append((path, label))

        groups = sorted(source_groups.keys())
        rng    = random.Random(seed)
        rng.shuffle(groups)

        n       = len(groups)
        n_test  = max(1, int(n * test_ratio))
        n_val   = max(1, int(n * val_ratio))
        n_train = n - n_test - n_val

        train_groups = groups[:n_train]
        val_groups   = groups[n_train: n_train + n_val]
        test_groups  = groups[n_train + n_val:]

        tr = [v for g in train_groups for v in source_groups[g]]
        vl = [v for g in val_groups   for v in source_groups[g]]
        te = [v for g in test_groups  for v in source_groups[g]]

        log.info(
            f"[GROUP-SPLIT] Source-id fallback: "
            f"train={len(tr)}  val={len(vl)}  test={len(te)}"
        )
        return tr, vl, te

    groups = sorted(group_to_vids.keys())
    rng    = random.Random(seed)
    rng.shuffle(groups)

    n       = len(groups)
    n_test  = max(1, int(n * test_ratio))
    n_val   = max(1, int(n * val_ratio))
    n_train = n - n_test - n_val

    train_groups = groups[:n_train]
    val_groups   = groups[n_train: n_train + n_val]
    test_groups  = groups[n_train + n_val:]

    tr = [v for g in train_groups for v in group_to_vids[g]]
    vl = [v for g in val_groups   for v in group_to_vids[g]]
    te = [v for g in test_groups  for v in group_to_vids[g]]

    log.info(
        f"[GROUP-SPLIT] Identity-aware: "
        f"groups={n}  train={len(train_groups)}  val={len(val_groups)}  test={len(test_groups)}"
    )
    log.info(f"  Videos: train={len(tr)}  val={len(vl)}  test={len(te)}")
    for split_name, split in [("train", tr), ("val", vl), ("test", te)]:
        n_r = sum(1 for _, l in split if l == 0)
        n_f = sum(1 for _, l in split if l == 1)
        log.info(f"  {split_name}: real={n_r}  fake={n_f}  ratio={n_f/(n_r+1e-9):.2f}:1")

    return tr, vl, te


def discover_videos(
    cfg: dict, logger: logging.Logger
) -> Tuple[List, List, List]:
    root         = Path(cfg["data_root"])
    real_folder  = cfg["real_folder"]
    fake_folders = cfg["fake_folders"]
    csv_dir      = cfg.get("csv_dir", "")
    exts         = {".mp4", ".avi", ".mov", ".mkv"}

    if csv_dir:
        csv_root = root / csv_dir
        paths    = [csv_root / "train.csv", csv_root / "val.csv", csv_root / "test.csv"]
        missing  = [p for p in paths if not p.exists()]
        if missing:
            logger.error(
                f"csv_dir='{csv_dir}' was set but these split files are missing: "
                f"{[str(p) for p in missing]}. Refusing to silently fall back to "
                "auto-discovery, since that path is more leak-prone. Fix csv_dir "
                "or unset it if you intend to use auto-discovery."
            )
            sys.exit(1)
        logger.info(f"CSV splits found at {csv_root}")
        ...
        return tr, vl, te

    logger.info("csv_dir not set — auto-discovering with identity-aware group split.")
    real_vids = sorted(p for p in (root / real_folder).rglob("*") if p.suffix.lower() in exts)
    fake_vids: List[Path] = []
    for ff in fake_folders:
        d = root / ff
        if d.exists():
            fake_vids.extend(sorted(p for p in d.rglob("*") if p.suffix.lower() in exts))
        else:
            logger.warning(f"Folder not found: {d}")

    logger.info(f"Found {len(real_vids)} real  +  {len(fake_vids)} fake videos.")
    if not real_vids and not fake_vids:
        logger.error(f"No videos found in {root} — check data_root in CONFIG.")
        sys.exit(1)

    all_vids = [(str(p), 0) for p in real_vids] + [(str(p), 1) for p in fake_vids]
    return group_aware_split(all_vids, cfg["seed"], logger=logger)


def _manipulation_bucket(path: str, real_folder: str, fake_folders: List[str]) -> str:

    parts = Path(path).parts
    for ff in fake_folders:
        if ff in parts:
            return ff
    if real_folder in parts:
        return real_folder
    return "unknown"


def make_weighted_sampler(
    dataset:      FFPPVideoDataset,
    real_folder:  str = "original",
    fake_folders: Optional[List[str]] = None,
    logger:       Optional[logging.Logger] = None,
) -> WeightedRandomSampler:
    fake_folders = fake_folders or []
    bucket_of = [
        _manipulation_bucket(p, real_folder, fake_folders)
        for p, _l, _w in dataset.samples
    ]
    bucket_counts: Dict[str, int] = {}
    for b in bucket_of:
        bucket_counts[b] = bucket_counts.get(b, 0) + 1

    fake_bucket_names = [b for b in bucket_counts if b != real_folder]
    n_fake_buckets    = max(1, len(fake_bucket_names))

    clip_weights: List[float] = []
    for b in bucket_of:
        if b == real_folder:

            w = 0.5 / bucket_counts[b]
        else:

            w = (0.5 / n_fake_buckets) / bucket_counts[b]
        clip_weights.append(w)

    total_mass      = sum(clip_weights)
    real_mass_share = sum(
        w for w, b in zip(clip_weights, bucket_of) if b == real_folder
    ) / max(total_mass, 1e-12)
    fake_mass_share = 1.0 - real_mass_share
    log = logger or logging.getLogger("ff_teacher")
    log.info(
        f"[FIX-13] Sampler realized mass — Real:{real_mass_share:.3f}  "
        f"Fake:{fake_mass_share:.3f}  (target 0.500 / 0.500)  "
        f"fake_buckets={n_fake_buckets}  bucket_counts={bucket_counts}"
    )
    if abs(real_mass_share - 0.5) > 0.02:
        log.warning(
            f"[FIX-13] Sampler Real mass share ({real_mass_share:.3f}) drifted "
            f">2pp from the intended 0.500 — double-check real_folder/"
            f"fake_folders match the actual dataset folder names."
        )

    return WeightedRandomSampler(
        weights=clip_weights, num_samples=len(clip_weights), replacement=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 8.  LOSS — FOCAL LOSS  (logic unchanged; see CONFIG.focal_alpha for FIX-14)
# ══════════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma:           float       = 2.0,
        alpha:           List[float] = None,
        num_classes:     int         = 2,
        reduction:       str         = "mean",
        label_smoothing: float       = 0.0,
    ) -> None:
        super().__init__()
        self.gamma           = gamma
        self.reduction       = reduction
        self.label_smoothing = label_smoothing
        if alpha is None:
            alpha = [1.0, 1.0]
        alpha_t = torch.tensor(alpha, dtype=torch.float32)
        alpha_t = alpha_t / alpha_t.sum()
        self.register_buffer("alpha_t", alpha_t)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            probs = F.softmax(logits.float(), dim=-1)
            pt    = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        ce_loss      = F.cross_entropy(
            logits, targets, reduction="none",
            label_smoothing=self.label_smoothing,
        )
        alpha_t      = self.alpha_t.to(logits.device)[targets]
        focal_weight = (1.0 - pt).pow(self.gamma)
        focal_loss   = alpha_t * focal_weight * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# ══════════════════════════════════════════════════════════════════════════════
# 9.  METRICS
# ══════════════════════════════════════════════════════════════════════════════

def eer(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        fnr = 1.0 - tpr
        i   = np.nanargmin(np.abs(fnr - fpr))
        return float((fpr[i] + fnr[i]) / 2)
    except Exception:
        return float("nan")


def find_optimal_threshold(
    y_true:  List[int],
    y_score: List[float],
    metric:  str = "balanced_acc",
) -> float:

    if not SKLEARN_OK:
        return 0.5
    try:
        y_true_arr  = np.asarray(y_true)
        y_score_arr = np.asarray(y_score)
        if len(np.unique(y_true_arr)) < 2:
            return 0.5

        _, _, roc_thr = roc_curve(y_true_arr, y_score_arr)
        _, _, pr_thr  = precision_recall_curve(y_true_arr, y_score_arr)
        thresholds = np.unique(np.concatenate([roc_thr, pr_thr]))
        thresholds = thresholds[(thresholds >= 0.05) & (thresholds <= 0.95)]
        if len(thresholds) == 0:
            return 0.5

        best_score = -np.inf
        best_thr   = 0.5

        for thr in thresholds:
            y_pred = (y_score_arr >= thr).astype(int)
            try:
                if metric == "mcc":
                    score = matthews_corrcoef(y_true_arr, y_pred)
                elif metric == "f1":
                    score = f1_score(y_true_arr, y_pred, zero_division=0)
                else:
                    score = balanced_accuracy_score(y_true_arr, y_pred)
                if score > best_score:
                    best_score = score
                    best_thr   = thr
            except Exception:
                continue

        return float(np.clip(best_thr, 0.05, 0.95))
    except Exception:
        return 0.5


def compute_metrics(
    y_true: List[int], y_score: List[float], thr: float = 0.5
) -> dict:
    y_true  = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred  = (y_score >= thr).astype(int)
    m: Dict[str, Any] = {}
    if not SKLEARN_OK:
        m["accuracy"] = float(np.mean(y_pred == y_true))
        return m
    try:
        m["accuracy"]     = float(accuracy_score(y_true, y_pred))
        m["roc_auc"]      = float(roc_auc_score(y_true, y_score))
        m["pr_auc"]       = float(average_precision_score(y_true, y_score))
        m["f1"]           = float(f1_score(y_true, y_pred, zero_division=0))
        m["precision"]    = float(precision_score(y_true, y_pred, zero_division=0))
        m["recall"]       = float(recall_score(y_true, y_pred, zero_division=0))
        m["balanced_acc"] = float(balanced_accuracy_score(y_true, y_pred))
        m["mcc"]          = float(matthews_corrcoef(y_true, y_pred))
        m["eer"]          = eer(y_true, y_score)
        tn, fp, fn, tp    = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        m["specificity"]  = float(tn / (tn + fp + 1e-9))
        m["tp"] = int(tp);  m["tn"] = int(tn)
        m["fp"] = int(fp);  m["fn"] = int(fn)
    except Exception:
        for k in ["accuracy", "roc_auc", "pr_auc", "f1", "precision", "recall",
                  "balanced_acc", "mcc", "eer", "specificity"]:
            m.setdefault(k, float("nan"))
    return m


def aggregate_clips(
    names: List[str], labels: List[int], probs: List[float]
) -> Tuple[List[str], List[int], List[float]]:
    pool: Dict[str, List[float]] = defaultdict(list)
    lbl_map: Dict[str, int]      = {}
    for n, l, p in zip(names, labels, probs):
        pool[n].append(p)
        lbl_map[n] = l
    ns = list(pool.keys())
    return ns, [lbl_map[n] for n in ns], [float(np.mean(pool[n])) for n in ns]


# ══════════════════════════════════════════════════════════════════════════════
# 9b.  TEST-TIME AUGMENTATION (TTA)  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_with_tta(
    model:     nn.Module,
    val_list:  List[Tuple[str, int]],
    criterion: nn.Module,
    device:    torch.device,
    cfg:       dict,
    logger:    logging.Logger,
    threshold: float = 0.5,
    n_views:   int   = 6,
) -> Tuple[dict, Tuple[List, List, List]]:
    n_views = min(n_views, len(_TTA_SCALES))
    logger.info(f"[TTA] Running {n_views} views  "
                f"scales={[_TTA_SCALES[v] for v in range(n_views)]}  "
                f"flips={[_TTA_FLIPS[v] for v in range(n_views)]}")
    all_probs_by_name: Dict[str, List[float]] = defaultdict(list)
    all_labels_by_name: Dict[str, int]         = {}

    nw      = cfg["num_workers"]
    use_pin = cfg["pin_memory"] and torch.cuda.is_available()

    for view_idx in range(n_views):
        tfm = ClipTransform(mode="val", size=cfg["input_size"], cfg=cfg, tta_view=view_idx)
        ds = FFPPVideoDataset(
            val_list, tfm,
            num_frames      = cfg["frames_per_clip"],
            frame_stride    = cfg["frame_stride"],
            clips_per_video = cfg["clips_per_video_val"],
            face_backend    = cfg["face_det_backend"],
            face_margin     = cfg["face_margin"],
            target_size     = cfg["input_size"],
            deterministic   = True,
        )
        loader = DataLoader(
            ds, batch_size=cfg["batch_size"], shuffle=False,
            num_workers=nw, pin_memory=use_pin,
            persistent_workers=(nw > 0),
            prefetch_factor=2 if nw > 0 else None,
            worker_init_fn=mute_worker_stderr,
        )

        model.eval()
        all_y: List[int]   = []
        all_p: List[float] = []
        all_n: List[str]   = []

        with torch.no_grad():
            for clips, labels, names in loader:
                clips  = clips.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with autocast(enabled=cfg["amp"]):
                    logits = model(clips)
                probs = torch.softmax(logits.float(), 1)[:, 1].cpu().numpy()
                all_p.extend(probs.tolist())
                all_y.extend(labels.cpu().numpy().tolist())
                all_n.extend(list(names))

        vnames, vlabels, vprobs = aggregate_clips(all_n, all_y, all_p)
        for nm, lb, pb in zip(vnames, vlabels, vprobs):
            all_probs_by_name[nm].append(pb)
            all_labels_by_name[nm] = lb

        view_auc_str = ""
        if SKLEARN_OK and len(set(vlabels)) > 1:
            try:
                view_auc_str = f"  auc={roc_auc_score(vlabels, vprobs):.4f}"
            except Exception:
                pass
        logger.info(
            f"  [TTA] view={view_idx}  scale={_TTA_SCALES[view_idx]}  "
            f"flip={_TTA_FLIPS[view_idx]}{view_auc_str}"
        )

        del ds, loader
        gc.collect()

    final_names  = sorted(all_probs_by_name.keys())
    final_labels = [all_labels_by_name[n] for n in final_names]
    final_probs  = [float(np.mean(all_probs_by_name[n])) for n in final_names]

    metrics             = compute_metrics(final_labels, final_probs, thr=threshold)
    metrics["val_loss"] = float("nan")
    logger.info(
        f"[TTA] Final averaged: "
        f"auc={metrics.get('roc_auc', float('nan')):.4f}  "
        f"f1={metrics.get('f1', float('nan')):.4f}  "
        f"mcc={metrics.get('mcc', float('nan')):.4f}  "
        f"spec={metrics.get('specificity', float('nan')):.4f}"
    )
    return metrics, (final_names, final_labels, final_probs)


# ══════════════════════════════════════════════════════════════════════════════
# 10.  CHECKPOINTING  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def save_ckpt(
    state:      dict,
    out_dir:    str,
    epoch:      int  = 0,
    best_auc:   bool = False,
    best_f1:    bool = False,
    prev_epoch: Optional[int] = None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    if prev_epoch is not None:
        prev_path = os.path.join(out_dir, f"checkpoint_epoch_{prev_epoch:03d}.pth")
        if os.path.isfile(prev_path):
            try:
                os.remove(prev_path)
            except Exception:
                pass
    per_epoch_path = os.path.join(out_dir, f"checkpoint_epoch_{epoch:03d}.pth")
    torch.save(state, per_epoch_path)
    torch.save(state, os.path.join(out_dir, "checkpoint_latest.pth"))
    if best_auc:
        torch.save(state, os.path.join(out_dir, "best_teacher_model.pth"))


def load_ckpt(
    path:      str,
    model:     nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler:    Optional[Any] = None,
) -> dict:
    ckpt  = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["model_state"]
    if not state:
        raise ValueError(f"Empty model_state in checkpoint '{path}'")
    first_key  = next(iter(state.keys()))
    is_dp_ckpt = first_key.startswith("module.")
    is_dp_mod  = isinstance(model, nn.DataParallel)
    if is_dp_ckpt and not is_dp_mod:
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    elif not is_dp_ckpt and is_dp_mod:
        state = {"module." + k: v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    if optimizer and "optimizer_state" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        except Exception as e:
            logging.getLogger("ff_teacher").warning(
                f"Could not restore optimizer state: {e}. Starting fresh optimizer."
            )
    if scheduler and "scheduler_state" in ckpt:
        try:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        except Exception as e:
            logging.getLogger("ff_teacher").warning(
                f"Could not restore scheduler state: {e}."
            )
    if scaler and "scaler_state" in ckpt:
        try:
            scaler.load_state_dict(ckpt["scaler_state"])
        except Exception as e:
            logging.getLogger("ff_teacher").warning(
                f"Could not restore scaler state: {e}."
            )
    return ckpt


def load_weights_only(path: str, model: nn.Module, logger: logging.Logger) -> None:
    ckpt  = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("ema_model_state") or ckpt["model_state"]
    is_dp_ckpt = next(iter(state.keys())).startswith("module.")
    is_dp_mod  = isinstance(model, nn.DataParallel)
    if is_dp_ckpt and not is_dp_mod:
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    elif not is_dp_ckpt and is_dp_mod:
        state = {"module." + k: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.info(
        f"[WARM-START] Loaded weights from '{path}' "
        f"(missing={len(missing)}, unexpected={len(unexpected)}). "
        f"Optimizer/scheduler/threshold state NOT restored — starting fresh "
        f"under the v4 recipe (fixed sampler)."
    )


# ══════════════════════════════════════════════════════════════════════════════
# 11.  SCHEDULER  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def build_scheduler(
    optimizer:       torch.optim.Optimizer,
    cfg:             dict,
    steps_per_epoch: int,
    swa_start_epoch: Optional[int] = None,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup = cfg["warmup_epochs"] * steps_per_epoch
    total_cosine_epochs = swa_start_epoch if swa_start_epoch else cfg["num_epochs"]
    total  = total_cosine_epochs * steps_per_epoch

    def lr_fn(step: int) -> float:
        if step < warmup:
            return float(step) / max(warmup, 1)
        prog = float(step - warmup) / max(total - warmup, 1)
        prog = min(prog, 1.0)
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * prog)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)


def compute_hf_weight(epoch: int, cfg: dict) -> float:
    target = cfg.get("hf_aux_weight", 0.20)
    warmup = cfg.get("hf_aux_warmup_epochs", 5)
    if epoch <= 0 or warmup <= 0:
        return target
    return target * min(1.0, epoch / warmup)


# ══════════════════════════════════════════════════════════════════════════════
# 12.  EMA MODEL  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def compute_ema_decay(epoch: int, cfg: dict) -> float:
    decay_max     = cfg.get("ema_decay_max",          0.999)
    warmup_epochs = cfg.get("ema_decay_warmup_epochs", 5)
    if epoch <= 0:
        return 0.90
    if epoch >= warmup_epochs:
        return decay_max
    alpha = epoch / warmup_epochs
    return 0.90 + alpha * (decay_max - 0.90)


def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(decay).add_(p.data, alpha=1.0 - decay)
        for ema_b, b in zip(ema_model.buffers(), model.buffers()):
            ema_b.copy_(b)


# ══════════════════════════════════════════════════════════════════════════════
# 12b.  STOCHASTIC WEIGHT AVERAGING (SWA)  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

class SWAModel:
    def __init__(self, model: nn.Module) -> None:
        self.avg_state: Dict[str, torch.Tensor] = {}
        self.n = 0
        self._init(model)

    def _init(self, model: nn.Module) -> None:
        for name, param in model.state_dict().items():
            self.avg_state[name] = param.float().clone()

    def update(self, model: nn.Module) -> None:
        self.n += 1
        with torch.no_grad():
            for name, param in model.state_dict().items():
                if name in self.avg_state:
                    target = self.avg_state[name]
                    if target.device != param.device:
                        target = target.to(param.device)
                        self.avg_state[name] = target
                    target += (param.float() - target) / self.n

    def apply(self, model: nn.Module) -> None:
        current_state = model.state_dict()
        new_state = {}
        for name, param in current_state.items():
            if name in self.avg_state:
                new_state[name] = self.avg_state[name].to(param.dtype).to(param.device)
            else:
                new_state[name] = param
        model.load_state_dict(new_state, strict=False)

    def state_dict(self) -> dict:
        return {
            "avg_state": {k: v.detach().cpu().clone() for k, v in self.avg_state.items()},
            "n": self.n,
        }

    def load_state_dict(self, d: dict) -> None:
        self.avg_state = {k: v.clone() for k, v in d["avg_state"].items()}
        self.n          = d["n"]

    def to(self, device: torch.device) -> "SWAModel":
        self.avg_state = {k: v.to(device) for k, v in self.avg_state.items()}
        return self


# ══════════════════════════════════════════════════════════════════════════════
# 13.  TRAINING LOOP  — [FIX-3] cutmix branch removed
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model:          nn.Module,
    loader:         DataLoader,
    criterion:      nn.Module,
    optimizer:      torch.optim.Optimizer,
    scaler:         Any,
    scheduler:      torch.optim.lr_scheduler.LambdaLR,
    device:         torch.device,
    cfg:            dict,
    epoch:          int,
    logger:         logging.Logger,
    timer:          Timer,
    is_first_epoch: bool = False,
    ema_model:      Optional[nn.Module] = None,
    ema_decay:      float               = 0.999,
    skip_scheduler_step: bool           = False,
) -> dict:
    model.train()
    loss_m     = AverageMeter()
    all_y: List[int]   = []
    all_p: List[float] = []
    nan_loss_n = nan_grad_n = oom_n = 0
    grad_accum_steps  = cfg.get("grad_accum_steps", 1)
    use_mixup         = cfg.get("use_mixup",  True)
    mc_prob           = cfg.get("mixup_cutmix_prob", 0.15)
    mixup_alpha       = cfg.get("mixup_alpha",  0.4)
    gnorm_display: float = 0.0

    pbar = (
        tqdm(total=len(loader),
             desc=f"Epoch {epoch}/{cfg['num_epochs']} [train]",
             leave=True, dynamic_ncols=True, unit="batch")
        if TQDM_OK else None
    )
    optimizer.zero_grad(set_to_none=True)

    for i, (clips, labels, _) in enumerate(loader):
        if timer.elapsed() > cfg["train_budget_sec"]:
            logger.warning("Training budget hit mid-epoch — stopping epoch early.")
            break

        if i == 0:
            n_pos = int(labels.sum().item())
            n_neg = int((labels == 0).sum().item())
            total = n_pos + n_neg
            ratio = n_pos / max(total, 1)
            logger.info(
                f"  E{epoch:03d} Batch[0] label dist — "
                f"Real:{n_neg}  Fake:{n_pos}  pos_ratio={ratio:.3f}"
            )
            if ratio < 0.2 or ratio > 0.8:
                logger.warning(
                    f"  [SAMPLER WARNING] First batch severely imbalanced "
                    f"(pos_ratio={ratio:.3f}). Sampler may be broken."
                )

        if is_first_epoch and i == 0:
            debug_path = save_debug_batch(clips, labels, cfg["output_dir"])
            if debug_path:
                logger.info(f"  Debug batch image saved → {debug_path}")

        clips  = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        labels_b  = labels
        lam       = 1.0
        # [FIX-3] mixup-only now (no cutmix branch)
        do_mix    = use_mixup and random.random() < mc_prob and clips.size(0) > 1
        if do_mix:
            clips, labels, labels_b, lam = mixup_data(clips, labels, mixup_alpha)

        is_accum_boundary = (
            (i + 1) % grad_accum_steps == 0 or (i + 1) == len(loader)
        )

        try:
            with autocast(enabled=cfg["amp"]):
                logits = model(clips)
                if do_mix and lam < 1.0 - 1e-5:
                    loss = mixup_criterion(criterion, logits, labels, labels_b, lam)
                else:
                    loss = criterion(logits, labels)
        except torch.cuda.OutOfMemoryError:
            oom_n += 1
            logger.warning(f"  [E{epoch} B{i}] CUDA OOM (count={oom_n}) — skipping batch.")
            torch.cuda.empty_cache()
            if is_accum_boundary:
                scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if pbar is not None:
                pbar.update(1)
            continue

        if not torch.isfinite(loss):
            nan_loss_n += 1
            logger.warning(f"  [E{epoch} B{i}] Non-finite loss — skipping backward.")
            if is_accum_boundary:
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            if pbar is not None:
                pbar.update(1)
            continue

        scaled_loss = loss / grad_accum_steps
        scaler.scale(scaled_loss).backward()

        if is_accum_boundary:
            scaler.unscale_(optimizer)
            gnorm_tensor  = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            gnorm_display = float(gnorm_tensor.item())

            if not math.isfinite(gnorm_display):
                nan_grad_n += 1
                logger.warning(f"  [E{epoch} B{i}] Non-finite grad norm — skipping step.")
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if pbar is not None:
                    pbar.update(1)
                continue

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if not skip_scheduler_step:
                scheduler.step()

            if ema_model is not None:
                update_ema(ema_model, model, ema_decay)

        with torch.no_grad():
            probs = torch.softmax(logits.detach().float(), 1)[:, 1].cpu().numpy()
        all_p.extend(probs.tolist())
        if do_mix and lam < 1.0 - 1e-5:
            blended = (lam * labels.float() + (1.0 - lam) * labels_b.float()).round().long()
            all_y.extend(blended.cpu().numpy().tolist())
        else:
            all_y.extend(labels.cpu().numpy().tolist())
        loss_m.update(loss.item(), clips.size(0))

        if pbar is not None:
            head_lr = get_head_lr(optimizer)
            pbar.set_postfix(
                loss=f"{loss_m.avg:.4f}",
                head_lr=f"{head_lr:.2e}",
                gnorm=f"{gnorm_display:.2f}",
            )
            pbar.update(1)

        if (i + 1) % cfg["log_interval"] == 0:
            head_lr = get_head_lr(optimizer)
            logger.info(
                f"  E{epoch:03d} [{i + 1:4d}/{len(loader)}] "
                f"loss={loss_m.avg:.4f}  gnorm={gnorm_display:.3f}  "
                f"head_lr={head_lr:.2e}  ema_decay={ema_decay:.4f}"
            )

    if pbar is not None:
        pbar.close()
    if nan_loss_n:
        logger.warning(f"  {nan_loss_n} non-finite-loss batches this epoch.")
    if nan_grad_n:
        logger.warning(f"  {nan_grad_n} non-finite-grad-norm batches this epoch.")
    if oom_n:
        logger.warning(f"  {oom_n} OOM batches.")

    train_acc = train_bal_acc = train_auc = 0.0
    if all_y:
        all_p_arr = np.array(all_p)
        all_y_arr = np.array(all_y, dtype=int)
        p_std = float(all_p_arr.std())
        logger.info(
            f"  E{epoch:03d} train score dist — "
            f"mean={all_p_arr.mean():.4f}  std={p_std:.4f}  "
            f"p>0.5: {(all_p_arr > 0.5).mean() * 100:.1f}%"
        )
        if p_std < 0.05:
            logger.warning(
                f"  [COLLAPSE WARNING] Train score std={p_std:.4f} < 0.05."
            )
        preds_bin = (all_p_arr >= 0.5).astype(int)
        train_acc = float(np.mean(preds_bin == all_y_arr))
        if SKLEARN_OK and len(np.unique(all_y_arr)) > 1:
            train_bal_acc = float(balanced_accuracy_score(all_y_arr, preds_bin))
            try:
                train_auc = float(roc_auc_score(all_y_arr, all_p_arr))
            except Exception:
                train_auc = float("nan")
        else:
            train_bal_acc = train_acc
            train_auc     = float("nan")
        if not math.isnan(train_auc) and train_auc < 0.55:
            logger.warning(
                f"  [LEARNING WARNING] Train AUC={train_auc:.4f} < 0.55 "
                f"after epoch {epoch}."
            )

    return {
        "train_loss":         loss_m.avg,
        "train_acc":          train_acc,
        "train_balanced_acc": train_bal_acc,
        "train_auc":          train_auc,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 14.  VALIDATION LOOP  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def validate(
    model:        nn.Module,
    loader:       DataLoader,
    criterion:    nn.Module,
    device:       torch.device,
    cfg:          dict,
    logger:       logging.Logger,
    return_preds: bool  = False,
    desc:         str   = "Validation",
    threshold:    float = 0.5,
) -> Any:
    model.eval()
    loss_m = AverageMeter()
    all_y: List[int]   = []
    all_p: List[float] = []
    all_n: List[str]   = []

    pbar = (
        tqdm(total=len(loader), desc=desc, leave=True,
             dynamic_ncols=True, unit="batch")
        if TQDM_OK else None
    )

    for clips, labels, names in loader:
        clips  = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast(enabled=cfg["amp"]):
            logits = model(clips)
            loss   = criterion(logits, labels)
        if torch.isfinite(loss):
            loss_m.update(loss.item(), clips.size(0))
        probs = torch.softmax(logits.float(), 1)[:, 1].cpu().numpy()
        all_p.extend(probs.tolist())
        all_y.extend(labels.cpu().numpy().tolist())
        all_n.extend(list(names))
        if pbar is not None:
            try:
                clip_auc = (
                    float(roc_auc_score(all_y, all_p))
                    if SKLEARN_OK and len(set(all_y)) > 1 else float("nan")
                )
            except Exception:
                clip_auc = float("nan")
            pbar.set_postfix(loss=f"{loss_m.avg:.4f}", clip_auc=f"{clip_auc:.4f}")
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    if all_p:
        p_arr = np.array(all_p)
        p_std = float(p_arr.std())
        logger.info(
            f"  Score dist: mean={p_arr.mean():.4f}  std={p_std:.4f}  "
            f"p>0.5: {(p_arr > 0.5).mean() * 100:.1f}%  "
            f"p>0.9: {(p_arr > 0.9).mean() * 100:.1f}%"
        )
        if p_std < 0.05:
            logger.warning(f"  [COLLAPSE WARNING] Score std={p_std:.4f} < 0.05.")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    vnames, vlabels, vprobs = aggregate_clips(all_n, all_y, all_p)

    try:
        clip_auc = (
            float(roc_auc_score(all_y, all_p))
            if SKLEARN_OK and len(set(all_y)) > 1 else float("nan")
        )
    except Exception:
        clip_auc = float("nan")

    metrics             = compute_metrics(vlabels, vprobs, thr=threshold)
    metrics["val_loss"] = loss_m.avg
    metrics["clip_auc"] = clip_auc
    metrics["val_acc"]  = metrics.get("accuracy", float("nan"))

    spec = metrics.get("specificity", float("nan"))
    bal  = metrics.get("balanced_acc", float("nan"))
    if not math.isnan(spec) and spec < 0.3:
        logger.warning(f"  [COLLAPSE WARNING] specificity={spec:.4f} < 0.3 — biased toward Fake.")
    if not math.isnan(bal) and bal < 0.55:
        logger.warning(f"  [COLLAPSE WARNING] balanced_acc={bal:.4f} near random.")

    if return_preds:
        return metrics, (vnames, vlabels, vprobs)
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# 15.  CSV LOGGING  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

HISTORY_COLS = [
    "epoch", "train_loss", "val_loss", "train_acc", "train_balanced_acc",
    "train_auc", "val_acc", "roc_auc", "pr_auc", "f1", "precision", "recall",
    "specificity", "balanced_acc", "mcc", "eer", "optimal_threshold",
    "ema_decay", "swa_active",
]


def append_history_csv(path: str, row: dict, write_header: bool = False) -> None:
    mode = "w" if write_header else "a"
    with open(path, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=HISTORY_COLS, extrasaction="ignore")
        if write_header:
            w.writeheader()
            return
        w.writerow({k: row.get(k, float("nan")) for k in HISTORY_COLS})


def write_predictions_csv(
    path:   str,
    names:  List[str],
    labels: List[int],
    probs:  List[float],
    thr:    float = 0.5,
) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_name", "true_label", "pred_probability", "pred_label"])
        for n, l, p in zip(names, labels, probs):
            w.writerow([n, l, f"{p:.6f}", int(p >= thr)])


# ══════════════════════════════════════════════════════════════════════════════
# 16.  FIGURES  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def _save_fig(fig: Any, out_dir: str, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(
            os.path.join(out_dir, f"{name}.{ext}"),
            dpi=150, bbox_inches="tight",
            facecolor="white", edgecolor="none",
        )


def plot_curves(history: List[dict], out_dir: str) -> None:
    if not MATPLOTLIB_OK or not history:
        return
    ep     = [h["epoch"]                                         for h in history]
    trl    = [h["train_loss"]                                    for h in history]
    vll    = [h["val_loss"]                                      for h in history]
    tra    = [h["train_acc"]                                     for h in history]
    trba   = [h.get("train_balanced_acc", float("nan"))          for h in history]
    trauc  = [h.get("train_auc",          float("nan"))          for h in history]
    vla    = [h.get("val_acc", h.get("accuracy", float("nan")))  for h in history]
    auc    = [h.get("roc_auc",            float("nan"))          for h in history]
    spe    = [h.get("specificity",        float("nan"))          for h in history]
    rec    = [h.get("recall",             float("nan"))          for h in history]
    mcc    = [h.get("mcc",                float("nan"))          for h in history]
    bal    = [h.get("balanced_acc",       float("nan"))          for h in history]
    thr_l  = [h.get("optimal_threshold",  0.5)                   for h in history]
    ema_d  = [h.get("ema_decay",          float("nan"))          for h in history]

    for fname, title, ys, labels, colours in [
        ("01_train_loss",  "Training Loss",
         [trl],            ["Train"],                          ["b"]),
        ("02_val_loss",    "Validation Loss",
         [vll],            ["Val"],                            ["r"]),
        ("03_accuracy",    "Accuracy (raw & balanced)",
         [tra, trba, vla], ["Train", "Train-Bal", "Val"],      ["b", "c", "r"]),
        ("04_roc_auc",     "ROC-AUC (train & val)",
         [trauc, auc],     ["Train AUC", "Val AUC"],           ["steelblue", "g"]),
        ("05_spec_recall", "Specificity vs Recall — key balance metric",
         [spe, rec],       ["Specificity", "Recall"],          ["darkorange", "steelblue"]),
        ("06_mcc",         "Matthews Correlation",
         [mcc],            ["MCC"],                            ["purple"]),
        ("07_balanced_acc","Balanced Accuracy",
         [bal],            ["Balanced Acc"],                   ["teal"]),
        ("08_threshold",   "Optimal Threshold over Epochs",
         [thr_l],          ["Threshold"],                      ["brown"]),
        ("09_ema_decay",   "Dynamic EMA Decay over Epochs",
         [ema_d],          ["EMA decay"],                      ["navy"]),
    ]:
        try:
            fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
            for y, lbl, c in zip(ys, labels, colours):
                ax.plot(ep, y, "-o", color=c, ms=4, label=lbl)
            ax.set_xlabel("Epoch")
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            _save_fig(fig, out_dir, fname)
            plt.close(fig)
        except Exception as exc:
            logging.getLogger("ff_teacher").warning(f"plot_curves({fname}): {exc}")


def plot_eval(
    y_true: List[int],
    y_score: List[float],
    out_dir: str,
    split:   str   = "test",
    thr:     float = 0.5,
) -> None:
    if not MATPLOTLIB_OK or not SKLEARN_OK:
        return
    y_true  = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred  = (y_score >= thr).astype(int)

    try:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_v       = roc_auc_score(y_true, y_score)
        fig, ax     = plt.subplots(figsize=(7, 6), facecolor="white")
        ax.plot(fpr, tpr, "b-", lw=2, label=f"AUC={auc_v:.4f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title(f"ROC Curve ({split})"); ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout(); _save_fig(fig, out_dir, f"10_roc_{split}"); plt.close(fig)
    except Exception:
        pass

    try:
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        ap           = average_precision_score(y_true, y_score)
        fig, ax      = plt.subplots(figsize=(7, 6), facecolor="white")
        ax.plot(rec, prec, "r-", lw=2, label=f"AP={ap:.4f}")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_title(f"PR Curve ({split})"); ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout(); _save_fig(fig, out_dir, f"11_pr_{split}"); plt.close(fig)
    except Exception:
        pass

    try:
        cm      = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(6, 5), facecolor="white")
        im = ax.imshow(cm, cmap=plt.cm.Blues)
        fig.colorbar(im, ax=ax)
        ax.set(
            xticks=[0, 1], yticks=[0, 1],
            xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"],
            xlabel="Predicted", ylabel="True",
            title=f"Confusion Matrix ({split}, thr={thr:.3f})",
        )
        th = cm.max() / 2
        for ii in range(2):
            for jj in range(2):
                ax.text(jj, ii, cm[ii, jj], ha="center", va="center",
                        color="white" if cm[ii, jj] > th else "black")
        fig.tight_layout(); _save_fig(fig, out_dir, f"12_cm_{split}"); plt.close(fig)
    except Exception:
        pass

    try:
        met  = compute_metrics(y_true.tolist(), y_score.tolist(), thr=thr)
        keys = ["accuracy", "roc_auc", "pr_auc", "f1", "precision",
                "recall", "specificity", "balanced_acc", "mcc"]
        vals = [met.get(k, 0.0) for k in keys]
        fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
        bars    = ax.bar(keys, vals, color="steelblue", edgecolor="black")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, 1.15); ax.set_ylabel("Score")
        ax.set_title(f"Metric Summary ({split}, thr={thr:.3f})")
        plt.xticks(rotation=30, ha="right")
        fig.tight_layout(); _save_fig(fig, out_dir, f"13_metrics_{split}"); plt.close(fig)
    except Exception:
        pass

    try:
        nb    = 10
        edges = np.linspace(0, 1, nb + 1)
        mids  = (edges[:-1] + edges[1:]) / 2
        fpos  = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (y_score >= lo) & (y_score < hi)
            fpos.append(y_true[m].mean() if m.sum() > 0 else np.nan)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="white")
        ax = axes[0]
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
        v = ~np.isnan(fpos)
        ax.plot(mids[v], np.array(fpos)[v], "b-o", ms=5, label="Model")
        ax.set_xlabel("Mean Predicted Prob"); ax.set_ylabel("Fraction Positives")
        ax.set_title(f"Calibration ({split})"); ax.legend(); ax.grid(True, alpha=0.3)
        ax = axes[1]
        ax.hist(y_score[y_true == 0], bins=30, alpha=0.6, label="Real",  color="blue")
        ax.hist(y_score[y_true == 1], bins=30, alpha=0.6, label="Fake",  color="red")
        ax.axvline(thr, color="k", linestyle="--", label=f"thr={thr:.3f}")
        ax.set_xlabel("P(Fake)"); ax.set_ylabel("Count")
        ax.set_title(f"Score Distribution ({split})"); ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout(); _save_fig(fig, out_dir, f"14_calib_{split}"); plt.close(fig)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# 17.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    cfg = CONFIG.copy()

    if NO_FACE_DET:
        cfg["face_det_backend"] = "none"
        cfg["use_face_cache"]   = False

    resume_from = RESUME_FROM
    mode        = MODE

    os.makedirs(cfg["output_dir"],     exist_ok=True)
    os.makedirs(cfg["face_cache_dir"], exist_ok=True)

    logger = setup_logging(cfg["output_dir"])
    set_seed(cfg["seed"])
    log_system_info(logger)
    save_config(cfg, os.path.join(cfg["output_dir"], "config.json"))

    logger.info(f"mode              : {mode}  |  resume: {resume_from}")
    logger.info(f"weights_only_init : {WEIGHTS_ONLY_INIT_FROM}")
    logger.info(f"face_det_backend  : {cfg['face_det_backend']}")
    logger.info(f"frames_per_clip   : {cfg['frames_per_clip']}")
    logger.info(f"batch_size        : {cfg['batch_size']}")
    logger.info(f"grad_accum_steps  : {cfg['grad_accum_steps']}  "
                f"→ effective_batch = {cfg['batch_size'] * cfg['grad_accum_steps']}")
    logger.info(f"base_lr (head)    : {cfg['learning_rate']}")
    logger.info(f"weight_decay      : {cfg['weight_decay']}")
    logger.info(f"dropout           : {cfg['dropout']}")
    logger.info(f"label_smoothing   : {cfg['label_smoothing']}")
    logger.info(f"focal_gamma       : {cfg['focal_gamma']}")
    logger.info(f"focal_alpha       : {cfg['focal_alpha']}  [FIX-14: was [1.0,1.05] in v3]")
    logger.info(f"stoch_depth_rate  : {cfg['stochastic_depth_rate']}")
    logger.info(f"use_mixup         : {cfg['use_mixup']}  (cutmix removed, [FIX-3])")
    logger.info(f"mixup_cutmix_prob : {cfg['mixup_cutmix_prob']}")
    logger.info(f"use_tta           : {cfg['use_tta']}")
    logger.info(f"tta_n_views       : {cfg.get('tta_n_views', 6)}")
    logger.info(f"threshold_metric  : {cfg.get('threshold_metric', 'balanced_acc')}")
    logger.info(f"THRESHOLD_EMA     : {THRESHOLD_EMA}  [FIX-2: was 0.5]")
    logger.info(f"use_swa           : {cfg.get('use_swa', False)}")
    logger.info(f"swa_start_frac    : {cfg.get('swa_start_frac', 0.65)}")
    logger.info(f"use_hf_aux        : {cfg.get('use_hf_aux', True)}  [FIX-5: was True]")
    logger.info(f"ema_decay_max     : {cfg.get('ema_decay_max', 0.999)}")
    logger.info(f"ema_warmup_epochs : {cfg.get('ema_decay_warmup_epochs', 5)}")
    logger.info(f"num_epochs        : {cfg['num_epochs']}")
    logger.info(f"total_budget      : {timedelta(seconds=int(cfg['total_budget_sec']))}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timer  = Timer()

    fail_log = os.path.join(cfg["output_dir"], "decode_failures.txt")
    try:
        fail_log_lock: Optional[Any] = multiprocessing.Lock()
    except Exception:
        fail_log_lock = None
        logger.warning("Could not create multiprocessing.Lock.")

    train_list, val_list, test_list = discover_videos(cfg, logger)

    n_real = sum(1 for _, l in train_list if l == 0)
    n_fake = sum(1 for _, l in train_list if l == 1)
    logger.info(
        f"WeightedRandomSampler active for class-balance  (n_real={n_real}, n_fake={n_fake})."
    )
    logger.info(
        f"focal_alpha={cfg['focal_alpha']} — small nudge only; the fixed sampler "
        f"[FIX-13] does the heavy lifting on class balance."
    )

    tr_tfm = ClipTransform(mode="train", size=cfg["input_size"], cfg=cfg)
    ev_tfm = ClipTransform(mode="val",   size=cfg["input_size"], cfg=cfg, tta_view=0)

    def make_ds(
        video_list:    List[Tuple[str, int]],
        tfm:           ClipTransform,
        clips_per:     int,
        deterministic: bool = False,
    ) -> FFPPVideoDataset:
        return FFPPVideoDataset(
            video_list, tfm,
            num_frames      = cfg["frames_per_clip"],
            frame_stride    = cfg["frame_stride"],
            clips_per_video = clips_per,
            face_backend    = cfg["face_det_backend"],
            face_margin     = cfg["face_margin"],
            target_size     = cfg["input_size"],
            fail_log        = fail_log,
            fail_log_lock   = fail_log_lock,
            face_cache_dir  = cfg.get("face_cache_dir"),
            use_face_cache  = cfg.get("use_face_cache", False),
            deterministic   = deterministic,
        )

    train_ds = make_ds(train_list, tr_tfm, cfg["clips_per_video_train"], deterministic=False)
    val_ds   = make_ds(val_list,   ev_tfm, cfg["clips_per_video_val"],   deterministic=True)
    test_ds  = make_ds(test_list,  ev_tfm, cfg["clips_per_video_val"],   deterministic=True)
    val_ds_epoch = make_ds(
        val_list, ev_tfm, cfg["clips_per_video_val_epoch"], deterministic=True
    )

    logger.info(
        f"Dataset sizes — train_ds={len(train_ds)}  "
        f"val_ds={len(val_ds)} (final-eval, {cfg['clips_per_video_val']} clip/video)  "
        f"val_ds_epoch={len(val_ds_epoch)} (per-epoch, {cfg['clips_per_video_val_epoch']} clip/video)  "
        f"test_ds={len(test_ds)}"
    )

    nw      = cfg["num_workers"]
    use_pin = cfg["pin_memory"] and torch.cuda.is_available()
    train_sampler = make_weighted_sampler(
        train_ds, real_folder=cfg["real_folder"], fake_folders=cfg["fake_folders"],
        logger=logger,
    )

    real_clips = sum(1 for _, l, _w in train_ds.samples if l == 0)
    fake_clips = sum(1 for _, l, _w in train_ds.samples if l == 1)
    logger.info(
        f"Clip class balance (raw counts, pre-sampler) — Real:{real_clips}  Fake:{fake_clips}  "
        f"ratio:{fake_clips / max(real_clips, 1):.2f}:1"
    )
    bucket_counts_log: Dict[str, int] = {}
    for p, _l, _w in train_ds.samples:
        b = _manipulation_bucket(p, cfg["real_folder"], cfg["fake_folders"])
        bucket_counts_log[b] = bucket_counts_log.get(b, 0) + 1
    logger.info(
        f"[FIX-13] Per-manipulation-method clip counts (Fake methods stratified "
        f"WITHIN a 50/50 Real/Fake top-level split): {bucket_counts_log}"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size         = cfg["batch_size"],
        sampler            = train_sampler,
        num_workers        = nw,
        pin_memory         = use_pin,
        drop_last          = True,
        persistent_workers = (nw > 0),
        prefetch_factor    = 2 if nw > 0 else None,
        worker_init_fn     = mute_worker_stderr,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size         = cfg["batch_size"],
        shuffle            = False,
        num_workers        = nw,
        pin_memory         = use_pin,
        persistent_workers = (nw > 0),
        prefetch_factor    = 2 if nw > 0 else None,
        worker_init_fn     = mute_worker_stderr,
    )
    val_loader_epoch = DataLoader(
        val_ds_epoch,
        batch_size         = cfg["batch_size"],
        shuffle            = False,
        num_workers        = nw,
        pin_memory         = use_pin,
        persistent_workers = (nw > 0),
        prefetch_factor    = 2 if nw > 0 else None,
        worker_init_fn     = mute_worker_stderr,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size         = cfg["batch_size"],
        shuffle            = False,
        num_workers        = nw,
        pin_memory         = use_pin,
        persistent_workers = (nw > 0),
        prefetch_factor    = 2 if nw > 0 else None,
        worker_init_fn     = mute_worker_stderr,
    )
    logger.info(
        f"Batches — train:{len(train_loader)}  "
        f"val(final-eval,3clip):{len(val_loader)}  "
        f"val_epoch(per-epoch,1clip):{len(val_loader_epoch)}  "
        f"test:{len(test_loader)}"
    )

    # ── Build backbone ────────────────────────────────────────────────────────
    use_backend = cfg.get("model_backend", "custom")
    _using_torchvision = False

    if use_backend == "torchvision" and _TORCHVISION_SWIN_OK:
        logger.info("Building torchvision Swin3D_T with Kinetics400 weights.")
        swin_backbone = swin3d_t(weights=Swin3D_T_Weights.KINETICS400_V1)
        if hasattr(swin_backbone.head, "in_features"):
            in_features = swin_backbone.head.in_features
        elif hasattr(swin_backbone.head, "__getitem__") and \
                hasattr(swin_backbone.head[-1], "in_features"):
            in_features = swin_backbone.head[-1].in_features
        else:
            in_features = 768
        swin_backbone.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(cfg["dropout"] * 0.6),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(cfg["dropout"] * 0.4),
            nn.Linear(256, cfg["num_classes"]),
        )
        logger.info(
            f"Swin3D_T head replaced: in_features={in_features}, "
            f"dropout={cfg['dropout']}, num_classes={cfg['num_classes']}"
        )
        _using_torchvision = True
    else:
        if use_backend == "torchvision" and not _TORCHVISION_SWIN_OK:
            logger.warning(
                "torchvision Swin3D_T unavailable — falling back to custom VideoSwinTiny."
            )
        logger.info("Building custom VideoSwinTiny.")
        swin_backbone = VideoSwinTiny(
            num_classes           = cfg["num_classes"],
            dropout               = cfg["dropout"],
            use_grad_checkpoint   = cfg.get("use_grad_checkpoint", False),
            stochastic_depth_rate = cfg.get("stochastic_depth_rate", 0.20),
        )
        if cfg["pretrained_weights"] == "kinetics":
            load_pretrained_kinetics(swin_backbone, logger)

    # ── Wrap backbone in DeepfakeDetector (+optional HF aux head) ────────────
    hf_head = None
    if cfg.get("use_hf_aux", False):
        hf_head = LaplacianHFHead(
            num_classes = cfg["num_classes"],
            dropout     = cfg["dropout"] * 0.7,
        )
        logger.info(
            f"LaplacianHFHead enabled  "
            f"hf_aux_weight={cfg.get('hf_aux_weight', 0.20):.2f}"
        )
    else:
        logger.info("LaplacianHFHead disabled [FIX-5].")

    detector_model = DeepfakeDetector(
        swin_model = swin_backbone,
        hf_head    = hf_head,
        hf_weight  = cfg.get("hf_aux_weight", 0.20),
    ).to(device)

    # ── Weights-only warm start from a prior checkpoint ───────────────────────
    if WEIGHTS_ONLY_INIT_FROM and os.path.isfile(WEIGHTS_ONLY_INIT_FROM) and not (
        resume_from and os.path.isfile(resume_from)
    ):
        try:
            load_weights_only(WEIGHTS_ONLY_INIT_FROM, detector_model, logger)
        except Exception as exc:
            logger.warning(f"Weights-only warm start failed: {exc}. Using fresh init.")

    # ── EMA model ─────────────────────────────────────────────────────────────
    ema_model = copy.deepcopy(detector_model)
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)
    logger.info(
        f"EMA model created — dynamic decay "
        f"0.90 → {cfg.get('ema_decay_max', 0.999)} "
        f"over {cfg.get('ema_decay_warmup_epochs', 5)} epochs"
    )

    # ── DataParallel ──────────────────────────────────────────────────────────
    if torch.cuda.device_count() > 1:
        detector_model = nn.DataParallel(detector_model)
        logger.info(f"DataParallel across {torch.cuda.device_count()} GPUs.")

    total_params     = sum(p.numel() for p in detector_model.parameters())
    trainable_params = sum(p.numel() for p in detector_model.parameters()
                           if p.requires_grad)
    logger.info(
        f"Model — total params:{total_params:,}  trainable:{trainable_params:,}"
    )

    # ── FocalLoss with [FIX-14] alpha ───────────────────────────────────────
    criterion = FocalLoss(
        gamma           = cfg["focal_gamma"],
        alpha           = cfg["focal_alpha"],
        num_classes     = cfg["num_classes"],
        label_smoothing = cfg.get("label_smoothing", 0.0),
    ).to(device)
    logger.info(
        f"FocalLoss  γ={cfg['focal_gamma']}  α={cfg['focal_alpha']}  "
        f"label_smoothing={cfg.get('label_smoothing', 0.0)}  "
        f"(pt from softmax; α normalised to "
        f"{[round(x/sum(cfg['focal_alpha']),3) for x in cfg['focal_alpha']]})"
    )

    # ── Optimizer ─────────────────────────────────────────────────────────────
    base_detector = (
        detector_model.module
        if isinstance(detector_model, nn.DataParallel)
        else detector_model
    )
    if _using_torchvision:
        optimizer = build_optimizer_torchvision(base_detector, cfg, logger)
    else:
        optimizer = build_optimizer_with_llrd(base_detector, cfg, logger)

    _llrd_ratios: Dict[str, float] = {
        g.get("name", f"group_{i}"): (g["lr"] / cfg["learning_rate"] if cfg["learning_rate"] > 0 else 1.0)
        for i, g in enumerate(optimizer.param_groups)
    }

    # ── AMP GradScaler ────────────────────────────────────────────────────────
    try:
        scaler = GradScaler(
            device          = _AMP_DEVICE,
            enabled         = cfg["amp"],
            init_scale      = 2 ** 16,
            growth_factor   = 2.0,
            backoff_factor  = 0.5,
            growth_interval = 2000,
        )
        logger.info("GradScaler: torch.amp new API (device='cuda')")
    except TypeError:
        scaler = GradScaler(  # type: ignore[call-arg]
            enabled         = cfg["amp"],
            init_scale      = 2 ** 16,
            growth_factor   = 2.0,
            backoff_factor  = 0.5,
            growth_interval = 2000,
        )
        logger.info("GradScaler: torch.cuda.amp legacy API")

    # ── Scheduler ─────────────────────────────────────────────────────────────
    use_swa       = cfg.get("use_swa", True)
    swa_start_ep  = max(1, int(cfg["num_epochs"] * cfg.get("swa_start_frac", 0.65)))
    effective_steps_per_epoch = math.ceil(
        len(train_loader) / cfg["grad_accum_steps"]
    )
    scheduler = build_scheduler(
        optimizer,
        cfg,
        effective_steps_per_epoch,
        swa_start_epoch=swa_start_ep if use_swa else None,
    )
    logger.info(
        f"Scheduler: effective_steps/epoch={effective_steps_per_epoch}  "
        f"warmup_steps={cfg['warmup_epochs'] * effective_steps_per_epoch}  "
        f"cosine_until_epoch={swa_start_ep if use_swa else cfg['num_epochs']}"
    )
    if use_swa:
        swa_model = SWAModel(base_detector)
        logger.info(
            f"SWA enabled — starts at epoch {swa_start_ep}  "
            f"(= {int(cfg.get('swa_start_frac',0.65)*100)}% of {cfg['num_epochs']} epochs)  "
            f"lr={cfg.get('swa_lr', 2e-6):.1e}"
        )
    else:
        swa_model = None

    # ── Training state ────────────────────────────────────────────────────────
    start_epoch       = 1
    best_auc          = 0.0
    best_f1           = 0.0
    history: List[dict] = []
    optimal_threshold = 0.5
    epochs_no_improve = 0
    wallclock_prior_sec = 0.0

    # ── Auto-resume ───────────────────────────────────────────────────────────
    _auto_resume = os.path.join(cfg["output_dir"], "checkpoint_latest.pth")
    if resume_from is None and os.path.isfile(_auto_resume):
        resume_from = _auto_resume
        logger.info(f"Auto-resuming from {_auto_resume}")

    if resume_from and os.path.isfile(resume_from):
        logger.info(f"[RESUME] Loading checkpoint: {resume_from}")
        ckpt = load_ckpt(resume_from, detector_model, optimizer, scheduler, scaler)

        start_epoch       = ckpt.get("epoch", 0) + 1
        best_auc          = ckpt.get("best_auc", 0.0)
        best_f1           = ckpt.get("best_f1",  0.0)
        history           = ckpt.get("history",  [])
        optimal_threshold = ckpt.get("optimal_threshold", 0.5)
        epochs_no_improve = ckpt.get("epochs_no_improve", 0)

        wallclock_prior_sec = ckpt.get("total_wallclock_sec", 0.0)

        if "ema_model_state" in ckpt:
            ema_model.load_state_dict(ckpt["ema_model_state"])
            logger.info("  [RESUME] Restored EMA model weights.")
        else:
            ema_model.load_state_dict(
                (detector_model.module
                 if isinstance(detector_model, nn.DataParallel)
                 else detector_model).state_dict()
            )
            logger.info("  [RESUME] No ema_model_state — synced EMA from model.")

        if "swa_state" in ckpt and swa_model is not None:
            swa_model.load_state_dict(ckpt["swa_state"])
            swa_model.to(device)
            logger.info(f"  [RESUME] Restored SWA state (n={swa_model.n}) and moved avg_state to {device}.")
        elif swa_model is not None:
            swa_model._init(base_detector)
            logger.info("  [RESUME] No swa_state in checkpoint — reinitialized SWA average from resumed weights.")

        for key, setter in [
            ("rng_state",        torch.set_rng_state),
            ("np_rng_state",     np.random.set_state),
            ("python_rng_state", random.setstate),
        ]:
            if key in ckpt:
                try:
                    setter(ckpt[key])
                except Exception:
                    pass

        collapse_reset_done = ckpt.get("collapse_reset_done", False)

        logger.info(
            f"[RESUME] Continuing from epoch {start_epoch}  "
            f"best_auc={best_auc:.4f}  threshold={optimal_threshold:.4f}  "
            f"epochs_no_improve={epochs_no_improve}  "
            f"wallclock_prior={timedelta(seconds=int(wallclock_prior_sec))}"
        )
        logger.info(
            f"[RESUME] History has {len(history)} epoch(s) — "
            f"will continue appending to training_history.csv"
        )
    elif resume_from and not os.path.isfile(resume_from):
        logger.warning(
            f"[RESUME] Checkpoint not found at '{resume_from}'. Starting fresh."
        )
        collapse_reset_done = False
    else:
        collapse_reset_done = False

    # ── Eval-only mode ────────────────────────────────────────────────────────

    if mode == "eval":
        logger.info("=== EVALUATION MODE ===")
        ema_model.hf_weight = cfg.get("hf_aux_weight", 0.20)
        ema_model = ema_model.to(device)

        if torch.cuda.device_count() > 1:
            eval_model = nn.DataParallel(ema_model)
            logger.info(f"[EVAL] Wrapped eval model across {torch.cuda.device_count()} GPUs.")
        else:
            eval_model = ema_model
        if cfg.get("use_tta", True):
            logger.info("[RECALIBRATE] Computing TTA-averaged validation scores "
                        "to recalibrate threshold before test eval ...")
            val_tta_met, (vn, vl_labels, vl_probs) = evaluate_with_tta(
                eval_model, val_list, criterion, device, cfg, logger,
                threshold=optimal_threshold,
                n_views=cfg.get("tta_n_views", 6),
            )
            recalibrated_thr = find_optimal_threshold(
                vl_labels, vl_probs, metric=cfg.get("threshold_metric", "balanced_acc")
            )
            logger.info(
                f"[RECALIBRATE] old_thr(single-view)={optimal_threshold:.4f}  "
                f"new_thr(TTA-val)={recalibrated_thr:.4f}  "
                f"val_tta_auc={val_tta_met.get('roc_auc', float('nan')):.4f}"
            )
            optimal_threshold = recalibrated_thr
            met, (ns, ls, ps) = evaluate_with_tta(
                eval_model, test_list, criterion, device, cfg, logger,
                threshold=optimal_threshold,
                n_views=cfg.get("tta_n_views", 6),
            )
        else:
            met, (ns, ls, ps) = validate(
                eval_model, test_loader, criterion, device, cfg, logger,
                return_preds=True, desc="Test evaluation",
                threshold=optimal_threshold,
            )
        for k, v in met.items():
            logger.info(f"  {k}: {v}")
        write_predictions_csv(
            os.path.join(cfg["output_dir"], "test_predictions.csv"),
            ns, ls, ps, thr=optimal_threshold,
        )
        plot_eval(ls, ps, cfg["output_dir"], thr=optimal_threshold)
        return

    # ── Training loop ─────────────────────────────────────────────────────────
    hist_csv = os.path.join(cfg["output_dir"], "training_history.csv")
    if start_epoch == 1:
        append_history_csv(hist_csv, {}, write_header=True)

    logger.info(f"Training epochs {start_epoch}–{cfg['num_epochs']}")

    epoch_times: List[float]   = []
    collapse_patience          = cfg.get("collapse_patience", 3)
    consecutive_collapse       = 0
    amp_low_scale_count        = 0
    AMP_LOW_SCALE_THRESH       = 1.0
    AMP_LOW_SCALE_PATIENCE     = 3
    threshold_metric           = cfg.get("threshold_metric", "balanced_acc")

    if history:
        for h in history[-collapse_patience:]:
            if h.get("specificity", 1.0) == 0.0:
                consecutive_collapse += 1
            else:
                consecutive_collapse = 0

    swa_active = False
    if start_epoch > swa_start_ep:
        swa_active = True
        logger.info(f"[SWA] Already active from resume (start_ep={swa_start_ep}).")

    def _build_ckpt_state(epoch_: int, val_metrics_: dict) -> dict:
        state = dict(
            epoch               = epoch_,
            model_state         = detector_model.state_dict(),
            ema_model_state     = ema_model.state_dict(),
            optimizer_state     = optimizer.state_dict(),
            scheduler_state     = scheduler.state_dict(),
            scaler_state        = scaler.state_dict(),
            best_auc            = best_auc,
            best_f1             = best_f1,
            val_metrics         = val_metrics_,
            config              = cfg,
            history             = history,
            rng_state           = torch.get_rng_state(),
            np_rng_state        = np.random.get_state(),
            python_rng_state    = random.getstate(),
            optimal_threshold   = optimal_threshold,
            epochs_no_improve   = epochs_no_improve,
            collapse_reset_done = collapse_reset_done,

            total_wallclock_sec = wallclock_prior_sec + timer.elapsed(),
        )
        if swa_model is not None:
            state["swa_state"] = swa_model.state_dict()
        return state

    training_complete = False

    for epoch in range(start_epoch, cfg["num_epochs"] + 1):
        elapsed = timer.elapsed()

        if len(epoch_times) >= 3:
            sorted_times = sorted(epoch_times)
            avg_ep = sorted_times[len(sorted_times) // 2]
        else:
            avg_ep = (sum(epoch_times) / len(epoch_times)) if epoch_times else 1800.0

        remaining_budget = cfg["total_budget_sec"] - elapsed
        if elapsed + avg_ep + cfg["eval_reserve_sec"] > cfg["total_budget_sec"]:
            logger.warning(
                f"[BUDGET] Epoch {epoch} would exceed budget "
                f"(elapsed={timedelta(seconds=int(elapsed))}, "
                f"median_epoch={timedelta(seconds=int(avg_ep))}, "
                f"remaining={timedelta(seconds=int(remaining_budget))}) — stopping."
            )
            break

        est_left = int((remaining_budget - cfg["eval_reserve_sec"]) / max(avg_ep, 1))
        logger.info(f"\n{'=' * 62}")
        logger.info(
            f"  EPOCH {epoch}/{cfg['num_epochs']}  |  "
            f"elapsed={timedelta(seconds=int(elapsed))}  |  "
            f"remaining={timer.eta_str(cfg['total_budget_sec'])}  |  "
            f"~{est_left} epochs left"
        )

        current_ema_decay = compute_ema_decay(epoch, cfg)
        current_hf_weight = compute_hf_weight(epoch, cfg)
        base_detector.hf_weight = current_hf_weight
        ema_model.hf_weight     = current_hf_weight
        logger.info(
            f"  median_epoch_time={timedelta(seconds=int(avg_ep))}  |  "
            f"optimal_thr={optimal_threshold:.4f}  |  "
            f"thr_metric={threshold_metric}  |  "
            f"ema_decay={current_ema_decay:.4f}  |  "
            f"hf_weight={current_hf_weight:.4f}"
        )

        if use_swa and epoch >= swa_start_ep:
            if not swa_active:
                swa_active = True
                logger.info(f"[SWA] Phase begins at epoch {epoch}.")

            swa_head_lr_max = cfg.get("swa_lr", 2e-5)
            swa_head_lr_min = swa_head_lr_max * 0.3
            cycle_len   = 2
            cycle_frac  = ((epoch - swa_start_ep) % cycle_len) / max(cycle_len - 1, 1)
            swa_head_lr = swa_head_lr_min + (swa_head_lr_max - swa_head_lr_min) * (
                0.5 * (1 + math.cos(math.pi * cycle_frac))
            )
            for g in optimizer.param_groups:
                ratio  = _llrd_ratios.get(g.get("name", ""), 1.0)
                g["lr"] = swa_head_lr * ratio
            logger.info(f"[SWA-LR] epoch={epoch}  cycle_frac={cycle_frac:.2f}  head_lr={swa_head_lr:.2e}")

        ep_start = time.time()
        tr_met   = train_one_epoch(
            detector_model, train_loader, criterion, optimizer, scaler,
            scheduler, device, cfg, epoch, logger, timer,
            is_first_epoch=(epoch == start_epoch),
            ema_model=ema_model,
            ema_decay=current_ema_decay,
            skip_scheduler_step=swa_active,  # [FIX-20]
        )

        train_only_time = time.time() - ep_start
        logger.info(f"  [TIMING] train_only={timedelta(seconds=int(train_only_time))}")

        if swa_active and swa_model is not None:
            swa_model.update(base_detector)
            logger.info(f"  [SWA] Updated average (n={swa_model.n}).")

        current_scale = scaler.get_scale()
        logger.info(f"  AMP scale: {current_scale:.0f}")
        if cfg["amp"] and current_scale < AMP_LOW_SCALE_THRESH:
            amp_low_scale_count += 1
            logger.warning(
                f"  AMP scale={current_scale:.2f} below threshold "
                f"({amp_low_scale_count}/{AMP_LOW_SCALE_PATIENCE})."
            )
            if amp_low_scale_count >= AMP_LOW_SCALE_PATIENCE:
                cfg["amp"] = False
                logger.error("  Disabling AMP due to persistently low scale.")
        else:
            amp_low_scale_count = 0

        vl_met, (vnames, vlabels, vprobs) = validate(
            ema_model, val_loader_epoch, criterion, device, cfg, logger,
            return_preds=True,
            desc=f"Epoch {epoch}/{cfg['num_epochs']} [val/EMA, 1-clip]",
            threshold=optimal_threshold,
        )

        full_epoch_time = time.time() - ep_start
        epoch_times.append(full_epoch_time)
        logger.info(
            f"  [TIMING] train_only={timedelta(seconds=int(train_only_time))}  "
            f"full_epoch(train+validate)={timedelta(seconds=int(full_epoch_time))}"
        )
        saved_val_loss = vl_met.get("val_loss", float("nan"))

        if cfg.get("use_optimal_threshold", True) and SKLEARN_OK:
            new_thr = find_optimal_threshold(
                vlabels, vprobs, metric=threshold_metric
            )
            met_old = compute_metrics(vlabels, vprobs, thr=optimal_threshold)
            met_new = compute_metrics(vlabels, vprobs, thr=new_thr)

            score_key = (
                "mcc"          if threshold_metric == "mcc"          else
                "balanced_acc" if threshold_metric == "balanced_acc" else "f1"
            )
            old_score = met_old.get(score_key, 0.0)
            new_score = met_new.get(score_key, 0.0)

            if new_score >= old_score - 0.005:
                # [FIX-2] THRESHOLD_EMA=0.85 now — sticky by design
                optimal_threshold = (
                    THRESHOLD_EMA * optimal_threshold
                    + (1 - THRESHOLD_EMA) * new_thr
                )
                optimal_threshold = float(np.clip(optimal_threshold, 0.10, 0.90))

            logger.info(
                f"  [THRESHOLD] metric={threshold_metric}  "
                f"new_thr_score={new_score:.4f}  "
                f"selected_thr={optimal_threshold:.4f}"
            )

            vl_met = compute_metrics(vlabels, vprobs, thr=optimal_threshold)
            vl_met["val_loss"] = saved_val_loss
            vl_met.setdefault("clip_auc", float("nan"))

        _current_epoch_auc = vl_met.get("roc_auc", float("nan"))
        if epoch >= 10 and not math.isnan(_current_epoch_auc) \
                and _current_epoch_auc < cfg["target_teacher_auc"] \
                and len(history) >= 3:
            _auc_3ago = history[-3].get("roc_auc", float("nan"))
            if not math.isnan(_auc_3ago):
                _auc_improve_3ep = _current_epoch_auc - _auc_3ago
                if _auc_improve_3ep < 0.01:
                    logger.error(
                        f"[DECISION-GATE] Epoch {epoch}: val AUC={_current_epoch_auc:.4f} "
                        f"still below target_teacher_auc={cfg['target_teacher_auc']:.2f}, "
                        f"and improved only {_auc_improve_3ep:+.4f} over the last 3 epochs "
                        f"(< 0.01). Recommend stopping and using this checkpoint as-is for "
                        f"distillation rather than continuing to chase AUC on this "
                        f"secondary teacher. Training will continue (this is advisory) — "
                        f"decide manually."
                    )

        _cumulative_wallclock_sec = wallclock_prior_sec + timer.elapsed()
        _budget_cap_sec = cfg["max_sessions_budget_hours"] * 3600.0
        if _cumulative_wallclock_sec > _budget_cap_sec:
            logger.error(
                f"[BUDGET-GATE] Cumulative wall-clock across sessions "
                f"({timedelta(seconds=int(_cumulative_wallclock_sec))}) has exceeded "
                f"max_sessions_budget_hours={cfg['max_sessions_budget_hours']:.1f}h "
                f"({timedelta(seconds=int(_budget_cap_sec))}). Recommend stopping "
                f"regardless of current AUC — decide manually."
            )

        logger.info(
            f"  Train loss={tr_met['train_loss']:.4f}  "
            f"acc={tr_met['train_acc']:.4f}  "
            f"bal_acc={tr_met['train_balanced_acc']:.4f}  "
            f"auc={tr_met.get('train_auc', float('nan')):.4f}"
        )
        logger.info(
            f"  Val   loss={vl_met.get('val_loss', float('nan')):.4f}  "
            f"auc={vl_met.get('roc_auc', float('nan')):.4f}  "
            f"f1={vl_met.get('f1', float('nan')):.4f}  "
            f"mcc={vl_met.get('mcc', float('nan')):.4f}  "
            f"bal_acc={vl_met.get('balanced_acc', float('nan')):.4f}  "
            f"spec={vl_met.get('specificity', float('nan')):.4f}  "
            f"rec={vl_met.get('recall', float('nan')):.4f}  "
            f"eer={vl_met.get('eer', float('nan')):.4f}  "
            f"thr={optimal_threshold:.4f}"
        )

        train_auc_val = tr_met.get("train_auc", float("nan"))
        val_auc_val   = vl_met.get("roc_auc",   float("nan"))
        if not math.isnan(train_auc_val) and not math.isnan(val_auc_val):
            gap = train_auc_val - val_auc_val
            msg = f"  Overfitting gap (train_auc - val_auc) = {gap:.4f}"
            if gap > 0.08:
                logger.warning(msg + "  [OVERFIT: consider stronger aug/dropout]")
            elif gap < 0.01 and val_auc_val < 0.92:
                logger.warning(
                    msg + "  [UNDERFIT: train_auc barely above val_auc while both "
                          "are still low — if this persists past epoch ~8-10, "
                          "double-check the FIX-13 sampler log line above shows "
                          "Real/Fake mass close to 0.500/0.500 before assuming "
                          "the recipe/architecture is the bottleneck]"
                )
            else:
                logger.info(msg)

        lr_report = "  LR per group: " + "  ".join(
            f"{g.get('name', f'g{j}')}={g['lr']:.2e}"
            for j, g in enumerate(optimizer.param_groups)
        )
        logger.info(lr_report)

        current_spec   = vl_met.get("specificity",  float("nan"))
        current_recall = vl_met.get("recall",        float("nan"))

        is_all_fake = not math.isnan(current_spec)   and current_spec   < 0.15
        is_all_real = not math.isnan(current_recall) and current_recall < 0.05

        if is_all_fake or is_all_real:
            consecutive_collapse += 1
            direction = "all-Fake (spec<0.15)" if is_all_fake else "all-Real (recall<0.05)"
            logger.warning(
                f"  MODE COLLAPSE: {direction}  "
                f"({consecutive_collapse}/{collapse_patience})"
            )
        else:
            consecutive_collapse = 0

        if consecutive_collapse >= collapse_patience:
            if not collapse_reset_done:
                collapse_reset_done = True
                logger.warning("MODE COLLAPSE — attempting LR reset (×3) ...")
                for g in optimizer.param_groups:
                    g["lr"] = g["lr"] * 3.0
                consecutive_collapse = 0
                try:
                    save_ckpt(
                        _build_ckpt_state(epoch, vl_met),
                        cfg["output_dir"], epoch=epoch,
                        best_auc=False, best_f1=False,
                        prev_epoch=epoch - 1 if epoch > start_epoch else None,
                    )
                except Exception as exc:
                    logger.warning(f"Could not checkpoint before collapse-reset continue: {exc}")
                try:
                    plot_curves(history, cfg["output_dir"])
                except Exception:
                    pass
                continue
            else:
                logger.error("MODE COLLAPSE PERSISTS after LR reset — stopping.")
                training_complete = True   # [FIX-8] genuine stop, not a budget cutoff
                try:
                    plot_curves(history, cfg["output_dir"])
                except Exception:
                    pass
                break

        if epoch >= 7 and vl_met.get("roc_auc", 1.0) < 0.55:
            logger.error(
                f"[EARLY_ABORT] Val AUC={vl_met.get('roc_auc'):.4f} < 0.55 "
                f"after epoch {epoch}. Stopping."
            )

            try:
                plot_curves(history, cfg["output_dir"])
            except Exception:
                pass
            break

        current_auc      = vl_met.get("roc_auc", 0.0)
        current_f1       = vl_met.get("f1",      0.0)
        best_auc_before  = best_auc
        is_best_auc      = (not math.isnan(current_auc)) and current_auc > best_auc
        is_best_f1       = (not math.isnan(current_f1))  and current_f1  > best_f1

        if is_best_auc:
            best_auc = current_auc
            logger.info(f"  ★ New best AUC {best_auc:.4f} → best_teacher_model.pth")
        if is_best_f1:
            best_f1 = current_f1
            logger.info(f"  ★ New best F1  {best_f1:.4f}  (tracked only)")

        early_stop_patience = cfg.get("early_stop_patience",  10)
        min_delta           = cfg.get("early_stop_min_delta", 0.0)
        # Don't let noisy per-epoch AUC during the SWA cyclic-LR phase trigger
        # early stopping — SWA's benefit only shows up in the *averaged* weights.
        in_swa_phase = swa_active
        meaningful_improvement = (
            not math.isnan(current_auc) and current_auc > best_auc_before + min_delta
        ) or in_swa_phase
        if meaningful_improvement:
            epochs_no_improve = 0
        elif not math.isnan(current_auc):
            epochs_no_improve += 1
            logger.info(
                f"  No AUC improvement ≥{min_delta}: {epochs_no_improve}/{early_stop_patience} "
                f"(best={best_auc:.4f}, current={current_auc:.4f})"
            )
            if epochs_no_improve >= early_stop_patience:
                logger.info(f"  Early stopping at epoch {epoch}.")
                training_complete = True   # [FIX-8] genuine stop, not a budget cutoff
                break

        row = {
            "epoch":             epoch,
            **tr_met,
            "val_loss":          vl_met.get("val_loss",      float("nan")),
            "val_acc":           vl_met.get("accuracy",      float("nan")),
            "roc_auc":           vl_met.get("roc_auc",       float("nan")),
            "pr_auc":            vl_met.get("pr_auc",        float("nan")),
            "f1":                vl_met.get("f1",            float("nan")),
            "precision":         vl_met.get("precision",     float("nan")),
            "recall":            vl_met.get("recall",        float("nan")),
            "specificity":       vl_met.get("specificity",   float("nan")),
            "balanced_acc":      vl_met.get("balanced_acc",  float("nan")),
            "mcc":               vl_met.get("mcc",           float("nan")),
            "eer":               vl_met.get("eer",           float("nan")),
            "optimal_threshold": optimal_threshold,
            "ema_decay":         current_ema_decay,
            "swa_active":        int(swa_active),
        }
        history.append(row)
        append_history_csv(hist_csv, row)

        ckpt_state = _build_ckpt_state(epoch, vl_met)

        save_ckpt(
            ckpt_state,
            cfg["output_dir"],
            epoch      = epoch,
            best_auc   = is_best_auc,
            best_f1    = is_best_f1,
            prev_epoch = epoch - 1 if epoch > start_epoch else None,
        )

        try:
            plot_curves(history, cfg["output_dir"])
        except Exception as exc:
            logger.warning(f"plot_curves failed: {exc}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:

        training_complete = True
        logger.info(f"[FIX-8] Full schedule ({cfg['num_epochs']} epochs) completed — "
                    f"running final evaluation.")

    if not training_complete:

        logger.info(
            "[FIX-8] Session ended via budget cutoff, not full completion — "
            "skipping the ~60-70min final TTA evaluation. Re-run to resume "
            "from checkpoint_latest.pth and continue the schedule."
        )
        logger.info(f"\nTotal time : {timedelta(seconds=int(timer.elapsed()))}")
        logger.info(f"Best val AUC so far: {best_auc:.6f}  |  Best val F1 so far: {best_f1:.6f}")
        return

    # ── Apply SWA weights before final evaluation ─────────────────────────────
    base_detector.hf_weight = cfg.get("hf_aux_weight", 0.20)
    ema_model.hf_weight     = cfg.get("hf_aux_weight", 0.20)
    if swa_active and swa_model is not None and swa_model.n > 0:
        logger.info(
            f"[SWA] Applying averaged weights from {swa_model.n} checkpoints "
            f"to base model and syncing to EMA model for test eval."
        )
        swa_model.apply(base_detector)
        ema_model.load_state_dict(base_detector.state_dict())
        bn_layers = [m for m in base_detector.modules() if isinstance(m, nn.BatchNorm2d)]
        if bn_layers:
            logger.info("[SWA] Updating BatchNorm stats on train set ...")
            base_detector.train()
            with torch.no_grad():
                for clips, _, _ in train_loader:
                    clips = clips.to(device, non_blocking=True)
                    base_detector(clips)
            base_detector.eval()
            ema_model.load_state_dict(base_detector.state_dict())
            logger.info("[SWA] BatchNorm stats updated.")

    # ── Final test evaluation ─────────────────────────────────────────────────
    logger.info("\n" + "=" * 62)
    logger.info("FINAL TEST EVALUATION")

    best_ckpt_path = os.path.join(cfg["output_dir"], "best_teacher_model.pth")
    if os.path.exists(best_ckpt_path) and not swa_active:
        logger.info("Loading best_teacher_model.pth for test evaluation.")
        try:
            best_ckpt         = torch.load(
                best_ckpt_path, map_location="cpu", weights_only=False
            )
            optimal_threshold = best_ckpt.get("optimal_threshold", optimal_threshold)
            if "ema_model_state" in best_ckpt:
                ema_model.load_state_dict(best_ckpt["ema_model_state"])
                logger.info("  Loaded EMA weights from best checkpoint.")
            else:
                load_ckpt(best_ckpt_path, detector_model)
                ema_model.load_state_dict(base_detector.state_dict())
            ema_model = ema_model.to(device)
            logger.info(f"  Using threshold from best checkpoint: {optimal_threshold:.4f}")
        except Exception as exc:
            logger.warning(
                f"Could not load best_teacher_model.pth: {exc}. "
                "Using current EMA/SWA weights."
            )

    ema_model = ema_model.to(device)
    if torch.cuda.device_count() > 1:
        eval_model = nn.DataParallel(ema_model)
        logger.info(f"[EVAL] Wrapped eval model across {torch.cuda.device_count()} GPUs.")
    else:
        eval_model = ema_model
    eval_model.eval()

    if cfg.get("use_tta", True):
        n_tta = cfg.get("tta_n_views", 6)
        logger.info("[RECALIBRATE] Computing TTA-averaged validation scores "
                    "to recalibrate threshold before final test eval ...")
        val_tta_met, (vn, vl_labels, vl_probs) = evaluate_with_tta(
            eval_model, val_list, criterion, device, cfg, logger,
            threshold=optimal_threshold,
            n_views=n_tta,
        )
        recalibrated_thr = find_optimal_threshold(
            vl_labels, vl_probs, metric=cfg.get("threshold_metric", "balanced_acc")
        )
        logger.info(
            f"[RECALIBRATE] old_thr(single-view-EMA)={optimal_threshold:.4f}  "
            f"new_thr(TTA-val)={recalibrated_thr:.4f}  "
            f"val_tta_auc={val_tta_met.get('roc_auc', float('nan')):.4f}  "
            f"val_tta_bal_acc(old_thr)={val_tta_met.get('balanced_acc', float('nan')):.4f}"
        )
        optimal_threshold = recalibrated_thr

        logger.info(f"Running TTA with {n_tta} views on test set ...")
        te_met, (ns, ls, ps) = evaluate_with_tta(
            eval_model, test_list, criterion, device, cfg, logger,
            threshold=optimal_threshold,
            n_views=n_tta,
        )
    else:
        te_met, (ns, ls, ps) = validate(
            eval_model, test_loader, criterion, device, cfg, logger,
            return_preds=True, desc="Test evaluation",
            threshold=optimal_threshold,
        )

    logger.info(f"Test results (video-level, thr={optimal_threshold:.4f}):")
    for k, v in te_met.items():
        logger.info(
            f"  {k:20s}: {v:.6f}" if isinstance(v, float) else f"  {k:20s}: {v}"
        )

    write_predictions_csv(
        os.path.join(cfg["output_dir"], "test_predictions.csv"),
        ns, ls, ps, thr=optimal_threshold,
    )

    logger.info("Generating figures ...")
    try:
        plot_curves(history, cfg["output_dir"])
        plot_eval(ls, ps, cfg["output_dir"], split="test", thr=optimal_threshold)
    except Exception as exc:
        logger.warning(f"Figure error: {exc}")

    logger.info(f"\nTotal time : {timedelta(seconds=int(timer.elapsed()))}")
    logger.info(f"Best val AUC: {best_auc:.6f}  |  Best val F1: {best_f1:.6f}")
    logger.info(f"Final threshold: {optimal_threshold:.4f}")
    logger.info("Done.")


if __name__ == "__main__":
    main()


# CELL 3
!python train_teacher_v6.py