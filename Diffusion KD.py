"""
kd_dpvit_diffusion.py — Knowledge Distillation of DP-ViT for Diffusion-Based
DeepFake Detection
=================================================================================
Architecture
────────────
  Teacher : DP-ViT (vit_base_patch16_224 × 2) + GatedFusionHead   ~175 M params
  Student : EfficientNet-B2 (dual-path, RGB + HF) + LightFusionHead ~16  M params

Knowledge-Distillation loss (per batch)
────────────────────────────────────────
  L = α · L_KL(student_logits, teacher_logits, T)
    + (1-α) · L_BCE(student_logits, hard_labels)

  where  T   = temperature (default 4.0)
         α   = KD weight   (default 0.7)

Dataset layout  (flat folders, no sub-folders)
──────────────────────────────────────────────
  BASE_PATH/
      Real/          *.png / *.jpg   → label 0
      DiffSwap/      *.png / *.jpg   → label 1
      SDv15_DS0.3/   *.png / *.jpg   → label 1


"""

# ── Standard library ──────────────────────────────────────────────────────────
import glob
import os
import random
import time
import warnings
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

class Config:
    """All hyper-parameters and paths in one place."""

    # ── Paths ─────────────────────────────────────────────────────────────────
    BASE_PATH: str = (
        "/kaggle/input/datasets/syedazmulhasansabbir/diffusion/diffusion"
    )
    REAL_FOLDER:  str       = "Real"
    FAKE_FOLDERS: List[str] = ["DiffSwap", "SDv15_DS0.3"]
    OUT_DIR: Path           = Path("/kaggle/working/kd_dpvit_diffusion_outputs")

    # ── Teacher checkpoint (required) ────────────────────────────────────────
    # Point this to the best_dpvit_diffusion_auc.pth produced by the teacher
    # training script.  The teacher is loaded in eval mode and never updated.
    TEACHER_CHECKPOINT: str = (
        "/kaggle/input/models/syedazmulhasansabbir/dp-vit-diffusion-ep08/tensorflow2/default/1/epoch_008.pth"
    )

    # ── Student resume / epoch cap ────────────────────────────────────────────
    # Set RESUME_STUDENT_CHECKPOINT to a student epoch_NNN.pth to resume.
    RESUME_STUDENT_CHECKPOINT: Optional[str] = None
    END_EPOCH: int = 20   # last epoch to train (inclusive)

    # ── Data split ────────────────────────────────────────────────────────────
    VAL_SIZE:  float = 0.10
    TEST_SIZE: float = 0.10

    # ── Training ──────────────────────────────────────────────────────────────
    IMG_SIZE:    int = 224
    BATCH_SIZE:  int = 32
    ACCUM_STEPS: int = 2       # effective batch = 32 × 2 = 64
    NUM_WORKERS: int = 4

    # ── Optimiser ─────────────────────────────────────────────────────────────
    # Differential LRs: backbone gets a lower LR to preserve pretrained
    # EfficientNet-B2 features; fusion head gets the full LR.
    LR_BACKBONE: float = 1e-5
    LR_HEAD:     float = 2e-4
    WD:          float = 1e-4

    # ── LR schedule ───────────────────────────────────────────────────────────
    WARMUP_EPOCHS: int = 2

    # ── Knowledge Distillation ────────────────────────────────────────────────
    KD_TEMPERATURE: float = 4.0   # T: softer teacher distribution
    KD_ALPHA:       float = 0.7   # weight on KL loss; (1-α) on BCE hard loss

    # ── Early stopping ────────────────────────────────────────────────────────
    PATIENCE: int = 7

    # ── Time budget ───────────────────────────────────────────────────────────
    # NOTE: the time-budget guard is only active when DEVICE == CUDA.
    # On CPU (no GPU quota), it is silently skipped so the job never aborts
    # before training even starts.
    TIME_BUDGET_HOURS:    float = 11.5
    EVAL_RESERVE_MINUTES: float = 30.0

    # ── Misc ──────────────────────────────────────────────────────────────────
    SEED:        int  = 42
    # FIX-G: AMP is only enabled on CUDA.  On CPU, torch.amp.autocast defaults
    # to bfloat16 which numpy cannot deserialise → TypeError on .numpy().
    # AMP provides zero speedup on CPU anyway, so this is safe to gate.
    AMP_ENABLED: bool = True   # will be overridden to False when DEVICE==cpu

    # ── Teacher backbone  (must match the saved teacher weights) ──────────────
    TEACHER_BACKBONE: str = "vit_base_patch16_224"

    # ── Student backbone ──────────────────────────────────────────────────────
    # EfficientNet-B2: ~7.7 M params per branch, num_features=1408.
    # Both RGB and HF branches share this backbone type (weights are NOT
    # tied — each branch has its own independent parameters).
    STUDENT_BACKBONE: str = "efficientnet_b2"


cfg = Config()
cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = cfg.OUT_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# FIX-G: disable AMP on CPU — bfloat16 output from autocast breaks .numpy()
if DEVICE.type == "cpu":
    cfg.AMP_ENABLED = False
    print("[Config] CPU detected — AMP disabled (bfloat16 unsupported by numpy).")

_TRAIN_START: float = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 2.  REPRODUCIBILITY
# ══════════════════════════════════════════════════════════════════════════════

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


seed_everything(cfg.SEED)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  GPU-SIDE LAPLACIAN HIGH-FREQUENCY EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════════

class LaplacianHF(nn.Module):
    """
    Differentiable Laplacian filter applied per-channel on a normalised
    image tensor.  Output is re-normalised to ImageNet stats so it can be
    fed directly into an EfficientNet-B2 backbone.
    """
    def __init__(self) -> None:
        super().__init__()
        kernel = torch.tensor(
            [[0.,  1., 0.],
             [1., -4., 1.],
             [0.,  1., 0.]],
        ).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        self.register_buffer("kernel", kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hf = F.conv2d(x, self.kernel, padding=1, groups=3).abs()
        b  = hf.size(0)
        lo = hf.view(b, -1).min(1).values.view(b, 1, 1, 1)
        hi = hf.view(b, -1).max(1).values.view(b, 1, 1, 1)
        hf = (hf - lo) / (hi - lo + 1e-6)
        mean = torch.tensor([0.485, 0.456, 0.406], device=hf.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=hf.device).view(1, 3, 1, 1)
        return (hf - mean) / std


laplacian_extractor = LaplacianHF().to(DEVICE)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  DATASET
# ══════════════════════════════════════════════════════════════════════════════

def _collect_images(folder: str) -> List[str]:
    paths: List[str] = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        paths.extend(glob.glob(os.path.join(folder, ext)))
    return paths


class DiffusionDeepfakeDataset(Dataset):
    """
    Flat-folder dataset.  Each image is loaded as RGB; the Laplacian HF stream
    is computed on the GPU inside the training loop (not here) to avoid
    CPU–GPU transfer overhead.
    """
    def __init__(
        self,
        paths:     List[str],
        labels:    List[int],
        transform: A.Compose,
    ) -> None:
        assert len(paths) == len(labels)
        self.paths     = paths
        self.labels    = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img    = np.array(Image.open(self.paths[idx]).convert("RGB"))
        tensor = self.transform(image=img)["image"]
        label  = torch.tensor(self.labels[idx], dtype=torch.float32)
        return tensor, label


# ══════════════════════════════════════════════════════════════════════════════
# 5.  ALBUMENTATIONS TRANSFORMS
# ══════════════════════════════════════════════════════════════════════════════

def _coarse_dropout_kwargs(img_size: int) -> dict:
    """Return CoarseDropout kwargs compatible with Albumentations ≥ 1.4."""
    import albumentations
    major, minor, *_ = [int(x) for x in albumentations.__version__.split(".")[:2]]
    hole_side = img_size // 16
    if (major, minor) >= (1, 4):
        return dict(
            max_num_holes=8,
            hole_height_range=(1, hole_side),
            hole_width_range=(1, hole_side),
            fill=0,
            p=0.2,
        )
    return dict(
        max_holes=8,
        max_height=hole_side,
        max_width=hole_side,
        fill_value=0,
        p=0.2,
    )


def get_train_transforms() -> A.Compose:
    s = cfg.IMG_SIZE
    return A.Compose([
        A.Resize(s, s),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.10, rotate_limit=10, p=0.4
        ),
        A.ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MedianBlur(blur_limit=5,        p=1.0),
        ], p=0.3),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),
        A.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.4
        ),
        A.CoarseDropout(**_coarse_dropout_kwargs(s)),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_eval_transforms() -> A.Compose:
    s = cfg.IMG_SIZE
    return A.Compose([
        A.Resize(s, s),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# 6.  DATA LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def build_dataloaders() -> Tuple[DataLoader, DataLoader, DataLoader]:
    real_paths = _collect_images(os.path.join(cfg.BASE_PATH, cfg.REAL_FOLDER))
    fake_paths: List[str] = []
    for folder_name in cfg.FAKE_FOLDERS:
        fake_paths.extend(
            _collect_images(os.path.join(cfg.BASE_PATH, folder_name))
        )

    all_paths  = real_paths + fake_paths
    all_labels = [0] * len(real_paths) + [1] * len(fake_paths)

    print(
        f"[Dataset] real={len(real_paths):,}  fake={len(fake_paths):,}  "
        f"total={len(all_paths):,}"
    )

    val_relative = cfg.VAL_SIZE / (1.0 - cfg.TEST_SIZE)

    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        all_paths, all_labels,
        test_size=cfg.TEST_SIZE, random_state=cfg.SEED, stratify=all_labels,
    )
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels,
        test_size=val_relative, random_state=cfg.SEED, stratify=train_val_labels,
    )

    print(
        f"[Split]   train={len(train_paths):,}  val={len(val_paths):,}  "
        f"test={len(test_paths):,}"
    )

    train_ds = DiffusionDeepfakeDataset(
        train_paths, train_labels, get_train_transforms()
    )
    val_ds = DiffusionDeepfakeDataset(
        val_paths, val_labels, get_eval_transforms()
    )
    test_ds = DiffusionDeepfakeDataset(
        test_paths, test_labels, get_eval_transforms()
    )

    counts   = [train_labels.count(0), train_labels.count(1)]
    weights  = [1.0 / c for c in counts]
    sample_w = [weights[lbl] for lbl in train_labels]
    sampler  = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

    use_persistent = cfg.NUM_WORKERS > 0
    loader_kw = dict(
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=DEVICE.type == "cuda",
        persistent_workers=use_persistent,
    )
    train_loader = DataLoader(train_ds, sampler=sampler, **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False,   **loader_kw)
    test_loader  = DataLoader(test_ds,  shuffle=False,   **loader_kw)

    return train_loader, val_loader, test_loader


# ══════════════════════════════════════════════════════════════════════════════
# 7.  MODELS
# ══════════════════════════════════════════════════════════════════════════════

# ── 7a. Teacher — full DP-ViT with GatedFusionHead ───────────────────────────

class GatedFusionHead(nn.Module):
    """
    Gated cross-stream fusion: learns to weight the RGB and HF features
    adaptively before the final projection.
    """
    def __init__(
        self, in_dim: int, hidden: int = 512, dropout: float = 0.3
    ) -> None:
        super().__init__()
        fused_dim = 2 * in_dim
        self.gate = nn.Sequential(nn.Linear(fused_dim, fused_dim), nn.Sigmoid())
        self.proj = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(
        self, f_rgb: torch.Tensor, f_hf: torch.Tensor
    ) -> torch.Tensor:
        fused = torch.cat([f_rgb, f_hf], dim=1)
        return self.proj(self.gate(fused) * fused)


class TeacherDPViT(nn.Module):
    """
    Full DP-ViT teacher.  Loaded from a pre-trained checkpoint and frozen —
    gradients are never computed for this model during KD training.
    """
    def __init__(
        self,
        backbone:   str  = "vit_base_patch16_224",
        pretrained: bool = False,          # weights come from the checkpoint
    ) -> None:
        super().__init__()
        self.rgb_enc = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.hf_enc  = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        feat_dim     = self.rgb_enc.num_features
        self.head    = GatedFusionHead(in_dim=feat_dim)

    def forward(self, x: torch.Tensor, hf: torch.Tensor) -> torch.Tensor:
        return self.head(self.rgb_enc(x), self.hf_enc(hf))


# ── 7b. Student — dual-path EfficientNet-B2 with LightFusionHead ─────────────

class LightFusionHead(nn.Module):
    """
    Lightweight fusion head for the EfficientNet-B2 student.
    Concatenates the RGB and HF feature vectors, applies LayerNorm, then
    projects through a 2-layer MLP to a single logit.

    in_dim  : num_features of one EfficientNet-B2 branch (1408)
    fused_dim = 2 × in_dim = 2816
    """
    def __init__(
        self, in_dim: int, hidden: int = 256, dropout: float = 0.3
    ) -> None:
        super().__init__()
        fused_dim = 2 * in_dim
        self.proj = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(
        self, f_rgb: torch.Tensor, f_hf: torch.Tensor
    ) -> torch.Tensor:
        return self.proj(torch.cat([f_rgb, f_hf], dim=1))


class StudentEfficientNetB2(nn.Module):
    """
    Dual-path EfficientNet-B2 student.

    Architecture
    ────────────
      RGB branch : EfficientNet-B2 feature extractor  (num_features = 1408)
      HF  branch : EfficientNet-B2 feature extractor  (num_features = 1408)
                   ↓ f_rgb (B, 1408)   ↓ f_hf (B, 1408)
                        LightFusionHead
                          (B, 2816) → LayerNorm → Linear(256) → GELU → Linear(1)

    The dual-path design mirrors the teacher's DP-ViT contract: both models
    accept (x_rgb, x_hf) and emit a single (B, 1) logit, keeping downstream
    MoE fusion fully compatible.

    Total parameters ≈ 16.3 M  (2 × 7.7 M backbone + ~1.8 M head).
    """
    def __init__(
        self,
        backbone:   str  = "efficientnet_b2",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        # Each branch is a separate EfficientNet-B2 instance — weights are NOT
        # shared so the HF branch can specialize on frequency artifacts while
        # the RGB branch handles semantic appearance cues.
        self.rgb_enc = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0
        )
        self.hf_enc = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0
        )
        feat_dim  = self.rgb_enc.num_features          # 1408 for EfficientNet-B2
        self.head = LightFusionHead(in_dim=feat_dim)

        total = sum(p.numel() for p in self.parameters())
        print(
            f"[Student] EfficientNet-B2 (dual-path)  "
            f"total params = {total / 1e6:.1f}M  "
            f"feat_dim (per branch) = {feat_dim}"
        )

    def forward(self, x: torch.Tensor, hf: torch.Tensor) -> torch.Tensor:
        """
        x  : (B, 3, H, W)  — normalised RGB image
        hf : (B, 3, H, W)  — Laplacian HF image (same spatial resolution)
        Returns: (B, 1) raw logit
        """
        f_rgb = self.rgb_enc(x)    # (B, 1408)
        f_hf  = self.hf_enc(hf)   # (B, 1408)
        return self.head(f_rgb, f_hf)


# ══════════════════════════════════════════════════════════════════════════════
# 8.  KNOWLEDGE-DISTILLATION LOSS
# ══════════════════════════════════════════════════════════════════════════════

class KDLoss(nn.Module):
    """
    Combined KD loss for binary classification with logit outputs.

    L = α · T² · KL( σ(s/T) ‖ σ(t/T) )  +  (1-α) · BCE(s, y_hard)

    where σ is the sigmoid function.  The T² factor re-scales the KL term to
    match the gradient magnitude of the BCE term (standard KD practice).

    Note: KLDivLoss expects log-probabilities for the *input* and
    probabilities for the *target*, matching PyTorch's convention.
    """
    def __init__(self, temperature: float = 4.0, alpha: float = 0.7) -> None:
        super().__init__()
        self.T     = temperature
        self.alpha = alpha
        self.kl    = nn.KLDivLoss(reduction="batchmean")
        self.bce   = nn.BCEWithLogitsLoss()

    def forward(
        self,
        student_logits: torch.Tensor,   # (B, 1)
        teacher_logits: torch.Tensor,   # (B, 1)
        hard_labels:    torch.Tensor,   # (B, 1)  float 0/1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (total_loss, kl_loss, bce_loss) for logging.
        """
        # ── Soft targets (sigmoid-based KL in logit space) ────────────────────
        # We model the binary problem as a 2-class distribution:
        #   p = [sigmoid(logit), 1 - sigmoid(logit)]
        # This keeps the temperature-scaling identical to the multi-class KD
        # formulation while staying in binary territory.
        s_prob_pos = torch.sigmoid(student_logits / self.T)
        t_prob_pos = torch.sigmoid(teacher_logits / self.T)

        # Stack into (B, 2) distributions: [P(fake), P(real)]
        s_dist = torch.cat([s_prob_pos, 1.0 - s_prob_pos], dim=1)   # (B, 2)
        t_dist = torch.cat([t_prob_pos, 1.0 - t_prob_pos], dim=1)   # (B, 2)

        # KLDivLoss expects log-probs for input, probs for target
        kl_loss = self.kl(
            torch.log(s_dist + 1e-8),
            t_dist.detach(),               # teacher is frozen; detach is explicit
        ) * (self.T ** 2)

        # ── Hard-label BCE ────────────────────────────────────────────────────
        bce_loss = self.bce(student_logits, hard_labels)

        total = self.alpha * kl_loss + (1.0 - self.alpha) * bce_loss
        return total, kl_loss, bce_loss


# ══════════════════════════════════════════════════════════════════════════════
# 9.  LR SCHEDULER — LINEAR WARMUP + COSINE DECAY
# ══════════════════════════════════════════════════════════════════════════════

def build_scheduler(
    opt:          optim.Optimizer,
    warmup_steps: int,
    total_steps:  int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


# ══════════════════════════════════════════════════════════════════════════════
# 10.  TIME-BUDGET HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _budget_seconds() -> float:
    return cfg.TIME_BUDGET_HOURS * 3600 - cfg.EVAL_RESERVE_MINUTES * 60


def _elapsed() -> float:
    return time.time() - _TRAIN_START


def _time_ok(epoch_duration_s: float) -> bool:
    """
    Returns True if there is enough wall-clock budget for one more epoch.

    FIX-J: always returns True on CPU.  The time-budget guard exists solely
    to protect Kaggle GPU quota; on a CPU-only kernel the estimate would be
    enormous and would abort training before the first epoch.
    """
    if DEVICE.type == "cpu":
        return True
    remaining = _budget_seconds() - _elapsed()
    return remaining > epoch_duration_s * 1.05


# ══════════════════════════════════════════════════════════════════════════════
# 11.  CHECKPOINT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _validate_checkpoint(path: Path) -> str:
    """
    Validate a PyTorch checkpoint and return its serialisation format
    ("zip" or "legacy_pickle").  Raises a clear RuntimeError on any issue.

    Logic mirrors FIX-A from the diffusion training script:
      • 0-byte files are rejected immediately.
      • ZIP magic (b'PK') → central-directory + namelist check.
      • Non-ZIP magic → legacy pickle; trust torch.load to handle it.
    """
    if not path.exists():
        raise RuntimeError(
            f"[Checkpoint] File not found: {path}\n"
            "  → Check that the path points to a valid Kaggle dataset/version."
        )

    file_size = path.stat().st_size
    if file_size == 0:
        raise RuntimeError(
            f"[Checkpoint] File is 0 bytes: {path}\n"
            "  → The checkpoint was never written (training crashed before "
            "the first torch.save call)."
        )

    with open(path, "rb") as fh:
        magic = fh.read(4)

    if magic[:2] == b"PK":
        if not zipfile.is_zipfile(str(path)):
            raise RuntimeError(
                f"[Checkpoint] ZIP magic present but central directory is "
                f"corrupt: {path}  ({file_size:,} bytes)\n"
                "  → Re-run the producing session and re-commit the dataset."
            )
        try:
            with zipfile.ZipFile(str(path), "r") as zf:
                members = zf.namelist()
            if not members:
                raise RuntimeError(
                    f"[Checkpoint] ZIP archive has no members: {path}"
                )
        except zipfile.BadZipFile as exc:
            raise RuntimeError(
                f"[Checkpoint] Bad ZIP file: {path}\n  {exc}"
            ) from exc
        fmt    = "zip"
        detail = f"{len(members)} zip members"
    else:
        fmt    = "legacy_pickle"
        detail = "legacy pickle (pre-1.6 serialisation)"

    print(
        f"[Checkpoint] Validation OK — {path.name}  "
        f"({file_size / 1e6:.1f} MB,  {detail})"
    )
    return fmt


def _save_checkpoint(
    path:      Path,
    epoch:     int,
    model:     StudentEfficientNetB2,
    opt:       optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler:    torch.amp.GradScaler,
    metrics:   Dict[str, float],
) -> None:
    torch.save(
        {
            "epoch":             epoch,
            "model_state_dict":  model.state_dict(),
            "optim_state_dict":  opt.state_dict(),
            "sched_state_dict":  scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            **metrics,
        },
        path,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 12.  SAFE TENSOR-TO-NUMPY HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _to_numpy(t: torch.Tensor) -> np.ndarray:
    """
    FIX-H: always cast to float32 before calling .cpu().numpy().

    torch.amp.autocast can leave tensors in bfloat16 (on CPU) or float16
    (on CUDA).  numpy does not support bfloat16 at all, and older numpy
    builds don't support float16 either.  A single .float() cast is the
    safest universal fix.
    """
    return t.detach().float().cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════════
# 13.  TRAIN / VALIDATE / TEST
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    student:   StudentEfficientNetB2,
    teacher:   TeacherDPViT,
    loader:    DataLoader,
    criterion: KDLoss,
    opt:       optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler:    torch.amp.GradScaler,
    hf_ext:    LaplacianHF,
    accum:     int,
    epoch_idx: int,
) -> Tuple[float, float, float, float]:
    """
    One KD training epoch.

    Returns (avg_total_loss, avg_kl_loss, avg_bce_loss, accuracy).
    """
    student.train()
    # Teacher stays in eval mode throughout — we never update its weights.
    teacher.eval()

    run_total = run_kl = run_bce = 0.0
    correct   = total  = 0
    opt.zero_grad(set_to_none=True)

    pbar = tqdm(
        loader,
        desc=f"Train E{epoch_idx:02d}",
        leave=False,
        dynamic_ncols=True,
    )
    for step, (x, y) in enumerate(pbar):
        x = x.to(DEVICE, non_blocking=True)
        y = y.unsqueeze(1).to(DEVICE, non_blocking=True)

        # ── Shared HF tensor (computed once, reused for both models) ──────────
        with torch.no_grad():
            hf = hf_ext(x)

        # ── Teacher inference (no gradient needed) ────────────────────────────
        with torch.no_grad():
            teacher_logits = teacher(x, hf)

        # ── Student forward + KD loss ─────────────────────────────────────────
        with torch.amp.autocast(device_type=DEVICE.type, enabled=cfg.AMP_ENABLED):
            student_logits = student(x, hf)
            loss, kl_loss, bce_loss = criterion(student_logits, teacher_logits, y)
            loss = loss / accum

        scaler.scale(loss).backward()

        if (step + 1) % accum == 0 or (step + 1) == len(loader):
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            opt.zero_grad(set_to_none=True)

        run_total += loss.item() * accum
        run_kl    += kl_loss.item()
        run_bce   += bce_loss.item()

        # FIX-H: cast to float32 before comparison / numpy conversion
        preds    = (torch.sigmoid(student_logits.float()) >= 0.5).float()
        correct += (preds == y).sum().item()
        total   += y.size(0)

        pbar.set_postfix(
            total=f"{run_total / (step + 1):.4f}",
            kl=f"{run_kl / (step + 1):.4f}",
        )

    n = len(loader)
    return run_total / n, run_kl / n, run_bce / n, correct / total


@torch.no_grad()
def evaluate(
    student:   StudentEfficientNetB2,
    loader:    DataLoader,
    hf_ext:    LaplacianHF,
    split:     str = "Val",
) -> Tuple[float, float, float]:
    """
    Evaluate the student on a split.

    Returns (accuracy, AUC, F1).
    Note: no KD loss here — teacher is not needed at eval time.
    """
    student.eval()
    bce_criterion = nn.BCEWithLogitsLoss()

    run_loss    = 0.0
    all_probs:  List[float] = []
    all_labels: List[int]   = []

    for x, y in tqdm(loader, desc=f"  {split:4s}", leave=False, dynamic_ncols=True):
        x = x.to(DEVICE, non_blocking=True)
        y = y.unsqueeze(1).to(DEVICE, non_blocking=True)
        hf = hf_ext(x)

        with torch.amp.autocast(device_type=DEVICE.type, enabled=cfg.AMP_ENABLED):
            logits = student(x, hf)
            loss   = bce_criterion(logits, y)

        run_loss += loss.item()

        # FIX-H: cast to float32 before .numpy() to handle fp16/bf16 outputs
        all_probs.extend(
            _to_numpy(torch.sigmoid(logits).squeeze(1)).tolist()
        )
        all_labels.extend(
            _to_numpy(y.squeeze(1)).astype(int).tolist()
        )

    probs_arr  = np.array(all_probs)
    labels_arr = np.array(all_labels)
    preds_hard = (probs_arr >= 0.5).astype(int)

    auc = roc_auc_score(labels_arr, probs_arr)
    f1  = f1_score(labels_arr, preds_hard, zero_division=0)
    acc = (preds_hard == labels_arr).mean()

    return float(acc), float(auc), float(f1)


@torch.no_grad()
def full_test_evaluation(
    student:   StudentEfficientNetB2,
    loader:    DataLoader,
    hf_ext:    LaplacianHF,
    log_lines: List[str],
) -> None:
    student.eval()
    all_probs:  List[float] = []
    all_labels: List[int]   = []

    for x, y in tqdm(loader, desc="  Test", leave=False, dynamic_ncols=True):
        x  = x.to(DEVICE, non_blocking=True)
        y  = y.unsqueeze(1).to(DEVICE, non_blocking=True)
        hf = hf_ext(x)

        with torch.amp.autocast(device_type=DEVICE.type, enabled=cfg.AMP_ENABLED):
            logits = student(x, hf)

        # FIX-H: cast to float32 before .numpy() to handle fp16/bf16 outputs
        all_probs.extend(
            _to_numpy(torch.sigmoid(logits).squeeze(1)).tolist()
        )
        all_labels.extend(
            _to_numpy(y.squeeze(1)).astype(int).tolist()
        )

    probs_arr  = np.array(all_probs)
    labels_arr = np.array(all_labels)
    preds_hard = (probs_arr >= 0.5).astype(int)

    auc  = roc_auc_score(labels_arr, probs_arr)
    acc  = (preds_hard == labels_arr).mean()
    f1   = f1_score(labels_arr, preds_hard, zero_division=0)
    prec = precision_score(labels_arr, preds_hard, zero_division=0)
    rec  = recall_score(labels_arr, preds_hard, zero_division=0)
    cm   = confusion_matrix(labels_arr, preds_hard)
    report = classification_report(
        labels_arr, preds_hard,
        target_names=["Real", "Fake"],
        digits=4,
    )

    lines = [
        "",
        "═" * 72,
        "  STUDENT TEST SET RESULTS",
        "═" * 72,
        f"  AUC       : {auc:.4f}",
        f"  Accuracy  : {acc:.4f}",
        f"  F1        : {f1:.4f}",
        f"  Precision : {prec:.4f}",
        f"  Recall    : {rec:.4f}",
        "",
        "  Confusion Matrix (rows=true, cols=pred):",
        f"  {cm}",
        "",
        "  Per-class Report:",
        report,
        "═" * 72,
    ]
    for ln in lines:
        print(ln)
        log_lines.append(ln)


# ══════════════════════════════════════════════════════════════════════════════
# 14.  EARLY STOPPING
# ══════════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    def __init__(self, patience: int = cfg.PATIENCE, min_delta: float = 1e-4) -> None:
        self.patience  = patience
        self.min_delta = min_delta
        self.best:     Optional[float] = None
        self.counter   = 0

    def __call__(self, score: float) -> bool:
        if self.best is None or score > self.best + self.min_delta:
            self.best    = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# ══════════════════════════════════════════════════════════════════════════════
# 15.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    global _TRAIN_START
    _TRAIN_START = time.time()

    print(
        f"[Config] device={DEVICE}  AMP={cfg.AMP_ENABLED}  "
        f"BS={cfg.BATCH_SIZE}  accum={cfg.ACCUM_STEPS}  "
        f"effBS={cfg.BATCH_SIZE * cfg.ACCUM_STEPS}  "
        f"T={cfg.KD_TEMPERATURE}  alpha={cfg.KD_ALPHA}  "
        f"budget={cfg.TIME_BUDGET_HOURS}h  "
        f"eval_reserve={cfg.EVAL_RESERVE_MINUTES}min"
    )

    train_loader, val_loader, test_loader = build_dataloaders()

    # ── Build teacher ─────────────────────────────────────────────────────────
    print("\n[Teacher] Loading …")
    teacher = TeacherDPViT(backbone=cfg.TEACHER_BACKBONE, pretrained=False).to(DEVICE)

    teacher_ckpt_path = Path(cfg.TEACHER_CHECKPOINT)
    _validate_checkpoint(teacher_ckpt_path)
    teacher_ckpt = torch.load(
        teacher_ckpt_path, map_location=DEVICE, weights_only=False
    )
    teacher.load_state_dict(teacher_ckpt["model_state_dict"])
    teacher.eval()

    # Freeze all teacher parameters — we never call opt.step() on them.
    for p in teacher.parameters():
        p.requires_grad_(False)

    t_total = sum(p.numel() for p in teacher.parameters())
    print(f"[Teacher] Loaded & frozen  ({t_total / 1e6:.1f}M params)")

    # ── Build student ─────────────────────────────────────────────────────────
    print("\n[Student] Building …")
    student = StudentEfficientNetB2(
        backbone=cfg.STUDENT_BACKBONE, pretrained=True
    ).to(DEVICE)

    # ── Loss, optimiser, scaler, scheduler ───────────────────────────────────
    criterion = KDLoss(temperature=cfg.KD_TEMPERATURE, alpha=cfg.KD_ALPHA)

    # Differential learning rates: backbone params get LR_BACKBONE (lower),
    # fusion head params get LR_HEAD (higher) to train the new head faster.
    backbone_params = (
        list(student.rgb_enc.parameters())
        + list(student.hf_enc.parameters())
    )
    head_params = list(student.head.parameters())

    opt = optim.AdamW(
        [
            {"params": backbone_params, "lr": cfg.LR_BACKBONE},
            {"params": head_params,     "lr": cfg.LR_HEAD},
        ],
        weight_decay=cfg.WD,
    )

    # FIX-B: non-deprecated GradScaler form; works on both CUDA and CPU.
    scaler = torch.amp.GradScaler(DEVICE.type, enabled=cfg.AMP_ENABLED)

    steps_per_epoch = len(train_loader)

    # FIX-C: scheduler total_steps uses END_EPOCH so the cosine decay is
    # correct whether we start from scratch or resume mid-run.
    scheduler = build_scheduler(
        opt,
        warmup_steps = cfg.WARMUP_EPOCHS * steps_per_epoch,
        total_steps  = cfg.END_EPOCH     * steps_per_epoch,
    )

    # ── Resume from student checkpoint ───────────────────────────────────────
    resume_epoch = 0
    best_auc     = 0.0

    if cfg.RESUME_STUDENT_CHECKPOINT:
        ckpt_path = Path(cfg.RESUME_STUDENT_CHECKPOINT)
        ckpt_fmt  = _validate_checkpoint(ckpt_path)

        print(f"\n[Resume] Loading student checkpoint ({ckpt_fmt}): {ckpt_path}")

        # FIX-D: explicit weights_only=False
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

        student.load_state_dict(ckpt["model_state_dict"])

        # FIX-E: move optimizer state tensors to DEVICE
        opt.load_state_dict(ckpt["optim_state_dict"])
        for state in opt.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(DEVICE)

        if "sched_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["sched_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])

        resume_epoch = int(ckpt["epoch"])
        best_auc     = float(ckpt.get("val_auc", 0.0))

        print(
            f"[Resume] Resumed from epoch {resume_epoch}  "
            f"(best val AUC so far = {best_auc:.4f})\n"
            f"[Resume] Will train epoch(s) {resume_epoch + 1} … {cfg.END_EPOCH}."
        )
    else:
        print("[Resume] No student checkpoint — training from scratch.")

    if resume_epoch >= cfg.END_EPOCH:
        print(
            f"[Resume] resume_epoch ({resume_epoch}) >= END_EPOCH ({cfg.END_EPOCH}). "
            "Nothing to train — proceeding directly to test evaluation."
        )

    early_stop = EarlyStopping(patience=cfg.PATIENCE)
    best_ckpt  = cfg.OUT_DIR / "best_student_diffusion_auc.pth"
    log_lines: List[str] = []

    header = (
        f"\n{'─' * 72}\n"
        f"  KD Training — Student EfficientNet-B2  |  "
        f"epochs {resume_epoch + 1} → {cfg.END_EPOCH}\n"
        f"  Teacher: {cfg.TEACHER_BACKBONE}  →  "
        f"Student: {cfg.STUDENT_BACKBONE}\n"
        f"  T={cfg.KD_TEMPERATURE}  α={cfg.KD_ALPHA}\n"
        f"{'─' * 72}"
    )
    print(header)
    log_lines.append(header)

    # ── FIX-F + FIX-I: timing dry-run ────────────────────────────────────────
    # Run in TRAIN mode (not eval) with actual forward + backward so the
    # timing reflects real GPU kernel warm-up and gradient computation.
    # Cap at 3 batches to avoid wasting quota.
    # FIX-J: on CPU the time-budget guard is unconditionally bypassed, so
    # even a large estimate will not abort training prematurely.
    try:
        student.train()
        teacher.eval()
        _dummy_opt = optim.SGD(student.parameters(), lr=0.0)  # no actual update
        _t0 = time.time()
        _batches_timed = 0
        for _i, (_xb, _yb) in enumerate(train_loader):
            _xb = _xb.to(DEVICE, non_blocking=True)
            _yb = _yb.unsqueeze(1).to(DEVICE, non_blocking=True)
            with torch.no_grad():
                _hb = laplacian_extractor(_xb)
                _t_logits = teacher(_xb, _hb)
            with torch.amp.autocast(device_type=DEVICE.type, enabled=cfg.AMP_ENABLED):
                _s_logits = student(_xb, _hb)
                _loss, _, _ = criterion(_s_logits, _t_logits, _yb)
                _loss = _loss / cfg.ACCUM_STEPS
            scaler.scale(_loss).backward()
            scaler.step(_dummy_opt)
            scaler.update()
            _dummy_opt.zero_grad(set_to_none=True)
            _batches_timed += 1
            if _i >= 2:   # time exactly 3 batches (indices 0, 1, 2)
                break
        _per_batch_s        = (time.time() - _t0) / max(1, _batches_timed)
        last_epoch_duration = _per_batch_s * len(train_loader)
        print(
            f"[Timing] Estimated epoch duration ≈ "
            f"{last_epoch_duration / 60:.1f} min  "
            f"(based on {_batches_timed} timed training batch(es))"
        )
        # Re-zero gradients so the actual training loop starts clean
        opt.zero_grad(set_to_none=True)
    except Exception as exc:
        last_epoch_duration = 600.0
        print(
            f"[Timing] Timing estimate failed ({exc}); "
            "defaulting to 600 s per epoch."
        )

    stop_reason = "completed all epochs"

    for epoch in range(resume_epoch + 1, cfg.END_EPOCH + 1):

        # ── Time-budget check (CUDA only — see FIX-J) ─────────────────────────
        if not _time_ok(last_epoch_duration):
            stop_reason = (
                f"time budget — less than "
                f"{cfg.EVAL_RESERVE_MINUTES:.0f} min remain for evaluation"
            )
            print(f"\n  ⏱  Stopping after epoch {epoch - 1}: {stop_reason}.")
            log_lines.append(f"\n  Stopped early: {stop_reason}.")
            break

        epoch_start = time.time()

        tr_total, tr_kl, tr_bce, tr_acc = train_one_epoch(
            student, teacher, train_loader, criterion,
            opt, scheduler, scaler, laplacian_extractor, cfg.ACCUM_STEPS, epoch,
        )

        val_acc, val_auc, val_f1 = evaluate(
            student, val_loader, laplacian_extractor, split="Val",
        )

        last_epoch_duration = time.time() - epoch_start
        elapsed_h = _elapsed() / 3600

        row = (
            f"Epoch {epoch:03d}/{cfg.END_EPOCH} | "
            f"TrTot {tr_total:.4f}  TrKL {tr_kl:.4f}  TrBCE {tr_bce:.4f}  "
            f"TrAcc {tr_acc:.4f} | "
            f"VaAcc {val_acc:.4f}  VaAUC {val_auc:.4f}  VaF1 {val_f1:.4f} | "
            f"LR_bb {opt.param_groups[0]['lr']:.2e}  "
            f"LR_hd {opt.param_groups[1]['lr']:.2e}  "
            f"[{elapsed_h:.2f}h elapsed]"
        )
        print(row)
        log_lines.append(row)

        # ── Per-epoch checkpoint (always saved after every epoch) ─────────────
        epoch_ckpt = CKPT_DIR / f"epoch_{epoch:03d}.pth"
        _save_checkpoint(
            epoch_ckpt, epoch, student, opt, scheduler, scaler,
            {
                "val_auc":  val_auc,
                "val_f1":   val_f1,
                "val_acc":  val_acc,
                "tr_total": tr_total,
                "tr_kl":    tr_kl,
                "tr_bce":   tr_bce,
                "tr_acc":   tr_acc,
            },
        )
        print(f"  💾 Checkpoint saved → {epoch_ckpt}")

        # ── Best-AUC checkpoint ───────────────────────────────────────────────
        if val_auc > best_auc:
            best_auc = val_auc
            _save_checkpoint(
                best_ckpt, epoch, student, opt, scheduler, scaler,
                {"val_auc": val_auc, "val_f1": val_f1, "val_acc": val_acc},
            )
            note = f"  ✓ New best AUC={best_auc:.4f} — saved → {best_ckpt}"
            print(note)
            log_lines.append(note)

        # ── Early stopping ────────────────────────────────────────────────────
        if early_stop(val_auc):
            stop_reason = (
                f"no AUC improvement for {cfg.PATIENCE} consecutive epochs"
            )
            msg = f"\n  Early stopping after epoch {epoch}: {stop_reason}."
            print(msg)
            log_lines.append(msg)
            break

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = [
        "",
        "─" * 72,
        f"  KD Training finished.  Best Student Val AUC = {best_auc:.4f}",
        f"  Best checkpoint  : {best_ckpt}",
        f"  Epoch checkpoints: {CKPT_DIR}",
        "─" * 72,
    ]
    for ln in summary:
        print(ln)
        log_lines.append(ln)

    # ── Load best student for test evaluation ─────────────────────────────────
    if best_ckpt.exists():
        print("\n  Loading best student checkpoint for test evaluation …")
        ckpt = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
        student.load_state_dict(ckpt["model_state_dict"])

    full_test_evaluation(student, test_loader, laplacian_extractor, log_lines)

    # ── Write training log ────────────────────────────────────────────────────
    log_path = cfg.OUT_DIR / "training_log.txt"
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))
    print(f"\n  Training log saved → {log_path}")


if __name__ == "__main__":
    main()