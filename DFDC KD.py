# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import io
import os
import cv2
import time
import math
import random
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from tqdm.notebook import tqdm
import timm

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, confusion_matrix,
    average_precision_score, precision_recall_curve,
    classification_report, brier_score_loss,
)
from sklearn.calibration import calibration_curve

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DATASET_DIR  = "/kaggle/input/datasets/itamargr/dfdc-faces-of-the-train-sample/train"
REAL_DIR     = os.path.join(DATASET_DIR, "real")
FAKE_DIR     = os.path.join(DATASET_DIR, "fake")
TEACHER_PATH = (
    "/kaggle/input/models/syedazmulhasansabbir/dfdc-dp-vit/tensorflow2/default/1/"
    "dpvit_epoch14_acc0.9974_auc0.9987.pth"
)
OUT_DIR = Path("/kaggle/working/kd_dfdc_v3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 224
SEED     = 42

BATCH_SIZE    = 24
EPOCHS        = 15
LEARNING_RATE = 2e-4
WEIGHT_DECAY  = 1e-4
GRAD_CLIP     = 1.0

# ── Staged unfreezing schedule ────────────────────────────────────────────────
# Stage 1: epochs 0..(FREEZE_EPOCHS-1)  — backbones frozen, heads only
# Stage 2: epochs FREEZE_EPOCHS..(WARMUP_EPOCHS-1) — RGB backbone unfrozen at LR/4
# Stage 3: epochs WARMUP_EPOCHS..      — HF backbone also unfrozen at LR/4,
#                                        RGB backbone promoted to LR/2
FREEZE_EPOCHS = 2
WARMUP_EPOCHS = 4
PATIENCE      = 5    # early stopping; suspended during Stage 1

# ── KD hyperparameters ────────────────────────────────────────────────────────
KD_TEMPERATURE = 4.0

# Final (Stage 3) loss weights — sum = 1.0
W_HARD_FINAL  = 0.40
W_SOFT_FINAL  = 0.30
W_FEAT_FINAL  = 0.15
W_INTER_FINAL = 0.08
W_ATTN_FINAL  = 0.05
W_RKD_FINAL   = 0.02

LABEL_SMOOTHING = 0.05

# Teacher logit clamping (Fix C)
LOGIT_CLIP         = 6.0   
LOGIT_CLIP_WARNING = 8.0

TEACHER_FEAT_DIM    = 1536
TEACHER_ATTN_LAYERS = [3, 6, 9, 11]

MAX_RUN_TIME = 9.0 * 3600
START_TIME   = time.time()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device        : {DEVICE}")
print(f"Student       : EfficientNet-B2 (dual-path RGB+HF, gated fusion)")
print(f"Freeze epochs : {FREEZE_EPOCHS}  |  Warmup epochs: {WARMUP_EPOCHS}")
print(f"Logit clip    : tanh(z / {LOGIT_CLIP}) × {LOGIT_CLIP}  (Fix C)")


# ─────────────────────────────────────────────────────────────────────────────
# FORENSIC AUGMENTATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

class JPEGCompression:
    def __init__(self, q_min: int = 30, q_max: int = 95):
        self.q_min, self.q_max = q_min, q_max

    def __call__(self, img: Image.Image) -> Image.Image:
        q = random.randint(self.q_min, self.q_max)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=q)
        buf.seek(0)
        return Image.open(buf).convert('RGB')


class ResizeRecompress:
    def __init__(self, sizes: list = (96, 112, 128)):
        self.sizes = sizes

    def __call__(self, img: Image.Image) -> Image.Image:
        small = random.choice(self.sizes)
        img = img.resize((small, small), Image.BILINEAR)
        return img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)


class GaussianNoise:
    def __init__(self, max_sigma: float = 0.03):
        self.max_sigma = max_sigma

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        sigma = random.uniform(0.0, self.max_sigma)
        return (t + sigma * torch.randn_like(t)).clamp(0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

class DFDCDataset(Dataset):
    VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp',
                  '.JPG', '.JPEG', '.PNG', '.BMP', '.WEBP'}

    def __init__(self, real_dir: str, fake_dir: str):
        self.all_paths:     list = []
        self.labels:        list = []
        self.path_to_video: dict = {}

        print("Scanning dataset ...")
        for label, folder in [(0, real_dir), (1, fake_dir)]:
            tag   = "real" if label == 0 else "fake"
            count = 0
            for root, _, files in os.walk(folder):
                for f in files:
                    if Path(f).suffix in self.VALID_EXTS:
                        fp = os.path.join(root, f)
                        self.all_paths.append(fp)
                        self.labels.append(label)
                        # Video ID: strip the trailing frame index after the
                        # last underscore so frames from the same video group
                        # together (v2 Bug 8 fix).
                        vid_name = f.rsplit('_', 1)[0] if '_' in f else Path(f).stem
                        self.path_to_video[fp] = f"{tag}_{vid_name}"
                        count += 1
            print(f"  {tag:>4}: {count:,}")

        n_real  = self.labels.count(0)
        n_fake  = self.labels.count(1)
        total   = n_real + n_fake
        print(f"  total: {total:,}  |  real={n_real/total:.1%}  fake={n_fake/total:.1%}")
        if n_real == 0 or n_fake == 0:
            raise ValueError(f"Missing class  real={n_real}  fake={n_fake}")

    def __len__(self) -> int:
        return len(self.all_paths)

    @staticmethod
    def _laplacian_hf(img: Image.Image) -> Image.Image:
        arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        lap = cv2.convertScaleAbs(cv2.Laplacian(arr, cv2.CV_64F))
        return Image.fromarray(cv2.cvtColor(lap, cv2.COLOR_BGR2RGB))

    def __getitem__(self, idx: int):
        try:
            img = Image.open(self.all_paths[idx]).convert('RGB')
        except Exception:
            img = Image.fromarray(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))
        return img, self._laplacian_hf(img), self.labels[idx]


# ── Transforms ────────────────────────────────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

train_tf_pil = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([JPEGCompression(q_min=30, q_max=95)], p=0.5),
    transforms.RandomApply([ResizeRecompress(sizes=[96, 112, 128])],  p=0.3),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
    transforms.RandomAffine(degrees=3, translate=(0.03, 0.03)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
])

train_tf_tensor = transforms.Compose([
    transforms.ToTensor(),
    GaussianNoise(max_sigma=0.02),
    transforms.Normalize(MEAN, STD),
])

eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class KDTransformWrapper(Dataset):
    def __init__(self, subset: Subset, is_train: bool):
        self.subset   = subset
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int):
        img_rgb, img_hf, label = self.subset[idx]

        # Teacher always receives clean eval-transformed images
        rgb_t = eval_tf(img_rgb)
        hf_t  = eval_tf(img_hf)

        if self.is_train:
            # Same random seed for RGB and HF so spatial augmentations match
            seed = random.randint(0, 2**32 - 1)
            random.seed(seed); aug_rgb = train_tf_pil(img_rgb)
            random.seed(seed); aug_hf  = train_tf_pil(img_hf)
            rgb_s = train_tf_tensor(aug_rgb)
            hf_s  = train_tf_tensor(aug_hf)
        else:
            rgb_s = eval_tf(img_rgb)
            hf_s  = eval_tf(img_hf)

        return rgb_t, hf_t, rgb_s, hf_s, torch.tensor(label, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# VIDEO-LEVEL DATASET SPLIT  (v2 Bug 6 fix — no frame leakage)
# ─────────────────────────────────────────────────────────────────────────────

full_dataset = DFDCDataset(real_dir=REAL_DIR, fake_dir=FAKE_DIR)

video_to_indices: dict = defaultdict(list)
for idx, path in enumerate(full_dataset.all_paths):
    vid_id = full_dataset.path_to_video[path]
    video_to_indices[vid_id].append(idx)

all_video_ids = list(video_to_indices.keys())
rng = random.Random(SEED)
rng.shuffle(all_video_ids)

n_vids    = len(all_video_ids)
n_train_v = int(0.80 * n_vids)
n_val_v   = int(0.10 * n_vids)

train_vids = set(all_video_ids[:n_train_v])
val_vids   = set(all_video_ids[n_train_v : n_train_v + n_val_v])
test_vids  = set(all_video_ids[n_train_v + n_val_v:])

train_indices = [i for v in train_vids for i in video_to_indices[v]]
val_indices   = [i for v in val_vids   for i in video_to_indices[v]]
test_indices  = [i for v in test_vids  for i in video_to_indices[v]]

train_sub = Subset(full_dataset, train_indices)
val_sub   = Subset(full_dataset, val_indices)
test_sub  = Subset(full_dataset, test_indices)

print(f"\nVideo-level split:"
      f"\n  train videos: {len(train_vids):,}  frames: {len(train_indices):,}"
      f"\n  val   videos: {len(val_vids):,}  frames: {len(val_indices):,}"
      f"\n  test  videos: {len(test_vids):,}  frames: {len(test_indices):,}")

train_data = KDTransformWrapper(train_sub, is_train=True)
val_data   = KDTransformWrapper(val_sub,   is_train=False)
test_data  = KDTransformWrapper(test_sub,  is_train=False)

# Weighted sampler for class balance
train_labels_list = [full_dataset.labels[i] for i in train_indices]
class_counts      = [train_labels_list.count(0), train_labels_list.count(1)]
sample_weights    = [1.0 / class_counts[l] for l in train_labels_list]
sampler = WeightedRandomSampler(sample_weights,
                                num_samples=len(sample_weights),
                                replacement=True)

_pin = DEVICE.type == "cuda"
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=4, pin_memory=_pin,
                          persistent_workers=True, prefetch_factor=2)
val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=_pin,
                          persistent_workers=True, prefetch_factor=2)
test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=_pin,
                          persistent_workers=True, prefetch_factor=2)

print(f"DataLoaders ready  |  train real/fake: {class_counts[0]:,}/{class_counts[1]:,}")


# ─────────────────────────────────────────────────────────────────────────────
# TEACHER MODEL  (frozen DP-ViT)
# ─────────────────────────────────────────────────────────────────────────────

class DPViTTeacher(nn.Module):
    def __init__(self, dropout: float = 0.4):
        super().__init__()
        self.rgb_enc = timm.create_model('vit_base_patch16_224',
                                         pretrained=False, num_classes=0)
        self.hf_enc  = timm.create_model('vit_base_patch16_224',
                                         pretrained=False, num_classes=0)
        feat_dim  = self.rgb_enc.num_features   # 768
        fused_dim = feat_dim * 2                # 1536
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim), nn.Dropout(dropout),
            nn.Linear(fused_dim, 512), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(512, 1),
        )

    def _extract_vit_intermediates(self, enc, x, layers):
        x = enc.patch_embed(x)
        cls_tokens = enc.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = enc.pos_drop(x + enc.pos_embed)

        intermediates, attn_maps = [], []
        for i, block in enumerate(enc.blocks):
            if i in layers:
                B, N, C = x.shape
                qkv = block.attn.qkv(block.norm1(x))
                qkv = qkv.reshape(B, N, 3,
                                   block.attn.num_heads,
                                   C // block.attn.num_heads).permute(2, 0, 3, 1, 4)
                q, k = qkv[0], qkv[1]
                attn_w = (q @ k.transpose(-2, -1)) * block.attn.scale
                attn_w = attn_w.softmax(dim=-1)
                attn_maps.append(attn_w.detach())
            x = block(x)
            if i in layers:
                intermediates.append(enc.norm(x)[:, 0])  # CLS token [B, 768]

        cls_final = enc.norm(x)[:, 0]
        return cls_final, intermediates, attn_maps

    def forward_full(self, rgb, hf, extract_layers=TEACHER_ATTN_LAYERS):
        rgb_cls, rgb_inter, rgb_attn = self._extract_vit_intermediates(
            self.rgb_enc, rgb, extract_layers)
        hf_cls = self.hf_enc(hf)
        fused  = torch.cat([rgb_cls, hf_cls], dim=1)
        logit  = self.head(fused)
        return logit, fused, rgb_inter, rgb_attn

    def get_fused_features(self, rgb, hf):
        return torch.cat([self.rgb_enc(rgb), self.hf_enc(hf)], dim=1)

    def forward(self, rgb, hf):
        return self.head(self.get_fused_features(rgb, hf))


print(f"\nLoading teacher: {TEACHER_PATH}")
if not os.path.exists(TEACHER_PATH):
    raise FileNotFoundError(
        f"Teacher checkpoint not found: {TEACHER_PATH}\n"
        "Mount the Kaggle model 'syedazmulhasansabbir/dfdc-dp-vit'."
    )
teacher = DPViTTeacher(dropout=0.4).to(DEVICE)
raw = torch.load(TEACHER_PATH, map_location=DEVICE, weights_only=False)
sd  = raw['model_state'] if isinstance(raw, dict) and 'model_state' in raw else raw
if any(k.startswith('_orig_mod.') for k in sd):
    sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
teacher.load_state_dict(sd, strict=True)
teacher.eval()
for p in teacher.parameters():
    p.requires_grad_(False)
print("Teacher loaded and frozen.")

with torch.no_grad():
    _d = torch.zeros(2, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
    _f = teacher.get_fused_features(_d, _d)
    assert _f.shape == (2, TEACHER_FEAT_DIM), f"Feature dim mismatch: {_f.shape}"
print(f"Teacher feature dim confirmed: {TEACHER_FEAT_DIM}")


# ─────────────────────────────────────────────────────────────────────────────
# TEACHER LOGIT INSPECTION  (Fix C)
# ─────────────────────────────────────────────────────────────────────────────

def inspect_teacher_logits(loader: DataLoader, n_batches: int = 15) -> float:
    """
    Runs the teacher on the first n_batches of a loader and reports logit
    statistics.  Returns mean |logit| so the caller can decide whether the
    tanh clamp is active.

    Key diagnostic:
      E[sigmoid(z/T)] close to 0.5  → soft targets carry entropy  (good)
      E[sigmoid(z/T)] close to 0/1  → soft targets are hard labels (bad — increase T
                                        or rely on the tanh clamp)
    """
    teacher.eval()
    logit_vals = []
    with torch.no_grad():
        for i, (rgb_t, hf_t, *_) in enumerate(loader):
            if i >= n_batches:
                break
            rgb_t = rgb_t.to(DEVICE, non_blocking=True)
            hf_t  = hf_t.to(DEVICE,  non_blocking=True)
            t_logit, *_ = teacher.forward_full(rgb_t, hf_t)
            logit_vals.append(t_logit.cpu().float())

    logit_vals = torch.cat(logit_vals)
    mean_abs   = logit_vals.abs().mean().item()
    p95        = logit_vals.abs().quantile(0.95).item()
    mn, mx     = logit_vals.min().item(), logit_vals.max().item()

    print(f"\n── Teacher logit inspection (first {n_batches} batches) ──")
    print(f"   mean |logit| : {mean_abs:.3f}")
    print(f"   95th pct     : {p95:.3f}")
    print(f"   range        : [{mn:.3f}, {mx:.3f}]")
    print(f"   After T={KD_TEMPERATURE}: mean |z/T| = {mean_abs/KD_TEMPERATURE:.3f}")
    soft_mean = torch.sigmoid(logit_vals / KD_TEMPERATURE).mean().item()
    print(f"   E[sigmoid(z/T)]          = {soft_mean:.4f}  "
          f"(0.5 = perfectly soft, ~0 or ~1 = collapsed to hard targets)")
    if mean_abs > LOGIT_CLIP_WARNING:
        print(f"   ⚠  OVERCONFIDENT TEACHER — tanh clamp ACTIVE "
              f"(ceiling ±{LOGIT_CLIP:.1f})")
    else:
        print(f"   ✓  Teacher logits within safe range — tanh clamp is a no-op")
    return mean_abs


print("\nInspecting teacher logits before training ...")
teacher_mean_abs = inspect_teacher_logits(train_loader, n_batches=15)


# ─────────────────────────────────────────────────────────────────────────────
# STUDENT MODEL  v3  — Staged Unfreezing + Gated HF Fusion + Better Init
# ─────────────────────────────────────────────────────────────────────────────

# EfficientNet-B2 channel sizes for out_indices=(1,2,3,4):
#   index 0 → stage 2:  [B,  24, 56, 56]
#   index 1 → stage 3:  [B,  48, 28, 28]
#   index 2 → stage 4:  [B, 120, 14, 14]  ← 14×14 = 196 patches, matches ViT
#   index 3 → stage 5:  [B, 352,  7,  7]
_B2_CH      = [24, 48, 120, 352]
_AT_STAGE   = 2           # index of 14×14 stage in rgb_feats list
_AT_CH      = _B2_CH[_AT_STAGE]   # 120
_AT_PATCHES = 14 * 14             # 196


class DualPathB2Student(nn.Module):
    """
    v3 changes:
      • hf_gate_logit: learnable scalar, sigmoid-activated, initialised to
        sigmoid(-4.6) ≈ 0.01.  In Stage 1 the model sees essentially pure RGB.
        The gate is logged each epoch so growth is visible.
      • freeze_backbones / unfreeze_rgb_backbone / unfreeze_hf_backbone helpers
        let the training loop drive staged unfreezing cleanly.
      • proj_head, inter_projs, attn_proj all use Xavier / Kaiming init so the
        first L_feat value is near 0.5 rather than near 1.0.
    """

    def __init__(self, dropout: float = 0.35):
        super().__init__()

        # Dual-path backbones — pre-trained ImageNet weights
        self.rgb_backbone = timm.create_model(
            'efficientnet_b2', pretrained=True,
            features_only=True, out_indices=(1, 2, 3, 4))
        self.hf_backbone = timm.create_model(
            'efficientnet_b2', pretrained=True,
            features_only=True, out_indices=(1, 2, 3, 4))

        feat_dim  = _B2_CH[-1]    # 352
        fused_dim = feat_dim * 2  # 704

        # ── Fix B: Gated HF fusion ────────────────────────────────────────────
        # Raw gate logit starts at -4.6 → sigmoid(-4.6) ≈ 0.01.
        # The model bootstraps on RGB only; the HF branch contribution grows
        # as the gate logit climbs toward 0 and beyond.
        # Using a raw logit (not directly a weight) means the optimizer sees
        # an unbounded parameter and gradients flow smoothly through sigmoid.
        self.hf_gate_logit = nn.Parameter(torch.tensor(-4.6))

        # Classification head
        self.fusion = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(512, 1),
        )

        # ── Fix E: proj_head with Xavier init ────────────────────────────────
        # Xavier-uniform keeps initial projections in a reasonable scale
        # relative to the teacher's L2-normalised feature space, so L_feat
        # starts near ~0.5 instead of ~0.997.
        self.proj_head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, TEACHER_FEAT_DIM),
        )
        nn.init.xavier_uniform_(self.proj_head[1].weight)
        nn.init.zeros_(self.proj_head[1].bias)

        # RGB-only intermediate projections (one per backbone stage)
        self.inter_projs = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(ch), nn.Linear(ch, 768))
            for ch in _B2_CH
        ])
        for ip in self.inter_projs:
            nn.init.xavier_uniform_(ip[1].weight)
            nn.init.zeros_(ip[1].bias)

        # Spatial attention proxy — 1×1 conv collapses 120 channels → 1 at 14×14
        self.attn_proj = nn.Conv2d(_AT_CH, 1, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.attn_proj.weight, a=0)

    # ── Staged freeze / unfreeze helpers (Fix A) ─────────────────────────────

    def freeze_backbones(self):
        """Stage 1: lock both backbones — only heads train."""
        for p in self.rgb_backbone.parameters():
            p.requires_grad_(False)
        for p in self.hf_backbone.parameters():
            p.requires_grad_(False)

    def unfreeze_rgb_backbone(self):
        """Stage 2: release RGB backbone for fine-tuning at reduced lr."""
        for p in self.rgb_backbone.parameters():
            p.requires_grad_(True)

    def unfreeze_hf_backbone(self):
        """Stage 3: release HF backbone for fine-tuning at reduced lr."""
        for p in self.hf_backbone.parameters():
            p.requires_grad_(True)

    @property
    def hf_gate(self) -> float:
        """Current gate value in [0, 1]; logged each epoch."""
        return float(torch.sigmoid(self.hf_gate_logit).item())

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _gap(feat_map: torch.Tensor) -> torch.Tensor:
        """Global average pool [B, C, H, W] → [B, C]."""
        return feat_map.mean(dim=[2, 3])

    def _fuse(self, rgb_pool: torch.Tensor,
              hf_pool: torch.Tensor) -> torch.Tensor:
        """
        Gated concatenation.
        In Stage 1 (gate ≈ 0.01): fused ≈ [rgb_pool | 0.01 * hf_pool]
        The LayerNorm inside self.fusion will normalise the near-zero HF half,
        and the classification head quickly learns to ignore it via weights.
        As gate grows the HF features ramp in gradually.
        """
        gate = torch.sigmoid(self.hf_gate_logit)
        return torch.cat([rgb_pool, gate * hf_pool], dim=1)

    # ── Forward passes ────────────────────────────────────────────────────────

    def forward(self, rgb: torch.Tensor, hf: torch.Tensor) -> torch.Tensor:
        rgb_pool = self._gap(self.rgb_backbone(rgb)[-1])
        hf_pool  = self._gap(self.hf_backbone(hf)[-1])
        return self.fusion(self._fuse(rgb_pool, hf_pool))

    def get_all_outputs(self, rgb: torch.Tensor, hf: torch.Tensor):
        """
        Returns (logit, proj, inter_rgb_list, attn_map).

        logit          : [B, 1]       — classification output
        proj           : [B, 1536]    — projected to teacher feature space
        inter_rgb_list : list of [B, 768] — RGB-only stage projections
        attn_map       : [B, 196]     — spatial attention proxy (14×14 stage)
        """
        rgb_feats = self.rgb_backbone(rgb)
        hf_feats  = self.hf_backbone(hf)

        rgb_pool = self._gap(rgb_feats[-1])
        hf_pool  = self._gap(hf_feats[-1])
        fused    = self._fuse(rgb_pool, hf_pool)

        logit = self.fusion(fused)
        proj  = self.proj_head(fused)

        # RGB-only intermediate projections (matches teacher's rgb_inter)
        inter_rgb_list = [
            ip(self._gap(rgb_feats[i]))
            for i, ip in enumerate(self.inter_projs)
        ]

        # Spatial attention proxy from the 14×14 stage
        raw_attn = self.attn_proj(rgb_feats[_AT_STAGE])  # [B, 1, 14, 14]
        attn_map = raw_attn.flatten(1)                    # [B, 196]

        return logit, proj, inter_rgb_list, attn_map


student = DualPathB2Student(dropout=0.35).to(DEVICE)

# Shape sanity check
with torch.no_grad():
    _d = torch.zeros(2, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
    _logit, _proj, _inters, _attn = student.get_all_outputs(_d, _d)
    assert _logit.shape == (2, 1)
    assert _proj.shape  == (2, TEACHER_FEAT_DIM)
    assert len(_inters) == 4
    assert _attn.shape  == (2, _AT_PATCHES)
    print(f"\nStudent output shapes OK — logit: {_logit.shape}  "
          f"proj: {_proj.shape}  inter×4  attn: {_attn.shape}")

print(f"Initial HF gate: {student.hf_gate:.4f}  "
      f"(should be ≈ 0.01, confirms RGB-first startup)")

teacher_params = sum(p.numel() for p in teacher.parameters()) / 1e6
student_params = sum(p.numel() for p in student.parameters()) / 1e6
print(f"Teacher : {teacher_params:.1f} M (frozen)")
print(f"Student : {student_params:.1f} M (trainable)")
print(f"Ratio   : {teacher_params / student_params:.1f}× compression")


# ─────────────────────────────────────────────────────────────────────────────
# DYNAMIC LOSS WEIGHT SCHEDULE  (Fix D)
# ─────────────────────────────────────────────────────────────────────────────

def get_loss_weights(epoch: int) -> dict:
    """
    Returns a weight dict (sums to 1.0) appropriate for the current epoch.

    Stage 1 (epoch < FREEZE_EPOCHS):
        Only hard + soft + feat.  The projection heads need to converge
        before we ask the backbone to satisfy spatial constraints.

    Stage 2 (FREEZE_EPOCHS ≤ epoch < WARMUP_EPOCHS):
        Linearly ramp inter in; attn and rkd remain off.

    Stage 3 (epoch ≥ WARMUP_EPOCHS):
        Full budget.  Attn and rkd ramp in over the first 3 epochs of this
        phase so they don't destabilise a backbone that just unfroze.
    """
    if epoch < FREEZE_EPOCHS:
        # Head-alignment phase — minimal losses, stable gradients
        raw = dict(hard=0.50, soft=0.30, feat=0.20,
                   inter=0.0,  attn=0.0,  rkd=0.0)

    elif epoch < WARMUP_EPOCHS:
        # Intermediate distillation ramps in linearly
        t       = (epoch - FREEZE_EPOCHS) / max(1, WARMUP_EPOCHS - FREEZE_EPOCHS)
        w_inter = W_INTER_FINAL * t
        # Smoothly blend feat weight from 0.20 toward its final value
        w_feat  = 0.20 + (W_FEAT_FINAL - 0.20) * t
        w_hard  = 0.50 - (0.50 - W_HARD_FINAL) * t
        w_soft  = 0.30
        raw = dict(hard=w_hard, soft=w_soft, feat=w_feat,
                   inter=w_inter, attn=0.0, rkd=0.0)

    else:
        # Full budget; ramp attn and rkd in over first 3 epochs of Stage 3
        t = min(1.0, (epoch - WARMUP_EPOCHS) / 3.0)
        raw = dict(
            hard  = W_HARD_FINAL,
            soft  = W_SOFT_FINAL,
            feat  = W_FEAT_FINAL,
            inter = W_INTER_FINAL,
            attn  = W_ATTN_FINAL * t,
            rkd   = W_RKD_FINAL  * t,
        )

    # Always normalise so weights sum exactly to 1.0
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}


# ─────────────────────────────────────────────────────────────────────────────
# KD LOSS COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

class RelationalKDLoss(nn.Module):
    """
    Safe pairwise distance matching.
    Fix 3 (v2): mu clamped to 1e-6 to avoid NaN when features collapse.
    """
    def forward(self, s_feats: torch.Tensor,
                t_feats: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            td = self._pairwise_dist(t_feats.detach())
        sd = self._pairwise_dist(s_feats)
        return F.smooth_l1_loss(sd, td)

    @staticmethod
    def _pairwise_dist(feats: torch.Tensor) -> torch.Tensor:
        diff = feats.unsqueeze(0) - feats.unsqueeze(1)
        dist = diff.pow(2).sum(-1).sqrt()
        mu   = dist.mean().clamp(min=1e-6)
        return dist / mu


class AttentionTransferLoss(nn.Module):
    """
    Student spatial attention map vs ViT patch attention.

    Teacher: attn_map [B, H, 197, 197] → CLS-to-patch row [B, 196]
    Student: attn_map [B, 196]          from the 14×14 spatial stage

    Both sides L2-normalised before MSE so the loss is O(1).
    Valid because ViT 14×14 patches == CNN stage 14×14 spatial grid.
    """
    def forward(self,
                teacher_attn_list: list,
                student_attn: torch.Tensor) -> torch.Tensor:
        if not teacher_attn_list:
            return torch.tensor(0.0, device=student_attn.device)

        n = min(len(teacher_attn_list), 4)
        t_attn_raw = teacher_attn_list[min(_AT_STAGE, n - 1)]  # [B, H, 197, 197]

        # Mean over heads; CLS row (index 0); drop CLS column → 196 patches
        t_att = t_attn_raw.mean(dim=1)[:, 0, 1:].float()
        t_att = F.normalize(t_att, dim=1)

        s_att = F.normalize(student_attn.float(), dim=1)
        return F.mse_loss(s_att, t_att.detach())


# ─────────────────────────────────────────────────────────────────────────────
# RESEARCH KD LOSS  v3
# ─────────────────────────────────────────────────────────────────────────────

class ResearchKDLoss(nn.Module):
    """
    v3 additions on top of v2 fixes:
      • weights are passed per forward call (from get_loss_weights)
      • teacher logits are tanh-clamped via _safe_teacher_logit (Fix C)
      • attn and rkd are skipped (zero-cost) when their weights are 0
    """

    def __init__(self, T: float, smoothing: float = 0.05):
        super().__init__()
        self.T         = T
        self.smoothing = smoothing
        self.bce       = nn.BCEWithLogitsLoss()
        self.cos_loss  = nn.CosineEmbeddingLoss()
        self.rkd_loss  = RelationalKDLoss()
        self.attn_loss = AttentionTransferLoss()

    @staticmethod
    def _safe_teacher_logit(t_logit: torch.Tensor) -> torch.Tensor:
        """
        Fix C: tanh clamp.
        sigmoid(+6) = 0.9975, sigmoid(-6) = 0.0025 — genuinely soft.
        sigmoid(+15) ≈ 1.0 — collapses to a hard target, defeats KD.

        tanh(z / LOGIT_CLIP) * LOGIT_CLIP maps any z to
        (-LOGIT_CLIP, +LOGIT_CLIP) smoothly, so the gradient is non-zero
        everywhere and the soft targets always carry entropy.
        """
        c = LOGIT_CLIP
        return c * torch.tanh(t_logit.float() / c)

    def forward(self,
                s_logit:     torch.Tensor,   # [B, 1]
                s_proj:      torch.Tensor,   # [B, 1536]
                s_inter:     list,           # list of [B, 768]  (RGB-only)
                s_attn_map:  torch.Tensor,   # [B, 196]
                t_logit:     torch.Tensor,   # [B, 1]
                t_feats:     torch.Tensor,   # [B, 1536]
                t_inter:     list,           # list of [B, 768]  (RGB-only)
                t_attn:      list,           # list of [B, H, 197, 197]
                true_labels: torch.Tensor,   # [B, 1]
                weights:     dict,           # from get_loss_weights(epoch)
                ):

        # ── L_hard ────────────────────────────────────────────────────────────
        t_smooth = true_labels * (1.0 - self.smoothing) + 0.5 * self.smoothing
        L_hard   = self.bce(s_logit, t_smooth)

        # ── L_soft (no T² — v2 Fix 1 retained; tanh clamp added in v3) ───────
        t_logit_safe = self._safe_teacher_logit(t_logit)
        soft_targets = torch.sigmoid(t_logit_safe / self.T).detach()
        L_soft = F.binary_cross_entropy_with_logits(
            s_logit.float() / self.T, soft_targets
        )

        # ── L_feat ────────────────────────────────────────────────────────────
        ones   = torch.ones(s_proj.size(0), device=s_proj.device)
        L_feat = self.cos_loss(s_proj, t_feats.detach(), ones)

        # ── L_inter (L2-normalised, RGB-only) ─────────────────────────────────
        L_inter = torch.tensor(0.0, device=s_logit.device)
        n_inter = min(len(s_inter), len(t_inter))
        if n_inter > 0 and weights['inter'] > 0:
            for si, ti in zip(s_inter[:n_inter], t_inter[:n_inter]):
                si_n = F.normalize(si.float(), dim=1)
                ti_n = F.normalize(ti.float(), dim=1).detach()
                L_inter = L_inter + F.mse_loss(si_n, ti_n)
            L_inter = L_inter / n_inter

        # ── L_attn (skipped in Stages 1-2 when weight = 0) ───────────────────
        if weights['attn'] > 0:
            L_attn = self.attn_loss(t_attn, s_attn_map)
        else:
            L_attn = torch.tensor(0.0, device=s_logit.device)

        # ── L_rkd (skipped in Stages 1-2 when weight = 0) ────────────────────
        if weights['rkd'] > 0:
            L_rkd = self.rkd_loss(s_proj, t_feats)
        else:
            L_rkd = torch.tensor(0.0, device=s_logit.device)

        total = (weights['hard']  * L_hard
               + weights['soft']  * L_soft
               + weights['feat']  * L_feat
               + weights['inter'] * L_inter
               + weights['attn']  * L_attn
               + weights['rkd']   * L_rkd)

        return total, {
            'L_hard':  L_hard.item(),
            'L_soft':  L_soft.item(),
            'L_feat':  L_feat.item(),
            'L_inter': L_inter.item(),
            'L_attn':  L_attn.item(),
            'L_rkd':   L_rkd.item(),
        }


kd_criterion = ResearchKDLoss(T=KD_TEMPERATURE, smoothing=LABEL_SMOOTHING)


# ─────────────────────────────────────────────────────────────────────────────
# OPTIMISER  — Four parameter groups for staged unfreezing  (Fix A)
# ─────────────────────────────────────────────────────────────────────────────
#
# Group 0  rgb_backbone  : lr starts at 0 (frozen); set to LR/4 at Stage 2,
#                          LR/2 at Stage 3.
# Group 1  hf_backbone   : lr starts at 0 (frozen); set to LR/4 at Stage 3.
# Group 2  all heads     : lr = LEARNING_RATE throughout (cosine with warmup).
# Group 3  hf_gate_logit : lr = LEARNING_RATE × 0.1 (slow gate movement so
#                          it doesn't jump to 1.0 before features are aligned).
#
# Setting lr=0 for frozen groups prevents Adam from building up stale
# momentum while those parameters are inactive.

head_params = (list(student.fusion.parameters()) +
               list(student.proj_head.parameters()) +
               list(student.inter_projs.parameters()) +
               list(student.attn_proj.parameters()))

optimizer = optim.AdamW([
    {'params': student.rgb_backbone.parameters(), 'lr': 0.0,
     'name': 'rgb_bb'},
    {'params': student.hf_backbone.parameters(),  'lr': 0.0,
     'name': 'hf_bb'},
    {'params': head_params,                        'lr': LEARNING_RATE,
     'name': 'heads'},
    {'params': [student.hf_gate_logit],            'lr': LEARNING_RATE * 0.1,
     'name': 'gate'},
], weight_decay=WEIGHT_DECAY)


def set_backbone_lr(rgb_lr: float, hf_lr: float):
    """Adjust backbone learning rates at stage boundaries."""
    optimizer.param_groups[0]['lr'] = rgb_lr
    optimizer.param_groups[1]['lr'] = hf_lr
    print(f"  Backbone lrs updated → rgb={rgb_lr:.2e}  hf={hf_lr:.2e}")


# Cosine scheduler with linear warmup applied to the head group only.
# Groups 0, 1, 3 get constant lambdas (their lrs are managed manually).
def lr_lambda_head(epoch: int) -> float:
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS
    progress = (epoch - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[
    lambda e: 1.0,      # rgb_bb  — managed manually
    lambda e: 1.0,      # hf_bb   — managed manually
    lr_lambda_head,     # heads   — cosine with warmup
    lambda e: 1.0,      # gate    — constant
])

scaler = torch.cuda.amp.GradScaler()

# Freeze backbones before Stage 1 begins
student.freeze_backbones()
trainable_params = sum(p.numel() for p in student.parameters()
                       if p.requires_grad) / 1e6
print(f"\nStage 1 start — both backbones frozen")
print(f"Trainable params (heads only): {trainable_params:.1f} M")


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_loader(loader: DataLoader, desc: str = "VAL",
                    epoch_weights: dict = None):
    """Returns (avg_loss, accuracy, auc)."""
    if epoch_weights is None:
        epoch_weights = get_loss_weights(WARMUP_EPOCHS)  # full-budget weights

    student.eval()
    total_loss, correct, total_n = 0.0, 0, 0
    all_probs, all_labels_ev = [], []

    with torch.no_grad():
        for rgb_t, hf_t, rgb_s, hf_s, labels in tqdm(loader, desc=desc, leave=False):
            rgb_t  = rgb_t.to(DEVICE, non_blocking=True)
            hf_t   = hf_t.to(DEVICE,  non_blocking=True)
            rgb_s  = rgb_s.to(DEVICE, non_blocking=True)
            hf_s   = hf_s.to(DEVICE,  non_blocking=True)
            labels = labels.unsqueeze(1).to(DEVICE, non_blocking=True)

            with torch.amp.autocast(device_type='cuda'):
                s_logit, s_proj, s_inter, s_attn = student.get_all_outputs(rgb_s, hf_s)
                t_logit, t_feats, t_inter, t_attn = teacher.forward_full(rgb_t, hf_t)
                loss, _ = kd_criterion(
                    s_logit, s_proj, s_inter, s_attn,
                    t_logit, t_feats, t_inter, t_attn, labels,
                    weights=epoch_weights,
                )

            if not math.isnan(loss.item()):
                total_loss += loss.item()

            probs  = torch.sigmoid(s_logit).squeeze(1).cpu().numpy()
            preds  = (probs >= 0.5).astype(int)
            all_probs.extend(probs.tolist())
            all_labels_ev.extend(labels.squeeze(1).cpu().numpy().astype(int).tolist())
            correct += (preds == labels.squeeze(1).cpu().numpy()).sum()
            total_n += labels.size(0)

    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
    acc      = correct / total_n if total_n > 0 else 0.0
    try:
        auc = roc_auc_score(all_labels_ev, all_probs)
    except ValueError:
        auc = float('nan')
    return avg_loss, acc, auc


def run_inference(loader: DataLoader, desc: str = "Infer"):
    student.eval()
    labels_all, probs_all = [], []
    with torch.no_grad():
        for rgb_t, hf_t, rgb_s, hf_s, labels in tqdm(loader, desc=desc, leave=False):
            rgb_s = rgb_s.to(DEVICE, non_blocking=True)
            hf_s  = hf_s.to(DEVICE,  non_blocking=True)
            with torch.amp.autocast(device_type='cuda'):
                logit = student(rgb_s, hf_s)
            probs = torch.sigmoid(logit).squeeze(1).cpu().numpy()
            probs_all.extend(probs.tolist())
            labels_all.extend(labels.numpy().tolist())
    return np.array(labels_all), np.array(probs_all)


def compute_frame_metrics(labels, probs, threshold=0.5):
    preds           = (probs >= threshold).astype(int)
    tn, fp, fn, tp  = confusion_matrix(labels, preds).ravel()
    return {
        'frame_accuracy':    accuracy_score(labels, preds),
        'frame_f1':          f1_score(labels, preds, zero_division=0),
        'frame_precision':   precision_score(labels, preds, zero_division=0),
        'frame_recall':      recall_score(labels, preds, zero_division=0),
        'frame_specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'frame_auc_roc':     roc_auc_score(labels, probs),
        'frame_ap':          average_precision_score(labels, probs),
        'frame_brier':       brier_score_loss(labels, probs),
        'frame_tp': int(tp), 'frame_fp': int(fp),
        'frame_tn': int(tn), 'frame_fn': int(fn),
    }


def compute_ece(labels, probs, n_bins=10):
    bins  = np.linspace(0, 1, n_bins + 1)
    ece   = 0.0
    total = len(labels)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs > lo) & (probs <= hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() / total * abs(labels[mask].mean() - probs[mask].mean())
    return float(ece)


def compute_video_metrics(labels, probs, subset, threshold=0.5):
    _nan = float('nan')
    video_probs:  dict = defaultdict(list)
    video_labels: dict = defaultdict(list)
    for i, gi in enumerate(subset.indices):
        path   = full_dataset.all_paths[gi]
        vid_id = full_dataset.path_to_video.get(path, f"unk_{gi}")
        video_probs[vid_id].append(probs[i])
        video_labels[vid_id].append(labels[i])

    vid_true, vid_prob_mean = [], []
    for vid_id in video_probs:
        vid_true.append(1 if 1 in set(video_labels[vid_id]) else 0)
        vid_prob_mean.append(float(np.mean(video_probs[vid_id])))

    vid_true      = np.array(vid_true)
    vid_prob_mean = np.array(vid_prob_mean)
    vid_pred      = (vid_prob_mean >= threshold).astype(int)
    n_cls         = len(np.unique(vid_true))
    return {
        'video_count':     len(vid_true),
        'video_accuracy':  accuracy_score(vid_true, vid_pred),
        'video_f1':        f1_score(vid_true, vid_pred, zero_division=0),
        'video_precision': precision_score(vid_true, vid_pred, zero_division=0),
        'video_recall':    recall_score(vid_true, vid_pred, zero_division=0),
        'video_auc_roc':   roc_auc_score(vid_true, vid_prob_mean) if n_cls > 1 else _nan,
        'video_ap':        average_precision_score(vid_true, vid_prob_mean) if n_cls > 1 else _nan,
    }, vid_true, vid_prob_mean


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP  v3
# ─────────────────────────────────────────────────────────────────────────────

_nan = float('nan')

best_val_auc      = 0.0
epochs_no_improve = 0

history: dict = {k: [_nan] * EPOCHS for k in [
    'train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_auc',
    'lr_heads', 'hf_gate',
    'L_hard', 'L_soft', 'L_feat', 'L_inter', 'L_attn', 'L_rkd',
    'w_hard',  'w_soft',  'w_feat', 'w_inter', 'w_attn', 'w_rkd',
]}

timed_out = False

print(f"\n{'='*70}")
print(f"  v3 KD  |  3-stage unfreezing  |  gated HF  |  dynamic weights")
print(f"  Stage 1 (ep 1–{FREEZE_EPOCHS}):   backbones FROZEN  — heads only")
print(f"  Stage 2 (ep {FREEZE_EPOCHS+1}–{WARMUP_EPOCHS}): RGB backbone  — LR/4")
print(f"  Stage 3 (ep {WARMUP_EPOCHS+1}+):  both backbones — RGB:LR/2  HF:LR/4")
print(f"{'='*70}\n")

for epoch in range(EPOCHS):
    if timed_out:
        break

    # ── Stage transitions ─────────────────────────────────────────────────────
    if epoch == FREEZE_EPOCHS:
        print(f"\n{'─'*70}")
        print(f"  ▶ Stage 2 start (epoch {epoch+1}): unfreezing RGB backbone at LR/4")
        student.unfreeze_rgb_backbone()
        set_backbone_lr(rgb_lr=LEARNING_RATE / 4, hf_lr=0.0)
        trainable_now = sum(p.numel() for p in student.parameters()
                            if p.requires_grad) / 1e6
        print(f"  Trainable now: {trainable_now:.1f} M")

    if epoch == WARMUP_EPOCHS:
        print(f"\n{'─'*70}")
        print(f"  ▶ Stage 3 start (epoch {epoch+1}): "
              f"RGB → LR/2, HF backbone unfrozen at LR/4")
        student.unfreeze_hf_backbone()
        set_backbone_lr(rgb_lr=LEARNING_RATE / 2, hf_lr=LEARNING_RATE / 4)
        trainable_now = sum(p.numel() for p in student.parameters()
                            if p.requires_grad) / 1e6
        print(f"  Trainable now: {trainable_now:.1f} M")

    # ── Loss weights for this epoch ───────────────────────────────────────────
    epoch_weights = get_loss_weights(epoch)

    # ── Train ─────────────────────────────────────────────────────────────────
    student.train()
    run     = defaultdict(float)
    correct = total_n = 0
    pbar    = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [TRAIN]")

    for rgb_t, hf_t, rgb_s, hf_s, labels in pbar:
        if time.time() - START_TIME > MAX_RUN_TIME:
            print("\n⏱️  Time limit — saving emergency checkpoint ...")
            timed_out = True
            break

        rgb_t  = rgb_t.to(DEVICE,  non_blocking=True)
        hf_t   = hf_t.to(DEVICE,   non_blocking=True)
        rgb_s  = rgb_s.to(DEVICE,  non_blocking=True)
        hf_s   = hf_s.to(DEVICE,   non_blocking=True)
        labels = labels.unsqueeze(1).to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type='cuda'):
            s_logit, s_proj, s_inter, s_attn = student.get_all_outputs(rgb_s, hf_s)
            with torch.no_grad():
                t_logit, t_feats, t_inter, t_attn = teacher.forward_full(rgb_t, hf_t)
            loss, parts = kd_criterion(
                s_logit, s_proj, s_inter, s_attn,
                t_logit, t_feats, t_inter, t_attn, labels,
                weights=epoch_weights,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        bs = labels.size(0)
        run['loss'] += loss.item()
        for k, v in parts.items():
            run[k] += v
        preds    = (torch.sigmoid(s_logit.detach()) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total_n += bs

        n_b = max(total_n / bs, 1)
        pbar.set_postfix(
            loss=f"{run['loss']/n_b:.4f}",
            acc=f"{correct/total_n:.4f}",
            Lh=f"{run['L_hard']/n_b:.3f}",
            Ls=f"{run['L_soft']/n_b:.3f}",
            gate=f"{student.hf_gate:.3f}",
        )

    if timed_out:
        fname = OUT_DIR / f'EMERGENCY_student_epoch{epoch+1:02d}.pth'
        torch.save({'epoch': epoch+1,
                    'state_dict':     student.state_dict(),
                    'optimizer':      optimizer.state_dict(),
                    'val_auc':        best_val_auc,
                    'history':        history,
                    'epoch_weights':  epoch_weights}, fname)
        print(f"Emergency checkpoint saved: {fname.name}")
        break

    # ── Validate ──────────────────────────────────────────────────────────────
    n_b        = len(train_loader)
    train_loss = run['loss'] / n_b
    train_acc  = correct / total_n

    val_loss, val_acc, val_auc = evaluate_loader(
        val_loader, desc=f"Epoch {epoch+1} [VAL]",
        epoch_weights=epoch_weights)
    scheduler.step()

    lr_heads = optimizer.param_groups[2]['lr']
    hf_gate  = student.hf_gate

    # ── Record history ────────────────────────────────────────────────────────
    for k, v in [('train_loss', train_loss), ('train_acc', train_acc),
                 ('val_loss', val_loss),   ('val_acc', val_acc),
                 ('val_auc', val_auc),     ('lr_heads', lr_heads),
                 ('hf_gate', hf_gate)]:
        history[k][epoch] = v
    for k in ['L_hard', 'L_soft', 'L_feat', 'L_inter', 'L_attn', 'L_rkd']:
        history[k][epoch] = run[k] / n_b
    for k, v in epoch_weights.items():
        history[f'w_{k}'][epoch] = v

    # ── Console report ────────────────────────────────────────────────────────
    stage_tag = ("S1-heads-only"  if epoch < FREEZE_EPOCHS else
                 "S2-rgb-unfrz"   if epoch < WARMUP_EPOCHS else
                 "S3-full")
    print(f"\nEpoch {epoch+1:02d}/{EPOCHS}  [{stage_tag}]")
    print(f"  train: loss={train_loss:.4f}  acc={train_acc:.4f}")
    print(f"  val:   loss={val_loss:.4f}   acc={val_acc:.4f}  auc={val_auc:.4f}")
    print(f"  gate={hf_gate:.4f}  lr_heads={lr_heads:.2e}  "
          f"lr_rgb={optimizer.param_groups[0]['lr']:.2e}  "
          f"lr_hf={optimizer.param_groups[1]['lr']:.2e}")
    print(f"  weights: "
          + "  ".join(f"{k}={v:.3f}" for k, v in epoch_weights.items()))
    print(f"  losses:  "
          + "  ".join(f"{k[2:]}={run[k]/n_b:.3f}"
                      for k in ['L_hard','L_soft','L_feat',
                                 'L_inter','L_attn','L_rkd']))

    # ── Collapse guard ────────────────────────────────────────────────────────
    # After Stage 1 the projection heads should have aligned.
    # L_feat > 0.90 at this point means they have not — warn and advise.
    if epoch == FREEZE_EPOCHS - 1:
        l_feat_val = run['L_feat'] / n_b
        if l_feat_val > 0.90:
            print(f"\n  ⚠  L_feat={l_feat_val:.4f} after Stage 1 — "
                  f"projection heads have not aligned with the teacher.\n"
                  f"     Recommendations:\n"
                  f"     1. Increase FREEZE_EPOCHS to 3-4.\n"
                  f"     2. Run inspect_teacher_logits() — if mean|logit|>10, "
                  f"       increase LOGIT_CLIP to 8.0.\n"
                  f"     3. Check that teacher features are not all-zero "
                  f"       (run teacher.get_fused_features on a batch).")
        else:
            print(f"\n  ✓  L_feat={l_feat_val:.4f} after Stage 1 — "
                  f"projection aligned, safe to unfreeze RGB backbone.")

    # ── Checkpoint ────────────────────────────────────────────────────────────
    ckpt_path = OUT_DIR / f'student_epoch{epoch+1:02d}_auc{val_auc:.4f}.pth'
    torch.save({'epoch':         epoch + 1,
                'state_dict':    student.state_dict(),
                'optimizer':     optimizer.state_dict(),
                'val_auc':       val_auc,
                'history':       history,
                'epoch_weights': epoch_weights,
                'hf_gate':       hf_gate}, ckpt_path)

    if val_auc > best_val_auc:
        best_val_auc      = val_auc
        epochs_no_improve = 0
        torch.save(student.state_dict(), OUT_DIR / 'best_student.pth')
        print(f"  ⭐  New best  AUC={best_val_auc:.4f} → best_student.pth")
    else:
        epochs_no_improve += 1
        # Early stopping is suspended during Stage 1: the val AUC may plateau
        # while backbones are frozen — that is expected and not a failure.
        if epoch >= FREEZE_EPOCHS and epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch+1}.")
            break
        elif epoch < FREEZE_EPOCHS:
            print(f"  (early stopping suspended during Stage 1 "
                  f"— no-improve streak: {epochs_no_improve})")
        else:
            print(f"  No improvement ({epochs_no_improve}/{PATIENCE})")

    print("-" * 70)

print(f"\n✅ Training complete  |  best val AUC: {best_val_auc:.4f}")
print(f"   Final HF gate  : {student.hf_gate:.4f}  "
      f"(1.0 = both paths fully active)")
print(f"   Elapsed        : {(time.time()-START_TIME)/3600:.2f} h")


# ─────────────────────────────────────────────────────────────────────────────
# TEST EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{'='*70}")
print("  Loading best_student.pth for final test evaluation ...")
best_path = OUT_DIR / 'best_student.pth'
if best_path.exists():
    student.load_state_dict(torch.load(best_path, map_location=DEVICE))
    print("  Loaded best_student.pth")
else:
    print("  best_student.pth not found — using current weights.")

test_labels, test_probs = run_inference(test_loader,  desc="TEST  inference")
val_labels,  val_probs  = run_inference(val_loader,   desc="VAL   inference")

# Optimal threshold via Youden's J on the validation set
fpr_val, tpr_val, thresh_val = roc_curve(val_labels, val_probs)
best_thresh = float(thresh_val[np.argmax(tpr_val - fpr_val)])
print(f"\n  Optimal threshold (Youden's J on val): {best_thresh:.4f}")

frame_metrics = compute_frame_metrics(test_labels, test_probs,
                                      threshold=best_thresh)
ece           = compute_ece(test_labels, test_probs)

print(f"\n── Frame-level metrics ──")
for k, v in frame_metrics.items():
    print(f"  {k:<25}: {v:.4f}" if isinstance(v, float) else f"  {k:<25}: {v}")
print(f"  {'frame_ece':<25}: {ece:.4f}")

video_metrics, vid_true, vid_prob_mean = compute_video_metrics(
    test_labels, test_probs, test_sub, threshold=best_thresh)

print(f"\n── Video-level metrics ──")
for k, v in video_metrics.items():
    print(f"  {k:<25}: {v:.4f}" if isinstance(v, float) else f"  {k:<25}: {v}")

preds_test = (test_probs >= best_thresh).astype(int)
print("\n── Classification Report (frame-level) ──")
print(classification_report(test_labels, preds_test,
                             target_names=['REAL', 'FAKE'], digits=4))


# ─────────────────────────────────────────────────────────────────────────────
# EXPORT CSVs
# ─────────────────────────────────────────────────────────────────────────────

all_metrics_dict = {
    'model':               'EfficientNet-B2 DualPath KD v3-COLLAPSE-FIX',
    'teacher':             'DPViT-DFDC',
    'kd_temperature':      KD_TEMPERATURE,
    'logit_clip':          LOGIT_CLIP,
    'freeze_epochs':       FREEZE_EPOCHS,
    'warmup_epochs':       WARMUP_EPOCHS,
    'final_hf_gate':       student.hf_gate,
    'w_hard_final':  W_HARD_FINAL,  'w_soft_final':  W_SOFT_FINAL,
    'w_feat_final':  W_FEAT_FINAL,  'w_inter_final': W_INTER_FINAL,
    'w_attn_final':  W_ATTN_FINAL,  'w_rkd_final':   W_RKD_FINAL,
    'optimal_threshold': best_thresh,
    'frame_ece':         ece,
    **{f'test_{k}': v for k, v in frame_metrics.items()},
    **{f'test_{k}': v for k, v in video_metrics.items()},
    'best_val_auc':     best_val_auc,
    'teacher_mean_abs_logit': teacher_mean_abs,
}
pd.DataFrame([all_metrics_dict]).to_csv(
    OUT_DIR / 'kd_metrics_summary.csv', index=False)

valid_ep = [e for e in range(EPOCHS) if not math.isnan(history['train_loss'][e])]
hist_df  = pd.DataFrame({k: [history[k][e] for e in valid_ep] for k in history})
hist_df.insert(0, 'epoch', [e + 1 for e in valid_ep])
hist_df.to_csv(OUT_DIR / 'kd_training_history.csv', index=False)

pd.DataFrame({
    'file_path':  [full_dataset.all_paths[i] for i in test_sub.indices],
    'true_label': test_labels.astype(int),
    'prob_fake':  np.round(test_probs, 6),
    'pred_label': preds_test,
    'correct':    (test_labels == preds_test).astype(int),
}).to_csv(OUT_DIR / 'kd_test_frame_predictions.csv', index=False)

print("\nSaved: kd_metrics_summary.csv  |  kd_training_history.csv  "
      "|  kd_test_frame_predictions.csv")


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    'figure.facecolor':  '#0a0e1a', 'axes.facecolor':    '#111827',
    'axes.edgecolor':    '#2d3748', 'axes.labelcolor':   '#e2e8f0',
    'text.color':        '#e2e8f0', 'xtick.color':       '#94a3b8',
    'ytick.color':       '#94a3b8', 'grid.color':        '#1e2a3a',
    'grid.linestyle':    '--',      'grid.alpha':        0.7,
    'font.family':       'DejaVu Sans', 'font.size':     11,
    'axes.titlesize':    13,        'axes.titleweight':  'bold',
    'legend.facecolor':  '#111827', 'legend.edgecolor':  '#2d3748',
    'savefig.facecolor': '#0a0e1a', 'savefig.dpi':       200,
    'savefig.bbox':      'tight',
})
C = dict(blue='#38bdf8', red='#f87171', green='#4ade80',
         yellow='#fbbf24', purple='#c084fc', orange='#fb923c', grey='#475569')
epochs_x = hist_df['epoch'].tolist()

# ── Fig 1: Loss curves ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 5))
ax = axes[0]
ax.plot(epochs_x, hist_df['train_loss'], color=C['blue'],  lw=2.5,
        marker='o', ms=5, label='Train')
ax.plot(epochs_x, hist_df['val_loss'],   color=C['red'],   lw=2.5,
        marker='s', ms=5, label='Val', ls='--')
# Mark stage boundaries
for ep, label in [(FREEZE_EPOCHS, 'S2'), (WARMUP_EPOCHS, 'S3')]:
    if ep < len(epochs_x):
        ax.axvline(ep + 0.5, color=C['grey'], lw=1.2, ls=':', alpha=0.7)
        ax.text(ep + 0.6, ax.get_ylim()[1] * 0.95, label,
                color=C['grey'], fontsize=9)
ax.set_title('Total KD Loss'); ax.set_xlabel('Epoch'); ax.legend(); ax.grid(True)

ax = axes[1]
for col, mk, key, lbl in zip(
    [C['blue'], C['green'], C['yellow'], C['purple'], C['orange'], C['red']],
    ['o', 's', 'D', '^', 'v', 'P'],
    ['L_hard','L_soft','L_feat','L_inter','L_attn','L_rkd'],
    ['L_hard', 'L_soft (no T²)', 'L_feat', 'L_inter (norm)',
     'L_attn (spatial)', 'L_rkd (safe)'],
):
    ax.plot(epochs_x, hist_df[key], color=col, lw=2, marker=mk, ms=4, label=lbl)
ax.set_title('KD Loss Components')
ax.set_xlabel('Epoch'); ax.legend(fontsize=8); ax.grid(True)
fig.suptitle('DFDC KD v3 — Loss Curves', fontsize=14)
fig.tight_layout()
fig.savefig(OUT_DIR / 'fig1_loss_curves.png')
plt.close(fig)

# ── Fig 2: Accuracy + AUC ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
ax = axes[0]
ax.plot(epochs_x, hist_df['train_acc'], color=C['green'],  lw=2.5,
        marker='o', ms=5, label='Train')
ax.plot(epochs_x, hist_df['val_acc'],   color=C['yellow'], lw=2.5,
        marker='s', ms=5, label='Val', ls='--')
ax.axhline(frame_metrics['frame_accuracy'], color=C['red'], lw=1.8, ls=':',
           label=f"Test Acc ({frame_metrics['frame_accuracy']:.4f})")
ax.set_title('Accuracy'); ax.set_xlabel('Epoch'); ax.set_ylim(0, 1.05)
ax.legend(); ax.grid(True)

ax = axes[1]
ax.plot(epochs_x, hist_df['val_auc'],  color=C['purple'], lw=2.5,
        marker='^', ms=5, label='Val AUC')
ax.axhline(frame_metrics['frame_auc_roc'], color=C['red'], lw=1.8, ls=':',
           label=f"Test AUC ({frame_metrics['frame_auc_roc']:.4f})")
ax.set_title('Val AUC-ROC (Primary Metric)'); ax.set_xlabel('Epoch')
ax.set_ylim(0.5, 1.02); ax.legend(); ax.grid(True)
fig.tight_layout()
fig.savefig(OUT_DIR / 'fig2_accuracy_auc.png')
plt.close(fig)

# ── Fig 3: HF gate + loss weights over time (v3-specific) ────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
ax = axes[0]
ax.plot(epochs_x, hist_df['hf_gate'], color=C['orange'], lw=2.5,
        marker='D', ms=5)
ax.axhline(0.5, color=C['grey'], lw=1.2, ls=':', alpha=0.6,
           label='gate = 0.5')
for ep, label in [(FREEZE_EPOCHS, 'S2'), (WARMUP_EPOCHS, 'S3')]:
    if ep < len(epochs_x):
        ax.axvline(ep + 0.5, color=C['grey'], lw=1.2, ls=':', alpha=0.7)
        ax.text(ep + 0.6, 0.05, label, color=C['grey'], fontsize=9)
ax.set_title('HF Gate Value (sigmoid of learnable logit)')
ax.set_xlabel('Epoch'); ax.set_ylabel('Gate  (0=RGB-only, 1=full HF)')
ax.set_ylim(-0.05, 1.05); ax.legend(); ax.grid(True)

ax = axes[1]
wkeys = ['w_hard', 'w_soft', 'w_feat', 'w_inter', 'w_attn', 'w_rkd']
wlbls = ['hard', 'soft', 'feat', 'inter', 'attn', 'rkd']
cols  = [C['blue'], C['green'], C['yellow'], C['purple'], C['orange'], C['red']]
for wk, wl, col in zip(wkeys, wlbls, cols):
    ax.plot(epochs_x, hist_df[wk], label=wl, color=col, lw=2, marker='o', ms=3)
ax.set_title('Dynamic Loss Weights per Epoch')
ax.set_xlabel('Epoch'); ax.set_ylabel('Weight')
ax.legend(fontsize=9); ax.grid(True)
fig.suptitle('v3 — Gate Growth & Weight Schedule', fontsize=14)
fig.tight_layout()
fig.savefig(OUT_DIR / 'fig3_gate_weights.png')
plt.close(fig)

# ── Figs 4–8: standard evaluation plots ──────────────────────────────────────
cm       = confusion_matrix(test_labels, preds_test)
cm_norm  = cm.astype(float) / cm.sum(axis=1, keepdims=True)
fpr_f, tpr_f, _ = roc_curve(test_labels, test_probs)
auc_f            = roc_auc_score(test_labels, test_probs)
prec_f, rec_f, _ = precision_recall_curve(test_labels, test_probs)
ap_f              = average_precision_score(test_labels, test_probs)
prob_true, prob_pred = calibration_curve(test_labels, test_probs,
                                         n_bins=10, strategy='uniform')
real_probs_arr = test_probs[test_labels == 0]
fake_probs_arr = test_probs[test_labels == 1]

# Fig 4: Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, data, fmt, title in zip(
        axes, [cm, cm_norm], ['d', '.3f'], ['Raw Counts', 'Normalised']):
    sns.heatmap(data, annot=True, fmt=fmt, ax=ax, cmap='Blues',
                xticklabels=['REAL', 'FAKE'], yticklabels=['REAL', 'FAKE'],
                linewidths=0.5, linecolor='#2d3748',
                annot_kws={'size': 14, 'weight': 'bold'})
    ax.set_title(f'Confusion Matrix — {title}')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
fig.suptitle('Frame-Level Confusion Matrix  [DFDC KD v3]',
             fontsize=14, y=1.02)
fig.tight_layout(); fig.savefig(OUT_DIR / 'fig4_confusion_matrix.png')
plt.close(fig)

# Fig 5: ROC
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(fpr_f, tpr_f, color=C['blue'], lw=2.5,
        label=f'Frame AUC = {auc_f:.4f}')
_nan = float('nan')
if not math.isnan(video_metrics['video_auc_roc']):
    fpr_v, tpr_v, _ = roc_curve(vid_true, vid_prob_mean)
    ax.plot(fpr_v, tpr_v, color=C['green'], lw=2.5, ls='--',
            label=f"Video AUC = {video_metrics['video_auc_roc']:.4f}")
ax.plot([0, 1], [0, 1], color=C['grey'], lw=1.5, ls=':')
opt_idx = np.argmax(tpr_f - fpr_f)
ax.scatter(fpr_f[opt_idx], tpr_f[opt_idx], color=C['red'], s=120, zorder=5,
           label=f'Optimal (τ={best_thresh:.3f})')
ax.set_title('ROC-AUC'); ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.05)
ax.legend(loc='lower right'); ax.grid(True)
fig.tight_layout(); fig.savefig(OUT_DIR / 'fig5_roc_auc.png')
plt.close(fig)

# Fig 6: Precision-Recall
fig, ax = plt.subplots(figsize=(8, 7))
ax.plot(rec_f, prec_f, color=C['yellow'], lw=2.5,
        label=f'Frame AP = {ap_f:.4f}')
if not math.isnan(video_metrics['video_ap']):
    prec_v, rec_v, _ = precision_recall_curve(vid_true, vid_prob_mean)
    ax.plot(rec_v, prec_v, color=C['green'], lw=2.5, ls='--',
            label=f"Video AP = {video_metrics['video_ap']:.4f}")
ax.axhline(test_labels.mean(), color=C['grey'], lw=1.5, ls=':',
           label=f'Baseline = {test_labels.mean():.3f}')
ax.set_title('Precision-Recall')
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.05)
ax.legend(); ax.grid(True)
fig.tight_layout(); fig.savefig(OUT_DIR / 'fig6_precision_recall.png')
plt.close(fig)

# Fig 7: Score distribution
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(real_probs_arr, bins=60, color=C['green'], alpha=0.65,
        label='REAL', density=True)
ax.hist(fake_probs_arr, bins=60, color=C['red'],   alpha=0.65,
        label='FAKE', density=True)
ax.axvline(best_thresh, color=C['yellow'], lw=2.5, ls='--',
           label=f'τ = {best_thresh:.3f}')
ax.set_title('Predicted Probability Distribution')
ax.set_xlabel('P(fake)'); ax.legend(); ax.grid(True)
fig.tight_layout(); fig.savefig(OUT_DIR / 'fig7_score_distribution.png')
plt.close(fig)

# Fig 8: Calibration
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ax = axes[0]
ax.plot(prob_pred, prob_true, color=C['blue'], lw=2.5, marker='o', ms=6,
        label='Student')
ax.plot([0, 1], [0, 1], color=C['grey'], lw=1.5, ls='--', label='Perfect')
ax.fill_between(prob_pred, prob_true, prob_pred, alpha=0.15,
                color=C['red'], label=f'ECE = {ece:.4f}')
ax.set_title('Reliability Diagram')
ax.set_xlabel('Mean Predicted Prob'); ax.legend(); ax.grid(True)
ax = axes[1]
bins = np.linspace(0, 1, 11)
hR, _ = np.histogram(real_probs_arr, bins=bins)
hF, _ = np.histogram(fake_probs_arr, bins=bins)
bc = 0.5 * (bins[:-1] + bins[1:]); w = 0.04
ax.bar(bc - w, hR, width=w*2, color=C['green'], alpha=0.7, label='REAL')
ax.bar(bc + w, hF, width=w*2, color=C['red'],   alpha=0.7, label='FAKE')
ax.axvline(best_thresh, color=C['yellow'], lw=2, ls='--',
           label=f'τ={best_thresh:.3f}')
ax.set_title(f'Score Histogram  Brier={frame_metrics["frame_brier"]:.4f}')
ax.set_xlabel('P(fake)'); ax.legend(); ax.grid(True)
fig.suptitle('Calibration Analysis  [DFDC KD v3]', fontsize=14)
fig.tight_layout(); fig.savefig(OUT_DIR / 'fig8_calibration.png')
plt.close(fig)

# ── Fig 9: Summary dashboard ──────────────────────────────────────────────────
_nan_str = lambda v: f"{v:.4f}" if not math.isnan(v) else 'N/A'

fig = plt.figure(figsize=(26, 18))
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.50, wspace=0.38)

ax0 = fig.add_subplot(gs[0, 0])
ax0.plot(epochs_x, hist_df['train_loss'], color=C['blue'],   lw=2,
         marker='o', ms=4, label='Train')
ax0.plot(epochs_x, hist_df['val_loss'],   color=C['red'],    lw=2,
         marker='s', ms=4, label='Val', ls='--')
ax0.set_title('Total Loss'); ax0.set_xlabel('Epoch')
ax0.legend(fontsize=8); ax0.grid(True)

ax1 = fig.add_subplot(gs[0, 1])
ax1.plot(epochs_x, hist_df['train_acc'], color=C['green'],   lw=2,
         marker='o', ms=4, label='Train')
ax1.plot(epochs_x, hist_df['val_acc'],   color=C['yellow'],  lw=2,
         marker='s', ms=4, label='Val', ls='--')
ax1.axhline(frame_metrics['frame_accuracy'], color=C['red'], lw=1.5,
            ls=':', label='Test')
ax1.set_title('Accuracy'); ax1.set_ylim(0, 1.05); ax1.set_xlabel('Epoch')
ax1.legend(fontsize=8); ax1.grid(True)

ax2 = fig.add_subplot(gs[0, 2])
ax2.plot(epochs_x, hist_df['val_auc'], color=C['purple'], lw=2,
         marker='^', ms=4, label='Val AUC')
ax2.axhline(frame_metrics['frame_auc_roc'], color=C['red'], lw=1.5,
            ls=':', label=f"Test={frame_metrics['frame_auc_roc']:.4f}")
ax2.set_title('AUC-ROC (Primary)'); ax2.set_ylim(0.5, 1.02)
ax2.set_xlabel('Epoch'); ax2.legend(fontsize=8); ax2.grid(True)

ax3 = fig.add_subplot(gs[0, 3])
ax3.plot(epochs_x, hist_df['hf_gate'], color=C['orange'], lw=2,
         marker='D', ms=4, label='HF gate')
ax3.set_title('HF Gate Growth'); ax3.set_ylim(-0.05, 1.05)
ax3.set_xlabel('Epoch'); ax3.legend(fontsize=8); ax3.grid(True)

ax4 = fig.add_subplot(gs[1, 0])
sns.heatmap(cm_norm, annot=True, fmt='.3f', ax=ax4, cmap='Blues',
            xticklabels=['REAL', 'FAKE'], yticklabels=['REAL', 'FAKE'],
            annot_kws={'size': 11, 'weight': 'bold'},
            linewidths=0.5, linecolor='#2d3748')
ax4.set_title('Confusion Matrix (norm.)')
ax4.set_xlabel('Pred'); ax4.set_ylabel('True')

ax5 = fig.add_subplot(gs[1, 1])
ax5.plot(fpr_f, tpr_f, color=C['blue'], lw=2, label=f'AUC={auc_f:.4f}')
ax5.plot([0, 1], [0, 1], color=C['grey'], lw=1, ls=':')
ax5.set_title('ROC-AUC'); ax5.set_xlabel('FPR'); ax5.set_ylabel('TPR')
ax5.legend(fontsize=9); ax5.grid(True)

ax6 = fig.add_subplot(gs[1, 2])
ax6.plot(rec_f, prec_f, color=C['yellow'], lw=2, label=f'AP={ap_f:.4f}')
ax6.set_title('Precision-Recall')
ax6.set_xlabel('Recall'); ax6.set_ylabel('Precision')
ax6.legend(fontsize=9); ax6.grid(True)

ax7 = fig.add_subplot(gs[1, 3])
ax7.plot(prob_pred, prob_true, color=C['blue'], lw=2.5,
         marker='o', ms=5, label='Student')
ax7.plot([0, 1], [0, 1], color=C['grey'], lw=1.5, ls='--', label='Perfect')
ax7.set_title(f'Reliability  ECE={ece:.4f}')
ax7.set_xlabel('Pred Prob'); ax7.legend(fontsize=9); ax7.grid(True)

ax8 = fig.add_subplot(gs[2, :2])
ax8.hist(real_probs_arr, bins=50, color=C['green'], alpha=0.65,
         label='REAL', density=True)
ax8.hist(fake_probs_arr, bins=50, color=C['red'],   alpha=0.65,
         label='FAKE', density=True)
ax8.axvline(best_thresh, color=C['yellow'], lw=2, ls='--',
            label=f'τ={best_thresh:.3f}')
ax8.set_title('Score Distribution')
ax8.set_xlabel('P(fake)'); ax8.legend(fontsize=9); ax8.grid(True)

ax9 = fig.add_subplot(gs[2, 2:])
ax9.axis('off')
table_data = [
    ['Metric',       'Frame',                                     'Video'],
    ['Accuracy',     f"{frame_metrics['frame_accuracy']:.4f}",
                     f"{video_metrics['video_accuracy']:.4f}"],
    ['F1',           f"{frame_metrics['frame_f1']:.4f}",
                     f"{video_metrics['video_f1']:.4f}"],
    ['Precision',    f"{frame_metrics['frame_precision']:.4f}",
                     f"{video_metrics['video_precision']:.4f}"],
    ['Recall',       f"{frame_metrics['frame_recall']:.4f}",
                     f"{video_metrics['video_recall']:.4f}"],
    ['Specificity',  f"{frame_metrics['frame_specificity']:.4f}",  '—'],
    ['AUC-ROC',      f"{frame_metrics['frame_auc_roc']:.4f}",
                     _nan_str(video_metrics['video_auc_roc'])],
    ['AP',           f"{frame_metrics['frame_ap']:.4f}",
                     _nan_str(video_metrics['video_ap'])],
    ['Brier Score',  f"{frame_metrics['frame_brier']:.4f}",        '—'],
    ['ECE',          f"{ece:.4f}",                                  '—'],
    ['TP/FP/TN/FN',
     f"{frame_metrics['frame_tp']}/{frame_metrics['frame_fp']}/"
     f"{frame_metrics['frame_tn']}/{frame_metrics['frame_fn']}", '—'],
    ['HF gate',      f"{student.hf_gate:.4f}",                     '—'],
    ['Logit clip',   f"±{LOGIT_CLIP}",                             '—'],
]
tbl = ax9.table(cellText=table_data[1:], colLabels=table_data[0],
                loc='center', cellLoc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1.1, 1.4)
for (r, c), cell in tbl.get_celld().items():
    cell.set_facecolor('#111827' if r % 2 == 0 else '#0a0e1a')
    cell.set_edgecolor('#2d3748')
    cell.set_text_props(color='#e2e8f0')
ax9.set_title('Evaluation Summary', pad=12)

fig.suptitle(
    f'EfficientNet-B2 Dual-Path KD Student — DFDC  |  v3 COLLAPSE FIX\n'
    f'Teacher: DPViT  |  T={KD_TEMPERATURE}  logit_clip=±{LOGIT_CLIP}  '
    f'freeze={FREEZE_EPOCHS}ep  warmup={WARMUP_EPOCHS}ep  '
    f'gate_final={student.hf_gate:.3f}',
    fontsize=13, y=1.01, color='#e2e8f0',
)
fig.savefig(OUT_DIR / 'fig9_summary_dashboard.png', dpi=200, bbox_inches='tight')
plt.close(fig)

print(f"\n{'='*70}")
print(f"  ALL OUTPUTS → {OUT_DIR}")
print(f"  Frame AUC      : {frame_metrics['frame_auc_roc']:.4f}")
print(f"  Frame Accuracy : {frame_metrics['frame_accuracy']:.4f}")
print(f"  Frame F1       : {frame_metrics['frame_f1']:.4f}")
print(f"  Frame Brier    : {frame_metrics['frame_brier']:.4f}")
print(f"  Frame ECE      : {ece:.4f}")
print(f"  Video Accuracy : {video_metrics['video_accuracy']:.4f}")
print(f"  Video AUC      : {_nan_str(video_metrics['video_auc_roc'])}")
print(f"  Final HF gate  : {student.hf_gate:.4f}")
print(f"  Teacher params : {teacher_params:.1f} M  →  "
      f"Student: {student_params:.1f} M  "
      f"({teacher_params/student_params:.1f}× compression)")
print(f"  Total session  : {(time.time()-START_TIME)/3600:.2f} h")
print(f"  Outputs: best_student.pth  |  9 figures  |  3 CSVs")
print(f"  Ready for Stage 2 MoE fusion.")