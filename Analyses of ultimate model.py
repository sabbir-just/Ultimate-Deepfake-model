# ======================= PURE-NUMPY HELPERS (shared) =======================
# CELL A  -  ANALYSES ONLY (no training).  Run in its OWN fresh Kaggle session (do NOT run after Cell B in the same kernel:
# the pipeline refuses to load if torch was already imported).
# Attach as notebook inputs: the dataset containing quadkd_offline_pipeline.py, the manifest dataset, the teacher_artifacts folder,
# the student checkpoints (best_auc.pth / best_ema_auc.pth), and - for A9/A10 only - the frame/video/audio data used by the manifest.
import os, sys, glob, json, time, math, random, re, types, warnings, hashlib
from pathlib import Path
from collections import defaultdict
if "torch" in sys.modules:
    raise RuntimeError("torch is already imported in this kernel. Restart the session (Run > Restart/Factory reset) and run this cell FIRST.")
import numpy as np, pandas as pd
from scipy.stats import rankdata, ks_2samp
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
DOMS = ["dfdc", "diffusionface", "ffpp", "lavdf"]

def fast_auc(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    m = np.isfinite(s); y, s = y[m], s[m]
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def bal_acc(y, pred):
    y = np.asarray(y).astype(int); pred = np.asarray(pred).astype(int)
    if (y == 1).sum() == 0 or (y == 0).sum() == 0: return float("nan")
    return float(((pred[y == 1] == 1).mean() + (pred[y == 0] == 0).mean()) / 2)

def eer(y, p):
    from sklearn.metrics import roc_curve
    y = np.asarray(y).astype(int)
    if len(set(y)) < 2: return float("nan")
    fpr, tpr, _ = roc_curve(y, p); fnr = 1 - tpr; i = np.argmin(np.abs(fpr - fnr))
    return float((fpr[i] + fnr[i]) / 2)

def ece_bins(y, p, nb=15):
    y = np.asarray(y); p = np.asarray(p); e = 0.0
    edges = np.linspace(0, 1, nb + 1)
    for i in range(nb):
        m = (p >= edges[i]) & ((p < edges[i + 1]) if i < nb - 1 else (p <= edges[i + 1]))
        if m.sum(): e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)

def sigmoid(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

def boot_ci(y, p, dom, groups=None, n=1000, seed=0, doms=DOMS):
    """Row-level (groups=None) or cluster bootstrap of pooled / per-domain / macro AUC."""
    y = np.asarray(y).astype(int); p = np.asarray(p, float); dom = np.asarray(dom)
    rng = np.random.RandomState(seed); N = len(y)
    if groups is not None:
        ug, inv = np.unique(np.asarray(groups), return_inverse=True)
        order = np.argsort(inv, kind="stable"); bounds = np.cumsum(np.bincount(inv))
        members = np.split(order, bounds[:-1])
    res = defaultdict(list)
    for _ in range(n):
        if groups is None: idx = rng.randint(0, N, N)
        else: idx = np.concatenate([members[g] for g in rng.randint(0, len(ug), len(ug))])
        yy, pp, dd = y[idx], p[idx], dom[idx]
        res["pooled"].append(fast_auc(yy, pp)); per = []
        for d in doms:
            m = dd == d; a = fast_auc(yy[m], pp[m]) if m.any() else float("nan")
            res[d].append(a); per.append(a)
        res["macro"].append(np.nanmean(per) if np.isfinite(per).any() else float("nan"))
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in res.items()}

def val_priors(val_dom, val_y, teacher_logits, ids, teachers, floor=0.05):
    """Validation-derived prior pi[d][t] = max(AUC_val(t | domain d) - 0.5, floor*0.5) / 0.5. No hand-set boost."""
    pri = {}
    ids = np.asarray(ids); val_dom = np.asarray(val_dom); val_y = np.asarray(val_y)
    for d in DOMS:
        pri[d] = {}
        for t in teachers:
            lg = teacher_logits.get(t, {})
            m = (val_dom == d)
            sel = [i for i in np.where(m)[0] if ids[i] in lg]
            a = fast_auc(val_y[sel], [lg[ids[i]] for i in sel]) if len(sel) > 20 else float("nan")
            pri[d][t] = (max(a - 0.5, floor * 0.5) / 0.5) if np.isfinite(a) else 1.0
    pri["unknown"] = {t: 1.0 for t in teachers}
    return pri

def load_teacher_logits(reader, t, ids):
    """Logit-only lookup: groups ids by shard, loads each shard ONCE and frees it right away (no embed/seq kept in RAM)."""
    idx = reader.index.get(t, {}); by = defaultdict(list)
    for i in ids:
        loc = idx.get(i)
        if loc is not None: by[loc["shard"]].append((i, loc["row"]))
    out = {}
    for sh in sorted(by):
        s = reader._get_shard(t, sh); lg = s["logit"].float().numpy().reshape(-1)
        for i, row in by[sh]: out[i] = float(lg[row])
        reader._shard_cache.pop((t, sh), None); del s
    return out

def make_group_fn(all_rows):
    """Heuristic source-video / identity key. EDIT if the printed audit shows groups == rows."""
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b): parent[find(a)] = find(b)
    def path_of(r): return str(r.get("video_path") or r.get("frame_path") or r.get("audio_path") or r.get("id"))
    ff_ids = {}
    for r in all_rows:
        if r.get("dataset") != "ffpp": continue
        p = Path(path_of(r)); ids = None
        for s in (p.stem, p.parent.name):
            m = re.match(r"^(\d{3})(?:_(\d{3}))?", s)
            if m: ids = [x for x in m.groups() if x]; break
        if ids:
            for x in ids[1:]: union(ids[0], x)
            ff_ids[r["id"]] = ids[0]
    def key(r):
        ds = r.get("dataset", "?"); p = Path(path_of(r)); stem = p.stem
        if ds == "lavdf": return f"lavdf:{stem}"
        if ds == "ffpp" and r["id"] in ff_ids: return f"ffpp:{find(ff_ids[r['id']])}"
        if re.fullmatch(r"(frame|f|img)?[_\-]?\d+", stem): g = p.parent.name
        else: g = re.sub(r"([_\-]?(frame|f|img)?[_\-]?\d+)$", "", stem) or p.parent.name
        return f"{ds}:{g}"
    return key

# ============================ CELL A: ANALYSES ============================
T0 = time.time(); BUDGET_H = 11.3
def left_h(): return BUDGET_H - (time.time() - T0) / 3600
CFG = dict(NUM_WORKERS=4, BS=32, VAL_PER_DOMAIN=3000, ROBUST_PER_DOMAIN=1000, N_BOOT=1000, MMD_N=2000,
           RUN_ROBUSTNESS=True, RUN_ADV=True, RUN_EFFICIENCY=True, RUN_SHORTCUT_AUDIT=True, GROUP_KEY_FN=None)
OUT = Path("/kaggle/working/quadkd_rev_analysis"); OUT.mkdir(parents=True, exist_ok=True)
def save(df, name): df.to_csv(OUT / name, index=False); print(f"[saved] {name} ({len(df)} rows)")
def log(*a): print(f"[{(time.time()-T0)/60:6.1f} min | {left_h():.1f}h left]", *a, flush=True)

def bfs_find(names, root="/kaggle/input", maxdepth=6, skip_big=3000):
    found = {n: [] for n in names}; q = [(root, 0)]
    while q:
        d, k = q.pop(0)
        try: ents = list(os.scandir(d))
        except Exception: continue
        if len(ents) > skip_big: continue
        for e in ents:
            if e.is_file() and e.name in found: found[e.name].append(e.path)
            elif e.is_dir(follow_symlinks=False) and k < maxdepth: q.append((e.path, k + 1))
    return found
FOUND = bfs_find(["quadkd_offline_pipeline.py", "best_auc.pth", "best_ema_auc.pth", "index.json"])
cands = FOUND["quadkd_offline_pipeline.py"] + glob.glob("/kaggle/working/quadkd_offline_pipeline.py") + ["quadkd_offline_pipeline.py"]
PIPE = next(p for p in ["/kaggle/input/datasets/syedazmulhasansabbir/script/quadkd_offline_pipeline.py"] + cands if os.path.exists(p)); log("pipeline file:", PIPE)
# NOTE: the pipeline refuses to load if torch was imported earlier in the kernel -> it is loaded BEFORE importing torch here.
src = open(PIPE, encoding="utf-8").read().replace("\r\n", "\n")
qkd = types.ModuleType("qkd"); qkd.__file__ = PIPE; sys.modules["qkd"] = qkd
exec(compile(src, PIPE, "exec"), qkd.__dict__)
import copy
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.isotonic import IsotonicRegression
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
DEV = qkd.DEVICE; TEACHERS = list(qkd.TEACHER_ORDER)
def exists(p): return bool(p) and os.path.exists(str(p))

# ---- small runtime patches of the pipeline (no file edits needed) ----
_orig_getitem = qkd.JointDeepfakeDataset.__getitem__
def _safe_getitem(self, i):                      # a null "manipulation" in the manifest would crash default_collate
    o = _orig_getitem(self, i)
    if o.get("manipulation") is None: o["manipulation"] = "none"
    return o
qkd.JointDeepfakeDataset.__getitem__ = _safe_getitem
def _b2_no_download():                           # weights come from the checkpoint -> no ImageNet download / internet needed
    import timm
    return timm.create_model("efficientnet_b2", pretrained=False, num_classes=0), "efficientnet_b2"
qkd._build_student_rgb_backbone = _b2_no_download

# ---------------- locate inputs ----------------
ART = next((p for p in [qkd.ARTIFACT_RESUME_INPUT_DIR, str(qkd.ARTIFACT_DIR)] if exists(p) and exists(Path(p) / "index.json")), None)
if ART is None:
    ART = next((str(Path(p).parent) for p in FOUND["index.json"] if (Path(p).parent / "metadata.json").exists()), None)
assert ART, "teacher_artifacts folder (index.json + metadata.json) not found in /kaggle/input"
CK_BEST = qkd.BEST_CKPT_RESUME_INPUT_PATH if exists(qkd.BEST_CKPT_RESUME_INPUT_PATH) else (FOUND["best_auc.pth"] or [None])[0]
CK_EMA = qkd.BEST_EMA_CKPT_RESUME_INPUT_PATH if exists(qkd.BEST_EMA_CKPT_RESUME_INPUT_PATH) else (FOUND["best_ema_auc.pth"] or [None])[0]
log("artifacts:", ART, "| ckpts:", CK_BEST, CK_EMA)
reader = qkd.TeacherArtifactReader(Path(ART))

# ---------------- manifest / splits (the split comes from the manifest 'split' field) ----------------
rows_all = qkd.load_manifest(qkd.MANIFEST_PATH)
by_split = defaultdict(list)
for r in rows_all: by_split[r.get("split", "train")].append(r)
train_rows, val_rows, test_rows = by_split["train"], by_split["val"], by_split["test"]
ood_rows = defaultdict(list)
for r in by_split["ood"]: ood_rows[r["dataset"]].append(r)
log({k: len(v) for k, v in by_split.items()}, {k: len(v) for k, v in ood_rows.items()})
gfn = CFG["GROUP_KEY_FN"] or make_group_fn(rows_all)
GK = {r["id"]: gfn(r) for r in rows_all}

def dom_balanced_val(n_per):
    rng = random.Random(777); out = []
    for d in DOMS:
        g = [r for r in val_rows if r["dataset"] == d]; out += rng.sample(g, min(n_per, len(g)))
    return out
val_sub = dom_balanced_val(CFG["VAL_PER_DOMAIN"])
log("val subset for threshold/prior/calibration fitting:", len(val_sub))

# ---------------- teacher logits (raw) for every row we need ----------------
need = {r["id"] for r in test_rows + val_sub + sum(ood_rows.values(), [])}
calib_pool = val_rows if len(val_rows) >= 8 else train_rows
if len(calib_pool) > 4000: calib_pool = random.Random(qkd.SEED).sample(calib_pool, 4000)   # same pool as the pipeline
need |= {r["id"] for r in calib_pool}
TL = {}
for t in TEACHERS:
    if t not in reader.available_teachers(): continue
    TL[t] = load_teacher_logits(reader, t, need); log("teacher logits", t, len(TL[t]))
Tfit = {}
for t in TL:
    ids_c = [r for r in calib_pool if r["id"] in TL[t]]
    s = qkd.TemperatureScaler(t); s.fit(np.array([TL[t][r["id"]] for r in ids_c]), np.array([int(r["label"]) for r in ids_c])); Tfit[t] = s.T
json.dump(dict(T_i=Tfit, calibration_split="random 4000-row subsample of the manifest 'val' split (rows where the teacher is present, all domains pooled)",
               n_calib=len(calib_pool)), open(OUT / "teacher_temperatures_check.json", "w"), indent=1)

# ---------------- student ----------------
student = qkd.MultimodalStudent(fusion_dim=512, n_sampled_frames=4).to(DEV)
cl = []
for pth, key in ((CK_BEST, "best_auc"), (CK_EMA, "best_ema_auc")):
    if exists(pth):
        raw = torch.load(qkd.resolve_ckpt_file(pth), map_location="cpu", weights_only=False); cl.append((raw.get(key, -1.0), pth, raw, key))
assert cl, "no student checkpoint found"
cl.sort(key=lambda c: c[0]); a_, pth_, payload, key_ = cl[-1]
try: qkd._assert_checkpoint_matches_current_architecture(payload, student)
except Exception as e: print("[WARNING] checkpoint guard:", e)
student.load_state_dict(payload["ema"] if (qkd.LOAD_EMA_WEIGHTS_FOR_EVAL and "ema" in payload) else payload["student"]); student.eval()
json.dump(dict(chosen=str(pth_), recorded_val_auc=float(a_), selection_set="validation subset (domain-balanced, <=6000 rows), NOT test",
               candidates=[(str(c[1]), float(c[0])) for c in cl]), open(OUT / "checkpoint_selection.json", "w"), indent=1)
log("student loaded from", pth_, "val AUC recorded:", a_)
del cl, payload

# ======================= PASS: run student on rows =======================
VARIANTS = ["clean", "no_audio", "no_temporal", "frame_only", "audio_swap", "audio_noise10dB", "temporal_swap"]
def variant(b, kind):
    nb = dict(b)
    if kind in ("no_audio", "frame_only"): nb["mask_audio"] = torch.zeros_like(b["mask_audio"])
    if kind in ("no_temporal", "frame_only"): nb["mask_temporal"] = torch.zeros_like(b["mask_temporal"])
    if kind == "audio_swap":
        idx = b["mask_audio"].nonzero(as_tuple=True)[0]
        if len(idx) > 1: a = b["audio"].clone(); a[idx] = b["audio"][idx.roll(1)]; nb["audio"] = a
    if kind == "temporal_swap":
        idx = b["mask_temporal"].nonzero(as_tuple=True)[0]
        if len(idx) > 1: c = b["clip"].clone(); c[idx] = b["clip"][idx.roll(1)]; nb["clip"] = c
    if kind == "audio_noise10dB":
        a = b["audio"]; p = a.pow(2).mean(1, keepdim=True).clamp(min=1e-10)
        nb["audio"] = a + torch.randn_like(a) * (p / 10.0).sqrt()
    return nb

def loader_for(rows, bs=None):
    return DataLoader(qkd.JointDeepfakeDataset(rows, train=False), batch_size=bs or CFG["BS"], shuffle=False,
                      num_workers=CFG["NUM_WORKERS"], pin_memory=torch.cuda.is_available(),
                      prefetch_factor=4, timeout=600)

def run_pass(rows, tag, variants=True):
    cache = OUT / f"cache_{tag}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True); log("loaded cache", tag); return {k: z[k] for k in z.files}
    rec = defaultdict(list); t1 = time.time(); nb_ = 0; partial = False
    for bi, batch in enumerate(loader_for(rows)):
        if left_h() < 1.5: log("time guard hit in pass", tag, "- result NOT cached (partial)"); partial = True; break
        b = qkd._to_device(batch)
        with torch.no_grad():
            for v in (VARIANTS if variants else ["clean"]):
                o = student(variant(b, v) if v != "clean" else b)
                rec[f"logit_{v}"].append(o["joint_logit"].float().cpu().numpy())
                if v == "clean": rec["rep"].append(o["joint_rep"].half().cpu().numpy())
        ids = [str(x) for x in batch["id"]]; rec["id"] += ids; rec["dataset"] += [str(x) for x in batch["dataset"]]; rec["manip"] += [str(x) for x in batch["manipulation"]]
        rec["label"].append(batch["label"].numpy()); nb_ += len(ids)
        for k in ("mask_frame", "mask_temporal", "mask_audio"): rec[k].append(batch[k].numpy())
        if bi % 50 == 0: log(f"{tag}: {nb_}/{len(rows)} rows, {nb_/(time.time()-t1):.1f} rows/s")
    out = {k: (np.concatenate(v) if k not in ("id", "dataset", "manip") else np.array(v)) for k, v in rec.items()}
    if not partial: np.savez_compressed(cache, **out)
    return out

P = {}
P["test"] = run_pass(test_rows, "test")
P["val"] = run_pass(val_sub, "val", variants=False)
for n_, rr in ood_rows.items(): P[n_] = run_pass(rr, f"ood_{n_}")
for k, v in P.items(): v["tl"] = {t: np.array([TL[t].get(i, np.nan) for i in v["id"]]) for t in TL}
def P_p(k, v="clean"): return sigmoid(P[k][f"logit_{v}"])

# ======================= A1  teacher audit (video_ff anomaly, FF++-only AUC) =======================
log("A1 teacher audit"); rows_o = []
for split in ("val", "test", *ood_rows.keys()):
    d = P[split]; y = d["label"]
    for t in TL:
        for dom in ["ALL"] + sorted(set(d["dataset"])):
            m = np.isfinite(d["tl"][t]) & ((d["dataset"] == dom) if dom != "ALL" else True)
            if m.sum() < 20 or len(set(y[m])) < 2: continue
            lg = d["tl"][t][m]
            rows_o.append(dict(split=split, teacher=t, domain=dom, n=int(m.sum()), auc=fast_auc(y[m], lg), auc_inverted=fast_auc(y[m], -lg),
                               mean_logit_fake=float(lg[y[m] == 1].mean()), mean_logit_real=float(lg[y[m] == 0].mean())))
A1 = pd.DataFrame(rows_o); save(A1, "A1_teacher_auc_by_split_domain.csv")
if len(A1): print(A1[(A1.teacher == "video_ff")].round(4).to_string(index=False))
d = P["test"]; mf = ((d["dataset"] == "ffpp") & np.isfinite(d["tl"]["video_ff"])) if "video_ff" in TL else None
if mf is not None and mf.any():
    pm = []
    for mn in sorted(set(d["manip"][mf & (d["label"] == 1)])):
        mm = mf & ((d["manip"] == mn) | (d["label"] == 0))
        pm.append(dict(manip=mn, n=int(mm.sum()), video_ff_auc=fast_auc(d["label"][mm], d["tl"]["video_ff"][mm]),
                       student_auc=fast_auc(d["label"][mm], d["logit_clean"][mm])))
    save(pd.DataFrame(pm), "A1b_ffpp_per_manip_within_domain.csv")

# ======================= A2  validation-only selection of priors / best teacher / thresholds =======================
log("A2 validation-based selection")
vd = P["val"]
pri = val_priors(vd["dataset"], vd["label"], TL, vd["id"], TEACHERS)
json.dump(pri, open(OUT / "A2_val_derived_priors.json", "w"), indent=1); print(json.dumps(pri, indent=1))
bt = []
for split in ("val", "test"):
    d = P[split]
    for t in TL:
        m = np.isfinite(d["tl"][t]); bt.append(dict(split=split, teacher=t, n=int(m.sum()), auc=fast_auc(d["label"][m], d["tl"][t][m])))
save(pd.DataFrame(bt), "A2_best_single_teacher_val_vs_test.csv")
thr_rows, curve_rows = [], []
for dom in DOMS:
    mv, mt = vd["dataset"] == dom, P["test"]["dataset"] == dom
    ths = np.round(np.arange(0.02, 0.99, 0.01), 2)
    bv = [bal_acc(vd["label"][mv], P_p("val")[mv] >= t) for t in ths]
    bt_ = [bal_acc(P["test"]["label"][mt], P_p("test")[mt] >= t) for t in ths]
    best = ths[int(np.nanargmax(bv))]
    for t, a, b in zip(ths, bv, bt_): curve_rows.append(dict(domain=dom, thr=t, val_bal_acc=a, test_bal_acc=b))
    thr_rows.append(dict(domain=dom, n_val=int(mv.sum()), n_test=int(mt.sum()), tuned_thr_from_val=float(best),
                         val_bal_acc_at_0p5=bal_acc(vd["label"][mv], P_p("val")[mv] >= .5), val_bal_acc_at_tuned=float(np.nanmax(bv)),
                         test_bal_acc_at_0p5=bal_acc(P["test"]["label"][mt], P_p("test")[mt] >= .5),
                         test_bal_acc_at_tuned=bal_acc(P["test"]["label"][mt], P_p("test")[mt] >= best),
                         val_fake_frac=float(vd["label"][mv].mean()), test_fake_frac=float(P["test"]["label"][mt].mean()),
                         ks_stat_fake_scores=float(ks_2samp(P_p("val")[mv & (vd["label"] == 1)], P_p("test")[mt & (P["test"]["label"] == 1)]).statistic),
                         ks_stat_real_scores=float(ks_2samp(P_p("val")[mv & (vd["label"] == 0)], P_p("test")[mt & (P["test"]["label"] == 0)]).statistic),
                         val_auc=fast_auc(vd["label"][mv], vd["logit_clean"][mv]), test_auc=fast_auc(P["test"]["label"][mt], P["test"]["logit_clean"][mt])))
A2 = pd.DataFrame(thr_rows); save(A2, "A2_thresholds_val_vs_test.csv"); save(pd.DataFrame(curve_rows), "A2_balacc_vs_threshold_curves.csv"); print(A2.round(4).to_string(index=False))
fig, ax = plt.subplots(1, 4, figsize=(16, 3.2))
cv = pd.DataFrame(curve_rows)
for a_, dom in zip(ax, DOMS):
    c = cv[cv.domain == dom]; a_.plot(c.thr, c.val_bal_acc, label="val"); a_.plot(c.thr, c.test_bal_acc, label="test"); a_.set_title(dom); a_.set_xlabel("threshold")
ax[0].legend(); ax[0].set_ylabel("balanced acc"); fig.savefig(OUT / "A2_balacc_vs_threshold.png", dpi=130, bbox_inches="tight"); plt.close(fig)

# ======================= A3  bootstrap CIs (row-level vs cluster by source video/identity) =======================
log("A3 bootstrap"); d = P["test"]; gk = np.array([GK[i] for i in d["id"]]); p = d["logit_clean"]
rl = boot_ci(d["label"], p, d["dataset"], None, n=CFG["N_BOOT"]); cl_ = boot_ci(d["label"], p, d["dataset"], gk, n=CFG["N_BOOT"])
pt = {"pooled": fast_auc(d["label"], p), "macro": np.nanmean([fast_auc(d["label"][d["dataset"] == x], p[d["dataset"] == x]) for x in DOMS])}
for x in DOMS: pt[x] = fast_auc(d["label"][d["dataset"] == x], p[d["dataset"] == x])
A3 = pd.DataFrame([dict(metric=k, auc=pt[k], row_ci_lo=rl[k][0], row_ci_hi=rl[k][1], cluster_ci_lo=cl_[k][0], cluster_ci_hi=cl_[k][1]) for k in pt])
save(A3, "A3_bootstrap_row_vs_cluster.csv"); print(A3.round(4).to_string(index=False))

# ======================= A8  leakage audit =======================
log("A8 split-leakage audit"); la = []
gtr = defaultdict(set); gva = defaultdict(set)
for r in train_rows: gtr[r["dataset"]].add(GK[r["id"]])
for r in val_rows: gva[r["dataset"]].add(GK[r["id"]])
for dom in DOMS:
    tr = [r for r in train_rows if r["dataset"] == dom]; te = [r for r in test_rows if r["dataset"] == dom]
    gte = {GK[r["id"]] for r in te}; ov = gte & gtr[dom]
    la.append(dict(domain=dom, rows_train=len(tr), groups_train=len(gtr[dom]), rows_test=len(te), groups_test=len(gte),
                   rows_per_group_test=len(te) / max(1, len(gte)), test_groups_also_in_train=len(ov),
                   frac_test_rows_in_train_groups=float(np.mean([GK[r["id"]] in gtr[dom] for r in te])) if te else float("nan")))
A8 = pd.DataFrame(la); print(A8.round(3).to_string(index=False))
d = P["test"]; leaky = np.array([GK[i] in gtr[ds_] for i, ds_ in zip(d["id"], d["dataset"])]); res = []
gk_test = np.array([GK[i] for i in d["id"]])
for dom in DOMS + ["ALL"]:
    m = (d["dataset"] == dom) if dom != "ALL" else np.ones(len(leaky), bool)
    for nm, mm in (("group_seen_in_train", m & leaky), ("group_NOT_in_train", m & ~leaky)):
        res.append(dict(domain=dom, subset=nm, n=int(mm.sum()), n_groups=len(set(gk_test[mm])), auc=fast_auc(d["label"][mm], d["logit_clean"][mm])))
A8b = pd.DataFrame(res); save(A8, "A8_leakage_group_overlap.csv"); save(A8b, "A8_auc_leaky_vs_clean_subset.csv"); print(A8b.round(4).to_string(index=False))
ex = {dom: [GK[r["id"]] for r in test_rows if r["dataset"] == dom][:3] for dom in DOMS}; print("example group keys:", ex)
print("!! If rows_per_group_test ~ 1 for a frame-based domain, the key function is not grouping frames -> set CFG['GROUP_KEY_FN'] for that dataset.")

# ======================= A4  calibration with per-bin counts + post-hoc =======================
log("A4 calibration")
from scipy.optimize import minimize_scalar
def nll(y, p): p = np.clip(p, 1e-7, 1 - 1e-7); return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
def fitT(z, y): return minimize_scalar(lambda T: nll(y, sigmoid(z / T)), bounds=(0.05, 20), method="bounded").x
zv, yv = vd["logit_clean"], vd["label"]; Tg = fitT(zv, yv); Td = {x: fitT(zv[vd["dataset"] == x], yv[vd["dataset"] == x]) for x in DOMS}
iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(sigmoid(zv), yv)
def posthoc(split, name, dom_arr):
    z = P[split]["logit_clean"]
    if name == "none": return sigmoid(z)
    if name == "global_T": return sigmoid(z / Tg)
    if name == "per_domain_T": return sigmoid(z / np.array([Td.get(x, Tg) for x in dom_arr]))
    if name == "isotonic": return iso.predict(sigmoid(z))
cal = []
for split in ["test"] + list(ood_rows):
    for name in ("none", "global_T", "per_domain_T", "isotonic"):
        pp = posthoc(split, name, P[split]["dataset"]); y = P[split]["label"]
        for dom in (["ALL"] + (DOMS if split == "test" else [])):
            m = (P[split]["dataset"] == dom) if dom != "ALL" else np.ones(len(y), bool)
            cal.append(dict(set=split, domain=dom, method=name, n=int(m.sum()), ece15=ece_bins(y[m], pp[m]), brier=float(((pp[m] - y[m]) ** 2).mean()), nll=nll(y[m], pp[m])))
A4 = pd.DataFrame(cal); save(A4, "A4_posthoc_calibration.csv"); print(f"global T={Tg:.3f}", {k: round(v, 3) for k, v in Td.items()})
print(A4[(A4.domain == "ALL")].round(4).to_string(index=False))
def reliability(y, p, nb=10):
    e = np.linspace(0, 1, nb + 1); idx = np.clip(np.digitize(p, e) - 1, 0, nb - 1)
    return [(0.5 * (e[i] + e[i + 1]), (y[idx == i].mean() if (idx == i).any() else np.nan), int((idx == i).sum())) for i in range(nb)]
sets = [("test(ID)", "test"), *[(k, k) for k in ood_rows]]
fig, ax = plt.subplots(2, len(sets), figsize=(4.2 * len(sets), 6.5), squeeze=False); rel_rows = []
for j, (nm, k) in enumerate(sets):
    rl_ = reliability(P[k]["label"], P_p(k)); xs, ys, ns = zip(*rl_)
    ax[0, j].plot([0, 1], [0, 1], "--", c="gray"); ax[0, j].plot(xs, ys, "o-"); ax[0, j].set_title(nm)
    for x_, y_, n_ in rl_:
        if np.isfinite(y_): ax[0, j].annotate(str(n_), (x_, y_), fontsize=6, xytext=(2, 4), textcoords="offset points")
        rel_rows.append(dict(set=nm, conf=x_, acc=y_, n=n_))
    ax[1, j].bar(xs, ns, width=0.08); ax[1, j].set_yscale("log"); ax[1, j].set_xlabel("confidence"); ax[1, j].set_ylabel("samples / bin")
fig.savefig(OUT / "A4_reliability_with_counts.png", dpi=140, bbox_inches="tight"); plt.close(fig); save(pd.DataFrame(rel_rows), "A4_reliability_bin_counts.csv")

# ======================= A5  quantitative OOD analysis =======================
log("A5 OOD quantification")
from scipy.linalg import sqrtm
Rv = vd["rep"].astype(np.float32); mu, sd = Rv.mean(0), Rv.std(0) + 1e-6
def Z(x): return (x.astype(np.float32) - mu) / sd
def mmd2(X, Y, nperm=100, seed=0):
    from sklearn.metrics.pairwise import euclidean_distances
    rng = np.random.RandomState(seed); XY = np.vstack([X, Y]); D = euclidean_distances(XY, squared=True)
    bw = np.median(D[D > 0]); K = np.exp(-D / bw); n = len(X)
    def stat(Kx, a, b):
        Kaa, Kbb, Kab = Kx[np.ix_(a, a)], Kx[np.ix_(b, b)], Kx[np.ix_(a, b)]
        return (Kaa.sum() - np.trace(Kaa)) / (len(a) * (len(a) - 1)) + (Kbb.sum() - np.trace(Kbb)) / (len(b) * (len(b) - 1)) - 2 * Kab.mean()
    idx = np.arange(len(XY)); obs = stat(K, idx[:n], idx[n:]); cnt = 0
    for _ in range(nperm):
        pm = rng.permutation(idx); cnt += stat(K, pm[:n], pm[n:]) >= obs
    return float(obs), float((cnt + 1) / (nperm + 1))
def frechet(X, Y):
    m1, m2 = X.mean(0), Y.mean(0); c1, c2 = np.cov(X, rowvar=False), np.cov(Y, rowvar=False)
    cs = sqrtm(c1 @ c2); cs = cs.real if np.iscomplexobj(cs) else cs
    return float(((m1 - m2) ** 2).sum() + np.trace(c1 + c2 - 2 * cs))
rng = np.random.RandomState(0); idt = P["test"]; Zt = Z(idt["rep"]); dom_t = idt["dataset"]
def sub(X, n): return X[rng.choice(len(X), min(n, len(X)), replace=False)]
# Mahalanobis (class-conditional means, tied covariance) fitted on VALIDATION in-domain reps
Zv = Z(Rv); yv_ = vd["label"].astype(int); m0, m1 = Zv[yv_ == 0].mean(0), Zv[yv_ == 1].mean(0)
cen = np.vstack([Zv[yv_ == 0] - m0, Zv[yv_ == 1] - m1]); prec = np.linalg.pinv(np.cov(cen, rowvar=False) + 1e-3 * np.eye(Zv.shape[1]))
def maha(X): return np.minimum((((X - m0) @ prec) * (X - m0)).sum(1), (((X - m1) @ prec) * (X - m1)).sum(1))
def energy(z): return -np.logaddexp(0, z)           # -logsumexp([0, z]); higher = more OOD-like
def conf(z): return -np.abs(sigmoid(z) - 0.5)       # low confidence = OOD-like
ood_tab = []
for k in ood_rows:
    Xo = Z(P[k]["rep"]); no = min(CFG["MMD_N"], len(Xo))
    for ref in ["ALL_ID"] + DOMS:
        Xi = Zt if ref == "ALL_ID" else Zt[dom_t == ref]
        if len(Xi) < 200: continue
        A_, B_ = sub(Xi, no), sub(Xo, no)
        X = np.vstack([A_, B_]); yd = np.r_[np.zeros(len(A_)), np.ones(len(B_))]
        pr = cross_val_predict(LogisticRegression(max_iter=400, C=0.5), X, yd, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
        mm, pv = mmd2(sub(Xi, 1000), sub(Xo, 1000))
        sc_id, sc_o = (maha(Xi), maha(Xo)); zi = idt["logit_clean"][(dom_t == ref) if ref != "ALL_ID" else slice(None)]; zo = P[k]["logit_clean"]
        ood_tab.append(dict(ood=k, id_ref=ref, domain_clf_acc=float(((pr > .5) == yd).mean()), domain_clf_auc=fast_auc(yd, pr), mmd2=mm, mmd_perm_p=pv,
                            frechet=frechet(A_, B_), maha_auroc=fast_auc(np.r_[np.zeros(len(sc_id)), np.ones(len(sc_o))], np.r_[sc_id, sc_o]),
                            energy_auroc=fast_auc(np.r_[np.zeros(len(zi)), np.ones(len(zo))], np.r_[energy(zi), energy(zo)]),
                            lowconf_auroc=fast_auc(np.r_[np.zeros(len(zi)), np.ones(len(zo))], np.r_[conf(zi), conf(zo)])))
half = rng.permutation(len(Zt)); h1, h2 = Zt[half[:len(Zt) // 2]], Zt[half[len(Zt) // 2:]]
nc = min(2000, len(h1), len(h2))
Xc = np.vstack([sub(h1, nc), sub(h2, nc)]); yc = np.r_[np.zeros(nc), np.ones(nc)]
prc = cross_val_predict(LogisticRegression(max_iter=400, C=0.5), Xc, yc, cv=5, method="predict_proba")[:, 1]
ood_tab.append(dict(ood="CONTROL ID-test half vs half", id_ref="ALL_ID", domain_clf_acc=float(((prc > .5) == yc).mean()), domain_clf_auc=fast_auc(yc, prc), mmd2=mmd2(sub(h1, 1000), sub(h2, 1000))[0]))
A5 = pd.DataFrame(ood_tab); save(A5, "A5_ood_quantification.csv"); print(A5.round(4).to_string(index=False))

# ======================= A6/A7  modality contribution + FF++ diagnostics =======================
log("A6 modality removal / corruption"); mr = []
for split in ["test"] + list(ood_rows):
    d = P[split]
    for v in VARIANTS:
        z = d[f"logit_{v}"]
        for dom in ["ALL"] + sorted(set(d["dataset"])):
            m = (d["dataset"] == dom) if dom != "ALL" else np.ones(len(z), bool)
            if len(set(d["label"][m])) < 2: continue
            lo, hi = boot_ci(d["label"][m], z[m], d["dataset"][m], None, n=200, doms=[dom])["pooled"] if dom != "ALL" else (np.nan, np.nan)
            mr.append(dict(set=split, domain=dom, variant=v, n=int(m.sum()), auc=fast_auc(d["label"][m], z[m]), ci_lo=lo, ci_hi=hi,
                           bal_acc=bal_acc(d["label"][m], z[m] > 0), has_audio=float(d["mask_audio"][m].mean()), has_temporal=float(d["mask_temporal"][m].mean())))
A6 = pd.DataFrame(mr); save(A6, "A6_modality_ablation_inference.csv")
print(A6[(A6.set == "test") & (A6.domain != "ALL")].pivot(index="domain", columns="variant", values="auc").round(4).to_string())
log("A7 FF++ diagnostics")
comp = pd.DataFrame([dict(split=r.get("split"), label=r["label"], manip=r.get("manipulation")) for r in rows_all if r.get("dataset") == "ffpp"]).fillna("none").value_counts().reset_index(name="n")
save(comp, "A7_ffpp_composition.csv")
pool = qkd.domain_balanced_sample(train_rows, 60000, seed=qkd.SEED + 1001); cnt = pd.Series([r["dataset"] for r in pool]).value_counts()
tr_cnt = pd.Series([r["dataset"] for r in train_rows]).value_counts(); print("Stage-B pool vs train availability:\n", pd.concat([cnt, tr_cnt], axis=1, keys=["in_stage_B_pool", "in_train_split"]))
wd = []
for split in ["test"] + list(ood_rows):
    d = P[split]
    for dom in sorted(set(d["dataset"])):
        for mn in sorted(set(d["manip"][(d["dataset"] == dom) & (d["label"] == 1)])):
            m = (d["dataset"] == dom) & ((d["manip"] == mn) | (d["label"] == 0))
            wd.append(dict(set=split, domain=dom, manipulation=mn, n_fake=int(((d["manip"] == mn) & (d["label"] == 1) & (d["dataset"] == dom)).sum()),
                           n_real_same_domain=int(((d["label"] == 0) & (d["dataset"] == dom)).sum()), auc_vs_same_domain_real=fast_auc(d["label"][m], d["logit_clean"][m])))
save(pd.DataFrame(wd), "A7_per_manipulation_auc_within_domain.csv")
if CFG["RUN_SHORTCUT_AUDIT"]:
    from PIL import Image
    feats = []
    for dom in DOMS:
        for spl in ("train", "test"):
            for lab in (0, 1):
                g = [r for r in rows_all if r.get("dataset") == dom and r.get("split") == spl and int(r["label"]) == lab]
                for r in random.Random(1).sample(g, min(1000, len(g))):
                    p_ = r.get("frame_path") or r.get("video_path")
                    try:
                        f = dict(domain=dom, split=spl, label=lab, bytes=os.path.getsize(p_))
                        if r.get("frame_path"):
                            im = Image.open(p_); f.update(w=im.size[0], h=im.size[1], bpp=f["bytes"] / (im.size[0] * im.size[1]),
                                                           q=float(np.mean(im.quantization[0])) if hasattr(im, "quantization") and im.quantization else np.nan)
                        feats.append(f)
                    except Exception: pass
    FD = pd.DataFrame(feats); sc = []
    for dom in DOMS:
        for spl in ("train", "test"):
            if not len(FD): break
            g = FD[(FD.domain == dom) & (FD.split == spl)]
            for c in ("bytes", "w", "h", "bpp", "q"):
                if c in g and g[c].notna().sum() > 50 and g.label.nunique() == 2: sc.append(dict(domain=dom, split=spl, feature=c, auc_for_label=fast_auc(g.label, g[c].fillna(g[c].median())), n=len(g)))
    SC = pd.DataFrame(sc)
    if len(SC):
        SC["flag"] = (SC.auc_for_label - 0.5).abs() > 0.2; save(SC, "A7_shortcut_metadata_auc.csv"); print(SC.round(3).to_string(index=False))
    else: print("[A7] shortcut audit: no readable files / features (data not attached?) - skipped")
    for lab in (0, 1): print("example ffpp paths label", lab, [(r.get("frame_path"), r.get("video_path")) for r in [x for x in rows_all if x.get("dataset") == "ffpp" and int(x["label"]) == lab][:2]])

# ======================= A9  robustness + white-box adversarial (frame path) =======================
def sub_stratified(rows, per):
    rg = random.Random(5); out = []
    for k in sorted({r["dataset"] for r in rows}):
        g = [r for r in rows if r["dataset"] == k]; out += rg.sample(g, min(per, len(g)))
    return out
if CFG["RUN_ROBUSTNESS"] and left_h() > 4:
    log("A9 robustness"); import torchvision.transforms.functional as TF
    from PIL import Image; import io
    MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
    to_pix = lambda x: x * STD + MEAN; from_pix = lambda x: (x - MEAN) / STD
    def jpeg(x, q):
        out = []
        for im in (x.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy():
            buf = io.BytesIO(); Image.fromarray(im).save(buf, format="JPEG", quality=q); buf.seek(0)
            out.append(torch.from_numpy(np.array(Image.open(buf).convert("RGB"))).permute(2, 0, 1).float() / 255)
        return torch.stack(out).to(x.device)
    def op(x, kind, par):
        if kind == "jpeg": return jpeg(x, par)
        if kind == "resize": h = x.shape[-1]; return F.interpolate(F.interpolate(x, scale_factor=par, mode="bilinear", antialias=True), size=(h, h), mode="bilinear")
        if kind == "noise": return (x + torch.randn_like(x) * par).clamp(0, 1)
        if kind == "blur": k = 2 * int(math.ceil(3 * par)) + 1; return TF.gaussian_blur(x, [k, k], [par, par])
    def corrupt(b, kind, par):
        nb = dict(b); nb["frame"] = from_pix(op(to_pix(b["frame"]), kind, par))
        mt = b["mask_temporal"].bool()
        if mt.any():
            idx = student.temporal_module.sample_idx.to(DEV); clip = b["clip"].clone(); sel = mt.nonzero(as_tuple=True)[0]
            sub_ = clip[sel][:, :, idx]; B_, C_, T_, H_, W_ = sub_.shape
            x = to_pix(sub_.permute(0, 2, 1, 3, 4).reshape(B_ * T_, C_, H_, W_)); x = from_pix(op(x, kind, par))
            sub_ = x.view(B_, T_, C_, H_, W_).permute(0, 2, 1, 3, 4)
            for j, ii in enumerate(idx.tolist()): clip[sel, :, ii] = sub_[:, :, j]
            nb["clip"] = clip
        return nb
    def pgd(b, eps, steps=10):
        x0 = to_pix(b["frame"]).detach(); alpha = eps / 4; delta = (torch.rand_like(x0) * 2 - 1) * eps; y = b["label"]
        for _ in range(steps):
            delta.requires_grad_(True); xa = (x0 + delta).clamp(0, 1); nb = dict(b); nb["frame"] = from_pix(xa)
            loss = F.binary_cross_entropy_with_logits(student(nb)["joint_logit"], y)
            g, = torch.autograd.grad(loss, delta); delta = (delta.detach() + alpha * g.sign()).clamp(-eps, eps)
            delta = ((x0 + delta).clamp(0, 1) - x0).detach()
        nb = dict(b); nb["frame"] = from_pix((x0 + delta).clamp(0, 1)); return nb
    PERT = [("clean", None, None), ("jpeg", "jpeg", 90), ("jpeg", "jpeg", 70), ("jpeg", "jpeg", 50), ("jpeg", "jpeg", 30), ("resize", "resize", 0.5), ("resize", "resize", 0.25),
            ("noise", "noise", 0.02), ("noise", "noise", 0.05), ("blur", "blur", 1.0), ("blur", "blur", 2.0)]
    if CFG["RUN_ADV"]: PERT += [("pgd", "pgd", e / 255) for e in (1, 2, 4)]
    rob_rows = sub_stratified(test_rows, CFG["ROBUST_PER_DOMAIN"]) + sub_stratified(sum(ood_rows.values(), []), CFG["ROBUST_PER_DOMAIN"])
    rec = defaultdict(list); meta = defaultdict(list); torch.manual_seed(0)
    for bi, batch in enumerate(loader_for(rob_rows, 24)):
        if left_h() < 2.0: log("time guard in robustness"); break
        b = qkd._to_device(batch)
        for nm, kind, par in PERT:
            key = f"{nm}_{par}"
            if kind is None:
                with torch.no_grad(): z = student(b)["joint_logit"]
            elif kind == "pgd":
                nb = pgd(b, par)
                with torch.no_grad(): z = student(nb)["joint_logit"]
            else:
                with torch.no_grad(): z = student(corrupt(b, kind, par))["joint_logit"]
            rec[key].append(z.float().cpu().numpy())
        meta["label"].append(batch["label"].numpy()); meta["dataset"] += [str(x) for x in batch["dataset"]]
        if bi % 20 == 0: log("robustness batch", bi)
    y = np.concatenate(meta["label"]); ds_ = np.array(meta["dataset"]); rb = []
    zc = np.concatenate(rec["clean_None"])
    for key, v in rec.items():
        z = np.concatenate(v)
        for dom in ["ALL"] + sorted(set(ds_)):
            m = (ds_ == dom) if dom != "ALL" else np.ones(len(y), bool); ok = (zc[m] > 0) == (y[m] == 1)
            rb.append(dict(perturbation=key, domain=dom, n=int(m.sum()), auc=fast_auc(y[m], z[m]), bal_acc=bal_acc(y[m], z[m] > 0),
                           flip_rate_of_clean_correct=float(((z[m] > 0) != (zc[m] > 0))[ok].mean()) if ok.any() else np.nan))
    save(pd.DataFrame(rb), "A9_robustness.csv"); print(pd.DataFrame(rb).query("domain=='ALL'").round(4).to_string(index=False))
    print("Threat model: white-box, digital, L_inf PGD-10 on the frame tensor only (clip/audio fixed); no adaptive/anti-forensic attacks; JPEG/resize/noise/blur applied to frame + the 4 clip frames the student reads.")

# ======================= A10  efficiency: FLOPs, memory, CPU/GPU latency, end-to-end =======================
if CFG["RUN_EFFICIENCY"]:
    log("A10 efficiency"); from torch.utils.flop_counter import FlopCounterMode
    def mk_inputs(B, dev):
        return dict(frame=torch.randn(B, 3, 224, 224, device=dev), clip=torch.randn(B, 3, 16, 224, 224, device=dev), audio=torch.randn(B, 64000, device=dev),
                    mask_frame=torch.ones(B, dtype=torch.bool, device=dev), mask_temporal=torch.ones(B, dtype=torch.bool, device=dev), mask_audio=torch.ones(B, dtype=torch.bool, device=dev))
    specs = {"student": (student, lambda m, x: m(x)), "frame_dfdc": ("frame_dfdc", lambda m, x: m.infer_all(x["frame"])), "frame_diff": ("frame_diff", lambda m, x: m.infer_all(x["frame"])),
             "video_ff": ("video_ff", lambda m, x: m.infer_all(x["clip"])), "audio_lavdf": ("audio_lavdf", lambda m, x: m.infer_all(x["audio"]))}
    eff = []
    for name, (mod, fn) in specs.items():
        try:
            m = mod if not isinstance(mod, str) else qkd.TEACHER_CLASSES[mod]()
            nparam = sum(p.numel() for p in m.parameters()) / 1e6
            for devn in (["cuda"] if torch.cuda.is_available() else []) + ["cpu"]:
                mm = copy.deepcopy(m).to(devn).eval(); B = 8 if devn == "cuda" else 2; x = mk_inputs(B, devn)
                with torch.no_grad():
                    for _ in range(2): fn(mm, x)
                    with FlopCounterMode(display=False) as fc: fn(mm, x)
                    if devn == "cuda": torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
                    n_it = 10 if devn == "cuda" else 3; t1 = time.time()
                    for _ in range(n_it): fn(mm, x)
                    if devn == "cuda": torch.cuda.synchronize()
                    ms = (time.time() - t1) / n_it * 1000
                eff.append(dict(model=name, device=devn, batch=B, params_M=nparam, gflops_per_sample=fc.get_total_flops() / B / 1e9, ms_per_batch=ms, ms_per_sample=ms / B,
                                peak_mem_MB=(torch.cuda.max_memory_allocated() / 1e6 if devn == "cuda" else np.nan)))
                del mm; torch.cuda.empty_cache()
        except Exception as e: print(f"[A10] {name} skipped: {e!r}")
    E = pd.DataFrame(eff)
    try:   # preprocessing (decode) cost on a real LAV-DF test sample = shared cost for student and every teacher
        ds = qkd.JointDeepfakeDataset([r for r in test_rows if r["dataset"] == "lavdf"][:40], train=False); t1 = time.time()
        for i in range(40): ds[i]
        E["decode_preproc_ms_per_sample_lavdf"] = (time.time() - t1) / 40 * 1000
    except Exception as e: print("[A10] preprocessing timing skipped:", e)
    save(E, "A10_efficiency.csv"); print(E.round(2).to_string(index=False))
    if len(E):
        for dv in E.device.unique():
            e = E[E.device == dv].set_index("model")
            if "student" in e.index:
                for nm, ts in (("4-teacher", ["frame_dfdc", "frame_diff", "video_ff", "audio_lavdf"]), ("3-teacher (no video_ff)", ["frame_dfdc", "frame_diff", "audio_lavdf"])):
                    if all(t in e.index for t in ts):
                        print(f"[{dv}] {nm}: params x{e.loc[ts,'params_M'].sum()/e.loc['student','params_M']:.1f}, GFLOPs x{e.loc[ts,'gflops_per_sample'].sum()/e.loc['student','gflops_per_sample']:.1f}, "
                              f"latency x{e.loc[ts,'ms_per_sample'].sum()/e.loc['student','ms_per_sample']:.1f}")
log("DONE. Outputs in", OUT); print(sorted(os.listdir(OUT)))