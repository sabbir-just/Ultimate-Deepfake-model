# ======================= PURE-NUMPY HELPERS (shared) =======================
# CELL B  -  RETRAINING EXPERIMENTS.  Run in its OWN fresh Kaggle session (GPU on). Set SESSION below (1..4), one session per notebook run.
# Session 1: leave FIXED_SCALE = None  -> it benchmarks speed and prints/saves the schedule SCALE.
# Sessions 2-4: attach session 1's (and every later) output as notebook input; the SCALE is then read automatically from
#               scale_session1.json (or paste it into FIXED_SCALE). Previously finished runs (result.json) are skipped and included in the final tables.
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

# ============================ CELL B: RETRAINING EXPERIMENTS ============================
# Multi-seed + KD-ablation + router + masking + baseline experiments, ALL on the SAME protocol
# (same manifest split, same cached teacher artifacts, same schedule scaled by one SCALE factor).
SESSION = 1              # 1..5  -> picks the experiment list below
FIXED_SCALE = 0.08       # hand-set because the auto benchmark underestimated cost ~3x; keep IDENTICAL in every session
SEEDS = [42, 1, 2]
SESSION = 1
SESSIONS = {
    1: [("full", 42), ("full", 1), ("full", 2)],
}
CFG = dict(NUM_WORKERS=4, BS_EVAL=32, VAL_PER_DOMAIN=3000, VAL_SUBSET=3000)
T0 = time.time(); BUDGET_H = 11.4
def left_h(): return BUDGET_H - (time.time() - T0) / 3600
def log(*a): print(f"[{(time.time()-T0)/60:6.1f} min | {left_h():.2f}h left]", *a, flush=True)
OUTROOT = Path("/kaggle/working/quadkd_rev_train"); OUTROOT.mkdir(parents=True, exist_ok=True)

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
FOUND = bfs_find(["quadkd_offline_pipeline.py", "index.json", "result.json", "scale_session1.json"])
PIPE = next(p for p in ["/kaggle/input/datasets/syedazmulhasansabbir/script/quadkd_offline_pipeline.py"] + FOUND["quadkd_offline_pipeline.py"] + glob.glob("/kaggle/working/quadkd_offline_pipeline.py") + ["quadkd_offline_pipeline.py"] if os.path.exists(p))
src = open(PIPE, encoding="utf-8").read().replace("\r\n", "\n")        # load the pipeline BEFORE importing torch (it requires that)
qkd = types.ModuleType("qkd"); qkd.__file__ = PIPE; sys.modules["qkd"] = qkd
exec(compile(src, PIPE, "exec"), qkd.__dict__)
import copy, shutil, gc
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
DEV = qkd.DEVICE; TEACHERS = list(qkd.TEACHER_ORDER)
def exists(p): return bool(p) and os.path.exists(str(p))

_orig_getitem = qkd.JointDeepfakeDataset.__getitem__
_quiet = {"done": False}
def _safe_getitem(self, i):                      # a null "manipulation" in the manifest would crash default_collate
    if not _quiet["done"] and torch.utils.data.get_worker_info() is not None:   # only inside DataLoader workers, never the main process
        try: os.dup2(os.open(os.devnull, os.O_WRONLY), 2)                      # silence native libavcodec/h264 stderr spam (decord)
        except Exception: pass
        _quiet["done"] = True
    o = _orig_getitem(self, i)
    if o.get("manipulation") is None: o["manipulation"] = "none"
    return o
qkd.JointDeepfakeDataset.__getitem__ = _safe_getitem
ART = next((p for p in [qkd.ARTIFACT_RESUME_INPUT_DIR, str(qkd.ARTIFACT_DIR)] if exists(p) and exists(Path(p) / "index.json")), None)
if ART is None: ART = next((str(Path(p).parent) for p in FOUND["index.json"] if (Path(p).parent / "metadata.json").exists()), None)
assert ART, "teacher_artifacts folder not found in /kaggle/input"
reader = qkd.TeacherArtifactReader(Path(ART)); log("artifacts:", ART, reader.available_teachers())

rows_all = qkd.load_manifest(qkd.MANIFEST_PATH); by_split = defaultdict(list)
for r in rows_all: by_split[r.get("split", "train")].append(r)
train_rows, val_rows, test_rows = by_split["train"], by_split["val"], by_split["test"]
ood_rows = defaultdict(list)
for r in by_split["ood"]: ood_rows[r["dataset"]].append(r)
log({k: len(v) for k, v in by_split.items()}, {k: len(v) for k, v in ood_rows.items()})

# ---- teacher logits on calibration + validation rows (for T_i fitting and validation-derived priors) ----
rg = random.Random(777); val_sub = []
for d in DOMS:
    g = [r for r in val_rows if r["dataset"] == d]; val_sub += rg.sample(g, min(CFG["VAL_PER_DOMAIN"], len(g)))
calib_pool = val_rows if len(val_rows) >= 8 else train_rows
if len(calib_pool) > 4000: calib_pool = random.Random(qkd.SEED).sample(calib_pool, 4000)
need = {r["id"] for r in calib_pool + val_sub}; TL = {}
for t in TEACHERS:
    if t in reader.available_teachers(): TL[t] = load_teacher_logits(reader, t, need)
scalers = {}
for t in TL:
    rr = [r for r in calib_pool if r["id"] in TL[t]]; s = qkd.TemperatureScaler(t)
    s.fit(np.array([TL[t][r["id"]] for r in rr]), np.array([int(r["label"]) for r in rr])); scalers[t] = s
embed_dims = {n: reader.metadata[n]["feature_dim"] for n in TEACHERS if n in reader.available_teachers()}
usable = {n: (n in embed_dims) for n in TEACHERS}
VAL_PRI = val_priors([r["dataset"] for r in val_sub], [int(r["label"]) for r in val_sub], TL, [r["id"] for r in val_sub], [t for t in TEACHERS if t in TL])
json.dump(VAL_PRI, open(OUTROOT / "validation_derived_priors.json", "w"), indent=1)

# ---- routers ----
class LearnedRouter(nn.Module):
    """Trainable attention-style router (the design the paper says was replaced). Used only to test the 'collapse' claim."""
    def __init__(self, names, hidden=32):
        super().__init__(); self.names = names; self.doms = DOMS + ["unknown"]
        self.net = nn.Sequential(nn.Linear(len(self.doms) + len(names), hidden), nn.ReLU(), nn.Linear(hidden, len(names)))
        self.domain_prior = "learned"; self.acc = defaultdict(lambda: np.zeros(len(names))); self.cnt = defaultdict(int)
    def forward(self, present_mask, datasets, log_epoch=None):
        dev = present_mask.device
        oh = torch.zeros(len(datasets), len(self.doms), device=dev)
        for i, d in enumerate(datasets): oh[i, self.doms.index(d) if d in self.doms else len(self.doms) - 1] = 1
        lg = self.net(torch.cat([oh, present_mask.float()], 1)).float()
        lg = lg.masked_fill(~present_mask, -1e4); w = torch.softmax(lg, 1) * present_mask.float()
        w = w / w.sum(1, keepdim=True).clamp(min=1e-8)
        with torch.no_grad():
            for i, d in enumerate(datasets): self.acc[d] += w[i].float().cpu().numpy(); self.cnt[d] += 1
        return w
    def flush_weight_log(self, path): pass
    def summary(self):
        out = {}
        for d in self.acc:
            m = self.acc[d] / max(1, self.cnt[d]); H = float(-(m.clip(1e-8) * np.log(m.clip(1e-8))).sum())
            out[d] = dict(mean_weights=dict(zip(self.names, m.round(4).tolist())), entropy=H, max_weight=float(m.max()))
        return out

def make_router(kind):
    if kind == "default": return qkd.DomainPriorRouter(TEACHERS, qkd.build_default_domain_prior())
    if kind == "uniform": return qkd.DomainPriorRouter(TEACHERS, {d: {t: 1.0 for t in TEACHERS} for d in DOMS + ["unknown"]})
    if kind == "valprior": return qkd.DomainPriorRouter(TEACHERS, {d: {t: VAL_PRI[d].get(t, 1.0) for t in TEACHERS} for d in VAL_PRI})
    if kind == "learned": return LearnedRouter(TEACHERS).to(DEV)

# ---- experiment specs ----
LW = qkd.LossWeights; BASE_SCHED = {k: copy.deepcopy(v) for k, v in qkd.STAGE_SCHEDULE_V2.items()}
ALL_B = {"frame", "temporal", "audio"}
def spec(name):
    s = dict(kind="student", stages=["A", "B", "C"], active={"A": {"frame"}, "B": ALL_B, "C": ALL_B}, sched=copy.deepcopy(BASE_SCHED), router="default", zero_fill=False, eval_active=None)
    hard = {k: LW(hard=1.0) for k in "ABC"}
    if name == "hard_only": s["sched"] = hard
    elif name == "frame_only": s["sched"] = hard; s["active"] = {k: {"frame"} for k in "ABC"}; s["eval_active"] = {"frame"}
    elif name == "no_stageC": s["stages"] = ["A", "B"]
    elif name == "kd_through_C": s["sched"]["C"] = copy.deepcopy(BASE_SCHED["B"])
    elif name == "zero_fill": s["zero_fill"] = True
    elif name == "uniform_router": s["router"] = "uniform"
    elif name == "valprior_router": s["router"] = "valprior"
    elif name == "learned_router": s["router"] = "learned"
    elif name.startswith("base_"): s["kind"] = "baseline"; s["eval_active"] = {"frame"}
    elif name != "full": raise ValueError(name)
    return s

# ---- evaluation ----
def loader_for(rows, active=None, bs=None):
    return DataLoader(qkd.JointDeepfakeDataset(rows, train=False, active_modalities=active), batch_size=bs or CFG["BS_EVAL"], shuffle=False,
                      num_workers=CFG["NUM_WORKERS"], pin_memory=torch.cuda.is_available(), prefetch_factor=4, timeout=600)
@torch.no_grad()
def predict(fn, rows, active=None):
    rec = defaultdict(list)
    for batch in loader_for(rows, active):
        b = qkd._to_device(batch); rec["logit"].append(fn(b).float().cpu().numpy()); rec["label"].append(batch["label"].numpy())
        rec["id"] += [str(x) for x in batch["id"]]; rec["dataset"] += [str(x) for x in batch["dataset"]]; rec["manip"] += [str(x) for x in batch["manipulation"]]
    return {k: (np.concatenate(v) if k in ("logit", "label") else np.array(v)) for k, v in rec.items()}
def summarize(test, oods, val):
    y, z, d = test["label"], test["logit"], test["dataset"]; p = sigmoid(z)
    per = {x: fast_auc(y[d == x], z[d == x]) for x in DOMS if (d == x).any()}
    m = dict(pooled_auc=fast_auc(y, z), macro_auc=float(np.nanmean(list(per.values()))), per_domain_auc=per, bal_acc=bal_acc(y, z > 0), eer=eer(y, p), ece=ece_bins(y, p),
             per_domain_bal_acc={x: bal_acc(y[d == x], z[d == x] > 0) for x in per}, val_auc=fast_auc(val["label"], val["logit"]))
    for k, o in oods.items(): m[f"ood_auc_{k}"] = fast_auc(o["label"], o["logit"]); m[f"ood_ece_{k}"] = ece_bins(o["label"], sigmoid(o["logit"]))
    return m

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s); qkd.SEED = s

def val_loader_for_trainer(active=None):
    vs = qkd.domain_balanced_sample(val_rows, CFG["VAL_SUBSET"], seed=qkd.SEED + 777) if len(val_rows) > CFG["VAL_SUBSET"] else val_rows
    return DataLoader(qkd.JointDeepfakeDataset(vs, train=False, active_modalities=active), batch_size=24, shuffle=False, num_workers=CFG["NUM_WORKERS"], pin_memory=True, prefetch_factor=4, timeout=600)

def configure_qkd(tag, sp, scale, seed):
    qkd.RUN_TAG = tag; d = OUTROOT / tag; (d / "ckpt").mkdir(parents=True, exist_ok=True); (d / "csv").mkdir(exist_ok=True); (d / "emerg").mkdir(exist_ok=True)
    qkd.CKPT_DIR, qkd.CSV_DIR, qkd.EMERGENCY_DIR = d / "ckpt", d / "csv", d / "emerg"; qkd.RESUME_CKPT_PATH = ""
    qkd.STAGE_ORDER = list(sp["stages"]); qkd.ACTIVE_BRANCHES_BY_STAGE_V2 = sp["active"]; qkd.STAGE_SCHEDULE_V2 = sp["sched"]
    qkd.STAGE_TRAIN_POOL_MAX_V2 = {k: max(2000, int(v * scale)) for k, v in {"A": 40000, "B": 60000, "C": 30000}.items()}
    seed_all(seed); return d

def build_student_parts():
    fp = nn.ModuleDict({n: nn.Sequential(nn.LayerNorm(embed_dims[n]), nn.Linear(embed_dims[n], 512)) for n in embed_dims}).to(DEV)
    return qkd.MultimodalStudent(fusion_dim=512, n_sampled_frames=4).to(DEV), fp, nn.Linear(512, 768).to(DEV)

def run_student(name, seed, scale, budget_h, eval_reserve_min):
    sp = spec(name); tag = f"{name}_s{seed}"; d = configure_qkd(tag, sp, scale, seed); t1 = time.time()
    student, fp, tproj = build_student_parts()
    if sp["zero_fill"]:
        class ZeroFillFusion(type(student.fusion)):
            def forward(self, tokens, kpm): return super().forward(tokens, torch.zeros_like(kpm))
        student.fusion.__class__ = ZeroFillFusion
    router = make_router(sp["router"])
    budget = qkd.BudgetManager(total_hours=max(0.1, budget_h), reserve_minutes=eval_reserve_min, start_time=time.time())
    trainer = qkd.KDTrainer(student, reader, usable, scalers, router, fp, tproj, train_rows, val_loader_for_trainer(sp["eval_active"]), budget)
    if sp["router"] == "learned":
        trainer.optimizer.add_param_group({"params": list(router.parameters())})
        try: trainer.scheduler.base_lrs.append(qkd.BASE_LR)       # keep the LR scheduler's per-group list in sync with the new group
        except Exception: pass
    trainer.run()
    if trainer.stage is not None:
        log(f"{tag}: cut off by budget; checkpoints KEPT in {d / 'ckpt'}")
        return dict(tag=tag, status="incomplete_budget", name=name, seed=seed), None
    trainer.ema.copy_to(student); student.eval()           # FINAL EMA weights, no checkpoint selection of any kind
    torch.save(student.state_dict(), d / "final_ema_student.pth")
    extra = dict(router_summary=router.summary()) if sp["router"] == "learned" else {}
    if sp["router"] != "learned":
        try: router.flush_weight_log(d / "router_weights_by_domain_epoch.csv")
        except Exception: pass
    train_h = (time.time() - t1) / 3600; log(f"{tag}: trained in {train_h:.2f}h; evaluating")
    fn = lambda b: student(b)["joint_logit"]
    val = predict(fn, qkd.domain_balanced_sample(val_rows, CFG["VAL_SUBSET"], seed=777), sp["eval_active"])
    test = predict(fn, test_rows, sp["eval_active"]); oods = {k: predict(fn, v, sp["eval_active"]) for k, v in ood_rows.items()}
    del trainer, student, fp, tproj; shutil.rmtree(d / "ckpt", ignore_errors=True); shutil.rmtree(d / "emerg", ignore_errors=True)
    return dict(tag=tag, name=name, seed=seed, scale=scale, status="done", train_hours=train_h, **extra), (test, oods, val)

def run_baseline(name, seed, scale, budget_h, eval_reserve_min):
    import timm; tag = f"{name}_s{seed}"; d = OUTROOT / tag; d.mkdir(exist_ok=True); seed_all(seed); t1 = time.time()
    archs = {"base_xception": ["legacy_xception", "xception", "efficientnet_b2"], "base_effb4": ["efficientnet_b4", "efficientnet_b2"]}[name]; net = None
    for a in archs:
        try: net = timm.create_model(a, pretrained=True, num_classes=1).to(DEV); used = a; break
        except Exception as e: log(f"[baseline] {a} unavailable: {e!r}")
    assert net is not None; log("baseline architecture:", used, sum(p.numel() for p in net.parameters()) / 1e6, "M params")
    if used not in archs[:1]: log(f"[baseline] WARNING: {name} fell back to '{used}' - report this in the paper, do not call it {name[5:]}")
    EP = 3; pool = int(310000 / EP * scale); opt = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=0.05)
    steps = EP * (pool // 32); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps); gs = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    step = 0; dl_budget_end = time.time() + max(0.1, budget_h - eval_reserve_min / 60) * 3600
    for ep in range(EP):
        rows = qkd.domain_balanced_sample(train_rows, pool, seed=seed + 1000 + ep); net.train()
        dl = DataLoader(qkd.JointDeepfakeDataset(rows, train=True, active_modalities={"frame"}), batch_size=32, shuffle=True, num_workers=CFG["NUM_WORKERS"], drop_last=True,
                        pin_memory=True, prefetch_factor=4, timeout=600)
        for bi, batch in enumerate(dl):
            if time.time() > dl_budget_end: return dict(tag=tag, status="incomplete_budget", name=name, seed=seed), None
            b = qkd._to_device(batch)
            with torch.autocast("cuda", dtype=torch.bfloat16 if qkd.BF16_OK else torch.float16, enabled=torch.cuda.is_available()):
                loss = qkd.safe_bce_logits(net(b["frame"]).squeeze(-1).float(), b["label"])
            opt.zero_grad(set_to_none=True); gs.scale(loss).backward(); gs.unscale_(opt); nn.utils.clip_grad_norm_(net.parameters(), 5.0); gs.step(opt); gs.update()
            if step < steps - 1: sch.step()
            step += 1
            if bi % 200 == 0: log(f"{tag} ep{ep} b{bi} loss {float(loss):.4f}")
    net.eval(); fn = lambda b: net(b["frame"]).squeeze(-1)
    val = predict(fn, qkd.domain_balanced_sample(val_rows, CFG["VAL_SUBSET"], seed=777), {"frame"})
    test = predict(fn, test_rows, {"frame"}); oods = {k: predict(fn, v, {"frame"}) for k, v in ood_rows.items()}
    return dict(tag=tag, name=name, seed=seed, scale=scale, status="done", arch=used, params_M=sum(p.numel() for p in net.parameters()) / 1e6, train_hours=(time.time() - t1) / 3600), (test, oods, val)

# ---- which experiments are already done (in this session's output or any attached previous output) ----
def read_results():
    out = {}
    for p in glob.glob(str(OUTROOT / "*" / "result.json")) + FOUND["result.json"]:
        try: j = json.load(open(p))
        except Exception: continue
        if isinstance(j, dict) and j.get("status") == "done" and j.get("tag") and "name" in j and "seed" in j: out[j["tag"]] = j
    return out
DONE = read_results(); plan = [(n, s) for n, s in SESSIONS[SESSION] if f"{n}_s{s}" not in DONE]
log("SESSION", SESSION, "plan:", plan, "| already done:", sorted(DONE))

# ---- benchmark -> SCALE (identical for all sessions) ----
EVAL_H = 0.6
if FIXED_SCALE is None and SESSION > 1:
    saved = []
    for p in FOUND["scale_session1.json"]:
        try: saved.append(json.load(open(p)))
        except Exception: pass
    assert saved, "Sessions >= 2 need session 1's output attached as an input (scale_session1.json), or paste the printed scale into FIXED_SCALE."
    FIXED_SCALE = float(saved[0]["scale"]); EVAL_H = float(saved[0].get("eval_h", EVAL_H)); log("scale read from attached session 1 output:", FIXED_SCALE, "| eval_h", EVAL_H)
if FIXED_SCALE is None:
    assert SESSION == 1, "Sessions >= 2 must have a scale (attach session 1 output or set FIXED_SCALE)."
    configure_qkd("bench", spec("full"), 1.0, 0); st, fp, tp = build_student_parts(); sp0 = spec("full")
    qkd.STAGE_ORDER = ["B"]; tr_loader = qkd.build_stage_train_loader(train_rows, "B"); opt = torch.optim.AdamW(list(st.parameters()) + list(fp.parameters()) + list(tp.parameters()), 1e-4)
    trainer_like = qkd.KDTrainer.__new__(qkd.KDTrainer)   # reuse _step without building a full trainer
    trainer_like.student, trainer_like.artifact_reader, trainer_like.usable, trainer_like.scalers = st, reader, usable, scalers
    trainer_like.router, trainer_like.feature_projections, trainer_like.temporal_proj, trainer_like.stage = make_router("default"), fp, tp, "B"
    trainer_like.optimizer = opt; trainer_like.scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available()); trainer_like.ema = qkd.EMA(st); trainer_like.global_epoch = 0
    st.train(); ts = []
    for i, batch in enumerate(tr_loader):
        t_ = time.time(); loss, _ = trainer_like._step(batch, sp0["sched"]["B"])
        if loss is not None: trainer_like._optimizer_step(loss)
        torch.cuda.synchronize(); ts.append(time.time() - t_)
        if i >= 25: break
    sB = float(np.median(ts[8:])) / tr_loader.batch_size    # s/sample, incl. data loading amortised by workers
    train_h = sB * (40000 * 0.6 + 180000 + 90000 * 0.9) / 3600 * 1.15                       # A is frame-only (cheaper), C has no teacher reads
    eval_h = sB * 0.45 * (len(test_rows) + sum(len(v) for v in ood_rows.values()) + CFG["VAL_SUBSET"]) / 3600 * 1.2
    n_runs = max(len(v) for v in SESSIONS.values())      # size the scale for the LARGEST session so every session fits in one 12h run
    per_run = (BUDGET_H - 0.35) / n_runs
    SCALE = float(np.clip((per_run - eval_h) / train_h, 0.08, 1.0))
    print(f"[bench] s/sample={sB:.4f} -> full-schedule training ~{train_h:.1f}h/run, eval ~{eval_h:.2f}h/run; sized for {n_runs} runs/session in {BUDGET_H:.1f}h -> SCALE={SCALE:.3f}")
    print(f"[bench] >>> later sessions read this automatically from scale_session1.json (or paste FIXED_SCALE = {SCALE:.3f}) <<<"); EVAL_H = eval_h
    del st, fp, tp, trainer_like, opt, tr_loader; torch.cuda.empty_cache(); gc.collect()
else:
    SCALE = float(FIXED_SCALE)
json.dump(dict(scale=SCALE, eval_h=EVAL_H, session=SESSION), open(OUTROOT / f"scale_session{SESSION}.json", "w"))
shutil.rmtree(OUTROOT / "bench", ignore_errors=True)

for k_, (name, seed) in enumerate(plan):
    tag = f"{name}_s{seed}"
    if left_h() < 1.0: log("not enough time left for", tag); continue
    budget_h = (left_h() - 0.08) / (len(plan) - k_)        # fair share of what is left; unused time rolls over to the next run
    log("=" * 20, "START", tag, f"scale={SCALE} | budget for this run {budget_h:.2f}h")
    try:
        meta, res = (run_baseline if spec(name)["kind"] == "baseline" else run_student)(name, seed, SCALE, budget_h, EVAL_H * 60 * 1.1)
    except Exception as e:
        import traceback; traceback.print_exc(); meta, res = dict(tag=tag, name=name, seed=seed, status=f"error: {e!r}"), None
    if res is not None:
        test, oods, val = res; meta["metrics"] = summarize(test, oods, val)
        np.savez_compressed(OUTROOT / tag / "preds.npz", **{f"test_{k}": v for k, v in test.items()}, **{f"ood_{o}_{k}": v for o, od in oods.items() for k, v in od.items()},
                            **{f"val_{k}": v for k, v in val.items()})
        print(json.dumps(meta["metrics"], indent=1, default=float))
    (OUTROOT / tag).mkdir(exist_ok=True); json.dump(meta, open(OUTROOT / tag / "result.json", "w"), indent=1, default=float)
    torch.cuda.empty_cache(); gc.collect()

# ---- aggregation over every result.json (this session + attached previous sessions) ----
ALL = {t: j for t, j in read_results().items() if "metrics" in j}
def flat(j):
    m = j["metrics"]; o = dict(config=j["name"], seed=j["seed"], scale=j.get("scale"), pooled_auc=m["pooled_auc"], macro_auc=m["macro_auc"], bal_acc=m["bal_acc"], eer=m["eer"], ece=m["ece"], val_auc=m["val_auc"])
    o.update({f"auc_{k}": v for k, v in m["per_domain_auc"].items()}); o.update({k: v for k, v in m.items() if k.startswith("ood_")}); return o
if ALL:
    R = pd.DataFrame([flat(j) for j in ALL.values()]).sort_values(["config", "seed"]); R.to_csv(OUTROOT / "B_all_runs.csv", index=False)
    cols = [c for c in R.columns if c not in ("config", "seed", "scale")]
    g = R.groupby("config")[cols]; agg = g.agg(["mean", "std"]); agg.columns = [f"{a}_{b}" for a, b in agg.columns]; agg["n_seeds"] = g.size(); agg.to_csv(OUTROOT / "B_summary_mean_sd.csv")
    print(R.round(4).to_string(index=False)); print(agg[[c for c in agg.columns if c.startswith(("pooled", "macro", "auc_ffpp", "ood_auc"))] + ["n_seeds"]].round(4).to_string())
    if "full" in set(R.config):
        base = R[R.config == "full"].set_index("seed"); dl = []
        for c in sorted(set(R.config) - {"full"}):
            o = R[R.config == c].set_index("seed")
            for m in cols:
                sh = base.index.intersection(o.index)
                if len(sh): dl.append(dict(config=c, metric=m, n_paired_seeds=len(sh), mean_delta_vs_full=float((o.loc[sh, m] - base.loc[sh, m]).mean()), sd=float((o.loc[sh, m] - base.loc[sh, m]).std()) if len(sh) > 1 else np.nan,
                                           n_runs_cfg=len(o), n_runs_full=len(base)))
        pd.DataFrame(dl).to_csv(OUTROOT / "B_delta_vs_full.csv", index=False)
        print("NOTE: configs with a single seed cannot be separated from run-to-run noise; compare their delta against the SD of 'full' across seeds.")
log("DONE", sorted(os.listdir(OUTROOT)))



SESSION = 2
SESSIONS = {
2: [("hard_only", 42), ("hard_only", 1), ("hard_only", 2)],
}
SESSION = 3
SESSIONS = {
3: [("frame_only", 42), ("no_stageC", 42), ("kd_through_C", 42), ("zero_fill", 42)],
}
SESSION = 4
SESSIONS = {
4: [("uniform_router", 42), ("valprior_router", 42), ("learned_router", 42)],
}
SESSION = 5
SESSIONS = {
5: [("base_xception", 42), ("base_effb4", 42)],
}