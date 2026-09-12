# Ultimate-Deepfake-model
Offline multi-teacher knowledge-distillation pipeline that compresses four frozen,
domain-specialist deepfake-detection teachers (frame/DFDC, frame/DiffusionFace,
video/FaceForensics++, audio/LAV-DF) into a single compact multimodal student
(EfficientNet-B2 frame path + small HF-stream CNN + short temporal transformer,
fused through a deterministic domain-prior router).

This repo contains the full pipeline used to precompute teacher features, train
the student under a 3-stage curriculum (warmup → multimodal KD → finetune), and
run comprehensive evaluation (in-domain, OOD, baselines, ablations, significance
tests) — as described in the accompanying paper.

## Pretrained models

All checkpoints referenced by this pipeline are hosted publicly on Kaggle:

| Checkpoint | Description | Link |
|---|---|---|
| Teacher checkpoints | Frozen teacher weights consumed by `PRECOMPUTE_TEACHERS` | [Kaggle Model](https://www.kaggle.com/models/syedazmulhasansabbir/ultimate-model-checkpoint) |
| Best-AUC student | Student checkpoint with the highest validation AUC | [Kaggle Model](https://www.kaggle.com/models/syedazmulhasansabbir/ultimate-model-auc-pth) |
| EMA-AUC student | EMA-averaged student weights, evaluated separately from the raw best-AUC checkpoint | [Kaggle Model](https://www.kaggle.com/models/syedazmulhasansabbir/ultimate-model-ema-auc-pth) |

To reproduce evaluation results, point `CKPT_DIR` (or the equivalent restore path)
at the downloaded checkpoint(s) above and run the pipeline in `EVALUATE` mode
(see below). `LOAD_EMA_WEIGHTS_FOR_EVAL` in the script controls whether the raw
or EMA student weights are loaded.

## Pipeline modes

The script runs one of three modes per session, set via the `MODE` variable near
the top of the configuration block:

1. **`PRECOMPUTE_TEACHERS`** — runs each frozen teacher once over the manifest
   and caches outputs to sharded `.pt` artifact files. Safe to re-run; resumes
   from remaining rows until every teacher reports 0 remaining.
2. **`TRAIN_STUDENT`** — trains the student against the cached teacher
   artifacts under the staged curriculum. Never runs evaluation.
3. **`EVALUATE`** — loads the best saved student checkpoint (`best_auc.pth`)
   and runs the full test / OOD / baseline / ablation / significance-testing
   suite. Never trains.

## Usage

```bash
pip install -r requirements.txt   # or let the script self-install decord/timm
python quadkd_offline_pipeline.py
```

Before running, edit the configuration section of `quadkd_offline_pipeline.py`
to set:
- `CKPT_PATHS` — paths to the teacher checkpoints (see table above)
- `MANIFEST_PATH` — path to the dataset manifest (FF++ / Celeb-DF / DFDC /
  DiffusionFace / LAV-DF)
- `MODE` — one of the three modes described above

The script is designed to run as a single Kaggle-notebook cell or standalone
script per session, with wall-clock budget tracking and mid-epoch checkpointing
so long runs can resume across sessions.

## Requirements

- PyTorch, timm, decord (auto-installed if missing), OpenCV/PyAV for video
  decoding
- Recommended: run on a Kaggle GPU environment (the pipeline includes several
  Kaggle-specific safeguards — file-descriptor sharing-strategy fixes,
  CPU-first checkpoint loading, subprocess-isolated video decoding, etc.)

## Citation

If you use this code or the released checkpoints, please cite our paper (citation
details to be added upon publication).

## License

See [LICENSE](./LICENSE).
