# Pressure-map posture recognition with YOLO + Transformer (PoPu dataset)

A two-stage pipeline that treats a body-pressure mat reading as a low-resolution image:

1. **YOLO** (YOLO11n) detects body-part blobs (head, torso, arms, legs) in each pressure frame.
2. A small **Transformer encoder** reads a window of per-frame YOLO tokens and predicts the lying posture
   (supine / left / right / prone) and whether a posture transition happens in the window.
3. **Uncertainty**: MC-dropout and a 5-member deep ensemble, with temperature scaling, ECE and
   error-detection metrics. Everything is evaluated **leave-subjects-out** (5 folds, 12 unseen subjects per fold).
4. **Stretch goal**: a toy simulated adjustable surface and a Q-learning policy driven by the predicted posture.

> **Scope warning.** PoPu contains only *lying* postures. There are no sitting postures in this dataset, and the
> transition data is *simulated* (see [Limitations](#limitations-read-before-citing-any-number)).

## Headline results (5 folds, 60 subjects, every test subject unseen in training)

| Task | Result |
|---|---|
| Posture classification (4 classes), deep ensemble | **97.6% ± 1.7%** accuracy, macro-F1 0.976, ECE 0.014 |
| Same, single model | 97.4% ± 1.8% accuracy |
| Error detection from predictive entropy | AUROC 0.94; accuracy 99.3% when keeping the 90% most confident windows |
| YOLO body-part detection (held-out subjects) | mAP50 0.720 ± 0.008, mAP50-95 0.330 ± 0.008 |
| Transition detection (simulated transitions) | AUROC 1.000, **but see limitations: this task is too easy to be informative** |
| Toy RL surface policy | ~11% simulated peak-pressure reduction; no benefit from posture or uncertainty information |

Full tables are in [`results/summary.md`](results/summary.md); per-fold numbers are in `results/fold*.json`.
(± is the standard deviation across the 5 folds.)

## Dataset

**PoPu** (Pressure Over & Pressure Under), CC0-licensed: 60 volunteers x 28 predetermined poses
(4 postures x 7 variations) x 30 samples per pose = **50,400 pressure frames**. This project uses the
*Tactilus* sheet placed over the mattress (64 x 27 grid, values 0-104) and the body-part segmentation annotations.
For each pose the volunteer shifted slightly twice, giving three "shift files" of 10 snapshots each.
The under-mattress Sensomatt data is not used.

## Method

### Stage 1: YOLO body-part detector
- **Input**: each 64x27 frame becomes a 3-channel image (raw, log-scaled, Sobel gradient), upsampled 4x and
  centred on a 256x256 canvas. The canvas is built explicitly, so training and manual inference see identical inputs.
- **Labels**: the dataset's body-part boxes (Head, Torso, Upper_Arm, Lower_Arm, Upper_Leg, Lower_Leg).
  Only one image per (subject, pose, variation) was annotated. The code finds which shift file that image came from
  (about 90% are the initial-pose file) and trains YOLO on every 2nd frame of that file: **8,280 labelled images**.
- **Training**: pretrained `yolo11n.pt`, up to 40 epochs with patience 15 (early stopping ended the runs at 22-25 epochs),
  no colour jitter or mosaic (not natural images), horizontal flip allowed (body-part classes have no left/right).
  **One detector per fold**, trained only on that fold's training subjects.

### Stage 2: per-frame tokens and Transformer
- **Token per frame (74 dims)** = top box per body-part class `[present, conf, cx, cy, w, h]` (36)
  + PCA(32) of global-pooled YOLO neck features + 6 pressure summary statistics.
- **Windows**: 10 frames, stride 5 over the real 30-frame sequences, plus simulated transition clips.
- **Model**: 2-layer pre-norm Transformer encoder (d=64, 4 heads, dropout 0.2) with a CLS token and two heads
  (posture, transition). Trained with AdamW, noise and frame-drop augmentation, early stopping on validation loss.
- **Transitions are simulated**: real frames of posture A, a blurred cross-fade, then real frames of posture B
  from the same subject. Real windows that span a small body shift (same posture) serve as hard negatives.

### Uncertainty and evaluation
- **Splits**: 5 leave-subjects-out folds; per fold 42 train / 6 validation / 12 test subjects. PCA, token
  standardisation, YOLO and the transformer only ever see training subjects. Validation subjects drive early stopping
  and temperature scaling.
- **Methods**: single model, MC-dropout (30 passes), deep ensemble (5 members), each raw and temperature-scaled.
- **Metrics**: accuracy, macro-F1, NLL, Brier, ECE (15 bins), entropy-based error detection (AUROC, AURC),
  selective accuracy at fixed coverage.
- Test set per fold: 2,256 windows (about 74% real, 26% simulated transitions), 11,280 in total.

## Results in detail

### Posture classification

| method | calibration | accuracy | macro-F1 | NLL | ECE |
|---|---|---|---|---|---|
| single | raw | 0.974 ± 0.018 | 0.974 | 0.133 ± 0.124 | 0.019 ± 0.015 |
| MC-dropout | raw | 0.974 ± 0.018 | 0.974 | 0.108 ± 0.100 | 0.013 ± 0.011 |
| deep ensemble | raw | **0.976 ± 0.017** | 0.976 | **0.109 ± 0.111** | 0.014 ± 0.013 |
| deep ensemble | temp-scaled | 0.976 ± 0.017 | 0.976 | 0.190 ± 0.261 | 0.014 ± 0.013 |

![Reliability diagrams](results/figures/reliability_posture.png)

- **Unseen-subject accuracy is high (97.6%)**, but the spread between subjects is large: per-subject accuracy has
  mean 0.976, std 0.053 and minimum 0.633 over the 60 subjects
  ([histogram](results/figures/per_subject_accuracy.png)).
- **Errors are concentrated in side-lying postures.** Per-class accuracy: supine 99.4%, prone 99.0%,
  left 96.1%, right 95.8%. Left and right windows get confused with each other and with prone (occasionally supine), as the
  confusion matrix shows. The worst subject (ID 15, 63.3%) has 38 of 94 left/right windows predicted as prone.
  The other low subjects (IDs 31, 47, 43; 86-90%) also lose accuracy mainly on left/right windows.
- **The model knows when it is unsure.** Mean predictive entropy is 0.026 on correct windows vs 0.373 on the
  274 wrong ones; selective accuracy rises from 97.6% (all) to 99.3% at 90% coverage.

![Confusion matrix](results/figures/confusion_matrix.png)
![Risk-coverage](results/figures/risk_coverage.png)

### Calibration: what actually helped
- **Raw models were already well calibrated** (ECE 1.3-1.9%). MC-dropout and the ensemble lowered ECE a little
  compared with a single model, but the differences are within the fold-to-fold spread.
- **Temperature scaling did not help reliably.** It helps the single model slightly (ECE 0.019 → 0.016), but for
  MC-dropout and the ensemble it left ECE unchanged and *increased* NLL (ensemble 0.109 → 0.190), with a large
  fold-to-fold spread. The cause is visible in the fitted temperatures: in folds 2 and 4 the validation set (only
  6 subjects, very few errors) pushed T down to 0.18-0.44, over-sharpening the test predictions.
  **Recommendation: use the raw ensemble predictions.** Temperature scaling would need a larger or cross-validated
  calibration set.
- The reported MCE (0.4-0.5) comes from sparsely populated bins and is not informative.

### Ablation: where does the accuracy come from? (single model, temperature-scaled)

| variant | accuracy | macro-F1 | NLL | transition AUROC |
|---|---|---|---|---|
| **full** (boxes + features + stats, 10 frames) | **0.974 ± 0.018** | 0.974 | 0.123 | 1.000 |
| YOLO boxes + features | 0.972 ± 0.019 | 0.972 | 0.120 | 1.000 |
| YOLO boxes only | 0.964 ± 0.018 | 0.964 | 0.146 | 0.999 |
| YOLO features only | 0.932 ± 0.011 | 0.932 | 0.197 | 1.000 |
| pixels PCA + stats (no YOLO) | 0.959 ± 0.019 | 0.959 | 0.151 | 1.000 |
| pixels PCA only (no YOLO) | 0.956 ± 0.020 | 0.956 | 0.165 | 0.999 |
| pressure stats only | 0.458 ± 0.030 | 0.450 | 1.184 | 0.999 |
| full, single frame (L=1) | 0.965 ± 0.017 | 0.965 | 0.154 | 0.511 |
| full, 5 frames | 0.970 ± 0.018 | 0.970 | 0.138 | 0.832 |

![Ablation](results/figures/ablation_accuracy.png)

How much to read into this:
- **YOLO tokens give a modest gain over a no-YOLO baseline**: the full model beats "pixels PCA only" in 5 of 5 folds
  (+1.8 points on average) and "pixels + stats" in 3 of 5 (+1.5 points). Boxes alone were 0.8 points above the pixel
  baseline on average. That is a consistent but small advantage, with one seed and 5 folds. The no-YOLO baseline is
  deliberately simple (PCA to 32 dimensions into the same Transformer); a CNN trained directly on the frames was
  **not** tested, so this does not show that YOLO beats a well-tuned end-to-end model.
- **Temporal context helps a little**: 96.5% (1 frame) → 97.0% (5) → 97.4% (10); the 10-frame model beats both
  shorter windows in 5 of 5 folds.
- **Ensembling adds almost nothing to accuracy** (+0.2 points over a single model, inconsistent sign across folds).
- Pressure summary statistics alone reach only 45.8% (chance is 25%), so the postures are not separable by
  simple load or contact-area measures.

### YOLO detection quality

mAP50 0.720 ± 0.008, mAP50-95 0.330 ± 0.008, precision 0.731, recall 0.716 on held-out subjects.

| class | mAP50-95 |
|---|---|
| Torso | 0.523 ± 0.015 |
| Upper_Leg | 0.519 ± 0.016 |
| Lower_Leg | 0.329 ± 0.023 |
| Head | 0.263 ± 0.033 |
| Upper_Arm | 0.198 ± 0.013 |
| Lower_Arm | 0.147 ± 0.011 |

Large body parts are detected well; arms and head (small blobs on a 64x27 grid, where a few cells decide the box)
are hard. Box quality at the strict IoU range is limited, which is expected given annotations were drawn on
colourised low-resolution images.

### Transition detection (read this carefully)
The detector reaches AUROC ≈ 1.000, accuracy 0.999, and flags only 0.0-0.3% of real stable or slightly shifted
windows. This number should **not** be taken as evidence that real posture transitions would be detected:
- The transitions are simulated cross-fades. Even the stats-only model (6 numbers per frame) reaches AUROC 0.999,
  so the simulated transitions carry an easy-to-spot artefact (blur, intermediate pressure).
- The single-frame model scores 0.511 (chance) and the 5-frame model 0.832, confirming that the detector
  uses frame-to-frame change as designed. The only mildly real signal is that natural small body shifts within a
  posture are almost never flagged (false-positive rate ≈ 0.2%).
- Real transitions (rolling over, in-between postures) are not in PoPu. Validating this head needs recordings of
  actual turns.

### Stretch goal: toy RL on a simulated adjustable surface

A hand-designed environment: four support zones (head/shoulders, torso/lumbar, hips/thighs, legs/heels), five
firmness levels each. Zone pressures come from the real measured frame; softening a zone lowers its peak and pushes
part of the load to its neighbours; side-lying gains more from softening than supine or prone. A tabular
Q-learning policy picks one zone adjustment per step over a 6-step episode.

| policy | mean reward | simulated peak pressure | peak reduction | adjustments / episode |
|---|---|---|---|---|
| static (no-op) | 0.000 | 0.477 | 0.0% | 0.0 |
| random | -0.140 | 0.488 | -2.2% | 5.3 |
| heuristic: soften hottest zone | 0.001 | 0.373 | **21.8%** | 3.0 |
| RL, posture-agnostic | 0.061 | 0.424 | 11.1% | 2.0 |
| RL, true posture (oracle) | 0.063 | 0.423 | 11.4% | 2.1 |
| RL, predicted posture | 0.063 | 0.423 | 11.4% | 2.1 |
| RL, predicted posture + uncertainty gate | 0.062 | 0.423 | 11.4% | 2.1 |

![RL](results/figures/rl_peak_reduction.png)

What this does and does not show:
- The RL policy learns something sensible (far better than random, fewer adjustments than the heuristic).
- **Knowing the posture, or being uncertain about it, made no measurable difference** here: posture-agnostic,
  oracle, predicted and gated policies are all within noise. The toy environment lets the policy infer what it needs
  from the pressure pattern itself. So this experiment does not demonstrate the value of the posture classifier
  or its calibrated uncertainty.
- The simple heuristic reaches a larger peak reduction (21.8%) than RL (11.4%). RL scores higher *reward* only
  because the reward I designed penalises extreme firmness levels and adjustments, which the heuristic ignores.
- The environment is **not a validated biomechanical model**. These numbers say nothing about real mattresses or
  clinical benefit.

## Limitations (read before citing any number)

1. **Lying postures only.** No sitting data exists in PoPu; results do not transfer to sitting.
2. **One sensor, one protocol.** All data comes from one pressure sheet on one mattress, with volunteers placed in
   instructed poses. Labels are the instructed pose, not independently verified. Other mats, bedding or
   unconstrained sleep may behave very differently.
3. **Simulated transitions** (see above). The transition head is unvalidated on real transitions.
4. **Windows are strongly correlated.** 30 frames per pose come from one short recording, and windows overlap
   (stride 5 of 10). The 11,280 test windows are far fewer independent samples than that count suggests; the
   subject-level split prevents leakage, and the 60-subject per-subject spread (std 0.053) is the more honest
   uncertainty measure.
5. **Approximate YOLO labels.** Only one image per pose was annotated, so labels are exact for the annotated
   shift file only; the detector is then applied to the other shift files, where boxes can be slightly off.
6. **Token distribution mismatch.** Each fold's YOLO was trained on the same subjects the Transformer trains on, so
   training tokens are cleaner than test tokens. This makes held-out numbers conservative, not optimistic, but
   nested cross-fitting would remove the mismatch.
7. **Single seed, one GPU run, 5 folds.** Ablation differences of about 1 point are suggestive, not conclusive.
8. **Small validation set for calibration** (6 subjects), which made temperature scaling unstable.

## Reproducing

Requirements: Python 3.10+, a CUDA GPU (a Colab T4 is enough), `pip install -r requirements.txt`.
Tested with ultralytics 8.4.190 and torch 2.11 on Colab.

```bash
# smoke test, ~10 min, 12 subjects, tiny settings (numbers meaningless)
python run_all.py --data /path/to/PoPu-main.zip --work ./work_debug --debug

# full run, 5 folds
python run_all.py --data /path/to/PoPu-main.zip --work ./work

# a single fold, or resume after prep
python run_all.py --data /path/to/PoPu-main.zip --work ./work --folds 0
python run_all.py --data /path/to/PoPu-main.zip --work ./work --skip_prep --folds 1 2 3 4
```

`--data` accepts the repository zip, `PoPu_data.zip`, or an extracted `PoPu_data/` folder.
`run_colab.ipynb` runs the same commands on Colab. Config overrides: `--set yolo.epochs=60 transformer.ensemble=3`
(see `configs/default.yaml`). The pipeline is seeded (`seed: 42`) but GPU kernels are not bit-exact, so expect small
differences.

## Project layout

```
configs/default.yaml          all hyper-parameters
run_all.py                    runs steps 1-6 + bundling
scripts/
  01_prepare_data.py          extract PoPu, cache frames, folds, YOLO dataset, synthetic transitions, windows
  02_train_yolo.py            fine-tune YOLO per fold, evaluate mAP on held-out subjects
  03_extract_tokens.py        run detector on all frames -> boxes, pooled features, stats
  04_train_eval.py            Transformer ensemble, MC-dropout, temperature scaling, metrics, ablations
  05_aggregate.py             summary.json / summary.md / figures over folds
  06_rl_demo.py               toy surface-adjustment policy
  99_collect_results.py       bundle results into a zip
src/
  data/       popu_io, render (3-channel YOLO input), splits, windows (synthetic transitions), tokens
  models/     yolo_stage (training + token extractor), transformer
  uncertainty/calibration.py  ECE, NLL, Brier, temperature scaling, AUROC, selective accuracy
  rl/surface_env.py           simulated surface + Q-learning
results/                      the run reported above (metrics, figures, predictions, Transformer weights)
```

## Data and licence
PoPu is released under CC0 (see the dataset repository). The dataset's own documentation should be consulted for
how to cite it.
