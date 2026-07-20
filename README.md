# Overstride

Injury-risk anomaly detection for runners. Builds a per-athlete baseline from their own training history and running form, measures deviation from that baseline, and converts deviation into an injury-risk signal.

The premise: injury risk is personal. A training load that's routine for one runner is a red flag for another. Rather than asking *"is this load risky in general?"*, Overstride asks *"is this unusual **for you**?"*

---

## Two signals, kept separate

Overstride produces two outputs. It deliberately does **not** fuse them into a single number, because they don't have the same evidence quality — and pretending they do would be dishonest.

| Signal | What it is | Evidence basis | Output |
|---|---|---|---|
| **Training-load risk** | Deviation from your normal training pattern | Calibrated against real injury outcomes (74 athletes, 7 years) | Probability (e.g. `63%`) |
| **Biomechanical anomaly** | Deviation from your normal running form | Unsupervised — no injury-labeled biomechanics data exists to calibrate against | Flag + direction (e.g. `elevated right-side asymmetry`) |

The first is a real probability. The second is an anomaly flag informed by known biomechanical risk markers from the literature. Showing them separately means an athlete can see *which* signal is driving a warning, and how much to trust it.

---

## How it works

```
SESSION LOGS ──┐
               ├─→ weekly aggregate ─→ Mahalanobis vs. your baseline ─→ frozen logistic ─→ P(injury)
               │
RUNNING VIDEO ─┴─→ pose estimation ──→ Mahalanobis vs. your kinematic baseline ─→ anomaly flag
```

Both tracks share the same core logic: **build a baseline, measure how far you've strayed, score the deviation.** They differ only in the feature space (training load vs. joint kinematics) and in whether the deviation-to-risk mapping is calibrated or heuristic.

Mahalanobis distance is used over a plain z-score because training features are correlated — distance and intensity move together, and a per-feature z-score would double-count that. Mahalanobis accounts for the covariance structure of *your* training pattern:

```
D² = (x − μ)ᵀ Σ⁻¹ (x − μ)
```

---

## Stages

### Stage 1 — Input

Athletes log sessions as they train. Scoring runs weekly.

| Field | Type | Example |
|---|---|---|
| Date | date | auto |
| Distance | km | `8.5` |
| Intensity zones | km per zone (Z1–Z4) | `Z1: 6.0, Z3: 2.5` |
| RPE (perceived exertion) | 1–10 | `7` |
| Perceived training success | 1–5 | `4` |
| Strength training | count | `1` |
| Cross-training | hours | `0.5` |
| Rest day | bool | `false` |

Raw session logs are aggregated into a 22-feature weekly vector, plus week-over-week relative features (`week 0 / week-1`, `week 0 / week-2`). The athlete never sees this layer — they just log runs.

### Stage 2 — Training-load deviation *(calibrated)*

**Offline, once:**
1. Load dataset #1 (Mid-Long Distance Runners Injuries)
2. Per athlete: build baseline `(μ, Σ)` from clean (non-injury) weeks
3. Compute `D²` for every athlete-week
4. Fit logistic regression `P(injury) = 1 / (1 + e^-(β₀ + β₁·D²))` against real injury labels
5. Freeze coefficients

**Runtime, per athlete, weekly:**
1. Aggregate the week's sessions → 22-feature vector
2. Compute week-over-week relative features
3. Look up + incrementally update `(μ, Σ)` (Welford's algorithm)
4. Compute `D²` against baseline
5. Apply frozen logistic regression → probability
6. Return score + feature-contribution breakdown

**Cold start:** ~8–10 clean weeks are needed before `D²` is statistically meaningful. Until then the athlete sees *"building baseline"* — no score is forced.

**Covariance stability:** athletes with too few clean weeks get a shrinkage estimator falling back toward a population-level `Σ`, rather than trusting a noisy individual covariance estimate.

**Baseline integrity:** anomalous weeks are excluded from baseline updates. A genuinely risky week shouldn't quietly redefine what "normal" means going forward.

### Stage 3 — Biomechanical deviation *(unsupervised)*

- Athlete submits running footage periodically (lower cadence than session logs — filming has real friction)
- Pose estimation → joint angles, stride asymmetry, ground-contact time, hip extension, cadence, arm swing, torso lean, overstriding
- Build per-athlete kinematic baseline from prior footage
- Mahalanobis distance on kinematic features → anomaly score
- Interpreted against literature-established risk markers where side-view footage makes them computable: overstriding, contact-time asymmetry. (Knee valgus is a known risk marker too, but needs frontal-plane camera footage this pipeline doesn't currently capture.)

No injury-labeled biomechanics dataset exists to calibrate this against, so the output is a **flag, not a probability**. The mechanism is grounded — anomalous biomechanics are established precursors to overuse injuries (shin splints, IT band, stress fractures) — but the specific deviation-to-risk mapping is not something this project has validated.

### Stage 4 — Output

```
Training-load risk:      63%  ↑  driven by rel_total_km_week_0_2
Biomechanical anomaly:   FLAG    elevated overstriding, right-side asymmetry
```

### Stage 5 — Self-improving baseline

Both baselines update incrementally as the athlete accumulates history. This is the part that improves over time **without any new external data** — it's pure accumulation of that person's own runs.

Important distinction: the *baseline* self-improves. The *risk calibration* does not — it was learned once, offline, from labeled injury data, and is frozen. No amount of unlabeled personal footage can improve that mapping. But since calibration only needs to happen once, the athlete-facing behavior is still *"keep logging, it keeps getting better."*

---

## Validation

Split **by athlete**, not by week. Random week-level splits leak the same athlete's weeks into both train and test, and the model partially memorizes their baseline — inflating accuracy artificially.

**Benchmark to beat:** Lövdal et al. (2021) achieved AUC `0.724` (day-level) and `0.678` (week-level) with bagged XGBoost on this same dataset.

**Hypothesis:** their model was trained on the pooled population — it learns what training patterns are risky *in general*. Overstride's baseline is athlete-relative. That better matches how coaches actually reason about injury risk, and may capture signal a population-level model averages away. Not guaranteed to win, but it's a real methodological argument, and it's testable.

**Class imbalance is real:** only 575 injured samples. Injuries are rare events. Handle with class weights or undersampling, and judge on recall/AUC rather than raw accuracy.

---

## Datasets

| Dataset | Role | Injury labels | Link |
|---|---|---|---|
| **Mid-Long Distance Runners Injuries** — 74 athletes, 7 years, 22 weekly features | Stage 2 calibration | ✅ 575 injured samples | [GitHub](https://github.com/josedv82/public_sport_science_datasets) |
| **1,798 Healthy & Injured Subjects** (Ferber et al.) | Possible Stage 3 calibration — injured-subject count not yet verified | ✅ subset | [Scientific Data](https://www.nature.com/articles/s41597-024-04011-7) |
| **Fukuchi et al. running biomechanics** — 28 runners, 3D mocap | Baseline reference | ❌ | [Figshare](https://doi.org/10.6084/m9.figshare.4543435) |
| **AthleticsPose** — 23 athletes, 8 synced cameras, real track events | Validate pose-extraction accuracy | n/a | [arXiv](https://arxiv.org/html/2507.12905v1) |
| **ASPset-510** — 330k frames, outdoor sports, 3D keypoints | Validate pose-extraction accuracy | n/a | — |
| Own / marathon footage | Population baseline + target-athlete footage | n/a | user-sourced, or `scripts/scrape_footage.py` |

All are publicly accessible. None require institutional credentials.

---

## Scope decisions

**Re-identification is out of scope.** The target use case is evaluating *one individual athlete*, not crowd-scale marathon analysis. That single decision eliminates the hardest problem in the pipeline — persistent identity across camera cuts, occlusion-heavy crowds, and appearance-based ReID. Within a single continuous shot, standard tracking (ByteTrack) holds IDs fine, and that's all this needs.

**Marathon footage's actual role** is bulk *anonymous* population-baseline samples — many independent examples of "normal running form." Identity doesn't matter for a pooled distribution the way it matters for a per-athlete one, so no tracking infrastructure is needed for that use.

**Multi-person ≠ more accurate.** More runners in frame means *more data volume*, but *noisier per-person extraction* (occlusion, ID switching). Two different meanings of "accuracy," and conflating them leads to bad architecture choices.

---

## Stack

| Layer | Tool |
|---|---|
| Data & stats | Python, pandas, NumPy, SciPy (Mahalanobis), scikit-learn |
| Pose estimation | MediaPipe (single-person) / YOLO-pose (multi-person) |
| Tracking | ByteTrack — single continuous shot only |

---

## Build order

1. **Stage 2 offline calibration** — dataset #1 → per-athlete baselines → Mahalanobis → logistic regression → athlete-level holdout validation against the published AUC benchmark
2. **Stage 2 runtime pipeline** — weekly aggregation → scoring function
3. **Stage 1 input schema** — session logging
4. **Stage 3 pose pipeline** — validate extraction accuracy on AthleticsPose/ASPset *before* trusting it on real footage
5. **Stage 3 baseline + anomaly scoring** — mirrors Stage 2's structure, unsupervised
6. **Stages 4 & 5** — combined output, incremental baseline updates

Steps 1–3 and 5 (training-load track) constitute a complete, defensible project on their own. The biomechanics track is a genuine scope expansion, not a prerequisite for a working, honest deliverable.

---

## References

- Lövdal, S., Den Hartigh, R., & Azzopardi, G. (2021). *Injury Prediction in Competitive Runners With Machine Learning.* — source of the primary dataset and the AUC benchmark.
- Ferber, R., Brett, A., Fukuchi, R. K., Hettinga, B., & Osis, S. T. (2024). *A Biomechanical Dataset of 1,798 Healthy and Injured Subjects During Treadmill Walking and Running.* Scientific Data.
- Fukuchi, R. K., Fukuchi, C. A., & Duarte, M. (2017). *A public dataset of running biomechanics and the effects of running speed on lower extremity kinematics and kinetics.* PeerJ.
