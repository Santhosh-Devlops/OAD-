# DyMETER — Java implementation

Faithful, dependency-free Java port of the online anomaly detection framework of

> **Catching Every Ripple: Enhanced Anomaly Awareness via Dynamic Concept Adaptation**
> J. Zhu, S. Cai, J. Chen, F. Deng, B. C. Ooi, W. Zhang — IEEE TPAMI, DOI 10.1109/TPAMI.2026.3682661

Every equation of the paper appears as a numbered comment next to the line that implements it, e.g.
`// Eq. (15): A(x) = R_e exp(lambda U_ctr (r_e - R_e))` in `DynamicThresholdOptimizer.score`.

| Module | Paper | Java |
|---|---|---|
| Static Concept-aware Detector (SCD) | Eq. 3–4, symmetric DNN autoencoder | `AutoEncoder` |
| Intelligent Evolution Controller (IEC) | Eq. 2, 5–8, evidential deep learning | `EvidentialController` |
| Dynamic Shift-aware Detector (DSD) | Eq. 9–14, instance-aware hypernetwork | `ShiftHypernetwork` |
| Dynamic Threshold Optimization (DTO) | Eq. 15–20, adaptive decision boundary | `DynamicThresholdOptimizer` |
| Training / inference | Algorithm 1, Algorithm 2 (drift re-initialisation, parallel offline update) | `DyMeterEngine` |
| Evaluation | AUCROC/AUCPR, 12 ablations, drift generator, robustness studies | `Metrics`, `Ablation`, `StreamGenerator`, `DyMeterExperiment` |
| Dashboard | 8 hash-routed pages, theme toggle, live progress and downloadable reports | `DyMeterServer`, `src/main/resources/index.html` |

The paper → code audit (including what the previous implementation was missing) is in
[`docs/EQUATION_TRACEABILITY.md`](docs/EQUATION_TRACEABILITY.md) and
[`docs/GAP_ANALYSIS.md`](docs/GAP_ANALYSIS.md); the extracted paper text is `docs/paper_ripple.txt`
and the superseded implementation is archived in `docs/legacy/`.

## Requirements

JDK 17 or later (`javac` and `java` on the `PATH`). No build tool and no third-party library: only
`java.base` plus the built-in `jdk.httpserver` for the web lab.

## Quick start

```powershell
New-Item -ItemType Directory -Force out | Out-Null
javac -d out src/main/java/dymeter/*.java

java -cp out dymeter.DyMeterSelfTest
java --add-modules jdk.httpserver -cp out dymeter.DyMeterServer
```

Open `http://localhost:8080` in a browser, upload a CSV, then choose **Run DyMETER**. The dashboard
and the server terminal both show live stage and percentage progress. Default settings are applied
automatically; optional tuning controls are tucked under **Advanced options**.

A labelled demo stream ships with the project (`data/sample_stream.csv`, 1500 rows, three concepts,
≈5 % anomalies). Regenerate it at any size:

```powershell
java -cp out dymeter.DyMeterExperiment --generate data/sample_stream.csv --rows 1500 --dimensions 8 --concepts 3
```

## Experiments

```powershell
# full ablation table, 3 independent runs (mean ± std), 900 Adam epochs, writes docs/results/*
java -cp out dymeter.DyMeterExperiment --data data/sample_stream.csv --variant all --runs 3 --epochs 900

# the paper's own budget (Adam lr 1e-2 decayed x0.96 over 2000 epochs, batch 64, window 64)
java -cp out dymeter.DyMeterExperiment --data data/sample_stream.csv --variant DyMETER --epochs 2000

# drift scenarios, robustness studies, hyper-parameter sweeps
java -cp out dymeter.DyMeterExperiment --data data/sample_stream.csv --drift mixed --variant all
java -cp out dymeter.DyMeterExperiment --data data/sample_stream.csv --noise 0.1 --coverage 0.5
java -cp out dymeter.DyMeterExperiment --data data/sample_stream.csv --variant all --runs 2 --tau 0.10

# any labelled CSV works (label/class/target/anomaly is used for scoring only)
java -cp out dymeter.DyMeterExperiment --data <path>\ionosphere.csv --variant all --runs 5
```

See [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md) for every option, the dashboard walkthrough and
the expected self-test output.

## The dashboard

Eight pages reachable from the side panel (`#/upload`, `#/pipeline`, `#/metrics`, `#/visuals`,
`#/drift`, `#/rows`, `#/report`, `#/about`):

1. **Upload & configure** — file picker with a preview, or the bundled sample stream served from
   `/data/…`.
2. **Run the pipeline** — select a run speed, keep the recommended defaults, or expand advanced
   options; follow the live progress bar and stage trace.
3. **Scores & metrics** — AUCROC, AUCPR, FPR, recall, precision, flagged records, drift counters,
   learned thresholds and the score histogram.
   Ranking metrics are marked **Not scored** with a reason if labels are missing or the evaluated rows
   contain only one class; the application does not invent metric values.
4. **Visualisations** — score vs adaptive threshold, uncertainty and reconstruction error, concept
   timeline. Every figure has a **Download PNG** button; the numbers export as CSV.
5. **Concept drift timeline** — each episode with the row number where it was declared, U_ctr, R_e,
   μ* at onset and a sentence explaining *why* it is concept drift.
6. **Row diagnostics** — the complete per-record trace with filters.
7. **Report & downloads** — a PDF report with dataset details, metrics, drift events and figures.
8. **How the method works** — a concise paper-to-code map.

## What the framework does, step by step

1. **SCD** (Eq. 3–4): a symmetric DNN autoencoder (depth 2–10, latent width set by PCA to ≥ 70 %
   explained variance) trains on the first 20 % of the stream with Adam (lr 1e-2, ×0.96 decay) and
   yields the reconstruction error `R_e`.
2. **IEC** (Eq. 2, 5–8): a two-layer ReLU evidential classifier trains with the focal evidential loss
   on pseudo-labels (`1` if `R_e > μ_p`, `0` otherwise, unknown if `U_ctr > μ_e`); its Dirichlet
   evidence gives the concept uncertainty `U_ctr` (mutual information).
3. **DSD** (Eq. 9–14): when `U_ctr` exceeds `μ_e` the instance is unfamiliar, so a hypernetwork turns
   it into per-layer parameter shifts `ΔΘ_d` and the dynamic detector `Θ_d = Θ_s + ΔΘ_d`
   re-estimates the reconstruction error without any retraining.
4. **DTO** (Eq. 15–20): `A(x) = R_e exp(λ U_ctr (r_e − R_e))` is compared with
   `μ* = μ⁰ + κ(μ⁰ − bAm)`, where `μ⁰` is the `(1−τ)`-quantile of the recent normal-score window
   `W_N` and `bAm` the median of the candidate window `W_C`. Drift re-initialises both windows;
   accumulated uncertainty (or `T_max`) triggers a parallel, non-blocking offline fine-tuning.

## Repository layout

```
src/main/java/dymeter/     Adam, Matrix, Numeric, Pca, AutoEncoder, EvidentialController,
                           ShiftHypernetwork, DynamicThresholdOptimizer, DyMeterEngine,
                           StreamData, StreamGenerator, Metrics, Ablation, DyMeterExperiment,
                           DyMeterServer, DyMeterSelfTest
src/main/resources/        index.html - the eight-page dashboard
data/sample_stream.csv     self-contained labelled drifting dataset
docs/                      EQUATION_TRACEABILITY.md, GAP_ANALYSIS.md, REPRODUCTION.md, REPORT.md,
                           results/, legacy/, paper_ripple.txt
```

## Validation

`java -cp out dymeter.DyMeterSelfTest` verifies the port, not just its wiring:

* **Gradients** — the analytic back-propagation of the SCD (`dL_SCD/dΘ_s`, Eq. 4), of the IEC
  (`dL_IEC/dΘ_c` for the focal and the cross-entropy objective, both evidence maps, Eq. 7) and of the
  DSD hypernetwork (`dL_DSD/dΘ_h`, Eq. 9–14) is compared with central finite differences on randomly
  perturbed parameters (agreement < 1e-6).
* **Equations** — closed forms of Eq. (6), (8), (15), (17), (18), (19); both triggers of Eq. (20);
  the drift branch of Algorithm 2 (windows discarded once per episode and re-seeded).
* **Metrics** — AUCROC/AUCPR/FPR/recall against hand-computed values.
* **Pipeline** — two synthetic drifting streams (ambiguous and independent drift): drift detection,
  DSD escalation, threshold re-initialisation, adaptive-threshold tracking and DTO calibration.

## Notes on fidelity

Every deviation from, or under-specification of, the paper is listed in
[`docs/GAP_ANALYSIS.md`](docs/GAP_ANALYSIS.md) §6: drift hysteresis and warm-up, how μ_e is obtained,
the cold-start rule after a threshold re-initialisation, the stale-window guard, the size of the
background fine-tuning pass, feature standardisation, the sign of τ and the two readings of the
evidence map `g(·)` of Eq. (2). Nothing was dropped silently; the four experimental families that are
*not* part of the DyMETER algorithm (18 third-party baselines, sequence SCD backbones, the Flink
deployment and the MNIST/FMNIST suites) are documented as out of scope in
`EQUATION_TRACEABILITY.md` §3.
#   O A D -  