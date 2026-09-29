# Reproducing the results

Everything runs from the project root (`C:\Users\MADHUMITHA\Documents\Phase1`) with JDK 17+.
The port has **no third-party dependencies**: only `java.base` and the built-in `jdk.httpserver`.

## 1. Build

```powershell
New-Item -ItemType Directory -Force out | Out-Null
javac -d out src/main/java/dymeter/*.java
```

## 2. Verify the port (gradient and equation checks)

```powershell
java -cp out dymeter.DyMeterSelfTest
```

Expected output (32 checks, all green):

```
1. Numerical gradient verification (analytic vs central differences)
  [ok]   SCD dL_SCD/dTheta_s matches finite differences (Eq. 4)
  [ok]   IEC dL_IEC/dTheta_c matches finite differences (focal Eq. 7)
  [ok]   IEC dL_IEC/dTheta_c matches finite differences (cross-entropy of DyMETER-ce)
  [ok]   DSD dL_DSD/dTheta_h matches finite differences (Eq. 9-14)
  [ok]   DSD starts from Delta_Theta_d = 0
2. Closed-form equation checks   (Eq. 6, 8, 15, 17, 18, 19, 20, drift branch)
3. Evaluation metrics            (AUCROC / AUCPR / FPR / recall)
4. End-to-end pipeline on a synthetic drifting stream
passed 32, failed 0
```

The finite-difference checks perturb individual weights and compare the measured loss change with the
analytically back-propagated gradient, so a wrong Jacobian cannot pass.

## 3. Interactive lab

```powershell
java --add-modules jdk.httpserver -cp out dymeter.DyMeterServer
```

Open <http://localhost:8080>. The page lets you

* upload a numeric CSV (a `label` / `class` / `target` / `anomaly` column is used for AUCROC/AUCPR
  only, never for training),
* inject abrupt / gradual / recurrent / mixed concept drift through the paper's stream generator,
* choose the variant, the Adam epoch budget, the minibatch size and τ, λ, κ,
* inspect the resolved architecture, the learned thresholds, the metrics, the score/threshold/
  uncertainty traces with concept boundaries and a per-row diagnostics table,
* run all 12 ablation variants in one click.

## 4. Command-line experiments

```powershell
# 1. create the self-contained labelled drifting dataset (1500 rows, 8 features, 3 concepts)
java -cp out dymeter.DyMeterExperiment --generate data/sample_stream.csv --rows 1500 --dimensions 8 --concepts 3

# 2. full ablation table, 3 independent runs, balanced budget
java -cp out dymeter.DyMeterExperiment --data data/sample_stream.csv --variant all --runs 3 --epochs 900

# 3. paper-faithful single run (Adam lr 1e-2, 2000 epochs, batch 64, window 64, tau 0.05)
java -cp out dymeter.DyMeterExperiment --data data/sample_stream.csv --variant DyMETER --epochs 2000

# 4. sensitivity study of a hyper-parameter (paper Fig. 4)
foreach ($tau in 0.02,0.05,0.10,0.20) {
  java -cp out dymeter.DyMeterExperiment --data data/sample_stream.csv --variant all --runs 2 --tau $tau --out docs/results
}
```

Results are written to `docs/results/<dataset>-results.csv` and `.md`. With `--data` pointing at one
of the benchmark CSVs (`ionosphere.csv`, `mammography.csv`, `satellite.csv`, `pima.csv`,
`machine_temperature_system_failure.csv`, `cpu_utilization_asg_misconfiguration.csv`,
`INSECTS_*.csv`, `nyc_taxi.csv`, `iot_arduino_four_sensor_stream.csv`) the same command reproduces
the paper's setting on those datasets; a `label` column is detected automatically
(`INSECTS_*` and the Numenta files are labelled).

The evaluation protocol of the CLI is the one of the paper: the first 20 % of the stream trains the
framework, the remaining 80 % is processed online, AUCROC/AUCPR are computed on the streamed part and
independent runs are reported as mean ± std.

## 5. Options

| Option | Meaning |
|---|---|
| `--data <csv>` | input stream (label columns used for scoring only) |
| `--generate <csv>` | write a labelled synthetic drifting dataset and exit |
| `--rows --dimensions --concepts --anomaly-rate --anomaly-shift --concept-shift` | synthetic dataset shape |
| `--variant <name\|all\|a,b>` | `--list` prints the 12 ablation variants |
| `--runs <n>` | independent runs (mean ± std) |
| `--drift none\|abrupt\|gradual\|recurrent\|mixed` | stream generator (paper §IV-A) |
| `--noise <ratio>` | noisy initialisation robustness study (paper §IV-J) |
| `--coverage <frac>` | insufficient concept coverage study (paper §IV-J) |
| `--epochs --batch --window --layers --lr` | training budget and architecture |
| `--tau --lambda --kappa --gamma --mu-o --mu-p` | DTO / IEC / pseudo-labelling hyper-parameters |
| `--mu-e <value>` | μ_e explicitly, instead of the derived 0.75-quantile of the training uncertainties |
| `--limit <rows>` | use only the first rows of the CSV |
| `--seed <n>` | seed of every random source |
| `--out <dir>` | result directory (`docs/results`, `none` disables writing) |

## 6. The dashboard

`java --add-modules jdk.httpserver -cp out dymeter.DyMeterServer` then open <http://localhost:8080>.
The lab is a nine-page, hash-routed site with a step-by-step side panel:

1. **Upload & configure** — file picker with a column/row preview, or the bundled `data/sample_stream.csv`
   (served from `/data/…`, so the browser does not re-upload it).
2. **Run the pipeline** — variant, Adam epochs, minibatch, drift generator, τ/λ/κ, μ_e, noise, seed;
   the page then shows which of the four stages ran.
3. **Scores & metrics** — AUCROC, AUCPR, FPR, recall, precision, concept/drift counters, learned
   thresholds and the anomaly-score histogram.
4. **Visualisations** — score vs adaptive threshold, concept uncertainty and reconstruction error,
   concept timeline; each figure is downloadable as **PNG** and its numbers as CSV.
5. **Concept drift timeline** — every episode with its row number, U_ctr, R_e, μ* at onset and the
   reason why the IEC declared drift.
6. **Row diagnostics** — the complete per-record trace with filters (anomalies, drift, resets).
7. **Ablation lab** — all 12 variants on the same stream in one click.
8. **Report & downloads** — the Markdown report (dataset details, epochs, timestamp, metrics, drift
   events with reasons, configuration) is generated and **downloaded automatically** after each run;
   the raw run JSON and the CSV exports are here too.
9. **Paper ↔ code map** — every equation with the class and method that implements it.

## 7. Preprocessing

`StreamData.standardise` z-scores every feature with the mean/standard deviation of the **historical
segment** and then applies the same transform to the stream, so the streaming statistics are
causally valid. The autoencoder, the uncertainty term and the DTO therefore operate on comparable
scales for datasets with heterogeneous units (e.g. Unix timestamps next to temperature values). The
transformation is applied by the CLI and the HTTP layer; the engine itself always receives plain
numeric vectors.
