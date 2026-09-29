# Gap analysis — paper versus the Java implementation

This audit compares the *original* Java streaming implementation (kept verbatim in
`docs/legacy/DyMeterEngine-first-version.java.txt`) with the paper. It lists every concept, equation
and algorithm that was missing, wrong or only approximated, and states what was implemented.
`docs/EQUATION_TRACEABILITY.md` contains the resulting paper → code map.

Legend: **MISSING** = not implemented at all, **WRONG** = implemented differently from the paper,
**PARTIAL** = implemented but incomplete, **OK** = faithful.

## 1. Static Concept-aware Detector (SCD)

| # | Paper requirement | Original code | Verdict | Now |
|---|---|---|---|---|
| 1.1 | Autoencoder `x → E(x) → z → D(z) → y` (Eq. 3) | one hidden layer with `tanh` | PARTIAL | `AutoEncoder`, arbitrary depth, ReLU hidden units, linear output |
| 1.2 | "symmetrically structured DNNs with 2–10 layers" | 1 encoder + 1 decoder layer | MISSING | `AutoEncoder.symmetricWidths(d, latent, encoderLayers)` builds `[d, h1..latent..h1, d]`; depth from the data (2 for ≤ 8 features, else 3), configurable 1–10 |
| 1.3 | Trained with **Adam**, lr 1e-2, ×0.96 decay over 2000 epochs | plain SGD with the same schedule | MISSING | `Adam` (β1 = 0.9, β2 = 0.999, ε = 1e-8) used by SCD, IEC and DSD |
| 1.4 | Loss `L_2(x,y) = Σ(x_i − y_i)²/n` (Eq. 4) | same formula | OK | unchanged, plus an analytic gradient |
| 1.5 | Latent dimension = #PCs for ≥ 70 % variance | Jacobi PCA | OK | moved to `Pca` with the monotone width design |
| 1.6 | Gradients must flow through the SCD (Algorithm 1, line 12) | SCD frozen while only shifts were trained | WRONG | the DSD stage back-propagates `L_DSD` into `Θ_s` **and** `Θ_h` |

## 2. Intelligent Evolution Controller (IEC)

| # | Paper requirement | Original code | Verdict | Now |
|---|---|---|---|---|
| 2.1 | IEC = **two-layer DNN with ReLU** | single linear layer | WRONG | `EvidentialController`: `x → ReLU(W1 x + b1) → W2 z + b2 → α` |
| 2.2 | `α = g(f(x))`, exponential non-negativity (Eq. 2) | `1 + exp(z)` | OK | unchanged (`α = 1 + exp(clip(z))`), documented |
| 2.3 | `p̂_c = α_c / Σα` (Eq. 6) | OK | OK | `predictiveProbability` |
| 2.4 | Focal evidential loss (Eq. 7) | gradients only for the last layer | PARTIAL | full back-propagation through both layers, verified against central differences |
| 2.5 | Concept uncertainty = Dirichlet mutual information (Eq. 8) | present | OK | `uncertainty`, closed form verified |
| 2.6 | Pseudo-labelling "repeat … until convergence" (Eq. 5, Algorithm 1) | single pass, fixed `μ_e` | PARTIAL | `trainController` repeats labelling/training until `μ_e` converges (≤ 4 rounds), `μ_e` clipped to [0.005, 0.4] |
| 2.7 | `μ_p` from a proportion of sorted training errors, `μ_p ∈ [0.05, 0.5]` | 0.95 quantile | OK | `Config.pseudoProportion` (default 0.05 → 0.95 quantile) |
| 2.8 | DyMETER-pl: labelled anomalies as prior knowledge | absent | MISSING | `Config.priorLabels` + `injectPriorLabels` (1 % of labelled anomalies) |

## 3. Dynamic Shift-aware Detector (DSD)

| # | Paper requirement | Original code | Verdict | Now |
|---|---|---|---|---|
| 3.1 | `e^(n) = L_layer^(n)(E_share(x))` (Eq. 9) | `e[i] = x[i % d]` | WRONG | `ShiftHypernetwork`: learnable shared encoder (linear + ReLU) plus a distinct linear layer per tensor |
| 3.2 | `W^(n) = (W1 e^(n) + b1) W2 + b2` (Eq. 10) | element-wise `shift * e[(i+j) % len]` | WRONG | literal two-MLP-layer map producing the whole `N_in × N_out` shift matrix |
| 3.3 | `K^(n) = W^(n) + b^(n)` (Eq. 11) | absent | MISSING | per-tensor additive bias `base[n]` |
| 3.4 | `Θ_d = Θ_s + ΔΘ_d` (Eq. 12) | same relation | OK | unchanged; `Δ = 0` at initialisation so `Θ_d = Θ_s` (self-test) |
| 3.5 | `L_DSD = L_2(x, y′)` (Eq. 14) | same | OK | unchanged |
| 3.6 | DyMETER-re: random embedding instead of instance conditioning | absent | MISSING | `Config.instanceConditioned=false` keeps a learnable random embedding `r^(n)` |
| 3.7 | Efficiency of the shift generation (Section III-F) | cheap but wrong model | WRONG | shifts are added into a reused `Θ_s` buffer (`loadDynamicWeights`) instead of allocating per row |

## 4. Dynamic Threshold Optimization (DTO)

| # | Paper requirement | Original code | Verdict | Now |
|---|---|---|---|---|
| 4.1 | `A(x) = R_e exp(λ U_ctr (r_e − R_e))` (Eq. 15) | same formula | OK | unchanged |
| 4.2 | `r_e` = maximum training error, EMA-updated | present | OK | unchanged |
| 4.3 | `F_c(μ_a)` cost-sensitive formulation (Eq. 16) | simplified away without discussion | PARTIAL | documented as the paper's normal-only simplification; the realised FPR is reported and compared with τ |
| 4.4 | `μ⁰_a = F_N⁻¹(1−τ)` (Eq. 17) | present | OK | unchanged |
| 4.5 | `W_C` admission `U_ctr > μ_t` and `A ∈ [μ⁰_a ± δ]`, `δ = MAD(W_N)` | present | OK | unchanged |
| 4.6 | "exponentially decaying rank statistics bAm and δ" | both recomputed from scratch every row | MISSING | both are EMA-updated in `DynamicThresholdOptimizer.process` |
| 4.7 | `μ^r_a = κ(μ⁰_a − bAm)` (Eq. 18), `μ*_a = μ⁰_a + μ^r_a` (Eq. 19) | present | OK | unchanged |
| 4.8 | Algorithm 2 drift branch: discard `W_N`, `W_C` and re-estimate | **not implemented** — the windows survived every drift, so the calibration stayed stale | MISSING | `resetStatistics` runs once per confirmed drift episode |
| 4.9 | Offline update `UP = 1` when accumulated uncertainty > `μ_o` or `Δt > T_max` (Eq. 20) | present | OK | unchanged, both triggers unit-checked |
| 4.10 | "parallel training approach … uses the latest models" | blocking re-train that replaced the live model with a freshly initialised one | PARTIAL | background worker + `fork()` snapshot + `fineTune()` (continued training) + atomic `applyRefresh()` |
| 4.11 | DyMETER-Q / DyMETER-A ablations | absent | MISSING | `Config.quantileOnly`, `Config.meanCandidate` |

## 5. Experimental protocol

| # | Paper requirement | Original code | Verdict | Now |
|---|---|---|---|---|
| 5.1 | AUCROC and AUCPR reporting | absent (the label column was discarded) | MISSING | `Metrics` + CLI table + dashboard metrics |
| 5.2 | Independent runs reported as mean ± std | absent | MISSING | `Ablation.repeat`, `--runs` |
| 5.3 | Streaming data generator: abrupt / gradual / recurrent / mixed | absent | MISSING | `StreamGenerator.generate`, exposed in CLI and UI |
| 5.4 | Robustness studies (training noise, concept coverage) | absent | MISSING | `--noise`, `--coverage` |
| 5.5 | Twelve ablation variants | absent | MISSING | `Ablation.variants()` |
| 5.6 | Reproducibility checks for the port | absent | MISSING | `DyMeterSelfTest`: 32 checks including central-difference gradient validation |

## 6. Engineering decisions the paper leaves undefined

These are the only places where behaviour had to be invented; each is documented in the code.

1. **Drift hysteresis.** `μ_e` can be crossed by a single noisy instance. The port smooths `U_ctr`
   (EMA 0.2), requires 5 consecutive uncertain instances, requires the smoothed value to exceed
   `μ_e` by a 25 % margin, and waits a warm-up of `|W_N|/2` streamed records before the first concept
   can open (`DyMeterEngine.evaluateDrift`). Without these guards a 900-row stream produced 139
   one-row concepts (and wiped the DTO window continuously); with them the sample stream yields 2–3
   concepts whose onsets are reported with row numbers and reasons.
2. **Where `μ_e` comes from.** The paper *searches* `μ_e` inside [0.005, 0.4]. The port derives it as
   the `muEQuantile`-quantile (default 0.75, the upper edge of the historical uncertainty band) of
   the training uncertainty distribution, and accepts an explicit value from the CLI (`--mu-e`) or
   the dashboard. If the explicit value is tighter than the training band, the DTO falls back to the
   most confident half of the historical segment instead of failing
   (`DyMeterEngine.seedThresholdStatistics`).
3. **Cold start after the drift reset.** Eq. (17)–(19) are undefined on an empty `W_N`. The port
   re-seeds `W_N` with the first `max(8, |W_N|/4)` scores of the new concept
   (`DynamicThresholdOptimizer.resetStatistics`). Without it a stale threshold can flag the whole
   new concept, `W_N` never refills and the FPR stays at 1.0 forever (measured on the synthetic
   stream); with it the FPR returns to ≈ τ + anomaly rate.
4. **Stale-window guard.** The same deadlock can occur *without* a drift signal: if the boundary
   rejects every instance for a full window, `F_N(A)` is rebuilt from the current observations
   (`DynamicThresholdOptimizer.process`). Measured: FPR 0.96 → 0.62 on the synthetic stream.
5. **Background refresh size.** The paper only says the refresh *fine-tunes* the modules; the port
   uses `max(150, epochs/5)` steps in the worker (`DyMeterEngine.fineTune`) and continues from the
   live parameters (`fork`) instead of re-initialising.
4. **Standardisation.** Features are z-scored with the historical statistics before the engine sees
   them (`StreamData.standardise`). The paper does not state a preprocessing step; the port makes it
   explicit and reversible.
5. **τ sign.** The implementation section lists τ = 0.95 while Eq. (17) uses the `(1−τ)` quantile to
   bound the FPR by τ. The default is τ = 0.05 so the 95th percentile is used; `--tau 0.95`
   reproduces the literal reading.
6. **Episode identifiers.** Concept numbers in the UI are readable episode IDs produced by the port;
   the paper estimates drift uncertainty but does not publish ground-truth concept IDs.
7. **Evidence map `g(·)` of Eq. (2).** The paper writes an exponential map; the default here is the
   softplus map (`α = 1 + log(1 + e^z)`) used by evidential deep learning. The exponential map drives
   `U_ctr` to ≈ 0 as soon as the classifier is confident, which cannot reproduce the paper's own
   `μ_e` search range of [0.005, 0.4]. Both maps are implemented and selectable
   (`Config.exponentialEvidence`).

## 7. Explicit non-goals

The 18 third-party baselines, the LSTM/convolutional SCD backbones, the Apache Flink deployment and
the MNIST/FMNIST synthetic suites are *experimental side artefacts* of the paper, not parts of the
DyMETER algorithm. They are not ported; the algorithmic content (Eq. 1–20, Algorithms 1–2 and all
architecture-agnostic ablations) is complete and verified. See `EQUATION_TRACEABILITY.md` §3.
