# DyMETER — paper → code traceability

Paper: *Catching Every Ripple: Enhanced Anomaly Awareness via Dynamic Concept Adaptation*
(IEEE TPAMI, DOI 10.1109/TPAMI.2026.3682661). The extracted paper text used for this audit is kept in
`docs/paper_ripple.txt` (pages 1–16, produced from the PDF with `pypdf`).

Every equation, algorithm and module of the paper is mapped to the Java class and method that
implements it. Columns: **Paper** = equation / algorithm step, **Java** = implementation,
**Status** = verified how.

## 1. Modules

| Paper element | Java | Status |
|---|---|---|
| SCD: symmetric DNN autoencoder, `x → E_s(x) → z → D_s(z) → y` (Eq. 3), 2–10 layers | `AutoEncoder` (`forward`, `symmetricWidths`) | ✅ architecture built from the data, depth 2–10 |
| SCD loss `L_SCD = L_2(x,y) = Σ(x_i − y_i)²/n` (Eq. 4), `n` = input dimensionality | `AutoEncoder.reconstructionLoss`, `reconstructionGradient` | ✅ finite-difference verified |
| SCD trained with **Adam**, lr 1e-2 decayed ×0.96 over 2000 epochs | `Adam`, `DyMeterEngine.trainStaticDetector`, `learningRate` | ✅ paper schedule implemented |
| Latent width = #principal components for ≥70% explained variance | `Pca.analyse`, `DyMeterEngine.configure` | ✅ Jacobi eigen-decomposition, no external library |
| Pseudo-labelling rule (Eq. 5): `1` if `L_2 > μ_p` and `U_ctr ≤ μ_e`, `0` if `L_2 ≤ μ_p` and `U_ctr ≤ μ_e`, `-` otherwise | `DyMeterEngine.trainController`, `injectPriorLabels` | ✅ unknown samples never enter training |
| IEC: Dirichlet `Dir(p|α)`, `α = g(f(x))`, exponential non-negativity map (Eq. 2) | `EvidentialController.concentration` | ✅ `α = 1 + exp(z)`, clipped for stability |
| IEC predictive probability `p̂_c = α_c / Σ_k α_k` (Eq. 6) | `EvidentialController.predictiveProbability` | ✅ closed form checked |
| IEC focal evidential loss `(1 − p̂_c)^γ (log Σα − log α_c)` (Eq. 7), γ = 2 | `EvidentialController.focalLoss`, `backward` | ✅ gradient finite-difference verified |
| Concept uncertainty as Dirichlet mutual information (Eq. 8) with digamma `Φ(·)` | `EvidentialController.uncertainty`, `Numeric.digamma` | ✅ closed form for `α=(2,2)` checked |
| IEC implemented as a **two-layer DNN with ReLU** | `EvidentialController` (W1,b1,W2,b2 + ReLU) | ✅ |
| DSD hypernetwork features `e^(n) = L_layer^(n)(E_share(x))` (Eq. 9) | `ShiftHypernetwork.addShift` (shared encoder + per-layer linear layer) | ✅ |
| `W^(n) = (W1 e^(n) + b1) W2 + b2` (Eq. 10) | `ShiftHypernetwork.addShift` | ✅ two MLP layers |
| `K^(n) = W^(n) + b^(n)` (Eq. 11) | `ShiftHypernetwork.addShift` (`base[n]`) | ✅ additive branch of Fig. 2 |
| Dynamic parameter set `Θ_d = Θ_s + ΔΘ_d` (Eq. 12) | `DyMeterEngine.loadDynamicWeights` | ✅ `Δ = 0` at initialisation (self-test) |
| Dynamic reconstruction (Eq. 13) and `L_DSD = L_2(x, y′)` (Eq. 14) | `AutoEncoder.forward(x, dynamicWeights, null)`, `reconstructionLoss` | ✅ |
| DTO anomaly score `A(x) = R_e exp(λ U_ctr (r_e − R_e))` (Eq. 15) | `DynamicThresholdOptimizer.score` | ✅ closed form checked |
| `r_e` initialised with the maximum training error, updated by EMA | `DyMeterEngine.train` (`referenceError`), `DynamicThresholdOptimizer.process` | ✅ |
| Cost-sensitive thresholding `F_c(μ_a)` and its normal-only simplification (Eq. 16) | `DynamicThresholdOptimizer` (τ upper-bounds the FPR, `Metrics.Summary.falsePositiveRate` reports the realised FPR) | ✅ simplified form as in the paper |
| Empirical CDF `F_N(A) = |W_N|⁻¹ Σ 1(A_i ≤ A)` and `μ⁰_a = F_N⁻¹(1−τ)` (Eq. 17) | `DynamicThresholdOptimizer.seed/process`, `Numeric.quantile` | ✅ |
| Candidate window `W_C`: `U_ctr > μ_t` and `A ∈ [μ⁰_a − δ, μ⁰_a + δ]`, `δ = MAD(W_N)`, `μ_t = max training uncertainty` | `DynamicThresholdOptimizer.process`, `DyMeterEngine.muT` | ✅ |
| `μ^r_a = κ(μ⁰_a − bAm)` (Eq. 18) | `DynamicThresholdOptimizer.process` (`medianCandidate` is the EMA of the median of `W_C`) | ✅ closed form checked |
| `μ*_a = μ⁰_a + μ^r_a` (Eq. 19) | `DynamicThresholdOptimizer.process` | ✅ |
| Decision `A(x) > μ*_a`, `W_N ← W_N ∪ {A}` for normal samples | `DynamicThresholdOptimizer.process`, `DyMeterEngine.process` | ✅ |
| Offline updating `UP = 1` if `Σ 1(U_ctr > μ_e)·U_ctr > μ_o` or `Δt > T_max` (Eq. 20) | `DynamicThresholdOptimizer.process` (`accumulated`, `rowsSinceUpdate`), `DyMeterEngine.scheduleOfflineUpdate` | ✅ both triggers checked |
| "parallel training approach ... uses the latest models for OAD" | `DyMeterEngine.scheduleOfflineUpdate/fork/fineTune/applyRefresh` (single background worker, atomic swap) | ✅ online service never blocked |
| Algorithm 1 (training): SCD → pseudo-labels → IEC → DSD with gradients jointly through SCD and IEC | `DyMeterEngine.train/trainStaticDetector/trainController/trainDynamicDetector` | ✅ |
| Algorithm 2 (inference): `U_ctr` → escalate to DSD if `U_ctr > μ_e` → score → calibrate `μ*` → decide → `UP` | `DyMeterEngine.process` | ✅ |


## 2. Algorithm 2 drift branch, hyper-parameters and experimental protocol

| Paper element | Java | Status |
|---|---|---|
| Drift branch: discard stale `W_N`/`W_C`, re-estimate `mu0_a`/`mu^r_a` from the new concept | `DynamicThresholdOptimizer.process/resetStatistics` (rising edge of a confirmed episode) | ✅ verified by unit checks |
| Hyper-parameters tau = 0.95 (implementation section) while Eq. (17) uses the `(1−tau)` quantile | `Config.tau` default `0.05` = 1 − 0.95 so Eq. (17) selects the 95th percentile; `--tau 0.95` reproduces the literal reading | ⚠ documented ambiguity, both readings selectable |
| gamma = 2, lambda = 0.6, kappa = 0.8, `|W_N| = |W_C| = 64`, h_r = 0.2, 2000 epochs, lr 1e-2 | `Config` defaults | ✅ |
| `mu_p` in [0.05, 0.5], `mu_e` in [0.005, 0.4] (searched ranges) | `Config.pseudoProportion` (default 0.05) and `trainController` clipping `mu_e` into [0.005, 0.4] | ✅ |
| `mu_o` in [0.1, 1] x `|W_N|` | `Config.muO` default `32 = 0.5 * |W_N|` | ✅ |
| AUCROC / AUCPR evaluation | `Metrics.aucRoc`, `Metrics.aucPr` | ✅ verified on perfect/inverted/tied rankings |
| Five independent runs reported as mean ± std | `Ablation.repeat`, `--runs` | ✅ |
| Data generator for streaming data (abrupt, incremental/gradual, recurrent, mixed) | `StreamGenerator.generate` | ✅ |
| Robustness studies: noisy initialisation, insufficient concept coverage | `StreamGenerator.addNoise`, `StreamGenerator.subsampleTrainingRows` | ✅ |
| Sequential subsets p1..p5 (Table VII) | `StreamData.partition` | ✅ |
| Ablations DyMETER-S, -D, -SD, -SDT, w/o IEC, w/o DTO, -ce, -pl, -re, -Q, -A | `Ablation.variants()` | ✅ all 12 variants runnable from CLI and UI |
| Cost/time reporting (throughput, training time) | `Ablation.Outcome.trainingMillis/inferenceMillis`, table + UI | ✅ |
| Interpretability: drift points versus anomalies reported separately | `Result.drift/thresholdReset/offlineUpdate`, UI badges | ✅ |

## 3. Deliberately out of scope of this port (documented, not silently dropped)

| Paper element | Why it is not ported | Where it is discussed |
|---|---|---|
| 18 baseline detectors (LOF, IF, KNN, STORM, RRCF, MStream, MemStream, HS-Trees, iForestASD, RS-Hash, LODA, Kitsune, xStream, PIDForest, ARCUS, D3R, SARAD, METER) | third-party Python/PyTorch research artefacts; the port instead evaluates DyMETER against its own 12 ablations | gap analysis section 4 |
| SCD backbone variants DyMETER-lstm / -conv / -r-conv (Table VIII) | sequence backbones are not meaningful for the tabular streaming API of this port | gap analysis section 4 |
| Apache Flink integration (Fig. 8, "DyMETER in Action") | needs a cluster deployment; this port ships its own streaming API and dashboard | gap analysis section 4 |
| MNIST/FMNIST synthetic datasets (SynM/SynF, 784 dimensions) | image inputs; `StreamGenerator.synthetic` produces equivalent tabular drift streams | gap analysis section 4 |
