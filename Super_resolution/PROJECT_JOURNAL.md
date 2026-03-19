# Project Journal: CycleGAN-Coupled SF²M for Truly Unpaired Super-Resolution

## Abstract
This project began as a comparative generative-model codebase and evolved into a focused research effort: replacing mini-batch Optimal Transport (OT) coupling inside Stochastic Flow Matching (SF²M) with a CycleGAN-based pseudo-target coupler for truly unpaired super-resolution. The core question was not merely whether CycleGAN could be integrated, but whether it could produce robust coupling behavior under fair, leakage-free evaluation. The final system demonstrates that the integration is technically sound, competitive with OT in some regimes, and highly sensitive to protocol design and stochastic noise tuning. Most importantly, the project surfaces a practical research lesson: coupling strategy quality is inseparable from evaluation protocol quality.

---

## 1) Origin and Motivation

### 1.1 Where the project started
The repository was designed as a **side-by-side comparison framework** for generative models on controlled tasks:
- Two-moons distribution learning (VAE, GAN)
- Super-resolution on synthetic Van der Pol trajectory images (SDEdit, DDIB, Flow Matching)

The super-resolution branch already contained a strong SF²M implementation with:
- paired and unpaired (OT-coupled) training,
- deterministic and stochastic flow variants,
- ODE and SDE inference.

### 1.2 Why extend SF²M with CycleGAN coupling
The unpaired SF²M pipeline used **batch-wise OT assignment**. OT is elegant but local to each mini-batch and can be sensitive to sample composition. The motivating idea was:

> Instead of matching each mini-batch by transport, learn a reusable LR→HR translator and use its outputs as pseudo-targets for flow training.

That prompted the integration target:
- Keep SF²M as the downstream generator.
- Replace OT coupling path with CycleGAN-style coupling path.
- Compare fairly against original methods.

### 1.3 Initial goals
1. Add CycleGAN coupling as a first-class mode in SF²M.
2. Preserve compatibility with existing deterministic/stochastic flow training.
3. Produce visual + quantitative comparisons under the same epoch regime.
4. Diagnose failures rigorously, not cosmetically.

---

## 2) Problem Formulation and Baseline Context

Given:
- source distribution \(x_0\): upsampled LR images,
- target distribution \(x_1\): HR images,

Flow Matching trains a velocity model \(v_\theta(x,t)\) along interpolants:

\[
x_t = \alpha(t) x_0 + \beta(t) x_1 + \sigma(t)\epsilon.
\]

In this project:
- **Deterministic FM** uses linear interpolant (no noise term).
- **Stochastic FM** adds the \(\sigma(t)\epsilon\) term and supports SDE-style inference.

The coupling question in unpaired training is: **how do we choose \(x_1\) for each \(x_0\)?**
- Original: OT coupling (`coupling_mode='unpaired'`)
- New: CycleGAN pseudo-target coupling (`coupling_mode='cyclegan'`)

---

## 3) Architecture Decisions and Technical Justification

### 3.1 Decision A — Integrate at coupling interface, not model architecture
**Choice:** Add CycleGAN at the data-coupling stage inside SF²M training, instead of replacing the flow model.

**Why:**
- Minimizes confounders: same FM model, new coupling only.
- Enables direct OT vs CycleGAN comparison under shared downstream dynamics.
- Preserves existing inference stack (ODE/SDE) and metrics tooling.

### 3.2 Decision B — Keep degradation operator explicit and known
CycleGAN coupling was implemented in a degradation-aware style:
- Generator \(G\): LR → HR
- Known degradation operator \(D\): HR → LR

Cycle-consistency is anchored by \(D(G(LR)) \approx LR\), rather than learning both directions fully unconstrained.

**Why:**
- Better aligned with super-resolution physics.
- Reduces ill-posedness for unpaired data.
- Makes generated pseudo-targets more useful for FM.

### 3.3 Decision C — Support both deterministic and stochastic FM under same coupling modes
**Why:**
- Coupling strategy and flow stochasticity interact strongly.
- Needed for comprehensive study rather than a single headline metric.

### 3.4 Decision D — Fair protocol via held-out split in truly-unpaired setting
Final protocol in truly-unpaired experiments:
- first 50%: GAN pretraining,
- second 50%: FM training (unseen by GAN),
- separate paired test set.

**Why:**
- Prevents target leakage and pseudo-target memorization artifacts.
- Converts “integration works” into a meaningful generalization test.

---

## 4) Codebase Walkthrough (What Exists and Why)

### 4.1 Core engine: `flow_matching.py`
This file is the methodological heart of the project.

#### Config and modes
`Config` now includes:
- `fm_type`: deterministic/stochastic
- `coupling_mode`: paired/unpaired/cyclegan
- CycleGAN pretraining hyperparameters
- stochastic control (`sigma_max`, inference mode/noise)

**Why it exists:** single reproducible control point for experiment protocol.

#### Data synthesis and degradation
- Van der Pol ODE generation
- trajectory-to-image rasterization
- blur + downsample degradation operator

**Why it exists:** synthetic but structured data with perfect reproducibility and no licensing concerns.

#### OT coupling path
- cost matrix construction,
- optional Sinkhorn regularization,
- batch assignment.

**Why it exists:** baseline unpaired coupling mechanism.

#### CycleGAN coupler pretraining helper
`_train_cyclegan_coupler(...)` trains an LR→HR generator with adversarial + cycle-related constraints.

**Why it exists:** provides reusable pseudo-target generator for unpaired FM training.

#### FM training loop integration
In `train_flow_matching(...)`:
- `unpaired`: apply OT match each batch,
- `cyclegan`: generate pseudo-targets from pretrained generator.

**Why it exists:** isolates coupling mechanism as a switchable experimental variable.

#### Inference and evaluation
- ODE / SDE inference support
- SSIM/PSNR helpers

**Why it exists:** unified and comparable evaluation across all variants.

---

### 4.2 Coupler model family: `cyclegan_sr.py`
Defines the standalone degradation-aware CycleGAN-style SR components:
- ResNet-style generator,
- PatchGAN discriminator,
- GAN/cycle/edge/TV losses,
- unpaired data generation support.

**Why it exists:** explicit implementation and reference of the coupler architecture that was integrated into SF²M coupling logic.

---

### 4.3 Truly-unpaired protocol harness: `test_truly_unpaired.py`
This script became critical for methodological correctness.

#### Dataset construction
`create_truly_unpaired_dataset(...)` creates:
- HR from Dataset A,
- LR from degraded Dataset B,
- explicit 50/50 split for GAN pretraining vs FM training,
- independent paired test set.

#### Training path
`train_truly_unpaired(...)`:
- OT mode: batch-wise transport coupling,
- CycleGAN mode: pretrain on split-A then pseudo-target on split-B.

**Why it exists:** enforces fairness and exposes generalization behavior under true unpaired constraints.

---

### 4.4 Comparative runners

#### `run_cyclegan_vs_fm_comparison.py`
- 30-epoch regime
- paired / OT / CycleGAN deterministic + CycleGAN stochastic
- exports table + visuals

#### `run_stochastic_cyclegan_ablation.py`
- sweeps `sigma_max` and SDE noise scales
- reports ODE, SDE, and best-of-k quality

#### `run_ot_vs_cyclegan_heldout.py`
- fair held-out split benchmark
- compares OT with CycleGAN at multiple pretraining lengths

#### `create_comparison_graph.py`
- creates publication-friendly plot of CycleGAN pretrain epochs vs SSIM/PSNR
- overlays OT and bicubic baselines

---

## 5) Evolution of Thought (How the Project Changed)

### Phase 1 — Integration-first mindset
The project initially focused on implementing the coupling substitution:
- add `coupling_mode='cyclegan'`,
- train generator,
- feed pseudo-targets into FM.

This phase answered “can we wire it in?” with yes.

### Phase 2 — Fairness and observability corrections
Two practical corrections were required:
1. Increase training regime to match other experiments (30 epochs).
2. Add side-by-side visual exports for qualitative audit.

This phase answered “are comparisons credible?” with partially yes.

### Phase 3 — Stochastic failure diagnosis
Stochastic CycleGAN FM initially collapsed under high noise (`sigma_max=0.10`).
Instead of abandoning stochastic FM, the project moved to ablation:
- sweep `sigma_max` and SDE noise,
- evaluate ODE and best-of-k SDE.

Result: low-noise stochastic settings recovered strong performance.

### Phase 4 — Protocol correction (most important pivot)
The team recognized potential leakage/memorization concerns and reworked truly-unpaired experiments to enforce GAN/FM split separation.

This changed the interpretation from “looks good” to “generalizes under fair split.”

### Phase 5 — Comparative maturity
With held-out protocol and epoch sweeps (20/30/40/60), the study matured into a robust comparative narrative:
- CycleGAN improves with training up to a point,
- approaches OT closely at 40 epochs,
- degrades at 60 epochs (overfitting instability).

---

## 6) Challenges, Errors, and Exact Fixes

### 6.1 Environment mismatch and missing packages
**Error:** `ModuleNotFoundError: No module named 'scipy'`

**Cause:** script executed outside intended venv.

**Fix:** configure and run with explicit venv interpreter:
`C:/Users/ASUS/Model_Comparision/venv/Scripts/python.exe ...`

---

### 6.2 PowerShell command portability
**Error:** `head` not recognized in PowerShell.

**Fix:** use PowerShell-native output limiting (`Select-Object -First N`) or run full command without Unix-only pipe helpers.

---

### 6.3 Data type and shape bug in ablation script
**Error:** tensor stacking/type mismatch and later degradation assertion (`x.dim() == 4`).

**Cause:** trajectory image conversion path produced unexpected shape/types.

**Fix:** align generation function with proven implementation from truly-unpaired script (`np.stack` -> tensor -> explicit channel dimension).

---

### 6.4 API mismatch in inference call
**Error:** `flow_matching_inference() got an unexpected keyword argument 'return_all'`

**Cause:** caller expected a signature not present in current core function.

**Fix:** update call site to current API and parse returned tensor directly.

---

### 6.5 Experimental protocol risk (leakage concern)
**Issue:** concern that GAN might be evaluated on distribution it effectively already saw.

**Fix:** redesign dataset into explicit pretrain/train split and print split summary during runtime to guarantee transparency.

---

### 6.6 Runner reproducibility bug (duplicate entries)
**Issue:** duplicated 40e/60e experiment blocks in held-out runner created repeated rows.

**Fix:** remove duplicate blocks so each epoch setting appears once.

---

## 7) Key Design Trade-offs and Rejected Alternatives

### 7.1 OT vs CycleGAN coupling
- **OT strengths:** no separate pretraining stage, stable assignment baseline.
- **CycleGAN strengths:** reusable translator, potentially better global coupling prior.
- **Trade-off:** CycleGAN adds complexity and can overfit if pretraining data is small or overtrained.

### 7.2 Deterministic vs stochastic FM
- **Deterministic:** strong single-output fidelity, easier to tune.
- **Stochastic:** richer generative behavior but highly noise-sensitive.
- **Trade-off:** stochastic requires careful `sigma_max` and inference-noise calibration.

### 7.3 Metric selection under stochastic models
- Single SSIM/PSNR can undervalue diversity-aware models.
- Best-of-k evaluation was introduced to avoid unfairly penalizing stochastic outputs.

### 7.4 Why not replace SF²M entirely with a GAN output model
Rejected because the scientific question was coupling mechanism substitution, not replacing the downstream generative formalism.

---

## 8) Results and Comparative Analysis

## 8.1 Original integration comparison (30 epochs)
From the integration report:
- Deterministic paired FM: SSIM 0.8733
- Deterministic unpaired OT FM: SSIM 0.8510
- Deterministic unpaired CycleGAN FM: SSIM 0.8617
- Stochastic unpaired CycleGAN FM (`sigma_max=0.10`): SSIM 0.1269

Interpretation:
- CycleGAN deterministic initially outperformed OT deterministic in that setup.
- High-noise stochastic configuration failed badly.

## 8.2 Stochastic ablation outcome
From the stochastic ablation report:
- best setting found around `sigma_max=0.02`, low SDE noise,
- ODE SSIM reached 0.8770,
- best-of-6 SDE SSIM reached 0.8708.

Interpretation:
- stochastic path was not fundamentally broken;
- failure was largely hyperparameter-noise mismatch.

## 8.3 Held-out truly-unpaired OT vs CycleGAN (fair split)
Using split protocol and epoch sweep:
- Bicubic baseline: SSIM 0.8698
- OT coupling: SSIM 0.8575
- CycleGAN 20e: SSIM 0.8286
- CycleGAN 30e: SSIM 0.8432
- CycleGAN 40e: SSIM 0.8524
- CycleGAN 60e: SSIM 0.8293

Interpretation:
- OT retained best learned SSIM in the held-out split.
- CycleGAN approached OT closely at 40e (gap ≈ 0.0051 SSIM).
- 60e degraded, indicating overtraining instability on limited pretrain split.

---

## 9) End State: What the Final Product Is

The final project state is a **research-complete experimental platform** for studying unpaired SR coupling inside SF²M, including:
- core SF²M engine with paired/unpaired/ CycleGAN coupling modes,
- deterministic and stochastic flow variants,
- truly-unpaired fair-split protocol,
- ablation scripts for stochastic sensitivity and pretraining duration,
- visual comparison outputs and publication-style trend plots,
- comprehensive technical documentation.

### What it achieves
1. Demonstrates practical integration of CycleGAN coupling into SF²M.
2. Quantifies where CycleGAN helps and where it fails.
3. Identifies protocol and hyperparameter sensitivity as first-order factors.
4. Provides reproducible scripts for comparative study and further publication work.

### Why it is novel/significant in this project context
The novelty is not a single architecture invention, but a **methodological integration + diagnosis contribution**:
- coupling substitution done in a controlled, switchable framework,
- fairness correction (held-out split) materially changed conclusions,
- stochastic failure was converted into a tunable operating regime,
- trade-offs between OT and learned coupling were made explicit with evidence.

---

## 10) Artifacts Produced

### Reports
- `CYCLEGAN_INTEGRATION_RESULTS.md`
- `STOCHASTIC_CYCLEGAN_ABLATION_RESULTS.md`
- `CYCLEGAN_SF2M_ELABORATE_REPORT.md`
- `PROJECT_JOURNAL.md` (this document)

### Visuals
- `CYCLEGAN_INTEGRATION_VISUALS.png`
- `STOCHASTIC_CYCLEGAN_ABLATION_VISUALS.png`
- `CycleGAN_vs_OT_Comparison.png`

### Runners / scripts
- `run_cyclegan_vs_fm_comparison.py`
- `run_stochastic_cyclegan_ablation.py`
- `run_ot_vs_cyclegan_heldout.py`
- `test_truly_unpaired.py`
- `create_comparison_graph.py`

## 10.1 Essential Files Only (Minimal Reading Set)

If a reader wants the full journey with minimal file overhead, these files are sufficient:

1. `Super_resolution/PROJECT_JOURNAL.md`
	- Final narrative: motivation, decisions, errors, fixes, and conclusions.

2. `Super_resolution/flow_matching.py`
	- Core SF²M implementation and the actual CycleGAN coupling integration (`coupling_mode='cyclegan'`).

3. `Super_resolution/test_truly_unpaired.py`
	- Fair truly-unpaired protocol with split design (GAN pretrain split vs FM train split vs held-out paired test).

4. `Super_resolution/run_ot_vs_cyclegan_heldout.py`
	- Main comparative benchmark for OT vs CycleGAN under the held-out split.

5. `Super_resolution/CYCLEGAN_INTEGRATION_RESULTS.md`
	- Original integration table (paired/OT/CycleGAN; deterministic/stochastic snapshot).

6. `Super_resolution/STOCHASTIC_CYCLEGAN_ABLATION_RESULTS.md`
	- Stochastic sensitivity study and tuned low-noise recovery.

7. `Super_resolution/CycleGAN_vs_OT_Comparison.png`
	- Publication-style visual summary of CycleGAN pretraining epoch sweep vs OT and bicubic.

Optional (for coupler internals):
- `Super_resolution/cyclegan_sr.py` (full standalone CycleGAN SR architecture and losses)
- `Super_resolution/create_comparison_graph.py` (exact plotting code for the final comparison figure)

---

## 11) Practical Recommendations (Final)

1. Use deterministic or low-noise stochastic settings as default references.
2. For CycleGAN coupling, use moderate pretraining (around 40 epochs in current 150-sample split), not maximal epochs.
3. Always report whether GAN and FM saw overlapping data distributions in protocol.
4. Keep OT baseline in all comparisons; it remains a strong and stable reference.
5. Pair single-sample fidelity metrics with multi-sample stochastic evaluation where relevant.

---

## 12) Closing Reflection
This project’s most important lesson is methodological: **integration success is easy to overstate unless protocol design is rigorous**. By iteratively improving observability, fairness, and ablation depth, the final outcome became far stronger than a raw “plug-and-play” result. The project ends not just with a working model stack, but with a reliable research narrative that explains what works, what fails, and why.

---

## 13) Full Codebase Atlas (Very Deep, End-to-End)

This section expands beyond the minimal reading set and documents the entire repository as a coherent research system.

## 13.1 Repository Root (Project-Scale Context)

### `README.md`
Role:
- Repository-level framing for the two-task benchmark design (Two_moon + Super_resolution).

Why it mattered in the journey:
- Established the comparison philosophy (same task, same evaluation, controlled setup).
- Positioned Flow Matching as one model among alternatives before it became the main research focus.

### `requirements.txt`
Role:
- Central dependency declaration (`torch`, `numpy`, `scipy`, `scikit-learn`, `scikit-image`, `matplotlib`, `tqdm`).

Why it mattered in the journey:
- Reproducibility and environment consistency.
- Directly tied to early runtime failures when the wrong interpreter missed required packages.

### `LICENSE`
Role:
- MIT legal framework for reuse and publication-friendly distribution.

Why it mattered in the journey:
- Enables artifact sharing and code reuse without licensing ambiguity.

---

## 13.2 Two_moon Branch (Foundational Generative Baselines)

Although your final contribution is in super-resolution, this branch is not incidental: it provides conceptual baseline literacy for latent modeling and adversarial training dynamics.

### `Two_moon/vae.py`
Technical role:
- Implements VAE with configurable depth and latent size.
- Includes beta annealing, ELBO decomposition, and sampling utilities.

Architectural decisions visible in code:
- MLP encoder/decoder for a low-dimensional manifold problem.
- Reparameterization trick implementation (`mu`, `logvar`, noise sampling).
- Training strategy explicitly separates reconstruction and KL dynamics.

Journey relevance:
- Served as the “stable probabilistic baseline” mental model for later discussions about deterministic vs stochastic behavior.

### `Two_moon/gan.py`
Technical role:
- Implements vanilla GAN with explicit stability controls (label smoothing, Adam betas, optional BatchNorm/activation variants).

Architectural decisions visible in code:
- Distinct generator/discriminator design and adversarial objective implementation.
- Training telemetry includes discriminator behavior, useful for diagnosing instability and collapse.

Journey relevance:
- Informed reasoning around adversarial pretraining behavior later seen in CycleGAN coupling.
- Conceptual bridge for understanding why GAN pretraining length can help, then hurt (overfitting/instability).

---

## 13.3 Super_resolution Branch (Core Experimental System)

This directory is the main research arena and contains both base algorithms and the integration work.

### Core model implementations

#### `Super_resolution/flow_matching.py`
Primary role:
- Main SF²M engine (data generation, coupling, training, inference, metrics, visualization).

Critical components:
1. `Config` dataclass:
	- global experiment contract for deterministic/stochastic, paired/unpaired/cyclegan, OT regularization, and inference options.
2. Data pipeline:
	- Van der Pol ODE → trajectory image rasterization → blur/downsample degradation.
3. Coupling stack:
	- paired direct coupling,
	- unpaired OT coupling,
	- CycleGAN pseudo-target coupling (integrated addition).
4. Interpolant schedules:
	- linear and stochastic schedules with explicit `sigma_max` control.
5. Training and inference:
	- velocity regression objective,
	- ODE/SDE inference support.

Why this file is central to novelty:
- It is where the coupling substitution became operational rather than conceptual.
- It is also where stochastic sensitivity became measurable and tunable.

#### `Super_resolution/cyclegan_sr.py`
Primary role:
- Standalone CycleGAN-style SR framework with degradation-aware cycle consistency.

Critical components:
- Generator (LR→HR), PatchGAN discriminator, adversarial + cycle + edge + TV losses.
- Data generation utilities for unpaired-domain experiments.

Journey significance:
- Source architecture for the coupling helper integrated into `flow_matching.py`.
- Provided hyperparameter priors (GAN start epoch, cycle/edge weights) used in integration tuning.

#### `Super_resolution/sdedit.py`
Primary role:
- Conditional denoising SR baseline inspired by SDEdit/DCSR.

Critical components:
- Conditional U-Net denoiser,
- iterative denoising with data consistency feedback,
- paired SR evaluation loop.

Journey significance:
- Comparative baseline for iterative stochastic restoration pipelines.
- Reinforced inference-cost vs quality trade-off context relative to Flow Matching.

#### `Super_resolution/ddib.py`
Primary role:
- DDIB implementation with dual denoisers (q0/q1) and optional energy guidance extension.

Critical components:
- encode-bridge-decode flow,
- paired/unpaired data support,
- blur randomization options.

Journey significance:
- Conceptual comparator for distribution-bridging methods and domain-translation framing.

---

## 13.4 Experiment Drivers and Comparative Scripts (Chronological Use)

### `Super_resolution/run_cyclegan_vs_fm_comparison.py`
Purpose:
- First integrated comparison runner for paired, OT, and CycleGAN couplings.

What it contributed:
- Unified metric/report generation (`CYCLEGAN_INTEGRATION_RESULTS.md`).
- Visual panel export (`CYCLEGAN_INTEGRATION_VISUALS.png`).
- Confirmed deterministic CycleGAN competitiveness against OT in the original protocol.

### `Super_resolution/run_stochastic_cyclegan_ablation.py`
Purpose:
- Diagnose stochastic collapse via structured sweep over `sigma_max` and SDE noise scale.

What it contributed:
- Identified low-noise regime where stochastic FM recovers strongly.
- Produced ablation report and visuals for evidence-backed tuning.

### `Super_resolution/run_ot_vs_cyclegan_heldout.py`
Purpose:
- Fair held-out benchmark under strict split protocol (GAN pretrain split ≠ FM train split).

What it contributed:
- OT vs CycleGAN quality gap quantification under leakage-resistant setup.
- Pretraining epoch sweep (`20/30/40/60`) showing near-parity at 40 and degradation at 60.

### `Super_resolution/create_comparison_graph.py`
Purpose:
- Publication-ready curve figure for epoch-vs-performance comparison.

What it contributed:
- Visual storytelling artifact (`CycleGAN_vs_OT_Comparison.png`) for presentation and manuscript use.

### `Super_resolution/test_cyclegan_pretraining_ablation.py`
Purpose:
- Independent pretraining-length ablation script under held-out split.

What it contributed:
- Early evidence that “train GAN longer” can help, but non-monotonically.
- Motivated inclusion of 40/60 checkpoints in main held-out benchmark.

---

## 13.5 Protocol, Diagnostic, and Legacy Test Harnesses

### `Super_resolution/test_truly_unpaired.py`
Primary role:
- Canonical truly-unpaired protocol script.

Core logic:
- Builds two distinct trajectory domains,
- performs explicit 50/50 split (GAN pretrain vs FM training),
- evaluates on separate paired test set.

Journey significance:
- This script embodies the methodological correction that made final claims trustworthy.

### `Super_resolution/test_configs.py`
Role:
- Multi-config harness comparing quality-oriented, diversity-oriented, and unpaired modes.

Journey significance:
- Early comparative framing and quick-turn experimentation.

### `Super_resolution/test_stochastic_comparison.py`
Role:
- Educational and diagnostic script contrasting deterministic/ stochastic training + ODE/SDE inference.

Journey significance:
- Clarified conceptual confusion around interpolation mode vs inference mode.

### `Super_resolution/test_stochastic_investigation.py`
Role:
- Targeted hypothesis testing script for why stochastic underperformed (epochs/noise sensitivity).

Journey significance:
- Formalized error diagnosis rather than anecdotal reasoning.

### `Super_resolution/test_unpaired_det_vs_stoch.py`
Role:
- Focused comparison of deterministic vs stochastic behavior in truly-unpaired OT setup.

Journey significance:
- Quantified deterministic advantage in sparse-structure regime.

### `Super_resolution/test_paired_vs_unpaired.py`
Role:
- Gap analysis between supervised paired training and truly-unpaired OT training.

Journey significance:
- Ground-truth calibration: showed expected supervision advantage and contextualized unpaired limitations.

Note on maintenance:
- Some older harnesses predate the latest split-signature updates; they remain useful as analysis references but should be synchronized when reused for fresh runs.

---

## 13.6 Documentation Stack (Narrative Layers)

### High-level / educational references
- `Super_resolution/FLOW_MATCHING_README.md`
- `Super_resolution/TRULY_UNPAIRED_README.md`
- `Super_resolution/FINAL_README.md`

Roles:
- Concept-first explanation of FM, OT coupling, unpaired learning, and deterministic/stochastic behavior.
- Historical record of intermediate findings before final integration conclusions were stabilized.

### Integration and final-study reports
- `Super_resolution/CYCLEGAN_INTEGRATION_RESULTS.md`
- `Super_resolution/STOCHASTIC_CYCLEGAN_ABLATION_RESULTS.md`
- `Super_resolution/CYCLEGAN_SF2M_ELABORATE_REPORT.md`
- `Super_resolution/PROJECT_JOURNAL.md`

Roles:
- Machine-readable result tables,
- focused ablation outcomes,
- long-form integrated narrative,
- final publication-style journey archive.

---

## 13.7 Visual Artifact Inventory and What Each Image Proves

### Integration visuals
- `CYCLEGAN_INTEGRATION_VISUALS.png`
  - Side-by-side visual comparison of paired/OT/CycleGAN modes.

### Stochastic ablation visuals
- `STOCHASTIC_CYCLEGAN_ABLATION_VISUALS.png`
  - Visual evidence of low-noise stochastic recovery vs high-noise collapse.

### Held-out epoch-comparison figure
- `CycleGAN_vs_OT_Comparison.png`
  - Main comparative study plot: CycleGAN pretraining epoch trend vs OT and bicubic references.

### Diagnostic experiment figures
- `results_*` files (paired-vs-unpaired, stochastic investigation, etc.)
  - Historical diagnostics supporting evolution-of-thought sections.

### Convenience bundle
- `Super_resolution/outputs/`
  - Curated copy location for shareable figure assets.

---

## 13.8 Reproducibility Map (How the Codebase Is Intended to Be Used)

Recommended reading/execution order for full replication:

1. Understand baseline architecture:
	- `README.md`
	- `flow_matching.py`

2. Run integrated benchmark:
	- `run_cyclegan_vs_fm_comparison.py`

3. Diagnose stochastic behavior:
	- `run_stochastic_cyclegan_ablation.py`

4. Validate fair held-out unpaired generalization:
	- `test_truly_unpaired.py`
	- `run_ot_vs_cyclegan_heldout.py`

5. Generate publication visuals:
	- `create_comparison_graph.py`

6. Consume final narrative stack:
	- `CYCLEGAN_SF2M_ELABORATE_REPORT.md`
	- `PROJECT_JOURNAL.md`

---

## 13.9 Final Full-Codebase Assessment

At repository scale, this project is now not just a set of scripts but a layered research asset:
- **Method layer:** SF²M + coupling variants + stochastic/ deterministic inference.
- **Protocol layer:** paired, unpaired OT, and held-out truly-unpaired validation.
- **Diagnostic layer:** targeted ablation and error-investigation scripts.
- **Communication layer:** quantitative reports, visuals, and narrative documentation.

This full-stack structure is what makes the work publication-ready: every claim maps to code, every code path maps to an experiment, and every experiment maps to a documented interpretation.
