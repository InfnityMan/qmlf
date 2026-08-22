# Changelog

## 1.4.0 — 2026-08-21

Everything between v1.2.0 (`f6a79c8`) and this release, landed as one commit.
Grouped by impact.

### Fixed — crashes and nondeterminism

- **SIGSEGV on the documented import order.** `import torch` followed by
  `run_quantum_benchmark()` killed the interpreter with no traceback (duplicate
  vendored OpenMP runtimes, macOS/arm64, 3/3 reproducible). The existing
  `OMP_NUM_THREADS=1` mitigation moved from `integration/q_pipeline.py` — where
  it only protected the CLI — into `qmlf/__init__.py`, so it lands before any
  submodule import. A caller's own `OMP_NUM_THREADS` is respected.
- **QNN read-outs were shot-noise by default.** qiskit-machine-learning
  defaults `EstimatorQNN` to `default_precision=0.015625`; on a circuit whose
  exact expectation was 0.032, five identical forward passes spanned 0.037 and
  seeded training did not reproduce. All three QNN-backed layers (`qnn`,
  `hybrid`, `transformer`) now default to exact statevector read-out and take
  `precision=` to opt back into sampling (`precision=0.015625` restores the old
  behaviour bit-for-bit in distribution).
- **`QuantumPipeline.run()` left dropout active at inference** — `no_grad()`
  disables autograd, not dropout — so two `run()` calls on identical data
  disagreed. It now calls `.eval()` and restores the layer's previous mode.
- **`compute_vqe_energy()` gains `initial_point=`** so the VQE can be made
  deterministic; the default (random start) is unchanged.
- **`AdvancedQIGASelector` gains `random_state=`** for self-contained
  reproducibility, and `convergence_history` now resets per `fit()` instead of
  accumulating across refits.

### Fixed — silent wrong answers

- **Noise:** unknown `strategy` raises at construction (a mis-cased `"ZNE"` used
  to return the input unmitigated with no signal); `zne` on a 1D vector raises
  instead of polyfitting probabilities-as-noise-scales and returning 1.0; `zne`
  with one scale warns; uncalibrated `readout` warns; `fit()` on a non-square
  calibration matrix warns instead of silently storing nothing;
  `noise_level` is validated to `[0, 1)`.
- **Kernel:** `n_qubits` != data width raises a message naming both numbers,
  instead of surfacing Qiskit's "Mismatching number of values and parameters";
  checked in both `fit()` and the un-fitted ZZ path.
- **`run_quantum_benchmark(n_qubits=...)`** warns when the argument disagrees
  with the data width it actually uses (it was silently overridden before).
- **Dead parameters now tell the truth:** `AdvancedQIGASelector(n_features=...)`
  warns that it does not bound the subset size; `FederatedQML` warns when the
  reporting cohort differs from `num_clients`; chem warns when the geometry's
  atom count differs from `num_atoms`.

### Fixed — API contracts and error quality

- Torch layers validate rank and width up front with messages that name the
  expected shape (`qnn`, `hybrid`, `transformer`, `QMLFPipeline`) instead of
  dying inside a reshape or matmul.
- Graph kernel raises a clear error when `n_neighbors > n_samples`.
- `qmlf.plot_hilbert_space` accepts `show=` / `save_path=` (headless use);
  the `viz` module-level wrappers forward all keyword arguments instead of
  silently dropping them.

### Packaging

- `pyproject.toml`: `setuptools>=77` (the declared 61 could not parse the SPDX
  license and failed the build), `requires-python>=3.10` (qiskit's own floor),
  version classifiers, keywords, Source/Issues URLs.
- `requirements.txt`: upper bounds on every dependency; qiskit floor lowered to
  2.3 to match what the suite actually runs against.
- Version 1.4.0; the private Kaggle wheelhouse dataset was re-versioned to
  match, and the in-sandbox install probe verified the wheel end to end.

### Added — the simplification layer

- **`QuantumClassifier`** — the three-line quantum-kernel SVM. Sizes its
  circuit from the data, computes and wires the precomputed Gram matrices
  internally, and (by default) selects `bandwidth` with a deterministic,
  leak-free stratified-CV sweep, fixing the documented trap where the
  zero-knowledge path scored chance level (0.40 -> 0.70+ on the reference
  dataset with no arguments). sklearn-compatible (`clone`, `get_params`),
  multiclass, deterministic; sweep table exposed in `cv_results_`. Whitened
  modes default to the safe `normalize="maxabs"`.
- **Automatic qubit sizing for wide data**: `QuantumClassifier` PCA-reduces
  inputs wider than `max_qubits` (default 8) with a deterministic, per-fold
  leak-free projection, so the qubit count is no longer a user decision at
  any width; `reduction_`/`n_qubits_` report the outcome, `max_qubits=None`
  restores the old warn-and-attempt behaviour.
- **Encoding search**: `mode="auto"` / `feature_map="auto"` extend the CV
  sweep across raw/whitened preprocessing and entangled/unentangled maps, so
  the encoding itself is selected by data; winners land in `mode_` /
  `feature_map_`, the full table in `cv_results_`.
- **Quantum-advantage screen** (Huang et al. 2021, Nat. Commun.):
  `quantum_advantage_report(X, y)` / `geometric_difference` /
  `model_complexity` in the new `qmlf.analysis` package — trace-normalised
  geometric difference g, both model complexities, and a conservative
  verdict, from the Gram matrices alone. Sanity-pinned: g(K, K) = 1.
- **Nystrom kernel approximation**: `QuantumKernel.nystrom_features(X,
  n_landmarks=m)` (deterministic landmarks, rank-safe W^{-1/2}, exact at
  m = n) and `circuit_budget(n, m)` resource accounting — n(n-1)/2 pairwise
  fidelity circuits vs ~n*m for the approximate path.
- **`bandwidth="alignment"`**: kernel-target-alignment-maximising bandwidth
  selection on the kernel itself; classification-only, with clear errors for
  missing labels or regression targets; composes with kernel="projected".
- **`QuantumRegressor`**: kernel ridge with the full auto stack (auto-sizing,
  PCA reduction, deterministic CV over MSE or median bandwidth, projected
  kernels); rejects the classification-only options with clear errors.
  qiskit's QSVR ships none of this.
- **Projected quantum kernels** (Huang et al. 2021):
  `QuantumKernel(kernel="projected", gamma="auto")` — RBF over exact per-qubit
  reduced density matrices with train-fitted median-heuristic `gamma`; PSD,
  deterministic, batched. Measured 100x higher off-diagonal mass than the
  fidelity kernel at 8 qubits/bandwidth 1.0 (0.356 vs 0.004). Absent from
  qiskit-machine-learning (verified against 0.9 source).
- **Per-feature (ARD) bandwidth**: `bandwidth` accepts a positive vector,
  applied per feature after whitening — a per-direction metric unreachable by
  any external input rescaling; wrong lengths fail at first sight of data.
- **Median-heuristic bandwidth**: `bandwidth="median"` fits the angle scale
  from training pairwise distances in one deterministic pass; fitted value in
  `bandwidth_`; supported by `QuantumClassifier` (skips the CV sweep).
- **Fisher whitening**: `mode="fisher"` whitens by the class-size-weighted
  within-class scatter (activates the previously unused `y` in
  `QuantumKernel.fit`); clear errors without labels or with only singleton
  classes; safe `maxabs` normalisation by default in the classifier.
- **`QuantumClassifier(kernel=...)`**: `"projected"` end to end, or
  `"auto"` to add the kernel family as a fourth searched axis; winner in
  `kernel_`, table column in `cv_results_`. Post-fit concentration guard
  warns when the fitted kernel is degenerate.
- **Kernel health diagnostics**: `kernel_diagnostics(gram, y)` and
  `clf.diagnose()` — concentration verdict, eigenvalue spectrum share, and
  centered kernel-target alignment (Cortes et al.), with the scale-vs-
  direction complementarity documented from measurement.
- **`QuantumKernel(n_qubits=None)`** — the kernel now sizes itself from the
  first data it sees (or from a supplied `QuantumCircuit`); explicit
  `n_qubits` behaves exactly as before, and internals accessed before sizing
  raise a clear error instead of `None`-ing.

### Added

- `tests/`: determinism suite, input-validation suite, viz/CLI/nisq/graph/chem
  smoke-and-contract suites, golden-value regression tests that pin kernel
  numerics against dependency drift, and a claims suite (`test_simplification`)
  that pins the three-line path itself.
- `kaggle/`: verified benchmark task (`qmlf-quantum-kernel-tuning`), two
  zero-quota sandbox probes, reference sweep, wheelhouse metadata, and
  `SANDBOX.md` with measured sandbox facts.
