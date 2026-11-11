# Changelog

## 1.5.0 (2026-11-11)

I measured every number in this section myself and quote each one with its
workload, baseline and method; none of it comes from anywhere else. The
timings were measured on Kaggle's standard notebooks. All of them used `OMP_NUM_THREADS=1`, which is qmlf's import default.

- **CPU:** a Kaggle CPU session on an AMD EPYC 7B12 (31 GB RAM) with Ubuntu
  24.04, Python 3.13.18, numpy 2.1.3 (OpenBLAS), qiskit 2.6.2, torch 2.11.0,
  sympy 1.14.0 and scikit-learn 1.6.1. Kaggle hands out hardware per session,
  and a later CPU session got an Intel Xeon @ 2.20GHz instead.
- **GPU:** a Kaggle "GPU T4 x2" session, meaning two NVIDIA Tesla T4 cards
  (15 GB each) on an Intel Xeon @ 2.00GHz host, with the same software and
  torch 2.11.0+cu130. `simulator="torch"` uses one of the two GPUs (`cuda`,
  i.e. `cuda:0`). Results get copied back to the CPU inside the timed region,
  which synchronises the device.

The test suite passed on a Kaggle CPU session (the Intel Xeon) and on the T4
machine: 299 passed both times, and the 2 skipped tests only run on Apple's
MPS backend. I haven't run it on Windows.

Kaggle has no Apple hardware, so this release doesn't quote any numbers for
`device="mps"`. The torch simulator still supports it, in complex64 because
MPS has no 64-bit floats, and clips single-precision Gram entries to [0, 1].

### Added: fused simulators (opt-in, `simulator=`)

`QuantumKernel`, `QuantumClassifier`, `QuantumRegressor`,
`run_quantum_benchmark` and the CLI (`qmlf benchmark|visualize --simulator`)
now take `simulator="qiskit" | "fused" | "torch"`. The default hasn't changed:
it's `"qiskit"`, bit for bit what every published number was computed with.
`"fused"` is plain numpy. `"torch"` picks CUDA, then Apple MPS, then CPU; on
the CPU-only Kaggle machine it ran on the CPU and matched qiskit's bandwidth
and predictions. `sympy` is a declared dependency now (torch installs it
anyway). If it's missing, the fused simulators fall back to qiskit with a
warning; I checked that by running without sympy.

`qmlf.ops.fused` lowers an encoding circuit once into whole-batch array
operations and runs every sample through it together. Qiskit simulates one
sample at a time, gate by gate, and that per-sample work took 99.4-99.8% of a
default 200-sample Gram matrix at 4, 8 and 12 qubits. Here's how the fused
version avoids it:

- A stretch of diagonal gates (`p`, `rz`, `cz`, `cp`, `rzz`, plus the
  `cx . p . cx` sandwiches a ZZ map is made of) becomes one phase multiply.
  `cx` and `swap` inside such a stretch are tracked as a linear map over GF(2),
  so the phase lands on the right amplitudes without moving any of them. When
  there are many distinct phase terms (a fully entangled ZZ map has
  q + q(q-1)/2), the phase comes from a Walsh-Hadamard transform of the term
  coefficients, which takes q passes over the amplitudes rather than one per
  term.
- Every other single-qubit gate is a butterfly along one qubit's axis. A
  Hadamard layer runs as an unnormalised Walsh-Hadamard transform with one
  exact power-of-two scale, and the first layer on |0...0> gets skipped
  outright because it's the uniform state.
- The batch lives amplitude-major, `(2**n, n_samples)`, with samples as the
  fastest-varying index. Sample-major, the butterflies on qubits 0-2 had
  worked on contiguous runs of just 1, 2 and 4 values. I timed it against a
  copy of the module that differs only in layout (1000 samples, 200 at 14
  qubits, median of 5, 4-14 qubits). On Kaggle's CPU the switch made numpy
  1.4-1.6x faster and torch CPU 1.3-1.5x, the T4 got 1.3-1.6x, and the output
  stayed bit-identical in all 110 circuit/backend combinations I tried (44 on
  the CPU machine, 66 on the GPU machine).
- Rows run in 4 MB chunks, which bounds the temporaries. Swept at 4, 8 and 12
  qubits, 4 MB came out between 16% faster and 3% slower than running the
  whole batch at once; 256 KB chunks were up to 14% slower.
- Each circuit gets lowered once per process and shared by every kernel built
  around it, keyed by the same structural digest as the state cache. That
  matters because lowering a 12-qubit ZZ map (156 parametrised gates) took
  0.10 s, two-thirds of the 0.15 s it takes to simulate 200 samples through
  it, and a default `QuantumClassifier().fit` builds 22 kernels around one
  circuit (I counted).
- Fused states go through the state cache too, under a key that also names
  the simulator. A CV sweep or `predict()` doesn't re-simulate rows it already
  has, and the default path never gets handed a fused state (tested).
- Circuits it can't handle warn once per circuit and fall back to qiskit
  instead of raising (tested). That covers anything outside the 25 supported
  gates, a lookalike gate that merely shares a library gate's name, an angle
  expression numpy can't evaluate, and any instruction that can't be
  identified exactly.

The first table times one Gram matrix of 200 samples (ZZ map, full
entanglement, 2 reps) against `simulator="qiskit"` on the same machine. I
cleared the state cache before every timed call so that every row really got
simulated, with the circuit already lowered, and took the median of 5 calls
after one untimed warm-up (qiskit got a single call). Kaggle CPU:

| qubits | qiskit | `fused` (numpy, complex128) | `torch` CPU (complex128) |
|---|---|---|---|
| 4  | 0.357 s | 0.0017 s (206x) | 0.0022 s (163x) |
| 6  | 0.766 s | 0.0036 s (214x) | 0.0035 s (219x) |
| 8  | 1.433 s | 0.0115 s (124x) | 0.0106 s (135x) |
| 10 | 2.541 s | 0.0472 s (54x)  | 0.0386 s (66x)  |
| 12 | 5.316 s | 0.167 s (32x)   | 0.119 s (45x)   |

Kaggle T4 x2, with qiskit and `fused` on the host CPU and `torch` on one T4:

| qubits | qiskit | `fused` (host CPU) | `torch` CUDA (complex128) |
|---|---|---|---|
| 4  | 0.527 s | 0.0027 s (194x) | 0.0033 s (159x) |
| 6  | 1.367 s | 0.0052 s (263x) | 0.0052 s (260x) |
| 8  | 2.510 s | 0.0151 s (166x) | 0.0091 s (277x) |
| 10 | 4.465 s | 0.0709 s (63x)  | 0.0243 s (184x) |
| 12 | 8.672 s | 0.258 s (34x)   | 0.099 s (87x)   |

Nobody runs a single Gram matrix, though, so I also timed all of
`QuantumClassifier(max_qubits=None).fit(X, y)`: the default bandwidth sweep
(7 candidates x 3 folds) on 200 samples, in one fresh process per simulator.
"First" is the first fit in that process. It includes importing sympy, and
torch where it applies (the import alone took 1.1 s on the CPU machine and
1.6 s on the GPU machine), plus lowering the circuit. "Later" is the mean of
two refits with the state cache cleared. Speedups compare against qiskit's fit
on the same machine:

| machine, qubits | qiskit | `fused` first / later | `torch` first / later |
|---|---|---|---|
| CPU, 8  | 10.1 s | 0.53 s (19x) / 0.20 s (50x) | CPU: 1.77 s (5.7x) / 0.21 s (48x) |
| CPU, 12 | 37.8 s | 2.11 s (18x) / 1.63 s (23x) | CPU: 3.16 s (12x) / 1.36 s (28x) |
| T4, 8   | 18.2 s | 0.71 s (26x) / 0.28 s (64x) | CUDA: 3.23 s (5.6x) / 0.27 s (67x) |
| T4, 12  | 61.7 s | 3.07 s (20x) / 2.44 s (25x) | CUDA: 4.19 s (15x) / 1.30 s (48x) |

After those fits, a single `predict` on 50 new rows ran 79x (8 qubits) and 26x
(12 qubits) faster with `fused` on the CPU machine.

**Correctness.** In the Gram runs the largest difference from qiskit was
1.2e-14 on the CPU machine and 1.5e-14 on CUDA, complex128 throughout. The
projected kernel on CUDA came within 2.0e-14, and CUDA states sat within
5.6e-17 of the numpy simulator's. Amplitudes match qiskit's `Statevector` to
3.4e-14 or better over 177 circuits with angles in [-pi, pi]: ZZ maps at 2-8
qubits with reps 1-3 and six entanglement patterns, Z maps,
`real_amplitudes`, four Pauli maps and two mixed-gate circuits. For
unnormalised features of size ~30 the gap grows to 7.7e-14. One test checks
all 25 supported gates in a single circuit against qiskit to 1e-12. Every
classifier fit above, on both machines, chose the same bandwidth, produced the
same CV table and made the same predictions as qiskit.

That's agreement on these workloads, and it isn't a guarantee for every
dataset: an SVM decision near a tie doesn't have to follow kernel values that
agree to 1e-14, and a near-tied CV sweep can pick a different bandwidth. That's
why `"qiskit"` stays the default.

**Batch invariance.** A row's state comes out as the same bits whether it's
simulated alone or among others. A test pins that, because the classifier
computes the same rows inside different batches (CV folds, fit, predict), and
on the T4 I checked that 300 rows simulated alone and together were
bit-identical as well. Getting there meant giving up complex dtypes. numpy's
complex multiply fuses one of its products, so `x * y` and `y * x` differed in
the imaginary part's last bit for 32.9% of 100,000 random pairs on both
Kaggle machines. numpy's temporary elision can also reuse a temporary operand
of 256 KiB or more as the output, which puts it first and can swap the order
once a batch gets big enough. Kaggle's numpy 2.1.3 on Python 3.13 never did
that in my tests, but it depends on the numpy build and on how you write the
expression, so the simulator stays out of complex arithmetic entirely: both
backends carry real and imaginary parts separately, because real multiplies
and adds are commutative and correctly rounded, and they choose every code
path from the circuit, never from the batch's values. The test uses 300 rows,
which keeps it above that 256 KiB line.

The invariance stops at the simulator, though. The Gram matrix is a BLAS
matmul, just like on the default path, and inherits its shape-dependent
rounding: a chunked Gram (`batch_size`) differed from the unchunked one by up
to 1.8e-15 at 8 qubits and 200 samples on Kaggle. Whitened modes get their
angles out of a matmul too, so there it covers the simulator and not the whole
pipeline.

**How I checked it.** Before release the fused simulator went through an
adversarial review. Five reviewer agents covered correctness fuzzing,
integration, batch invariance, performance claims and torch edge cases,
and two skeptic agents re-examined each finding. They confirmed 26 findings,
some of them duplicates, and this release fixes all of them:

- copies of a model (deep-copied, pickled, or returned by joblib) silently
  skipped the CX-network permutation, since it was looked up by the `id()` of
  an object every copy replaces; some models couldn't be pickled at all;
- gates with non-angle parameters (`UnitaryGate`, `Initialize`, `2**x` angles)
  crashed instead of falling back, and a circuit parameter named like a numpy
  function shadowed that function;
- float constants were printed with 15 digits, which put states 1.7e-12 off
  for features of size ~30 (7.7e-14 with all 17);
- every kernel re-lowered the circuit and re-simulated cached rows, and the
  fallback warning repeated once per kernel;
- smaller things: `batch_size <= 0` returned a wrong-shaped matrix on torch
  without complaint, an empty projected-kernel input crashed, signs of zero
  depended on the batch, the batch-invariance test was too small to catch the
  regression it guards, and nothing bounded whole-batch memory.

The reviewers also flagged claims in these notes that the measurements didn't
support, and those are gone. Every behavioural fix has a regression test.

Running the suite on Kaggle then caught three tests that asserted more than
the code promises. Two wanted a chunked Gram matrix to equal the unchunked one
bit for bit. For those shapes that held on one Kaggle machine, but on the
other two the results came out up to 4.4e-16 apart. The third wanted torch-CPU
states to equal numpy's bit for bit, which held on the AMD EPYC but not on two
Intel Xeons; there they were up to 3.9e-17 apart, because torch and numpy
evaluate cos/sin with different vector math. The tests now check equality to 1e-14 and 1e-15.

### Performance: same numbers, less work

Every change below is exact. I captured qmlf's outputs with the original 1.4.0
wheel (`dist/`, the same wheel the Kaggle benchmark installs) and with this
release, and all 110 arrays came out **bit-identical** on both the Kaggle CPU
and T4 machines. They cover Gram matrices, predictions,
decision functions, CV tables, and QNN/hybrid/transformer read-outs and
gradients. The only thing that moved is kernel-target alignment, in its last
digit (~1e-16), because it now centres a matrix without building the centring
matrix.

Against 1.4.0 on the Kaggle CPU machine, with the default simulator, 60
samples x 6 qubits, and the median of 3 fresh processes per task:

| task | 1.4.0 | 1.5.0 | speedup |
|---|---|---|---|
| `QuantumClassifier().fit` (bandwidth sweep) | 8.38 s | 1.72 s | 4.9x |
| `predict` on 60 new rows | 0.477 s | 0.231 s | 2.1x |
| `QuantumClassifier(kernel="projected").fit` | 12.02 s | 1.74 s | 6.9x |
| `nystrom_features` (20 landmarks) | 0.371 s | 0.229 s | 1.6x |
| one cold Gram matrix | 0.226 s | 0.226 s | unchanged |

The T4 machine's CPU, an Intel Xeon, gave 5.0x, 2.0x, 6.9x and 1.6x on the
first four tasks and left the cold Gram matrix unchanged as well.

- **Simulated states get reused.** The state a row encodes to is a pure
  function of the circuit and the angles, so qmlf caches it under both. Before
  this, `predict()` re-simulated the entire training set on every call, a
  cross kernel re-simulated the training rows it had just been fitted on, and
  a CV sweep re-simulated every fold's training rows once per fold. Now each
  row gets simulated once, and the tests count simulations to prove it.
  Whitened modes prepare different angles per fold, so they miss the cache, as
  they should. `QMLF_STATE_CACHE_MB` sets the budget (64 by default), and
  `qmlf.state_cache_info`, `qmlf.clear_state_cache` and
  `qmlf.set_state_cache_limit(0)` (which turns it off) are public.

  The key is the circuit's full structure rather than a summary of it: gates,
  wires, parameter values (arrays by their bytes, not by how numpy prints
  them), parameter binding order, and the contents of composite instructions
  all the way down. If something can't be pinned down exactly, such as an
  opaque instruction with no definition, caching switches off for that circuit
  instead of risking a collision. A lock guards the cache, so threaded callers
  get the same answers instead of a `KeyError` from inside
  `compute_kernel_matrix`. The projected kernel keeps only the Bloch vectors
  it reuses and drops the statevectors it reduces them from (4,096 bytes per
  row at 8 qubits, against 192 for the Bloch vector).
- **`batch_size` has to be a positive integer.** `compute_kernel_matrix` now
  raises `ValueError` for 0, negatives, floats and booleans on every path.
  Each path used to fail in its own way: a `range()` error, an empty `vstack`,
  or on torch a wrong-shaped matrix with no error at all.
- **`batch_size` stopped multiplying the work it's meant to bound.** On the
  statevector path each chunk used to re-simulate the *entire* target table,
  so batching cost O(n^2 / batch_size) simulations instead of O(n) and the
  smaller the batch, the slower it ran. Both tables now get simulated once and
  only the Gram matmul is chunked.
- **The quantum layers simulate each sample once instead of once per
  observable.** Qiskit's `StatevectorEstimator` builds a fresh statevector per
  (parameter set, observable) pair, while the layers read one Z per qubit, so
  `qmlf.ops.shared_estimator.SharedStateEstimator` shares the state across a
  pub's observables. I counted one forward pass of the default 8-qubit layer on
  a batch of 4 going from 32 simulations to 4. Timed against 1.4.0 on the
  Kaggle CPU machine (layer `real_amplitudes` with reps 5, 48 parameters,
  batch 4; median of 5 forwards and 3 backwards in one process):

  | layer | forward | parameter-shift backward |
  |---|---|---|
  | 8 qubits, batch 4 | 0.226 s -> 0.036 s (6.4x) | 36.1 s -> 5.26 s (6.9x) |
  | 4 qubits, reps 2, batch 6 | 0.043 s -> 0.015 s (2.8x) | 1.81 s -> 0.54 s (3.4x) |

  Outputs and gradients are bit-identical (they're among the 110 arrays
  above). The shortcut steps aside for shot emulation (`precision > 0`) and for
  circuits containing `reset` or `measure`, where the simulation itself is
  stochastic.
- **`QuantumClassifier.fit` lost an O(n^3) step.** The post-fit concentration
  guard called the full diagnostics report, which eigendecomposes the training
  Gram matrix, and then used only the verdict. That verdict is the
  off-diagonal mean, an O(n^2) read. `diagnose()` still reports the spectrum.
- **Kernel-target alignment centres in O(n^2)** where it used to do two dense
  n x n matmuls per candidate bandwidth.

Two things I left alone on purpose. `geometric_difference` still forms an
explicit inverse, because switching to a solve moves the value on
ill-conditioned classical Gram matrices, and that's a numerics decision rather
than a speed one. And `OMP_NUM_THREADS=1` is still set process-wide at import:
relaxing it risks the macOS crash it was added for, and it can change threaded
BLAS results.

## 1.4.0 (2026-08-21)

This covers everything between v1.2.0 (`f6a79c8`) and this release, which
landed as one commit, grouped by how much it matters.

### Fixed: crashes and nondeterminism

- **A SIGSEGV on the documented import order.** `import torch` followed by
  `run_quantum_benchmark()` killed the interpreter without a traceback
  (duplicate vendored OpenMP runtimes on macOS/arm64, reproducible 3 times out
  of 3). The existing `OMP_NUM_THREADS=1` mitigation used to live in
  `integration/q_pipeline.py`, where it only protected the CLI. It now sits in
  `qmlf/__init__.py`, so it lands before any submodule import, and it respects
  a caller's own `OMP_NUM_THREADS`.
- **QNN read-outs were shot noise by default.** qiskit-machine-learning
  defaults `EstimatorQNN` to `default_precision=0.015625`. On a circuit whose
  exact expectation was 0.032, five identical forward passes spanned 0.037, and
  seeded training didn't reproduce. All three QNN-backed layers (`qnn`,
  `hybrid`, `transformer`) now default to exact statevector read-out and take
  `precision=` if you want sampling back (`precision=0.015625` restores the old
  behaviour, bit for bit in distribution).
- **`QuantumPipeline.run()` left dropout on at inference.** `no_grad()` turns
  off autograd, not dropout, so two `run()` calls on identical data disagreed.
  It now calls `.eval()` and restores the layer's previous mode afterwards.
- **`compute_vqe_energy()` takes `initial_point=`**, so you can make the VQE
  deterministic; the default random start hasn't changed.
- **`AdvancedQIGASelector` takes `random_state=`** for self-contained
  reproducibility, and `convergence_history` resets on each `fit()` instead of
  piling up across refits.

### Fixed: silent wrong answers

- **Noise.** An unknown `strategy` raises at construction; a mis-cased
  `"ZNE"` used to hand back the input unmitigated without a word. `zne` on a
  1D vector raises instead of polyfitting probabilities as if they were noise
  scales and returning 1.0, and `zne` with a single scale warns. Uncalibrated
  `readout` warns, `fit()` on a non-square calibration matrix warns instead of
  quietly storing nothing, and `noise_level` is validated to `[0, 1)`.
- **Kernel.** When `n_qubits` doesn't match the data width, the error names
  both numbers instead of surfacing Qiskit's "Mismatching number of values and
  parameters". Both `fit()` and the un-fitted ZZ path check it.
- **`run_quantum_benchmark(n_qubits=...)`** warns when the argument disagrees
  with the data width it actually uses. It used to override it silently.
- **Dead parameters tell the truth now.** `AdvancedQIGASelector(n_features=...)`
  warns that it doesn't bound the subset size, `FederatedQML` warns when the
  reporting cohort differs from `num_clients`, and chem warns when the
  geometry's atom count differs from `num_atoms`.

### Fixed: API contracts and error quality

- Torch layers (`qnn`, `hybrid`, `transformer`, `QMLFPipeline`) check rank and
  width up front, with messages that name the expected shape, instead of dying
  inside a reshape or matmul.
- The graph kernel raises a clear error when `n_neighbors > n_samples`.
- `qmlf.plot_hilbert_space` accepts `show=` and `save_path=` for headless use,
  and the `viz` module-level wrappers pass every keyword argument through
  instead of silently dropping some.

### Packaging

- `pyproject.toml` requires `setuptools>=77`, since the declared 61 couldn't
  parse the SPDX license and broke the build. It also sets
  `requires-python>=3.10` (qiskit's own floor) and adds version classifiers,
  keywords and Source/Issues URLs.
- `requirements.txt` puts upper bounds on every dependency and lowers the
  qiskit floor to 2.3, matching what the suite actually runs against.
- The version is 1.4.0. I re-versioned the private Kaggle wheelhouse dataset to
  match, and the in-sandbox install probe verified the wheel end to end.

### Added: the simplification layer

The three-line path:

- **`QuantumClassifier`** is the three-line quantum-kernel SVM. It sizes its
  circuit from the data, computes and wires the precomputed Gram matrices
  itself, and by default picks `bandwidth` with a deterministic, leak-free
  stratified-CV sweep. That fixes the documented trap where the
  zero-knowledge path scored at chance (0.40 -> 0.70+ on the reference dataset
  with no arguments). It's sklearn-compatible (`clone`, `get_params`),
  handles multiclass, is deterministic, and exposes the sweep table in
  `cv_results_`. Whitened modes default to the safe `normalize="maxabs"`.
- **Wide data gets fewer qubits automatically.** `QuantumClassifier`
  PCA-reduces inputs wider than `max_qubits` (8 by default) with a
  deterministic projection fitted per fold without leakage, so you never have
  to decide the qubit count, whatever the width. `reduction_` and `n_qubits_`
  report what happened, and `max_qubits=None` restores the old
  warn-and-attempt behaviour.
- **Encoding search.** `mode="auto"` and `feature_map="auto"` extend the CV
  sweep across raw versus whitened preprocessing and entangled versus
  unentangled maps, so the data picks the encoding. The winners land in
  `mode_` and `feature_map_`, the full table in `cv_results_`.
- **`QuantumClassifier(kernel=...)`** runs `"projected"` end to end, or
  `"auto"` to search the kernel family as a fourth axis, with the winner in
  `kernel_` and a column in `cv_results_`. A post-fit concentration guard warns
  when the fitted kernel is degenerate.
- **`QuantumRegressor`** is kernel ridge with the whole auto stack: it sizes
  itself, PCA-reduces, runs deterministic CV over MSE or uses the median
  bandwidth, and takes projected kernels. It rejects the classification-only
  options with clear errors. qiskit's QSVR ships none of this.

Kernels:

- **Projected quantum kernels** (Huang et al. 2021).
  `QuantumKernel(kernel="projected", gamma="auto")` is an RBF over exact
  per-qubit reduced density matrices, with `gamma` fitted on the training data
  by the median heuristic; it's PSD, deterministic and batched. At release I
  measured 100x more off-diagonal mass than the fidelity kernel at 8 qubits and
  bandwidth 1.0 (0.356 vs 0.004). Re-measured for 1.5.0 on a stated dataset,
  `make_classification(n_samples=100, n_features=8, random_state=0)`, it's
  0.362 vs 0.004, about 87x; the README quotes that one. qiskit-machine-learning
  doesn't have it (checked against the 0.9 source).
- **Per-feature (ARD) bandwidth.** `bandwidth` accepts a positive vector,
  applied per feature after whitening. That makes it a per-direction metric no
  external rescaling of the input can reach, and a wrong length fails as soon
  as data shows up.
- **Median-heuristic bandwidth.** `bandwidth="median"` fits the angle scale
  from pairwise distances in the training data in one deterministic pass. The
  fitted value goes in `bandwidth_`, and `QuantumClassifier` supports it by
  skipping the CV sweep.
- **`bandwidth="alignment"`** picks the bandwidth that maximises
  kernel-target alignment on the kernel itself. It's classification-only,
  raises clear errors for missing labels or regression targets, and composes
  with `kernel="projected"`.
- **Fisher whitening.** `mode="fisher"` whitens by the class-size-weighted
  within-class scatter, which finally puts the previously unused `y` in
  `QuantumKernel.fit` to work. It raises clear errors without labels or with
  only singleton classes, and the classifier defaults it to the safe `maxabs`
  normalisation.
- **`QuantumKernel(n_qubits=None)`** sizes the kernel from the first data it
  sees, or from a supplied `QuantumCircuit`. An explicit `n_qubits` behaves
  exactly as before, and touching internals before sizing raises a clear error
  instead of handing back `None`.

Analysis:

- **The quantum-advantage screen** (Huang et al. 2021, Nat. Commun.):
  `quantum_advantage_report(X, y)`, `geometric_difference` and
  `model_complexity` in the new `qmlf.analysis` package. From the Gram
  matrices alone it computes the trace-normalised geometric difference g and
  both model complexities and gives a conservative verdict. A sanity test pins
  g(K, K) = 1.
- **Nyström kernel approximation.** `QuantumKernel.nystrom_features(X,
  n_landmarks=m)` uses deterministic landmarks and a rank-safe W^{-1/2}, and is
  exact at m = n. `circuit_budget(n, m)` does the resource accounting:
  n(n-1)/2 pairwise fidelity circuits against roughly n*m for the approximate
  path.
- **Kernel health diagnostics.** `kernel_diagnostics(gram, y)` and
  `clf.diagnose()` report a concentration verdict, the eigenvalue spectrum
  share and centered kernel-target alignment (Cortes et al.). The docstring
  explains, from measurement, why the scale and direction checks complement
  each other.

### Added

- `tests/`: a determinism suite, an input-validation suite, smoke-and-contract
  suites for viz, the CLI, nisq, graph and chem, golden-value regression tests
  that pin kernel numerics against dependency drift, and a claims suite
  (`test_simplification`) that pins the three-line path itself.
- `kaggle/`: a verified benchmark task (`qmlf-quantum-kernel-tuning`), two
  zero-quota sandbox probes, the reference sweep, wheelhouse metadata, and
  `SANDBOX.md` with measured facts about the sandbox.
