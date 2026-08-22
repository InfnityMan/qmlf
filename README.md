# qmlf

A quantum machine learning framework built on Qiskit, PyTorch and
scikit-learn. It pulls together a few building blocks I keep reaching for:
quantum kernels, a trainable quantum neural network layer, a couple of hybrid
models, plus some tools for noise mitigation, circuit optimization, feature
selection and plotting.


## Requirements

Python 3.10+ and Qiskit 2.3 or newer (both floors are what the test suite
actually runs against). Install the dependencies with:

    pip install -r requirements.txt

or install the package itself (which also registers the `qmlf` command):

    pip install .

It uses the current Qiskit primitives (`StatevectorEstimator`) and the
`zz_feature_map` / `real_amplitudes` circuit builders, so it runs on Qiskit 2.x
without any of the old V1 shims.

## Quick start

Classify with a quantum kernel in three lines — no circuits, no Gram-matrix
wiring, no qiskit imports:

```python
from qmlf import QuantumClassifier

clf = QuantumClassifier()                # qubits sized from the data
clf.fit(X_train, y_train)                # bandwidth tuned by deterministic CV
clf.predict(X_test)
```

Everything that used to require insider knowledge happens inside `fit`: the
encoding circuit is sized from the feature count, the fidelity Gram matrices
are computed in the right orientation and wired into an SVM, and `bandwidth`
— the knob that separates a working fidelity kernel from a degenerate one —
is selected by a leak-free, deterministic cross-validation sweep. On the
reference dataset this takes the zero-knowledge path from 0.40 (chance, the
old defaults) to 0.70+ with no arguments at all. The chosen value is in
`clf.bandwidth_`, the full sweep table in `clf.cv_results_`, and every knob
(`mode`, `feature_map`, `entanglement`, `bandwidths`, `C`) is still there for
power users.

**Nothing must be chosen; everything can be.** Every automatic decision has
an override and reports what it decided:

| automatic | override | decision visible in |
|---|---|---|
| qubit count (wide data is PCA-reduced to fit the simulator) | `max_qubits=` / `None` | `n_qubits_`, `reduction_` |
| bandwidth sweep | `bandwidth=0.05` | `bandwidth_`, `cv_results_` |
| sweep grid | `bandwidths=(...)` | `cv_results_` |
| encoding search | `mode=` / `feature_map=` (or `"auto"` to search) | `mode_`, `feature_map_` |
| safe normalisation for whitened modes | `normalize=` | — |

With `mode="auto", feature_map="auto"` the *encoding itself* is selected by
cross-validation — entangled vs unentangled map, raw vs Mahalanobis-whitened
data — so the choice qiskit tutorials hand to the reader becomes a searched
hyperparameter. After fitting, `clf.diagnose()` reports whether the kernel is
concentrated (the classic silent fidelity-kernel failure), its spectrum, and
its centered kernel-target alignment in one call.

Or work with the kernel directly:

```python
import numpy as np
from qmlf import QuantumKernel

X = np.random.rand(20, 4)
kernel = QuantumKernel().fit(X)          # n_qubits inferred from the data
gram = kernel.compute_kernel_matrix(X)   # 20 x 20 kernel matrix
```

(`n_qubits` can still be passed explicitly; a mismatch with the data raises
an error naming both numbers.)

**Tune the bandwidth.** Fidelity kernels concentrate: as the encoding angles
span a wider range, off-diagonal similarities collapse toward zero, the Gram
matrix approaches the identity, and a downstream SVM memorises the training set
instead of generalising. `bandwidth` scales the angles down and is usually the
single most valuable thing to sweep — on a real benchmark it was worth roughly
0.3 AUROC:

```python
kernel = QuantumKernel(n_qubits=4, mode="ZZ", bandwidth=0.1).fit(X)
```

For `mode="covariant"` (alias `"mahalanobis"` — Mahalanobis/ZCA whitening then a
ZZ map, *not* the group-covariant kernel of Glick et al.) also pass
`normalize="maxabs"`. Whitening cancels any scaling applied to its input, so
without normalisation the whitened angles can exceed `[-pi, pi]` and alias:

```python
kernel = QuantumKernel(
    n_qubits=4, mode="mahalanobis", bandwidth=0.5, normalize="maxabs"
).fit(X)
```

Both are available from the CLI too (`--bandwidth`, `--normalize`). To run
against real hardware or a noisy simulator, pass `fidelity=` or `sampler=`; the
exact statevector fast path is used automatically only when neither is given.

**Choose the encoding.** The feature map and its entanglement structure are
hyperparameters, not fixed choices. Entanglement drives the concentration that
`bandwidth` compensates for, so the two interact:

```python
QuantumKernel(n_qubits=8, feature_map="zz", entanglement="linear")  # or "full", "circular"
QuantumKernel(n_qubits=8, feature_map="z")                          # no entanglement, much faster
QuantumKernel(n_qubits=8, feature_map=my_circuit)                   # any Qiskit QuantumCircuit
```

On at least one real dataset the unentangled `"z"` map matched or beat `"zz"`
while running 6-11x faster, so the entangled default is not automatically the
right choice — sweep it. CLI: `--feature-map`, `--entanglement`.

> **Don't hand `.fidelity_quantum_kernel` to `QSVC`.** Doing so bypasses
> `_prepare`, silently discarding whitening, `normalize` and `bandwidth` — so
> `mode="covariant"` becomes indistinguishable from `mode="ZZ"` with no error.
> Use `compute_kernel_matrix` with `SVC(kernel="precomputed")` instead. Accessing
> the attribute now warns when it would change your answer.

Drop a trainable quantum layer into a torch model:

```python
import torch
from qmlf import create_advanced_qnn_layer

layer = create_advanced_qnn_layer(n_qubits=4, output_dim=8)
layer.eval()                             # see the note below
y = layer(torch.randn(16, 4))            # (16, 8)
```

**Read-outs are exact by default.** The quantum layers evaluate their
observables straight off the statevector, so the same input gives the same
answer every time. That is what you want when you are seeding an experiment or
asserting on a number.

To emulate a finite-shot device instead, ask for it:

```python
create_advanced_qnn_layer(n_qubits=4, precision=0.015625)   # sampled, noisy
```

Before 1.2.1 the sampled path was the *default*, inherited from
qiskit-machine-learning, and there was no way to turn it off. On a circuit whose
exact expectation value was 0.032, five successive forward passes spanned 0.037
— more drift than signal — and training would not reproduce under a fixed seed.
If you were relying on that noise, pass `precision=0.015625` to get it back.

One PyTorch footgun worth repeating, because the quantum layers carry dropout in
their classical heads: `torch.no_grad()` turns off gradients, **not** dropout.
Call `.eval()` for inference. `QuantumPipeline.run()` does this for you and
restores the layer's previous mode afterwards.

Or chain a kernel, a QNN and a Hilbert-space plot together in one call:

```python
import numpy as np
from qmlf import create_quantum_pipeline

X = np.random.rand(20, 4)
result = create_quantum_pipeline().run(X, labels=None)
result["kernel_matrix"]                  # 20 x 20 quantum kernel
result["qnn_output"]                     # QNN read-out for each sample
```

The number of qubits is taken from the data when it isn't given.

## Command line

Installing the package puts a `qmlf` command on the path (or run it as
`python -m qmlf`). It works out of the box on a synthetic dataset, or point it
at your own CSV with `--csv` (the last column is the target unless you pass
`--target`):

    qmlf benchmark --n-qubits 4
    qmlf visualize --mode covariant --save kernel.html

## What's included

* `models`: `QuantumClassifier` and `QuantumRegressor` — three-line quantum-kernel estimators (auto-sized, auto-tuned, sklearn-compatible).
* `analysis`: the Huang-et-al. quantum-advantage screen (`quantum_advantage_report`, `geometric_difference`, `model_complexity`).
* `ops`: quantum fidelity kernel over a ZZ feature map, plus a QSVC vs. XGBoost benchmark.
* `graph`: graph-diffusion kernel, with an optional quantum kernel backend (`use_quantum=True`).
* `qnn`, `hybrid`, `transformer`: torch layers wrapping a real `EstimatorQNN`, read out from per-qubit Z expectation values.
* `chem`: charge-aware Coulomb-matrix descriptors, a model Hamiltonian, classical ground-state energy, and an optional VQE.
* `nisq`: transpiler-based circuit depth and gate reduction.
* `noise`: readout calibration, depolarizing-channel inversion, and zero-noise extrapolation.
* `optimizers`: quantum-inspired evolutionary feature selection.
* `federated`: sample-weighted federated averaging.
* `viz`: Plotly charts (kernels, Bloch sphere, training curves, barren-plateau diagnostics).
* `integration`: end-to-end pipelines, including a one-line `QuantumPipeline` that runs kernel -> QNN -> visualization.

## What this adds over raw Qiskit ML / TensorFlow Quantum

Everything here runs *on* Qiskit, so nothing is impossible elsewhere — the
point is what exists as a built-in versus what you would have to build:

* **Projected quantum kernels** (Huang et al. 2021) as a mode string:
  `QuantumKernel(kernel="projected")` builds the RBF over per-qubit reduced
  density matrices, with a train-fitted median-heuristic `gamma`. This is the
  literature's answer to exponential fidelity-kernel concentration, and it
  measurably works: at 8 qubits and bandwidth 1.0 the fidelity Gram matrix
  collapses to 0.004 mean off-diagonal while the projected kernel holds 0.356
  — a 100x difference on the exact failure mode. Not implemented anywhere in
  qiskit-machine-learning (verified: zero source references).
* **Per-feature (ARD) bandwidth.** `bandwidth=np.array([...])` scales each
  feature's encoding angle independently — and because it lands *after*
  whitening, it is a per-direction metric that no external rescaling of the
  input can reproduce (whitening cancels input scaling exactly).
* **CV-free bandwidth via the median heuristic.** `bandwidth="median"` fits
  the scale from the training angles' pairwise distances — one deterministic
  pass, no folds, no SVM fits.
* **Supervised (Fisher) whitening.** `mode="fisher"` whitens by the
  within-class scatter instead of the global covariance, so noise directions
  shrink and class-separating directions keep their scale — metric learning
  inside the quantum kernel, using the labels qiskit's kernels never see.
* **Bandwidth as a first-class, auto-tuned parameter.** `FidelityQuantumKernel`
  has no bandwidth; the literature (and our measured sweeps: 0.40 -> 0.75+)
  says it is the single most important knob. Here it is a constructor argument
  and, by default, tuned for you.
* **Searchable encodings.** `mode="auto", feature_map="auto"` cross-validates
  entangled/unentangled maps and raw/whitened preprocessing jointly. QSVC asks
  you to hand-build the feature map before you start.
* **Mahalanobis-whitened fidelity kernels** with leak-free, train-fitted
  normalisation — covariance-adapted quantum kernels as a mode string, not a
  research project.
* **O(n) exact statevector fast path.** Pure-state fidelity Gram matrices are
  computed from n statevectors and one matmul instead of n² circuit pairs —
  measured 6–11x faster than the pairwise path, and bit-for-bit reproducible.
* **The quantum-advantage screen** (Huang et al. 2021). One call —
  `quantum_advantage_report(X, y)` — answers the question every quantum
  kernel project should ask before running a single circuit: *can this
  quantum kernel beat a classical RBF on this data?* Computes the paper's
  geometric difference g and both model complexities from the Gram matrices
  alone, with a conservative verdict. No framework ships this.
* **Nystrom kernel approximation with a circuit budget.**
  `kernel.nystrom_features(X, n_landmarks=m)` gives an explicit feature map
  usable with any linear model, cutting a hardware run from n(n-1)/2
  fidelity circuits to ~n*m; `kernel.circuit_budget(n)` prints the
  arithmetic for every evaluation path. qiskit-ml has no kernel
  approximation and no resource accounting.
* **Alignment-selected bandwidth.** `bandwidth="alignment"` picks the scale
  that maximises centered kernel-target alignment — one Gram matrix per
  candidate, no folds, no SVM fits; a third selection principle beside CV
  and the median heuristic.
* **A regressor with the full auto stack.** qiskit's QSVR exists but bare —
  no bandwidth (the parameter does not exist there), no tuning, no
  dimensionality handling. `QuantumRegressor` is kernel ridge with
  everything above: auto-sized, auto-reduced, CV- or median-tuned, projected
  kernels included.
* **A post-fit concentration guard.** `QuantumClassifier` warns at fit time
  when the kernel it just built is degenerate, instead of letting a
  memorising model reach the test set silently.
* **Kernel health diagnostics.** `kernel_diagnostics()` / `clf.diagnose()`
  detect concentration and report centered kernel-target alignment. Neither
  qiskit-machine-learning nor TFQ ships a built-in check for the failure mode
  their own kernels exhibit.
* **Exact-by-default QNN read-outs** with shot noise as an explicit opt-in
  (`precision=`), where the upstream default silently samples.
* **Deterministic everything**: same data, same answer, across processes —
  the property a benchmark or a paper actually needs.

## A few caveats

The quantum kernels and the VQE run on simulators, and the cost grows with the
number of circuit evaluations, so keep datasets and qubit counts small. The
chemistry module works from a geometry-derived model Hamiltonian, not a full
electronic-structure calculation.

`n_qubits` must equal the number of input features — the feature map encodes one
angle per feature. Passing a mismatch raises and tells you both numbers.

A few constructor arguments are declarative and do not resize anything:
`AdvancedQIGASelector(n_features=...)` does not request a subset of that size
(the selector keeps whatever scores best), and `FederatedQML(num_clients=...)`
and `AdvancedQuantumChemistryLayer(num_atoms=...)` are sized from the data you
pass. All three warn when what you declared disagrees with what arrived.

Torch, scikit-learn and xgboost each vendor their own OpenMP runtime, and more
than one thread pool in a process can kill the interpreter with no traceback.
`import qmlf` sets `OMP_NUM_THREADS=1` unless you have already chosen a value.

## License

See [LICENSE](LICENSE).
