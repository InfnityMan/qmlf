# qmlf

qmlf is a quantum machine learning framework built on Qiskit, PyTorch and
scikit-learn. It collects the pieces I kept rebuilding by hand: quantum
kernels, a trainable quantum neural network layer, a couple of hybrid models,
and some tools for noise mitigation, circuit optimization, feature selection
and plotting.


## Requirements

You need Python 3.10+ and Qiskit 2.3 or newer; those floors are what the test
suite actually runs against. Install the dependencies with:

    pip install -r requirements.txt

or install the package itself, which also registers the `qmlf` command:

    pip install .

qmlf uses the current Qiskit primitives (`StatevectorEstimator`) and the
`zz_feature_map` / `real_amplitudes` circuit builders, so it runs on Qiskit 2.x
without any of the old V1 shims.

## Quick start

A quantum-kernel classifier takes three lines. You don't build a circuit, you
don't wire up a Gram matrix, and you don't import qiskit:

```python
from qmlf import QuantumClassifier

clf = QuantumClassifier()                # qubits sized from the data
clf.fit(X_train, y_train)                # bandwidth tuned by deterministic CV
clf.predict(X_test)
```

`fit` handles the parts that used to need insider knowledge. It sizes the
encoding circuit from the feature count, computes the fidelity Gram matrices
in the right orientation and hands them to an SVM, then picks `bandwidth` with
a leak-free, deterministic cross-validation sweep. Bandwidth is the knob that
separates a working fidelity kernel from a degenerate one, which is why it
gets tuned for you: on the reference dataset, running with no arguments at
all now scores 0.70+ where the old defaults sat at chance (0.40). The chosen
value lands in `clf.bandwidth_` and the full sweep in `clf.cv_results_`, and
every knob (`mode`, `feature_map`, `entanglement`, `bandwidths`, `C`) is still
there if you'd rather set it yourself.

You never have to choose anything, but you can override every choice, and
each one tells you what it decided:

| automatic | override | decision visible in |
|---|---|---|
| qubit count (wide data is PCA-reduced to fit the simulator) | `max_qubits=` / `None` | `n_qubits_`, `reduction_` |
| bandwidth sweep | `bandwidth=0.05` | `bandwidth_`, `cv_results_` |
| sweep grid | `bandwidths=(...)` | `cv_results_` |
| encoding search | `mode=` / `feature_map=` (or `"auto"` to search) | `mode_`, `feature_map_` |
| safe normalisation for whitened modes | `normalize=` | `_resolved_normalize()` |

With `mode="auto", feature_map="auto"`, cross-validation picks the encoding as
well: entangled or unentangled map, raw or Mahalanobis-whitened data. Qiskit's
tutorials leave that decision to you; here it's one more hyperparameter in the
search. Once the model is fitted, `clf.diagnose()` tells you in one call
whether the kernel has concentrated (the classic way a fidelity kernel fails
without raising anything), what its spectrum looks like, and its centered
kernel-target alignment.

You can also work with the kernel directly:

```python
import numpy as np
from qmlf import QuantumKernel

X = np.random.rand(20, 4)
kernel = QuantumKernel().fit(X)          # n_qubits inferred from the data
gram = kernel.compute_kernel_matrix(X)   # 20 x 20 kernel matrix
```

Passing `n_qubits` yourself still works, and if it doesn't match the data the
error names both numbers.

**Tune the bandwidth.** Fidelity kernels concentrate. As the encoding angles
spread over a wider range, the off-diagonal similarities fall toward zero, the
Gram matrix drifts toward the identity, and the SVM downstream memorises the
training set instead of generalising. `bandwidth` scales the angles down, and
it's usually the most valuable single thing you can sweep; on a real benchmark
it was worth roughly 0.3 AUROC:

```python
kernel = QuantumKernel(n_qubits=4, mode="ZZ", bandwidth=0.1).fit(X)
```

`mode="covariant"` (alias `"mahalanobis"`) applies Mahalanobis/ZCA whitening
and then a ZZ map, so it isn't the group-covariant kernel of Glick et al. Pass
`normalize="maxabs"` along with it. Whitening cancels any scaling you apply to
its input, and without normalisation the whitened angles can run past
`[-pi, pi]` and alias:

```python
kernel = QuantumKernel(
    n_qubits=4, mode="mahalanobis", bandwidth=0.5, normalize="maxabs"
).fit(X)
```

The CLI has both (`--bandwidth`, `--normalize`). To run on real hardware or a
noisy simulator, pass `fidelity=` or `sampler=`; qmlf takes the exact
statevector fast path only when you've given neither.

**Choose the encoding.** Treat the feature map and its entanglement pattern as
hyperparameters rather than fixed choices. Entanglement drives the
concentration that `bandwidth` compensates for, so the two interact:

```python
QuantumKernel(n_qubits=8, feature_map="zz", entanglement="linear")  # or "full", "circular"
QuantumKernel(n_qubits=8, feature_map="z")                          # no entanglement, much faster
QuantumKernel(n_qubits=8, feature_map=my_circuit)                   # any Qiskit QuantumCircuit
```

On a Kaggle CPU session (Intel Xeon @ 2.20GHz) the unentangled `"z"` map
computed a 200-sample Gram matrix 2.8–8.5x faster than `"zz"` at 4–12 qubits,
and which one classifies better depends on your data. So don't assume the
entangled default is right; sweep it (the CLI flags are `--feature-map` and
`--entanglement`).

> **Don't hand `.fidelity_quantum_kernel` to `QSVC`.** That skips `_prepare`
> and silently throws away whitening, `normalize` and `bandwidth`, which makes
> `mode="covariant"` behave exactly like `mode="ZZ"` with nothing to tell you.
> Use `compute_kernel_matrix` with `SVC(kernel="precomputed")` instead. Reading
> the attribute now warns whenever it would change your answer.

Dropping a trainable quantum layer into a torch model looks like this:

```python
import torch
from qmlf import create_advanced_qnn_layer

layer = create_advanced_qnn_layer(n_qubits=4, output_dim=8)
layer.eval()                             # see the note below
y = layer(torch.randn(16, 4))            # (16, 8)
```

**Read-outs are exact by default.** The quantum layers read their observables
straight off the statevector, so the same input gives the same answer every
time, which is what you want when you're seeding an experiment or asserting on
a number. If you'd like to emulate a finite-shot device, ask for it:

```python
create_advanced_qnn_layer(n_qubits=4, precision=0.015625)   # sampled, noisy
```

Before 1.2.1 the sampled path was the *default*, inherited from
qiskit-machine-learning, and you couldn't turn it off. On one circuit with an
exact expectation value of 0.032, five forward passes in a row spanned 0.037:
more drift than signal, and training wouldn't reproduce under a fixed seed. If
you were relying on that noise, `precision=0.015625` brings it back.

One PyTorch trap bears repeating, because the quantum layers carry dropout in
their classical heads. `torch.no_grad()` turns off gradients but leaves
dropout on, so call `.eval()` for inference. `QuantumPipeline.run()` does that
for you and puts the layer back in its previous mode afterwards.

Or chain a kernel, a QNN and a Hilbert-space plot in one call:

```python
import numpy as np
from qmlf import create_quantum_pipeline

X = np.random.rand(20, 4)
result = create_quantum_pipeline().run(X, labels=None)
result["kernel_matrix"]                  # 20 x 20 quantum kernel
result["qnn_output"]                     # QNN read-out for each sample
```

If you don't give the number of qubits, it comes from the data.

## Command line

Installing the package puts a `qmlf` command on your path (`python -m qmlf`
works too). Out of the box it runs on a synthetic dataset. Point it at your own
CSV with `--csv` and it treats the last column as the target unless you pass
`--target`:

    qmlf benchmark --n-qubits 4
    qmlf visualize --mode covariant --save kernel.html

## What's included

| module | what's in it |
|---|---|
| `models` | `QuantumClassifier` and `QuantumRegressor`: three-line quantum-kernel estimators that size and tune themselves and work with sklearn |
| `analysis` | the quantum-advantage screen from Huang et al. (`quantum_advantage_report`, `geometric_difference`, `model_complexity`) |
| `ops` | a quantum fidelity kernel over a ZZ feature map, and a QSVC vs. XGBoost benchmark |
| `graph` | a graph-diffusion kernel with an optional quantum backend (`use_quantum=True`) |
| `qnn`, `hybrid`, `transformer` | torch layers wrapping a real `EstimatorQNN`, read out from per-qubit Z expectation values |
| `chem` | charge-aware Coulomb-matrix descriptors, a model Hamiltonian, the classical ground-state energy, and an optional VQE |
| `nisq` | transpiler-based reduction of circuit depth and gate count |
| `noise` | readout calibration, depolarizing-channel inversion and zero-noise extrapolation |
| `optimizers` | quantum-inspired evolutionary feature selection |
| `federated` | sample-weighted federated averaging |
| `viz` | Plotly charts of kernels, the Bloch sphere, training curves and barren-plateau diagnostics |
| `integration` | end-to-end pipelines, including a one-line `QuantumPipeline` that runs kernel -> QNN -> visualization |

## What this adds over raw Qiskit ML / TensorFlow Quantum

All of it runs on Qiskit, so nothing here is impossible elsewhere. What
differs is how much comes built in and how much you'd have to write yourself.

**Kernels that don't collapse.** `QuantumKernel(kernel="projected")` gives you
the projected kernel from Huang et al. (2021), an RBF over per-qubit reduced
density matrices with `gamma` fitted on the training data by the median
heuristic. It's the literature's answer to the exponential concentration of
fidelity kernels, and you can see it work. On
`make_classification(n_samples=100, n_features=8, random_state=0)` at 8 qubits
and bandwidth 1.0, the fidelity Gram matrix collapses to a mean off-diagonal of
0.004 while the projected kernel holds 0.362, about 87x more. You won't find
one in qiskit-machine-learning: I checked 0.9.0, and every kernel it ships is
a fidelity kernel.

**Bandwidth as a real parameter.** `FidelityQuantumKernel` has no bandwidth,
even though the literature (and our own sweeps: 0.40 -> 0.75+) says it's the
most important knob. Here it's a constructor argument, tuned for you unless
you say otherwise, and there's more than one way to pick it:

* cross-validation, the default;
* `bandwidth="median"`, which fits the scale from the training angles'
  pairwise distances in one deterministic pass, with no folds and no SVM fits;
* `bandwidth="alignment"`, which takes the scale that maximises centered
  kernel-target alignment and costs one Gram matrix per candidate (again no
  folds or SVM fits);
* `bandwidth=np.array([...])`, a per-feature (ARD) scale. Because it applies
  *after* whitening, it's a per-direction metric that no rescaling of the
  input can reproduce; whitening cancels input scaling exactly.

**Encodings and whitening you can search.** `mode="auto", feature_map="auto"`
cross-validates entangled against unentangled maps and raw against whitened
preprocessing, jointly, where QSVC expects you to hand-build the feature map
before you start. The whitened modes are mode strings rather than research
projects. Mahalanobis whitening comes with leak-free normalisation fitted on
the training data. `mode="fisher"` whitens by the within-class scatter instead
of the global covariance, so noise directions shrink while class-separating
directions keep their scale; that's metric learning inside the quantum kernel,
using labels qiskit's kernels never see.

**Fewer circuits.** Pure-state fidelity Gram matrices come from n statevectors
and one matmul rather than n² circuit pairs. That O(n) path measured 61–96x
faster than qiskit-machine-learning's pairwise path at 2–8 qubits and 40
samples on a Kaggle CPU session (Intel Xeon @ 2.20GHz; the gap grows with n),
agreed with it to 2e-11, and reproduces bit for bit. For hardware runs,
`kernel.nystrom_features(X, n_landmarks=m)` gives an explicit Nyström feature
map that works with any linear model and cuts the job from n(n-1)/2 fidelity
circuits to roughly n*m, and `kernel.circuit_budget(n)` prints the arithmetic
for every evaluation path. qiskit-ml has no kernel approximation and does no
resource accounting.

**Checks before and after you fit.** `quantum_advantage_report(X, y)`
implements the screen from Huang et al. (2021) and answers the question every
quantum-kernel project should ask before running a single circuit: *can this
quantum kernel beat a classical RBF on this data?* It computes the paper's
geometric difference g and both model complexities from the Gram matrices
alone and gives a conservative verdict; no framework ships this. After fitting,
`QuantumClassifier` warns when the kernel it just built is degenerate, instead
of letting a memorising model reach your test set quietly, and
`kernel_diagnostics()` / `clf.diagnose()` report concentration and centered
kernel-target alignment. Neither qiskit-machine-learning nor TFQ has a built-in
check for the failure mode their own kernels exhibit.

**A regressor that gets the same treatment.** qiskit's QSVR exists, but bare:
it has no bandwidth parameter at all, no tuning and no dimensionality handling.
`QuantumRegressor` is kernel ridge with everything above, so it sizes itself,
reduces wide data, tunes by CV or the median heuristic, and takes projected
kernels.

**Answers you can reproduce.** QNN read-outs are exact unless you opt into
shot noise with `precision=`, where the upstream default samples without
telling you. And everything is deterministic: same data, same answer, across
processes, which is the property a benchmark or a paper actually needs.

## Speed without changing any answer

Every number in this section was measured on Kaggle's standard notebook
machines: a CPU session (AMD EPYC 7B12) and a "GPU T4 x2" session, both on
Python 3.13 with `OMP_NUM_THREADS=1`. The CHANGELOG lists the software
versions, workloads and method. Other hardware will give other numbers.

Simulating one statevector per sample is nearly the whole cost of a quantum
kernel (99.4–99.8% of a 200-sample Gram matrix at 4–12 qubits), and qmlf used
to simulate the same rows over and over. It now caches each state under its
encoding circuit and angles. That's exact, since the same row through the same
circuit is the same state, so no result changes: 110 outputs compared against
1.4.0 are bit-identical.

`predict()` used to re-simulate the entire training set on every call. Now it
simulates only the rows you hand it, and nothing at all for rows the model was
fitted on. A CV sweep used to re-simulate each fold's training rows once per
fold; a data-independent kernel now simulates each row once per candidate,
however many folds there are. (Whitened modes prepare different angles per
fold, so they still simulate per fold, as correctness requires.) Against
1.4.0, on 60 samples at 6 qubits, `fit()` with the default bandwidth sweep is
4.9x faster, a projected-kernel fit 6.9x, `predict()` on new rows 2.1x and
Nyström features 1.6x. A single cold Gram matrix is unchanged.

The cache holds 64 MB by default. One entry is `2**n_qubits` amplitudes, which
is 256 bytes at 4 qubits and 64 KB at 12. Set `QMLF_STATE_CACHE_MB` to change
the budget or call `qmlf.set_state_cache_limit(0)` to switch it off;
`qmlf.state_cache_info()` reports what it holds and `qmlf.clear_state_cache()`
empties it.

Two circuits share an entry only if they really are the same encoding. The key
covers gates, wires, parameter values, binding order and what's inside
composite instructions, so a `PauliEvolutionGate` over `XX` never gets confused
with one over `ZZ`, and neither do two large `UnitaryGate` matrices that happen
to print the same under numpy's summary. A circuit that can't be identified
exactly (an opaque instruction with no definition) just isn't cached. The cache
is safe to use from several threads.

The torch layers had the opposite problem. qiskit's estimator builds a fresh
statevector for every (sample, observable) pair, yet these layers read one Z
expectation per qubit off a single state. Sharing that state cut one forward
pass of the default 8-qubit layer (batch 4) from 32 simulations to 4. Against
1.4.0 the forward pass is 6.4x faster and the parameter-shift backward pass
6.9x (2.8x and 3.4x at 4 qubits), with bit-identical outputs and gradients.
Shot emulation (`precision > 0`) keeps the stock path, because there the noise
draws are part of the answer.

`batch_size` on `compute_kernel_matrix` isn't a speed setting; it bounds
memory. It used to re-simulate the whole target table for every chunk, which
made a smaller batch slower. Now it chunks only the matmul.

### A faster simulator, if you ask for it

Everything above keeps qiskit as the simulator, and every number stays bit for
bit what it was. For more speed, ask for the fused simulator:

```python
QuantumClassifier(simulator="fused")              # numpy, complex128
QuantumKernel(n_qubits=10, simulator="torch")     # CUDA, then Apple MPS, then CPU
```

It lowers the encoding circuit once into whole-batch array operations and runs
every sample through together instead of one at a time. Diagonal gates fold
into a single phase multiply, and each single-qubit gate becomes a butterfly
along one axis. With 200 samples and the default fully entangled ZZ map, on the
Kaggle CPU machine:

| `QuantumClassifier().fit(X, y)` | 8 qubits | 12 qubits |
|---|---|---|
| `"qiskit"` (default) | 10.1 s | 37.8 s |
| `"fused"`, first fit in a fresh process | 0.53 s (19x) | 2.11 s (18x) |
| `"fused"`, later fits | 0.20 s (50x) | 1.63 s (23x) |
| one 200 x 200 Gram matrix alone, `"fused"` | 124x | 32x |

The first fit pays a one-time cost for lowering the circuit, because sympy has
to compile the angle expressions. That happens once per circuit per process,
and every kernel built around the circuit shares the result. Every fit above
chose the same bandwidth and made the same predictions as qiskit.

A GPU pulls further ahead as the qubit count grows. On the Kaggle T4 machine,
`simulator="torch"` (CUDA, complex128) refit the 12-qubit classifier in 1.30 s.
That's 48x faster than qiskit on the same machine and nearly twice as fast as
`"fused"` on its CPU (2.44 s), while at 8 qubits the two were level (0.27 s and
0.28 s). One 12-qubit Gram matrix took 0.099 s on the T4 against 8.67 s for
qiskit, 87x. torch uses one GPU, so the second one sits idle.

Here's what was measured for each choice:

| `simulator=` | precision | measured difference from `"qiskit"` | use it for |
|---|---|---|---|
| `"qiskit"` (default) | complex128 | none, it's the reference | published numbers, benchmarks |
| `"fused"` | complex128 | Gram ≤ 2e-14, states ≤ 8e-14 | everyday speed |
| `"torch"` on CPU | complex128 | Gram ≤ 2e-14 | 10–12 qubits on CPU, where it beat `"fused"` |
| `"torch"` on CUDA | complex128 | Gram ≤ 2e-14 (Kaggle T4) | large batches at 10+ qubits |
| `"torch"` on Apple MPS | complex64 (MPS has no 64-bit floats) | not measured; Kaggle has no Apple hardware | speed where single precision is enough |

Kernel values that close still don't promise that every decision matches,
since an SVM near a tie can tip either way on a 1e-14 difference. Use the
default when you need to reproduce published numbers.

A row's state doesn't depend on which other rows share its batch. The fused
simulator is batch-invariant by construction (tested on CPU, checked on a T4),
so the same prepared angles give the same state bits in `fit`,
cross-validation and `predict`. Whitened modes are the exception on every
simulator, because they compute those angles with a matrix product whose last
bit can depend on the batch. Fitted models pickle and copy like any other, and
a circuit the fused simulator can't lower falls back to qiskit with one warning
rather than an error. The test suite passes on Kaggle's Linux CPU and T4
machines; nobody has run it on Windows yet.

## A few caveats

The quantum kernels and the VQE run on simulators whose cost grows with the
number of circuit evaluations, so keep datasets and qubit counts small. And the
chemistry module works from a model Hamiltonian derived from the geometry, not
a full electronic-structure calculation.

`n_qubits` has to equal the number of input features, because the feature map
encodes one angle per feature. A mismatch raises and tells you both numbers.

Some constructor arguments are declarative and don't resize anything.
`AdvancedQIGASelector(n_features=...)` doesn't ask for a subset of that size
(the selector keeps whatever scores best), and `FederatedQML(num_clients=...)`
and `AdvancedQuantumChemistryLayer(num_atoms=...)` take their size from the
data you pass. All of them warn when what you declared disagrees with what
arrived.

Torch, scikit-learn and xgboost each vendor their own OpenMP runtime, and more
than one thread pool in a process can kill the interpreter without a
traceback. `import qmlf` sets `OMP_NUM_THREADS=1` unless you've already chosen
a value.

## License

See [LICENSE](LICENSE).
