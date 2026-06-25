# qmlf

A quantum machine learning framework built on Qiskit, PyTorch and
scikit-learn. It pulls together a few building blocks I keep reaching for:
quantum kernels, a trainable quantum neural network layer, a couple of hybrid
models, plus some tools for noise mitigation, circuit optimization, feature
selection and plotting.


## Requirements

Python 3.9+ and a recent Qiskit (2.4 or newer). Install with:

    pip install -r requirements.txt

It uses the current Qiskit primitives (`StatevectorEstimator`) and the
`zz_feature_map` / `real_amplitudes` circuit builders, so it runs on Qiskit 2.x
without any of the old V1 shims.

## Quick start

Build a quantum fidelity kernel:

```python
import numpy as np
from qmlf import QuantumKernel

X = np.random.rand(20, 4)
kernel = QuantumKernel(n_qubits=4, mode="ZZ").fit(X)
gram = kernel.compute_kernel_matrix(X)   # 20 x 20 kernel matrix
```

Drop a trainable quantum layer into a torch model:

```python
import torch
from qmlf import create_advanced_qnn_layer

layer = create_advanced_qnn_layer(n_qubits=4, output_dim=8)
y = layer(torch.randn(16, 4))            # (16, 8)
```

## What's included

* `ops`: quantum fidelity kernel over a ZZ feature map, plus a QSVC vs. XGBoost benchmark.
* `graph`: graph-diffusion kernel, with an optional quantum kernel backend (`use_quantum=True`).
* `qnn`, `hybrid`, `transformer`: torch layers wrapping a real `EstimatorQNN`, read out from per-qubit Z expectation values.
* `chem`: charge-aware Coulomb-matrix descriptors, a model Hamiltonian, classical ground-state energy, and an optional VQE.
* `nisq`: transpiler-based circuit depth and gate reduction.
* `noise`: readout calibration, depolarizing-channel inversion, and zero-noise extrapolation.
* `optimizers`: quantum-inspired evolutionary feature selection.
* `federated`: sample-weighted federated averaging.
* `viz`: Plotly charts (kernels, Bloch sphere, training curves, barren-plateau diagnostics).

## A few caveats

The quantum kernels and the VQE run on simulators, and the cost grows with the
number of circuit evaluations, so keep datasets and qubit counts small. The
chemistry module works from a geometry-derived model Hamiltonian, not a full
electronic-structure calculation.

## License

See [LICENSE](LICENSE).
