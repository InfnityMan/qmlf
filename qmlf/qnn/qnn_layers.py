import numpy as np
import torch
from torch import nn

from qiskit.circuit.library import zz_feature_map, real_amplitudes
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.gradients import ParamShiftEstimatorGradient


class AdvancedIBIInitializer:
    def __init__(self, num_qubits, reps=5, scale=0.1):
        self.num_qubits = num_qubits
        self.reps = reps
        self.scale = scale

    def initialize_parameters(self, num_params):
        """Identity-Block Initialization: small-angle, near-identity weights whose per-block std shrinks with reps to mitigate barren plateaus."""
        params = np.zeros(num_params)
        block_size = max(1, num_params // self.reps)
        std = self.scale / np.sqrt(self.reps)

        for i in range(self.reps):
            start = i * block_size
            end = min(start + block_size, num_params)

            if start >= num_params:
                break

            params[start:end] += np.random.normal(0, std, end - start)

        return params


class AdvancedQuantumNNLayer(nn.Module):
    """Variational quantum layer with a classical read-out head.

    Precision
    ---------
    ``precision`` is the sampling precision of the underlying ``EstimatorQNN``.

    - ``0.0`` (default) evaluates the observables **exactly** from the
      statevector. Repeated forward passes on the same input return the same
      value, which is what a seeded experiment or an assertion needs.
    - A positive value samples the expectation instead, emulating a finite-shot
      device: the read-out picks up Gaussian noise of roughly that scale and no
      two forward passes agree.

    qiskit-machine-learning defaults this to ``0.015625``, which is why earlier
    releases returned noisy read-outs whether or not that was wanted -- on a
    circuit whose exact expectation value was 0.032, five successive calls
    spanned 0.037. That behaviour is still available, but it is now something
    you ask for rather than the silent default:

        create_advanced_qnn_layer(precision=0.015625)   # pre-1.2.1 behaviour
    """

    def __init__(self, n_qubits=8, reps=5, output_dim=16, *, precision=0.0):
        super().__init__()

        if precision < 0:
            raise ValueError(f"precision must be non-negative, got {precision}")

        self.n_qubits = n_qubits
        self.reps = reps
        self.output_dim = output_dim
        self.precision = precision

        self.feature_map = zz_feature_map(feature_dimension=n_qubits, reps=1)
        self.ansatz = real_amplitudes(num_qubits=n_qubits, reps=reps)
        self.circuit = self.feature_map.compose(self.ansatz)

        self.num_params = len(self.ansatz.parameters)

        self.estimator = StatevectorEstimator()
        self.gradient = ParamShiftEstimatorGradient(self.estimator)

        self.observables = self._create_observables()

        self.qnn = EstimatorQNN(
            circuit=self.circuit,
            input_params=list(self.feature_map.parameters),
            weight_params=list(self.ansatz.parameters),
            observables=self.observables,
            estimator=self.estimator,
            gradient=self.gradient,
            input_gradients=True,
            default_precision=self.precision
        )

        self.initial_weights = AdvancedIBIInitializer(
            self.n_qubits,
            self.reps
        ).initialize_parameters(self.num_params)

        self.quantum_layer = TorchConnector(
            self.qnn,
            initial_weights=self.initial_weights
        )

        self.classical_head = nn.Sequential(
            nn.Linear(n_qubits, 48),
            nn.LayerNorm(48),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(48, output_dim),
            nn.LayerNorm(output_dim)
        )

    def _create_observables(self):
        observables = []

        for i in range(self.n_qubits):
            pauli = ["I"] * self.n_qubits
            pauli[self.n_qubits - i - 1] = "Z"
            observables.append(SparsePauliOp.from_list([("".join(pauli), 1.0)]))

        return observables

    def forward(self, x):
        if x.dim() != 2:
            raise ValueError(
                f"Expected a 2D (batch, n_qubits) tensor, got a {x.dim()}D "
                f"tensor of shape {tuple(x.shape)}. For a single sample use "
                "x.unsqueeze(0)."
            )

        if x.shape[1] != self.n_qubits:
            raise ValueError(
                f"Expected input with {self.n_qubits} features, got {x.shape[1]}"
            )

        quantum_out = self.quantum_layer(x)
        return self.classical_head(quantum_out)


def create_advanced_qnn_layer(n_qubits=8, reps=5, output_dim=16, *, precision=0.0):
    return AdvancedQuantumNNLayer(n_qubits, reps, output_dim, precision=precision)