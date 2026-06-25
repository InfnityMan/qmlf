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
    def __init__(self, n_qubits=8, reps=5, output_dim=16):
        super().__init__()

        self.n_qubits = n_qubits
        self.reps = reps
        self.output_dim = output_dim

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
            input_gradients=True
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
        if x.shape[1] != self.n_qubits:
            raise ValueError(
                f"Expected input with {self.n_qubits} features, got {x.shape[1]}"
            )

        quantum_out = self.quantum_layer(x)
        return self.classical_head(quantum_out)


def create_advanced_qnn_layer(n_qubits=8, reps=5, output_dim=16):
    return AdvancedQuantumNNLayer(n_qubits, reps, output_dim)