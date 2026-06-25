import numpy as np
import torch
from torch import nn

from qiskit.circuit.library import zz_feature_map
from qiskit.circuit.library import real_amplitudes

from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp

from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

from qiskit_algorithms.gradients import (
    ParamShiftEstimatorGradient
)


class AdvancedHybridLayer(nn.Module):
    def __init__(
        self,
        input_dim=24,
        n_qubits=8,
        reps=4,
        output_dim=32
    ):
        super().__init__()

        self.input_dim = input_dim
        self.n_qubits = n_qubits
        self.reps = reps
        self.output_dim = output_dim

        self.preprocessor = nn.Sequential(
            nn.Linear(input_dim, 48),
            nn.LayerNorm(48),
            nn.ReLU(),
            nn.Linear(48, n_qubits),
            nn.Tanh()
        )

        self.feature_map = zz_feature_map(
            feature_dimension=n_qubits,
            reps=2
        )

        self.ansatz = real_amplitudes(
            num_qubits=n_qubits,
            reps=reps
        )

        self.circuit = self.feature_map.compose(
            self.ansatz
        )

        self.estimator = StatevectorEstimator()

        self.gradient = (
            ParamShiftEstimatorGradient(
                self.estimator
            )
        )

        self.observables = self._create_observables()

        self.qnn = EstimatorQNN(
            circuit=self.circuit,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            observables=self.observables,
            estimator=self.estimator,
            gradient=self.gradient,
            input_gradients=True
        )

        initial_weights = np.random.normal(
            0,
            0.1,
            self.qnn.num_weights
        )

        self.quantum_layer = TorchConnector(
            self.qnn,
            initial_weights=initial_weights
        )

        self.postprocessor = nn.Sequential(
            nn.Linear(n_qubits, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, output_dim),
            nn.LayerNorm(output_dim)
        )

    def _create_observables(self):
        observables = []

        for i in range(self.n_qubits):
            pauli = ["I"] * self.n_qubits
            pauli[self.n_qubits - i - 1] = "Z"

            observables.append(
                SparsePauliOp.from_list([("".join(pauli), 1.0)])
            )

        return observables

    def forward(self, x):
        encoded = self.preprocessor(x)

        quantum_output = self.quantum_layer(
            encoded
        )

        return self.postprocessor(
            quantum_output
        )

    def get_quantum_circuit(self):
        return self.circuit

    def get_num_parameters(self):
        return self.qnn.num_weights

    def get_quantum_feature_dimension(self):
        return self.n_qubits


def create_advanced_hybrid_layer(
    input_dim=24,
    n_qubits=8,
    reps=4,
    output_dim=32
):
    return AdvancedHybridLayer(
        input_dim=input_dim,
        n_qubits=n_qubits,
        reps=reps,
        output_dim=output_dim
    )
