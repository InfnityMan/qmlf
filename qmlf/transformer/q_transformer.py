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


class AdvancedQuantumTransformerLayer(nn.Module):
    """Transformer block whose feed-forward stage passes through a quantum layer.

    ``forward`` expects a 3D ``(batch, seq_len, embed_dim)`` tensor.

    ``precision`` is the sampling precision of the underlying ``EstimatorQNN``:
    ``0.0`` (default) reads the observables exactly off the statevector and is
    reproducible; a positive value emulates a finite-shot device and makes every
    forward pass differ. See :class:`~qmlf.qnn.qnn_layers.AdvancedQuantumNNLayer`
    for the full note.
    """

    def __init__(
        self,
        n_qubits=8,
        heads=4,
        reps=3,
        embed_dim=32,
        *,
        precision=0.0
    ):
        super().__init__()

        if precision < 0:
            raise ValueError(f"precision must be non-negative, got {precision}")

        self.n_qubits = n_qubits
        self.heads = heads
        self.reps = reps
        self.embed_dim = embed_dim
        self.precision = precision

        self.input_projection = nn.Linear(
            embed_dim,
            n_qubits
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=n_qubits,
            num_heads=heads,
            batch_first=True
        )

        self.feature_map = zz_feature_map(
            feature_dimension=n_qubits,
            reps=1
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
            input_gradients=True,
            default_precision=self.precision
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

        self.norm1 = nn.LayerNorm(
            n_qubits
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(
                n_qubits,
                n_qubits * 4
            ),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(
                n_qubits * 4,
                n_qubits
            )
        )

        self.norm2 = nn.LayerNorm(
            n_qubits
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
        # Without this the reshape below fails with a bare
        # "shape '[N, n_qubits]' is invalid for input of size M", which says
        # nothing about the rank the layer actually wants.
        if x.dim() != 3:
            raise ValueError(
                f"Expected a 3D (batch, seq_len, embed_dim) tensor, got a "
                f"{x.dim()}D tensor of shape {tuple(x.shape)}. For a single "
                f"sequence use x.unsqueeze(0)."
            )

        if x.shape[-1] != self.embed_dim:
            raise ValueError(
                f"Expected embed_dim={self.embed_dim} in the last dimension, "
                f"got {x.shape[-1]}"
            )

        projected = self.input_projection(
            x
        )

        attention_out, _ = self.attention(
            projected,
            projected,
            projected
        )

        x1 = self.norm1(
            projected + attention_out
        )

        batch_size = x1.shape[0]
        seq_len = x1.shape[1]

        quantum_input = x1.reshape(
            batch_size * seq_len,
            self.n_qubits
        )

        quantum_output = self.quantum_layer(
            quantum_input
        )

        quantum_output = quantum_output.reshape(
            batch_size,
            seq_len,
            self.n_qubits
        )

        x2 = x1 + quantum_output

        ff_output = self.feed_forward(
            x2
        )

        output = self.norm2(
            x2 + ff_output
        )

        return output

    def get_quantum_circuit(self):
        return self.circuit

    def get_num_parameters(self):
        return self.qnn.num_weights


def create_advanced_quantum_transformer(
    n_qubits=8,
    heads=4,
    reps=3,
    embed_dim=32,
    *,
    precision=0.0
):
    return AdvancedQuantumTransformerLayer(
        n_qubits=n_qubits,
        heads=heads,
        reps=reps,
        embed_dim=embed_dim,
        precision=precision
    )
