import numpy as np
import torch

from qmlf.ops.q_ops_core import QuantumKernel
from qmlf.qnn.qnn_layers import create_advanced_qnn_layer
from qmlf.viz.q_viz_pro import QVizPro


class QuantumPipeline:
    """One-line quantum workflow: kernel -> QNN -> visualization.

    Wires a :class:`QuantumKernel`, an advanced variational quantum
    neural-network layer and the :class:`QVizPro` plotting helpers together so a
    full quantum feature pipeline can be run and inspected in a single call.

    The number of qubits is inferred from the data when it is not given, since
    both the ZZ feature map and the QNN feature map read one angle per input
    feature.
    """

    def __init__(self, n_qubits=None, mode="ZZ", reps=2, output_dim=16):
        self.n_qubits = n_qubits
        self.mode = mode
        self.reps = reps
        self.output_dim = output_dim

        self.kernel = None
        self.qnn_layer = None

        self.kernel_matrix_ = None
        self.qnn_output_ = None

    def _build(self, n_features):
        n_qubits = self.n_qubits if self.n_qubits is not None else n_features
        self.n_qubits = n_qubits

        self.kernel = QuantumKernel(
            n_qubits=n_qubits,
            mode=self.mode,
            reps=self.reps
        )

        self.qnn_layer = create_advanced_qnn_layer(
            n_qubits=n_qubits,
            reps=self.reps,
            output_dim=self.output_dim
        )

    def run(self, X, labels=None, visualize=True, show=True, save_path=None):
        X_np = np.asarray(X, dtype=float)

        if X_np.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features)")

        if self.kernel is None or self.qnn_layer is None:
            self._build(X_np.shape[1])

        if X_np.shape[1] != self.n_qubits:
            raise ValueError(
                f"Expected input with {self.n_qubits} features, got {X_np.shape[1]}"
            )

        # 1. quantum fidelity kernel over the (optionally covariance-adapted) data.
        self.kernel.fit(X_np)
        self.kernel_matrix_ = self.kernel.compute_kernel_matrix(X_np)

        # 2. per-qubit Z read-out from the variational quantum neural network.
        x_tensor = torch.tensor(X_np, dtype=torch.float32)

        with torch.no_grad():
            self.qnn_output_ = self.qnn_layer(x_tensor).detach().cpu().numpy()

        # 3. project the kernel into Hilbert space for inspection.
        figure = None

        if visualize:
            figure = QVizPro.plot_hilbert_space(
                self.kernel_matrix_,
                labels=labels,
                show=show,
                save_path=save_path
            )

        return {
            "kernel_matrix": self.kernel_matrix_,
            "qnn_output": self.qnn_output_,
            "figure": figure
        }

    def get_quantum_kernel(self):
        return self.kernel

    def get_qnn_layer(self):
        return self.qnn_layer


def create_quantum_pipeline(n_qubits=None, mode="ZZ", reps=2, output_dim=16):
    return QuantumPipeline(
        n_qubits=n_qubits,
        mode=mode,
        reps=reps,
        output_dim=output_dim
    )
