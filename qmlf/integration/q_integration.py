import numpy as np
import torch

from torch import nn

from qmlf.ops.q_ops_core import QuantumKernel
from qmlf.qnn.qnn_layers import (
    create_advanced_qnn_layer
)
from qmlf.viz.q_viz_pro import QVizPro


class QMLFPipeline(nn.Module):
    """End-to-end pipeline: classical encoding -> quantum QNN + quantum kernel.

    The input is projected to ``n_qubits`` features, run through the variational
    quantum neural network, and combined with a quantum-kernel similarity to a
    fixed set of landmark points. Both feature branches are per-sample (no
    cross-batch coupling), so the output of one sample does not depend on its
    batch-mates.
    """

    def __init__(
        self,
        input_dim,
        n_qubits=8,
        reps=5,
        output_dim=16
    ):
        super().__init__()

        self.input_dim = input_dim
        self.n_qubits = n_qubits

        self.quantum_kernel = QuantumKernel(
            n_qubits=n_qubits,
            mode="ZZ",
            reps=reps
        )

        # Fixed landmark points used for the quantum-kernel similarity features.
        self.landmarks = np.random.uniform(
            -1.0,
            1.0,
            size=(n_qubits, n_qubits)
        )

        self.input_projection = nn.Sequential(
            nn.Linear(
                input_dim,
                n_qubits
            ),
            nn.Tanh()
        )

        self.kernel_projection = nn.Sequential(
            nn.Linear(
                n_qubits,
                n_qubits
            ),
            nn.LayerNorm(
                n_qubits
            ),
            nn.ReLU()
        )

        self.qnn_layer = (
            create_advanced_qnn_layer(
                n_qubits=n_qubits,
                reps=reps,
                output_dim=output_dim
            )
        )

        self.feature_fusion = nn.Sequential(
            nn.Linear(
                output_dim + n_qubits,
                64
            ),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.final_head = nn.Sequential(
            nn.Linear(
                64,
                64
            ),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(
                64,
                output_dim
            )
        )

    def forward(self, x):
        if isinstance(
            x,
            torch.Tensor
        ):
            x_tensor = x
        else:
            x_tensor = torch.tensor(
                np.asarray(x),
                dtype=torch.float32
            )

        if x_tensor.dim() != 2:
            raise ValueError(
                f"Expected 2D (batch, input_dim) input, got a "
                f"{x_tensor.dim()}D tensor of shape {tuple(x_tensor.shape)}"
            )

        if x_tensor.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim} features, got "
                f"{x_tensor.shape[-1]}"
            )

        encoded = self.input_projection(
            x_tensor
        )

        qnn_features = self.qnn_layer(
            encoded
        )

        # Per-sample quantum-kernel similarity to the fixed landmarks.
        encoded_np = (
            encoded.detach()
            .cpu()
            .numpy()
        )

        kernel_matrix = (
            self.quantum_kernel
            .compute_kernel_matrix(
                encoded_np,
                self.landmarks
            )
        )

        kernel_features = torch.tensor(
            kernel_matrix,
            dtype=torch.float32,
            device=x_tensor.device
        )

        kernel_features = (
            self.kernel_projection(
                kernel_features
            )
        )

        fused = torch.cat(
            [
                qnn_features,
                kernel_features
            ],
            dim=1
        )

        fused = self.feature_fusion(
            fused
        )

        output = self.final_head(
            fused
        )

        return output

    def compute_kernel_features(
        self,
        x
    ):
        return (
            self.quantum_kernel
            .compute_kernel_matrix(x)
        )

    def visualize_pipeline(
        self,
        kernel_matrix,
        labels=None
    ):
        return (
            QVizPro.plot_hilbert_space(
                kernel_matrix,
                labels=labels
            )
        )

    def get_quantum_kernel(
        self
    ):
        return self.quantum_kernel

    def get_qnn_layer(
        self
    ):
        return self.qnn_layer


def create_full_pipeline(
    input_dim,
    n_qubits=8,
    reps=5,
    output_dim=16
):
    return QMLFPipeline(
        input_dim=input_dim,
        n_qubits=n_qubits,
        reps=reps,
        output_dim=output_dim
    )
