import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA

from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.kernels import FidelityQuantumKernel


class AdvancedQuantumGraphKernel(
    BaseEstimator,
    TransformerMixin
):
    """A graph-diffusion kernel with an optional quantum backend.

    By default the base similarity is a classical RBF kernel, structured by a
    kNN graph-diffusion operator (fast and practical for V1). When
    ``use_quantum`` is True the base similarity is instead a quantum fidelity
    kernel (ZZ feature map) on PCA-reduced node features, still structured by the
    same graph diffusion. The quantum backend evaluates O(n^2) circuits, so it
    is opt-in.
    """

    def __init__(
        self,
        n_qubits=8,
        kernel_type="zz",
        n_neighbors=5,
        gamma=1.0,
        diffusion_steps=3,
        use_quantum=False
    ):
        self.n_qubits = n_qubits
        self.kernel_type = kernel_type

        self.n_neighbors = n_neighbors
        self.gamma = gamma
        self.diffusion_steps = diffusion_steps
        self.use_quantum = use_quantum

        self.kernel_matrix = None
        self.adjacency_matrix = None
        self.diffused_graph = None

        self.pca = None
        self.feature_map = None
        self.quantum_kernel = None
        self.X_train_ = None

    def _build_graph(self, X):
        adjacency = kneighbors_graph(
            X,
            n_neighbors=self.n_neighbors,
            mode="connectivity",
            include_self=True
        )

        return adjacency.toarray().astype(float)

    def _graph_diffusion(self, adjacency):
        degree = np.diag(
            adjacency.sum(axis=1)
        )

        degree_inv = np.linalg.pinv(degree)

        transition = (
            degree_inv @ adjacency
        )

        diffusion = np.eye(
            adjacency.shape[0]
        )

        current = np.eye(
            adjacency.shape[0]
        )

        for _ in range(
            self.diffusion_steps
        ):
            current = (
                current @ transition
            )

            diffusion += current

        diffusion /= (
            self.diffusion_steps + 1
        )

        return diffusion

    def _reduce_features(self, X):
        X_np = np.asarray(X, dtype=float)

        if self.pca is not None:
            reduced = self.pca.transform(X_np)
        else:
            reduced = X_np

        if reduced.shape[1] < self.n_qubits:
            padded = np.zeros(
                (reduced.shape[0], self.n_qubits)
            )

            padded[:, :reduced.shape[1]] = reduced

            return padded

        return reduced

    def _base_kernel(self, X, X2=None):
        if self.use_quantum:
            X_reduced = self._reduce_features(X)

            if X2 is None:
                return self.quantum_kernel.evaluate(X_reduced)

            X2_reduced = self._reduce_features(X2)

            return self.quantum_kernel.evaluate(X_reduced, X2_reduced)

        if X2 is None:
            return rbf_kernel(X, gamma=self.gamma)

        return rbf_kernel(X, X2, gamma=self.gamma)

    def fit(self, X, y=None):
        X_np = np.asarray(X, dtype=float)

        if self.kernel_type != "zz":
            raise ValueError(
                f"Unknown kernel_type: {self.kernel_type}"
            )

        self.X_train_ = X_np

        # include_self=True counts the sample itself as one of its neighbours,
        # so the ceiling is n_samples, not n_samples - 1. Checked here because
        # sklearn's own error surfaces from inside kneighbors and does not name
        # the constructor argument that caused it.
        if self.n_neighbors > X_np.shape[0]:
            raise ValueError(
                f"n_neighbors={self.n_neighbors} but X has only "
                f"{X_np.shape[0]} samples; the kNN graph needs "
                f"n_neighbors <= n_samples (self-loops included)."
            )

        if self.use_quantum:
            n_components = min(self.n_qubits, X_np.shape[1])

            self.pca = PCA(n_components=n_components)
            self.pca.fit(X_np)

            self.feature_map = zz_feature_map(
                feature_dimension=self.n_qubits,
                reps=2
            )

            self.quantum_kernel = FidelityQuantumKernel(
                feature_map=self.feature_map
            )

        adjacency = self._build_graph(
            X_np
        )

        diffusion = self._graph_diffusion(
            adjacency
        )

        base_kernel = self._base_kernel(
            X_np
        )

        graph_kernel = (
            diffusion
            @ base_kernel
            @ diffusion.T
        )

        self.adjacency_matrix = adjacency
        self.diffused_graph = diffusion
        self.kernel_matrix = graph_kernel

        return self

    def transform(self, X):
        if self.kernel_matrix is None:
            raise ValueError(
                "Kernel has not been fitted yet"
            )

        X_np = np.asarray(X, dtype=float)

        # Out-of-sample similarity of X to the training nodes (the diffusion
        # operator is defined on the training graph).
        return self._base_kernel(
            X_np,
            self.X_train_
        )

    def fit_transform(self, X, y=None):
        return self.fit(
            X,
            y
        ).transform(X)

    def get_kernel_matrix(self):
        return self.kernel_matrix

    def get_adjacency_matrix(self):
        return self.adjacency_matrix

    def get_diffused_graph(self):
        return self.diffused_graph


def create_advanced_graph_kernel(
    n_qubits=8,
    n_neighbors=5,
    gamma=1.0,
    use_quantum=False
):
    return AdvancedQuantumGraphKernel(
        n_qubits=n_qubits,
        n_neighbors=n_neighbors,
        gamma=gamma,
        use_quantum=use_quantum
    )
