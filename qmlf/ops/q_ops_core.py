import numpy as np
import pandas as pd
from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb
from qiskit_machine_learning.algorithms import QSVC
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.graph_objects as go


class QuantumKernel:
    def __init__(self, n_qubits, mode="ZZ", reps=2, shrinkage=1e-3):
        self.n_qubits = n_qubits
        self.mode = mode
        self.reps = reps
        self.shrinkage = shrinkage
        self.mean_ = None
        self.cov_matrix = None
        self.whitening_matrix = None
        self.X_train_ = None

        if mode == "ZZ" or mode == "covariant":
            self.feature_map = zz_feature_map(feature_dimension=n_qubits, reps=reps)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.fidelity_quantum_kernel = FidelityQuantumKernel(feature_map=self.feature_map)

    def fit(self, X_train, y=None):
        X_np = np.asarray(X_train, dtype=float)
        self.X_train_ = X_np

        if self.mode == "covariant":
            self.mean_ = X_np.mean(axis=0)
            centered = X_np - self.mean_
            self.cov_matrix = np.cov(centered, rowvar=False)
            self.whitening_matrix = self._compute_whitening(self.cov_matrix)

        return self

    def _compute_whitening(self, cov_matrix):
        # Mahalanobis whitening from the training-data covariance. The covariance
        # is first shrunk toward a scaled identity (Ledoit-Wolf-style) so the
        # inverse square root stays well-conditioned even when features are
        # collinear, then W = V diag(1/sqrt(lambda)) V^T. Mapping the centered
        # data through W gives it identity covariance, so a Euclidean distance in
        # the transformed space is exactly the Mahalanobis distance of the raw
        # data, d(x, y) = sqrt((x - y)^T Sigma^-1 (x - y)).
        cov_matrix = np.atleast_2d(cov_matrix)
        dim = cov_matrix.shape[0]

        target = np.trace(cov_matrix) / dim
        shrunk = (1.0 - self.shrinkage) * cov_matrix + self.shrinkage * target * np.eye(dim)

        eigenvalues, eigenvectors = np.linalg.eigh(shrunk)
        inv_sqrt = 1.0 / np.sqrt(np.maximum(eigenvalues, 0) + 1e-8)
        whitening = eigenvectors @ np.diag(inv_sqrt) @ eigenvectors.T

        return whitening

    def _prepare(self, X):
        X_np = np.asarray(X, dtype=float)

        if self.mode == "covariant":
            if self.whitening_matrix is None:
                raise ValueError("Must call .fit() before using covariant mode")
            return (X_np - self.mean_) @ self.whitening_matrix

        return X_np

    def compute_kernel_matrix(self, X, X2=None):
        # ZZ mode: plain quantum fidelity kernel.
        # covariant mode: covariance-adapted (whitened) quantum fidelity kernel.
        X_prepared = self._prepare(X)

        if X2 is None:
            return self.fidelity_quantum_kernel.evaluate(X_prepared)

        X2_prepared = self._prepare(X2)

        return self.fidelity_quantum_kernel.evaluate(X_prepared, X2_prepared)


def plot_hilbert_space(kernel_matrix, labels=None):
    kernel_matrix = np.asarray(kernel_matrix, dtype=float)
    n_samples = kernel_matrix.shape[0]

    if n_samples > 3:
        perplexity = min(30, max(2, n_samples - 1))
        tsne = TSNE(n_components=3, random_state=42, perplexity=perplexity)
        embedding = tsne.fit_transform(kernel_matrix)
    else:
        pca = PCA(n_components=min(3, n_samples))
        embedding = pca.fit_transform(kernel_matrix)
        if embedding.shape[1] < 3:
            padded = np.zeros((embedding.shape[0], 3))
            padded[:, :embedding.shape[1]] = embedding
            embedding = padded

    fig = go.Figure(data=[go.Scatter3d(
        x=embedding[:, 0],
        y=embedding[:, 1],
        z=embedding[:, 2],
        mode='markers',
        marker=dict(
            size=8,
            color=labels if labels is not None else embedding[:, 0],
            colorscale='Viridis',
            opacity=0.8
        )
    )])

    fig.update_layout(title="Quantum Hilbert Space Visualization")
    fig.show()
    return fig


def run_quantum_benchmark(X, y, n_qubits=4, reps=2, test_size=0.2):
    X = np.asarray(X, dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    xgb_model = xgb.XGBClassifier(eval_metric="logloss")
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)

    # The ZZ feature map dimension must equal the number of input features,
    # so the quantum kernel is built over all features (n_qubits is a hint).
    n_features = X.shape[1]
    quantum_kernel = QuantumKernel(n_qubits=n_features, mode="ZZ", reps=reps)
    quantum_kernel.fit(X_train)

    qsvc = QSVC(quantum_kernel=quantum_kernel.fidelity_quantum_kernel)
    qsvc.fit(X_train, y_train)
    y_pred_qsvc = qsvc.predict(X_test)

    results = {
        "Model": ["XGBoost", "Quantum SVC"],
        "Accuracy": [
            accuracy_score(y_test, y_pred_xgb),
            accuracy_score(y_test, y_pred_qsvc)
        ],
        "F1 Score": [
            f1_score(y_test, y_pred_xgb, average='weighted'),
            f1_score(y_test, y_pred_qsvc, average='weighted')
        ]
    }

    return pd.DataFrame(results)
