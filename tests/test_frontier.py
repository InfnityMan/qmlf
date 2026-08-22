"""Tests for the advantage screen, Nystrom path, alignment bandwidth, and the
regressor -- all verified absent from qiskit-machine-learning 0.9 before
implementation (zero source references to geometric difference, Nystrom,
landmarks, or any bandwidth machinery; QSVR exists but with no tuning at all).

Run with:  python -m pytest tests/ -q
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.metrics import r2_score

from qmlf import (
    QuantumKernel,
    QuantumRegressor,
    geometric_difference,
    model_complexity,
    quantum_advantage_report,
)


@pytest.fixture(scope="module")
def X():
    return np.random.default_rng(0).normal(size=(20, 4))


@pytest.fixture(scope="module")
def y(X):
    return (X[:, 0] > 0).astype(int)


# --------------------------------------------------------------------------
# bandwidth="alignment"
# --------------------------------------------------------------------------

def test_alignment_bandwidth_fits_from_labels(X, y):
    kernel = QuantumKernel(bandwidth="alignment").fit(X, y)

    assert kernel.bandwidth_ in (1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01)


def test_alignment_bandwidth_is_deterministic(X, y):
    a = QuantumKernel(bandwidth="alignment").fit(X, y)
    b = QuantumKernel(bandwidth="alignment").fit(X, y)

    assert a.bandwidth_ == b.bandwidth_
    assert np.array_equal(a.compute_kernel_matrix(X), b.compute_kernel_matrix(X))


def test_alignment_bandwidth_requires_labels(X):
    with pytest.raises(ValueError, match="needs labels"):
        QuantumKernel(bandwidth="alignment").fit(X)


def test_alignment_bandwidth_rejects_regression_targets(X):
    with pytest.raises(ValueError, match="regression"):
        QuantumKernel(bandwidth="alignment").fit(X, np.arange(len(X), dtype=float))


def test_alignment_works_with_projected_kernel(X, y):
    kernel = QuantumKernel(bandwidth="alignment", kernel="projected").fit(X, y)

    assert kernel.bandwidth_ > 0
    assert kernel.compute_kernel_matrix(X).shape == (20, 20)


# --------------------------------------------------------------------------
# Nystrom features + circuit budget
# --------------------------------------------------------------------------

def test_nystrom_with_all_landmarks_is_exact(X):
    kernel = QuantumKernel(bandwidth=0.1).fit(X)
    exact = kernel.compute_kernel_matrix(X)
    features = kernel.nystrom_features(X, n_landmarks=len(X))

    assert np.allclose(features @ features.T, exact, atol=1e-8)


def test_nystrom_preserves_downstream_accuracy(X, y):
    """The honest contract: approximation quality is judged by the model it
    feeds, not by elementwise error on a full-rank Gram matrix."""
    from sklearn.svm import SVC, LinearSVC

    kernel = QuantumKernel(bandwidth=0.05).fit(X)

    exact_acc = SVC(kernel="precomputed").fit(
        kernel.compute_kernel_matrix(X), y
    ).score(kernel.compute_kernel_matrix(X), y)

    features = kernel.nystrom_features(X, n_landmarks=10)
    approx_acc = LinearSVC(dual="auto").fit(features, y).score(features, y)

    assert approx_acc >= exact_acc - 0.15


def test_nystrom_explicit_landmark_indices(X):
    kernel = QuantumKernel(bandwidth=0.1).fit(X)
    features = kernel.nystrom_features(X, landmark_indices=[0, 5, 10, 15])

    assert features.shape == (20, 4)


def test_nystrom_is_deterministic(X):
    kernel = QuantumKernel(bandwidth=0.1).fit(X)

    assert np.array_equal(
        kernel.nystrom_features(X, n_landmarks=8),
        kernel.nystrom_features(X, n_landmarks=8),
    )


def test_nystrom_validates_landmark_count(X):
    kernel = QuantumKernel(bandwidth=0.1).fit(X)

    with pytest.raises(ValueError, match="n_landmarks"):
        kernel.nystrom_features(X, n_landmarks=0)

    with pytest.raises(ValueError, match="n_landmarks"):
        kernel.nystrom_features(X)


def test_circuit_budget_arithmetic():
    budget = QuantumKernel(4).circuit_budget(200, n_landmarks=20)

    assert budget["pairwise_fidelity_circuits"] == 200 * 199 // 2
    assert budget["statevector_simulations"] == 200
    assert budget["nystrom_fidelity_circuits"] == 200 * 20 + 20 * 19 // 2
    # the point of the whole exercise:
    assert budget["nystrom_fidelity_circuits"] < budget["pairwise_fidelity_circuits"] / 4


# --------------------------------------------------------------------------
# Geometric difference / advantage screen (Huang et al. 2021)
# --------------------------------------------------------------------------

def test_identical_kernels_give_g_of_one():
    gram = np.eye(10) + 0.5

    assert geometric_difference(gram, gram) == pytest.approx(1.0, abs=1e-3)


def test_geometric_difference_validates_shapes():
    with pytest.raises(ValueError, match="square"):
        geometric_difference(np.eye(4), np.eye(5))


def test_model_complexity_orders_easy_before_hard(X):
    """Labels aligned with the kernel's top eigendirection must cost less
    than random labels."""
    gram = QuantumKernel(bandwidth=0.05).fit(X).compute_kernel_matrix(X)

    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    easy = np.sign(eigenvectors[:, -1])
    hard = np.where(np.arange(len(X)) % 2 == 0, 1.0, -1.0)

    assert model_complexity(gram, easy) < model_complexity(gram, hard)


def test_advantage_report_structure_and_determinism(X, y):
    a = quantum_advantage_report(X, y)
    b = quantum_advantage_report(X, y)

    for key in ("geometric_difference", "s_classical", "s_quantum",
                "verdict", "g_matchable_below", "g_advantage_scale"):
        assert key in a

    assert a["geometric_difference"] == b["geometric_difference"]
    assert a["verdict"] == b["verdict"]
    assert a["geometric_difference"] > 0


def test_advantage_report_without_labels_skips_complexities(X):
    report = quantum_advantage_report(X)

    assert report["s_classical"] is None
    assert report["s_quantum"] is None
    assert "verdict" in report


# --------------------------------------------------------------------------
# QuantumRegressor
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def reg_data():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (40, 4))
    y = X[:, 0] + 0.5 * X[:, 1] ** 2 + 0.05 * rng.normal(size=40)
    return X[:30], X[30:], y[:30], y[30:]


def test_regressor_learns_a_smooth_target(reg_data):
    X_train, X_test, y_train, y_test = reg_data

    reg = QuantumRegressor(bandwidths=(0.5, 0.25, 0.1)).fit(X_train, y_train)

    assert r2_score(y_test, reg.predict(X_test)) >= 0.8
    assert reg.cv_results_ is not None and "mean_cv_mse" in reg.cv_results_


def test_regressor_is_deterministic(reg_data):
    X_train, X_test, y_train, _ = reg_data

    a = QuantumRegressor(bandwidths=(0.5, 0.1)).fit(X_train, y_train)
    b = QuantumRegressor(bandwidths=(0.5, 0.1)).fit(X_train, y_train)

    assert np.array_equal(a.predict(X_test), b.predict(X_test))


def test_regressor_median_bandwidth(reg_data):
    X_train, X_test, y_train, _ = reg_data

    reg = QuantumRegressor(bandwidth="median").fit(X_train, y_train)

    assert reg.bandwidth_ > 0
    assert reg.cv_results_ is None
    assert reg.predict(X_test).shape == (10,)


def test_regressor_projected_kernel(reg_data):
    X_train, X_test, y_train, y_test = reg_data

    reg = QuantumRegressor(kernel="projected", bandwidth=0.25)
    reg.fit(X_train, y_train)

    assert reg.kernel_ == "projected"
    assert np.isfinite(reg.predict(X_test)).all()


def test_regressor_rejects_classification_only_options(reg_data):
    X_train, _, y_train, _ = reg_data

    with pytest.raises(ValueError, match="classification-only"):
        QuantumRegressor(mode="fisher").fit(X_train, y_train)

    with pytest.raises(ValueError, match="classification-only"):
        QuantumRegressor(bandwidth="alignment").fit(X_train, y_train)


def test_regressor_reduces_wide_data():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(24, 20))
    y = X[:, 0]

    reg = QuantumRegressor(bandwidth=0.25, cv=2).fit(X, y)

    assert reg.n_qubits_ == 8
    assert reg.reduction_ == "pca"


def test_regressor_sklearn_compatibility(reg_data):
    X_train, _, y_train, _ = reg_data

    reg = clone(QuantumRegressor(bandwidth=0.25, alpha=1e-2))
    assert reg.get_params()["alpha"] == 1e-2

    reg.fit(X_train, y_train)
    assert hasattr(reg, "coef_")


def test_regressor_predict_before_fit_errors():
    with pytest.raises(ValueError, match="not been fitted"):
        QuantumRegressor().predict(np.zeros((2, 4)))
