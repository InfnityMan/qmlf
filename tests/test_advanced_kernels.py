"""Capability tests for the kernels qiskit-machine-learning does not have.

Verified against the installed qiskit-ml 0.9 source before implementation:
zero occurrences of projected kernels, reduced density matrices, ARD/
anisotropic scaling, median-heuristic bandwidth, or within-class whitening.
Each capability below therefore exists in qmlf as a built-in and nowhere in
the upstream stack.

Run with:  python -m pytest tests/ -q
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import warnings

import numpy as np
import pytest

from qmlf import QuantumClassifier, QuantumKernel


@pytest.fixture(scope="module")
def X():
    return np.random.default_rng(0).normal(size=(14, 4))


@pytest.fixture(scope="module")
def y(X):
    return (X[:, 0] > 0).astype(int)


# --------------------------------------------------------------------------
# Projected quantum kernel (Huang et al. 2021)
# --------------------------------------------------------------------------

def test_projected_kernel_is_a_valid_kernel(X):
    kernel = QuantumKernel(kernel="projected").fit(X)
    gram = kernel.compute_kernel_matrix(X)

    assert np.allclose(np.diag(gram), 1.0, atol=1e-10)
    assert np.allclose(gram, gram.T, atol=1e-12)
    assert np.linalg.eigvalsh((gram + gram.T) / 2).min() >= -1e-9
    assert kernel.gamma_ > 0  # median heuristic fitted


def test_projected_kernel_resists_concentration():
    """The reason it exists: global fidelity kernels concentrate as qubits
    grow; local RDM kernels do not. Measured at 8 qubits, bandwidth 1.0."""
    X8 = np.random.default_rng(7).normal(size=(20, 8))
    mask = ~np.eye(20, dtype=bool)

    fidelity = QuantumKernel().fit(X8).compute_kernel_matrix(X8)
    projected = QuantumKernel(kernel="projected").fit(X8).compute_kernel_matrix(X8)

    assert fidelity[mask].mean() < 0.02       # measured 0.0036: dead
    assert projected[mask].mean() > 0.20      # measured 0.3558: healthy
    assert projected[mask].mean() > 10 * fidelity[mask].mean()


def test_projected_kernel_is_deterministic_and_rectangular(X):
    a = QuantumKernel(kernel="projected").fit(X)
    b = QuantumKernel(kernel="projected").fit(X)

    assert np.array_equal(a.compute_kernel_matrix(X), b.compute_kernel_matrix(X))
    assert a.compute_kernel_matrix(X[:5], X).shape == (5, 14)
    # batch path agrees with the full path
    assert np.allclose(
        a.compute_kernel_matrix(X, batch_size=4), a.compute_kernel_matrix(X)
    )


def test_projected_rejects_custom_fidelity():
    with pytest.raises(ValueError, match="projected"):
        QuantumKernel(kernel="projected", sampler=object())


def test_projected_gamma_validation():
    with pytest.raises(ValueError, match="gamma"):
        QuantumKernel(kernel="projected", gamma=-1.0)

    kernel = QuantumKernel(kernel="projected", gamma=2.0)
    assert kernel.gamma_ == 2.0


def test_unknown_kernel_rejected():
    with pytest.raises(ValueError, match="Unknown kernel"):
        QuantumKernel(kernel="banana")


# --------------------------------------------------------------------------
# Per-feature (ARD) bandwidth
# --------------------------------------------------------------------------

def test_ard_bandwidth_differs_from_scalar(X):
    ard = QuantumKernel(bandwidth=np.array([0.5, 0.5, 0.01, 0.01])).fit(X) \
        .compute_kernel_matrix(X)
    scalar = QuantumKernel(bandwidth=0.5).fit(X).compute_kernel_matrix(X)

    assert not np.allclose(ard, scalar)


def test_ard_bandwidth_suppresses_a_feature(X):
    """Sending one feature's bandwidth toward zero must converge to the
    kernel computed without that feature's variation."""
    tiny = np.array([0.5, 0.5, 0.5, 1e-9])
    ard = QuantumKernel(bandwidth=tiny).fit(X).compute_kernel_matrix(X)

    X_flat = X.copy()
    X_flat[:, 3] = 0.0
    flat = QuantumKernel(bandwidth=np.array([0.5, 0.5, 0.5, 1.0])).fit(X_flat) \
        .compute_kernel_matrix(X_flat)

    assert np.allclose(ard, flat, atol=1e-6)


def test_ard_bandwidth_wrong_length_raises(X):
    kernel = QuantumKernel(bandwidth=np.array([0.5, 0.5]))

    with pytest.raises(ValueError, match="entries"):
        kernel.fit(X)


def test_ard_bandwidth_rejects_nonpositive():
    with pytest.raises(ValueError, match="positive"):
        QuantumKernel(bandwidth=np.array([0.5, -0.1]))


# --------------------------------------------------------------------------
# Median-heuristic bandwidth
# --------------------------------------------------------------------------

def test_median_bandwidth_is_fitted_and_deterministic(X):
    a = QuantumKernel(bandwidth="median").fit(X)
    b = QuantumKernel(bandwidth="median").fit(X)

    assert a.bandwidth_ == b.bandwidth_ > 0
    assert np.array_equal(a.compute_kernel_matrix(X), b.compute_kernel_matrix(X))


def test_median_bandwidth_decontracts_the_kernel(X):
    """The whole point: the median heuristic lands the angles at a usable
    scale without any cross-validation."""
    mask = ~np.eye(len(X), dtype=bool)

    naive = QuantumKernel(bandwidth=1.0).fit(X).compute_kernel_matrix(X)
    median = QuantumKernel(bandwidth="median").fit(X).compute_kernel_matrix(X)

    assert median[mask].mean() > naive[mask].mean()


def test_median_bandwidth_requires_fit(X):
    with pytest.raises(ValueError, match="fit"):
        QuantumKernel(bandwidth="median").compute_kernel_matrix(X)


def test_bad_bandwidth_string_rejected():
    with pytest.raises(ValueError, match="median"):
        QuantumKernel(bandwidth="widest")


# --------------------------------------------------------------------------
# Fisher (within-class) whitening
# --------------------------------------------------------------------------

def test_fisher_differs_from_mahalanobis(X, y):
    fisher = QuantumKernel(mode="fisher", normalize="maxabs").fit(X, y) \
        .compute_kernel_matrix(X)
    maha = QuantumKernel(mode="mahalanobis", normalize="maxabs").fit(X) \
        .compute_kernel_matrix(X)

    assert not np.allclose(fisher, maha)


def test_fisher_without_labels_is_a_clear_error(X):
    with pytest.raises(ValueError, match="fit\\(X_train, y_train\\)"):
        QuantumKernel(mode="fisher", normalize="maxabs").fit(X)


def test_fisher_all_singleton_classes_rejected(X):
    with pytest.raises(ValueError, match="two or more"):
        QuantumKernel(mode="fisher", normalize="maxabs").fit(X, np.arange(len(X)))


def test_fisher_is_deterministic(X, y):
    a = QuantumKernel(mode="fisher", normalize="maxabs").fit(X, y)
    b = QuantumKernel(mode="fisher", normalize="maxabs").fit(X, y)

    assert np.array_equal(a.compute_kernel_matrix(X), b.compute_kernel_matrix(X))


# --------------------------------------------------------------------------
# Classifier integration
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def clf_data():
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(
        n_samples=80, n_features=6, n_informative=4, n_redundant=0,
        class_sep=1.2, random_state=7
    )
    return train_test_split(X, y, test_size=0.25, random_state=7, stratify=y)


def test_classifier_projected_kernel(clf_data):
    X_train, X_test, y_train, y_test = clf_data

    clf = QuantumClassifier(kernel="projected", bandwidth=0.02)
    clf.fit(X_train, y_train)

    assert clf.kernel_ == "projected"
    assert clf.score(X_test, y_test) >= 0.6


def test_classifier_kernel_auto_searches_both(clf_data):
    X_train, _, y_train, _ = clf_data

    clf = QuantumClassifier(kernel="auto", bandwidths=(0.5, 0.02))
    clf.fit(X_train, y_train)

    assert set(clf.cv_results_["kernel"]) == {"fidelity", "projected"}
    assert clf.kernel_ in ("fidelity", "projected")


@pytest.mark.filterwarnings(
    "ignore:The fitted training kernel is severely concentrated"
)
def test_classifier_median_bandwidth(clf_data):
    X_train, _, y_train, _ = clf_data

    clf = QuantumClassifier(bandwidth="median").fit(X_train, y_train)

    assert clf.bandwidth_ > 0
    assert clf.cv_results_ is None  # no sweep ran


def test_classifier_fisher_mode(clf_data):
    X_train, X_test, y_train, y_test = clf_data

    clf = QuantumClassifier(mode="fisher", bandwidth=0.05).fit(X_train, y_train)

    assert clf.score(X_test, y_test) >= 0.6


def test_classifier_warns_on_concentrated_fit(clf_data):
    X_train, _, y_train, _ = clf_data

    with pytest.warns(UserWarning, match="concentrated"):
        QuantumClassifier(bandwidth=1.0).fit(X_train, y_train)
