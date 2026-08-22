"""Claims tests: qmlf must remove the underneath circuit-building process.

The framework's promise is that a user gets quantum ML without touching
encoding circuits, Gram-matrix orientation, precomputed-SVC wiring, or the
bandwidth pathology of fidelity kernels. Measured before these APIs existed,
the zero-knowledge path required six pieces of insider knowledge and scored
0.40 (chance) on the reference dataset; the tests below pin the promise so it
cannot silently rot.

Run with:  python -m pytest tests/ -q
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import qmlf
from qmlf import QuantumClassifier, QuantumKernel

# Several tests below pin small fixed bandwidths on tiny datasets, which the
# post-fit concentration guard rightly flags; the guard itself has its own
# assertion in test_advanced_kernels.
pytestmark = pytest.mark.filterwarnings(
    "ignore:The fitted training kernel is severely concentrated"
)


@pytest.fixture(scope="module")
def data():
    X, y = make_classification(
        n_samples=80, n_features=6, n_informative=4, n_redundant=0,
        class_sep=1.2, random_state=7
    )
    return train_test_split(X, y, test_size=0.25, random_state=7, stratify=y)


# --------------------------------------------------------------------------
# The three-line claim
# --------------------------------------------------------------------------

def test_three_lines_no_quantum_knowledge_beats_the_old_default_trap(data):
    """fit/predict with zero arguments must clear the 0.40 chance-level score
    the old zero-knowledge path produced on this exact dataset."""
    X_train, X_test, y_train, y_test = data

    clf = QuantumClassifier()
    clf.fit(X_train, y_train)

    assert clf.score(X_test, y_test) >= 0.65


def test_fit_is_deterministic(data):
    X_train, X_test, y_train, _ = data

    a = QuantumClassifier().fit(X_train, y_train)
    b = QuantumClassifier().fit(X_train, y_train)

    assert a.bandwidth_ == b.bandwidth_
    assert np.array_equal(a.predict(X_test), b.predict(X_test))


def test_auto_bandwidth_exposes_its_working(data):
    X_train, _, y_train, _ = data

    clf = QuantumClassifier().fit(X_train, y_train)

    table = clf.cv_results_
    assert len(table["bandwidth"]) == len(table["mean_cv_accuracy"])
    # The winner must actually be the argmax of the table it reports.
    best = table["bandwidth"][int(np.argmax(table["mean_cv_accuracy"]))]
    assert clf.bandwidth_ == best


def test_explicit_bandwidth_skips_the_sweep(data):
    X_train, _, y_train, _ = data

    clf = QuantumClassifier(bandwidth=0.1).fit(X_train, y_train)

    assert clf.bandwidth_ == 0.1
    assert clf.cv_results_ is None


def test_whitened_mode_gets_safe_normalisation_by_default(data):
    """mode='mahalanobis' without normalize used to be a documented
    degenerate configuration; the simple API must not reproduce it."""
    X_train, X_test, y_train, y_test = data

    clf = QuantumClassifier(mode="mahalanobis", bandwidth=0.05)
    clf.fit(X_train, y_train)

    assert clf._resolved_normalize() == "maxabs"
    assert clf.score(X_test, y_test) >= 0.65


def test_multiclass_works(data):
    X_train, _, y_train, _ = data
    y3 = (np.arange(len(y_train)) % 3)  # deterministic 3-class labels

    clf = QuantumClassifier(bandwidth=0.1).fit(X_train, y3)

    assert set(clf.classes_) == {0, 1, 2}
    assert clf.predict(X_train[:5]).shape == (5,)


def test_sklearn_compatibility(data):
    X_train, _, y_train, _ = data

    clf = QuantumClassifier(bandwidth=0.1, C=2.0)
    cloned = clone(clf)  # requires verbatim get_params/set_params

    assert cloned.get_params()["C"] == 2.0
    cloned.fit(X_train, y_train)
    assert hasattr(cloned, "classes_")


def test_predict_before_fit_is_a_clear_error(data):
    _, X_test, _, _ = data

    with pytest.raises(ValueError, match="not been fitted"):
        QuantumClassifier().predict(X_test)


def test_wide_data_is_reduced_automatically():
    """Qubit count is no longer a user decision: 20 features -> max_qubits."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(24, 20))
    y = np.arange(24) % 2

    clf = QuantumClassifier(bandwidth=0.1, cv=2).fit(X, y)

    assert clf.n_qubits_ == 8          # default max_qubits
    assert clf.reduction_ == "pca"
    assert clf.predict(X[:3]).shape == (3,)


def test_wide_data_reduction_is_deterministic():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(24, 20))
    y = np.arange(24) % 2

    a = QuantumClassifier(bandwidth=0.1, cv=2).fit(X, y)
    b = QuantumClassifier(bandwidth=0.1, cv=2).fit(X, y)

    assert np.array_equal(a.predict(X), b.predict(X))


def test_max_qubits_none_disables_reduction_and_warns():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 14))
    y = np.arange(10) % 2

    with pytest.warns(UserWarning, match="max_qubits=None"):
        try:
            QuantumClassifier(bandwidth=0.1, cv=2, max_qubits=None).fit(X, y)
        except Exception:
            pass  # only the warning is under test; 14 qubits may be slow


def test_narrow_data_is_not_reduced(data):
    X_train, _, y_train, _ = data  # 6 features <= max_qubits=8

    clf = QuantumClassifier(bandwidth=0.1).fit(X_train, y_train)

    assert clf.reduction_ is None
    assert clf.n_qubits_ == X_train.shape[1]


def test_encoding_search_selects_mode_and_map(data):
    """mode/feature_map='auto' makes the encoding a searched hyperparameter."""
    X_train, X_test, y_train, y_test = data

    clf = QuantumClassifier(mode="auto", feature_map="auto",
                            bandwidths=(0.5, 0.02))
    clf.fit(X_train, y_train)

    assert clf.mode_ in ("ZZ", "mahalanobis")
    assert clf.feature_map_ in ("zz", "z")
    # 2 modes x 2 maps x 2 bandwidths
    assert len(clf.cv_results_["bandwidth"]) == 8
    winner = int(np.argmax(clf.cv_results_["mean_cv_accuracy"]))
    assert clf.cv_results_["mode"][winner] == clf.mode_
    assert clf.cv_results_["feature_map"][winner] == clf.feature_map_
    assert clf.score(X_test, y_test) >= 0.6


def test_encoding_search_is_deterministic(data):
    X_train, _, y_train, _ = data

    a = QuantumClassifier(mode="auto", feature_map="auto",
                          bandwidths=(0.5, 0.02)).fit(X_train, y_train)
    b = QuantumClassifier(mode="auto", feature_map="auto",
                          bandwidths=(0.5, 0.02)).fit(X_train, y_train)

    assert (a.mode_, a.feature_map_, a.bandwidth_) ==         (b.mode_, b.feature_map_, b.bandwidth_)


def test_diagnose_flags_the_trap_and_clears_the_tuned_kernel(data):
    X_train, _, y_train, _ = data

    trap = QuantumClassifier(bandwidth=1.0).fit(X_train, y_train).diagnose()
    tuned = QuantumClassifier().fit(X_train, y_train).diagnose()

    assert trap["verdict"] == "severely concentrated"
    assert tuned["verdict"] == "healthy"
    assert -1.0 <= trap["kernel_target_alignment"] <= 1.0
    assert 0.0 < trap["top_eigenvalue_fraction"] <= 1.0


def test_kernel_diagnostics_standalone_validates_input():
    from qmlf import kernel_diagnostics

    with pytest.raises(ValueError, match="square"):
        kernel_diagnostics(np.ones((3, 4)))

    report = kernel_diagnostics(np.eye(6))
    assert report["verdict"] == "severely concentrated"
    assert report["kernel_target_alignment"] is None


# --------------------------------------------------------------------------
# Kernel auto-sizing
# --------------------------------------------------------------------------

def test_kernel_with_no_arguments_sizes_itself(data):
    X_train, _, _, _ = data

    kernel = QuantumKernel()
    gram = kernel.fit(X_train).compute_kernel_matrix(X_train)

    assert kernel.n_qubits == X_train.shape[1]
    assert gram.shape == (len(X_train), len(X_train))


def test_auto_sized_kernel_matches_explicit(data):
    X_train, _, _, _ = data

    auto = QuantumKernel().fit(X_train).compute_kernel_matrix(X_train)
    explicit = QuantumKernel(X_train.shape[1]).fit(X_train) \
        .compute_kernel_matrix(X_train)

    assert np.array_equal(auto, explicit)


def test_auto_sized_kernel_locks_to_first_width(data):
    X_train, _, _, _ = data

    kernel = QuantumKernel().fit(X_train)

    with pytest.raises(ValueError, match="features"):
        kernel.compute_kernel_matrix(X_train[:, :3])


def test_unsized_kernel_internals_error_clearly():
    with pytest.raises(ValueError, match="not built"):
        QuantumKernel().feature_map

    with pytest.raises(ValueError, match="not built"):
        QuantumKernel().fidelity_quantum_kernel


def test_circuit_feature_map_sizes_immediately():
    from qiskit.circuit.library import z_feature_map

    kernel = QuantumKernel(feature_map=z_feature_map(feature_dimension=3, reps=1))

    assert kernel.n_qubits == 3


# --------------------------------------------------------------------------
# README quick-start snippets must actually run
# --------------------------------------------------------------------------

def test_readme_kernel_snippet():
    X = np.random.rand(20, 4)
    gram = QuantumKernel(n_qubits=4, mode="ZZ").fit(X).compute_kernel_matrix(X)

    assert gram.shape == (20, 20)


def test_readme_classifier_snippet(data):
    X_train, X_test, y_train, _ = data

    clf = qmlf.QuantumClassifier().fit(X_train, y_train)

    assert clf.predict(X_test).shape == (len(X_test),)
