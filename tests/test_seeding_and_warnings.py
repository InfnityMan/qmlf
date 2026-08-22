"""Seeded-reproducibility knobs and the newest honest-failure warnings.

Run with:  python -m pytest tests/ -q
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import warnings

import numpy as np
import pytest

import qmlf

X = np.random.RandomState(2).uniform(-1, 1, (30, 6))
y = np.random.RandomState(1).randint(0, 2, 30)


def test_qiga_random_state_is_self_contained():
    """Same seed, same answer -- without touching global numpy state."""
    def run():
        selector = qmlf.create_advanced_qiga_selector(
            population_size=4, generations=2, random_state=7
        )
        selector.fit(X, y)
        return selector.selected_features.copy(), selector.best_score

    mask_a, score_a = run()
    mask_b, score_b = run()

    assert np.array_equal(mask_a, mask_b)
    assert score_a == score_b


def test_qiga_refit_resets_convergence_history():
    selector = qmlf.create_advanced_qiga_selector(
        population_size=4, generations=2, random_state=0
    )

    selector.fit(X, y)
    first_length = len(selector.convergence_history)
    selector.fit(X, y)

    # Used to be 2x after a refit: the old curve was never cleared.
    assert len(selector.convergence_history) == first_length == 2


def test_qiga_default_rng_path_unchanged():
    """random_state=None must keep the historical global-RNG behaviour."""
    def run():
        np.random.seed(11)
        selector = qmlf.create_advanced_qiga_selector(
            population_size=4, generations=2
        )
        selector.fit(X, y)
        return selector.selected_features.copy()

    assert np.array_equal(run(), run())


def test_noise_fit_warns_on_malformed_calibration():
    mitigator = qmlf.create_advanced_noise_mitigator()

    with pytest.warns(UserWarning, match="square"):
        mitigator.fit(np.array([0.9, 0.1]))

    assert mitigator.get_calibration_matrix() is None


def test_noise_fit_square_matrix_is_quiet():
    mitigator = qmlf.create_advanced_noise_mitigator()

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        mitigator.fit(np.array([[0.9, 0.1], [0.05, 0.95]]))

    assert mitigator.get_calibration_matrix() is not None


def test_benchmark_warns_when_n_qubits_disagrees_with_data():
    with pytest.warns(UserWarning, match="n_qubits is ignored"):
        qmlf.run_quantum_benchmark(X[:, :4], y, n_qubits=8)


def test_benchmark_matching_n_qubits_is_quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        result = qmlf.run_quantum_benchmark(X[:, :4], y, n_qubits=4)

    assert set(result["Model"]) == {"XGBoost", "Quantum SVC"}
