"""Golden-value regression tests: pin the numbers the framework produces.

These values were computed on the stack the release was verified against
(qiskit 2.3.1/2.5.2, numpy 2.3/2.5 -- identical to 1e-10 on both). Benchmark
task thresholds are calibrated against these exact numerics, so a dependency
bump that moves them should FAIL here first, not silently shift a published
leaderboard. Tolerance is 1e-9: loose enough for BLAS/platform last-ulp noise,
tight enough that any algorithmic change trips it.

Run with:  python -m pytest tests/ -q
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import warnings

import numpy as np
import pytest

import qmlf

ATOL = 1e-9


@pytest.fixture(scope="module")
def X():
    return np.random.default_rng(42).uniform(-1.0, 1.0, size=(8, 3))


def test_zz_kernel_values(X):
    gram = qmlf.QuantumKernel(3, mode="ZZ", bandwidth=0.5).fit(X) \
        .compute_kernel_matrix(X)

    assert np.allclose(
        gram[0, 1:4],
        [0.045374904406943, 0.003287672037708, 0.437275911378585],
        atol=ATOL
    )
    assert abs(gram[~np.eye(8, dtype=bool)].mean() - 0.1747301176893954) < ATOL


def test_mahalanobis_kernel_values(X):
    gram = qmlf.QuantumKernel(
        3, mode="mahalanobis", normalize="maxabs", bandwidth=0.5
    ).fit(X).compute_kernel_matrix(X)

    assert np.allclose(
        gram[0, 1:4],
        [0.348802976978685, 0.1290887033797, 0.031672649632157],
        atol=ATOL
    )
    assert abs(gram[~np.eye(8, dtype=bool)].mean() - 0.16500650973106218) < ATOL


def test_z_map_kernel_values(X):
    gram = qmlf.QuantumKernel(3, feature_map="z", bandwidth=0.5).fit(X) \
        .compute_kernel_matrix(X)

    assert np.allclose(
        gram[0, 1:4],
        [0.851361571221219, 0.671373059817707, 0.893565458800256],
        atol=ATOL
    )


def test_kernel_matrix_properties(X):
    """Structural invariants: symmetric, unit diagonal, PSD, values in [0, 1]."""
    gram = qmlf.QuantumKernel(3, bandwidth=0.5).fit(X).compute_kernel_matrix(X)

    assert np.allclose(gram, gram.T, atol=1e-12)
    assert np.allclose(np.diag(gram), 1.0, atol=1e-10)
    assert gram.min() >= -1e-12 and gram.max() <= 1.0 + 1e-12
    assert np.linalg.eigvalsh((gram + gram.T) / 2).min() >= -1e-9


def test_depolarizing_inversion_value():
    mitigator = qmlf.create_advanced_noise_mitigator(
        strategy="depolarizing", noise_level=0.1
    )
    out = mitigator.mitigate(np.array([0.7, 0.3]))

    assert np.allclose(out[0], [0.7222222222215, 0.2777777777775], atol=ATOL)


def test_nisq_fidelity_model_values():
    report = qmlf.create_advanced_nisq_optimizer().optimize_circuit(20, 40, 10)

    assert abs(report["original_fidelity"] - 0.6703200460356393) < ATOL
    assert abs(report["optimized_fidelity"] - 0.7788007830714049) < ATOL
    assert abs(report["mitigated_fidelity"] - 0.8341005873035536) < ATOL


def test_chem_ground_state_value():
    coordinates = np.array([[0., 0., 0.], [0., 0., .74], [0., .74, 0.]])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # num_atoms=3 matches, no warning path
        energy = qmlf.create_advanced_chem_layer(num_atoms=3) \
            .compute_ground_state_energy(coordinates)

    assert abs(energy - (-1.9476910021168328)) < ATOL
