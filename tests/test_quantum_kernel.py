"""Regression tests for QuantumKernel.

These pin the behaviour of the bandwidth / normalisation fix. The bugs they
guard against were silent -- covariant mode ignored every input scaling and the
kernel had no bandwidth knob at all, which cost roughly 0.3 AUROC on a real
benchmark without raising anything. Keep them passing.

Run with:  python -m pytest tests/ -q
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import warnings

import numpy as np
import pytest

from qmlf.ops.q_ops_core import QuantumKernel, run_quantum_benchmark


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    # Correlated features, so whitening actually has something to do.
    mixing = np.array([[1.0, 0.8, 0.1, 0.0],
                       [0.0, 0.6, 0.3, 0.1],
                       [0.0, 0.0, 0.9, 0.4],
                       [0.0, 0.0, 0.0, 0.7]])
    return rng.normal(size=(24, 4)) @ mixing


# --------------------------------------------------------------------------
# Backwards compatibility
# --------------------------------------------------------------------------

def test_positional_signature_still_works(data):
    # The benchmark notebook constructs the kernel positionally.
    kernel = QuantumKernel(4, "ZZ", 2, 1e-3)
    assert kernel.bandwidth == 1.0
    assert kernel.normalize is None


def test_defaults_match_pairwise_fidelity_kernel(data):
    # The statevector fast path must be numerically indistinguishable from the
    # original FidelityQuantumKernel evaluation it replaces.
    fast = QuantumKernel(4, mode="ZZ", reps=2).fit(data)
    slow = QuantumKernel(4, mode="ZZ", reps=2, fast_statevector=False).fit(data)
    assert np.abs(fast.compute_kernel_matrix(data)
                  - slow.compute_kernel_matrix(data)).max() < 1e-9


@pytest.mark.filterwarnings("ignore:mode=")
def test_defaults_match_pairwise_covariant(data):
    fast = QuantumKernel(4, mode="covariant", reps=2).fit(data)
    slow = QuantumKernel(4, mode="covariant", reps=2, fast_statevector=False).fit(data)
    assert np.abs(fast.compute_kernel_matrix(data)
                  - slow.compute_kernel_matrix(data)).max() < 1e-9


def test_rectangular_and_batched_paths_agree(data):
    kernel = QuantumKernel(4, mode="ZZ", reps=2).fit(data[:12])
    full = kernel.compute_kernel_matrix(data[12:], data[:12])
    batched = kernel.compute_kernel_matrix(data[12:], data[:12], batch_size=5)
    assert np.abs(full - batched).max() < 1e-12


@pytest.mark.filterwarnings("ignore:mode=")
def test_mahalanobis_is_an_alias_for_covariant(data):
    a = QuantumKernel(4, mode="covariant").fit(data).compute_kernel_matrix(data)
    b = QuantumKernel(4, mode="mahalanobis").fit(data).compute_kernel_matrix(data)
    assert np.abs(a - b).max() < 1e-12


# --------------------------------------------------------------------------
# Bandwidth
# --------------------------------------------------------------------------

def test_bandwidth_scales_zz_angles(data):
    plain = QuantumKernel(4, mode="ZZ").fit(data)._prepare(data)
    scaled = QuantumKernel(4, mode="ZZ", bandwidth=0.25).fit(data)._prepare(data)
    assert np.allclose(scaled, plain * 0.25)


def test_bandwidth_widens_the_kernel(data):
    # Lower bandwidth -> less concentration -> larger off-diagonal similarity.
    def median_offdiag(bw):
        K = QuantumKernel(4, mode="ZZ", bandwidth=bw).fit(data).compute_kernel_matrix(data)
        return np.median(K[~np.eye(len(K), dtype=bool)])

    assert median_offdiag(0.1) > median_offdiag(0.5) > median_offdiag(1.0)


def test_bandwidth_rejects_nonsense():
    for bad in (0, -1, np.nan, np.inf):
        with pytest.raises(ValueError):
            QuantumKernel(4, bandwidth=bad)


# --------------------------------------------------------------------------
# The covariant scale-invariance bug
# --------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore:mode=")
def test_covariant_without_normalize_is_scale_invariant(data):
    # Documents the original behaviour: whitening cancels input scaling, which
    # is exactly why bandwidth has to be applied *after* whitening.
    a = QuantumKernel(4, mode="covariant").fit(data)._prepare(data)
    b = QuantumKernel(4, mode="covariant").fit(data * 100)._prepare(data * 100)
    assert np.abs(a - b).max() < 1e-2


@pytest.mark.filterwarnings("ignore:mode=")
def test_bandwidth_survives_whitening(data):
    # The actual regression: a bandwidth applied after whitening must NOT be
    # cancelled the way a pre-scaled input is.
    full = QuantumKernel(4, mode="covariant", bandwidth=1.0).fit(data)._prepare(data)
    half = QuantumKernel(4, mode="covariant", bandwidth=0.5).fit(data)._prepare(data)
    assert np.allclose(half, full * 0.5)
    assert not np.allclose(half, full)


def test_covariant_kernel_responds_to_bandwidth(data):
    def median_offdiag(bw):
        k = QuantumKernel(4, mode="covariant", bandwidth=bw, normalize="maxabs").fit(data)
        K = k.compute_kernel_matrix(data)
        return np.median(K[~np.eye(len(K), dtype=bool)])

    values = [median_offdiag(bw) for bw in (1.0, 0.5, 0.1)]
    assert len(set(np.round(values, 6))) == 3, "covariant is ignoring bandwidth again"


# --------------------------------------------------------------------------
# Normalisation: bounded angles, fitted on train only
# --------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore:mode=")
def test_maxabs_bounds_angles_within_pi(data):
    prepared = QuantumKernel(4, mode="covariant", normalize="maxabs").fit(data)._prepare(data)
    assert prepared.min() >= -np.pi and prepared.max() <= np.pi
    assert np.isclose(np.abs(prepared).max(), 1.0)


def test_normalize_scale_is_fitted_on_train_only(data):
    # The scale must come from .fit() and be reused verbatim, never recomputed
    # from the data handed to _prepare -- that would leak test statistics.
    kernel = QuantumKernel(4, mode="covariant", normalize="maxabs").fit(data[:12])
    scale = kernel.scale_
    kernel._prepare(data[12:] * 5.0)
    assert kernel.scale_ == scale


def test_normalize_requires_fit(data):
    with pytest.raises(ValueError):
        QuantumKernel(4, mode="ZZ", normalize="maxabs")._prepare(data)


def test_degenerate_fold_does_not_produce_nans():
    constant = np.ones((8, 4))
    kernel = QuantumKernel(4, mode="ZZ", normalize="maxabs").fit(constant)
    assert np.isfinite(kernel._prepare(constant)).all()


def test_invalid_normalize_rejected():
    with pytest.raises(ValueError):
        QuantumKernel(4, normalize="minmax")


# --------------------------------------------------------------------------
# Backend injection
# --------------------------------------------------------------------------

def test_custom_fidelity_disables_the_exact_fast_path():
    from qiskit.primitives import StatevectorSampler
    from qiskit_machine_learning.state_fidelities import ComputeUncompute

    kernel = QuantumKernel(
        4, mode="ZZ", fidelity=ComputeUncompute(sampler=StatevectorSampler())
    )
    assert kernel._use_statevector() is False


def test_default_uses_the_fast_path():
    assert QuantumKernel(4, mode="ZZ")._use_statevector() is True


# --------------------------------------------------------------------------
# Round 2: encoding is a hyperparameter (brief section 5, criteria 1-4)
# --------------------------------------------------------------------------

def test_z_feature_map_matches_qiskit_exactly(data):
    """Criterion 1: feature_map='z' reproduces z_feature_map(n_qubits, reps)."""
    from qiskit.circuit.library import z_feature_map

    kernel = QuantumKernel(4, mode="ZZ", reps=2, feature_map="z")
    expected = z_feature_map(feature_dimension=4, reps=2)
    assert kernel.feature_map.num_qubits == expected.num_qubits
    assert kernel.feature_map.num_parameters == expected.num_parameters
    assert dict(kernel.feature_map.count_ops()) == dict(expected.count_ops())
    # No entangling gates at all -- that is the point of this map.
    assert not {"cx", "cz"} & set(kernel.feature_map.count_ops())


def test_entanglement_is_not_swallowed(data):
    """Criterion 2: 'linear' must give a different Gram matrix from 'full'."""
    full = QuantumKernel(4, mode="ZZ", reps=2, bandwidth=0.5,
                         entanglement="full").fit(data).compute_kernel_matrix(data)
    linear = QuantumKernel(4, mode="ZZ", reps=2, bandwidth=0.5,
                           entanglement="linear").fit(data).compute_kernel_matrix(data)
    assert np.abs(full - linear).max() > 1e-6


def test_z_map_differs_from_zz(data):
    zz = QuantumKernel(4, reps=2, bandwidth=0.5).fit(data).compute_kernel_matrix(data)
    z = QuantumKernel(4, reps=2, bandwidth=0.5,
                      feature_map="z").fit(data).compute_kernel_matrix(data)
    assert np.abs(zz - z).max() > 1e-6


def test_defaults_unchanged_from_previous_release(data):
    """Criterion 3: the default encoding is still zz/full."""
    from qiskit.circuit.library import zz_feature_map

    kernel = QuantumKernel(4, mode="ZZ", reps=2)
    expected = zz_feature_map(feature_dimension=4, reps=2)
    assert dict(kernel.feature_map.count_ops()) == dict(expected.count_ops())
    assert kernel.entanglement == "full"


def test_custom_circuit_accepted_and_validated(data):
    from qiskit.circuit.library import z_feature_map

    custom = z_feature_map(feature_dimension=4, reps=3)
    kernel = QuantumKernel(4, feature_map=custom).fit(data)
    assert kernel.feature_map is custom
    assert kernel.compute_kernel_matrix(data).shape == (len(data), len(data))

    with pytest.raises(ValueError, match="qubits"):
        QuantumKernel(4, feature_map=z_feature_map(feature_dimension=6))


def test_unknown_feature_map_rejected():
    with pytest.raises(ValueError, match="Unknown feature_map"):
        QuantumKernel(4, feature_map="bogus")


def test_entanglement_with_z_map_warns():
    with pytest.warns(UserWarning, match="no effect"):
        QuantumKernel(4, feature_map="z", entanglement="linear")


def test_assigning_feature_map_rebuilds_both_paths(data):
    """A swapped map must affect the pairwise path too, not just statevector."""
    from qiskit.circuit.library import z_feature_map

    kernel = QuantumKernel(4, reps=2, bandwidth=0.5, fast_statevector=False).fit(data)
    before = kernel.compute_kernel_matrix(data)
    kernel.feature_map = z_feature_map(feature_dimension=4, reps=2)
    after = kernel.compute_kernel_matrix(data)
    assert np.abs(before - after).max() > 1e-6


# --------------------------------------------------------------------------
# Round 2: the silent-bypass bug
# --------------------------------------------------------------------------

def test_covariant_without_normalize_warns(data):
    with pytest.warns(UserWarning, match="normalize='maxabs'"):
        QuantumKernel(4, mode="covariant")


def test_direct_fidelity_kernel_access_warns(data):
    """Using .fidelity_quantum_kernel bypasses _prepare -- that must not be silent."""
    kernel = QuantumKernel(4, mode="ZZ", bandwidth=0.1)
    with pytest.warns(UserWarning, match="bypasses _prepare"):
        _ = kernel.fidelity_quantum_kernel


def test_plain_kernel_access_does_not_warn(data):
    # Nothing would be bypassed here, so no warning should fire.
    kernel = QuantumKernel(4, mode="ZZ")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _ = kernel.fidelity_quantum_kernel


def test_run_quantum_benchmark_respects_mode():
    """Criterion 4: covariant must give a different answer from ZZ."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 3)) @ np.array([[1.0, 0.7, 0.2],
                                             [0.0, 0.8, 0.3],
                                             [0.0, 0.0, 0.6]])
    y = (X[:, 0] + 0.5 * rng.normal(size=40) > 0).astype(int)

    zz = run_quantum_benchmark(X, y, reps=1, bandwidth=0.3)
    with pytest.warns(UserWarning):
        cov = run_quantum_benchmark(X, y, reps=1, bandwidth=0.3, mode="covariant")

    zz_acc = zz.loc[zz["Model"] == "Quantum SVC", "Accuracy"].iloc[0]
    cov_acc = cov.loc[cov["Model"] == "Quantum SVC", "Accuracy"].iloc[0]
    assert zz_acc != cov_acc, "run_quantum_benchmark is still bypassing _prepare"


def test_run_quantum_benchmark_respects_bandwidth():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(40, 3))
    y = (X[:, 0] > 0).astype(int)
    wide = run_quantum_benchmark(X, y, reps=1, bandwidth=1.0)
    narrow = run_quantum_benchmark(X, y, reps=1, bandwidth=0.05)
    assert not wide.equals(narrow), "bandwidth is being discarded again"
