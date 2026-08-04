"""Regression tests for QuantumKernel.

These pin the behaviour of the bandwidth / normalisation fix. The bugs they
guard against were silent -- covariant mode ignored every input scaling and the
kernel had no bandwidth knob at all, which cost roughly 0.3 AUROC on a real
benchmark without raising anything. Keep them passing.

Run with:  python -m pytest tests/ -q
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pytest

from qmlf.ops.q_ops_core import QuantumKernel


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

def test_covariant_without_normalize_is_scale_invariant(data):
    # Documents the original behaviour: whitening cancels input scaling, which
    # is exactly why bandwidth has to be applied *after* whitening.
    a = QuantumKernel(4, mode="covariant").fit(data)._prepare(data)
    b = QuantumKernel(4, mode="covariant").fit(data * 100)._prepare(data * 100)
    assert np.abs(a - b).max() < 1e-2


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
