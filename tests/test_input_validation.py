"""Tests for the paths that used to fail silently or unhelpfully.

Each of these previously returned a plausible-looking value, or an error that
named nothing the caller had written:

  * an unrecognised mitigation strategy returned the input unmitigated
  * zne on a 1D vector read probabilities as noise scales and returned 1.0
  * readout without a calibration matrix passed through with no warning
  * AdvancedQIGASelector(n_features=2) selected 5 of 6 features
  * a QuantumKernel width mismatch surfaced from inside Qiskit's parameter
    binding as "Mismatching number of values and parameters"
  * a 2D tensor into the transformer raised a bare reshape error

Run with:  python -m pytest tests/ -q
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pytest

import qmlf
from qmlf.noise.q_noise import AdvancedNoiseMitigator
from qmlf.ops.q_ops_core import QuantumKernel


# --------------------------------------------------------------------------
# Noise mitigation
# --------------------------------------------------------------------------

def test_unknown_strategy_rejected_at_construction():
    with pytest.raises(ValueError, match="Unknown strategy"):
        AdvancedNoiseMitigator(strategy="nonsense")


def test_strategy_is_case_sensitive_and_says_so():
    # "ZNE" used to sail through and return the input unmitigated.
    with pytest.raises(ValueError, match="lower-case"):
        AdvancedNoiseMitigator(strategy="ZNE")


def test_noise_level_of_one_rejected():
    # The depolarizing inverse divides by (1 - noise_level).
    with pytest.raises(ValueError, match="noise_level"):
        AdvancedNoiseMitigator(noise_level=1.0, strategy="depolarizing")


def test_zne_rejects_a_1d_vector():
    """A 1D input used to be read as one measurement per noise scale, so
    mitigate([0.7, 0.3]) polyfitted the two probabilities and returned 1.0."""
    mitigator = AdvancedNoiseMitigator(strategy="zne")

    with pytest.raises(ValueError, match="2D"):
        mitigator.mitigate(np.array([0.7, 0.3]))


def test_zne_over_several_scales_still_works():
    mitigator = AdvancedNoiseMitigator(strategy="zne")

    measured = np.array([[0.70, 0.30],
                         [0.60, 0.40],
                         [0.55, 0.45]])
    extrapolated = mitigator.mitigate(measured)

    assert extrapolated.shape == (2,)
    assert extrapolated.sum() == pytest.approx(1.0)


def test_zne_with_one_scale_warns_rather_than_inventing_a_result():
    mitigator = AdvancedNoiseMitigator(strategy="zne")

    with pytest.warns(UserWarning, match="single noise scale"):
        out = mitigator.mitigate(np.array([[0.7, 0.3]]))

    assert np.allclose(out, [[0.7, 0.3]])


def test_uncalibrated_readout_warns():
    mitigator = AdvancedNoiseMitigator(strategy="readout")

    with pytest.warns(UserWarning, match="no calibration matrix"):
        mitigator.mitigate(np.array([0.7, 0.3]))


def test_calibrated_readout_does_not_warn_and_still_mitigates():
    confusion = np.array([[0.90, 0.10],
                          [0.05, 0.95]])
    mitigator = AdvancedNoiseMitigator(strategy="readout").fit(confusion)

    with warnings_as_errors():
        out = mitigator.mitigate(np.array([0.7, 0.3]))

    assert out.shape == (1, 2)
    assert out.sum() == pytest.approx(1.0)
    # Inverting the confusion matrix must move the distribution somewhere.
    assert not np.allclose(out, [[0.7, 0.3]])


# --------------------------------------------------------------------------
# Parameters that never did anything
# --------------------------------------------------------------------------

def test_qiga_n_features_warns_because_it_has_no_effect():
    with pytest.warns(UserWarning, match="no effect"):
        qmlf.create_advanced_qiga_selector(n_features=2)


def test_qiga_default_construction_is_quiet():
    with warnings_as_errors():
        qmlf.create_advanced_qiga_selector()


def test_federated_warns_when_the_cohort_size_disagrees():
    federated = qmlf.create_federated_qml(num_clients=5)

    with pytest.warns(UserWarning, match="num_clients"):
        federated.aggregate([[1.0, 2.0], [3.0, 4.0]])


def test_federated_full_cohort_is_quiet():
    federated = qmlf.create_federated_qml(num_clients=2)

    with warnings_as_errors():
        result = federated.aggregate([[1.0, 2.0], [3.0, 4.0]])

    assert np.allclose(result, [2.0, 3.0])


def test_chem_warns_when_the_geometry_size_disagrees():
    layer = qmlf.create_advanced_chem_layer(num_atoms=8)
    coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])

    with pytest.warns(UserWarning, match="num_atoms"):
        layer.compute_ground_state_energy(coordinates)


# --------------------------------------------------------------------------
# Shape contracts
# --------------------------------------------------------------------------

def test_kernel_width_mismatch_names_both_numbers():
    X = np.random.default_rng(0).uniform(-1.0, 1.0, size=(10, 4))

    with pytest.raises(ValueError) as excinfo:
        QuantumKernel(n_qubits=8).fit(X)

    message = str(excinfo.value)
    assert "n_qubits=8" in message
    assert "4 features" in message


def test_kernel_width_checked_without_fit_in_zz_mode():
    # ZZ mode never requires fit(), so the check cannot live only there.
    X = np.random.default_rng(0).uniform(-1.0, 1.0, size=(10, 4))

    with pytest.raises(ValueError, match="n_qubits=8"):
        QuantumKernel(n_qubits=8).compute_kernel_matrix(X)


def test_kernel_rejects_a_1d_array():
    with pytest.raises(ValueError, match="2D"):
        QuantumKernel(n_qubits=4).fit(np.arange(4.0))


def test_matching_width_still_works():
    X = np.random.default_rng(0).uniform(-1.0, 1.0, size=(8, 4))
    gram = QuantumKernel(n_qubits=4).fit(X).compute_kernel_matrix(X)

    assert gram.shape == (8, 8)


def test_transformer_rejects_a_2d_tensor():
    torch = pytest.importorskip("torch")

    layer = qmlf.create_advanced_quantum_transformer(
        n_qubits=4, heads=2, reps=1, embed_dim=8
    )

    with pytest.raises(ValueError, match="3D"):
        layer(torch.randn(2, 8))


def test_transformer_rejects_a_wrong_embed_dim():
    torch = pytest.importorskip("torch")

    layer = qmlf.create_advanced_quantum_transformer(
        n_qubits=4, heads=2, reps=1, embed_dim=8
    )

    with pytest.raises(ValueError, match="embed_dim"):
        layer(torch.randn(2, 3, 16))


# --------------------------------------------------------------------------
# Headless plotting
# --------------------------------------------------------------------------

def test_plot_hilbert_space_can_be_suppressed():
    """show=False must return the figure without trying to open a browser."""
    figure = qmlf.plot_hilbert_space(np.eye(6), show=False)

    assert figure is not None


def test_plot_hilbert_space_writes_to_a_path(tmp_path):
    target = tmp_path / "kernel.html"
    qmlf.plot_hilbert_space(np.eye(6), show=False, save_path=str(target))

    assert target.exists()
    assert target.stat().st_size > 0


# --------------------------------------------------------------------------

def warnings_as_errors():
    import warnings as _warnings
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        with _warnings.catch_warnings():
            _warnings.simplefilter("error", UserWarning)
            yield

    return _ctx()
