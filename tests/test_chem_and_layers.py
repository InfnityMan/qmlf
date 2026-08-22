"""Chemistry module contracts and the torch layers' new shape validation.

Run with:  python -m pytest tests/ -q
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import warnings

import numpy as np
import pytest

import qmlf

torch = pytest.importorskip("torch")

COORDS_3 = np.array([[0., 0., 0.], [0., 0., .74], [0., .74, 0.]])


@pytest.fixture(scope="module")
def chem():
    return qmlf.create_advanced_chem_layer(num_atoms=3)


# --------------------------------------------------------------------------
# Chemistry
# --------------------------------------------------------------------------

def test_ground_state_is_min_eigenvalue(chem):
    energy = chem.compute_ground_state_energy(COORDS_3)
    hamiltonian = chem.get_hamiltonian()

    assert np.allclose(hamiltonian, hamiltonian.T)
    assert abs(energy - np.linalg.eigvalsh(hamiltonian).min()) < 1e-12


def test_coulomb_matrix_structure(chem):
    charges = np.array([1.0, 6.0, 8.0])
    coulomb = chem.build_coulomb_matrix(COORDS_3, charges)

    assert np.allclose(coulomb, coulomb.T)
    assert np.allclose(np.diag(coulomb), 0.5 * charges ** 2.4)


def test_descriptor_is_sorted_descending(chem):
    descriptor = chem.molecular_descriptor(COORDS_3)

    assert descriptor.shape == (3,)
    assert np.all(np.diff(descriptor) <= 1e-12)


def test_operator_has_one_qubit_per_atom_and_full_term_count(chem):
    operator = chem._hamiltonian_operator(COORDS_3)

    n = 3
    assert operator.num_qubits == n
    # C(n, 2) ZZ couplings plus n transverse-field X terms.
    assert len(operator.paulis) == n * (n - 1) // 2 + n


def test_vqe_with_fixed_initial_point_is_deterministic_and_variational(chem):
    """Fixed start -> identical energies; variational principle -> never below
    the exact ground state of the operator."""
    operator = chem._hamiltonian_operator(COORDS_3)
    exact_min = np.linalg.eigvalsh(operator.to_matrix()).min()

    from qiskit.circuit.library import real_amplitudes
    n_params = real_amplitudes(num_qubits=3, reps=2).num_parameters
    start = np.zeros(n_params)

    first = chem.compute_vqe_energy(COORDS_3, initial_point=start)
    second = chem.compute_vqe_energy(COORDS_3, initial_point=start)

    assert first == second
    assert first >= exact_min - 1e-6


def test_geometry_size_mismatch_warns():
    layer = qmlf.create_advanced_chem_layer(num_atoms=8)

    with pytest.warns(UserWarning, match="num_atoms"):
        layer.compute_ground_state_energy(COORDS_3)


def test_1d_coordinates_rejected(chem):
    with pytest.raises(ValueError, match="2D"):
        chem.compute_ground_state_energy(np.arange(3.0))


# --------------------------------------------------------------------------
# Torch layer shape contracts (all raise BEFORE reaching torch internals)
# --------------------------------------------------------------------------

def test_qnn_layer_rejects_1d_input():
    layer = qmlf.create_advanced_qnn_layer(n_qubits=3, reps=1, output_dim=2)

    with pytest.raises(ValueError, match="2D"):
        layer(torch.randn(3))


def test_hybrid_layer_rejects_wrong_width():
    layer = qmlf.create_advanced_hybrid_layer(
        input_dim=6, n_qubits=3, reps=1, output_dim=4
    )

    with pytest.raises(ValueError, match="input_dim=6"):
        layer(torch.randn(2, 4))


def test_hybrid_layer_rejects_3d_input():
    layer = qmlf.create_advanced_hybrid_layer(
        input_dim=6, n_qubits=3, reps=1, output_dim=4
    )

    with pytest.raises(ValueError, match="2D"):
        layer(torch.randn(2, 3, 6))


def test_hybrid_layer_forward_shape():
    layer = qmlf.create_advanced_hybrid_layer(
        input_dim=6, n_qubits=3, reps=1, output_dim=4
    )
    layer.eval()

    with torch.no_grad():
        out = layer(torch.randn(5, 6))

    assert out.shape == (5, 4)


def test_full_pipeline_rejects_wrong_width():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline = qmlf.create_full_pipeline(
            input_dim=5, n_qubits=3, reps=1, output_dim=4
        )

    with pytest.raises(ValueError, match="input_dim=5"):
        pipeline(torch.randn(2, 3))


def test_full_pipeline_forward_shape():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline = qmlf.create_full_pipeline(
            input_dim=5, n_qubits=3, reps=1, output_dim=4
        )
        pipeline.eval()

        with torch.no_grad():
            out = pipeline(torch.randn(4, 5))

    assert out.shape == (4, 4)
