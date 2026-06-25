import numpy as np

from qiskit.circuit.library import real_amplitudes
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA


class AdvancedQuantumChemistryLayer:
    """Molecular feature and energy utilities.

    Builds charge-aware Coulomb-matrix descriptors and a geometry-derived model
    Hamiltonian, and estimates its ground-state energy. The default energy path
    (``compute_ground_state_energy``) is a stable classical diagonalization; a
    genuine Qiskit VQE (``compute_vqe_energy``) is also provided for small
    systems. ``basis`` is accepted for API compatibility and is reserved for a
    future electronic-structure backend: the V1 uses a model spin Hamiltonian,
    not a full electronic-structure calculation.
    """

    def __init__(
        self,
        num_atoms=8,
        basis="sto-3g"
    ):
        self.num_atoms = num_atoms
        self.basis = basis

        self.hamiltonian = None
        self.coulomb_matrix = None
        self.vqe_result = None

    def _pairwise_distances(
        self,
        coordinates
    ):
        coordinates = np.asarray(
            coordinates,
            dtype=float
        )

        n_atoms = coordinates.shape[0]

        distances = np.zeros(
            (n_atoms, n_atoms)
        )

        for i in range(n_atoms):
            for j in range(n_atoms):
                distances[i, j] = np.linalg.norm(
                    coordinates[i]
                    - coordinates[j]
                )

        return distances

    def _resolve_charges(
        self,
        n_atoms,
        charges
    ):
        if charges is None:
            return np.ones(n_atoms)

        return np.asarray(
            charges,
            dtype=float
        )

    def build_coulomb_matrix(
        self,
        coordinates,
        charges=None
    ):
        distances = self._pairwise_distances(
            coordinates
        )

        n_atoms = distances.shape[0]

        charges = self._resolve_charges(
            n_atoms,
            charges
        )

        coulomb = np.zeros_like(
            distances
        )

        for i in range(n_atoms):
            for j in range(n_atoms):
                if i == j:
                    coulomb[i, j] = 0.5 * charges[i] ** 2.4
                else:
                    coulomb[i, j] = (
                        charges[i] * charges[j]
                        / (
                            distances[i, j]
                            + 1e-8
                        )
                    )

        self.coulomb_matrix = coulomb

        return coulomb

    def build_hamiltonian(
        self,
        coordinates,
        charges=None
    ):
        """Build a symmetric model Hamiltonian matrix from the geometry.

        A simplified tight-binding-style model: diagonal on-site terms from the
        charge-aware self-energy, off-diagonal couplings that decay with
        interatomic distance. This is NOT a full electronic-structure
        Hamiltonian.
        """
        distances = self._pairwise_distances(
            coordinates
        )

        n_atoms = distances.shape[0]

        charges = self._resolve_charges(
            n_atoms,
            charges
        )

        hamiltonian = np.zeros(
            (n_atoms, n_atoms)
        )

        for i in range(n_atoms):
            for j in range(n_atoms):
                if i == j:
                    hamiltonian[i, j] = 0.5 * charges[i] ** 2.4
                else:
                    hamiltonian[i, j] = (
                        -charges[i] * charges[j]
                        / (
                            distances[i, j]
                            + 1e-8
                        )
                    )

        self.hamiltonian = hamiltonian

        return hamiltonian

    def compute_ground_state_energy(
        self,
        coordinates,
        charges=None
    ):
        """Classical ground-state energy of the model Hamiltonian.

        Returns the smallest eigenvalue of the geometry-derived model
        Hamiltonian via exact diagonalization. This is the stable, recommended
        energy path.
        """
        hamiltonian = self.build_hamiltonian(
            coordinates,
            charges
        )

        eigenvalues = np.linalg.eigvalsh(
            hamiltonian
        )

        return float(
            np.min(eigenvalues)
        )

    def _hamiltonian_operator(
        self,
        coordinates,
        charges=None
    ):
        distances = self._pairwise_distances(
            coordinates
        )

        n_atoms = distances.shape[0]

        charges = self._resolve_charges(
            n_atoms,
            charges
        )

        terms = []

        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                coupling = (
                    charges[i] * charges[j]
                    / (
                        distances[i, j]
                        + 1e-8
                    )
                )

                pauli = ["I"] * n_atoms
                pauli[i] = "Z"
                pauli[j] = "Z"

                terms.append(
                    ("".join(pauli), coupling)
                )

        for i in range(n_atoms):
            pauli = ["I"] * n_atoms
            pauli[i] = "X"

            terms.append(
                ("".join(pauli), 1.0)
            )

        return SparsePauliOp.from_list(
            terms
        )

    def compute_vqe_energy(
        self,
        coordinates,
        charges=None
    ):
        """Estimate the ground-state energy with a real VQE.

        Runs a genuine variational quantum eigensolver (real_amplitudes ansatz +
        COBYLA over a statevector Estimator) on a geometry-derived model spin
        Hamiltonian (one qubit per atom). Practical only for small systems; for
        routine use prefer ``compute_ground_state_energy``.
        """
        hamiltonian = self._hamiltonian_operator(
            coordinates,
            charges
        )

        n_qubits = hamiltonian.num_qubits

        ansatz = real_amplitudes(
            num_qubits=n_qubits,
            reps=2
        )

        optimizer = COBYLA(
            maxiter=100
        )

        vqe = VQE(
            StatevectorEstimator(),
            ansatz,
            optimizer
        )

        result = vqe.compute_minimum_eigenvalue(
            operator=hamiltonian
        )

        self.vqe_result = result

        return float(
            result.eigenvalue.real
        )

    def molecular_descriptor(
        self,
        coordinates,
        charges=None
    ):
        coulomb = self.build_coulomb_matrix(
            coordinates,
            charges
        )

        eigenvalues = np.linalg.eigvalsh(
            coulomb
        )

        return np.sort(
            eigenvalues
        )[::-1]

    def fit(
        self,
        molecular_data
    ):
        return self

    def transform(
        self,
        coordinates
    ):
        return self.compute_ground_state_energy(
            coordinates
        )

    def fit_transform(
        self,
        coordinates
    ):
        return self.fit(
            coordinates
        ).transform(
            coordinates
        )

    def get_hamiltonian(self):
        return self.hamiltonian

    def get_coulomb_matrix(self):
        return self.coulomb_matrix


def create_advanced_chem_layer(
    num_atoms=8,
    basis="sto-3g"
):
    return AdvancedQuantumChemistryLayer(
        num_atoms=num_atoms,
        basis=basis
    )
