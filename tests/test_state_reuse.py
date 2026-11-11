"""Tests for the simulation-reuse paths: same numbers, fewer simulations.

Every optimisation here is exact, so each test pins BOTH halves of that claim:
the values still match the unoptimised path, and the number of statevector
simulations actually drops. Counting the simulations is the point -- a cache
that silently stops hitting still returns the right answer, just slowly.

Run with:  python -m pytest tests/ -q
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from qmlf.models.q_models import QuantumClassifier, kernel_diagnostics
from qmlf.ops import q_ops_core
from qmlf.ops.q_ops_core import (
    QuantumKernel,
    clear_state_cache,
    set_state_cache_limit,
    state_cache_info,
)


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    return rng.uniform(-1.0, 1.0, size=(16, 3))


@pytest.fixture(autouse=True)
def fresh_cache():
    """Every test starts from a cold cache and restores the default budget."""
    clear_state_cache()
    yield
    set_state_cache_limit(64)
    clear_state_cache()


class _CountingStatevector:
    """Wraps Statevector so the tests can count simulations."""

    def __init__(self):
        self.calls = 0

    def __call__(self, circuit):
        self.calls += 1
        return Statevector(circuit)


@pytest.fixture
def counter(monkeypatch):
    counting = _CountingStatevector()
    monkeypatch.setattr(q_ops_core, "Statevector", counting)
    return counting


def _assert_same_gram(expected, chunked):
    """batch_size chunks a BLAS matmul, and BLAS rounding depends on the block shape.

    So a chunked Gram is equal to the unchunked one to rounding, not to the
    bit -- on every platform. Measured on Kaggle: these 3-qubit cases were
    bit-equal on one machine but differed by up to 4.4e-16 on two others; at
    8 qubits and 200 samples, batch_size=1 moved entries by up to 1.8e-15.
    The simulation counts in each test are what pin the chunking.
    """
    np.testing.assert_allclose(chunked, expected, rtol=0, atol=1e-14)


def test_batching_simulates_each_row_once(data, counter):
    """batch_size chunks the Gram matmul; it must not re-simulate the table.

    This used to re-simulate the whole target table per chunk -- O(n^2 /
    batch_size) simulations instead of O(n), so a smaller batch was slower.
    """
    kernel = QuantumKernel(n_qubits=3, bandwidth=0.5).fit(data)

    full = kernel.compute_kernel_matrix(data)
    simulations_unbatched = counter.calls

    clear_state_cache()
    counter.calls = 0
    batched = kernel.compute_kernel_matrix(data, batch_size=4)

    _assert_same_gram(full, batched)
    assert simulations_unbatched == len(data)
    assert counter.calls == len(data)

    # With the cache on, a re-simulating chunk loop would still hit the cache
    # and count once per row -- so the count that actually pins the chunking is
    # this one, taken with the cache switched off.
    set_state_cache_limit(0)
    clear_state_cache()
    counter.calls = 0
    uncached = kernel.compute_kernel_matrix(data, batch_size=4)

    _assert_same_gram(full, uncached)
    assert counter.calls == len(data)


def test_batched_cross_kernel_matches_and_reuses(data, counter):
    kernel = QuantumKernel(n_qubits=3, bandwidth=0.5).fit(data[:10])

    full = kernel.compute_kernel_matrix(data[10:], data[:10])
    clear_state_cache()
    counter.calls = 0
    batched = kernel.compute_kernel_matrix(data[10:], data[:10], batch_size=2)

    _assert_same_gram(full, batched)
    # Six test rows plus ten training rows, each simulated exactly once.
    assert counter.calls == len(data)


def test_predict_reuses_the_training_simulations(data, counter):
    """predict() must simulate its own rows only, not the training set again."""
    y = np.array([i % 2 for i in range(len(data))])
    unseen = np.random.default_rng(1).uniform(-1.0, 1.0, size=(4, 3))
    clf = QuantumClassifier(bandwidth=0.5, max_qubits=None).fit(data, y)

    counter.calls = 0
    first = clf.predict(unseen)
    after_first = counter.calls
    second = clf.predict(unseen)

    assert np.array_equal(first, second)
    assert after_first == len(unseen)      # the training rows come from the cache
    assert counter.calls == after_first    # and the second call simulates nothing

    # Rows the model was fitted on cost nothing at all.
    counter.calls = 0
    clf.predict(data[:4])
    assert counter.calls == 0


def test_cross_validation_shares_simulations_between_folds(data, counter):
    """A data-independent kernel prepares the same angles in every fold."""
    y = np.array([i % 2 for i in range(len(data))])
    bandwidths = (1.0, 0.25)

    QuantumClassifier(bandwidths=bandwidths, max_qubits=None, cv=3).fit(data, y)
    shared = counter.calls

    clear_state_cache()
    set_state_cache_limit(0)
    counter.calls = 0
    QuantumClassifier(bandwidths=bandwidths, max_qubits=None, cv=3).fit(data, y)

    # One simulation per row per candidate bandwidth, however many folds ran.
    assert shared == len(data) * len(bandwidths)
    assert counter.calls > shared


def test_disabled_cache_gives_identical_values(data):
    kernel = QuantumKernel(n_qubits=3, bandwidth=0.5).fit(data)
    cached = kernel.compute_kernel_matrix(data)

    set_state_cache_limit(0)
    uncached = QuantumKernel(n_qubits=3, bandwidth=0.5).fit(data) \
        .compute_kernel_matrix(data)

    assert np.array_equal(cached, uncached)


def test_cache_is_keyed_by_encoding_not_just_angles(data):
    """Two feature maps on the same rows must not share simulations."""
    zz = QuantumKernel(n_qubits=3, feature_map="zz", bandwidth=0.5).fit(data)
    z = QuantumKernel(n_qubits=3, feature_map="z", bandwidth=0.5).fit(data)

    assert not np.allclose(zz.compute_kernel_matrix(data), z.compute_kernel_matrix(data))


def test_mutating_a_supplied_circuit_is_not_served_from_cache(data):
    """A caller's own circuit can change under the kernel; results must follow."""
    from qiskit.circuit import Parameter, QuantumCircuit

    parameters = [Parameter(f"x{i}") for i in range(3)]
    circuit = QuantumCircuit(3)
    for i, parameter in enumerate(parameters):
        circuit.ry(parameter, i)

    kernel = QuantumKernel(feature_map=circuit, bandwidth=0.5).fit(data)
    before = kernel.compute_kernel_matrix(data)

    # Mutated in place after the kernel was built and used. It has to be a
    # PARAMETERISED gate: a fixed unitary applied to every state leaves
    # |<a|b>|^2 untouched, so it could not reveal a stale cache.
    circuit.rz(parameters[1], 0)
    after = kernel.compute_kernel_matrix(data)

    fresh = QuantumKernel(feature_map=circuit, bandwidth=0.5).fit(data) \
        .compute_kernel_matrix(data)

    assert not np.allclose(before, after)
    assert np.array_equal(after, fresh)


def test_composite_gates_are_not_confused_with_each_other(data):
    """Two encodings that differ only INSIDE a composite gate are different.

    A digest built from instruction names alone reads both circuits as the
    same one, and the second kernel is then served the first one's states --
    silently wrong values, not a slow path.
    """
    from qiskit.circuit import Parameter, QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp

    def kernel_for(pauli):
        # The composite gate goes FIRST: a fixed unitary applied after the
        # encoding cancels in |<a|b>|^2, so a kernel could not tell the two
        # circuits apart even with a perfect cache.
        circuit = QuantumCircuit(3)
        for qubit in range(3):
            circuit.ry(0.3 * (qubit + 1), qubit)  # non-uniform: H|0> would be
        circuit.append(PauliEvolutionGate(SparsePauliOp([pauli]), time=0.7), range(3))
        for qubit in range(3):                     # an eigenstate of XXI
            circuit.ry(Parameter(f"x{qubit}"), qubit)

        return QuantumKernel(feature_map=circuit, bandwidth=0.5).fit(data[:, :3])

    xx = kernel_for("XXI").compute_kernel_matrix(data[:, :3])
    zz = kernel_for("ZZI").compute_kernel_matrix(data[:, :3])

    set_state_cache_limit(0)
    clear_state_cache()
    xx_uncached = kernel_for("XXI").compute_kernel_matrix(data[:, :3])
    zz_uncached = kernel_for("ZZI").compute_kernel_matrix(data[:, :3])

    assert np.array_equal(xx, xx_uncached)
    assert np.array_equal(zz, zz_uncached)
    assert not np.allclose(xx, zz)


def test_array_valued_gate_parameters_are_hashed_by_value(data):
    """Two big unitaries print the same under numpy's summary; they are not."""
    from qiskit.circuit import Parameter, QuantumCircuit
    from qiskit.circuit.library import UnitaryGate

    def kernel_for(swap):
        # Two basis states swapped in the middle of a 32x32 matrix, where
        # numpy's summary printing does not reach. The gate goes before the
        # rotations, for the same reason as above.
        matrix = np.eye(32)
        matrix[[12, swap]] = matrix[[swap, 12]]
        circuit = QuantumCircuit(5)
        for qubit in range(5):
            circuit.ry(0.3 * (qubit + 1), qubit)  # non-uniform: permuting the
        circuit.append(UnitaryGate(matrix), range(5))  # amplitudes of a uniform
                                                       # state changes nothing
        for qubit in range(5):
            circuit.ry(Parameter(f"x{qubit}"), qubit)

        return QuantumKernel(feature_map=circuit, bandwidth=0.5).fit(wide)

    wide = np.random.default_rng(3).uniform(-1.0, 1.0, size=(6, 5))
    first = kernel_for(19).compute_kernel_matrix(wide)
    second = kernel_for(27).compute_kernel_matrix(wide)

    set_state_cache_limit(0)
    clear_state_cache()

    assert np.array_equal(first, kernel_for(19).compute_kernel_matrix(wide))
    assert np.array_equal(second, kernel_for(27).compute_kernel_matrix(wide))
    assert not np.allclose(first, second)


def test_cache_budget_tracks_what_is_actually_held(data):
    kernel = QuantumKernel(n_qubits=3, bandwidth=0.5).fit(data)

    for _ in range(4):  # repeated puts for rows that are already cached
        kernel.compute_kernel_matrix(data)

    info = state_cache_info()
    held = sum(
        value.nbytes for value in q_ops_core._STATE_CACHE.values()
    )

    assert info["bytes"] == held
    assert info["entries"] == len(data)


def test_concurrent_kernels_do_not_corrupt_the_cache(data):
    """The cache is module-level; threaded callers must not break it."""
    from concurrent.futures import ThreadPoolExecutor

    expected = QuantumKernel(n_qubits=3, bandwidth=0.5).fit(data) \
        .compute_kernel_matrix(data)
    set_state_cache_limit(1)  # small budget, so eviction runs constantly

    def work(_):
        return QuantumKernel(n_qubits=3, bandwidth=0.5).fit(data) \
            .compute_kernel_matrix(data)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(work, range(24)))

    assert all(np.array_equal(expected, result) for result in results)
    assert state_cache_info()["bytes"] >= 0


def test_projected_kernel_reuses_bloch_vectors(data, counter):
    kernel = QuantumKernel(n_qubits=3, kernel="projected", bandwidth=0.5).fit(data)

    first = kernel.compute_kernel_matrix(data)
    simulations = counter.calls
    second = kernel.compute_kernel_matrix(data)

    assert np.array_equal(first, second)
    assert simulations == len(data)
    assert counter.calls == simulations
    # Only the Bloch vectors are kept. Keeping the statevectors too would fill
    # the budget with entries this path never reads again -- 21x larger than
    # the vectors they would evict, at 8 qubits (4,096 bytes against 192).
    assert state_cache_info()["entries"] == len(data)


def test_alignment_selection_matches_explicit_centering(data):
    """Pin the scorer that picks bandwidth='alignment', not just its plumbing."""
    y = np.array([i % 2 for i in range(len(data))])
    candidates = (1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01)
    n = len(data)
    centering = np.eye(n) - np.ones((n, n)) / n
    target = np.where(y[:, None] == y[None, :], 1.0, -1.0)
    target_c = centering @ target @ centering

    scores = []
    for candidate in candidates:
        gram = QuantumKernel(n_qubits=3, bandwidth=candidate).fit(data) \
            .compute_kernel_matrix(data)
        gram_c = centering @ gram @ centering
        scores.append((gram_c * target_c).sum() / (
            np.linalg.norm(gram_c) * np.linalg.norm(target_c)
        ))

    fitted = QuantumKernel(n_qubits=3, bandwidth="alignment").fit(data, y)

    assert fitted.bandwidth_ == candidates[int(np.argmax(scores))]


def test_double_centering_matches_the_explicit_matrix(data):
    y = np.array([i % 2 for i in range(len(data))])
    gram = QuantumKernel(n_qubits=3, bandwidth=0.5).fit(data).compute_kernel_matrix(data)

    n = len(gram)
    centering = np.eye(n) - np.ones((n, n)) / n
    target = np.where(y[:, None] == y[None, :], 1.0, -1.0)
    gram_c = centering @ gram @ centering
    target_c = centering @ target @ centering
    expected = (gram_c * target_c).sum() / (
        np.linalg.norm(gram_c) * np.linalg.norm(target_c)
    )

    assert abs(kernel_diagnostics(gram, y)["kernel_target_alignment"] - expected) < 1e-12
