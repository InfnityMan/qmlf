"""The shared-simulation estimator must be a drop-in for the stock one.

It exists to stop simulating the same state once per observable. That is only
legitimate if the numbers are untouched, so these tests compare it against
qiskit's own estimator on the same circuits, and pin the cases where it has to
step aside: shot emulation and stochastic circuits.

Run with:  python -m pytest tests/ -q
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import real_amplitudes, zz_feature_map
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp

from qmlf.ops.shared_estimator import SharedStateEstimator

N_QUBITS = 3


@pytest.fixture(scope="module")
def problem():
    feature_map = zz_feature_map(feature_dimension=N_QUBITS, reps=1)
    ansatz = real_amplitudes(num_qubits=N_QUBITS, reps=2)
    circuit = feature_map.compose(ansatz)
    observables = [
        SparsePauliOp.from_list(
            [("I" * i + "Z" + "I" * (N_QUBITS - i - 1), 1.0)]
        )
        for i in range(N_QUBITS)
    ]
    values = np.random.default_rng(0).uniform(
        -1.0, 1.0, size=(4, circuit.num_parameters)
    )

    return circuit, observables, values


def _evs(estimator, circuit, observables, values, precision=0.0):
    pubs = [(circuit, observable, values) for observable in observables]
    result = estimator.run(pubs, precision=precision).result()

    return np.stack([pub.data.evs for pub in result])


def test_matches_the_stock_estimator_exactly(problem):
    circuit, observables, values = problem

    stock = _evs(StatevectorEstimator(), circuit, observables, values)
    shared = _evs(SharedStateEstimator(), circuit, observables, values)

    assert np.array_equal(stock, shared)


def test_multiple_observables_on_one_pub(problem):
    """Observables broadcast against parameters inside a single pub."""
    circuit, observables, values = problem
    pub = (circuit, observables, values[:1])

    stock = StatevectorEstimator().run([pub]).result()[0].data.evs
    shared = SharedStateEstimator().run([pub]).result()[0].data.evs

    assert np.array_equal(stock, shared)


def test_parameter_free_circuit():
    circuit = QuantumCircuit(1)
    circuit.h(0)
    observable = SparsePauliOp.from_list([("Z", 1.0)])

    stock = StatevectorEstimator().run([(circuit, observable)]).result()[0].data.evs
    shared = SharedStateEstimator().run([(circuit, observable)]).result()[0].data.evs

    assert np.array_equal(stock, shared)


def test_sampling_still_samples(problem):
    """precision > 0 emulates a device: the shortcut must step aside."""
    circuit, observables, values = problem
    estimator = SharedStateEstimator()

    first = _evs(estimator, circuit, observables, values, precision=0.05)
    second = _evs(estimator, circuit, observables, values, precision=0.05)

    assert not np.array_equal(first, second)
    assert np.allclose(first, second, atol=0.5)


def _reset_circuit():
    """A circuit whose reset outcome is visible in the expectation values.

    Entangled before the reset, so which branch the reset collapses to decides
    the answer, and read with several observables, so sharing one simulation
    across them would collapse what must be independent draws.
    """
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.reset(0)
    circuit.ry(0.7, 0)

    return circuit


@pytest.mark.parametrize("seed", [3, 7, 11])
def test_stochastic_circuit_falls_back(seed):
    """A reset makes simulation itself stochastic; sharing it would change it."""
    observables = [
        SparsePauliOp.from_list([(pauli, 1.0)]) for pauli in ("ZI", "IZ", "ZZ")
    ]
    pub = (_reset_circuit(), observables)

    shared = SharedStateEstimator(seed=seed).run([pub]).result()[0].data.evs
    stock = StatevectorEstimator(seed=seed).run([pub]).result()[0].data.evs

    assert np.array_equal(shared, stock)


def test_reset_inside_a_composite_instruction_falls_back():
    """The reset is one level down, where a name check cannot see it."""
    inner = _reset_circuit().to_instruction(label="prep")
    circuit = QuantumCircuit(2)
    circuit.append(inner, [0, 1])
    observables = [
        SparsePauliOp.from_list([(pauli, 1.0)]) for pauli in ("ZI", "IZ", "ZZ")
    ]
    pub = (circuit, observables)

    shared = SharedStateEstimator(seed=5).run([pub]).result()[0].data.evs
    stock = StatevectorEstimator(seed=5).run([pub]).result()[0].data.evs

    assert np.array_equal(shared, stock)


# The matching wiring guard -- that the three quantum layers actually use this
# estimator -- lives in test_chem_and_layers.py, which imports torch and
# qiskit-machine-learning at module scope. Importing them late, from inside a
# test in this module, has been observed to stall.
