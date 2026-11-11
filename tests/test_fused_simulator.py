"""The fused simulators must agree with qiskit, and stay opt-in.

qmlf.ops.fused lowers a circuit to whole-batch array operations. Its claims,
each pinned here: amplitudes agree with qiskit's Statevector to 1e-12 in
complex128 across every entanglement pattern and supported gate; a row's state
does not depend on which other rows share its batch (bit-for-bit); unsupported
circuits fall back to qiskit rather than failing; a fitted model survives
pickling and copying; the default simulator is still qiskit, so published
numbers do not move.

Run with:  python -m pytest tests/ -q
"""
import copy
import os
import pickle

os.environ.setdefault("OMP_NUM_THREADS", "1")

import warnings

import numpy as np
import pytest
from qiskit.circuit import Gate, Parameter, ParameterVector, QuantumCircuit
from qiskit.circuit.library import (CU1Gate, DiagonalGate, HGate, IGate, PauliGate,
                                    U1Gate, U2Gate, U3Gate, UnitaryGate,
                                    real_amplitudes, z_feature_map, zz_feature_map)
from qiskit.quantum_info import Statevector

import qmlf
from qmlf.models.q_models import QuantumClassifier
from qmlf.ops import fused
from qmlf.ops.fused import Unsupported, lower
from qmlf.ops.q_ops_core import QuantumKernel

TOL = 1e-12


def _reference(circuit, values):
    return np.stack([
        np.asarray(Statevector(circuit.assign_parameters(row)).data) for row in values
    ])


def _values(circuit, n_rows=6, seed=0):
    return np.random.default_rng(seed).uniform(
        -np.pi, np.pi, size=(n_rows, circuit.num_parameters)
    )


@pytest.mark.parametrize("entanglement",
                         ["full", "linear", "reverse_linear", "circular", "sca", "pairwise"])
@pytest.mark.parametrize("reps", [1, 2, 3])
def test_zz_feature_map_matches_qiskit(entanglement, reps):
    circuit = zz_feature_map(5, reps=reps, entanglement=entanglement)
    values = _values(circuit)

    assert np.max(np.abs(lower(circuit).statevectors(values) - _reference(circuit, values))) < TOL


@pytest.mark.parametrize("make", [
    lambda: z_feature_map(4, reps=2),
    lambda: real_amplitudes(4, reps=3),  # ry + cx: exercises the permutation path
    lambda: zz_feature_map(9, reps=2),   # enough masks to take the Walsh-Hadamard route
])
def test_other_circuits_match_qiskit(make):
    circuit = make()
    values = _values(circuit)

    assert np.max(np.abs(lower(circuit).statevectors(values) - _reference(circuit, values))) < TOL


def test_every_supported_gate_matches_qiskit():
    x = ParameterVector("x", 4)
    circuit = QuantumCircuit(4)
    circuit.h(range(4))
    circuit.rx(x[0], 0); circuit.ry(x[1] * 0.5, 1); circuit.rz(x[2], 2); circuit.p(x[3] ** 2, 3)
    circuit.cx(0, 1); circuit.cz(1, 2); circuit.cp(x[0] - x[1], 2, 3); circuit.rzz(x[2] * x[3], 0, 3)
    circuit.swap(0, 2); circuit.x(1); circuit.s(2); circuit.sdg(0); circuit.t(3); circuit.tdg(1)
    circuit.z(2); circuit.y(0); circuit.sx(3); circuit.sxdg(2); circuit.u(x[0], x[1], x[2], 1)
    circuit.cx(3, 0); circuit.cx(2, 1); circuit.ry(x[3], 2)
    circuit.append(IGate(), [0]); circuit.append(U1Gate(x[1]), [3])
    circuit.append(U2Gate(x[2], x[3]), [0]); circuit.append(U3Gate(x[3], x[0], x[1]), [2])
    circuit.append(CU1Gate(x[0] * x[2]), [1, 3])
    circuit.global_phase = x[0] * 0.3
    values = _values(circuit)

    used = {instruction.operation.name for instruction in circuit.data}
    assert used == fused._SUPPORTED  # every gate the simulator claims to lower

    assert np.max(np.abs(lower(circuit).statevectors(values) - _reference(circuit, values))) < TOL


def _bits(states):
    return np.asarray(states).view(np.uint64)


def test_rows_do_not_depend_on_their_batch(monkeypatch):
    """Bit-for-bit: a row simulated alone equals the same row in a batch.

    numpy's complex multiply fuses one of its products, so x * y and y * x
    differ in the last bit -- and numpy's temporary elision can swap the
    operand order once a temporary reaches 256 KiB (256 rows of a 6-qubit
    state, 256 * 64 * 16 bytes), depending on the numpy build. 300 rows keeps
    the test above that line. The simulator keeps real and imaginary parts
    separate for exactly this reason.
    Chunking is exercised too: rows are run a chunk at a time, and the chunk
    boundaries must not show either.
    """
    program = lower(zz_feature_map(6, reps=2))
    values = np.random.default_rng(1).uniform(-1.0, 1.0, size=(300, 6))

    together = program.statevectors(values)
    alone = np.stack([program.statevectors(values[i:i + 1])[0] for i in range(len(values))])
    assert np.array_equal(_bits(together), _bits(alone))

    monkeypatch.setattr(fused, "_CHUNK_BYTES", 4096)  # 8 rows per chunk
    assert np.array_equal(_bits(program.statevectors(values)), _bits(together))


def test_gate_formula_is_chosen_per_gate_not_per_batch():
    """u() is real for some angles; one row's angles must not pick the
    arithmetic another row is computed with (that moved signs of zero)."""
    u = ParameterVector("u", 3)
    circuit = QuantumCircuit(2)
    circuit.h(0); circuit.u(u[0], u[1], u[2], 0); circuit.cx(0, 1); circuit.u(u[0], u[1], u[2], 1)
    values = np.random.default_rng(4).uniform(-1.0, 1.0, size=(40, 3))
    values[::3, 1:] = 0.0  # these rows' u matrices are real
    program = lower(circuit)

    together = program.statevectors(values)
    alone = np.stack([program.statevectors(values[i:i + 1])[0] for i in range(len(values))])

    assert np.array_equal(_bits(together), _bits(alone))


class _LookalikeRX(Gate):
    """Calls itself "rx" but is an RY: lowering must not trust the name."""

    def __init__(self, theta):
        super().__init__("rx", 1, [theta])

    def _define(self):
        definition = QuantumCircuit(1)
        definition.ry(self.params[0], 0)
        self.definition = definition


def _with(gate):
    circuit = QuantumCircuit(gate.num_qubits)
    for i, parameter in enumerate(ParameterVector("w", gate.num_qubits)):
        circuit.ry(parameter, i)
    circuit.append(gate, range(gate.num_qubits))
    return circuit


@pytest.mark.parametrize("bad", ["ccx", "measure", "unitary", "pauli", "diagonal",
                                 "initialize", "exponent", "lookalike"])
def test_unsupported_circuits_are_rejected(bad):
    """Every one of these must raise Unsupported -- the exception the kernel
    catches to fall back -- not a TypeError or AttributeError from inside."""
    if bad in ("ccx", "measure"):
        circuit = QuantumCircuit(3, 1)
        circuit.ry(Parameter("a"), 0)
        if bad == "ccx":
            circuit.ccx(0, 1, 2)
        else:
            circuit.measure(0, 0)
    elif bad == "unitary":
        circuit = _with(UnitaryGate(np.eye(2)))      # params are matrices
    elif bad == "pauli":
        circuit = _with(PauliGate("XY"))              # params are strings
    elif bad == "diagonal":
        circuit = _with(DiagonalGate([1, 1j]))        # params are complex
    elif bad == "initialize":
        circuit = QuantumCircuit(2)
        circuit.initialize([1, 0, 0, 0])
        circuit.ry(Parameter("a"), 0)
    elif bad == "exponent":
        circuit = QuantumCircuit(1)
        circuit.ry(2 ** Parameter("a"), 0)            # qiskit's sympify gives NotImplemented
    else:
        circuit = QuantumCircuit(1)
        circuit.append(_LookalikeRX(Parameter("a")), [0])

    with pytest.raises(Unsupported):
        lower(circuit)


def test_labelled_library_gate_is_still_lowered():
    circuit = QuantumCircuit(2)
    h = ParameterVector("h", 2)
    circuit.append(HGate(label="mine"), [0]); circuit.h(1); circuit.rz(h[0], 0); circuit.rz(h[1], 1)
    values = _values(circuit)

    assert np.max(np.abs(lower(circuit).statevectors(values) - _reference(circuit, values))) < TOL


@pytest.mark.parametrize("name", ["sin", "exp", "cos", "arctan", "numpy"])
def test_parameter_names_cannot_shadow_numpy(name):
    a, b = Parameter(name), Parameter("other")
    circuit = QuantumCircuit(1)
    circuit.h(0); circuit.p(a.sin() * b + a, 0)
    values = _values(circuit)

    assert np.max(np.abs(lower(circuit).statevectors(values) - _reference(circuit, values))) < TOL


def test_float_constants_keep_every_digit():
    """sympy prints a Float with 15 digits by default; 1000*pi*x then lands
    2e-12 away from qiskit. With all 17 it is within rounding."""
    x = Parameter("x")
    circuit = QuantumCircuit(1)
    circuit.h(0); circuit.p(1000 * np.pi * x, 0)
    values = np.array([[1.0], [0.37]])

    assert np.max(np.abs(lower(circuit).statevectors(values) - _reference(circuit, values))) < 1e-15

    zz = zz_feature_map(4, reps=2)
    wide = np.array([[30.0, 28.7, 20.4, 5.9]])  # unnormalised features
    assert np.max(np.abs(lower(zz).statevectors(wide) - _reference(zz, wide))) < 1e-12


def test_unsupported_feature_map_falls_back_to_qiskit():
    a, b, c = Parameter("a"), Parameter("b"), Parameter("c")
    circuit = QuantumCircuit(3)  # one parameter per qubit, as QuantumKernel requires
    circuit.ry(a, 0); circuit.ry(b, 1); circuit.ry(c, 2); circuit.ccx(0, 1, 2); circuit.rz(a, 2)
    X = np.random.default_rng(2).uniform(-1.0, 1.0, size=(8, 3))

    exact = QuantumKernel(feature_map=circuit, bandwidth=0.5).fit(X).compute_kernel_matrix(X)

    with pytest.warns(UserWarning, match="falling back to the qiskit simulator"):
        fallback = QuantumKernel(feature_map=circuit, bandwidth=0.5, simulator="fused") \
            .fit(X).compute_kernel_matrix(X)

    assert np.array_equal(exact, fallback)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(3)
    return rng.uniform(-1.0, 1.0, size=(24, 4)), np.array([i % 2 for i in range(24)])


@pytest.mark.parametrize("kernel", ["fidelity", "projected"])
@pytest.mark.parametrize("mode", ["ZZ", "mahalanobis"])
def test_kernels_match_the_default_simulator(data, kernel, mode):
    X, _ = data
    kwargs = dict(n_qubits=4, bandwidth=0.4, kernel=kernel, mode=mode,
                  normalize="maxabs" if mode != "ZZ" else None)

    exact = QuantumKernel(**kwargs).fit(X)
    fast = QuantumKernel(**kwargs, simulator="fused").fit(X)

    assert np.max(np.abs(exact.compute_kernel_matrix(X) - fast.compute_kernel_matrix(X))) < TOL
    assert np.max(np.abs(
        exact.compute_kernel_matrix(X[:5], X) - fast.compute_kernel_matrix(X[:5], X, batch_size=2)
    )) < TOL


def test_classifier_makes_the_same_decisions(data):
    X, y = data
    exact = QuantumClassifier(max_qubits=None).fit(X, y)
    fast = QuantumClassifier(max_qubits=None, simulator="fused").fit(X, y)

    assert exact.bandwidth_ == fast.bandwidth_
    assert exact.cv_results_["mean_cv_accuracy"] == fast.cv_results_["mean_cv_accuracy"]
    assert np.array_equal(exact.predict(X), fast.predict(X))


def test_default_simulator_is_unchanged(data):
    """Opt-in means opt-in: the default path is still qiskit's, bit for bit."""
    X, _ = data
    default = QuantumKernel(n_qubits=4, bandwidth=0.4).fit(X)

    assert default.simulator == "qiskit"
    assert default._fused_program() is None


def test_mutated_circuit_is_relowered(data):
    X, _ = data
    p = [Parameter(f"x{i}") for i in range(4)]
    circuit = QuantumCircuit(4)
    for i in range(4):
        circuit.ry(p[i], i)

    kernel = QuantumKernel(feature_map=circuit, bandwidth=0.5, simulator="fused").fit(X)
    kernel.compute_kernel_matrix(X)
    circuit.rz(p[1], 0)  # parameterised, so the fidelity actually changes

    expected = QuantumKernel(feature_map=circuit, bandwidth=0.5).fit(X).compute_kernel_matrix(X)

    assert np.max(np.abs(kernel.compute_kernel_matrix(X) - expected)) < TOL


def test_invalid_combinations_are_rejected():
    with pytest.raises(ValueError, match="Unknown simulator"):
        QuantumKernel(n_qubits=2, simulator="gpu")

    with pytest.raises(ValueError, match="sampler"):
        QuantumKernel(n_qubits=2, simulator="fused", sampler=object())

    with pytest.raises(ValueError, match="device"):
        QuantumKernel(n_qubits=2, simulator="fused", device="cpu")

    with pytest.raises(ValueError, match="fast_statevector"):
        QuantumKernel(n_qubits=2, simulator="fused", fast_statevector=False)


@pytest.mark.parametrize("simulator", ["qiskit", "fused"])
@pytest.mark.parametrize("batch_size", [0, -5, 2.5, True])
def test_bad_batch_size_is_rejected_on_every_path(data, simulator, batch_size):
    X, _ = data
    kernel = QuantumKernel(n_qubits=4, bandwidth=0.4, simulator=simulator).fit(X)

    with pytest.raises(ValueError, match="batch_size"):
        kernel.compute_kernel_matrix(X, batch_size=batch_size)


@pytest.mark.parametrize("kernel", ["fidelity", "projected"])
def test_empty_input_gives_an_empty_matrix(data, kernel):
    X, _ = data
    fast = QuantumKernel(n_qubits=4, bandwidth=0.4, kernel=kernel, simulator="fused").fit(X)

    assert fast.compute_kernel_matrix(X[:0], X).shape == (0, len(X))


def _permuting_circuit():
    """RY layer, an un-undone CX ladder, RY layer: the CX network leaves a
    permutation the simulator must apply as a gather."""
    x = ParameterVector("x", 4)
    circuit = QuantumCircuit(4)
    for i in range(4):
        circuit.ry(x[i], i)
    circuit.cx(0, 1); circuit.cx(1, 2); circuit.cx(2, 3)
    for i in range(4):
        circuit.ry(x[i] * 0.7, i)
    return circuit


@pytest.mark.parametrize("duplicate", [copy.deepcopy, lambda obj: pickle.loads(pickle.dumps(obj))],
                         ids=["deepcopy", "pickle"])
def test_copies_of_a_fitted_model_compute_the_same_kernel(data, duplicate):
    """The permutation used to be looked up by id(segment), which every copy
    changes: a deep-copied or unpickled model silently skipped it and
    computed different (wrong) states. Copies must now agree bit for bit,
    and pickling must work at all."""
    X, y = data
    circuit = _permuting_circuit()
    kernel = QuantumKernel(feature_map=circuit, bandwidth=0.8, simulator="fused").fit(X)
    gram = kernel.compute_kernel_matrix(X)
    qmlf.clear_state_cache()  # the copy must recompute, not read the original's rows

    assert np.array_equal(duplicate(kernel).compute_kernel_matrix(X), gram)
    assert np.max(np.abs(gram - QuantumKernel(feature_map=circuit, bandwidth=0.8)
                         .fit(X).compute_kernel_matrix(X))) < TOL

    model = QuantumClassifier(max_qubits=None, bandwidth=0.5, simulator="fused").fit(X, y)
    assert np.array_equal(duplicate(model).decision_function(X), model.decision_function(X))


def test_a_circuit_is_lowered_once_and_warned_about_once(data, monkeypatch):
    """A classifier sweep builds a kernel per fold and candidate around the
    same circuit; lowering it (and warning about it) per kernel dominated the
    fit and flooded stderr."""
    X, y = data
    calls = []
    real_lower = fused.lower
    monkeypatch.setattr(fused, "lower", lambda circuit: calls.append(1) or real_lower(circuit))

    unique = zz_feature_map(4, reps=2, parameter_prefix="once")
    for _ in range(5):
        QuantumKernel(feature_map=unique, bandwidth=0.3, simulator="fused").fit(X) \
            .compute_kernel_matrix(X)
    assert len(calls) == 1

    a, b, c = Parameter("wa"), Parameter("wb"), Parameter("wc")
    unsupported = QuantumCircuit(3)
    unsupported.ry(a, 0); unsupported.ry(b, 1); unsupported.ry(c, 2); unsupported.ccx(0, 1, 2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(5):
            QuantumKernel(feature_map=unsupported, bandwidth=0.3, simulator="fused") \
                .fit(X[:, :3]).compute_kernel_matrix(X[:5, :3])
    assert sum("falling back" in str(w.message) for w in caught) == 1


def test_fused_rows_are_reused_but_never_served_to_qiskit(data, monkeypatch):
    """Fused states go through the row cache like qiskit's (a predict or a
    second Gram simulates nothing new) -- under a key that names the
    simulator, so the default path keeps returning exactly qiskit's states."""
    X, _ = data
    qmlf.clear_state_cache()
    exact = QuantumKernel(n_qubits=4, bandwidth=0.4).fit(X).compute_kernel_matrix(X)

    simulated = []
    real = fused.Program.statevectors
    monkeypatch.setattr(fused.Program, "statevectors",
                        lambda self, values, *a, **k: simulated.append(len(values))
                        or real(self, values, *a, **k))

    fast = QuantumKernel(n_qubits=4, bandwidth=0.4, simulator="fused").fit(X)
    fast.compute_kernel_matrix(X)
    fast.compute_kernel_matrix(X[:5], X)
    assert sum(simulated) == len(X)

    qmlf.clear_state_cache()
    fast.compute_kernel_matrix(X)  # fused rows now cached; qiskit must not see them
    assert np.array_equal(QuantumKernel(n_qubits=4, bandwidth=0.4).fit(X).compute_kernel_matrix(X),
                          exact)


torch = pytest.importorskip("torch")


def test_torch_cpu_matches_numpy():
    """Same algorithm, two libraries: equal to rounding, not always to the bit.

    The phases go through cos/sin, which numpy and torch evaluate with their
    own vectorised math. Measured on Kaggle: bit-equal on an AMD EPYC 7B12; on
    two Intel Xeons, 18 and 42 of 2,048 amplitudes (this circuit, 64 rows)
    differed, by at most 3.9e-17.
    """
    circuit = zz_feature_map(5, reps=2)
    values = _values(circuit)
    program = lower(circuit)

    on_torch = program.statevectors(values, backend="torch", device="cpu").numpy()

    np.testing.assert_allclose(on_torch, program.statevectors(values), rtol=0, atol=1e-15)


def _devices():
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


@pytest.mark.parametrize("device", _devices())
def test_torch_rows_do_not_depend_on_their_batch(device):
    program = lower(zz_feature_map(6, reps=2))
    values = np.random.default_rng(1).uniform(-1.0, 1.0, size=(300, 6))

    together = program.statevectors(values, backend="torch", device=device).cpu().numpy()
    probes = [0, 1, 150, 298, 299]
    alone = np.stack([program.statevectors(values[i:i + 1], backend="torch", device=device)
                      .cpu().numpy()[0] for i in probes])

    view = np.uint64 if together.dtype == np.complex128 else np.uint32
    assert np.array_equal(together[probes].view(view), alone.view(view))


def test_bad_torch_device_fails_at_construction():
    with pytest.raises(ValueError, match="torch device"):
        QuantumKernel(n_qubits=2, simulator="torch", device="bogus")


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple MPS")
def test_mps_is_single_precision_but_correct(data):
    """MPS has no 64-bit floats; complex64 must still be right to 1e-5.

    Guards a torch 2.8 MPS trap: batched complex matmul ignored conj()'s lazy
    flag and silently computed a @ b instead of conj(a) @ b. The projected
    kernel's Bloch reduction is a batched product, so it must not run there
    with a lazy conjugate.
    """
    X, _ = data

    for kernel in ("fidelity", "projected"):
        exact = QuantumKernel(n_qubits=4, bandwidth=0.4, kernel=kernel).fit(X) \
            .compute_kernel_matrix(X)
        on_mps = QuantumKernel(n_qubits=4, bandwidth=0.4, kernel=kernel,
                               simulator="torch", device="mps").fit(X).compute_kernel_matrix(X)

        assert np.max(np.abs(exact - on_mps)) < 1e-5


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple MPS")
def test_mps_gram_at_twelve_qubits_is_within_its_documented_error():
    """Each MPS Gram entry sums 4096 float32 products, so rounding piles up
    and near-1 entries can land above 1 before clipping. None may exceed 1."""
    X = np.random.default_rng(5).uniform(-np.pi, np.pi, size=(256, 12))
    reference = QuantumKernel(n_qubits=12, bandwidth=1.0, simulator="fused").fit(X) \
        .compute_kernel_matrix(X)
    on_mps = QuantumKernel(n_qubits=12, bandwidth=1.0, simulator="torch", device="mps") \
        .fit(X).compute_kernel_matrix(X)

    assert np.max(np.abs(on_mps - reference)) < 5e-5
    assert on_mps.max() <= 1.0
