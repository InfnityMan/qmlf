"""Batched statevector simulation for encoding circuits.

Qiskit's ``Statevector`` evolves one sample at a time, gate by gate. On the
default kernel path that per-sample simulation measured 99.4-99.8% of the
time of a 200-sample Gram matrix at 4-12 qubits (Kaggle CPU). This module lowers a circuit
ONCE into a short program of whole-array operations, then runs every sample
through it together:

* a stretch of diagonal gates -- ``p``/``rz``, ``cz``/``cp``/``rzz``, and the
  ``cx . p . cx`` sandwiches a ZZ feature map is built from -- collapses into a
  single phase multiply. ``cx`` and ``swap`` inside such a stretch are tracked
  as a linear map over GF(2) (which parity of the input bits each wire
  carries), so the phase lands on the right amplitudes without moving any.
  This is the phase-polynomial view of CNOT+phase circuits;
* every other single-qubit gate becomes a butterfly along one qubit's axis
  (a Hadamard on every qubit is therefore a Walsh-Hadamard transform).

A ZZ feature map then costs, per repetition, one butterfly per qubit plus one
phase multiply -- over the whole batch at once.

Layout. The state of a batch is held amplitude-major, ``(2**n, n_samples)``:
samples are the fastest-varying index, as threads would be on a GPU. A
butterfly on qubit q then works on contiguous runs of ``2**q * n_samples``
values; sample-major, the runs for qubits 0-2 were 1, 2 and 4 values long.
Measured on Kaggle against a copy differing only in layout (4-14 qubits):
amplitude-major is 1.4-1.6x faster on numpy, 1.3-1.5x on torch CPU and
1.3-1.6x on a T4, and bit-identical, since every scalar operation is the
same.

Exactness. In complex128 the states agreed with qiskit's to 3.4e-14 or better
over 177 circuits with angles in [-pi, pi], and to 7.7e-14 for unnormalised
features of size ~30: summing phase angles and exponentiating once rounds
differently from multiplying phase factors one gate at a time, and the error
grows with the size of the angles. Not bit-for-bit, which is why this is an
opt-in simulator and qiskit stays the default.

Batch invariance. Every operation is elementwise per sample, and every choice
of code path is made from the circuit, never from the batch's values -- so a
row's state is the same bits whether it is simulated alone or among ten
thousand others. Pinned by a test, because QuantumClassifier computes the same
rows inside different batches (CV folds, fit, predict).

Backends. ``numpy`` (complex128, CPU) and ``torch`` (complex128 on CPU and
CUDA; complex64 on Apple's MPS, which has no 64-bit floats at all). The two
backends compute cos/sin with different vector math, so their states agree
to rounding (3.9e-17 measured), not always to the bit.
"""
import numbers
from collections import namedtuple

import numpy as np
from qiskit.circuit import Barrier, Delay, ParameterExpression

_SQRT_HALF = 1.0 / np.sqrt(2.0)

# Rows are run through the program this many bytes (per real array) at a
# time, which bounds the temporaries to a few times this size however large
# the batch is. Rows are independent, so chunking cannot change any result.
# Swept at 4, 8 and 12 qubits on Kaggle's CPU, 4 MB timed between 16% faster
# and 3% slower than the whole batch at once; 256 KB chunks were up to 14%
# slower.
_CHUNK_BYTES = 1 << 22


class Unsupported(Exception):
    """The circuit uses something this simulator does not lower."""


# ---------------------------------------------------------------------------
# angles: every data-dependent gate parameter, evaluated as one table
# ---------------------------------------------------------------------------

# A gate angle that depends on the data: ``factor * table[index]``.
_Ref = namedtuple("_Ref", "index factor")

_PRINTER = None


def _exact_printer():
    """sympy's NumPyPrinter, but printing float constants with all 17 digits.

    The stock printer writes a sympy Float with 15 significant digits, so a
    constant like pi (as a Float) becomes 3.14159265358979 -- 3.2e-15 wrong
    before any arithmetic. Measured: a 4-qubit ZZ map with features of size
    ~30 then lands 1.7e-12 from qiskit's state; with all 17 digits, 7.7e-14.
    """
    global _PRINTER

    if _PRINTER is None:
        from sympy.printing.numpy import NumPyPrinter

        class ExactPrinter(NumPyPrinter):
            def _print_Float(self, expr):
                value = float(expr)
                return repr(value) if np.isfinite(value) else super()._print_Float(expr)

        _PRINTER = ExactPrinter

    return _PRINTER


def _sympify(parameter):
    try:
        import sympy  # imported here: only the fused path needs it
    except ImportError as exc:
        raise Unsupported("sympy is not installed") from exc

    try:
        expression = parameter.sympify()
    except Exception as exc:
        raise Unsupported(f"cannot translate the gate angle {parameter}") from exc

    # qiskit 2.5 returns the NotImplemented singleton, rather than raising,
    # for a constant raised to a parameter (2**x).
    if not isinstance(expression, sympy.Basic):
        raise Unsupported(f"cannot translate the gate angle {parameter}")

    return expression


class _Angles:
    """The data-dependent gate angles of one circuit.

    Lowering registers each one and refers to it by index; at run time a
    single generated numpy function evaluates all of them for the whole batch
    as an ``(n_angles, n_samples)`` table. Columns of the value matrix follow
    ``circuit.parameters`` -- the order ``assign_parameters(row)`` binds in.
    """

    def __init__(self, circuit):
        self._columns = {parameter.name: i for i, parameter in enumerate(circuit.parameters)}
        self._expressions = []  # sympy expressions; they pickle, unlike functions
        self._inputs = []       # value column feeding each argument of _function
        self._function = None   # compiled lazily, and again after unpickling

    def reference(self, parameter):
        """A gate parameter as a float constant, or a _Ref into the table."""
        if isinstance(parameter, ParameterExpression) and parameter.parameters:
            self._expressions.append(_sympify(parameter))
            self._function = None
            return _Ref(len(self._expressions) - 1, 1.0)

        try:
            return float(parameter)
        except (TypeError, ValueError) as exc:
            raise Unsupported(f"gate parameter {parameter!r} is not a real number") from exc

    def _compile(self):
        import sympy

        columns = {}
        for expression in self._expressions:
            for symbol in expression.free_symbols:
                if symbol.name not in self._columns:
                    raise Unsupported(f"angle uses {symbol.name!r}, not a circuit parameter")
                columns[symbol] = self._columns[symbol.name]

        arguments = sorted(columns, key=lambda symbol: (columns[symbol], symbol.name))
        # dummify: the generated function names its arguments _Dummy_N, not
        # after the parameters -- a parameter called "sin" or "exp" would
        # otherwise shadow the numpy function of that name.
        function = sympy.lambdify(
            arguments, self._expressions, modules="numpy",
            printer=_exact_printer(), dummify=True,
        )
        # Inputs first: a program is shared across threads, and another
        # thread may call table() the moment _function is set.
        self._inputs = [columns[symbol] for symbol in arguments]
        self._function = function

    def table(self, values):
        n_samples = len(values)
        table = np.empty((len(self._expressions), n_samples))

        if not self._expressions:
            return table

        if self._function is None:
            self._compile()

        results = self._function(*(values[:, column] for column in self._inputs))

        for k, result in enumerate(results):
            result = np.asarray(result)
            if np.iscomplexobj(result):
                raise Unsupported("a gate angle evaluates to a complex number")
            table[k] = result  # broadcasts an expression that came out constant

        return table

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_function"] = None  # generated code; recompiled on first use
        return state


def _evaluate(angle, table, n_samples):
    if isinstance(angle, _Ref):
        row = table[angle.index]
        return row if angle.factor == 1.0 else angle.factor * row

    return np.full(n_samples, angle)


def _scaled(angle, factor):
    if isinstance(angle, _Ref):
        return _Ref(angle.index, angle.factor * factor)

    return factor * angle


# ---------------------------------------------------------------------------
# lowering
# ---------------------------------------------------------------------------

# 2x2 matrices for the non-diagonal single-qubit gates, as functions of the
# gate's (already evaluated) parameter arrays.
def _rx(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return [[c, -1j * s], [-1j * s, c]]


def _ry(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return [[c, -s], [s, c]]


def _u(theta, phi, lam):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return [
        [c, -np.exp(1j * lam) * s],
        [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c],
    ]


def _u2(phi, lam):
    return _u(np.full_like(phi, np.pi / 2), phi, lam)


_FIXED_1Q = {
    "h": [[_SQRT_HALF, _SQRT_HALF], [_SQRT_HALF, -_SQRT_HALF]],
    "x": [[0, 1], [1, 0]],
    "y": [[0, -1j], [1j, 0]],
    "sx": [[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]],
    "sxdg": [[0.5 - 0.5j, 0.5 + 0.5j], [0.5 + 0.5j, 0.5 - 0.5j]],
}

# name -> (arity, builder, whether the matrix is real for EVERY angle). The
# real-arithmetic shortcut is chosen per gate type, never from the values in
# a batch: deciding it from the batch would let one row's angles pick the
# formula another row is computed with.
_PARAMETRIC_1Q = {
    "rx": (1, _rx, False),
    "ry": (1, _ry, True),
    "u": (3, _u, False),
    "u3": (3, _u, False),
    "u2": (2, _u2, False),
}

# Diagonal single-qubit gates: phase angle applied when the wire reads 1.
_FIXED_PHASE = {"z": np.pi, "s": np.pi / 2, "sdg": -np.pi / 2,
                "t": np.pi / 4, "tdg": -np.pi / 4}

_SUPPORTED = ({"id", "p", "u1", "rz", "cz", "cp", "cu1", "rzz", "cx", "swap"}
              | set(_FIXED_1Q) | set(_PARAMETRIC_1Q) | set(_FIXED_PHASE))

_STANDARD_BASES = None


def _is_library_gate(operation):
    """The operation is qiskit's own gate of that name, not a lookalike.

    Matching on the name alone would let a custom Gate that calls itself
    "rx" be simulated as an RX. ``base_class`` sees through qiskit's singleton
    and labelled variants of the same gate.
    """
    global _STANDARD_BASES

    if _STANDARD_BASES is None:
        from qiskit.circuit.library import get_standard_gate_name_mapping

        _STANDARD_BASES = {name: gate.base_class
                           for name, gate in get_standard_gate_name_mapping().items()}

    base = _STANDARD_BASES.get(operation.name)
    return base is not None and getattr(operation, "base_class", None) is base


class _Segment:
    """A stretch of diagonal and linear-reversible gates, then a 1q gate."""

    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        # Wire q currently carries the parity of the segment's input bits
        # selected by wires[q]; cx and swap update these masks.
        self.wires = [1 << q for q in range(n_qubits)]
        self.terms = []         # (pattern key, angle)
        self.global_terms = []  # angles added to every amplitude
        self.gather = None      # set by Program when the wires end permuted

    def is_empty(self):
        identity = all(mask == 1 << q for q, mask in enumerate(self.wires))
        return identity and not self.terms and not self.global_terms


def lower(circuit):
    """Compile ``circuit`` into a :class:`Program`, or raise Unsupported."""
    n = circuit.num_qubits
    angles = _Angles(circuit)
    steps = []
    segment = _Segment(n)

    def flush():
        nonlocal segment
        if not segment.is_empty():
            steps.append(("diagonal", segment))
        segment = _Segment(n)

    if circuit.global_phase:
        segment.global_terms.append(angles.reference(circuit.global_phase))

    for instruction in circuit.data:
        operation = instruction.operation
        name = operation.name

        if isinstance(operation, (Barrier, Delay)):
            continue

        if instruction.clbits:
            raise Unsupported(f"{name!r} acts on classical bits")

        # Checked BEFORE reading params: a gate outside the set (UnitaryGate,
        # Initialize, ...) may carry params that are not angles at all.
        if name not in _SUPPORTED or not _is_library_gate(operation):
            raise Unsupported(f"gate {name!r}")

        if name == "id":
            continue

        qubits = [circuit.find_bit(qubit).index for qubit in instruction.qubits]
        params = [angles.reference(parameter) for parameter in operation.params]
        wires = segment.wires

        if name in ("p", "u1"):
            segment.terms.append((("lin", wires[qubits[0]]), params[0]))
        elif name == "rz":
            segment.terms.append((("lin", wires[qubits[0]]), params[0]))
            segment.global_terms.append(_scaled(params[0], -0.5))
        elif name in _FIXED_PHASE:
            segment.terms.append((("lin", wires[qubits[0]]), _FIXED_PHASE[name]))
        elif name in ("cz", "cp", "cu1"):
            angle = np.pi if name == "cz" else params[0]
            a, b = sorted((wires[qubits[0]], wires[qubits[1]]))
            segment.terms.append((("and", a, b), angle))
        elif name == "rzz":
            # exp(-i t/2 Z(x)Z) = exp(-i t/2) * exp(i t [a XOR b])
            segment.terms.append((("lin", wires[qubits[0]] ^ wires[qubits[1]]), params[0]))
            segment.global_terms.append(_scaled(params[0], -0.5))
        elif name == "cx":
            wires[qubits[1]] ^= wires[qubits[0]]
        elif name == "swap":
            wires[qubits[0]], wires[qubits[1]] = wires[qubits[1]], wires[qubits[0]]
        elif name == "h":
            flush()
            if steps and steps[-1][0] == "hadamard" and qubits[0] not in steps[-1][1]:
                steps[-1][1].append(qubits[0])  # same layer: disjoint qubits commute
            else:
                steps.append(("hadamard", [qubits[0]]))
        elif name in _FIXED_1Q:
            flush()
            matrix = np.asarray(_FIXED_1Q[name], dtype=complex)
            steps.append(("gate", qubits[0], matrix, not np.any(matrix.imag)))
        else:
            arity, build, real = _PARAMETRIC_1Q[name]
            if len(params) != arity:
                raise Unsupported(f"{name!r} with {len(params)} parameters")
            flush()
            steps.append(("gate", qubits[0], (build, params), real))

    flush()

    program = Program(n, len(circuit.parameters), steps, angles)

    # Evaluate every angle once now, so an expression numpy cannot evaluate
    # surfaces here -- where the caller falls back to qiskit -- rather than
    # in the middle of a kernel computation.
    probe = np.linspace(0.1, 0.9, max(1, program.n_parameters))[None, :program.n_parameters]
    try:
        with np.errstate(all="ignore"):
            angles.table(probe)
    except Unsupported:
        raise
    except Exception as exc:
        raise Unsupported(f"cannot evaluate the gate angles ({exc})") from exc

    return program


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------


class Program:
    """A lowered circuit: run it on a batch of parameter rows.

    Holds no reference to the circuit and no generated code that survives
    pickling, so a fitted kernel can be pickled, deep-copied or shipped to a
    worker process and still compute the same states.
    """

    def __init__(self, n_qubits, n_parameters, steps, angles):
        self.n_qubits = n_qubits
        self.n_parameters = n_parameters
        self.steps = steps
        self._angles = angles
        self._index = np.arange(2 ** n_qubits)
        self._bits = {}  # mask -> 0/1 parity pattern over basis index

        for step in steps:
            if step[0] == "diagonal":
                self._prepare_segment(step[1])

        self._index = self._bits = None  # only needed while preparing

    def _parity(self, mask):
        pattern = self._bits.get(mask)

        if pattern is None:
            bits = np.zeros(len(self._index), dtype=np.int64)
            for q in range(self.n_qubits):
                if mask >> q & 1:
                    bits ^= (self._index >> q) & 1
            pattern = self._bits[mask] = bits.astype(float)

        return pattern

    def _prepare_segment(self, segment):
        # Reduce every phase term to parities of the segment's input bits:
        # a "lin" term on mask m contributes angle * parity(z & m); a product
        # of two parities (cz/cp) is (p_a + p_b - p_{a^b}) / 2, since for bits
        # a*b == (a + b - (a XOR b)) / 2. Mask 0 is the empty parity, always
        # 0, so it contributes nothing -- NOT a global phase.
        masks = {}  # mask -> [(weight, angle), ...]

        def add(mask, weight, angle):
            if mask:
                masks.setdefault(mask, []).append((weight, angle))

        for key, angle in segment.terms:
            if key[0] == "lin":
                add(key[1], 1.0, angle)
            elif key[1] == key[2]:
                add(key[1], 1.0, angle)  # p * p == p
            else:
                add(key[1], 0.5, angle)
                add(key[2], 0.5, angle)
                add(key[1] ^ key[2], -0.5, angle)

        segment.masks = sorted(masks.items())

        # Few distinct masks: one pass each. Many (a fully entangled ZZ map
        # has q + q(q-1)/2): build the coefficient vector and take its
        # Walsh-Hadamard transform, which costs q passes however many masks
        # there are, using sum_m c_m parity(z & m) == (sum_m c_m - (W c)(z)) / 2.
        segment.use_transform = len(segment.masks) > self.n_qubits

        if not segment.use_transform:
            segment.patterns = [self._parity(mask) for mask, _ in segment.masks]

        # The input basis state z ends up at L(z): output bit q is the parity
        # wires[q] selects. Phases are applied in the input basis, then the
        # amplitudes are gathered into place. Stored ON the segment: a lookup
        # keyed by id(segment) would silently miss in any copy of the program.
        if any(mask != 1 << q for q, mask in enumerate(segment.wires)):
            destination = np.zeros(len(self._index), dtype=np.int64)
            for q, mask in enumerate(segment.wires):
                destination |= self._parity(mask).astype(np.int64) << q
            gather = np.empty_like(destination)
            gather[destination] = self._index
            segment.gather = gather

    def _segment_angles(self, segment, table, n_samples):
        dimension = 2 ** self.n_qubits
        coefficients = []

        for _, contributions in segment.masks:
            coefficient = np.zeros(n_samples)
            for weight, angle in contributions:
                coefficient = coefficient + weight * _evaluate(angle, table, n_samples)
            coefficients.append(coefficient)

        offset = np.zeros(n_samples)
        for angle in segment.global_terms:
            offset = offset + _evaluate(angle, table, n_samples)

        if segment.use_transform:
            vector = np.zeros((dimension, n_samples))
            total = np.zeros(n_samples)
            for (mask, _), coefficient in zip(segment.masks, coefficients):
                vector[mask] = coefficient
                total = total + coefficient
            transformed = _walsh_hadamard(vector, self.n_qubits)
            return (0.5 * total + offset)[None, :] - 0.5 * transformed

        angles = np.zeros((dimension, n_samples))
        for pattern, coefficient in zip(segment.patterns, coefficients):
            angles += pattern[:, None] * coefficient[None, :]

        return angles + offset[None, :]

    def _gate_matrix(self, payload, table, n_samples):
        if isinstance(payload, np.ndarray):
            return payload  # fixed 2x2

        build, params = payload
        matrix = build(*(_evaluate(angle, table, n_samples) for angle in params))

        # (n_samples, 2, 2), broadcasting any constant entries.
        return np.stack([
            np.stack([np.broadcast_to(np.asarray(entry, dtype=complex), (n_samples,))
                      for entry in row], axis=-1)
            for row in matrix
        ], axis=-2)

    def statevectors(self, values, backend="numpy", device=None):
        """States for every row of ``values`` (shape ``(n_samples, n_parameters)``).

        Returns a numpy complex128 array for ``backend="numpy"``, or a torch
        tensor on ``device`` for ``backend="torch"``.
        """
        values = np.ascontiguousarray(values, dtype=float)

        if values.ndim != 2 or values.shape[1] != self.n_parameters:
            raise ValueError(
                f"expected values of shape (n, {self.n_parameters}), got {values.shape}"
            )

        if backend == "numpy":
            ops = _NumpyOps()
        elif backend == "torch":
            ops = _TorchOps(device)
        else:
            raise ValueError(f"unknown backend {backend!r}")

        n_samples, dimension = len(values), 2 ** self.n_qubits
        table = self._angles.table(values)
        out = ops.empty(n_samples, dimension)
        rows = max(1, _CHUNK_BYTES // (8 * dimension))

        for start in range(0, n_samples, rows):
            stop = min(start + rows, n_samples)
            ops.store(out, start, self._run(table[:, start:stop], stop - start, ops))

        return out

    def _run(self, table, n_samples, ops):
        dimension = 2 ** self.n_qubits
        steps = self.steps

        if steps and steps[0][0] == "hadamard" and len(steps[0][1]) == self.n_qubits:
            # A Hadamard on every qubit of |0...0> is the uniform state; no
            # need to run n butterflies to find that out.
            state = ops.uniform(n_samples, dimension, 2.0 ** (-self.n_qubits / 2))
            steps = steps[1:]
        else:
            state = ops.basis_zero(n_samples, dimension)

        for step in steps:
            if step[0] == "hadamard":
                state = ops.hadamard(state, step[1], self.n_qubits)
            elif step[0] == "diagonal":
                segment = step[1]
                angles = self._segment_angles(segment, table, n_samples)
                state = ops.apply_phase(state, angles)
                if segment.gather is not None:
                    state = ops.gather(state, segment.gather)
            else:
                _, qubit, payload, real = step
                matrix = self._gate_matrix(payload, table, n_samples)
                state = ops.apply_1q(state, qubit, self.n_qubits, matrix, real)

        return state


# ---------------------------------------------------------------------------
# array backends -- the same operations on numpy and on torch
#
# Both carry the state as SEPARATE real and imaginary arrays and spell every
# complex product out in real multiplies and adds. That is what makes the
# simulation batch-invariant. numpy's complex multiply is not, for two reasons
# that combine: it fuses one of its products (x * y and y * x differed in the
# last bit for 32.9% of 100,000 random pairs), and numpy's temporary elision
# reuses a temporary operand of 256 KiB or more as the output, which can swap
# the operand order once the batch is big enough. Whether that swap happens
# depends on the numpy build (Kaggle's numpy 2.1.3 never did it in testing),
# so this code does not rely on it either way.
# Real multiplies and adds are commutative and correctly rounded, so in this
# form each amplitude depends on its own sample and nothing else.
# ---------------------------------------------------------------------------


def _split(n_samples, qubit, n_qubits):
    # State tables are (2**n, n_samples): amplitude-major, samples fastest.
    # Little-endian basis index: qubit q is the bit worth 2**q, so viewing
    # the table as (2**(n-1-q), 2, 2**q, n_samples) puts qubit q on axis 1
    # -- and every slice keeps a contiguous run of 2**q * n_samples values.
    return (2 ** (n_qubits - 1 - qubit), 2, 2 ** qubit, n_samples)


def _entries(matrix):
    """The four entries of a 2x2 (or per-sample (n, 2, 2)) matrix as
    (real, imag) pairs, shaped to broadcast against (n, A, B) blocks."""
    if matrix.ndim == 2:
        return [(float(matrix[i, j].real), float(matrix[i, j].imag))
                for i, j in ((0, 0), (0, 1), (1, 0), (1, 1))]

    return [(matrix[:, i, j].real.copy(), matrix[:, i, j].imag.copy())
            for i, j in ((0, 0), (0, 1), (1, 0), (1, 1))]


def _butterfly(ar, ai, br, bi, first, second, real_only):
    """(first * a + second * b) in real arithmetic; first/second are (re, im)."""
    (fr, fi), (sr, si) = first, second

    if real_only:
        return fr * ar + sr * br, fr * ai + sr * bi

    return ((fr * ar - fi * ai) + (sr * br - si * bi),
            (fr * ai + fi * ar) + (sr * bi + si * br))


def _walsh_hadamard(vector, n_qubits):
    """Unnormalised Walsh-Hadamard transform along the last axis (real)."""
    n_samples = vector.shape[1]
    for qubit in range(n_qubits):
        view = vector.reshape(_split(n_samples, qubit, n_qubits))
        a, b = view[:, 0], view[:, 1]
        top, bottom = a + b, a - b
        view[:, 0], view[:, 1] = top, bottom
    return vector


class _NumpyOps:
    def empty(self, n_samples, dimension):
        return np.empty((n_samples, dimension), dtype=complex)

    def store(self, out, start, state):
        real, imag = state
        out.real[start:start + real.shape[1]] = real.T
        out.imag[start:start + imag.shape[1]] = imag.T

    def uniform(self, n_samples, dimension, amplitude):
        return np.full((dimension, n_samples), amplitude), np.zeros((dimension, n_samples))

    def hadamard(self, state, qubits, n_qubits):
        # Sum-and-difference butterflies, then ONE scale by 2**(-k/2): for an
        # even number of qubits that scale is a power of two and exact, and it
        # replaces k rounded multiplications by 1/sqrt(2).
        # In place: every state array reaching here was freshly produced by
        # the previous step, so nothing else holds a reference to it.
        real, imag = state
        for qubit in qubits:
            for part in (real, imag):
                view = part.reshape(_split(part.shape[1], qubit, n_qubits))
                a, b = view[:, 0], view[:, 1]
                top, bottom = a + b, a - b
                view[:, 0], view[:, 1] = top, bottom
        scale = 2.0 ** (-len(qubits) / 2)
        return real * scale, imag * scale

    def basis_zero(self, n_samples, dimension):
        real = np.zeros((dimension, n_samples))
        real[0] = 1.0
        return real, np.zeros((dimension, n_samples))

    def apply_phase(self, state, angles):
        real, imag = state
        cos, sin = np.cos(angles), np.sin(angles)
        return real * cos - imag * sin, real * sin + imag * cos

    def gather(self, state, gather):
        real, imag = state
        return real[gather], imag[gather]

    def apply_1q(self, state, qubit, n_qubits, matrix, real_only):
        real, imag = state
        shape = _split(real.shape[1], qubit, n_qubits)
        vr, vi = real.reshape(shape), imag.reshape(shape)
        ar, br, ai, bi = vr[:, 0], vr[:, 1], vi[:, 0], vi[:, 1]
        m00, m01, m10, m11 = _entries(matrix)

        out_r, out_i = np.empty(shape), np.empty(shape)
        out_r[:, 0], out_i[:, 0] = _butterfly(ar, ai, br, bi, m00, m01, real_only)
        out_r[:, 1], out_i[:, 1] = _butterfly(ar, ai, br, bi, m10, m11, real_only)

        return out_r.reshape(real.shape), out_i.reshape(imag.shape)


class _TorchOps:
    def __init__(self, device):
        import torch

        self.torch = torch
        self.device = resolve_device(device)
        # MPS has no 64-bit floats, so complex128 does not exist there.
        self.real = torch.float32 if self.device.type == "mps" else torch.float64
        self.complex = torch.complex64 if self.device.type == "mps" else torch.complex128

    def _tensor(self, array, dtype=None):
        return self.torch.as_tensor(
            np.ascontiguousarray(array), dtype=dtype or self.real, device=self.device
        )

    def empty(self, n_samples, dimension):
        return self.torch.empty((n_samples, dimension), dtype=self.complex, device=self.device)

    def store(self, out, start, state):
        real, imag = state
        out[start:start + real.shape[1]] = self.torch.complex(real, imag).T

    def uniform(self, n_samples, dimension, amplitude):
        real = self.torch.full((dimension, n_samples), amplitude, dtype=self.real, device=self.device)
        return real, self.torch.zeros_like(real)

    def hadamard(self, state, qubits, n_qubits):
        # In place: every state array reaching here was freshly produced by
        # the previous step, so nothing else holds a reference to it.
        real, imag = state
        for qubit in qubits:
            for part in (real, imag):
                view = part.reshape(_split(part.shape[1], qubit, n_qubits))
                a, b = view[:, 0], view[:, 1]
                top, bottom = a + b, a - b
                view[:, 0], view[:, 1] = top, bottom
        scale = 2.0 ** (-len(qubits) / 2)
        return real * scale, imag * scale

    def basis_zero(self, n_samples, dimension):
        real = self.torch.zeros((dimension, n_samples), dtype=self.real, device=self.device)
        real[0] = 1.0
        return real, self.torch.zeros_like(real)

    def apply_phase(self, state, angles):
        real, imag = state

        if self.real == self.torch.float32:
            # Reduce in float64 BEFORE narrowing. The phases of a 12-qubit ZZ
            # map measured up to ~900 rad (features in [-1, 1]) and ~1500 rad
            # (in [-pi, pi]), where float32 rounds by up to 3e-5 and 6e-5 rad.
            # Reduced to [-pi, pi], the rounding is at most 1.2e-7.
            angles = angles - 2.0 * np.pi * np.round(angles / (2.0 * np.pi))

        angles = self._tensor(angles)
        cos, sin = self.torch.cos(angles), self.torch.sin(angles)
        return real * cos - imag * sin, real * sin + imag * cos

    def gather(self, state, gather):
        index = self._tensor(gather, self.torch.long)
        real, imag = state
        return real[index], imag[index]

    def apply_1q(self, state, qubit, n_qubits, matrix, real_only):
        real, imag = state
        shape = _split(real.shape[1], qubit, n_qubits)
        vr, vi = real.reshape(shape), imag.reshape(shape)
        ar, br, ai, bi = vr[:, 0], vr[:, 1], vi[:, 0], vi[:, 1]
        entries = _entries(matrix)

        if matrix.ndim == 3:
            entries = [(self._tensor(re), self._tensor(im)) for re, im in entries]

        m00, m01, m10, m11 = entries
        top = _butterfly(ar, ai, br, bi, m00, m01, real_only)
        bottom = _butterfly(ar, ai, br, bi, m10, m11, real_only)

        return (self.torch.stack((top[0], bottom[0]), dim=1).reshape(real.shape),
                self.torch.stack((top[1], bottom[1]), dim=1).reshape(imag.shape))


def resolve_device(device=None):
    """A torch.device for ``device``, or the best available when None.

    Fails here, with a clear message, for a device torch cannot parse or one
    this machine does not have -- not at the first kernel computation.
    """
    import torch

    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    try:
        resolved = torch.device(device)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"not a torch device: {device!r}") from exc

    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("device='cuda' but torch reports no CUDA device")

    if resolved.type == "mps" and not (getattr(torch.backends, "mps", None) is not None
                                      and torch.backends.mps.is_available()):
        raise ValueError("device='mps' but torch reports MPS unavailable")

    return resolved


def as_tensor(states, device=None):
    """A numpy state table as a torch tensor on ``device`` (a no-op for a tensor)."""
    import torch

    if isinstance(states, torch.Tensor):
        return states

    return torch.as_tensor(np.ascontiguousarray(states), device=resolve_device(device))


def to_numpy(states):
    """A state table as numpy, wherever it lives."""
    if isinstance(states, np.ndarray):
        return states

    return states.resolve_conj().to("cpu").numpy()


# ---------------------------------------------------------------------------
# kernels from batched states
# ---------------------------------------------------------------------------


def _check_batch_size(batch_size):
    if batch_size is None:
        return

    if isinstance(batch_size, bool) or not isinstance(batch_size, numbers.Integral) \
            or batch_size < 1:
        raise ValueError(f"batch_size must be a positive integer or None, got {batch_size!r}")


def fidelity_gram(states_a, states_b, batch_size=None):
    """|<a_i|b_j>|^2 for numpy or torch state tables; returns float64 numpy.

    Torch tables multiply on their own device. In single precision (Apple
    MPS) each entry sums 2**n_qubits float32 products, so the rounding grows
    with the qubit count and depends on ``batch_size``, and near-1 entries can
    come out above 1; those are clipped to the [0, 1] a fidelity lies in.
    """
    _check_batch_size(batch_size)
    n_rows = len(states_a)
    step = max(1, n_rows if batch_size is None else int(batch_size))

    if isinstance(states_a, np.ndarray):
        blocks = [np.abs(states_a[start:start + step].conj() @ states_b.T) ** 2
                  for start in range(0, n_rows, step)]
        return blocks[0] if len(blocks) == 1 else (
            np.vstack(blocks) if blocks else np.zeros((0, len(states_b))))

    blocks = []
    for start in range(0, n_rows, step):
        # resolve_conj(): torch's conj() is a lazy flag, and Apple MPS in
        # torch 2.8 ignored it in BATCHED complex matmuls, computing a @ b
        # instead of conj(a) @ b. Materialising the conjugate removes the
        # dependence on every backend honouring the flag.
        conjugated = states_a[start:start + step].conj().resolve_conj()
        block = (conjugated @ states_b.T).abs() ** 2
        if block.element_size() < 8:  # single precision
            block = block.clamp(max=1.0)
        blocks.append(block.to("cpu").double().numpy())

    return np.vstack(blocks) if blocks else np.zeros((0, len(states_b)))


def bloch_features(states, n_qubits):
    """Per-qubit Bloch vectors, ordered exactly like QuantumKernel's.

    QuantumKernel reshapes each state to (2,)*n and takes block j from tensor
    axis j (i.e. qubit n-1-j). Same order here, so the projected kernel sums
    its distances over the same sequence of terms. Reduced on the CPU in
    complex128 whatever the states' precision: on MPS batched complex matmul
    drops lazy conjugation (see fidelity_gram), and a float64 reduction is
    simply more accurate.
    """
    states = to_numpy(states)

    if states.dtype != np.complex128:
        states = states.astype(np.complex128)

    n_samples = states.shape[0]
    tensor = states.reshape((n_samples,) + (2,) * n_qubits)
    blocks = []

    for axis in range(n_qubits):
        matrix = np.moveaxis(tensor, axis + 1, 1).reshape(n_samples, 2, 2 ** (n_qubits - 1))
        rdm = matrix @ matrix.conj().transpose(0, 2, 1)

        blocks.append(np.stack([
            2.0 * rdm[:, 0, 1].real,
            -2.0 * rdm[:, 0, 1].imag,
            (rdm[:, 0, 0] - rdm[:, 1, 1]).real,
        ], axis=1))

    return np.concatenate(blocks, axis=1)
