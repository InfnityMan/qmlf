import hashlib
import numbers
import os
import threading
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
from qiskit.circuit import Barrier, Delay, Gate, ParameterExpression, QuantumCircuit
from qiskit.circuit.library import z_feature_map, zz_feature_map
from qiskit.quantum_info import Statevector

from qmlf.ops import fused
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# NOTE: xgboost and plotly are imported lazily inside the functions that use
# them. Importing torch (pulled in by qmlf.qnn) before xgboost segfaults the
# interpreter on macOS, because torch and scikit-learn each vendor their own
# libomp.dylib. Keeping the heavy optional deps out of module scope means
# `from qmlf.ops.q_ops_core import QuantumKernel` stays cheap and safe.

# Mode aliases. "covariant" is kept as the original spelling; "mahalanobis" is
# the accurate name and is preferred in new code (see the class docstring).
_COVARIANT_MODES = ("covariant", "mahalanobis")

# All modes that whiten the data before encoding. "fisher" whitens by the
# WITHIN-CLASS scatter instead of the global covariance, so directions that
# vary inside a class (noise) are compressed and directions that separate the
# classes keep their scale -- supervised metric learning inside the kernel.
_WHITENED_MODES = _COVARIANT_MODES + ("fisher",)
_VALID_MODES = ("ZZ",) + _WHITENED_MODES

_VALID_KERNELS = ("fidelity", "projected")

_BUILTIN_FEATURE_MAPS = ("zz", "z")

# "qiskit" simulates each sample gate by gate through qiskit's Statevector --
# the reference, and bit-for-bit what every published number was computed
# with. "fused" (numpy, complex128) and "torch" (complex128 on CPU/CUDA,
# complex64 on Apple MPS) run the whole batch through qmlf.ops.fused instead.
_VALID_SIMULATORS = ("qiskit", "fused", "torch")

# ---------------------------------------------------------------------------
# Simulated-state cache
#
# Simulating one statevector per sample is the dominant cost of every
# evaluation path, and the same rows are simulated over and over: a cross
# kernel re-simulates the training rows it was just fitted on, a CV sweep
# re-simulates each fold's training rows once per fold, and every predict()
# call re-simulates the entire training set. The state a row encodes to is a
# pure function of (encoding circuit, angles), so reusing it is exact rather
# than approximate -- entries are keyed by both, and eviction costs time, never
# accuracy. Whitened modes prepare different angles per fold and simply miss,
# which is the correct behaviour.
#
# The limit is a byte budget (QMLF_STATE_CACHE_MB, default 64) because one
# entry is 2**n_qubits complex amplitudes: 256 bytes at 4 qubits, 64 KB at 12.
# ---------------------------------------------------------------------------
_STATE_CACHE = OrderedDict()
_STATE_CACHE_BYTES = 0
# The dict, the byte counter and the budget are module-level mutable state, and
# a caller may well reach them from several threads (joblib's threading
# backend, a thread per request in a service). Every read-modify-write below
# takes this lock: without it, "is the dict non-empty" followed by popitem, or
# .get followed by .move_to_end, can lose the race and raise KeyError out of
# the middle of compute_kernel_matrix.
_CACHE_LOCK = threading.Lock()


def _budget_from_environment():
    raw = os.environ.get("QMLF_STATE_CACHE_MB", "64")

    try:
        megabytes = max(0.0, float(raw))
    except (TypeError, ValueError):
        # A performance knob must never be the reason `import qmlf` fails.
        warnings.warn(
            f"QMLF_STATE_CACHE_MB={raw!r} is not a number; using the default 64 MB.",
            UserWarning,
            stacklevel=2,
        )
        megabytes = 64.0

    return int(megabytes * 1024 ** 2)


_STATE_CACHE_LIMIT = _budget_from_environment()


def state_cache_info():
    """Entries, bytes held, and the byte budget of the simulated-state cache."""
    with _CACHE_LOCK:
        return {
            "entries": len(_STATE_CACHE),
            "bytes": _STATE_CACHE_BYTES,
            "limit_bytes": _STATE_CACHE_LIMIT,
        }


def clear_state_cache():
    """Drop every cached statevector. Results are unaffected, only timings."""
    global _STATE_CACHE_BYTES

    with _CACHE_LOCK:
        _STATE_CACHE.clear()
        _STATE_CACHE_BYTES = 0


def set_state_cache_limit(megabytes):
    """Resize the cache budget; ``0`` disables caching entirely."""
    global _STATE_CACHE_LIMIT

    with _CACHE_LOCK:
        _STATE_CACHE_LIMIT = max(0, int(float(megabytes) * 1024 ** 2))
        _evict_states()


def _evict_states():
    """Evict oldest-first until the budget holds. Caller holds the lock."""
    global _STATE_CACHE_BYTES

    while _STATE_CACHE and _STATE_CACHE_BYTES > _STATE_CACHE_LIMIT:
        _, evicted = _STATE_CACHE.popitem(last=False)
        _STATE_CACHE_BYTES -= evicted.nbytes

    if not _STATE_CACHE:
        # Re-anchor the counter: an empty cache holds exactly zero bytes, so
        # any drift cannot survive past here and silently shrink the budget.
        _STATE_CACHE_BYTES = 0


def _cache_get(key):
    with _CACHE_LOCK:
        cached = _STATE_CACHE.get(key)

        if cached is not None:
            _STATE_CACHE.move_to_end(key)

        return cached


def _cache_put(key, value):
    global _STATE_CACHE_BYTES

    # A single entry larger than the whole budget would be evicted immediately.
    if _STATE_CACHE_LIMIT <= 0 or value.nbytes > _STATE_CACHE_LIMIT:
        return

    value.setflags(write=False)  # cached arrays are shared; nobody may edit one

    with _CACHE_LOCK:
        # Two threads can miss on the same row and both put it. Drop whatever
        # the key held first, so the counter tracks what is actually stored
        # instead of drifting upward until the cache evicts itself to nothing.
        previous = _STATE_CACHE.pop(key, None)

        if previous is not None:
            _STATE_CACHE_BYTES -= previous.nbytes

        _STATE_CACHE[key] = value
        _STATE_CACHE_BYTES += value.nbytes
        _evict_states()


def _double_center(matrix):
    """H M H for H = I - 11^T/n, without building H.

    Used by both kernel-target alignment paths; the explicit centring matrix
    turned an O(n^2) operation into two dense n x n matmuls.
    """
    row_means = matrix.mean(axis=1, keepdims=True)
    column_means = matrix.mean(axis=0, keepdims=True)

    return matrix - row_means - column_means + matrix.mean()


class _Unidentifiable(Exception):
    """Raised when a circuit cannot be described well enough to key a cache."""


_STANDARD_GATE_TYPES = None
_MAX_DEFINITION_DEPTH = 12


def _standard_gate_types():
    global _STANDARD_GATE_TYPES

    if _STANDARD_GATE_TYPES is None:
        from qiskit.circuit.library import get_standard_gate_name_mapping

        _STANDARD_GATE_TYPES = {
            name: type(gate) for name, gate in get_standard_gate_name_mapping().items()
        }

    return _STANDARD_GATE_TYPES


def _is_standard(operation):
    """A library gate whose (name, params) pin down its unitary exactly."""
    return type(operation) is _standard_gate_types().get(operation.name)


def _parameter_key(parameter):
    """Identify a gate parameter by value, not by how it prints.

    ``str()`` is right for a ParameterExpression (it is exact) and wrong for an
    array: numpy abbreviates anything past 1000 elements and rounds to 8
    digits, so two different unitaries print the same and would share a cache
    entry.
    """
    if isinstance(parameter, ParameterExpression):
        return str(parameter)

    array = np.asarray(parameter)

    if array.dtype == object:
        raise _Unidentifiable(f"parameter of type {type(parameter)!r}")

    return array.shape, str(array.dtype), array.tobytes()


def _circuit_structure(circuit, depth=0):
    if depth > _MAX_DEFINITION_DEPTH:
        raise _Unidentifiable("definition nested too deeply")

    entries = [
        circuit.num_qubits,
        circuit.num_clbits,
        str(circuit.global_phase),
        # assign_parameters binds positionally against circuit.parameters, so
        # the binding ORDER is part of what a row means. Two circuits whose
        # gates print identically can still bind the same row differently.
        tuple(str(parameter) for parameter in circuit.parameters),
    ]

    for instruction in circuit.data:
        operation = instruction.operation
        entry = (
            operation.name,
            tuple(circuit.find_bit(qubit).index for qubit in instruction.qubits),
            tuple(circuit.find_bit(clbit).index for clbit in instruction.clbits),
            tuple(_parameter_key(parameter) for parameter in operation.params),
        )

        if not _is_standard(operation) and not isinstance(operation, (Barrier, Delay)):
            # A composite instruction carries its content in its definition,
            # not its name: PauliEvolutionGate("XX", t) and
            # PauliEvolutionGate("ZZ", t) are both named "PauliEvolution" with
            # the same params. Descend, or refuse to key the circuit at all.
            definition = getattr(operation, "definition", None)

            if definition is None:
                raise _Unidentifiable(f"opaque instruction {operation.name!r}")

            entry += (_circuit_structure(definition, depth + 1),)

        entries.append(entry)

    return tuple(entries)


def _circuit_key(circuit):
    """Structural identity of an encoding circuit, as a short digest.

    Object identity is useless here: a CV sweep builds one kernel (and one
    circuit) per fold from the same arguments, and those folds are exactly
    where the reuse is. So the key has to be what the circuit DOES -- gates,
    wires, parameter values and binding order, all the way down through
    composite instructions.

    Returns ``None`` for a circuit this function cannot pin down exactly
    (an opaque instruction with no definition, a parameter it cannot hash).
    Callers then skip the cache and simulate, which is slower and always
    right; a digest that is merely probable would hand one encoding another
    encoding's states.

    The structure is hashed to 16 bytes because it is half of every cache key,
    and every key is held for as long as its entry is: a short digest keeps
    the keys small and cheap to compare, where the structure itself is a
    nested tuple as long as the circuit.
    """
    try:
        structure = _circuit_structure(circuit)
    except _Unidentifiable:
        return None

    return hashlib.blake2b(repr(structure).encode(), digest_size=16).digest()


# ---------------------------------------------------------------------------
# Lowered circuits for the fused simulators
#
# Lowering is dominated by sympy printing every angle expression into numpy
# code: measured 0.10 s for a 12-qubit ZZ map (156 parametrised gates) on
# Kaggle's CPU, two-thirds of the 0.15 s it takes to simulate 200 samples
# through it.
# QuantumClassifier builds a fresh kernel per CV fold and candidate (22 for
# the default sweep, counted) around the SAME
# circuit, so a program is shared by circuit digest, not owned by a kernel: a
# circuit is lowered once per process. The digest is the state cache's, which
# already pins down everything lowering reads.
# ---------------------------------------------------------------------------
_PROGRAMS = OrderedDict()  # digest -> Program, or the reason it cannot be lowered
_PROGRAMS_LIMIT = 32
_PROGRAMS_LOCK = threading.Lock()
_FALLBACK_WARNED = set()

# The projected kernel on a fused simulator reduces states to Bloch vectors a
# chunk of rows at a time, so (as on the qiskit path) the whole state table
# never exists at once: 16 MB is 256 rows at 12 qubits, 16 at 16.
_FUSED_BLOCH_CHUNK_BYTES = 16 * 2 ** 20


def _lowered(signature, circuit):
    """``(program, None)`` for a circuit digest, or ``(None, reason)``."""
    with _PROGRAMS_LOCK:
        entry = _PROGRAMS.get(signature)

        if entry is not None:
            _PROGRAMS.move_to_end(signature)

    if entry is None:
        # Outside the lock: lowering takes a while, and two threads lowering
        # the same circuit at once only duplicate work.
        try:
            entry = fused.lower(circuit)
        except fused.Unsupported as exc:
            entry = str(exc)

        with _PROGRAMS_LOCK:
            entry = _PROGRAMS.setdefault(signature, entry)
            _PROGRAMS.move_to_end(signature)

            while len(_PROGRAMS) > _PROGRAMS_LIMIT:
                _PROGRAMS.popitem(last=False)

    if isinstance(entry, str):
        return None, entry

    return entry, None


def _warn_fallback(key, simulator, reason):
    """Warn once per circuit, not once per kernel built around it."""
    with _PROGRAMS_LOCK:
        if (key, simulator) in _FALLBACK_WARNED:
            return
        _FALLBACK_WARNED.add((key, simulator))

    warnings.warn(
        f"simulator={simulator!r} cannot run this feature map ({reason}); "
        "falling back to the qiskit simulator for it.",
        UserWarning,
        stacklevel=5,
    )

# Distinguishes "caller did not pass entanglement" from "caller explicitly asked
# for the default", so a meaningless combination can be reported instead of
# silently swallowed.
_UNSET = object()


class QuantumKernel:
    """Quantum fidelity kernel over a ZZ feature map.

    Two modes are supported:

    ``"ZZ"``
        A plain fidelity kernel: the input angles are passed to the ZZ feature
        map unchanged (subject to ``normalize``/``bandwidth`` below).

    ``"covariant"`` / ``"mahalanobis"``
        The data is first Mahalanobis/ZCA-whitened using the training
        covariance, so Euclidean distance in the transformed space equals
        Mahalanobis distance in the raw space, and the whitened values are then
        used as ZZ angles.

        NOTE ON NAMING: this is *not* the "covariant quantum kernel" of Glick
        et al., which is a group-covariant construction built on a fiducial
        state. This mode is covariance-adapted preprocessing followed by a
        standard ZZ map. ``"mahalanobis"`` is the accurate alias; ``"covariant"``
        is retained for backwards compatibility.

    Bandwidth
    ---------
    Fidelity kernels concentrate: as the encoding angles span a wider range,
    off-diagonal similarities collapse toward zero, the Gram matrix approaches
    the identity, and an SVM on top of it memorises the training set instead of
    generalising. ``bandwidth`` scales the encoding angles and is the standard
    remedy; it is a genuine hyperparameter and is usually worth tuning over
    something like ``[1.0, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01]``. Tuners have
    been observed selecting the bottom of a grid that stopped at 0.05, so extend
    the range downward rather than assuming 0.05 is a floor.

    It is applied as the *last* step of :meth:`_prepare`, i.e. after whitening.
    Applying it before whitening would be cancelled exactly: scaling ``X`` by
    ``c`` scales the covariance by ``c**2`` and the whitening matrix by ``1/c``,
    so ``(cX - c*mean) @ (W/c) == (X - mean) @ W``. That cancellation is why
    scaling the input has no effect on covariant mode.

    Normalisation
    -------------
    Whitening produces unit-variance output whose range grows with the sample
    size and can exceed the ``[-pi, pi]`` interval the ZZ map's angles are
    meaningful over, at which point distinct inputs alias onto the same state.
    ``normalize`` rescales the prepared array by a scale fitted **on the
    training data only** (in :meth:`fit`) and reused at predict time, so it does
    not leak. ``normalize="maxabs"`` bounds the output to ``[-1, 1]`` before
    ``bandwidth`` is applied.

    Encoding
    --------
    ``feature_map`` selects the encoding circuit and ``entanglement`` is passed
    through to it:

    - ``"zz"`` (default) -- :func:`zz_feature_map`, honouring ``entanglement``
      (``"full"``, ``"linear"``, ``"circular"``, ...).
    - ``"z"`` -- :func:`z_feature_map`, a product encoding with **no**
      entangling gates. ``entanglement`` has no effect here and passing it
      warns.
    - a Qiskit :class:`QuantumCircuit` -- used directly, so the encoding is a
      first-class hyperparameter. It must have ``n_qubits`` qubits and exactly
      ``n_qubits`` free parameters.

    Entanglement drives the concentration that ``bandwidth`` compensates for, so
    the two interact; sweeping them together is reasonable. The unentangled
    ``"z"`` map computed a 200-sample Gram matrix 2.8-8.5x faster than
    ``"zz"`` at 4-12 qubits (Kaggle, Intel Xeon CPU), and which one
    classifies better depends on the data, so the entangled default is not
    automatically the right choice.

    Assigning to ``.feature_map`` after construction rebuilds the underlying
    fidelity kernel, so the swap applies to both the statevector and pairwise
    paths.

    Sizing
    ------
    ``n_qubits`` may be omitted entirely: the feature map reads one angle per
    feature, so the width is knowable from the data and ``QuantumKernel()``
    sizes itself at the first ``fit()`` / ``compute_kernel_matrix()`` call
    (or immediately, from the circuit, when ``feature_map`` is a
    ``QuantumCircuit``). Passing an explicit ``n_qubits`` behaves exactly as
    before, including raising on a width mismatch.

    All parameters after ``shrinkage`` are keyword-only, and their defaults
    reproduce the previous numeric behaviour exactly.
    """

    def __init__(self, n_qubits=None, mode="ZZ", reps=2, shrinkage=1e-3, *,
                 bandwidth=1.0, normalize=None, feature_map="zz",
                 entanglement=_UNSET, fidelity=None, sampler=None,
                 fast_statevector="auto", kernel="fidelity", gamma="auto",
                 simulator="qiskit", device=None):
        # A concrete circuit already knows its own width, so there is nothing
        # to defer.
        if n_qubits is None and isinstance(feature_map, QuantumCircuit):
            n_qubits = feature_map.num_qubits

        self.n_qubits = n_qubits
        self.mode = mode
        self.reps = reps
        self.shrinkage = shrinkage
        self.bandwidth = bandwidth
        self.normalize = normalize
        self.feature_map_spec = feature_map
        self.entanglement = "full" if entanglement is _UNSET else entanglement
        self.fast_statevector = fast_statevector
        self.mean_ = None
        self.cov_matrix = None
        self.whitening_matrix = None
        self.scale_ = None
        self.X_train_ = None

        if mode not in _VALID_MODES:
            raise ValueError(f"Unknown mode: {mode}")

        if normalize not in (None, "maxabs", "std"):
            raise ValueError(
                f"Unknown normalize: {normalize} (expected None, 'maxabs' or 'std')"
            )

        # bandwidth: a positive scalar (isotropic, the historical behaviour),
        # a positive vector with one entry per feature (ARD / anisotropic --
        # neither exists in qiskit's kernels), or "median" to fit it from the
        # training data via the median heuristic (1 / median pairwise
        # distance of the prepared angles) with no cross-validation at all.
        if isinstance(bandwidth, str):
            if bandwidth not in ("median", "alignment"):
                raise ValueError(
                    f"bandwidth must be a positive number, a positive vector, "
                    f"'median', or 'alignment', got {bandwidth!r}"
                )
        elif np.ndim(bandwidth) == 1:
            bandwidth = np.asarray(bandwidth, dtype=float)

            if not np.all(np.isfinite(bandwidth)) or np.any(bandwidth <= 0):
                raise ValueError(
                    "per-feature bandwidth must be positive and finite in "
                    f"every entry, got {bandwidth}"
                )
        elif not np.isfinite(bandwidth) or bandwidth <= 0:
            raise ValueError(f"bandwidth must be a positive finite number, got {bandwidth}")

        if kernel not in _VALID_KERNELS:
            raise ValueError(
                f"Unknown kernel: {kernel!r} (expected one of {_VALID_KERNELS})"
            )

        if kernel == "projected" and (fidelity is not None or sampler is not None):
            raise ValueError(
                "kernel='projected' is computed from exact per-qubit reduced "
                "density matrices on the statevector and does not use a "
                "fidelity or sampler; drop those arguments."
            )

        if simulator not in _VALID_SIMULATORS:
            raise ValueError(
                f"Unknown simulator: {simulator!r} (expected one of {_VALID_SIMULATORS})"
            )

        if simulator != "qiskit" and (fidelity is not None or sampler is not None):
            raise ValueError(
                f"simulator={simulator!r} is an exact statevector simulator; a "
                "custom fidelity or sampler means shots or hardware. Drop one."
            )

        if device is not None and simulator != "torch":
            raise ValueError("device= only applies to simulator='torch'")

        if simulator != "qiskit" and fast_statevector is False:
            raise ValueError(
                f"simulator={simulator!r} runs on the statevector path, and "
                "fast_statevector=False asks for qiskit's pairwise fidelity "
                "circuits instead. Drop one."
            )

        if simulator == "torch":
            fused.resolve_device(device)  # a missing torch or a bad device fails here

        if gamma != "auto" and (not np.isscalar(gamma) or not np.isfinite(gamma) or gamma <= 0):
            raise ValueError(
                f"gamma must be a positive finite number or 'auto', got {gamma!r}"
            )

        self.kernel = kernel
        self.gamma = gamma
        self.gamma_ = None if gamma == "auto" else float(gamma)
        self.simulator = simulator
        self.device = device
        self.bandwidth_ = None  # fitted value when bandwidth == "median"

        # Whitened angles routinely leave [-pi, pi] and alias, which collapses
        # the kernel; without a normalisation this mode has been observed to go
        # fully degenerate (every sample predicted as one class). Warn rather
        # than change the default, which would be a breaking numeric change.
        if mode in _WHITENED_MODES and normalize is None:
            warnings.warn(
                f"mode={mode!r} with normalize=None: whitening produces angles "
                "outside [-pi, pi] that alias in the feature map, and bandwidth "
                "cannot compensate. Pass normalize='maxabs' unless you are "
                "deliberately reproducing pre-1.2 behaviour.",
                UserWarning,
                stacklevel=2
            )

        # A caller-supplied fidelity or sampler means the kernel may be
        # shot-based or noisy, so the exact statevector shortcut no longer
        # applies and "auto" falls back to the pairwise path.
        self._custom_fidelity = fidelity is not None or sampler is not None
        self._fidelity = fidelity
        self._sampler = sampler

        # Kept for deferred building when n_qubits is None; entanglement must
        # stay the raw argument (possibly the _UNSET sentinel) so the warning
        # logic in _build_feature_map still sees what the caller actually
        # passed.
        self._feature_map_arg = feature_map
        self._entanglement_arg = entanglement
        self._feature_map = None
        self._fidelity_quantum_kernel = None
        self._circuit_signature = None

        if n_qubits is not None:
            self._set_feature_map(
                self._build_feature_map(feature_map, entanglement)
            )

    def _ensure_sized(self, n_features):
        # Deferred sizing: the first data this kernel sees fixes its width,
        # after which mismatches raise exactly as they do for an explicit
        # n_qubits.
        if self.n_qubits is not None:
            return

        self.n_qubits = int(n_features)
        self._set_feature_map(
            self._build_feature_map(self._feature_map_arg, self._entanglement_arg)
        )

    def _build_feature_map(self, feature_map, entanglement):
        # A caller-supplied circuit makes the encoding a first-class
        # hyperparameter without this class having to enumerate every option.
        if isinstance(feature_map, QuantumCircuit):
            if feature_map.num_qubits != self.n_qubits:
                raise ValueError(
                    f"feature_map circuit has {feature_map.num_qubits} qubits, "
                    f"expected {self.n_qubits}"
                )

            if feature_map.num_parameters != self.n_qubits:
                raise ValueError(
                    f"feature_map circuit has {feature_map.num_parameters} free "
                    f"parameters, expected {self.n_qubits} (one angle per feature)"
                )

            if entanglement is not _UNSET:
                warnings.warn(
                    "entanglement is ignored when feature_map is a QuantumCircuit; "
                    "build the entanglement into the circuit instead.",
                    UserWarning,
                    stacklevel=3
                )

            return feature_map

        if feature_map == "zz":
            return zz_feature_map(
                feature_dimension=self.n_qubits,
                reps=self.reps,
                entanglement=self.entanglement
            )

        if feature_map == "z":
            # z_feature_map accepts an entanglement argument but has no
            # multi-qubit terms, so it is a genuine no-op. Say so rather than
            # letting a swept grid silently produce duplicate rows.
            if entanglement is not _UNSET:
                warnings.warn(
                    f"entanglement={self.entanglement!r} has no effect with "
                    "feature_map='z' (no entangling gates); results will be "
                    "identical across entanglement values.",
                    UserWarning,
                    stacklevel=3
                )

            return z_feature_map(feature_dimension=self.n_qubits, reps=self.reps)

        raise ValueError(
            f"Unknown feature_map: {feature_map!r} "
            f"(expected one of {_BUILTIN_FEATURE_MAPS} or a QuantumCircuit)"
        )

    def _set_feature_map(self, circuit):
        # The fidelity kernel caches the circuit, so it has to be rebuilt
        # alongside it. Without this, assigning to .feature_map would change the
        # statevector path but not the pairwise path -- the same result silently
        # depending on which backend happened to be selected.
        self._feature_map = circuit
        # Cached simulations belong to a circuit, so the key is recomputed
        # whenever the circuit is swapped.
        self._circuit_signature = _circuit_key(circuit)

        if self._fidelity is None and self._sampler is not None:
            from qiskit_machine_learning.state_fidelities import ComputeUncompute

            self._fidelity = ComputeUncompute(sampler=self._sampler)

        if self._fidelity is None:
            self._fidelity_quantum_kernel = FidelityQuantumKernel(
                feature_map=circuit
            )
        else:
            self._fidelity_quantum_kernel = FidelityQuantumKernel(
                feature_map=circuit,
                fidelity=self._fidelity
            )

    @property
    def feature_map(self):
        if self._feature_map is None:
            raise ValueError(
                "n_qubits was not given, so the encoding circuit is not built "
                "until the first fit() or compute_kernel_matrix() call sizes "
                "the kernel from the data."
            )

        return self._feature_map

    @feature_map.setter
    def feature_map(self, circuit):
        if self.n_qubits is None:
            self.n_qubits = circuit.num_qubits

        self._set_feature_map(circuit)

    @property
    def fidelity_quantum_kernel(self):
        if self._fidelity_quantum_kernel is None:
            raise ValueError(
                "n_qubits was not given, so the underlying fidelity kernel is "
                "not built until the first fit() or compute_kernel_matrix() "
                "call sizes it from the data."
            )

        # Handing this straight to QSVC bypasses _prepare entirely, which
        # silently discards whitening, normalize and bandwidth -- the caller
        # gets a plain unscaled kernel and no error. Warn when that would
        # actually change the answer.
        if (self.mode in _COVARIANT_MODES
                or self.bandwidth != 1.0
                or self.normalize is not None):
            warnings.warn(
                "Using .fidelity_quantum_kernel directly bypasses _prepare, so "
                f"mode={self.mode!r}, bandwidth={self.bandwidth} and "
                f"normalize={self.normalize!r} will be IGNORED. Use "
                ".compute_kernel_matrix(X) / .compute_kernel_matrix(X_test, "
                "X_train) with sklearn's SVC(kernel='precomputed') instead.",
                UserWarning,
                stacklevel=2
            )

        return self._fidelity_quantum_kernel

    def _check_width(self, X_np, name="X"):
        # The feature map reads one angle per feature, so a width mismatch is
        # fatal. Without this it surfaces from inside Qiskit's parameter binding
        # as "Mismatching number of values and parameters. For partial binding
        # please pass a dictionary of {parameter: value} pairs." -- a message
        # that names neither n_qubits, nor the feature count, nor this class.
        if X_np.ndim != 2:
            raise ValueError(
                f"{name} must be 2D (n_samples, n_features), got a "
                f"{X_np.ndim}D array of shape {X_np.shape}"
            )

        self._ensure_sized(X_np.shape[1])

        # Checked here rather than in _prepare so that fit() catches it too:
        # a wrong-length ARD vector should fail at the first sight of data,
        # not at the first kernel evaluation.
        if np.ndim(self.bandwidth) == 1 and len(self.bandwidth) != self.n_qubits:
            raise ValueError(
                f"per-feature bandwidth has {len(self.bandwidth)} entries but "
                f"the data has {X_np.shape[1]} features"
            )

        if X_np.shape[1] != self.n_qubits:
            raise ValueError(
                f"QuantumKernel(n_qubits={self.n_qubits}) but {name} has "
                f"{X_np.shape[1]} features. The feature map encodes one angle "
                f"per feature, so these must match -- either pass "
                f"n_qubits={X_np.shape[1]}, or project {name} to "
                f"{self.n_qubits} features before fitting."
            )

    def fit(self, X_train, y=None):
        X_np = np.asarray(X_train, dtype=float)
        self._check_width(X_np, "X_train")
        self.X_train_ = X_np

        if self.mode in _WHITENED_MODES:
            self.mean_ = X_np.mean(axis=0)
            centered = X_np - self.mean_

            if self.mode == "fisher":
                self.cov_matrix = self._within_class_scatter(X_np, y)
            else:
                self.cov_matrix = np.cov(centered, rowvar=False)

            self.whitening_matrix = self._compute_whitening(self.cov_matrix)

        # The normalisation scale is fitted here, on training data only, and
        # reused verbatim by _prepare at predict time -- same no-leakage
        # contract as mean_ / whitening_matrix above.
        if self.normalize is not None:
            self.scale_ = self._compute_scale(self._transform_mode(X_np))

        # Median-heuristic bandwidth, fitted on exactly the angles bandwidth
        # will later multiply (post-whitening, post-normalisation), so the
        # heuristic sees what the circuit sees.
        if isinstance(self.bandwidth, str):
            prepared = self._transform_mode(X_np)

            if self.scale_ is not None:
                prepared = prepared / self.scale_

            if self.bandwidth == "median":
                self.bandwidth_ = self._median_bandwidth(prepared)
            else:
                self.bandwidth_ = self._alignment_bandwidth(prepared, y)

        # Median-heuristic gamma for the projected kernel, from the training
        # Bloch-vector geometry.
        if self.kernel == "projected" and self.gamma == "auto":
            self.gamma_ = self._median_gamma(self._prepare(X_np))

        return self

    def _within_class_scatter(self, X_np, y):
        # Class-size-weighted mean of per-class covariances. Whitening by this
        # instead of the global covariance is Fisher-style metric learning:
        # noise directions (large within-class variance) are compressed,
        # class-separating directions keep their scale. Requires labels --
        # qiskit's kernels have no supervised path at all.
        if y is None:
            raise ValueError(
                "mode='fisher' whitens by the within-class scatter and "
                "therefore needs labels: call .fit(X_train, y_train)."
            )

        y_arr = np.asarray(y)
        dim = X_np.shape[1]
        scatter = np.zeros((dim, dim))
        counted = 0

        for cls in np.unique(y_arr):
            members = X_np[y_arr == cls]

            if len(members) < 2:
                continue  # a singleton class has no within-class spread

            scatter += (len(members) / len(X_np)) * np.cov(members, rowvar=False)
            counted += len(members)

        if counted == 0:
            raise ValueError(
                "mode='fisher' needs at least one class with two or more "
                "samples to estimate the within-class scatter."
            )

        return scatter

    def _alignment_bandwidth(self, prepared, y):
        # Third selection principle after CV and the median heuristic: pick
        # the bandwidth whose training Gram matrix maximises centered
        # kernel-target alignment (Cortes et al. 2012). One Gram per
        # candidate, no folds, no SVM fits. qiskit's only alignment use is an
        # SVC loss for hand-designed trainable circuits; it cannot tune a
        # bandwidth because its kernels have none.
        if y is None:
            raise ValueError(
                "bandwidth='alignment' maximises kernel-target alignment and "
                "therefore needs labels: call .fit(X_train, y_train)."
            )

        y_arr = np.asarray(y)

        if len(np.unique(y_arr)) == len(y_arr):
            raise ValueError(
                "bandwidth='alignment' needs class labels (every label is "
                "unique -- this looks like a regression target). Use "
                "bandwidth='median' or a numeric bandwidth instead."
            )

        candidates = (1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01)
        target = np.where(y_arr[:, None] == y_arr[None, :], 1.0, -1.0)
        scores = []

        # Temporarily bind each candidate so the normal compute path (fast
        # statevector, projected, everything) is what gets scored.
        for candidate in candidates:
            angles = prepared * candidate

            if self.kernel == "projected":
                if self.gamma_ is None:
                    self.gamma_ = self._median_gamma(angles)
                gram = self._projected_kernel_matrix(angles, None, None)
            elif self._use_statevector():
                gram = self._statevector_kernel(angles)
            else:
                gram = self._fidelity_quantum_kernel.evaluate(angles)

            scores.append(self._centered_alignment(gram, target))

        return candidates[int(np.argmax(scores))]

    @staticmethod
    def _centered_alignment(gram, target):
        # H M H with H = I - 11^T/n is M minus its row means, minus its column
        # means, plus its grand mean -- so the same centring costs O(n^2)
        # instead of the two dense n x n matmuls (2n^3 flops) the explicit H
        # required, and the alignment sweep centres one Gram matrix per
        # candidate bandwidth.
        gram_c = _double_center(gram)
        target_c = _double_center(target)
        denom = np.linalg.norm(gram_c) * np.linalg.norm(target_c)

        return float((gram_c * target_c).sum() / denom) if denom > 0 else 0.0

    def _median_bandwidth(self, prepared):
        distances = np.sqrt(np.maximum(self._sq_distances(prepared, prepared), 0))
        off = distances[~np.eye(len(distances), dtype=bool)]
        median = float(np.median(off)) if off.size else 0.0

        # Scale angles so the median pairwise separation is ~1 radian; a
        # degenerate (all-identical) training set falls back to no scaling.
        return 1.0 / median if median > 0 else 1.0

    def _median_gamma(self, prepared):
        bloch = self._bloch_features(prepared)
        rdm_sq = 0.5 * self._sq_distances(bloch, bloch)
        off = rdm_sq[~np.eye(len(rdm_sq), dtype=bool)]
        median = float(np.median(off)) if off.size else 0.0

        return 1.0 / median if median > 0 else 1.0

    @staticmethod
    def _sq_distances(A, B):
        sq = (
            np.sum(A ** 2, axis=1)[:, None]
            + np.sum(B ** 2, axis=1)[None, :]
            - 2.0 * (A @ B.T)
        )

        return np.maximum(sq, 0.0)

    def _compute_whitening(self, cov_matrix):
        # Mahalanobis whitening from the training-data covariance. The covariance
        # is first shrunk toward a scaled identity (Ledoit-Wolf-style) so the
        # inverse square root stays well-conditioned even when features are
        # collinear, then W = V diag(1/sqrt(lambda)) V^T. Mapping the centered
        # data through W gives it identity covariance, so a Euclidean distance in
        # the transformed space is exactly the Mahalanobis distance of the raw
        # data, d(x, y) = sqrt((x - y)^T Sigma^-1 (x - y)).
        cov_matrix = np.atleast_2d(cov_matrix)
        dim = cov_matrix.shape[0]

        target = np.trace(cov_matrix) / dim
        shrunk = (1.0 - self.shrinkage) * cov_matrix + self.shrinkage * target * np.eye(dim)

        eigenvalues, eigenvectors = np.linalg.eigh(shrunk)
        inv_sqrt = 1.0 / np.sqrt(np.maximum(eigenvalues, 0) + 1e-8)
        whitening = eigenvectors @ np.diag(inv_sqrt) @ eigenvectors.T

        return whitening

    def _compute_scale(self, prepared):
        # A single global scalar, not a per-feature vector: the point is to
        # bound the overall angle range, and whitened data already has unit
        # per-feature variance by construction, so a per-feature scale would be
        # a no-op there.
        if self.normalize == "maxabs":
            scale = np.abs(prepared).max()
        else:
            scale = prepared.std()

        # A degenerate (all-constant) training fold would otherwise divide by
        # zero and turn every angle into a NaN.
        if not np.isfinite(scale) or scale == 0:
            return 1.0

        return float(scale)

    def _transform_mode(self, X_np):
        # The mode-specific part of _prepare, split out so fit() can compute the
        # normalisation scale on exactly what _prepare will later normalise.
        if self.mode in _COVARIANT_MODES:
            if self.whitening_matrix is None:
                raise ValueError("Must call .fit() before using covariant mode")
            return (X_np - self.mean_) @ self.whitening_matrix

        return X_np

    def _prepare(self, X, name="X"):
        X_np = np.asarray(X, dtype=float)
        # Also checked here, not just in fit(), because ZZ mode never requires
        # a fit() call and would otherwise reach Qiskit unvalidated.
        self._check_width(X_np, name)
        X_np = self._transform_mode(X_np)

        if self.normalize is not None:
            if self.scale_ is None:
                raise ValueError("Must call .fit() before using normalize")
            X_np = X_np / self.scale_

        # Bandwidth is applied last so it survives whitening (see class
        # docstring); at the default of 1.0 this is an exact no-op. A vector
        # bandwidth broadcasts per feature (ARD): because it also lands AFTER
        # whitening, it is a per-direction metric no external rescaling of
        # the input can reproduce -- whitening would cancel that exactly.
        return X_np * self._effective_bandwidth()

    def _effective_bandwidth(self):
        if isinstance(self.bandwidth, str):
            if self.bandwidth_ is None:
                raise ValueError(
                    "bandwidth='median' is fitted from the training data; "
                    "call .fit() first."
                )

            return self.bandwidth_

        return self.bandwidth

    def _use_statevector(self):
        if self.fast_statevector is True:
            return True

        if self.fast_statevector is False:
            return False

        # "auto": the shortcut is only exact for a pure-state, noiseless
        # fidelity, which is what the default FidelityQuantumKernel gives.
        return not self._custom_fidelity

    def _fused_program(self):
        """The feature map lowered for the fused simulators, or None.

        None means "use the qiskit simulator": either it was asked for, or the
        circuit contains something qmlf.ops.fused does not lower, in which case
        this warns once and the kernel stays exact rather than failing. The
        lowering is redone whenever the circuit's digest changes, so a caller
        mutating their own circuit in place is followed, as on the cached path.
        """
        if self.simulator == "qiskit":
            return None

        # Also the state-cache key for the rows simulated with this program.
        self._circuit_signature = signature = _circuit_key(self.feature_map)

        if signature is None:
            program = None
            reason = "it contains an instruction that cannot be identified exactly"
            warn_key = ("unidentifiable", id(self.feature_map))
        else:
            program, reason = _lowered(signature, self.feature_map)
            warn_key = signature

        if program is None:
            _warn_fallback(warn_key, self.simulator, reason)

        return program

    def _fused_tag(self):
        # Part of every cache key the fused paths write. Their states differ
        # from qiskit's (and complex64 from complex128) in the last bits, so
        # they must never answer a lookup made by another simulator -- least
        # of all the default one.
        if self.simulator == "fused":
            return ("fused",)

        return ("torch", str(fused.resolve_device(self.device)))

    def _run_fused(self, program, rows):
        """Simulate ``rows`` with the fused program; a numpy table."""
        if self.simulator == "fused":
            return program.statevectors(rows)

        return fused.to_numpy(
            program.statevectors(rows, backend="torch", device=self.device)
        )

    def _fused_states(self, program, prepared):
        """Encoded states of every prepared row, via the fused simulator.

        Through the same row cache as the qiskit path, so a CV sweep, a cross
        kernel or a predict() does not re-simulate rows it already has. Rows
        missing from the cache are simulated together in one batch, which is
        safe because the simulator is batch-invariant.
        """
        signature = self._circuit_signature

        if signature is None or _STATE_CACHE_LIMIT <= 0 or len(prepared) == 0:
            return self._run_fused(program, prepared)

        tag = self._fused_tag()
        keys = [(signature, tag, row.tobytes()) for row in prepared]
        states = [_cache_get(key) for key in keys]
        missing = [i for i, state in enumerate(states) if state is None]

        if not missing:
            return np.stack(states)

        fresh = self._run_fused(program, prepared[missing])

        for i, state in zip(missing, fresh):
            state = state.copy()  # its own memory: a view would pin the batch
            _cache_put(keys[i], state)
            states[i] = state

        return fresh if len(missing) == len(prepared) else np.stack(states)

    def _fused_gram(self, states_a, states_b, batch_size=None):
        if self.simulator == "torch":
            same = states_b is states_a
            states_a = fused.as_tensor(states_a, self.device)
            states_b = states_a if same else fused.as_tensor(states_b, self.device)

        return fused.fidelity_gram(states_a, states_b, batch_size)

    def _fused_bloch(self, program, prepared):
        """Bloch features via the fused simulator, cached per row like the
        qiskit path's, and never holding more than one chunk of states."""
        n = self.n_qubits
        signature = self._circuit_signature
        use_cache = signature is not None and _STATE_CACHE_LIMIT > 0
        tag = self._fused_tag() if use_cache else None
        keys = [(signature, tag, "bloch", row.tobytes()) for row in prepared] \
            if use_cache else None

        features = np.empty((len(prepared), 3 * n))
        missing = []

        for i in range(len(prepared)):
            cached = _cache_get(keys[i]) if use_cache else None

            if cached is None:
                missing.append(i)
            else:
                features[i] = cached

        chunk = max(1, _FUSED_BLOCH_CHUNK_BYTES // (16 * 2 ** n))

        for start in range(0, len(missing), chunk):
            rows = missing[start:start + chunk]
            vectors = fused.bloch_features(self._run_fused(program, prepared[rows]), n)
            features[rows] = vectors

            if use_cache:
                for i, vector in zip(rows, vectors):
                    _cache_put(keys[i], vector.copy())

        return features

    def _refresh_signature(self):
        # Recomputed once per table rather than trusted from construction: a
        # caller can hand in their own QuantumCircuit and then mutate it in
        # place, and a cache keyed on a stale digest would answer with the old
        # circuit's states. Re-digesting costs one pass over the gate list --
        # nothing next to a single simulation.
        self._circuit_signature = _circuit_key(self.feature_map)

    def _simulate_row(self, row, store=True):
        """The encoded state of one prepared angle row, cached across kernels.

        ``store=False`` still reads the cache but does not fill it: the
        projected kernel only needs the 3 * n_qubits Bloch vector it reduces
        the state to, and keeping the state as well (2**n_qubits complex
        amplitudes: 4,096 bytes at 8 qubits against the vector's 192, 21x)
        would evict the Bloch vectors it does reuse.
        """
        if self._circuit_signature is None:
            return np.asarray(
                Statevector(self.feature_map.assign_parameters(row)).data
            )

        key = (self._circuit_signature, row.tobytes())
        state = _cache_get(key)

        if state is None:
            state = np.asarray(
                Statevector(self.feature_map.assign_parameters(row)).data
            )

            if store:
                _cache_put(key, state)

        return state

    def _simulate(self, prepared):
        """Encoded states for every prepared row, one simulation per new row."""
        self._refresh_signature()
        states = np.empty((len(prepared), 2 ** self.n_qubits), dtype=complex)

        for i, row in enumerate(prepared):
            states[i] = self._simulate_row(row)

        return states

    def _statevector_kernel(self, A, B=None):
        # For a pure-state fidelity kernel, K[i, j] = |<phi(x_i)|phi(x_j)>|^2.
        # Simulating n statevectors once and taking |Psi Psi^dagger|^2 is
        # mathematically identical to evaluating n^2 circuit pairs, but costs
        # O(n) simulations plus one matmul instead of O(n^2) circuit builds.
        # The result is PSD by the Schur product theorem, so enforce_psd would
        # be a no-op on it.
        program = self._fused_program()

        if program is not None:
            states_a = self._fused_states(program, A)
            states_b = states_a if B is None else self._fused_states(program, B)
            return self._fused_gram(states_a, states_b)

        states_a = self._simulate(A)
        states_b = states_a if B is None else self._simulate(B)

        return np.abs(states_a.conj() @ states_b.T) ** 2

    def nystrom_features(self, X, n_landmarks=None, landmark_indices=None):
        """Explicit feature map Phi with Phi @ Phi.T ~= the kernel matrix.

        Nystrom approximation from m landmark rows of ``X``: computes only the
        (n x m) cross block and the (m x m) landmark block instead of the full
        (n x n) Gram matrix. On a statevector this is a convenience; on
        hardware (a custom ``fidelity``/``sampler``) it is the difference
        between n(n-1)/2 fidelity circuits and roughly n*m -- see
        :meth:`circuit_budget`. Neither qiskit-machine-learning nor TFQ ships
        any kernel approximation.

        Landmarks default to ``n_landmarks`` evenly spaced rows of ``X`` in
        the given order (deterministic); pass ``landmark_indices`` to choose
        them yourself. Returns an ``(n, m)`` array usable with any linear
        model, e.g. ``LinearSVC``.
        """
        X_np = np.asarray(X, dtype=float)
        self._check_width(X_np, "X")
        n = len(X_np)

        if landmark_indices is not None:
            indices = np.asarray(landmark_indices, dtype=int)
        else:
            if n_landmarks is None:
                raise ValueError(
                    "pass n_landmarks (how many rows to use) or explicit "
                    "landmark_indices"
                )

            if not 1 <= n_landmarks <= n:
                raise ValueError(
                    f"n_landmarks must be in [1, {n}], got {n_landmarks}"
                )

            indices = np.linspace(0, n - 1, n_landmarks).round().astype(int)
            indices = np.unique(indices)

        landmarks = X_np[indices]

        cross = self.compute_kernel_matrix(X_np, landmarks)
        landmark_gram = self.compute_kernel_matrix(landmarks)

        # W^(-1/2) via eigendecomposition with clipping, so a rank-deficient
        # landmark set degrades gracefully instead of exploding.
        eigenvalues, eigenvectors = np.linalg.eigh(
            (landmark_gram + landmark_gram.T) / 2
        )
        inv_sqrt = np.where(eigenvalues > 1e-12, 1.0 / np.sqrt(np.maximum(eigenvalues, 1e-12)), 0.0)

        return cross @ (eigenvectors * inv_sqrt) @ eigenvectors.T

    def circuit_budget(self, n_samples, n_landmarks=None):
        """How many circuit evaluations each evaluation path needs.

        A planning tool for hardware runs: the pairwise fidelity path needs a
        circuit per sample pair, the exact statevector path one simulation per
        sample, and the Nystrom path only the landmark blocks. qiskit-ml
        offers no kernel-level resource accounting.
        """
        n = int(n_samples)

        budget = {
            "pairwise_fidelity_circuits": n * (n - 1) // 2,
            "statevector_simulations": n,
        }

        if n_landmarks is not None:
            m = int(n_landmarks)
            budget["nystrom_fidelity_circuits"] = n * m + m * (m - 1) // 2

        return budget

    def _bloch_features(self, X_prepared):
        """Per-qubit Bloch vectors of every encoded state: (n, 3 * n_qubits).

        For each sample the encoding circuit is simulated once and every
        qubit's reduced density matrix is read off the statevector. A 1-qubit
        state rho = (I + r . sigma) / 2 is fully described by its Bloch vector
        r, and ||rho_a - rho_b||_F^2 == |r_a - r_b|^2 / 2, so distances between
        these features ARE the reduced-density-matrix distances the projected
        kernel is defined over.
        """
        program = self._fused_program()

        if program is not None:
            # Batched simulation and one batched reduction per qubit, in the
            # same feature order as the loop below.
            return self._fused_bloch(program, X_prepared)

        self._refresh_signature()
        n = self.n_qubits
        features = np.empty((len(X_prepared), 3 * n))

        for i, row in enumerate(X_prepared):
            # Cached per row like the statevectors, under a separate key: the
            # Bloch reduction costs a reshape and a matmul per qubit on top of
            # the simulation, and a CV sweep asks for the same rows repeatedly.
            # One row at a time, so a projected kernel at 12 qubits never holds
            # the whole state table.
            key = None if self._circuit_signature is None else (
                self._circuit_signature, "bloch", row.tobytes()
            )
            cached = None if key is None else _cache_get(key)

            if cached is not None:
                features[i] = cached
                continue

            tensor = self._simulate_row(row, store=False).reshape((2,) * n)
            vector = np.empty(3 * n)

            for q in range(n):
                # 2 x 2^(n-1) slice with qubit q leading; rho_q = M M^dagger.
                matrix = np.moveaxis(tensor, q, 0).reshape(2, -1)
                rdm = matrix @ matrix.conj().T

                vector[3 * q] = 2.0 * rdm[0, 1].real
                vector[3 * q + 1] = -2.0 * rdm[0, 1].imag
                vector[3 * q + 2] = (rdm[0, 0] - rdm[1, 1]).real

            features[i] = vector

            if key is not None:
                _cache_put(key, vector)

        return features

    def _projected_kernel_matrix(self, X_prepared, X2_prepared, batch_size):
        # Projected quantum kernel (Huang et al. 2021, "Power of data"):
        # K(x, y) = exp(-gamma * sum_q ||rho_q(x) - rho_q(y)||_F^2). Working
        # from local reduced density matrices instead of global fidelity
        # sidesteps the exponential concentration global fidelity kernels
        # suffer as qubit count grows -- the failure mode diagnose() flags.
        if self.gamma_ is None:
            raise ValueError(
                "gamma='auto' is fitted from the training data; call .fit() "
                "first (or pass a numeric gamma)."
            )

        bloch_a = self._bloch_features(X_prepared)
        bloch_b = bloch_a if X2_prepared is None else self._bloch_features(X2_prepared)

        def rows(chunk):
            rdm_sq = 0.5 * self._sq_distances(chunk, bloch_b)
            return np.exp(-self.gamma_ * rdm_sq)

        if batch_size is None:
            return rows(bloch_a)

        return np.vstack([
            rows(bloch_a[start:start + batch_size])
            for start in range(0, len(bloch_a), batch_size)
        ])

    def compute_kernel_matrix(self, X, X2=None, batch_size=None):
        # ZZ mode: plain quantum fidelity kernel.
        # covariant mode: covariance-adapted (whitened) quantum fidelity kernel.
        if batch_size is not None and (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, numbers.Integral)
            or batch_size < 1
        ):
            # Every path used to fail on its own terms (a range() error, an
            # empty vstack, or -- on torch -- a silently wrong-shaped matrix).
            raise ValueError(
                f"batch_size must be a positive integer or None, got {batch_size!r}"
            )

        X_prepared = self._prepare(X, "X")
        X2_prepared = self._prepare(X2, "X2") if X2 is not None else None

        if self.kernel == "projected":
            return self._projected_kernel_matrix(X_prepared, X2_prepared, batch_size)

        if self._use_statevector():
            return self._compute_kernel_matrix_statevector(
                X_prepared, X2_prepared, batch_size
            )

        if batch_size is None:
            if X2_prepared is None:
                return self._fidelity_quantum_kernel.evaluate(X_prepared)

            return self._fidelity_quantum_kernel.evaluate(X_prepared, X2_prepared)

        return self._compute_kernel_matrix_batched(X_prepared, X2_prepared, batch_size)

    def _compute_kernel_matrix_statevector(self, X_prepared, X2_prepared, batch_size):
        # Unlike the pairwise path there is no per-entry circuit to bound, so
        # batch_size only chunks the final matmul; the statevector table itself
        # is O(n * 2^n_qubits) and stays small at the qubit counts this kernel
        # is usable at.
        #
        # Both tables are simulated ONCE, before chunking. Passing each chunk
        # through _statevector_kernel (as this did) re-simulated the entire
        # target table per chunk, so batching cost O(n^2 / batch_size)
        # simulations instead of O(n) -- the smaller the batch, the slower it
        # got, which is backwards. The chunk shapes below are unchanged, so the
        # numbers are too.
        program = self._fused_program()

        if program is not None:
            states_a = self._fused_states(program, X_prepared)
            states_b = (states_a if X2_prepared is None
                        else self._fused_states(program, X2_prepared))
            return self._fused_gram(states_a, states_b, batch_size)

        states_a = self._simulate(X_prepared)
        states_b = states_a if X2_prepared is None else self._simulate(X2_prepared)

        if batch_size is None:
            return np.abs(states_a.conj() @ states_b.T) ** 2

        rows = [
            np.abs(states_a[start:start + batch_size].conj() @ states_b.T) ** 2
            for start in range(0, states_a.shape[0], batch_size)
        ]

        return np.vstack(rows)

    def _compute_kernel_matrix_batched(self, X_prepared, X2_prepared, batch_size):
        # Evaluates the kernel a row-chunk of X at a time so that no single
        # fidelity evaluation has to build circuits for the full cross product
        # at once. This bounds peak memory to batch_size * len(target) pairs
        # instead of len(X) * len(target), at the cost of the symmetry shortcut
        # a plain evaluate(X) can take when X2 is None.
        target = X_prepared if X2_prepared is None else X2_prepared

        rows = [
            self._fidelity_quantum_kernel.evaluate(X_prepared[start:start + batch_size], target)
            for start in range(0, X_prepared.shape[0], batch_size)
        ]

        return np.vstack(rows)


def plot_hilbert_space(kernel_matrix, labels=None, *, show=True, save_path=None):
    """Project a kernel matrix into 3D and plot it.

    ``show`` defaults to True, matching every previous release. Pass
    ``show=False`` when running headless (CI, a benchmark sandbox, a batch job)
    so the figure is returned without trying to open a browser, and
    ``save_path`` to write it to an HTML file.

    :class:`~qmlf.viz.q_viz_pro.QVizPro.plot_hilbert_space` is the fuller
    version of this, with a choice of projection method and perplexity.
    """
    import plotly.graph_objects as go

    kernel_matrix = np.asarray(kernel_matrix, dtype=float)
    n_samples = kernel_matrix.shape[0]

    if n_samples > 3:
        perplexity = min(30, max(2, n_samples - 1))
        tsne = TSNE(n_components=3, random_state=42, perplexity=perplexity)
        embedding = tsne.fit_transform(kernel_matrix)
    else:
        pca = PCA(n_components=min(3, n_samples))
        embedding = pca.fit_transform(kernel_matrix)
        if embedding.shape[1] < 3:
            padded = np.zeros((embedding.shape[0], 3))
            padded[:, :embedding.shape[1]] = embedding
            embedding = padded

    fig = go.Figure(data=[go.Scatter3d(
        x=embedding[:, 0],
        y=embedding[:, 1],
        z=embedding[:, 2],
        mode='markers',
        marker=dict(
            size=8,
            color=labels if labels is not None else embedding[:, 0],
            colorscale='Viridis',
            opacity=0.8
        )
    )])

    fig.update_layout(title="Quantum Hilbert Space Visualization")

    if save_path is not None:
        fig.write_html(save_path)

    if show:
        fig.show()

    return fig


def run_quantum_benchmark(X, y, n_qubits=4, reps=2, test_size=0.2, *,
                          mode="ZZ", bandwidth=1.0, normalize=None,
                          feature_map="zz", entanglement=_UNSET, simulator="qiskit"):
    # Imported here rather than at module scope: importing torch (via qmlf.qnn)
    # before xgboost segfaults the interpreter on macOS over duplicate libomp.
    import xgboost as xgb
    from sklearn.svm import SVC

    X = np.asarray(X, dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    xgb_model = xgb.XGBClassifier(eval_metric="logloss")
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)

    # The feature map dimension must equal the number of input features, so the
    # quantum kernel is built over all features. n_qubits is only honoured when
    # it already agrees with the data; a disagreement used to be swallowed
    # silently, which made the parameter look respected when it never was.
    n_features = X.shape[1]

    if n_qubits != n_features:
        warnings.warn(
            f"run_quantum_benchmark(n_qubits={n_qubits}) but X has "
            f"{n_features} features; the kernel is built over all "
            f"{n_features} features and n_qubits is ignored.",
            UserWarning,
            stacklevel=2
        )
    quantum_kernel = QuantumKernel(
        n_qubits=n_features,
        mode=mode,
        reps=reps,
        bandwidth=bandwidth,
        normalize=normalize,
        feature_map=feature_map,
        entanglement=entanglement,
        simulator=simulator
    )
    quantum_kernel.fit(X_train)

    # Routed through compute_kernel_matrix, NOT by handing
    # .fidelity_quantum_kernel to QSVC. The latter bypasses _prepare, which
    # silently discards whitening, normalize and bandwidth -- making
    # mode="covariant" indistinguishable from mode="ZZ" with no error raised.
    # A precomputed kernel keeps every preprocessing step in the path.
    kernel_train = quantum_kernel.compute_kernel_matrix(X_train)
    kernel_test = quantum_kernel.compute_kernel_matrix(X_test, X_train)

    svc = SVC(kernel="precomputed")
    svc.fit(kernel_train, y_train)
    y_pred_qsvc = svc.predict(kernel_test)

    results = {
        "Model": ["XGBoost", "Quantum SVC"],
        "Accuracy": [
            accuracy_score(y_test, y_pred_xgb),
            accuracy_score(y_test, y_pred_qsvc)
        ],
        "F1 Score": [
            f1_score(y_test, y_pred_xgb, average='weighted'),
            f1_score(y_test, y_pred_qsvc, average='weighted')
        ]
    }

    return pd.DataFrame(results)
