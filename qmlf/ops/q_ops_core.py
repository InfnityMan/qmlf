import warnings

import numpy as np
import pandas as pd
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import z_feature_map, zz_feature_map
from qiskit.quantum_info import Statevector
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
    the two interact; sweeping them together is reasonable. On at least one real
    dataset the unentangled ``"z"`` map matched or beat ``"zz"`` while running
    6-11x faster, so the entangled default is not automatically the right
    choice.

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
                 fast_statevector="auto", kernel="fidelity", gamma="auto"):
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

        if gamma != "auto" and (not np.isscalar(gamma) or not np.isfinite(gamma) or gamma <= 0):
            raise ValueError(
                f"gamma must be a positive finite number or 'auto', got {gamma!r}"
            )

        self.kernel = kernel
        self.gamma = gamma
        self.gamma_ = None if gamma == "auto" else float(gamma)
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
        n = gram.shape[0]
        centering = np.eye(n) - np.ones((n, n)) / n
        gram_c = centering @ gram @ centering
        target_c = centering @ target @ centering
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

    def _statevector_kernel(self, A, B=None):
        # For a pure-state fidelity kernel, K[i, j] = |<phi(x_i)|phi(x_j)>|^2.
        # Simulating n statevectors once and taking |Psi Psi^dagger|^2 is
        # mathematically identical to evaluating n^2 circuit pairs, but costs
        # O(n) simulations plus one matmul instead of O(n^2) circuit builds.
        # The result is PSD by the Schur product theorem, so enforce_psd would
        # be a no-op on it.
        def states(M):
            return np.stack([
                np.asarray(Statevector(self.feature_map.assign_parameters(row)).data)
                for row in M
            ])

        states_a = states(A)
        states_b = states_a if B is None else states(B)

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
        n = self.n_qubits
        features = np.empty((len(X_prepared), 3 * n))

        for i, row in enumerate(X_prepared):
            amplitudes = np.asarray(
                Statevector(self.feature_map.assign_parameters(row)).data
            )
            tensor = amplitudes.reshape((2,) * n)

            for q in range(n):
                # 2 x 2^(n-1) slice with qubit q leading; rho_q = M M^dagger.
                matrix = np.moveaxis(tensor, q, 0).reshape(2, -1)
                rdm = matrix @ matrix.conj().T

                features[i, 3 * q] = 2.0 * rdm[0, 1].real
                features[i, 3 * q + 1] = -2.0 * rdm[0, 1].imag
                features[i, 3 * q + 2] = (rdm[0, 0] - rdm[1, 1]).real

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
        if batch_size is None:
            return self._statevector_kernel(X_prepared, X2_prepared)

        rows = [
            self._statevector_kernel(X_prepared[start:start + batch_size],
                                     X_prepared if X2_prepared is None else X2_prepared)
            for start in range(0, X_prepared.shape[0], batch_size)
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
                          feature_map="zz", entanglement=_UNSET):
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
        entanglement=entanglement
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
