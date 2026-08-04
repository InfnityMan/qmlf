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
_VALID_MODES = ("ZZ",) + _COVARIANT_MODES

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

    All parameters after ``shrinkage`` are keyword-only, and their defaults
    reproduce the previous numeric behaviour exactly.
    """

    def __init__(self, n_qubits, mode="ZZ", reps=2, shrinkage=1e-3, *,
                 bandwidth=1.0, normalize=None, feature_map="zz",
                 entanglement=_UNSET, fidelity=None, sampler=None,
                 fast_statevector="auto"):
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

        if not np.isfinite(bandwidth) or bandwidth <= 0:
            raise ValueError(f"bandwidth must be a positive finite number, got {bandwidth}")

        # Whitened angles routinely leave [-pi, pi] and alias, which collapses
        # the kernel; without a normalisation this mode has been observed to go
        # fully degenerate (every sample predicted as one class). Warn rather
        # than change the default, which would be a breaking numeric change.
        if mode in _COVARIANT_MODES and normalize is None:
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

        self._set_feature_map(
            self._build_feature_map(feature_map, entanglement)
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
        return self._feature_map

    @feature_map.setter
    def feature_map(self, circuit):
        self._set_feature_map(circuit)

    @property
    def fidelity_quantum_kernel(self):
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

    def fit(self, X_train, y=None):
        X_np = np.asarray(X_train, dtype=float)
        self.X_train_ = X_np

        if self.mode in _COVARIANT_MODES:
            self.mean_ = X_np.mean(axis=0)
            centered = X_np - self.mean_
            self.cov_matrix = np.cov(centered, rowvar=False)
            self.whitening_matrix = self._compute_whitening(self.cov_matrix)

        # The normalisation scale is fitted here, on training data only, and
        # reused verbatim by _prepare at predict time -- same no-leakage
        # contract as mean_ / whitening_matrix above.
        if self.normalize is not None:
            self.scale_ = self._compute_scale(self._transform_mode(X_np))

        return self

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

    def _prepare(self, X):
        X_np = np.asarray(X, dtype=float)
        X_np = self._transform_mode(X_np)

        if self.normalize is not None:
            if self.scale_ is None:
                raise ValueError("Must call .fit() before using normalize")
            X_np = X_np / self.scale_

        # Bandwidth is applied last so it survives whitening (see class
        # docstring); at the default of 1.0 this is an exact no-op.
        return X_np * self.bandwidth

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

    def compute_kernel_matrix(self, X, X2=None, batch_size=None):
        # ZZ mode: plain quantum fidelity kernel.
        # covariant mode: covariance-adapted (whitened) quantum fidelity kernel.
        X_prepared = self._prepare(X)
        X2_prepared = self._prepare(X2) if X2 is not None else None

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


def plot_hilbert_space(kernel_matrix, labels=None):
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
    # quantum kernel is built over all features (n_qubits is a hint).
    n_features = X.shape[1]
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
