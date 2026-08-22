import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from qmlf.ops.q_ops_core import QuantumKernel

# Whitened modes take their safe normalisation by default (see __init__).
# fisher whitens too (by within-class scatter), so it needs the same guard.
_COVARIANT_MODES = ("covariant", "mahalanobis", "fisher")

# The grid the QuantumKernel docstring tells humans to sweep by hand. Ordered
# widest-first so an argmax tie resolves to the least aggressive scaling.
_DEFAULT_BANDWIDTH_GRID = (1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01)

# What "auto" expands to. Kept small on purpose: these four combinations cover
# the entangled/unentangled and raw/whitened axes, which is where the real
# accuracy differences live (see the README's measured sweeps).
_AUTO_MODES = ("ZZ", "mahalanobis")
_AUTO_FEATURE_MAPS = ("zz", "z")
_AUTO_KERNELS = ("fidelity", "projected")

# Each extra qubit doubles the statevector, so wide data is reduced to this
# many dimensions before encoding unless the caller says otherwise.
_DEFAULT_MAX_QUBITS = 8

# Off-diagonal Gram mass below these marks a concentrated kernel; the levels
# come from the measured reference sweep (library defaults 0.017 = chance-level
# accuracy, healthy tuned kernels 0.17-0.77).
_SEVERE_CONCENTRATION = 0.05
_MILD_CONCENTRATION = 0.15


def kernel_diagnostics(gram, y=None):
    """Health report for a quantum kernel matrix.

    Fidelity kernels fail in a characteristic way -- concentration: the
    off-diagonal mass collapses toward zero, the Gram matrix approaches the
    identity, and a downstream SVM memorises instead of generalising. Neither
    qiskit-machine-learning nor TensorFlow Quantum ships a built-in check for
    it; this is the one-call version.

    Returns a dict with:

    - ``offdiag_mean`` -- mean off-diagonal Gram value. The single best
      concentration indicator.
    - ``top_eigenvalue_fraction`` -- share of spectral mass in the largest
      eigenvalue; near ``1/n`` for an identity-like (concentrated) kernel.
    - ``kernel_target_alignment`` -- when ``y`` is given: centered kernel
      alignment (Cortes et al. 2012) between the Gram matrix and the ideal
      same-class/different-class matrix (+1/-1).

      Read the two numbers TOGETHER: alignment is scale-invariant, so it
      measures whether the kernel's structure points in a class-consistent
      *direction*, while ``offdiag_mean`` measures whether there is enough
      *scale* for an SVM to use that direction. A concentrated kernel can
      show respectable alignment (its residual off-diagonal structure is
      class-correlated) and still be unusable -- measured on the reference
      dataset, the bandwidth=1.0 trap kernel had alignment 0.13 yet scored
      chance-level accuracy, because its off-diagonal mass was 0.017. The
      ``verdict`` is therefore driven by concentration, with alignment as
      the complementary direction diagnostic.
    - ``verdict`` -- ``"severely concentrated"`` / ``"concentrated"`` /
      ``"healthy"``, thresholds calibrated on the measured reference sweep.
    """
    gram = np.asarray(gram, dtype=float)

    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError(
            f"gram must be a square kernel matrix, got shape {gram.shape}"
        )

    n = gram.shape[0]
    off_mask = ~np.eye(n, dtype=bool)
    offdiag_mean = float(gram[off_mask].mean())

    eigenvalues = np.linalg.eigvalsh((gram + gram.T) / 2)
    total = float(eigenvalues.sum())
    top_fraction = float(eigenvalues.max() / total) if total > 0 else float("nan")

    alignment = None

    if y is not None:
        y = np.asarray(y)
        target = np.where(y[:, None] == y[None, :], 1.0, -1.0)

        # Centered alignment: H = I - 11^T/n double-centres both matrices,
        # removing the diagonal/mean bias that lets an identity-like kernel
        # score well against the +1 diagonal of the target.
        centering = np.eye(n) - np.ones((n, n)) / n
        gram_centered = centering @ gram @ centering
        target_centered = centering @ target @ centering

        denom = np.linalg.norm(gram_centered) * np.linalg.norm(target_centered)
        alignment = float((gram_centered * target_centered).sum() / denom) \
            if denom > 0 else float("nan")

    if offdiag_mean < _SEVERE_CONCENTRATION:
        verdict = "severely concentrated"
    elif offdiag_mean < _MILD_CONCENTRATION:
        verdict = "concentrated"
    else:
        verdict = "healthy"

    return {
        "offdiag_mean": offdiag_mean,
        "top_eigenvalue_fraction": top_fraction,
        "kernel_target_alignment": alignment,
        "verdict": verdict,
    }


class QuantumClassifier(BaseEstimator, ClassifierMixin):
    """Quantum-kernel SVM with no circuit plumbing: ``fit``, ``predict``, done.

    The full quantum pipeline -- dimensionality reduction when the data is
    wider than the simulator can afford, the encoding circuit sized from the
    (reduced) feature count, fidelity Gram matrices in the right orientation,
    the ``SVC(kernel="precomputed")`` wiring, and the hyperparameter selection
    that separates a working fidelity kernel from a degenerate one -- happens
    inside ``fit``:

        clf = QuantumClassifier()
        clf.fit(X_train, y_train)
        clf.predict(X_test)

    Nothing must be chosen; everything can be. Every automatic decision has an
    override and reports what it decided:

    ==================  =========================  ========================
    automatic           override                   decision visible in
    ==================  =========================  ========================
    qubit count         ``max_qubits=`` / ``None``  ``n_qubits_``, ``reduction_``
    bandwidth sweep     ``bandwidth=0.05``          ``bandwidth_``, ``cv_results_``
    sweep grid          ``bandwidths=(...)``        ``cv_results_``
    encoding search     ``mode=``/``feature_map=``  ``mode_``, ``feature_map_``
    safe normalisation  ``normalize=``              ``_resolved_normalize()``
    ==================  =========================  ========================

    Selection
    ---------
    ``bandwidth="auto"`` (default) sweeps ``bandwidths`` by deterministic
    stratified CV -- each fold fits its own reducer and kernel, so nothing
    leaks. ``mode="auto"`` and/or ``feature_map="auto"`` extend the same sweep
    across the entangled/unentangled (``"zz"``/``"z"``) and raw/whitened
    (``"ZZ"``/``"mahalanobis"``) axes, so the *encoding itself* is selected by
    data rather than by the user -- the encoding is a first-class searchable
    hyperparameter here, which neither qiskit-machine-learning's ``QSVC`` nor
    TensorFlow Quantum provides as a built-in. Defaults stay ``"ZZ"``/``"zz"``
    so the zero-argument path remains fast.

    Wide data
    ---------
    One qubit per feature is unaffordable past a dozen qubits on an exact
    simulator. When ``X`` is wider than ``max_qubits`` (default 8), a
    deterministic PCA fitted on training data only brings it down first;
    ``reduction_`` says so and ``max_qubits=None`` disables it (then wide data
    warns instead).

    Diagnostics
    -----------
    ``diagnose()`` returns the :func:`kernel_diagnostics` health report for
    the fitted training Gram matrix -- concentration, spectrum, and
    kernel-target alignment in one call.

    Everything is exact-simulator based and deterministic: two identical fits
    produce identical predictions.
    """

    def __init__(self, mode="ZZ", bandwidth="auto", normalize=None,
                 feature_map="zz", entanglement=None, reps=2, C=1.0,
                 bandwidths=None, cv=3, max_qubits=_DEFAULT_MAX_QUBITS,
                 kernel="fidelity", gamma="auto"):
        self.kernel = kernel
        self.gamma = gamma
        self.mode = mode
        self.bandwidth = bandwidth
        self.normalize = normalize
        self.feature_map = feature_map
        self.entanglement = entanglement
        self.reps = reps
        self.C = C
        self.bandwidths = bandwidths
        self.cv = cv
        self.max_qubits = max_qubits

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------

    def _resolved_normalize(self, mode=None):
        mode = self.mode if mode is None else mode

        if self.normalize is None and mode in _COVARIANT_MODES:
            return "maxabs"

        if self.normalize == "none":
            return None

        return self.normalize

    def _make_kernel(self, kernel, mode, feature_map, bandwidth):
        kwargs = {}

        if self.entanglement is not None:
            kwargs["entanglement"] = self.entanglement

        return QuantumKernel(
            n_qubits=None,
            mode=mode,
            reps=self.reps,
            bandwidth=bandwidth,
            normalize=self._resolved_normalize(mode),
            feature_map=feature_map,
            kernel=kernel,
            gamma=self.gamma,
            **kwargs
        )

    def _make_reducer(self, X):
        """A PCA to ``max_qubits`` dimensions, or None when X already fits.

        svd_solver='full' keeps the projection deterministic; the randomized
        solver sklearn may otherwise pick would make two identical fits
        disagree.
        """
        if self.max_qubits is None or X.shape[1] <= self.max_qubits:
            return None

        return PCA(n_components=self.max_qubits, svd_solver="full").fit(X)

    def _candidate_grid(self):
        kernels = _AUTO_KERNELS if self.kernel == "auto" else (self.kernel,)
        modes = _AUTO_MODES if self.mode == "auto" else (self.mode,)
        maps = _AUTO_FEATURE_MAPS if self.feature_map == "auto" else (self.feature_map,)

        if self.bandwidth == "auto":
            bandwidths = tuple(self.bandwidths) if self.bandwidths is not None \
                else _DEFAULT_BANDWIDTH_GRID
        elif self.bandwidth == "median":
            # A single candidate; the kernel fits the value itself from the
            # training data (median heuristic), no sweep needed.
            bandwidths = ("median",)
        else:
            bandwidths = (float(self.bandwidth),)

        return [
            (k, m, fm, bw)
            for k in kernels for m in modes for fm in maps for bw in bandwidths
        ]

    # ------------------------------------------------------------------
    # selection
    # ------------------------------------------------------------------

    def _select_configuration(self, X, y):
        grid = self._candidate_grid()

        if len(grid) == 1:
            self.cv_results_ = None
            return grid[0]

        _, class_counts = np.unique(y, return_counts=True)
        n_splits = int(min(self.cv, class_counts.min()))

        if n_splits < 2:
            # Nothing to cross-validate against; the first candidate is the
            # widest bandwidth of the default mode -- the least-degenerate
            # blind choice.
            self.cv_results_ = None
            return grid[0]

        # shuffle=False: the sweep must give the same answer every run.
        folds = list(StratifiedKFold(n_splits=n_splits, shuffle=False).split(X, y))
        means = []

        for kernel_type, mode, feature_map, bandwidth in grid:
            scores = []

            for train_idx, valid_idx in folds:
                X_train, X_valid = X[train_idx], X[valid_idx]

                # Reducer AND kernel are refitted per fold on that fold's
                # training rows only, so neither the projection nor the
                # whitening/normalisation ever sees validation data.
                reducer = self._make_reducer(X_train)

                if reducer is not None:
                    X_train = reducer.transform(X_train)
                    X_valid = reducer.transform(X_valid)

                kernel = self._make_kernel(
                    kernel_type, mode, feature_map, bandwidth
                ).fit(X_train, y[train_idx])

                gram_train = kernel.compute_kernel_matrix(X_train)
                gram_valid = kernel.compute_kernel_matrix(X_valid, X_train)

                svc = SVC(kernel="precomputed", C=self.C)
                svc.fit(gram_train, y[train_idx])
                scores.append(svc.score(gram_valid, y[valid_idx]))

            means.append(float(np.mean(scores)))

        self.cv_results_ = {
            "kernel": [k for k, _, _, _ in grid],
            "mode": [m for _, m, _, _ in grid],
            "feature_map": [fm for _, _, fm, _ in grid],
            "bandwidth": [bw for _, _, _, bw in grid],
            "mean_cv_accuracy": means,
        }

        # argmax takes the first maximum; the grid is ordered widest-bandwidth
        # first within each combination, so ties resolve away from aggressive
        # scaling.
        return grid[int(np.argmax(means))]

    # ------------------------------------------------------------------
    # sklearn surface
    # ------------------------------------------------------------------

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError(
                f"X must be 2D (n_samples, n_features), got a {X.ndim}D "
                f"array of shape {X.shape}"
            )

        if len(X) != len(y):
            raise ValueError(
                f"X has {len(X)} samples but y has {len(y)} labels"
            )

        if len(np.unique(y)) < 2:
            raise ValueError("y must contain at least two classes")

        if self.max_qubits is None and X.shape[1] > 12:
            warnings.warn(
                f"X has {X.shape[1]} features -> {X.shape[1]} qubits with "
                "max_qubits=None; the exact statevector simulation doubles in "
                "cost per qubit and is impractical much past 12.",
                UserWarning,
                stacklevel=2
            )

        kernel_type, mode, feature_map, bandwidth = self._select_configuration(X, y)

        self.kernel_ = kernel_type
        self.mode_ = mode
        self.feature_map_ = feature_map

        self._reducer = self._make_reducer(X)
        self.reduction_ = None if self._reducer is None else "pca"

        X_encoded = X if self._reducer is None else self._reducer.transform(X)

        self._kernel = self._make_kernel(
            kernel_type, mode, feature_map, bandwidth
        ).fit(X_encoded, y)

        # "median" is resolved by the kernel during fit; report the number
        # actually used either way.
        self.bandwidth_ = float(
            self._kernel.bandwidth_ if isinstance(bandwidth, str) else bandwidth
        )
        self._X_fit = X_encoded
        self._y_fit = y

        self._gram_fit = self._kernel.compute_kernel_matrix(X_encoded)

        self._svc = SVC(kernel="precomputed", C=self.C)
        self._svc.fit(self._gram_fit, y)

        self.classes_ = self._svc.classes_
        self.n_qubits_ = self._kernel.n_qubits

        # Post-fit guard: a severely concentrated kernel means the SVM is
        # memorising, whatever the training accuracy says. Better to hear it
        # now than on the test set.
        report = kernel_diagnostics(self._gram_fit)

        if report["verdict"] == "severely concentrated":
            warnings.warn(
                f"The fitted training kernel is severely concentrated "
                f"(off-diagonal mean {report['offdiag_mean']:.4f}); the model "
                "will generalise poorly. A smaller bandwidth, bandwidth='auto', "
                "or kernel='projected' usually fixes this. See .diagnose().",
                UserWarning,
                stacklevel=2
            )

        return self

    def _cross_gram(self, X):
        if not hasattr(self, "_svc"):
            raise ValueError(
                "This QuantumClassifier has not been fitted yet; call "
                ".fit(X, y) first."
            )

        X = np.asarray(X, dtype=float)

        if self._reducer is not None:
            X = self._reducer.transform(X)

        return self._kernel.compute_kernel_matrix(X, self._X_fit)

    def predict(self, X):
        # The gram call carries the fitted-check, so it must run before the
        # self._svc attribute lookup or an unfitted call surfaces as a bare
        # AttributeError instead of the explanation.
        gram = self._cross_gram(X)
        return self._svc.predict(gram)

    def decision_function(self, X):
        gram = self._cross_gram(X)
        return self._svc.decision_function(gram)

    def kernel_matrix(self, X, X2=None):
        """The fitted kernel's Gram matrix, for inspection or plotting."""
        if not hasattr(self, "_svc"):
            raise ValueError(
                "This QuantumClassifier has not been fitted yet; call "
                ".fit(X, y) first."
            )

        X = np.asarray(X, dtype=float)

        if self._reducer is not None:
            X = self._reducer.transform(X)
            X2 = None if X2 is None else self._reducer.transform(
                np.asarray(X2, dtype=float)
            )

        return self._kernel.compute_kernel_matrix(X, X2)

    def diagnose(self):
        """Health report for the fitted training kernel (see
        :func:`kernel_diagnostics`)."""
        if not hasattr(self, "_svc"):
            raise ValueError(
                "This QuantumClassifier has not been fitted yet; call "
                ".fit(X, y) first."
            )

        return kernel_diagnostics(self._gram_fit, self._y_fit)


def create_quantum_classifier(**kwargs):
    return QuantumClassifier(**kwargs)


class QuantumRegressor(BaseEstimator, RegressorMixin):
    """Quantum-kernel ridge regression with the same no-plumbing contract.

    qiskit-machine-learning does ship a QSVR -- but bare: the user still
    builds the feature map, has no bandwidth (the parameter does not exist
    there), no dimensionality handling, no projected kernels and no tuning.
    This regressor carries the full auto stack: qubits sized (and wide data
    PCA-reduced) from the data, bandwidth selected by deterministic K-fold CV
    over the ridge fit (or ``"median"`` for the CV-free heuristic), and
    ``kernel="projected"`` available end to end.

    The model is kernel ridge: ``coef = (K + alpha I)^{-1} y``, prediction
    ``K_cross @ coef``. ``alpha`` is the ridge regularisation.

    ``mode="fisher"`` and ``bandwidth="alignment"`` are classification-only
    (both need class labels) and are rejected with a clear error.
    """

    def __init__(self, mode="ZZ", bandwidth="auto", normalize=None,
                 feature_map="zz", entanglement=None, reps=2, alpha=1e-3,
                 bandwidths=None, cv=3, max_qubits=_DEFAULT_MAX_QUBITS,
                 kernel="fidelity", gamma="auto"):
        self.kernel = kernel
        self.gamma = gamma
        self.mode = mode
        self.bandwidth = bandwidth
        self.normalize = normalize
        self.feature_map = feature_map
        self.entanglement = entanglement
        self.reps = reps
        self.alpha = alpha
        self.bandwidths = bandwidths
        self.cv = cv
        self.max_qubits = max_qubits

    _resolved_normalize = QuantumClassifier._resolved_normalize
    _make_kernel = QuantumClassifier._make_kernel
    _make_reducer = QuantumClassifier._make_reducer
    _candidate_grid = QuantumClassifier._candidate_grid

    def _validate_options(self):
        if self.mode == "fisher":
            raise ValueError(
                "mode='fisher' whitens by within-class scatter and needs "
                "class labels; it is classification-only. Use 'ZZ' or "
                "'mahalanobis' for regression."
            )

        if self.bandwidth == "alignment":
            raise ValueError(
                "bandwidth='alignment' maximises class-label alignment and is "
                "classification-only. Use 'auto', 'median', or a number."
            )

    def _ridge_coef(self, gram, y):
        n = len(gram)
        return np.linalg.solve(gram + self.alpha * np.eye(n), y)

    def _select_configuration(self, X, y):
        grid = self._candidate_grid()

        if len(grid) == 1:
            self.cv_results_ = None
            return grid[0]

        n_splits = int(min(self.cv, len(X)))

        if n_splits < 2:
            self.cv_results_ = None
            return grid[0]

        from sklearn.model_selection import KFold

        # shuffle=False: deterministic, same folds every run.
        folds = list(KFold(n_splits=n_splits, shuffle=False).split(X))
        means = []

        for kernel_type, mode, feature_map, bandwidth in grid:
            errors = []

            for train_idx, valid_idx in folds:
                X_train, X_valid = X[train_idx], X[valid_idx]

                reducer = self._make_reducer(X_train)

                if reducer is not None:
                    X_train = reducer.transform(X_train)
                    X_valid = reducer.transform(X_valid)

                kernel = self._make_kernel(
                    kernel_type, mode, feature_map, bandwidth
                ).fit(X_train)

                gram_train = kernel.compute_kernel_matrix(X_train)
                gram_valid = kernel.compute_kernel_matrix(X_valid, X_train)

                coef = self._ridge_coef(gram_train, y[train_idx])
                predictions = gram_valid @ coef
                errors.append(float(np.mean((predictions - y[valid_idx]) ** 2)))

            means.append(float(np.mean(errors)))

        self.cv_results_ = {
            "kernel": [k for k, _, _, _ in grid],
            "mode": [m for _, m, _, _ in grid],
            "feature_map": [fm for _, _, fm, _ in grid],
            "bandwidth": [bw for _, _, _, bw in grid],
            "mean_cv_mse": means,
        }

        # Lowest validation MSE wins; ties resolve to the widest bandwidth.
        return grid[int(np.argmin(means))]

    def fit(self, X, y):
        self._validate_options()

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2:
            raise ValueError(
                f"X must be 2D (n_samples, n_features), got a {X.ndim}D "
                f"array of shape {X.shape}"
            )

        if len(X) != len(y):
            raise ValueError(
                f"X has {len(X)} samples but y has {len(y)} targets"
            )

        kernel_type, mode, feature_map, bandwidth = self._select_configuration(X, y)

        self.kernel_ = kernel_type
        self.mode_ = mode
        self.feature_map_ = feature_map

        self._reducer = self._make_reducer(X)
        self.reduction_ = None if self._reducer is None else "pca"

        X_encoded = X if self._reducer is None else self._reducer.transform(X)

        self._kernel = self._make_kernel(
            kernel_type, mode, feature_map, bandwidth
        ).fit(X_encoded)

        self.bandwidth_ = float(
            self._kernel.bandwidth_ if isinstance(bandwidth, str) else bandwidth
        )

        self._X_fit = X_encoded
        gram = self._kernel.compute_kernel_matrix(X_encoded)
        self.coef_ = self._ridge_coef(gram, y)
        self.n_qubits_ = self._kernel.n_qubits

        return self

    def predict(self, X):
        if not hasattr(self, "coef_"):
            raise ValueError(
                "This QuantumRegressor has not been fitted yet; call "
                ".fit(X, y) first."
            )

        X = np.asarray(X, dtype=float)

        if self._reducer is not None:
            X = self._reducer.transform(X)

        gram = self._kernel.compute_kernel_matrix(X, self._X_fit)

        return gram @ self.coef_


def create_quantum_regressor(**kwargs):
    return QuantumRegressor(**kwargs)
