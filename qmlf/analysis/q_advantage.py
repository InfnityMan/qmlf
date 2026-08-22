"""Quantum-advantage screening for kernel methods.

Implements the geometric-difference test of Huang et al., "Power of data in
quantum machine learning" (Nature Communications 12, 2631 (2021)): a
data-driven answer to the question every quantum kernel project should ask
first -- CAN a quantum kernel outperform a classical one on THIS dataset?

Neither qiskit-machine-learning nor TensorFlow Quantum ships any form of this
(verified: zero source references). The usual workflow is to train both models
end to end and compare test scores; the geometric difference answers the
question from the Gram matrices alone, before any training.
"""
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

from qmlf.ops.q_ops_core import QuantumKernel


def geometric_difference(classical_gram, quantum_gram, regularization=1e-7):
    """g(K_C || K_Q) of Huang et al. 2021, eq. (5).

    ``g = sqrt(|| sqrt(K_Q) (K_C + lambda I)^{-1} sqrt(K_Q) ||_inf)`` (spectral
    norm), with both Gram matrices trace-normalised to ``n`` first, as the
    paper prescribes, so the scale of either kernel cannot fake a separation.

    Interpretation (theirs, not ours): if ``g`` is O(1), a classical learner
    with kernel ``K_C`` can match anything the quantum kernel does on this
    data, so there is no point running circuits. A potential quantum advantage
    requires ``g`` on the order of ``sqrt(n)``.
    """
    K_c = np.asarray(classical_gram, dtype=float)
    K_q = np.asarray(quantum_gram, dtype=float)

    if K_c.shape != K_q.shape or K_c.ndim != 2 or K_c.shape[0] != K_c.shape[1]:
        raise ValueError(
            f"both Gram matrices must be square and same-shaped, got "
            f"{K_c.shape} and {K_q.shape}"
        )

    n = K_c.shape[0]

    # Trace-normalise to n (paper convention: tr(K) = n).
    K_c = n * K_c / np.trace(K_c)
    K_q = n * K_q / np.trace(K_q)

    # sqrt(K_Q) via eigendecomposition with clipping.
    eigenvalues, eigenvectors = np.linalg.eigh((K_q + K_q.T) / 2)
    sqrt_q = (eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))) @ eigenvectors.T

    inverse_c = np.linalg.inv(K_c + regularization * n * np.eye(n))
    inner = sqrt_q @ inverse_c @ sqrt_q

    spectral_norm = float(np.linalg.eigvalsh((inner + inner.T) / 2).max())

    return float(np.sqrt(max(spectral_norm, 0.0)))


def model_complexity(gram, y, regularization=1e-7):
    """s_K(y) = y^T (K + lambda I)^{-1} y, trace-normalised (Huang et al.).

    The kernel's cost of fitting these labels: small when the labels lie along
    the kernel's top eigendirections, large (up to ~n) when the kernel finds
    the labels hard. Comparing s for the classical and quantum kernels says
    which geometry the labels actually live in.
    """
    K = np.asarray(gram, dtype=float)
    y = np.asarray(y, dtype=float)
    n = K.shape[0]

    K = n * K / np.trace(K)

    return float(y @ np.linalg.solve(K + regularization * n * np.eye(n), y))


def quantum_advantage_report(X, y=None, quantum_kernel=None, gamma="median",
                             regularization=1e-7):
    """One-call screening: is a quantum kernel worth trying on this data?

    Builds the quantum Gram matrix (default: an auto-sized
    :class:`~qmlf.ops.q_ops_core.QuantumKernel`; pass your own fitted or
    unfitted kernel to screen a specific configuration) and a
    median-heuristic classical RBF Gram matrix, then reports the geometric
    difference and -- when labels are given -- both model complexities.

    The verdict follows the paper's reading conservatively:

    - ``g`` near 1 (below ``n**0.25``): the classical RBF can match the
      quantum kernel on this dataset; save the circuits.
    - ``g`` of order ``sqrt(n)`` or larger AND the labels harder for the
      classical kernel (``s_classical > s_quantum``): the quantum kernel sees
      structure the classical one does not -- a real candidate.
    - in between: geometry differs but not decisively.

    Deterministic throughout.
    """
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError(
            f"X must be 2D (n_samples, n_features), got shape {X.shape}"
        )

    n = len(X)

    kernel = QuantumKernel() if quantum_kernel is None else quantum_kernel
    labels_arg = y if kernel.mode == "fisher" else None
    kernel.fit(X, labels_arg)
    quantum_gram = kernel.compute_kernel_matrix(X)

    # Median-heuristic RBF: the strongest generic classical baseline that
    # needs no tuning, and deterministic.
    sq_distances = (
        np.sum(X ** 2, axis=1)[:, None]
        + np.sum(X ** 2, axis=1)[None, :]
        - 2.0 * (X @ X.T)
    )
    off = sq_distances[~np.eye(n, dtype=bool)]
    median_sq = float(np.median(off)) if off.size else 1.0
    rbf_gamma = 1.0 / median_sq if gamma == "median" else float(gamma)
    classical_gram = rbf_kernel(X, gamma=rbf_gamma)

    g = geometric_difference(classical_gram, quantum_gram, regularization)

    report = {
        "n_samples": n,
        "geometric_difference": g,
        "g_matchable_below": float(n ** 0.25),
        "g_advantage_scale": float(np.sqrt(n)),
        "classical_rbf_gamma": rbf_gamma,
        "s_classical": None,
        "s_quantum": None,
    }

    if y is not None:
        y_arr = np.asarray(y, dtype=float)
        # +-1 encoding for binary labels, per the paper; other targets pass
        # through as-is.
        unique = np.unique(y_arr)
        if len(unique) == 2:
            y_arr = np.where(y_arr == unique[0], -1.0, 1.0)

        report["s_classical"] = model_complexity(classical_gram, y_arr, regularization)
        report["s_quantum"] = model_complexity(quantum_gram, y_arr, regularization)

    if g < report["g_matchable_below"]:
        verdict = (
            "classical kernel can match: the geometric difference is small, "
            "so a classical RBF reproduces anything this quantum kernel "
            "does on this data"
        )
    elif g >= report["g_advantage_scale"] and (
        report["s_classical"] is not None
        and report["s_quantum"] is not None
        and report["s_classical"] > report["s_quantum"]
    ):
        verdict = (
            "quantum candidate: large geometric separation AND the labels "
            "are easier in the quantum geometry"
        )
    else:
        verdict = (
            "inconclusive: the geometries differ, but not decisively for "
            "these labels"
        )

    report["verdict"] = verdict

    return report
