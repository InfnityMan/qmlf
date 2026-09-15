# %%
"""Kaggle Benchmark: can a model tune a quantum fidelity kernel that actually works?

Fidelity kernels concentrate. As the encoding angles widen, off-diagonal
similarities collapse toward zero, the Gram matrix approaches the identity, and
an SVM on top of it memorises the training set. The library's defaults sit
squarely in that regime, so a model that accepts them scores near chance.

Measured on this exact split (see kaggle/reference_sweep.txt):

    mode          bandwidth  normalize   offdiag_mean   test_acc
    ZZ            1.0        None        0.017          0.400   <- library default
    ZZ            0.02       None        0.617          0.750
    mahalanobis   0.05       maxabs      0.771          0.800   <- best known

The model never sees those numbers. It has to know that fidelity kernels
concentrate, that bandwidth is the remedy, that whitening cancels any input
scaling unless a train-fitted normalisation is applied, and that handing
`.fidelity_quantum_kernel` to a classifier silently discards all of it.

Grading is deterministic: the model returns a configuration, this task builds
exactly that configuration and measures it. There is no self-reported score to
game.
"""
import os
import sys

# %%
import kaggle_benchmarks as kbench
from pydantic import BaseModel, Field


# %%
def _ensure_qmlf():
    """Install qmlf and its runtime deps inside the benchmark sandbox.

    Verified against the live sandbox (see kaggle/SANDBOX.md): Python 3.11.15,
    linux, internet available, and only numpy/pandas preinstalled -- scikit-learn,
    scipy and the whole qiskit stack are absent.

    The attached dataset lands at /kaggle/input/datasets/<owner>/<slug>/, one
    level deeper than /kaggle/input/<slug>/, so the wheel is located by walking
    the tree rather than by a fixed glob.

    Installing --no-index from the wheelhouse takes ~15s and needs no network.
    qmlf goes in with --no-deps deliberately: torch and xgboost are declared
    dependencies but are imported lazily and are not on this task's code path,
    so pulling them would cost an 800MB download for nothing.
    """
    import subprocess
    import sys

    # Idempotent: a re-run in a warm sandbox must not reinstall, and this also
    # lets the task be exercised locally where /kaggle/input does not exist.
    try:
        import qmlf
        return qmlf
    except ImportError:
        pass

    def pip(*args):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])

    wheel = None

    for root, _dirs, files in os.walk("/kaggle/input"):
        for name in files:
            if name.startswith("qmlf-") and name.endswith(".whl"):
                wheel = os.path.join(root, name)
                break
        if wheel:
            break

    if wheel:
        found = os.path.dirname(wheel)
        pip("--no-index", "--find-links", found,
            "scikit-learn", "qiskit", "qiskit-machine-learning", "qiskit-algorithms")
        pip("--no-deps", wheel)
    else:
        # No wheelhouse attached: fall back to PyPI. Works today for the
        # dependencies; needs qmlf published for the last line to resolve.
        pip("scikit-learn", "qiskit>=2.4", "qiskit-machine-learning>=0.9",
            "qiskit-algorithms>=0.4")
        pip("--no-deps", "qmlf")

    import qmlf
    return qmlf


# %%
class KernelPlan(BaseModel):
    """The configuration the model proposes. This task builds exactly this."""

    mode: str = Field(description="'ZZ', 'covariant', or 'mahalanobis'")
    bandwidth: float = Field(description="Scale factor on the encoding angles")
    normalize: str = Field(description="'none', 'maxabs', or 'std'")
    feature_map: str = Field(description="'zz' or 'z'")
    entanglement: str = Field(description="'full', 'linear', or 'circular'")
    svm_wiring: str = Field(
        description="'precomputed' to use compute_kernel_matrix with "
                    "SVC(kernel='precomputed'), or 'fidelity_quantum_kernel' to "
                    "hand the raw kernel object to the classifier"
    )
    rationale: str = Field(description="Why this configuration avoids concentration")


PROMPT = """\
You are configuring a quantum fidelity kernel from the `qmlf` library for a
binary classification task, then classifying with a support vector machine.

Data: 80 samples, 6 continuous correlated features, 2 balanced classes,
moderate class separation. 60 train / 20 test, stratified.

The kernel is built as:

    QuantumKernel(n_qubits=6, mode=..., bandwidth=..., normalize=...,
                  feature_map=..., entanglement=...)

You will be scored on two things measured from the Gram matrix your
configuration produces:

  1. Test accuracy of the resulting SVM.
  2. The mean off-diagonal value of the training Gram matrix. A fidelity kernel
     whose off-diagonal mass has collapsed toward zero is degenerate: the Gram
     matrix approaches the identity and the SVM memorises rather than
     generalises. A configuration that scores well on accuracy by luck while
     leaving the kernel concentrated will not pass.

Think carefully about how the encoding angle range affects fidelity between
distinct samples, what whitening does to any scaling applied before it, and how
the kernel must be handed to scikit-learn so that no preprocessing step is
silently discarded.

Return your configuration.\
"""


# %%
# This task is hand-written and is not regenerated by either generator, so the
# scoring helpers are defined here rather than inherited from common.py.
_RECORDED_CHECKS = []


def _assert(ok, expectation):
    """Record a check, then forward it to kbench as a normal assertion."""
    ok = bool(ok)
    _RECORDED_CHECKS.append(ok)
    kbench.assertions.assert_true(ok, expectation=expectation)

    return ok


def _fraction_recorded():
    """Fraction of recorded checks that passed -- this is what Kaggle scores."""
    if not _RECORDED_CHECKS:
        return 0.0

    return float(sum(_RECORDED_CHECKS)) / float(len(_RECORDED_CHECKS))


@kbench.task(
    name="qmlfb-kernel-concentration-rescue",
    description="Configure a non-degenerate quantum fidelity kernel and beat the "
                "library defaults on a 6-qubit classification task."
)
def qmlf_quantum_kernel_tuning(llm) -> float:
    import warnings

    _ensure_qmlf()

    # Imported only after _ensure_qmlf(): the sandbox ships numpy and pandas but
    # not scikit-learn, so these names do not exist until the install has run.
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC

    from qmlf import QuantumKernel

    # No reasoning= flag here on purpose: with a structured schema some models
    # emit their <think> trace into the message body, and the JSON parser then
    # fails on "expected value at line 1 column 1" before the task ever runs.
    plan = llm.prompt(PROMPT, schema=KernelPlan)

    X, y = make_classification(
        n_samples=80, n_features=6, n_informative=4, n_redundant=0,
        class_sep=1.2, random_state=7
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=7, stratify=y
    )

    # Build exactly what the model asked for. An invalid configuration is a
    # failed attempt, not a crashed task.
    build_error = None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            kernel = QuantumKernel(
                n_qubits=6,
                mode=plan.mode,
                bandwidth=plan.bandwidth,
                normalize=None if str(plan.normalize).lower() in ("none", "null", "") else plan.normalize,
                feature_map=plan.feature_map,
                entanglement=plan.entanglement,
            ).fit(X_train)

            gram_train = kernel.compute_kernel_matrix(X_train)
            gram_test = kernel.compute_kernel_matrix(X_test, X_train)

        offdiag = float(gram_train[~np.eye(len(gram_train), dtype=bool)].mean())

        svc = SVC(kernel="precomputed").fit(gram_train, y_train)
        accuracy = float(accuracy_score(y_test, svc.predict(gram_test)))

    except Exception as exc:
        build_error = f"{type(exc).__name__}: {exc}"
        offdiag, accuracy = 0.0, 0.0

    # Reference points measured on this exact split.
    LIBRARY_DEFAULT_ACCURACY = 0.40
    BEST_KNOWN_ACCURACY = 0.80

    _assert(
        build_error is None,
        expectation=f"The configuration must be valid. Got: {build_error}"
    )

    _assert(
        plan.svm_wiring == "precomputed",
        expectation="Must use compute_kernel_matrix with SVC(kernel='precomputed'). "
                    "Passing .fidelity_quantum_kernel to the classifier bypasses "
                    "_prepare and silently discards whitening, normalize and "
                    f"bandwidth. Model chose: {plan.svm_wiring!r}"
    )

    _assert(
        offdiag >= 0.10,
        expectation="The training Gram matrix must not be concentrated: mean "
                    f"off-diagonal >= 0.10. Library defaults give 0.017. Got {offdiag:.4f}"
    )

    _assert(
        accuracy >= 0.70,
        expectation=f"Test accuracy must beat the library default of "
                    f"{LIBRARY_DEFAULT_ACCURACY:.2f} by a clear margin (>= 0.70). "
                    f"Best known is {BEST_KNOWN_ACCURACY:.2f}. Got {accuracy:.4f}"
    )

    # Kaggle scores the RETURN value and accepts only a number or a bool;
    # a dict is stored as resultCase "none" and scores 0.00.
    return _fraction_recorded()


# %%
qmlf_quantum_kernel_tuning.run(kbench.llm)
