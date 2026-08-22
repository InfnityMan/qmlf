# %%
"""Tier F3 -- Repair a broken quantum pipeline (four planted bugs)

A script with four literature-documented bugs: n_qubits disagrees with
the data, the raw fidelity kernel is handed to QSVC (silently discarding
whitening and bandwidth: accuracy 0.45 instead of
0.95), a mis-cased mitigation strategy, and a 1-D input to
ZNE. The model must return a working version.
"""
import kaggle_benchmarks as kbench
from pydantic import BaseModel, Field

# %%
# ---- qmlf benchmark: shared helpers (embedded verbatim into every task) ----
import os
import sys


def _ensure_qmlf(with_torch=False):
    """Install qmlf + runtime deps in the Kaggle Benchmarks sandbox.

    Measured facts (kaggle/SANDBOX.md): Python 3.11, internet on, only
    numpy/pandas preinstalled. Wheelhouse dataset lands at
    /kaggle/input/datasets/<owner>/<slug>/ -> --no-index install in ~15s.
    Fallback: the public repo (git+https), verified to serve qmlf 1.4.0.
    torch CPU from the PyTorch index measured at 33s.
    """
    import subprocess

    def pip(*args):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])

    try:
        import qmlf  # noqa: F401
        have_qmlf = True
    except ImportError:
        have_qmlf = False

    if not have_qmlf:
        wheel = None
        for root, _dirs, files in os.walk("/kaggle/input"):
            for name in files:
                if name.startswith("qmlf-") and name.endswith(".whl"):
                    wheel = os.path.join(root, name)
        if wheel:
            pip("--no-index", "--find-links", os.path.dirname(wheel),
                "scikit-learn", "qiskit", "qiskit-machine-learning", "qiskit-algorithms")
            pip("--no-deps", wheel)
        else:
            pip("scikit-learn", "qiskit>=2.3,<3", "qiskit-machine-learning>=0.9,<1",
                "qiskit-algorithms>=0.4,<1")
            pip("--no-deps", "git+https://github.com/InfnityMan/qmlf")

    if with_torch:
        try:
            import torch  # noqa: F401
        except ImportError:
            pip("torch", "--index-url", "https://download.pytorch.org/whl/cpu")

    import qmlf
    return qmlf


def _extract_code(text):
    """First ```python fence, else first ``` fence, else the whole text."""
    import re

    for pattern in (r"```python\s*\n(.*?)```", r"```\s*\n(.*?)```"):
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1)

    return text


def _run_model_function(source, entry, args, timeout=900):
    """exec the model's source, fetch `entry`, call it under a wall-clock
    limit. Returns (result, error_string_or_None). Never raises."""
    import signal
    import traceback

    def _alarm(signum, frame):
        raise TimeoutError(f"model code exceeded {timeout}s")

    namespace = {"__name__": "__model__"}

    try:
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(timeout)
        exec(compile(source, "<model_code>", "exec"), namespace)
        function = namespace.get(entry)
        if function is None:
            raise NameError(f"model code must define a function named {entry!r}")
        return function(*args), None
    except BaseException as exc:  # noqa: BLE001 - grading must survive anything
        tail = traceback.format_exc().strip().splitlines()[-1]
        return None, f"{type(exc).__name__}: {exc} | {tail}"[:600]
    finally:
        signal.alarm(0)


def _jsonable(obj):
    """Recursively convert numpy scalars/arrays/bools to plain Python.

    kbench serialises the task's return value to JSON; a numpy.bool_ (which
    json.dumps rejects with "Object of type bool is not JSON serializable")
    silently produces 'notebook completed but no run output'. Measured, not
    guessed: two tasks died exactly this way."""
    import numpy as np

    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


# ---- deterministic datasets ------------------------------------------------

def _split(X, y, test_size=0.25, seed=7, stratify=True):
    from sklearn.model_selection import train_test_split

    return train_test_split(X, y, test_size=test_size, random_state=seed,
                            stratify=y if stratify else None)


def data_projected_at_scale():
    """10 features, no reduction: the regime where fidelity kernels die."""
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=120, n_features=10, n_informative=6,
                               n_redundant=0, class_sep=1.5, random_state=11)
    return _split(X, y)


def data_ard_noise():
    """4 informative + 6 pure-noise features; which is which is NOT told.

    Built explicitly (class-conditional shifts on every informative feature,
    one mild interaction) so that each informative feature is individually
    detectable by an honest univariate analysis -- make_classification's
    'informative' features are often only jointly informative, which would
    make the hidden-noise criterion unfair."""
    import numpy as np

    rng = np.random.default_rng(33)
    y = rng.integers(0, 2, 110)
    X_inf = rng.normal(size=(110, 4)) + np.outer(y, np.array([1.1, 0.9, 1.0, 1.2]))
    X_inf[:, 1] = X_inf[:, 1] * (1 + 0.3 * X_inf[:, 0])
    noise = rng.normal(scale=2.0, size=(110, 6))
    X = np.hstack([X_inf, noise])
    return _split(X, y)


def data_entanglement():
    """A dataset on which the unentangled 'z' map matches the 'zz' map."""
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=100, n_features=5, n_informative=3,
                               n_redundant=0, class_sep=1.3, random_state=5)
    return _split(X, y)


def data_advantage_pair():
    """Two datasets for the advantage screen: a geometric one and a plain one."""
    import numpy as np
    from sklearn.datasets import make_classification, make_moons

    X_a, y_a = make_moons(n_samples=80, noise=0.12, random_state=3)
    X_b, y_b = make_classification(n_samples=80, n_features=4, n_informative=2,
                                   n_redundant=0, class_sep=2.5, random_state=8)
    X_b = X_b[:, :4]
    return (X_a, y_a), (X_b, y_b)


def data_honesty():
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=120, n_features=6, n_informative=4,
                               n_redundant=0, class_sep=1.0, random_state=13)
    return _split(X, y)


def data_budget():
    """400 samples: pairwise fidelity needs 79,800 circuits; budget is 25,000."""
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=400, n_features=6, n_informative=4,
                               n_redundant=0, class_sep=1.4, random_state=17)
    return _split(X, y, test_size=0.2)


def data_mitigation():
    """Ideal 4-state distribution, a readout confusion matrix, and noisy
    measurements at three depolarizing noise scales (noise then readout)."""
    import numpy as np

    ideal = np.array([0.55, 0.25, 0.15, 0.05])
    confusion = np.array([
        [0.92, 0.04, 0.03, 0.01],
        [0.05, 0.90, 0.02, 0.03],
        [0.03, 0.02, 0.91, 0.04],
        [0.02, 0.04, 0.04, 0.90],
    ])
    base_p = 0.12
    scales = np.array([1.0, 2.0, 3.0])
    uniform = np.full(4, 0.25)
    observed = []
    for scale in scales:
        p = min(base_p * scale, 0.95)
        depolarized = (1 - p) * ideal + p * uniform
        observed.append(depolarized @ confusion)  # row-stochastic readout
    return ideal, confusion, scales, np.array(observed)


def data_qnn():
    import numpy as np
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=24, n_features=4, n_informative=3,
                               n_redundant=0, class_sep=1.8, random_state=9)
    return np.asarray(X, dtype=float), np.asarray(y)


def data_vqe():
    import numpy as np

    coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74], [0.0, 0.74, 0.0]])
    charges = np.array([1.0, 1.0, 6.0])
    return coordinates, charges


def data_industrial():
    """Breast-cancer: 30 real features, 200 rows (stratified), standardised."""
    import numpy as np
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X, y = load_breast_cancer(return_X_y=True)
    X_sub, _, y_sub, _ = train_test_split(X, y, train_size=200, random_state=23, stratify=y)
    X_train, X_test, y_train, y_test = train_test_split(
        X_sub, y_sub, test_size=0.25, random_state=23, stratify=y_sub)
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test), y_train, y_test


def data_regression():
    """Damped-oscillator style target over 8 features; wide enough to force
    reduction, smooth enough for a tuned quantum kernel ridge to fit."""
    import numpy as np

    rng = np.random.default_rng(31)
    X = rng.uniform(-1.0, 1.0, size=(90, 8))
    t = X[:, 0] + 0.5 * X[:, 1]
    y = np.exp(-0.8 * np.abs(t)) * np.cos(3.0 * t) + 0.3 * X[:, 2] ** 2
    y = y + 0.02 * rng.normal(size=90)
    return X[:65], X[65:], y[:65], y[65:]


def data_federated():
    import numpy as np

    rng = np.random.default_rng(41)
    client_params = rng.normal(size=(3, 6))   # 3 of 5 clients reported
    sample_counts = np.array([120, 40, 240])
    return client_params, sample_counts


def data_debug():
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=80, n_features=5, n_informative=3,
                               n_redundant=0, class_sep=1.4, random_state=29)
    return _split(X, y)


# %%

BROKEN = '''
import numpy as np
from qiskit_machine_learning.algorithms import QSVC
import qmlf

def run_pipeline(X_train, y_train, X_test, observed):
    # X has 5 features
    kernel = qmlf.QuantumKernel(n_qubits=8, mode="mahalanobis", normalize="maxabs", bandwidth=0.1)
    kernel.fit(X_train)
    clf = QSVC(quantum_kernel=kernel.fidelity_quantum_kernel)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    # observed: distributions measured at noise scales 1, 2, 3 -> shape (3, 4)
    mitigator = qmlf.create_advanced_noise_mitigator(strategy="ZNE")
    mitigated = mitigator.mitigate(observed[0])
    return {"predictions": predictions, "mitigated": mitigated}
'''

PROMPT = """\
The following qmlf pipeline is broken in several places (it may raise, and
where it does not raise it silently computes the wrong thing). Fix it.
Preserve the intent: a whitened ('mahalanobis') quantum kernel with
normalize='maxabs' and bandwidth=0.1 classifying 5-feature data, and a
zero-noise extrapolation across the three measured noise scales returning a
length-4 distribution. Hints about qmlf: the feature map encodes one angle per
feature; preprocessing lives in compute_kernel_matrix (use
compute_kernel_matrix(X) and compute_kernel_matrix(X_test, X_train) with
sklearn SVC(kernel='precomputed')); strategy names are lower-case; zne needs
the full (n_scales, n_states) stack.

""" + BROKEN + """

Return the complete corrected code defining run_pipeline(X_train, y_train,
X_test, observed) in one ```python code block.\
"""


@kbench.task(name="qmlfb-debug-broken-pipeline",
             description="Find and fix four planted bugs in a quantum kernel + mitigation pipeline.")
def qmlf_debug_broken_pipeline(llm) -> dict:
    _ensure_qmlf()
    import numpy as np
    from sklearn.metrics import accuracy_score

    X_train, X_test, y_train, y_test = data_debug()
    _ideal, _A, _scales, observed = data_mitigation()
    source = _extract_code(llm.prompt(PROMPT))

    # Static fast-fail: handing the raw fidelity kernel to QSVC is one of the
    # planted bugs (it silently discards whitening/normalize/bandwidth), and
    # the pairwise sampler path it triggers is the slow route through the
    # sandbox. Grading it by inspection is both correct and cheap.
    bypass = "fidelity_quantum_kernel" in source or "QSVC(" in source
    if bypass:
        result, error = None, "bypass: code still hands the raw fidelity kernel to QSVC"
    else:
        result, error = _run_model_function(source, "run_pipeline", (X_train, y_train, X_test, observed), timeout=600)

    acc, mit_ok = 0.0, False
    if error is None:
        try:
            acc = float(accuracy_score(y_test, np.asarray(result["predictions"]).ravel()))
            m = np.asarray(result["mitigated"], dtype=float).ravel()
            mit_ok = m.shape == (4,) and abs(m.sum() - 1) < 1e-6 and np.all(m >= -1e-9)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

    kbench.assertions.assert_true(not bypass, expectation="Must not hand .fidelity_quantum_kernel to QSVC: it bypasses whitening, normalize and bandwidth (one of the planted bugs)")
    kbench.assertions.assert_true(error is None, expectation=f"Repaired code must run. Got: {error}")
    kbench.assertions.assert_true(acc >= 0.90, expectation=f"Accuracy >= 0.90 (correct pipeline 0.95; the QSVC bypass gives 0.45). Got {acc:.3f}")
    kbench.assertions.assert_true(mit_ok, expectation="mitigated must be a valid length-4 distribution from ZNE over all three scales")
    return _jsonable({"accuracy": acc, "mitigated_ok": mit_ok, "error": error})


# %%
qmlf_debug_broken_pipeline.run(kbench.llm)
