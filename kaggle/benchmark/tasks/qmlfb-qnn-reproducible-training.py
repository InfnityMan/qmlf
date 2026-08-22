# %%
"""Tier E1 -- Reproducible variational QNN training

Train qmlf's variational quantum layer to a loss reduction of >= 20%
(reference 36.6% in 12 Adam steps) such that two runs
with the same seed are bit-identical in both loss curve and predictions.
Traps: dropout left active at inference (no_grad is not eval), sampled
read-outs (precision > 0), unseeded torch/numpy.
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

PROMPT = """\
Train a hybrid quantum-classical classifier reproducibly. Dataset: 24 samples,
4 features, binary labels. torch and qmlf are installed.

qmlf.create_advanced_qnn_layer(n_qubits=4, reps=2, output_dim=2) returns a
torch.nn.Module whose forward pass runs a 4-qubit variational circuit with
exact statevector read-out by default (precision=0.0) followed by a small
classical head containing Dropout. Its quantum weights and classical weights
are both trainable torch parameters.

Write:

    def train_and_predict(X, y, seed):
        ...
        return {"losses": <list of per-step training losses>,
                "predictions": <array of class predictions on X>,
                "model": <the trained torch module, ready for inference>}

Scoring: the grader calls your function TWICE with the same seed; both loss
lists must be exactly identical and both prediction arrays identical; the
final loss must be at least 20% below the first; and the returned model must
be deterministic at inference -- two forward passes on the same input must
give identical outputs (it will be called as-is, under torch.no_grad()).
Use a modest number of steps (10-15; each step simulates circuits). Return
only one ```python code block.\
"""


@kbench.task(name="qmlfb-qnn-reproducible-training",
             description="Train a variational QNN layer so two seeded runs are bit-identical and the loss falls >= 20%.")
def qmlf_qnn_reproducible_training(llm) -> dict:
    _ensure_qmlf(with_torch=True)
    import numpy as np

    X, y = data_qnn()
    source = _extract_code(llm.prompt(PROMPT))
    a, error = _run_model_function(source, "train_and_predict", (X, y, 0), timeout=1200)
    b = None
    if error is None:
        b, error = _run_model_function(source, "train_and_predict", (X, y, 0), timeout=1200)

    drop, same_losses, same_preds, inference_det = 0.0, False, False, False
    if error is None:
        try:
            import torch
            la, lb = [float(v) for v in a["losses"]], [float(v) for v in b["losses"]]
            pa, pb = np.asarray(a["predictions"]).ravel(), np.asarray(b["predictions"]).ravel()
            same_losses = la == lb and len(la) >= 2
            same_preds = pa.shape == pb.shape and np.array_equal(pa, pb)
            drop = 100.0 * (la[0] - la[-1]) / la[0] if la and la[0] != 0 else 0.0
            model = a["model"]
            Xt = torch.tensor(X, dtype=torch.float32)
            with torch.no_grad():
                out1, out2 = model(Xt), model(Xt)   # as returned: dropout left on shows here
            inference_det = bool(torch.equal(out1, out2))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

    kbench.assertions.assert_true(error is None, expectation=f"Model code must run (twice). Got: {error}")
    kbench.assertions.assert_true(same_losses, expectation="Loss curves of two same-seed runs must be identical")
    kbench.assertions.assert_true(same_preds, expectation="Predictions of two same-seed runs must be identical (dropout must be off at inference)")
    kbench.assertions.assert_true(inference_det, expectation="Returned model must be deterministic at inference: two forward passes differ, so dropout is still active (torch.no_grad() is not .eval())")
    kbench.assertions.assert_true(drop >= 20.0, expectation=f"Final loss must be >= 20% below the first (reference 36.6%). Got {drop:.1f}%")
    return _jsonable({"loss_drop_pct": drop, "reproducible": bool(same_losses and same_preds), "inference_deterministic": inference_det, "error": error})


# %%
qmlf_qnn_reproducible_training.run(kbench.llm)
