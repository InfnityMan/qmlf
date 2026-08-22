# %%
"""Tier E -- VQE: determinism and the variational principle

exact 32.265209; operator ground state -29.565827; VQE from zeros -29.565388.
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


# ---- parametric makers (suite families) ------------------------------------

def make_dataset(kind, n_samples=100, n_features=6, seed=0, class_sep=1.4,
                 n_informative=None, noise_features=0, noise_scale=2.0):
    """Deterministic classification data. Returns X_train, X_test, y_train, y_test.

    kinds: 'blobs' (make_classification), 'moons', 'circles', 'xor',
    'shift' (explicit class-conditional shifts, each feature individually
    informative), 'quantiles' (gaussian_quantiles, concentric).
    """
    import numpy as np
    from sklearn import datasets as skd

    rng = np.random.default_rng(seed)
    if kind == "blobs":
        inf = n_informative or max(2, n_features // 2)
        X, y = skd.make_classification(n_samples=n_samples, n_features=n_features,
                                       n_informative=inf, n_redundant=0,
                                       class_sep=class_sep, random_state=seed)
    elif kind == "moons":
        X2, y = skd.make_moons(n_samples=n_samples, noise=0.12, random_state=seed)
        X = np.hstack([X2, rng.normal(scale=0.3, size=(n_samples, max(0, n_features - 2)))])
    elif kind == "circles":
        X2, y = skd.make_circles(n_samples=n_samples, noise=0.08, factor=0.45, random_state=seed)
        X = np.hstack([X2, rng.normal(scale=0.3, size=(n_samples, max(0, n_features - 2)))])
    elif kind == "xor":
        X = rng.uniform(-1, 1, size=(n_samples, n_features))
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    elif kind == "shift":
        y = rng.integers(0, 2, n_samples)
        inf = n_informative or n_features
        shift = 0.9 + 0.3 * rng.random(inf)
        X = rng.normal(size=(n_samples, inf)) + np.outer(y, shift)
        if inf > 1:
            X[:, 1] = X[:, 1] * (1 + 0.3 * X[:, 0])
        if n_features > inf:
            X = np.hstack([X, rng.normal(scale=noise_scale, size=(n_samples, n_features - inf))])
    elif kind == "quantiles":
        X, y = skd.make_gaussian_quantiles(n_samples=n_samples, n_features=n_features,
                                           n_classes=2, random_state=seed)
    else:
        raise ValueError(kind)

    if noise_features:
        X = np.hstack([X, rng.normal(scale=noise_scale, size=(n_samples, noise_features))])

    return _split(np.asarray(X, dtype=float), np.asarray(y), test_size=0.25, seed=7)


def make_regression(kind, n_samples=90, n_features=8, seed=31):
    import numpy as np

    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n_samples, n_features))
    if kind == "oscillator":
        t = X[:, 0] + 0.5 * X[:, 1]
        y = np.exp(-0.8 * np.abs(t)) * np.cos(3.0 * t) + 0.3 * X[:, 2] ** 2
    elif kind == "sinc":
        r = 3.0 * np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2) + 1e-9
        y = np.sin(r) / r + 0.2 * X[:, 2]
    elif kind == "friedman":
        y = (np.sin(np.pi * X[:, 0] * X[:, 1]) + 2 * (X[:, 2] - 0.5) ** 2
             + X[:, 3] + 0.5 * X[:, 4])
    elif kind == "quadratic":
        y = X[:, 0] ** 2 - 0.5 * X[:, 1] * X[:, 2] + 0.3 * X[:, 3]
    else:
        raise ValueError(kind)
    y = y + 0.02 * rng.normal(size=n_samples)
    n_train = int(0.72 * n_samples)
    return X[:n_train], X[n_train:], y[:n_train], y[n_train:]


def make_mitigation(seed=0, n_states=4, base_p=0.12, readout_err=0.08,
                    scales=(1.0, 2.0, 3.0)):
    """Ideal distribution -> depolarizing at each scale -> readout confusion."""
    import numpy as np

    rng = np.random.default_rng(seed)
    ideal = rng.dirichlet(np.ones(n_states) * 0.8)
    confusion = np.eye(n_states) * (1 - readout_err)
    off = rng.random((n_states, n_states)); np.fill_diagonal(off, 0)
    confusion += readout_err * off / off.sum(axis=1, keepdims=True)
    scales = np.asarray(scales, dtype=float)
    uniform = np.full(n_states, 1.0 / n_states)
    observed = []
    for scale in scales:
        p = min(base_p * scale, 0.95)
        observed.append(((1 - p) * ideal + p * uniform) @ confusion)
    return ideal, confusion, scales, np.array(observed)


def make_geometry(seed=0, n_atoms=3):
    import numpy as np

    rng = np.random.default_rng(seed)
    coordinates = rng.uniform(-0.9, 0.9, size=(n_atoms, 3))
    charges = rng.choice([1.0, 1.0, 6.0, 7.0, 8.0], size=n_atoms)
    return coordinates, charges


def make_federated(seed=0, n_clients=5, n_reporting=3, dim=6):
    import numpy as np

    rng = np.random.default_rng(seed)
    params = rng.normal(size=(n_reporting, dim))
    counts = rng.integers(20, 300, size=n_reporting)
    return params, counts, n_clients


def make_real(name):
    """Standardised real tabular data subsets (sklearn bundled)."""
    import numpy as np
    from sklearn import datasets as skd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    if name == "breast_cancer":
        X, y = skd.load_breast_cancer(return_X_y=True); n = 200
    elif name == "wine":
        X, y = skd.load_wine(return_X_y=True); n = 150
    elif name == "iris":
        X, y = skd.load_iris(return_X_y=True); n = 150
    elif name == "digits_3v8":
        X, y = skd.load_digits(return_X_y=True)
        keep = (y == 3) | (y == 8); X, y = X[keep], (y[keep] == 8).astype(int); n = 200
    else:
        raise ValueError(name)
    if n < len(X):
        X, _, y, _ = train_test_split(X, y, train_size=n, random_state=23, stratify=y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=23, stratify=y)
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test), y_train, y_test


# %%
REF = {'exact': 32.265208909871866, 'operator_min': -29.56582746415843, 'vqe_zeros': -29.565387685055786, 'keep': True}
PARAMS = {'n_atoms': 2}



PROMPT = """\
Two ground-state energies for a 2-atom geometry with charges: the exact diagonalisation of qmlf's model Hamiltonian and a genuine VQE estimate on the corresponding spin operator. qmlf.create_advanced_chem_layer(num_atoms=2) exposes compute_ground_state_energy(coordinates, charges) and compute_vqe_energy(coordinates, charges, initial_point=None) (real_amplitudes reps=2 + COBYLA; a RANDOM start by default, so pass a fixed initial_point vector of length = ansatz parameter count for determinism). Write
    def energies(coordinates, charges):
        return {"exact": <float>, "vqe": <float>}
Scoring: called twice; 'exact' to 1e-9; the two 'vqe' values identical; 'vqe' never below the operator's true ground state and within 0.3 of it. Return only one ```python code block.\
"""


def score(source, data, ref, params):
    coordinates, charges = data
    a, error = _run_model_function(source, "energies", (coordinates, charges)); b = None
    if error is None: b, error = _run_model_function(source, "energies", (coordinates, charges))
    exact = va = vb = None
    if error is None:
        try: exact, va, vb = float(a["exact"]), float(a["vqe"]), float(b["vqe"])
        except Exception as exc: error = f"{type(exc).__name__}: {exc}"
    E, M = ref["exact"], ref["operator_min"]
    checks = [(error is None, f"Model code must run (twice). Got: {error}"),
              (exact is not None and abs(exact - E) < 1e-9, f"exact must equal {E:.9f}. Got {exact}"),
              (va is not None and va == vb, f"VQE must be deterministic across calls. Got {va} vs {vb}"),
              (va is not None and M - 1e-6 <= va <= M + 0.3, f"VQE must lie in [{M:.4f}, {M + 0.3:.4f}] (variational bound, converged). Got {va}")]
    return {"exact": exact, "vqe": va, "vqe_repeat": vb, "error": error}, checks



@kbench.task(name="qmlfb-vqe-s4n2", description='Deliver exact and deterministic VQE energies that respect the variational bound.')
def qmlfb_vqe_s4n2(llm) -> dict:
    _ensure_qmlf()
    data = make_geometry(seed=4, n_atoms=2)
    answer = llm.prompt(PROMPT)
    if True:
        answer = _extract_code(answer)
    metrics, checks = score(answer, data, REF, PARAMS)
    for ok, expectation in checks:
        kbench.assertions.assert_true(bool(ok), expectation=expectation)
    return _jsonable(metrics)


# %%
qmlfb_vqe_s4n2.run(kbench.llm)
