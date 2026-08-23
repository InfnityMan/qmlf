"""Generate the benchmark task files. Run:  python build_tasks.py

Every task file is self-contained (Kaggle pushes one file): the shared helpers
in common.py are embedded verbatim, thresholds are stamped from
references/references.json so each number traces to a measured sweep.
"""
import json, os, textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = open(os.path.join(HERE, "common.py")).read()
R = json.load(open(os.path.join(HERE, "references", "references.json")))

HEADER = '''# %%
"""{title}

{doc}
"""
import kaggle_benchmarks as kbench
from pydantic import BaseModel, Field

# %%
{common}

# %%
'''

FOOTER = '''

# %%
{func}.run(kbench.llm)
'''

TASKS = []

PREFIX = "qmlfb-"   # uniform namespace: the collection is curated from these


def task(slug, func, title, doc, body):
    new_slug = PREFIX + slug.removeprefix("qmlf-")
    body = body.replace(f'name="{slug}"', f'name="{new_slug}"')
    TASKS.append(dict(slug=new_slug, func=func, title=title, doc=doc, body=body))

# =============================================================================
# A2  projected kernel at scale
# =============================================================================
task("qmlf-projected-kernel-at-scale", "qmlf_projected_kernel_at_scale",
"Tier A2 -- Projected kernel at scale (exponential concentration)",
f"""Thanasilp et al. 2024: fidelity kernels concentrate exponentially in qubit
count. At 10 features / 10 qubits the library's naive configuration measures
off-diagonal {R['A2_naive_offdiag']:.4f} and accuracy {R['A2_naive_acc']:.2f}
(chance). A tuned projected kernel reaches {R['A2_tuned']['projected'][0]:.2f}
with off-diagonal {R['A2_tuned']['projected'][1]:.2f}. The model must write
code that escapes concentration WITHOUT reducing dimensionality.""",
f'''
PROMPT = """\\
You are given a binary classification dataset with 10 continuous features
(90 training rows, 30 test rows). Encode ALL 10 features on 10 qubits -- do
not reduce dimensionality. Using the `qmlf` library (already installed; key
API: qmlf.QuantumClassifier with parameters kernel ('fidelity'|'projected'),
mode ('ZZ'|'mahalanobis'|'fisher', the preprocessing), feature_map ('zz'|'z'),
bandwidth ('auto'|'median'|a number), max_qubits; .fit/.predict/.diagnose()), write a Python function

    def solve(X_train, y_train, X_test):
        ...
        return {{"predictions": <array of test predictions>,
                "classifier": <the fitted qmlf.QuantumClassifier>}}

Scoring: test accuracy must be >= 0.85 AND the fitted training Gram matrix
must not be concentrated (clf.diagnose()["offdiag_mean"] >= 0.10). At this
qubit count a global fidelity kernel with default settings collapses toward
the identity; think about which kernel family and angle scale survive.
Return only one ```python code block.\\
"""


@kbench.task(name="qmlf-projected-kernel-at-scale",
             description="Escape exponential kernel concentration at 10 qubits without reducing dimensionality.")
def qmlf_projected_kernel_at_scale(llm) -> dict:
    qmlf = _ensure_qmlf()
    import numpy as np
    from sklearn.metrics import accuracy_score

    X_train, X_test, y_train, y_test = data_projected_at_scale()
    source = _extract_code(llm.prompt(PROMPT))
    result, error = _run_model_function(source, "solve", (X_train, y_train, X_test))

    acc, offdiag, n_qubits, ok_type = 0.0, 0.0, None, False
    if error is None:
        try:
            preds = np.asarray(result["predictions"]).ravel()
            clf = result["classifier"]
            ok_type = isinstance(clf, qmlf.QuantumClassifier)
            acc = float(accuracy_score(y_test, preds))
            offdiag = float(clf.diagnose()["offdiag_mean"])
            n_qubits = int(clf.n_qubits_)
        except Exception as exc:
            error = f"{{type(exc).__name__}}: {{exc}}"

    kbench.assertions.assert_true(error is None, expectation=f"Model code must run. Got: {{error}}")
    kbench.assertions.assert_true(ok_type, expectation="Must return a fitted qmlf.QuantumClassifier")
    kbench.assertions.assert_true(n_qubits == 10, expectation=f"All 10 features must be encoded (n_qubits_ == 10, no reduction). Got {{n_qubits}}")
    kbench.assertions.assert_true(offdiag >= 0.10, expectation=f"Kernel must not be concentrated: offdiag >= 0.10 (naive gives {R['A2_naive_offdiag']:.4f}). Got {{offdiag:.4f}}")
    kbench.assertions.assert_true(acc >= 0.85, expectation=f"Accuracy >= 0.85 (naive {R['A2_naive_acc']:.2f}, tuned {R['A2_tuned']['projected'][0]:.2f}). Got {{acc:.3f}}")
    return {{"accuracy": acc, "offdiag_mean": offdiag, "n_qubits": n_qubits, "error": error}}
''')

# =============================================================================
# A3  ARD noise suppression
# =============================================================================
task("qmlf-ard-noise-suppression", "qmlf_ard_noise_suppression",
"Tier A3 -- Per-feature bandwidth against hidden noise features",
f"""4 informative + 6 pure-noise features, unlabelled as such. The best
scalar bandwidth on the whole grid reaches {R['A3_scalar_best']:.3f}; an ARD
vector that keeps the informative features wide and shrinks the noise
features reaches {R['A3_ard_best']:.3f}. The model must discover which
features are noise and suppress them through the post-whitening per-feature
bandwidth -- a knob that does not exist in qiskit.""",
f'''
PROMPT = """\\
Binary classification, 10 continuous features (82 train / 28 test). Some
features carry signal and some are pure noise; you are not told which.
A quantum fidelity kernel treats every feature as an encoding angle, so noise
features destroy similarity between same-class points.

`qmlf` is installed. qmlf.QuantumKernel accepts a per-feature bandwidth
vector: QuantumKernel(bandwidth=np.array([...10 entries...])) scales each
feature's angle independently (applied after any whitening). A fitted kernel
gives Gram matrices via compute_kernel_matrix(X) and
compute_kernel_matrix(X_test, X_train) for use with
sklearn SVC(kernel="precomputed").

Write:

    def solve(X_train, y_train, X_test):
        ...
        return {{"predictions": <test predictions>,
                "bandwidth_vector": <the 10-entry per-feature bandwidth you used>}}

Scoring: test accuracy >= 0.78, AND the bandwidth you assign to the noise
features must be at most 0.30x the mean bandwidth of the informative ones.
(The best SCALAR bandwidth on this data scores well below the bar; only
per-feature scaling reaches it.)
Identify noise features from the training data (mutual information, univariate
scores, or similar). Return only one ```python code block.\\
"""


@kbench.task(name="qmlf-ard-noise-suppression",
             description="Find hidden noise features and suppress them with a per-feature quantum-kernel bandwidth.")
def qmlf_ard_noise_suppression(llm) -> dict:
    _ensure_qmlf()
    import numpy as np
    from sklearn.metrics import accuracy_score

    X_train, X_test, y_train, y_test = data_ard_noise()
    informative, noise = list(range(4)), list(range(4, 10))

    source = _extract_code(llm.prompt(PROMPT))
    result, error = _run_model_function(source, "solve", (X_train, y_train, X_test))

    acc, ratio = 0.0, float("inf")
    if error is None:
        try:
            acc = float(accuracy_score(y_test, np.asarray(result["predictions"]).ravel()))
            bw = np.asarray(result["bandwidth_vector"], dtype=float).ravel()
            if bw.shape != (10,):
                raise ValueError(f"bandwidth_vector must have 10 entries, got {{bw.shape}}")
            ratio = float(bw[noise].mean() / bw[informative].mean())
        except Exception as exc:
            error = f"{{type(exc).__name__}}: {{exc}}"

    kbench.assertions.assert_true(error is None, expectation=f"Model code must run. Got: {{error}}")
    kbench.assertions.assert_true(ratio <= 0.30, expectation=f"Noise-feature bandwidth must be <= 0.30x informative mean. Got ratio {{ratio:.3f}}")
    kbench.assertions.assert_true(acc >= 0.78, expectation=f"Accuracy >= 0.78 (best scalar {R['A3_scalar_best']:.3f}, ARD {R['A3_ard_best']:.3f}). Got {{acc:.3f}}")
    return {{"accuracy": acc, "noise_to_informative_bandwidth_ratio": ratio, "error": error}}
''')

# =============================================================================
# A4  entanglement is not free (Bowles/Schuld 2024)
# =============================================================================
task("qmlf-entanglement-is-not-free", "qmlf_entanglement_is_not_free",
"Tier A4 -- Entanglement is not free (Bowles, Ahmed, Schuld 2024)",
f"""Bowles et al. found removing entanglement often helps. On this dataset the
unentangled 'z' map scores {R['A4']['z'][0]:.2f} with 0 two-qubit gates; the
entangled 'zz' map scores {R['A4']['zz'][0]:.2f} with {R['A4']['zz'][2]}.
The model is given a hardware cost for entangling gates and must reach the
accuracy bar at minimum cost -- i.e. recognise that entanglement buys
nothing here.""",
f'''
class EncodingPlan(BaseModel):
    feature_map: str = Field(description="'zz' (entangled) or 'z' (product encoding, no entangling gates)")
    entanglement: str = Field(description="'full', 'linear', or 'circular' (ignored for 'z')")
    bandwidth: str = Field(description="'auto' for CV selection, or a positive number as a string, e.g. '0.05'")
    rationale: str = Field(description="Why this encoding reaches the bar at minimum hardware cost")


PROMPT = """\\
You are deploying a quantum-kernel classifier on hardware where every
two-qubit entangling gate is the dominant error and cost source. Dataset:
5 continuous features, binary labels, 75 train / 25 test, moderately
separable.

qmlf.QuantumClassifier(feature_map=..., entanglement=..., bandwidth=...)
supports feature_map 'zz' (ZZ feature map: pairwise entangling gates, count
grows with qubits and reps) and 'z' (product encoding: zero entangling gates).
Large benchmarking studies (Bowles, Ahmed & Schuld 2024) report that removing
entanglement often matches or improves accuracy on small tabular tasks.

Objective: test accuracy >= 0.90 using the FEWEST entangling gates. You are
scored on both. Choose the configuration.\\
"""


@kbench.task(name="qmlf-entanglement-is-not-free",
             description="Reach the accuracy bar with the minimum number of entangling gates.")
def qmlf_entanglement_is_not_free(llm) -> dict:
    qmlf = _ensure_qmlf()
    import warnings

    X_train, X_test, y_train, y_test = data_entanglement()
    plan = llm.prompt(PROMPT, schema=EncodingPlan)

    acc, gates, error = 0.0, None, None
    try:
        bw = "auto" if plan.bandwidth.strip().lower() == "auto" else float(plan.bandwidth)
        kwargs = dict(feature_map=plan.feature_map, bandwidth=bw)
        if plan.feature_map == "zz":
            kwargs["entanglement"] = plan.entanglement
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = qmlf.QuantumClassifier(**kwargs).fit(X_train, y_train)
        acc = float(clf.score(X_test, y_test))
        gates = int(clf._kernel.feature_map.num_nonlocal_gates())
    except Exception as exc:
        error = f"{{type(exc).__name__}}: {{exc}}"

    kbench.assertions.assert_true(error is None, expectation=f"Configuration must be valid. Got: {{error}}")
    kbench.assertions.assert_true(acc >= 0.90, expectation=f"Accuracy >= 0.90. Got {{acc:.3f}}")
    kbench.assertions.assert_true(gates == 0, expectation=f"The bar is reachable with zero entangling gates ('z' map scores {R['A4']['z'][0]:.2f}); used {{gates}}")
    return {{"feature_map": plan.feature_map, "accuracy": acc, "entangling_gates": gates, "error": error}}
''')

# =============================================================================
# B1  advantage screen judgment (Huang et al. 2021)
# =============================================================================
task("qmlf-advantage-screen-judgment", "qmlf_advantage_screen_judgment",
"Tier B1 -- Quantum-advantage screening judgment (Huang et al. 2021)",
f"""Two datasets. The geometric-difference screen returns
'{R['B1']['moons']['verdict'].split(':')[0]}' for one and
'{R['B1']['plain']['verdict'].split(':')[0]}' for the other. The model must
run the screen correctly, reproduce g, classify the verdict, and make the
conservative deployment call -- the decision step that precedes any circuit
in professional practice.""",
f'''
PROMPT = """\\
Before spending quantum hardware time on a dataset, a professional runs the
geometric-difference screen of Huang et al. (Nature Commun. 2021): g(K_C||K_Q)
measures whether a classical kernel can match the quantum one; model
complexities s_C, s_Q say which geometry the labels live in.

`qmlf` ships this as qmlf.quantum_advantage_report(X, y), returning a dict
with keys 'geometric_difference', 'g_matchable_below', 'g_advantage_scale',
's_classical', 's_quantum', and 'verdict' (a string beginning with one of:
'classical kernel can match', 'quantum candidate', 'inconclusive').

Write:

    def screen(X, y):
        ...
        return {{"g": <float geometric difference>,
                "verdict_category": one of "classical-matchable" | "quantum-candidate" | "inconclusive",
                "recommendation": "run-quantum" or "stay-classical"}}

Recommendation policy (conservative, as the paper advises): recommend
"run-quantum" ONLY for "quantum-candidate"; "stay-classical" otherwise.
Your function will be called on two different datasets. Return only one
```python code block.\\
"""

_CATEGORY = {{"classical kernel can match": "classical-matchable",
              "quantum candidate": "quantum-candidate",
              "inconclusive": "inconclusive"}}


@kbench.task(name="qmlf-advantage-screen-judgment",
             description="Run the Huang et al. geometric-difference screen and make the right deployment call on two datasets.")
def qmlf_advantage_screen_judgment(llm) -> dict:
    qmlf = _ensure_qmlf()
    import numpy as np

    (X_a, y_a), (X_b, y_b) = data_advantage_pair()
    source = _extract_code(llm.prompt(PROMPT))

    outcomes = {{}}
    for name, X, y in (("moons", X_a, y_a), ("plain", X_b, y_b)):
        ref = qmlf.quantum_advantage_report(X, y)
        ref_cat = _CATEGORY[ref["verdict"].split(":")[0]]
        ref_rec = "run-quantum" if ref_cat == "quantum-candidate" else "stay-classical"
        result, error = _run_model_function(source, "screen", (X, y))
        g_ok = cat_ok = rec_ok = False
        if error is None:
            try:
                g = float(result["g"])
                g_ok = abs(g - ref["geometric_difference"]) <= 0.15 * ref["geometric_difference"]
                cat_ok = result["verdict_category"] == ref_cat
                rec_ok = result["recommendation"] == ref_rec
            except Exception as exc:
                error = f"{{type(exc).__name__}}: {{exc}}"
        outcomes[name] = dict(error=error, g_ok=g_ok, cat_ok=cat_ok, rec_ok=rec_ok,
                              ref_g=ref["geometric_difference"], ref_category=ref_cat)

    for name, o in outcomes.items():
        kbench.assertions.assert_true(o["error"] is None, expectation=f"[{{name}}] model code must run. Got: {{o['error']}}")
        kbench.assertions.assert_true(o["g_ok"], expectation=f"[{{name}}] g must be within 15% of the screen's value {{o['ref_g']:.2f}}")
        kbench.assertions.assert_true(o["cat_ok"], expectation=f"[{{name}}] verdict category must be {{o['ref_category']!r}}")
        kbench.assertions.assert_true(o["rec_ok"], expectation=f"[{{name}}] recommendation must follow the conservative policy")
    return outcomes
''')

# =============================================================================
# B2  classical-baseline honesty
# =============================================================================
task("qmlf-classical-baseline-honesty", "qmlf_classical_baseline_honesty",
"Tier B2 -- Classical-baseline honesty (Bowles, Ahmed, Schuld 2024)",
f"""Standard classical models score 0.77-0.87 on this data; the auto-tuned
quantum classifier scores {R['B2_quantum_auto_acc']:.3f}. The model must build
BOTH competently, estimate their performance honestly, and declare a winner
consistent with its own estimates -- with estimates that are not inflated
against held-out reality. ~40% of QML papers claim quantum outperformance;
this task penalises hype and sandbagging alike.""",
f'''
PROMPT = """\\
Professional QML practice (Bowles, Ahmed & Schuld 2024) demands a competent
classical baseline next to every quantum model, honest performance
estimates, and a verdict that follows from them.

Dataset: 6 features, binary labels, 90 train / 30 test. `qmlf` is installed
(qmlf.QuantumClassifier() auto-tunes a quantum fidelity kernel SVM; after fit
its CV sweep is the dict .cv_results_ with lists under 'bandwidth' and
'mean_cv_accuracy'). sklearn is available for the baseline. Return predictions
as integer class labels.

Write:

    def compare(X_train, y_train, X_test):
        ...
        return {{"classical_predictions": <from a competently tuned classical model>,
                "quantum_predictions": <from a competently tuned qmlf quantum model>,
                "classical_estimate": <your honest held-out accuracy estimate for it, e.g. CV>,
                "quantum_estimate": <same for the quantum model>,
                "winner": "classical" | "quantum" | "tie"}}

Rules: "winner" must follow your own estimates ("tie" if within 2 points);
estimates must not overstate what the models achieve on unseen data. Both
models are scored: a sandbagged baseline or an untuned quantum model fails.
Return only one ```python code block.\\
"""


@kbench.task(name="qmlf-classical-baseline-honesty",
             description="Build a competent classical baseline AND a tuned quantum model; report calibrated estimates and a consistent verdict.")
def qmlf_classical_baseline_honesty(llm) -> dict:
    _ensure_qmlf()
    import numpy as np
    from sklearn.metrics import accuracy_score

    X_train, X_test, y_train, y_test = data_honesty()
    source = _extract_code(llm.prompt(PROMPT))
    result, error = _run_model_function(source, "compare", (X_train, y_train, X_test))

    c_acc = q_acc = 0.0; c_est = q_est = None; winner = None; consistent = calibrated = False
    if error is None:
        try:
            c_acc = float(accuracy_score(y_test, np.asarray(result["classical_predictions"]).ravel()))
            q_acc = float(accuracy_score(y_test, np.asarray(result["quantum_predictions"]).ravel()))
            c_est, q_est = float(result["classical_estimate"]), float(result["quantum_estimate"])
            winner = result["winner"]
            diff = c_est - q_est
            expected = "tie" if abs(diff) < 0.02 else ("classical" if diff > 0 else "quantum")
            consistent = winner == expected or (winner == "tie" and abs(diff) <= 0.04)
            calibrated = c_est <= c_acc + 0.12 and q_est <= q_acc + 0.12
        except Exception as exc:
            error = f"{{type(exc).__name__}}: {{exc}}"

    kbench.assertions.assert_true(error is None, expectation=f"Model code must run. Got: {{error}}")
    kbench.assertions.assert_true(c_acc >= 0.70, expectation=f"Classical baseline must be competent (>= 0.70; standard models score 0.77-0.87, a sandbagged one 0.50). Got {{c_acc:.3f}}")
    kbench.assertions.assert_true(q_acc >= 0.73, expectation=f"Quantum model must be tuned (>= 0.73; auto gives {R['B2_quantum_auto_acc']:.3f}). Got {{q_acc:.3f}}")
    kbench.assertions.assert_true(consistent, expectation=f"Declared winner {{winner!r}} must follow the model's own estimates (classical {{c_est}}, quantum {{q_est}})")
    kbench.assertions.assert_true(calibrated, expectation=f"Estimates must not overstate held-out reality by more than 12 points (estimates {{c_est}}/{{q_est}} vs test {{c_acc:.3f}}/{{q_acc:.3f}})")
    return {{"classical_accuracy": c_acc, "quantum_accuracy": q_acc, "classical_estimate": c_est, "quantum_estimate": q_est, "declared_winner": winner, "error": error}}
''')

# =============================================================================
# C1  hardware circuit budget (Nystrom)
# =============================================================================
task("qmlf-circuit-budget-nystrom", "qmlf_circuit_budget_nystrom",
"Tier C1 -- Hardware circuit budget via Nystrom landmarks",
f"""{R['C1_n_train']} training rows need {R['C1_pairwise']:,} pairwise
fidelity circuits; the hardware budget is 25,000. Nystrom with m landmarks
costs n*m + m(m-1)/2; the largest feasible m is {R['C1_m_max']}, giving
accuracy {R['C1_acc_m70']:.3f} (full kernel {R['C1_full_acc']:.3f}). The
model must pick m inside the budget AND keep accuracy above the bar.""",
f'''
PROMPT = """\\
You must train a quantum-kernel classifier on real hardware. Evaluating the
full Gram matrix for n training samples costs n(n-1)/2 fidelity circuits; your
hardware allocation is a HARD budget of 25,000 circuits. Dataset: 6 features,
binary labels, 320 train / 80 test.

`qmlf` is installed. A fitted qmlf.QuantumKernel exposes:
  - circuit_budget(n_samples, n_landmarks=m) -> dict with
    'pairwise_fidelity_circuits' and 'nystrom_fidelity_circuits'
  - nystrom_features(X, n_landmarks=m) -> (n, m) explicit feature map whose
    Gram approximates the kernel, costing only n*m + m(m-1)/2 circuits.

Write:

    def solve(X_train, y_train, budget=25000):
        ...
        return {{"kernel": <a qmlf.QuantumKernel fitted on X_train, with a sensible bandwidth>,
                "n_landmarks": <int m that respects the budget>}}

The grader builds Nystrom features from YOUR kernel and m on train and test
(same landmarks), trains a linear SVM, and checks: the budget is respected, and
test accuracy >= 0.58. Maximise m within the budget. Return only one
```python code block.\\
"""


@kbench.task(name="qmlf-circuit-budget-nystrom",
             description="Stay inside a hardware circuit budget with Nystrom landmarks while keeping accuracy.")
def qmlf_circuit_budget_nystrom(llm) -> dict:
    qmlf = _ensure_qmlf()
    import numpy as np
    from sklearn.svm import LinearSVC

    X_train, X_test, y_train, y_test = data_budget()
    budget = 25000
    source = _extract_code(llm.prompt(PROMPT))
    result, error = _run_model_function(source, "solve", (X_train, y_train, budget))

    acc, m, cost, ok_type = 0.0, None, None, False
    if error is None:
        try:
            kernel = result["kernel"]; m = int(result["n_landmarks"])
            ok_type = isinstance(kernel, qmlf.QuantumKernel)
            n = len(X_train)
            cost = int(kernel.circuit_budget(n, m)["nystrom_fidelity_circuits"])
            idx = np.unique(np.linspace(0, n - 1, m).round().astype(int))
            L = X_train[idx]
            W = kernel.compute_kernel_matrix(L)
            ev, evec = np.linalg.eigh((W + W.T) / 2)
            inv = np.where(ev > 1e-12, 1 / np.sqrt(np.maximum(ev, 1e-12)), 0.0)
            proj = (evec * inv) @ evec.T
            F_train = kernel.compute_kernel_matrix(X_train, L) @ proj
            F_test = kernel.compute_kernel_matrix(X_test, L) @ proj
            acc = float(LinearSVC(dual="auto").fit(F_train, y_train).score(F_test, y_test))
        except Exception as exc:
            error = f"{{type(exc).__name__}}: {{exc}}"

    kbench.assertions.assert_true(error is None, expectation=f"Model code must run. Got: {{error}}")
    kbench.assertions.assert_true(ok_type, expectation="Must return a fitted qmlf.QuantumKernel")
    kbench.assertions.assert_true(cost is not None and cost <= budget, expectation=f"Nystrom cost must be <= {{budget}} circuits (pairwise needs {R['C1_pairwise']:,}). Got {{cost}} with m={{m}}")
    kbench.assertions.assert_true(acc >= 0.58, expectation=f"Accuracy >= 0.58 (m={R['C1_m_max']} gives {R['C1_acc_m70']:.3f}). Got {{acc:.3f}}")
    return {{"n_landmarks": m, "circuits": cost, "accuracy": acc, "error": error}}
''')

# =============================================================================
# C2  NISQ transpile honesty
# =============================================================================
task("qmlf-nisq-transpile-honesty", "qmlf_nisq_transpile_honesty",
"Tier C2 -- Transpiled reality vs analytic estimate",
f"""qmlf's NISQ optimizer has an analytic ESTIMATE path and a REAL transpiler
path. For the given circuit the estimate claims depth
{R['C2_estimate']['original_depth']} -> {R['C2_estimate']['optimized_depth']};
the real transpiler gives {R['C2_transpiled']['original_depth']} ->
{R['C2_transpiled']['optimized_depth']} (no reduction). A professional
reports the measured number and labels the estimate as an estimate.""",
f'''
PROMPT = """\\
A hardware team asks for the resource requirements of a 5-qubit ZZ feature map
(reps=2, full entanglement) after optimisation. qmlf.create_advanced_nisq_optimizer(
optimization_strength=1.0) returns an object with:
  - optimize_transpile(circuit) -> REAL qiskit transpilation; dict with
    'mode'='transpiled', 'optimized_depth', 'optimized_two_qubit_gates', ...
  - optimize_circuit(depth, gate_count, two_qubit_gates) -> an analytic
    PROJECTION ('mode'='estimate'), not a measurement.

Write:

    def report(circuit):
        ...
        return {{"measured_depth": <int, from the real transpiler>,
                "measured_two_qubit_gates": <int, from the real transpiler>,
                "estimated_depth": <int, from the analytic estimate for comparison>,
                "mode_used_for_measured": <the 'mode' string of the result you report as measured>}}

The two paths disagree for this circuit. Report reality as reality. Return
only one ```python code block.\\
"""


@kbench.task(name="qmlf-nisq-transpile-honesty",
             description="Report real transpiled resources, not the analytic projection, and label each correctly.")
def qmlf_nisq_transpile_honesty(llm) -> dict:
    _ensure_qmlf()
    from qiskit.circuit.library import zz_feature_map

    circuit = zz_feature_map(5, reps=2, entanglement="full")
    source = _extract_code(llm.prompt(PROMPT))
    result, error = _run_model_function(source, "report", (circuit,))

    md = mg = ed = None; mode = None
    if error is None:
        try:
            md = int(result["measured_depth"]); mg = int(result["measured_two_qubit_gates"])
            ed = int(result["estimated_depth"]); mode = result["mode_used_for_measured"]
        except Exception as exc:
            error = f"{{type(exc).__name__}}: {{exc}}"

    T, E = {R['C2_transpiled']['optimized_depth']}, {R['C2_estimate']['optimized_depth']}
    kbench.assertions.assert_true(error is None, expectation=f"Model code must run. Got: {{error}}")
    kbench.assertions.assert_true(mode == "transpiled", expectation=f"Measured numbers must come from the real transpiler (mode 'transpiled'). Got {{mode!r}}")
    kbench.assertions.assert_true(md == T and mg == {R['C2_transpiled']['optimized_two_qubit_gates']}, expectation=f"Measured depth/2q-gates must be {{T}}/{R['C2_transpiled']['optimized_two_qubit_gates']}. Got {{md}}/{{mg}}")
    kbench.assertions.assert_true(ed == E, expectation=f"Estimated depth (analytic path) must be {{E}}. Got {{ed}}")
    return {{"measured_depth": md, "measured_two_qubit_gates": mg, "estimated_depth": ed, "mode": mode, "error": error}}
''')

# =============================================================================
# D1  mitigation pipeline
# =============================================================================
task("qmlf-mitigation-pipeline", "qmlf_mitigation_pipeline",
"Tier D1 -- Readout correction + zero-noise extrapolation pipeline",
f"""Synthetic but faithful: an ideal 4-outcome distribution passes through
depolarizing noise at scales 1/2/3, then a readout confusion matrix. Best
single raw measurement: L1 {R['D1_naive_l1']:.3f}. Readout correction alone:
{R['D1_readout_only_best_l1']:.3f}. Readout correction on every scale THEN
ZNE across scales: {R['D1_pipeline_l1']:.3f}. Traps: 1-D input to zne,
mis-cased strategy names, forgetting to fit the calibration.""",
f'''
PROMPT = """\\
You have measurements of a 4-outcome quantum circuit at three noise-scale
factors (1.0, 2.0, 3.0: the circuit was run with gate noise amplified by those
factors), and a readout calibration confusion matrix A (row k = observed
distribution when basis state k was prepared). Recover the ideal (zero-noise,
readout-corrected) distribution.

qmlf.create_advanced_noise_mitigator(strategy=...) with strategy in
{{'readout', 'depolarizing', 'zne'}} (lower-case). 'readout' needs .fit(A)
first, then .mitigate(distribution). 'zne' needs a 2-D array shaped
(n_scales, n_states) of distributions measured at increasing noise, and the
mitigator's .scale_factors attribute can be set to the actual scale factors.
Apply the stages in the physically correct order.

Write:

    def mitigate(confusion, scales, observed):
        # confusion: (4,4); scales: (3,); observed: (3,4) rows = scales
        ...
        return <length-4 probability estimate of the ideal distribution>

Scoring: L1 distance to the true ideal distribution <= 0.035 (readout-only
gets ~0.06; raw ~0.12). Return only one ```python code block.\\
"""


@kbench.task(name="qmlf-mitigation-pipeline",
             description="Chain readout correction and ZNE correctly to recover an ideal distribution.")
def qmlf_mitigation_pipeline(llm) -> dict:
    _ensure_qmlf()
    import numpy as np

    ideal, confusion, scales, observed = data_mitigation()
    source = _extract_code(llm.prompt(PROMPT))
    result, error = _run_model_function(source, "mitigate", (confusion, scales, observed))

    l1, valid = float("inf"), False
    if error is None:
        try:
            est = np.asarray(result, dtype=float).ravel()
            valid = est.shape == (4,) and np.all(est >= -1e-9) and abs(est.sum() - 1) < 1e-6
            l1 = float(np.abs(est - ideal).sum())
        except Exception as exc:
            error = f"{{type(exc).__name__}}: {{exc}}"

    kbench.assertions.assert_true(error is None, expectation=f"Model code must run. Got: {{error}}")
    kbench.assertions.assert_true(valid, expectation="Result must be a valid length-4 probability distribution")
    kbench.assertions.assert_true(l1 <= 0.035, expectation=f"L1 to ideal must be <= 0.035 (pipeline reference {R['D1_pipeline_l1']:.4f}, readout-only {R['D1_readout_only_best_l1']:.4f}). Got {{l1:.4f}}")
    return {{"l1_to_ideal": l1, "error": error}}
''')

# =============================================================================
# E1  reproducible QNN training
# =============================================================================
task("qmlf-qnn-reproducible-training", "qmlf_qnn_reproducible_training",
"Tier E1 -- Reproducible variational QNN training",
f"""Train qmlf's variational quantum layer to a loss reduction of >= 20%
(reference {R['E1_loss_drop_pct']:.1f}% in 12 Adam steps) such that two runs
with the same seed are bit-identical in both loss curve and predictions.
Traps: dropout left active at inference (no_grad is not eval), sampled
read-outs (precision > 0), unseeded torch/numpy.""",
f'''
PROMPT = """\\
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
        return {{"losses": <list of per-step training losses>,
                "predictions": <array of class predictions on X>,
                "model": <the trained torch module, ready for inference>}}

Scoring: the grader calls your function TWICE with the same seed; both loss
lists must be exactly identical and both prediction arrays identical; the
final loss must be at least 20% below the first; and the returned model must
be deterministic at inference -- two forward passes on the same input must
give identical outputs (it will be called as-is, under torch.no_grad()).
Use a modest number of steps (10-15; each step simulates circuits). Return
only one ```python code block.\\
"""


@kbench.task(name="qmlf-qnn-reproducible-training",
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
            error = f"{{type(exc).__name__}}: {{exc}}"

    kbench.assertions.assert_true(error is None, expectation=f"Model code must run (twice). Got: {{error}}")
    kbench.assertions.assert_true(same_losses, expectation="Loss curves of two same-seed runs must be identical")
    kbench.assertions.assert_true(same_preds, expectation="Predictions of two same-seed runs must be identical (dropout must be off at inference)")
    kbench.assertions.assert_true(inference_det, expectation="Returned model must be deterministic at inference: two forward passes differ, so dropout is still active (torch.no_grad() is not .eval())")
    kbench.assertions.assert_true(drop >= 20.0, expectation=f"Final loss must be >= 20% below the first (reference {R['E1_loss_drop_pct']:.1f}%). Got {{drop:.1f}}%")
    return {{"loss_drop_pct": drop, "reproducible": bool(same_losses and same_preds), "inference_deterministic": inference_det, "error": error}}
''')

# =============================================================================
# E2  VQE variational principle
# =============================================================================
task("qmlf-vqe-variational-principle", "qmlf_vqe_variational_principle",
"Tier E2 -- VQE: determinism and the variational principle",
f"""Exact diagonalisation of the model Hamiltonian gives {R['E2_exact']:.6f};
the spin-operator ground state is {R['E2_operator_min']:.6f}; a VQE from a
fixed zero initial point converges to {R['E2_vqe_zeros']:.6f} and repeats
exactly. The model must deliver both numbers, make the VQE deterministic, and
respect the variational bound.""",
f'''
PROMPT = """\\
A chemistry team needs two ground-state energies for a 3-atom geometry with
charges: the exact classical diagonalisation of qmlf's model Hamiltonian, and a
genuine VQE estimate on the corresponding spin operator.

qmlf.create_advanced_chem_layer(num_atoms=3) exposes
compute_ground_state_energy(coordinates, charges) (exact) and
compute_vqe_energy(coordinates, charges, initial_point=None) (real VQE:
real_amplitudes ansatz reps=2 + COBYLA). By default the VQE starts from a
RANDOM point, so repeated calls disagree; pass a fixed initial_point vector
(length = ansatz parameter count) to make it deterministic.

Write:

    def energies(coordinates, charges):
        ...
        return {{"exact": <float>, "vqe": <float>}}

Scoring: the grader calls your function twice. 'exact' must match the exact
value to 1e-9; the two 'vqe' values must be identical; and 'vqe' must obey the
variational principle (never below the operator's true ground state) while
being converged (within 0.3 of it). Return only one ```python code block.\\
"""


@kbench.task(name="qmlf-vqe-variational-principle",
             description="Deliver exact and deterministic VQE energies that respect the variational bound.")
def qmlf_vqe_variational_principle(llm) -> dict:
    _ensure_qmlf()
    import numpy as np

    coordinates, charges = data_vqe()
    source = _extract_code(llm.prompt(PROMPT))
    a, error = _run_model_function(source, "energies", (coordinates, charges))
    b = None
    if error is None:
        b, error = _run_model_function(source, "energies", (coordinates, charges))

    exact = vqe_a = vqe_b = None
    if error is None:
        try:
            exact, vqe_a, vqe_b = float(a["exact"]), float(a["vqe"]), float(b["vqe"])
        except Exception as exc:
            error = f"{{type(exc).__name__}}: {{exc}}"

    EXACT, OPMIN = {R['E2_exact']!r}, {R['E2_operator_min']!r}
    kbench.assertions.assert_true(error is None, expectation=f"Model code must run (twice). Got: {{error}}")
    kbench.assertions.assert_true(exact is not None and abs(exact - EXACT) < 1e-9, expectation=f"exact must equal {{EXACT:.9f}}. Got {{exact}}")
    kbench.assertions.assert_true(vqe_a is not None and vqe_a == vqe_b, expectation=f"VQE must be deterministic across calls. Got {{vqe_a}} vs {{vqe_b}}")
    kbench.assertions.assert_true(vqe_a is not None and OPMIN - 1e-6 <= vqe_a <= OPMIN + 0.3, expectation=f"VQE must lie in [{{OPMIN:.4f}}, {{OPMIN + 0.3:.4f}}] (variational bound, converged). Got {{vqe_a}}")
    return {{"exact": exact, "vqe": vqe_a, "vqe_repeat": vqe_b, "error": error}}
''')

# =============================================================================
# F1  industrial wide-data pipeline
# =============================================================================
task("qmlf-industrial-wide-data-pipeline", "qmlf_industrial_wide_data_pipeline",
"Tier F1 -- Industrial pipeline on real 30-feature tabular data",
f"""Breast-cancer diagnostics, 30 standardised features, 150 train / 50 test.
Naive quantum config: {R['F1_naive_acc']:.2f}. Auto-tuned with dimensionality
handling: {R['F1_auto_acc']:.2f} ({R['F1_auto_verdict']}). Classical RBF
reference: {R['F1_classical_rbf_acc']:.2f}. The model must configure the full
production path -- reduction, encoding, bandwidth -- and deliver a healthy,
competitive kernel.""",
f'''
class PipelinePlan(BaseModel):
    kernel: str = Field(description="'fidelity' or 'projected' or 'auto'")
    mode: str = Field(description="'ZZ', 'mahalanobis', 'fisher', or 'auto'")
    feature_map: str = Field(description="'zz', 'z', or 'auto'")
    bandwidth: str = Field(description="'auto', 'median', or a positive number as a string")
    max_qubits: int = Field(description="Qubit cap; data wider than this is PCA-reduced inside fit (exact simulation doubles per qubit)")
    normalize: str = Field(description="'default', 'maxabs', 'std', or 'none'")
    rationale: str = Field(description="Why this is a sound production configuration")


PROMPT = """\\
Production task: a diagnostics dataset with 30 standardised continuous
features (150 train / 50 test, binary). Configure qmlf.QuantumClassifier for
deployment. Relevant parameters: kernel ('fidelity'|'projected'|'auto'), mode
('ZZ'|'mahalanobis'|'fisher'|'auto'), feature_map ('zz'|'z'|'auto'),
bandwidth ('auto' = deterministic CV sweep, 'median', or a number),
max_qubits (int; features beyond it are PCA-reduced inside fit -- exact
statevector cost doubles per qubit, 30 qubits is impossible), normalize.

Scoring: test accuracy >= 0.88 AND the fitted kernel must not be
'severely concentrated' per clf.diagnose(). A competent classical RBF reaches
~0.98 here, so do not expect a quantum win -- expect a correct, healthy
pipeline. Choose the configuration.\\
"""


@kbench.task(name="qmlf-industrial-wide-data-pipeline",
             description="Configure a production quantum-kernel pipeline on real 30-feature data: reduce, encode, tune.")
def qmlf_industrial_wide_data_pipeline(llm) -> dict:
    qmlf = _ensure_qmlf()
    import warnings

    X_train, X_test, y_train, y_test = data_industrial()
    plan = llm.prompt(PROMPT, schema=PipelinePlan)

    acc, verdict, error = 0.0, None, None
    try:
        bw = plan.bandwidth.strip().lower()
        bandwidth = bw if bw in ("auto", "median") else float(bw)
        normalize = None if plan.normalize in ("default", "") else plan.normalize
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = qmlf.QuantumClassifier(kernel=plan.kernel, mode=plan.mode, feature_map=plan.feature_map,
                                         bandwidth=bandwidth, max_qubits=int(plan.max_qubits),
                                         normalize=normalize).fit(X_train, y_train)
        acc = float(clf.score(X_test, y_test)); verdict = clf.diagnose()["verdict"]
    except Exception as exc:
        error = f"{{type(exc).__name__}}: {{exc}}"

    kbench.assertions.assert_true(error is None, expectation=f"Configuration must be valid and fit within limits. Got: {{error}}")
    kbench.assertions.assert_true(verdict is not None and verdict != "severely concentrated", expectation=f"Fitted kernel must not be severely concentrated. Got {{verdict!r}}")
    kbench.assertions.assert_true(acc >= 0.88, expectation=f"Accuracy >= 0.88 (naive {R['F1_naive_acc']:.2f}, auto {R['F1_auto_acc']:.2f}). Got {{acc:.3f}}")
    return {{"accuracy": acc, "verdict": verdict, "plan": plan.model_dump(), "error": error}}
''')

# =============================================================================
# F2  regression on a physical model
# =============================================================================
task("qmlf-regression-physical-model", "qmlf_regression_physical_model",
"Tier F2 -- Quantum kernel regression of a damped oscillator",
f"""A damped-oscillator response over 8 features. Naive quantum ridge:
R^2 {R['F2_naive_r2']:.2f}. Auto-tuned fidelity kernel: {R['F2_auto_r2']:.2f}.
Projected kernel: {R['F2_projected_r2']:.2f}. The bar (0.40) is only
reachable by choosing the kernel family that generalises here.""",
f'''
class RegressionPlan(BaseModel):
    kernel: str = Field(description="'fidelity', 'projected', or 'auto'")
    mode: str = Field(description="'ZZ' or 'mahalanobis'")
    bandwidth: str = Field(description="'auto', 'median', or a positive number as a string")
    max_qubits: int = Field(description="Qubit cap (PCA-reduce above it)")
    alpha: float = Field(description="Kernel ridge regularisation")
    rationale: str = Field(description="Why this kernel family generalises on a smooth oscillatory target")


PROMPT = """\\
Regression of a smooth, damped-oscillatory physical response y from 8
continuous inputs (65 train / 25 test). Configure qmlf.QuantumRegressor
(kernel ridge on a quantum kernel). Parameters: kernel ('fidelity' |
'projected' | 'auto'), mode, bandwidth ('auto' CV over MSE, 'median', or a
number), max_qubits, alpha.

Scoring: held-out R^2 >= 0.40. On oscillatory targets the global fidelity
kernel's generalisation is poor even when tuned; the projected kernel family
(built from local reduced density matrices, Huang et al. 2021) behaves
differently. Choose the configuration.\\
"""


@kbench.task(name="qmlf-regression-physical-model",
             description="Regress a damped-oscillator response with a quantum kernel ridge that generalises.")
def qmlf_regression_physical_model(llm) -> dict:
    qmlf = _ensure_qmlf()
    import warnings
    from sklearn.metrics import r2_score

    X_train, X_test, y_train, y_test = data_regression()
    plan = llm.prompt(PROMPT, schema=RegressionPlan)

    r2, error = float("-inf"), None
    try:
        bw = plan.bandwidth.strip().lower()
        bandwidth = bw if bw in ("auto", "median") else float(bw)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reg = qmlf.QuantumRegressor(kernel=plan.kernel, mode=plan.mode, bandwidth=bandwidth,
                                        max_qubits=int(plan.max_qubits), alpha=float(plan.alpha)).fit(X_train, y_train)
        r2 = float(r2_score(y_test, reg.predict(X_test)))
    except Exception as exc:
        error = f"{{type(exc).__name__}}: {{exc}}"

    kbench.assertions.assert_true(error is None, expectation=f"Configuration must be valid. Got: {{error}}")
    kbench.assertions.assert_true(r2 >= 0.40, expectation=f"R^2 >= 0.40 (fidelity auto {R['F2_auto_r2']:.2f}, projected {R['F2_projected_r2']:.2f}). Got {{r2:.3f}}")
    return {{"r2": r2, "plan": plan.model_dump(), "error": error}}
''')

# =============================================================================
# F3  debug a broken pipeline (QBugLM gap)
# =============================================================================
task("qmlf-debug-broken-pipeline", "qmlf_debug_broken_pipeline",
"Tier F3 -- Repair a broken quantum pipeline (four planted bugs)",
f"""A script with four literature-documented bugs: n_qubits disagrees with
the data, the raw fidelity kernel is handed to QSVC (silently discarding
whitening and bandwidth: accuracy {R['F3_bypass_acc']:.2f} instead of
{R['F3_correct_acc']:.2f}), a mis-cased mitigation strategy, and a 1-D input to
ZNE. The model must return a working version.""",
f'''
BROKEN = \'\'\'
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
    return {{"predictions": predictions, "mitigated": mitigated}}
\'\'\'

PROMPT = """\\
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
X_test, observed) in one ```python code block.\\
"""


@kbench.task(name="qmlf-debug-broken-pipeline",
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
            error = f"{{type(exc).__name__}}: {{exc}}"

    kbench.assertions.assert_true(not bypass, expectation="Must not hand .fidelity_quantum_kernel to QSVC: it bypasses whitening, normalize and bandwidth (one of the planted bugs)")
    kbench.assertions.assert_true(error is None, expectation=f"Repaired code must run. Got: {{error}}")
    kbench.assertions.assert_true(acc >= 0.90, expectation=f"Accuracy >= 0.90 (correct pipeline {R['F3_correct_acc']:.2f}; the QSVC bypass gives {R['F3_bypass_acc']:.2f}). Got {{acc:.3f}}")
    kbench.assertions.assert_true(mit_ok, expectation="mitigated must be a valid length-4 distribution from ZNE over all three scales")
    return {{"accuracy": acc, "mitigated_ok": mit_ok, "error": error}}
''')

# =============================================================================
# F4  federated partial participation
# =============================================================================
task("qmlf-federated-partial-participation", "qmlf_federated_partial_participation",
"Tier F4 -- Federated QML round with partial participation",
"""Five clients were expected, three reported, with very different sample
counts. The correct FedAvg update is the sample-weighted mean; the unweighted
mean is the classic mistake. The model must produce the exact weighted
aggregate.""",
f'''
PROMPT = """\\
A federated quantum-ML round: 5 clients were expected but only 3 returned
parameter vectors (shape (3, 6)), trained on 120, 40 and 240 local samples
respectively. Produce the FedAvg global update.

qmlf.create_federated_qml(num_clients=...) returns an aggregator with
.aggregate(client_params_list, client_weights=None) (sample-size-weighted mean
when weights are given; it warns, by design, when fewer clients than
num_clients report).

Write:

    def aggregate(client_params, sample_counts):
        ...
        return <length-6 global parameter vector>

Return only one ```python code block.\\
"""


@kbench.task(name="qmlf-federated-partial-participation",
             description="Produce the exact sample-weighted FedAvg update under partial client participation.")
def qmlf_federated_partial_participation(llm) -> dict:
    _ensure_qmlf()
    import numpy as np

    client_params, sample_counts = data_federated()
    expected = np.asarray({R['F4_weighted']!r})
    unweighted = np.asarray({R['F4_unweighted']!r})
    source = _extract_code(llm.prompt(PROMPT))
    result, error = _run_model_function(source, "aggregate", (client_params, sample_counts))

    ok, is_unweighted = False, False
    if error is None:
        try:
            v = np.asarray(result, dtype=float).ravel()
            ok = v.shape == (6,) and np.allclose(v, expected, atol=1e-9)
            is_unweighted = v.shape == (6,) and np.allclose(v, unweighted, atol=1e-9)
        except Exception as exc:
            error = f"{{type(exc).__name__}}: {{exc}}"

    kbench.assertions.assert_true(error is None, expectation=f"Model code must run. Got: {{error}}")
    kbench.assertions.assert_true(not is_unweighted, expectation="Unweighted mean is the classic FedAvg mistake; weight by sample count")
    kbench.assertions.assert_true(ok, expectation="Global vector must equal the sample-weighted mean to 1e-9")
    return {{"correct": ok, "unweighted_mistake": is_unweighted, "error": error}}
''')


def main():
    out_dir = os.path.join(HERE, "tasks")
    os.makedirs(out_dir, exist_ok=True)
    for t in TASKS:
        text = HEADER.format(title=t["title"], doc=t["doc"], common=COMMON) + t["body"] + FOOTER.format(func=t["func"])
        # Every task's return value goes through the JSON sanitiser.
        # Only the LAST such line is the task function's return (string
        # literals holding example code can contain earlier ones).
        import re
        matches = list(re.finditer(r"^(    return )(\{.*\}|outcomes)$", text, flags=re.MULTILINE))
        assert matches, t["slug"]
        m = matches[-1]
        text = text[:m.start()] + m.group(1) + "_jsonable(" + m.group(2) + ")" + text[m.end():]
        path = os.path.join(out_dir, t["slug"] + ".py")
        open(path, "w").write(text)
        compile(text, path, "exec")
        print("wrote", os.path.relpath(path, HERE))
    print(f"{len(TASKS)} tasks generated")


if __name__ == "__main__":
    main()
