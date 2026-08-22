"""Reference sweeps: every task threshold traces to a number printed here."""
import json, sys, time, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import numpy as np
from common import *  # noqa
import qmlf
from qmlf import QuantumClassifier, QuantumKernel, QuantumRegressor, kernel_diagnostics

R = {}
def offdiag(g): return float(g[~np.eye(len(g), dtype=bool)].mean())

# A2 projected at scale ------------------------------------------------------
Xtr, Xte, ytr, yte = data_projected_at_scale()
naive = QuantumClassifier(bandwidth=1.0, max_qubits=None).fit(Xtr, ytr)
R["A2_naive_acc"] = naive.score(Xte, yte); R["A2_naive_offdiag"] = naive.diagnose()["offdiag_mean"]
best = {}
for kern in ("fidelity", "projected"):
    c = QuantumClassifier(kernel=kern, max_qubits=None).fit(Xtr, ytr)
    best[kern] = (c.score(Xte, yte), c.diagnose()["offdiag_mean"], c.bandwidth_)
R["A2_tuned"] = best

# A3 ARD ----------------------------------------------------------------------
Xtr, Xte, ytr, yte = data_ard_noise()
from sklearn.svm import SVC
def _acc(bw):
    k = QuantumKernel(bandwidth=bw).fit(Xtr)
    return SVC(kernel="precomputed").fit(k.compute_kernel_matrix(Xtr), ytr).score(k.compute_kernel_matrix(Xte, Xtr), yte)
R["A3_scalar_grid"] = {str(b): _acc(b) for b in (0.5, 0.25, 0.1, 0.05, 0.02, 0.01)}
R["A3_scalar_best"] = max(R["A3_scalar_grid"].values())
ard = {}
for inf_bw in (0.25, 0.05):
    for ratio in (0.1, 0.02):
        bw = np.full(10, inf_bw * ratio); bw[:4] = inf_bw
        ard[f"inf{inf_bw}_ratio{ratio}"] = _acc(bw)
R["A3_ard"] = ard; R["A3_ard_best"] = max(ard.values())

# A4 entanglement ------------------------------------------------------------
Xtr, Xte, ytr, yte = data_entanglement()
ent = {}
for fm in ("zz", "z"):
    c = QuantumClassifier(feature_map=fm).fit(Xtr, ytr)
    ent[fm] = (c.score(Xte, yte), c.bandwidth_, c._kernel.feature_map.num_nonlocal_gates())
R["A4"] = ent

# B1 advantage pair ----------------------------------------------------------
(Xa, ya), (Xb, yb) = data_advantage_pair()
R["B1"] = {name: qmlf.quantum_advantage_report(X, y) for name, (X, y) in (("moons", (Xa, ya)), ("plain", (Xb, yb)))}

# B2 honesty -----------------------------------------------------------------
Xtr, Xte, ytr, yte = data_honesty()
from sklearn.svm import SVC
sq = ((Xtr[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1); gam = 1.0 / np.median(sq[~np.eye(len(Xtr), dtype=bool)])
R["B2_classical_rbf_acc"] = SVC(kernel="rbf", gamma=gam).fit(Xtr, ytr).score(Xte, yte)
R["B2_quantum_auto_acc"] = QuantumClassifier().fit(Xtr, ytr).score(Xte, yte)

# C1 budget ------------------------------------------------------------------
Xtr, Xte, ytr, yte = data_budget()
k = QuantumKernel(bandwidth=0.1).fit(Xtr)
n = len(Xtr); budget = 25000
m_max = max(m for m in range(1, n + 1) if k.circuit_budget(n, m)["nystrom_fidelity_circuits"] <= budget)
R["C1_n_train"] = n; R["C1_pairwise"] = k.circuit_budget(n)["pairwise_fidelity_circuits"]; R["C1_m_max"] = m_max
from sklearn.svm import LinearSVC
for m in (m_max, m_max // 2, 16):
    F = k.nystrom_features(Xtr, n_landmarks=m); idx = np.unique(np.linspace(0, n - 1, m).round().astype(int))
    # test features against same landmarks
    L = Xtr[idx]; W = k.compute_kernel_matrix(L); ev, evec = np.linalg.eigh((W + W.T) / 2)
    inv = np.where(ev > 1e-12, 1 / np.sqrt(np.maximum(ev, 1e-12)), 0); Ft = k.compute_kernel_matrix(Xte, L) @ (evec * inv) @ evec.T
    R[f"C1_acc_m{m}"] = LinearSVC(dual="auto").fit(F, ytr).score(Ft, yte)
R["C1_full_acc"] = SVC(kernel="precomputed").fit(k.compute_kernel_matrix(Xtr), ytr).score(k.compute_kernel_matrix(Xte, Xtr), yte)

# C2 transpile ---------------------------------------------------------------
from qiskit.circuit.library import zz_feature_map
circ = zz_feature_map(5, reps=2, entanglement="full")
opt = qmlf.create_advanced_nisq_optimizer(optimization_strength=1.0)
R["C2_transpiled"] = opt.optimize_transpile(circ); R["C2_estimate"] = opt.optimize_circuit(circ.depth(), sum(circ.count_ops().values()), circ.num_nonlocal_gates())

# D1 mitigation --------------------------------------------------------------
ideal, A, scales, obs = data_mitigation()
l1 = lambda p: float(np.abs(np.asarray(p).ravel() - ideal).sum())
R["D1_naive_l1"] = min(l1(o) for o in obs)
ro = qmlf.create_advanced_noise_mitigator(strategy="readout").fit(A)
corrected = np.vstack([ro.mitigate(o) for o in obs])
R["D1_readout_only_best_l1"] = min(l1(c) for c in corrected)
zne = qmlf.create_advanced_noise_mitigator(strategy="zne"); zne.scale_factors = scales
R["D1_pipeline_l1"] = l1(zne.mitigate(corrected))

# E1 qnn ---------------------------------------------------------------------
import torch
X, y = data_qnn()
def train(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    layer = qmlf.create_advanced_qnn_layer(n_qubits=4, reps=2, output_dim=2); layer.eval()
    opt = torch.optim.Adam(layer.parameters(), lr=0.05); lf = torch.nn.CrossEntropyLoss()
    Xt = torch.tensor(X, dtype=torch.float32); yt = torch.tensor(y)
    losses = []
    for _ in range(12):
        opt.zero_grad(); loss = lf(layer(Xt), yt); loss.backward(); opt.step(); losses.append(float(loss))
    with torch.no_grad(): pred = layer(Xt).argmax(1).numpy()
    return losses, pred
t0 = time.time(); la, pa = train(0); lb, pb = train(0)
R["E1_loss_drop_pct"] = 100 * (la[0] - la[-1]) / la[0]; R["E1_reproducible"] = bool(la == lb and np.array_equal(pa, pb)); R["E1_seconds"] = time.time() - t0

# E2 vqe ---------------------------------------------------------------------
co, ch = data_vqe(); chem = qmlf.create_advanced_chem_layer(num_atoms=3)
R["E2_exact"] = chem.compute_ground_state_energy(co, ch)
op = chem._hamiltonian_operator(co, ch); R["E2_operator_min"] = float(np.linalg.eigvalsh(op.to_matrix()).min())
from qiskit.circuit.library import real_amplitudes
z = np.zeros(real_amplitudes(num_qubits=3, reps=2).num_parameters)
R["E2_vqe_zeros"] = chem.compute_vqe_energy(co, ch, initial_point=z)
R["E2_vqe_zeros_repeat_equal"] = chem.compute_vqe_energy(co, ch, initial_point=z) == R["E2_vqe_zeros"]

# F1 industrial --------------------------------------------------------------
Xtr, Xte, ytr, yte = data_industrial()
t0 = time.time(); c = QuantumClassifier().fit(Xtr, ytr)
R["F1_auto_acc"] = c.score(Xte, yte); R["F1_auto_seconds"] = time.time() - t0; R["F1_auto_verdict"] = c.diagnose()["verdict"]
R["F1_naive_acc"] = QuantumClassifier(bandwidth=1.0).fit(Xtr, ytr).score(Xte, yte)
R["F1_classical_rbf_acc"] = SVC().fit(Xtr, ytr).score(Xte, yte)
cp = QuantumClassifier(kernel="projected").fit(Xtr, ytr); R["F1_projected_acc"] = cp.score(Xte, yte)

# F2 regression --------------------------------------------------------------
from sklearn.metrics import r2_score
Xtr, Xte, ytr, yte = data_regression()
R["F2_naive_r2"] = r2_score(yte, QuantumRegressor(bandwidth=1.0).fit(Xtr, ytr).predict(Xte))
r = QuantumRegressor().fit(Xtr, ytr); R["F2_auto_r2"] = r2_score(yte, r.predict(Xte)); R["F2_auto_bw"] = r.bandwidth_
rp = QuantumRegressor(kernel="projected").fit(Xtr, ytr); R["F2_projected_r2"] = r2_score(yte, rp.predict(Xte))

# F3 debug (correct pipeline outputs) ---------------------------------------
Xtr, Xte, ytr, yte = data_debug()
k = QuantumKernel(mode="mahalanobis", normalize="maxabs", bandwidth=0.1).fit(Xtr)
R["F3_correct_acc"] = SVC(kernel="precomputed").fit(k.compute_kernel_matrix(Xtr), ytr).score(k.compute_kernel_matrix(Xte, Xtr), yte)
kb = QuantumKernel(mode="mahalanobis", normalize="maxabs", bandwidth=0.1).fit(Xtr)
R["F3_bypass_acc"] = float(__import__("qiskit_machine_learning.algorithms", fromlist=["QSVC"]).QSVC(quantum_kernel=kb._fidelity_quantum_kernel).fit(Xtr, ytr).score(Xte, yte))

# F4 federated ---------------------------------------------------------------
cp_, counts = data_federated()
R["F4_weighted"] = (counts / counts.sum() @ cp_).tolist(); R["F4_unweighted"] = cp_.mean(0).tolist()

def _clean(o):
    if isinstance(o, dict): return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)): return o.item()
    if isinstance(o, np.ndarray): return o.tolist()
    return o
json.dump(_clean(R), open(sys.argv[1], "w"), indent=2)
print(json.dumps(_clean(R), indent=1)[:6000])
