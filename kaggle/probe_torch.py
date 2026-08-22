# %%
"""Zero-quota probe: can the sandbox get torch (CPU) fast enough for QNN tasks?"""
import kaggle_benchmarks as kbench


# %%
@kbench.task(name="qmlf-torch-probe", description="Time a CPU torch install and a QNN forward pass.")
def qmlf_torch_probe(llm) -> dict:
    import json, os, subprocess, sys, time

    def pip(*args, timeout=1500):
        t0 = time.time()
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args],
                           capture_output=True, text=True, timeout=timeout)
        return {"rc": r.returncode, "seconds": round(time.time() - t0, 1),
                "tail": (r.stdout or r.stderr or "")[-300:]}

    wheel = None
    for root, _d, files in os.walk("/kaggle/input"):
        for f in files:
            if f.startswith("qmlf-") and f.endswith(".whl"):
                wheel = os.path.join(root, f)
    steps = {}
    steps["deps"] = pip("--no-index", "--find-links", os.path.dirname(wheel),
                        "scikit-learn", "qiskit", "qiskit-machine-learning", "qiskit-algorithms")
    steps["qmlf"] = pip("--no-deps", wheel)
    steps["torch_cpu"] = pip("torch", "--index-url", "https://download.pytorch.org/whl/cpu")

    check = {}
    try:
        import numpy as np, torch, qmlf
        t0 = time.time()
        layer = qmlf.create_advanced_qnn_layer(n_qubits=4, reps=2, output_dim=2)
        layer.eval()
        x = torch.tensor(np.random.default_rng(0).uniform(-1, 1, (8, 4)), dtype=torch.float32)
        with torch.no_grad():
            a, b = layer(x), layer(x)
        check = {"torch": torch.__version__, "forward_seconds": round(time.time() - t0, 2),
                 "deterministic": bool(torch.equal(a, b)), "ok": True}
    except Exception as exc:
        check = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    report = {"steps": steps, "check": check}
    print(json.dumps(report, indent=2))
    return report


# %%
qmlf_torch_probe.run(kbench.llm)
