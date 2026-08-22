# %%
"""Zero-quota probe #2: locate the attached dataset and prove the install works."""
import kaggle_benchmarks as kbench


# %%
@kbench.task(name="qmlf-install-probe", description="Locate wheels and install qmlf in-sandbox.")
def qmlf_install_probe(llm) -> dict:
    import glob
    import json
    import os
    import subprocess
    import sys
    import time

    tree = []
    for root, dirs, files in os.walk("/kaggle/input"):
        depth = root.rstrip("/").count("/") - 2
        if depth > 4:
            dirs[:] = []
            continue
        for f in files:
            tree.append(os.path.join(root, f))

    wheels = sorted(p for p in tree if p.endswith(".whl"))
    qmlf_wheel = next((p for p in wheels if os.path.basename(p).startswith("qmlf-")), None)

    steps = []
    t0 = time.time()

    def sh(label, args):
        r = subprocess.run([sys.executable, "-m", "pip"] + args,
                           capture_output=True, text=True, timeout=1800)
        steps.append({"step": label, "rc": r.returncode,
                      "tail": (r.stdout or r.stderr or "")[-400:]})
        return r.returncode

    # The kernel path needs numpy/pandas/sklearn/qiskit. torch is imported lazily
    # by qmlf and is not needed here, so --no-deps keeps an 800MB download out.
    if qmlf_wheel:
        sh("deps from wheelhouse (offline)",
           ["install", "-q", "--no-index", "--find-links", os.path.dirname(qmlf_wheel),
            "scikit-learn", "qiskit", "qiskit-machine-learning", "qiskit-algorithms"])
        sh("qmlf from wheelhouse (no-deps)",
           ["install", "-q", "--no-deps", qmlf_wheel])
    else:
        sh("deps from PyPI (fallback)",
           ["install", "-q", "scikit-learn", "qiskit>=2.4", "qiskit-machine-learning>=0.9",
            "qiskit-algorithms>=0.4"])

    install_seconds = round(time.time() - t0, 1)

    check = {}
    try:
        import numpy as np
        import qmlf
        from qmlf import QuantumKernel

        X = np.random.default_rng(0).uniform(-1.0, 1.0, size=(12, 4))
        gram = QuantumKernel(n_qubits=4, bandwidth=0.25).fit(X).compute_kernel_matrix(X)
        gram2 = QuantumKernel(n_qubits=4, bandwidth=0.25).fit(X).compute_kernel_matrix(X)

        check = {
            "qmlf_version": qmlf.__version__,
            "gram_shape": list(gram.shape),
            "gram_offdiag_mean": round(float(gram[~np.eye(12, dtype=bool)].mean()), 6),
            "bit_reproducible": gram.tobytes() == gram2.tobytes(),
            "import_ok": True,
        }
    except Exception as exc:
        check = {"import_ok": False, "error": f"{type(exc).__name__}: {exc}"}

    report = {
        "input_tree_sample": tree[:15],
        "wheels_found": len(wheels),
        "qmlf_wheel": qmlf_wheel,
        "install_seconds": install_seconds,
        "steps": steps,
        "check": check,
    }
    print(json.dumps(report, indent=2))
    return report


# %%
qmlf_install_probe.run(kbench.llm)
