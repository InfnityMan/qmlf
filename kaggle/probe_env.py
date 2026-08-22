# %%
"""Zero-quota probe: report what the benchmark sandbox actually provides.

Never calls llm.prompt, so this costs no model credits.
"""
import kaggle_benchmarks as kbench


# %%
@kbench.task(name="qmlf-env-probe", description="Report sandbox python, packages, internet.")
def qmlf_env_probe(llm) -> dict:
    import glob
    import importlib
    import json
    import os
    import subprocess
    import sys

    present = {}
    for mod in ("numpy", "pandas", "sklearn", "scipy", "torch", "xgboost",
                "plotly", "qiskit", "qiskit_machine_learning", "qiskit_algorithms", "qmlf"):
        try:
            m = importlib.import_module(mod)
            present[mod] = getattr(m, "__version__", "unknown")
        except Exception as exc:
            present[mod] = f"MISSING ({type(exc).__name__})"

    # Is the attached dataset visible, and where?
    inputs = sorted(glob.glob("/kaggle/input/*"))
    wheels = sorted(os.path.basename(p) for p in glob.glob("/kaggle/input/*/*.whl"))

    # Internet? Try PyPI's JSON API with a short timeout.
    try:
        import urllib.request
        urllib.request.urlopen("https://pypi.org/pypi/pip/json", timeout=8).read(64)
        internet = "YES"
    except Exception as exc:
        internet = f"NO ({type(exc).__name__})"

    try:
        pip_list = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            capture_output=True, text=True, timeout=90
        )
        installed = sorted(p["name"] for p in json.loads(pip_list.stdout or "[]"))
    except Exception as exc:
        installed = [f"ERR {exc}"]

    report = {
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "executable": sys.executable,
        "internet": internet,
        "kaggle_input_dirs": inputs,
        "wheels_visible": wheels,
        "key_packages": present,
        "installed_count": len(installed),
        "installed": installed,
    }

    print(json.dumps(report, indent=2))
    return report


# %%
qmlf_env_probe.run(kbench.llm)
