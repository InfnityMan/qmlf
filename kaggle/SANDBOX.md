# Kaggle Benchmarks sandbox — measured facts

Everything here was read out of a live run, not inferred. Probe task:
`probe_env.py` (task slug `qmlf-env-probe`), run 2026-08-21.

## Environment

| | |
|---|---|
| Python | 3.11.15 |
| Platform | linux |
| Interpreter | `/benchmarks/.venv/bin/python` |
| Internet | **available** (PyPI reachable) |
| Installed packages | 178 |

## What is NOT preinstalled

This is **not** the Kaggle notebook data-science image. It is a minimal
Jupyter/nbconvert environment.

| Package | Status |
|---|---|
| numpy | 2.3.5 present |
| pandas | 2.3.3 present |
| **scikit-learn** | **missing** |
| **scipy** | **missing** |
| **torch** | **missing** |
| **xgboost** | **missing** |
| **plotly** | **missing** |
| **qiskit / qiskit-machine-learning / qiskit-algorithms** | **missing** |

A task that does `from sklearn...` at the top of its body dies at task-creation
time with `ModuleNotFoundError`, before any model is called.

## Attached-dataset path

Datasets attached with `-d owner/slug` land at:

```
/kaggle/input/datasets/<owner>/<slug>/
```

**not** `/kaggle/input/<slug>/`. A glob of `/kaggle/input/*/*.whl` finds nothing.
Walk the tree instead.

## Install cost

Installing the qiskit stack + scikit-learn + qmlf from the attached wheelhouse
with `--no-index` (no network):

```
install_seconds: 14.6      both pip steps rc=0
```

qmlf goes in with `--no-deps` on purpose. torch and xgboost are declared
dependencies but are imported lazily and are not on the kernel code path, so
installing them would cost an ~800MB download for nothing. A task that exercises
the QNN layers does need torch and should drop `--no-deps`.

## Verified in-sandbox behaviour

```
qmlf_version      : 1.2.1
gram_shape        : [12, 12]
gram_offdiag_mean : 0.226194
bit_reproducible  : true      <- identical Gram matrices across two builds
```

## Platform constraint worth knowing

qiskit 2.4.0 moved its linux wheels from `manylinux_2_17` to `manylinux_2_28`,
so `qiskit>=2.4` requires a host with **glibc >= 2.28**. The benchmark sandbox
satisfies this. Build the wheelhouse with
`--platform manylinux_2_28_x86_64 --python-version 311 --abi cp311 --abi abi3 --abi none`;
omitting the abi3 tag silently excludes qiskit, which ships `cp310-abi3` wheels.

## Structured output

`llm.prompt(..., schema=..., reasoning="high")` failed: the model emitted its
`<think>` trace into the message body and the JSON parser raised
`expected value at line 1 column 1`. Drop `reasoning=` when using a schema, and
prefer plain `str` fields with a `"none"` sentinel over `str | None`.
