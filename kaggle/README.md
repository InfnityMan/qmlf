# Kaggle Benchmarks setup for qmlf

> **The benchmark suite lives in [`benchmark/`](benchmark/BENCHMARK.md)** — 15 private
> `qmlfb-*` tasks, their research basis, reference sweeps, generator, and a
> zero-LLM local validator. This file covers the install/sandbox plumbing.

Everything here is verified against the live Kaggle Benchmarks sandbox. See
[SANDBOX.md](SANDBOX.md) for the measured environment facts.

## Status

| Piece | State |
|---|---|
| `hemakarapu/qmlf-wheelhouse` dataset | created, **private**, 15 wheels, status `ready` |
| `qmlf-quantum-kernel-tuning` task | pushed, **private**, v3, runs green |
| Offline install in sandbox | verified, 14.6s, no network needed |
| Determinism in sandbox | verified, `bit_reproducible: true` |
| Wheelhouse sync | dataset holds **qmlf-1.4.0**, matching this tree — re-version after any future framework change (command below) |
| PyPI | **not published** — the wheelhouse is what makes this work today |

## Files

- `task_quantum_kernel_tuning.py` — the benchmark task
- `probe_env.py` / `probe_install.py` — zero-quota probes that report the sandbox
  environment and prove the install path. Neither calls `llm.prompt`, so neither
  costs model credits. Keep them; re-run after any Kaggle image change.
- `reference_sweep.py` / `.txt` — the bandwidth/concentration sweep the task's
  thresholds are derived from. Regenerate if the dataset or split changes.
- `qmlf-wheelhouse/` — the dataset payload (wheels + `dataset-metadata.json`)

## The task

The model configures a quantum fidelity kernel and is graded on the kernel it
actually produces, not on a number it reports. It must work out that fidelity
kernels concentrate, that bandwidth is the remedy, and that handing
`.fidelity_quantum_kernel` to a classifier silently discards preprocessing.

Measured discrimination on this split:

| configuration | off-diag | accuracy | verdict |
|---|---|---|---|
| library defaults (`ZZ`, bandwidth 1.0) | 0.017 | 0.400 | fail |
| correct wiring but raw kernel object | 0.771 | 0.800 | fail — bypass trap |
| `mahalanobis`, `maxabs`, bandwidth 0.05 | 0.771 | 0.800 | pass |
| `ZZ`, bandwidth 0.02 | 0.617 | 0.750 | pass |

First live result — Gemini 3 Flash Preview, 3 of 4 assertions passed:

```
mode=mahalanobis  normalize=std  bandwidth=0.15  entanglement=linear
svm_wiring=precomputed
offdiag_mean=0.06428   test_accuracy=0.80
```

It reached the best-known accuracy while its Gram matrix was still concentrated.
The off-diagonal assertion is what caught that; on accuracy alone it would have
looked like a perfect answer.

## Rebuilding the wheelhouse

Needed whenever qmlf changes. The abi3 tags are not optional — leave them out and
qiskit is silently excluded.

```bash
pip download --dest kaggle/qmlf-wheelhouse --only-binary=:all: \
  --platform manylinux_2_28_x86_64 --platform manylinux_2_17_x86_64 \
  --python-version 311 --implementation cp --abi cp311 --abi abi3 --abi none \
  "qiskit>=2.4" "qiskit-machine-learning>=0.9" "qiskit-algorithms>=0.4"
```

```bash
python -m build && cp dist/qmlf-*.whl kaggle/qmlf-wheelhouse/
```

```bash
kaggle datasets version -p kaggle/qmlf-wheelhouse -m "qmlf $(python -c 'import qmlf;print(qmlf.__version__)')" -r skip
```

## Pushing and running

```bash
kaggle b t push qmlf-quantum-kernel-tuning -f kaggle/task_quantum_kernel_tuning.py -d hemakarapu/qmlf-wheelhouse
```

```bash
kaggle b t run qmlf-quantum-kernel-tuning -m claude-opus-5-default --wait
```

Tasks stay private until you explicitly run `kaggle b t publish`.

## Once qmlf is on PyPI

`_ensure_qmlf()` already falls back to PyPI when no wheelhouse is attached, so
publishing lets you drop the `-d` flag. The sandbox has internet, so that path
works — it just needs the package to exist.
