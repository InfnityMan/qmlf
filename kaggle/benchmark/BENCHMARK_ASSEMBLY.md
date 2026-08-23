# Assembling the benchmark on Kaggle (web UI)

Neither the Kaggle CLI nor the SDK exposes a create-benchmark or
add-task-to-benchmark call -- the entire API surface is task-level
(`create_benchmark_task`, `list`, `run`, `publish`, leaderboard reads;
verified against `kagglesdk.benchmarks.services`). Collections are assembled
in the web UI. Every task below already exists on the account as a **private,
Completed** task (114/114 verified).

## Benchmark 1 -- "qmlf" (the public-candidate set, 91 tasks)

1. Go to https://www.kaggle.com/benchmarks -> **New Benchmark**.
2. Name: `qmlf` (slug becomes `hemakarapu/qmlf`). Keep it **private**.
3. Add tasks: search by slug and add each of the 91 below (all `qmlfb-*`,
   listed by family so you can tick them in order). Do **not** add the 22
   held-out slugs (`private/heldout.txt` on the `benchmark-private` branch) or
   the canary `qmlfb-kernel-concentration-rescue`.
4. Save. Do not publish yet.

### advantage (6)
- `qmlfb-advantage-blobs4`
- `qmlfb-advantage-circles2`
- `qmlfb-advantage-moons2`
- `qmlfb-advantage-quantiles4`
- `qmlfb-advantage-screen-judgment`
- `qmlfb-advantage-xor3`

### ard (2)
- `qmlfb-ard-4of12s38`
- `qmlfb-ard-noise-suppression`

### budget (2)
- `qmlfb-budget-n360b9000`
- `qmlfb-budget-n400b25000`

### bwselect (7)
- `qmlfb-bwselect-alignment-blobs6`
- `qmlfb-bwselect-alignment-moons4`
- `qmlfb-bwselect-cv-blobs6`
- `qmlfb-bwselect-cv-moons4`
- `qmlfb-bwselect-cv-shift5`
- `qmlfb-bwselect-median-blobs6`
- `qmlfb-bwselect-median-shift5`

### chem (2)
- `qmlfb-chem-s10n3`
- `qmlfb-chem-s11n5`

### conc (7)
- `qmlfb-conc-blobs10`
- `qmlfb-conc-blobs6`
- `qmlfb-conc-blobs8`
- `qmlfb-conc-circles6`
- `qmlfb-conc-moons6`
- `qmlfb-conc-quantiles6`
- `qmlfb-conc-shift6`

### debug (7)
- `qmlfb-debug-broken-pipeline`
- `qmlfb-debug-set1`
- `qmlfb-debug-set2`
- `qmlfb-debug-set3`
- `qmlfb-debug-set4`
- `qmlfb-debug-set5`
- `qmlfb-debug-set6`

### depol (3)
- `qmlfb-depol-n2p40`
- `qmlfb-depol-n4p10`
- `qmlfb-depol-n8p25`

### determinism (2)
- `qmlfb-determinism-qiga`
- `qmlfb-determinism-vqe`

### diagnostics (3)
- `qmlfb-diagnostics-fidelitybw5`
- `qmlfb-diagnostics-fidelitybw50`
- `qmlfb-diagnostics-projectedbw25`

### entangle (3)
- `qmlfb-entangle-blobs5s5`
- `qmlfb-entangle-blobs6s43`
- `qmlfb-entangle-shift5s42`

### federated (4)
- `qmlfb-federated-c3r3`
- `qmlfb-federated-c5r3`
- `qmlfb-federated-c8r5`
- `qmlfb-federated-partial-participation`

### graph (2)
- `qmlfb-graph-ck3`
- `qmlfb-graph-qk3`

### hand-built pilot (7)
- `qmlfb-circuit-budget-nystrom`
- `qmlfb-classical-baseline-honesty`
- `qmlfb-entanglement-is-not-free`
- `qmlfb-mitigation-pipeline`
- `qmlfb-nisq-transpile-honesty`
- `qmlfb-projected-kernel-at-scale`
- `qmlfb-qnn-reproducible-training`

### honesty (3)
- `qmlfb-honesty-blobs6`
- `qmlfb-honesty-moons4`
- `qmlfb-honesty-shift6`

### industrial (4)
- `qmlfb-industrial-breastcancer`
- `qmlfb-industrial-iris`
- `qmlfb-industrial-wide-data-pipeline`
- `qmlfb-industrial-wine`

### mitigate (5)
- `qmlfb-mitigate-s0n4`
- `qmlfb-mitigate-s1n4`
- `qmlfb-mitigate-s2n8`
- `qmlfb-mitigate-s3n4`
- `qmlfb-mitigate-s5n2`

### nisqplan (3)
- `qmlfb-nisqplan-n1000m50`
- `qmlfb-nisqplan-n200m20`
- `qmlfb-nisqplan-n64m8`

### proj (5)
- `qmlfb-proj-blobs10`
- `qmlfb-proj-blobs12`
- `qmlfb-proj-blobs8`
- `qmlfb-proj-moons10`
- `qmlfb-proj-shift10`

### regression (3)
- `qmlfb-regression-physical-model`
- `qmlfb-regression-quadratic`
- `qmlfb-regression-sinc`

### transpile (6)
- `qmlfb-transpile-n4r2fulls65`
- `qmlfb-transpile-n4r3linears100`
- `qmlfb-transpile-n5r2fulls100`
- `qmlfb-transpile-n5r3circulars65`
- `qmlfb-transpile-n6r1circulars65`
- `qmlfb-transpile-n6r2linears100`

### viz (2)
- `qmlfb-viz-eigen`
- `qmlfb-viz-hilbert`

### vqe (3)
- `qmlfb-vqe-s1n3`
- `qmlfb-vqe-s4n2`
- `qmlfb-vqe-variational-principle`

## Benchmark 2 -- "qmlf-heldout" (contamination control, never published)

Same steps; name it `qmlf-heldout`; add the 22 slugs from
`kaggle/benchmark/private/heldout.txt` plus `qmlfb-kernel-concentration-rescue`
(the canary). This benchmark stays private permanently and is run against
every model the public one is run against.

## After assembly

- Leaderboard: `kaggle b leaderboard hemakarapu/qmlf -s`
- Running models is the next step and spends quota: pick models from
  `kaggle b t models` and run per task with
  `kaggle b t run <slug> -m <model> ...`, or from the benchmark page.
