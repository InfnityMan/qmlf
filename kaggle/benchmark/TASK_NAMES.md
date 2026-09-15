# Task naming

Kaggle has no separate title field for a benchmark task: **the slug is the
displayed name**. The generator derives slugs from family and parameters,
which produced unreadable names like `qmlfb-transpile-n4r2fulls65`. Every
task now ships under a self-describing slug instead.

## Rules

- the concept comes first, in words (`kernel-concentration`, `readout-zne`),
  never an abbreviation the reader has to decode
- the distinguishing configuration follows in readable units: `<n>q` qubits,
  `<n>states`, `<n>reps`, `<n>clients`, `<n>pct`, `seed<n>`
- a seed appears only where it is the sole thing separating two variants
- **nothing in a name may reveal an answer.** Two families were checked
  explicitly before naming: the graph prompt states `use_quantum` verbatim,
  and the repair prompt already tells the model how many bugs there are, so
  the concept and the bug *count* are safe to name. Bug *identities* are
  deliberately excluded — `qmlfb-pipeline-repair-4bugs-a`, never the bug list.

The mapping is applied by the generator at build time, so regenerating cannot
silently revert it.

This table covers the public task set only. Held-out variants are renamed by
the same rules but are not listed here — naming them would hand over the
contamination-control set.

## Old -> new

### Kernel concentration (Tier A)

| was | now |
|---|---|
| `qmlfb-conc-blobs10` | `qmlfb-kernel-concentration-blobs-10q` |
| `qmlfb-conc-blobs6` | `qmlfb-kernel-concentration-blobs-6q` |
| `qmlfb-conc-blobs8` | `qmlfb-kernel-concentration-blobs-8q` |
| `qmlfb-conc-circles6` | `qmlfb-kernel-concentration-circles-6q` |
| `qmlfb-conc-moons6` | `qmlfb-kernel-concentration-moons-6q` |
| `qmlfb-conc-quantiles6` | `qmlfb-kernel-concentration-quantiles-6q` |
| `qmlfb-conc-shift6` | `qmlfb-kernel-concentration-shift-6q` |

### Projected kernel at scale (Tier A)

| was | now |
|---|---|
| `qmlfb-proj-blobs10` | `qmlfb-projected-kernel-blobs-10q` |
| `qmlfb-proj-blobs12` | `qmlfb-projected-kernel-blobs-12q` |
| `qmlfb-proj-blobs8` | `qmlfb-projected-kernel-blobs-8q` |
| `qmlfb-proj-moons10` | `qmlfb-projected-kernel-moons-10q` |
| `qmlfb-proj-shift10` | `qmlfb-projected-kernel-shift-10q` |

### Bandwidth selection (Tier A)

| was | now |
|---|---|
| `qmlfb-bwselect-alignment-blobs6` | `qmlfb-bandwidth-alignment-blobs-6q` |
| `qmlfb-bwselect-alignment-moons4` | `qmlfb-bandwidth-alignment-moons-4q` |
| `qmlfb-bwselect-cv-blobs6` | `qmlfb-bandwidth-cv-blobs-6q` |
| `qmlfb-bwselect-cv-moons4` | `qmlfb-bandwidth-cv-moons-4q` |
| `qmlfb-bwselect-cv-shift5` | `qmlfb-bandwidth-cv-shift-5q` |
| `qmlfb-bwselect-median-blobs6` | `qmlfb-bandwidth-median-blobs-6q` |
| `qmlfb-bwselect-median-shift5` | `qmlfb-bandwidth-median-shift-5q` |

### Kernel diagnostics (Tier A)

| was | now |
|---|---|
| `qmlfb-diagnostics-fidelitybw5` | `qmlfb-diagnostics-fidelity-bw0p05` |
| `qmlfb-diagnostics-fidelitybw50` | `qmlfb-diagnostics-fidelity-bw0p5` |
| `qmlfb-diagnostics-projectedbw25` | `qmlfb-diagnostics-projected-bw0p25` |

### ARD / entanglement / graph (Tier A)

| was | now |
|---|---|
| `qmlfb-ard-4of12s38` | `qmlfb-ard-4-informative-of-12` |
| `qmlfb-entangle-blobs5s5` | `qmlfb-entanglement-blobs-5q-seed5` |
| `qmlfb-entangle-blobs6s43` | `qmlfb-entanglement-blobs-6q-seed43` |
| `qmlfb-entangle-shift5s42` | `qmlfb-entanglement-shift-5q-seed42` |
| `qmlfb-graph-ck3` | `qmlfb-graph-kernel-classical-k3` |
| `qmlfb-graph-qk3` | `qmlfb-graph-kernel-quantum-k3` |

### Advantage screening & baseline honesty (Tier B)

| was | now |
|---|---|
| `qmlfb-advantage-blobs4` | `qmlfb-advantage-screen-blobs-4q` |
| `qmlfb-advantage-circles2` | `qmlfb-advantage-screen-circles-2q` |
| `qmlfb-advantage-moons2` | `qmlfb-advantage-screen-moons-2q` |
| `qmlfb-advantage-quantiles4` | `qmlfb-advantage-screen-quantiles-4q` |
| `qmlfb-advantage-xor3` | `qmlfb-advantage-screen-xor-3q` |
| `qmlfb-honesty-blobs6` | `qmlfb-classical-baseline-blobs-6q` |
| `qmlfb-honesty-moons4` | `qmlfb-classical-baseline-moons-4q` |
| `qmlfb-honesty-shift6` | `qmlfb-classical-baseline-shift-6q` |

### Budgets & transpilation (Tier C)

| was | now |
|---|---|
| `qmlfb-budget-n360b9000` | `qmlfb-hardware-budget-270train-9000circuits` |
| `qmlfb-budget-n400b25000` | `qmlfb-hardware-budget-300train-25000circuits` |
| `qmlfb-nisqplan-n1000m50` | `qmlfb-nisq-planning-1000samples-50landmarks` |
| `qmlfb-nisqplan-n200m20` | `qmlfb-nisq-planning-200samples-20landmarks` |
| `qmlfb-nisqplan-n64m8` | `qmlfb-nisq-planning-64samples-8landmarks` |
| `qmlfb-transpile-n4r2fulls65` | `qmlfb-transpile-4q-2reps-full` |
| `qmlfb-transpile-n4r3linears100` | `qmlfb-transpile-4q-3reps-linear` |
| `qmlfb-transpile-n5r2fulls100` | `qmlfb-transpile-5q-2reps-full` |
| `qmlfb-transpile-n5r3circulars65` | `qmlfb-transpile-5q-3reps-circular` |
| `qmlfb-transpile-n6r1circulars65` | `qmlfb-transpile-6q-1rep-circular` |
| `qmlfb-transpile-n6r2linears100` | `qmlfb-transpile-6q-2reps-linear` |

### Noise channels & mitigation (Tier D)

| was | now |
|---|---|
| `qmlfb-depol-n2p40` | `qmlfb-depolarizing-2states-40pct` |
| `qmlfb-depol-n4p10` | `qmlfb-depolarizing-4states-10pct` |
| `qmlfb-depol-n8p25` | `qmlfb-depolarizing-8states-25pct` |
| `qmlfb-mitigate-s5n2` | `qmlfb-readout-zne-2states-seed5` |
| `qmlfb-mitigate-s0n4` | `qmlfb-readout-zne-4states-seed0` |
| `qmlfb-mitigate-s1n4` | `qmlfb-readout-zne-4states-seed1` |
| `qmlfb-mitigate-s3n4` | `qmlfb-readout-zne-4states-seed3` |
| `qmlfb-mitigate-s2n8` | `qmlfb-readout-zne-8states-seed2` |

### Chemistry, VQE, reproducibility (Tier E)

| was | now |
|---|---|
| `qmlfb-chem-s10n3` | `qmlfb-chemistry-3atoms` |
| `qmlfb-chem-s11n5` | `qmlfb-chemistry-5atoms` |
| `qmlfb-determinism-qiga` | `qmlfb-determinism-feature-selector` |
| `qmlfb-determinism-vqe` | `qmlfb-determinism-vqe-energy` |
| `qmlfb-vqe-s4n2` | `qmlfb-vqe-2atoms` |
| `qmlfb-vqe-s1n3` | `qmlfb-vqe-3atoms` |

### Pipelines, federated, figures, regression (Tier F)

| was | now |
|---|---|
| `qmlfb-federated-c3r3` | `qmlfb-federated-3clients-3reporting` |
| `qmlfb-federated-c5r3` | `qmlfb-federated-5clients-3reporting` |
| `qmlfb-federated-c8r5` | `qmlfb-federated-8clients-5reporting` |
| `qmlfb-viz-eigen` | `qmlfb-figure-eigenspectrum` |
| `qmlfb-viz-hilbert` | `qmlfb-figure-hilbert-space` |
| `qmlfb-industrial-breastcancer` | `qmlfb-industrial-breast-cancer` |
| `qmlfb-debug-set2` | `qmlfb-pipeline-repair-2bugs-a` |
| `qmlfb-debug-set6` | `qmlfb-pipeline-repair-2bugs-b` |
| `qmlfb-debug-set3` | `qmlfb-pipeline-repair-3bugs-a` |
| `qmlfb-debug-set4` | `qmlfb-pipeline-repair-3bugs-b` |
| `qmlfb-debug-set1` | `qmlfb-pipeline-repair-4bugs-a` |
| `qmlfb-debug-set5` | `qmlfb-pipeline-repair-4bugs-b` |
| `qmlfb-regression-sinc` | `qmlfb-regression-radial-sinc` |

## Already self-describing (unchanged)

- `qmlfb-advantage-screen-judgment`
- `qmlfb-ard-noise-suppression`
- `qmlfb-circuit-budget-nystrom`
- `qmlfb-classical-baseline-honesty`
- `qmlfb-debug-broken-pipeline`
- `qmlfb-entanglement-is-not-free`
- `qmlfb-federated-partial-participation`
- `qmlfb-industrial-iris`
- `qmlfb-industrial-wide-data-pipeline`
- `qmlfb-industrial-wine`
- `qmlfb-mitigation-pipeline`
- `qmlfb-nisq-transpile-honesty`
- `qmlfb-projected-kernel-at-scale`
- `qmlfb-qnn-reproducible-training`
- `qmlfb-regression-physical-model`
- `qmlfb-regression-quadratic`
- `qmlfb-vqe-variational-principle`
