# Suite manifest (generated)

77 public generated tasks across 22 families, plus the 15 hand-built tasks. A further 22 held-out variants exist and are not listed.

| family | task | mode | measured reference |
|---|---|---|---|
| conc | `qmlfb-conc-blobs6` | plan | Naive (bandwidth 1.0): acc 0.54, off-diagonal 0.0158. Tuned: 0.93 / 0.17. Data: make_dataset("blobs", n_samples=110, n_features=6, seed=1) |
| conc | `qmlfb-conc-blobs8` | plan | Naive (bandwidth 1.0): acc 0.54, off-diagonal 0.0041. Tuned: 0.93 / 0.38. Data: make_dataset("blobs", n_samples=110, n_features=8, seed=2) |
| conc | `qmlfb-conc-moons6` | plan | Naive (bandwidth 1.0): acc 0.43, off-diagonal 0.0197. Tuned: 0.96 / 0.30. Data: make_dataset("moons", n_samples=110, n_features=6, seed=3) |
| conc | `qmlfb-conc-circles6` | plan | Naive (bandwidth 1.0): acc 0.71, off-diagonal 0.0204. Tuned: 0.96 / 0.36. Data: make_dataset("circles", n_samples=110, n_features=6, seed=4) |
| conc | `qmlfb-conc-shift6` | plan | Naive (bandwidth 1.0): acc 0.50, off-diagonal 0.0171. Tuned: 0.93 / 0.84. Data: make_dataset("shift", n_samples=110, n_features=6, seed=5) |
| conc | `qmlfb-conc-blobs10` | plan | Naive (bandwidth 1.0): acc 0.46, off-diagonal 0.0010. Tuned: 0.96 / 0.37. Data: make_dataset("blobs", n_samples=110, n_features=10, seed=6) |
| conc | `qmlfb-conc-quantiles6` | plan | Naive (bandwidth 1.0): acc 0.50, off-diagonal 0.0167. Tuned: 0.82 / 0.38. Data: make_dataset("quantiles", n_samples=110, n_features=6, seed=8) |
| proj | `qmlfb-proj-blobs8` | code | 8 qubits. Naive acc 0.40 (offdiag 0.0039); projected kernel 0.97 (offdiag 0.38). |
| proj | `qmlfb-proj-blobs10` | code | 10 qubits. Naive acc 0.50 (offdiag 0.0011); projected kernel 0.80 (offdiag 0.38). |
| proj | `qmlfb-proj-shift10` | code | 10 qubits. Naive acc 0.60 (offdiag 0.0012); projected kernel 0.97 (offdiag 0.42). |
| proj | `qmlfb-proj-blobs12` | code | 12 qubits. Naive acc 0.43 (offdiag 0.0003); projected kernel 0.67 (offdiag 0.37). |
| proj | `qmlfb-proj-moons10` | code | 10 qubits. Naive acc 0.47 (offdiag 0.0019); projected kernel 0.83 (offdiag 0.38). |
| ard | `qmlfb-ard-4of12s38` | code | 4 informative + 8 noise features. Best scalar 0.607; ARD 0.750. |
| entangle | `qmlfb-entangle-blobs5s5` | plan | 'z' map 0.96 with 0 two-qubit gates; 'zz' map 0.92 with 40. |
| entangle | `qmlfb-entangle-shift5s42` | plan | 'z' map 0.88 with 0 two-qubit gates; 'zz' map 0.84 with 40. |
| entangle | `qmlfb-entangle-blobs6s43` | plan | 'z' map 0.96 with 0 two-qubit gates; 'zz' map 0.88 with 60. |
| bwselect | `qmlfb-bwselect-cv-blobs6` | code | Principle cv -> 0.02 on this training set. |
| bwselect | `qmlfb-bwselect-cv-moons4` | code | Principle cv -> 0.1 on this training set. |
| bwselect | `qmlfb-bwselect-cv-shift5` | code | Principle cv -> 0.02 on this training set. |
| bwselect | `qmlfb-bwselect-median-blobs6` | code | Principle median -> 0.221605 on this training set. |
| bwselect | `qmlfb-bwselect-median-shift5` | code | Principle median -> 0.282678 on this training set. |
| bwselect | `qmlfb-bwselect-alignment-blobs6` | code | Principle alignment -> 0.05 on this training set. |
| bwselect | `qmlfb-bwselect-alignment-moons4` | code | Principle alignment -> 0.02 on this training set. |
| advantage | `qmlfb-advantage-moons2` | code | Screen verdict on this data: inconclusive (g = 611.60). |
| advantage | `qmlfb-advantage-circles2` | code | Screen verdict on this data: inconclusive (g = 391.09). |
| advantage | `qmlfb-advantage-blobs4` | code | Screen verdict on this data: quantum candidate (g = 266.49). |
| advantage | `qmlfb-advantage-xor3` | code | Screen verdict on this data: quantum candidate (g = 278.33). |
| advantage | `qmlfb-advantage-quantiles4` | code | Screen verdict on this data: quantum candidate (g = 153.91). |
| honesty | `qmlfb-honesty-blobs6` | code | Standard classical models: 0.87-1.00; auto quantum: 0.97. |
| honesty | `qmlfb-honesty-moons4` | code | Standard classical models: 0.90-0.97; auto quantum: 0.97. |
| honesty | `qmlfb-honesty-shift6` | code | Standard classical models: 0.93-0.97; auto quantum: 0.93. |
| budget | `qmlfb-budget-n400b25000` | code | 300 rows need 44,850 pairwise circuits; budget 25,000; max feasible m=74 gives 0.760. |
| budget | `qmlfb-budget-n360b9000` | code | 270 rows need 36,315 pairwise circuits; budget 9,000; max feasible m=31 gives 0.867. |
| transpile | `qmlfb-transpile-n5r2fulls100` | code | Real transpiler depth 40, 2q gates 40; analytic estimate claims depth 16. |
| transpile | `qmlfb-transpile-n4r3linears100` | code | Real transpiler depth 27, 2q gates 18; analytic estimate claims depth 16. |
| transpile | `qmlfb-transpile-n6r1circulars65` | code | Real transpiler depth 20, 2q gates 12; analytic estimate claims depth 13. |
| transpile | `qmlfb-transpile-n4r2fulls65` | code | Real transpiler depth 31, 2q gates 24; analytic estimate claims depth 16. |
| transpile | `qmlfb-transpile-n6r2linears100` | code | Real transpiler depth 25, 2q gates 20; analytic estimate claims depth 16. |
| transpile | `qmlfb-transpile-n5r3circulars65` | code | Real transpiler depth 51, 2q gates 30; analytic estimate claims depth 16. |
| mitigate | `qmlfb-mitigate-s0n4` | code | Raw 0.076, readout-only 0.050, full pipeline 0.0322 (L1 to ideal). |
| mitigate | `qmlfb-mitigate-s1n4` | code | Raw 0.207, readout-only 0.096, full pipeline 0.0532 (L1 to ideal). |
| mitigate | `qmlfb-mitigate-s2n8` | code | Raw 0.101, readout-only 0.064, full pipeline 0.0193 (L1 to ideal). |
| mitigate | `qmlfb-mitigate-s3n4` | code | Raw 0.181, readout-only 0.120, full pipeline 0.0324 (L1 to ideal). |
| mitigate | `qmlfb-mitigate-s5n2` | code | Raw 0.129, readout-only 0.046, full pipeline 0.0000 (L1 to ideal). |
| depol | `qmlfb-depol-n4p10` | code | Raw L1 0.097; exact inversion recovers the ideal. |
| depol | `qmlfb-depol-n8p25` | code | Raw L1 0.231; exact inversion recovers the ideal. |
| depol | `qmlfb-depol-n2p40` | code | Raw L1 0.224; exact inversion recovers the ideal. |
| vqe | `qmlfb-vqe-s1n3` | code | exact -6.491856; operator ground state -27.630314; VQE from zeros -27.619598. |
| vqe | `qmlfb-vqe-s4n2` | code | exact 32.265209; operator ground state -29.565827; VQE from zeros -29.565388. |
| chem | `qmlfb-chem-s10n3` | code | 3-atom geometry with nuclear charges. |
| chem | `qmlfb-chem-s11n5` | code | 5-atom geometry with nuclear charges. |
| industrial | `qmlfb-industrial-breastcancer` | plan | breast_cancer: naive 0.62, auto 0.92, projected 0.88, classical RBF 0.98. |
| industrial | `qmlfb-industrial-wine` | plan | wine: naive 0.39, auto 0.95, projected 0.95, classical RBF 0.95. |
| industrial | `qmlfb-industrial-iris` | plan | iris: naive 0.50, auto 0.92, projected 0.89, classical RBF 0.95. |
| regression | `qmlfb-regression-sinc` | plan | sinc: naive R2 -0.19, fidelity auto 0.92, projected 0.82. |
| regression | `qmlfb-regression-quadratic` | plan | quadratic: naive R2 -0.55, fidelity auto 0.98, projected 0.87. |
| debug | `qmlfb-debug-set1` | code | Planted: n_qubits disagrees with the 5-feature data; raw fidelity kernel handed to QSVC (discards whitening/normalize/bandwidth); mis-cased strategy name; 1-D input to ZNE. Correct pipeline: acc 0.95, L1 0.0322. |
| debug | `qmlfb-debug-set2` | code | Planted: raw fidelity kernel handed to QSVC (discards whitening/normalize/bandwidth); readout mitigator used without .fit(confusion). Correct pipeline: acc 0.95, L1 0.0334. |
| debug | `qmlfb-debug-set3` | code | Planted: n_qubits disagrees with the 5-feature data; 1-D input to ZNE; whitened mode without normalisation (angles alias). Correct pipeline: acc 0.90, L1 0.0186. |
| debug | `qmlfb-debug-set4` | code | Planted: mis-cased strategy name; readout mitigator used without .fit(confusion); whitened mode without normalisation (angles alias). Correct pipeline: acc 0.90, L1 0.0539. |
| debug | `qmlfb-debug-set5` | code | Planted: n_qubits disagrees with the 5-feature data; raw fidelity kernel handed to QSVC (discards whitening/normalize/bandwidth); readout mitigator used without .fit(confusion); 1-D input to ZNE. Correct pipeline: acc 0.85, L1 0.0138. |
| debug | `qmlfb-debug-set6` | code | Planted: whitened mode without normalisation (angles alias); 1-D input to ZNE. Correct pipeline: acc 0.90, L1 0.0280. |
| federated | `qmlfb-federated-c5r3` | code | 3 of 5 clients reported with unequal sample counts. |
| federated | `qmlfb-federated-c8r5` | code | 5 of 8 clients reported with unequal sample counts. |
| federated | `qmlfb-federated-c3r3` | code | 3 of 3 clients reported with unequal sample counts. |
| determinism | `qmlfb-determinism-qiga` | code | Two runs of the selector must produce identical feature importances (use random_state=...) |
| determinism | `qmlfb-determinism-vqe` | code | Two VQE runs must return identical energies (fix initial_point) |
| diagnostics | `qmlfb-diagnostics-fidelitybw5` | code | fidelity kernel, bandwidth 0.05: offdiag 0.2496, KTA 0.4726, verdict healthy. |
| diagnostics | `qmlfb-diagnostics-projectedbw25` | code | projected kernel, bandwidth 0.25: offdiag 0.3669, KTA 0.0920, verdict healthy. |
| diagnostics | `qmlfb-diagnostics-fidelitybw50` | code | fidelity kernel, bandwidth 0.5: offdiag 0.0340, KTA 0.1562, verdict severely concentrated. |
| graph | `qmlfb-graph-ck3` | code | classical backend, k=3, n_qubits=4. |
| graph | `qmlfb-graph-qk3` | code | quantum backend, k=3, n_qubits=4. |
| viz | `qmlfb-viz-hilbert` | code | QVizPro.plot_hilbert_space(gram, labels=y_train, ...) written to HTML, no display. |
| viz | `qmlfb-viz-eigen` | code | QVizPro.plot_kernel_eigenvalues(gram, ...) written to HTML, no display. |
| nisqplan | `qmlfb-nisqplan-n200m20` | code | n=200, m=20: {'pairwise_fidelity_circuits': 19900, 'statevector_simulations': 200, 'nystrom_fidelity_circuits': 4190} |
| nisqplan | `qmlfb-nisqplan-n1000m50` | code | n=1000, m=50: {'pairwise_fidelity_circuits': 499500, 'statevector_simulations': 1000, 'nystrom_fidelity_circuits': 51225} |
| nisqplan | `qmlfb-nisqplan-n64m8` | code | n=64, m=8: {'pairwise_fidelity_circuits': 2016, 'statevector_simulations': 64, 'nystrom_fidelity_circuits': 540} |
