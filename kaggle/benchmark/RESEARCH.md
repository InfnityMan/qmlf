# Research basis for the qmlf benchmark

Sources consulted (built-in web search; no firecrawl), and how each shaped a task.

## Kernel geometry and concentration

- **Thanasilp, Wang, Cerezo, Holmes — "Exponential concentration in quantum
  kernel methods", Nature Communications 15, 5200 (2024).**
  https://www.nature.com/articles/s41467-024-49287-w
  Fidelity kernel values concentrate exponentially in qubit count under
  expressive encodings, entanglement, global measurements, or noise; the
  model becomes trivial with polynomial shots. → Tasks A1 (concentration
  rescue), A2 (projected kernel at scale), D1 (shot-noise awareness).
- **Shaydulin & Wild, "Importance of kernel bandwidth in quantum machine
  learning" (2022); Canatar et al., "Bandwidth enables generalization in
  quantum kernel models" (2022).** Bandwidth is the single most important
  hyperparameter of a fidelity kernel and the standard concentration remedy.
  → Task A1 thresholds; A3 (per-feature bandwidth).
- **Huang et al., "Power of data in quantum machine learning", Nature
  Communications 12, 2631 (2021).** Projected quantum kernels; the geometric
  difference g and model complexity s as the principled advantage test.
  → Tasks A2, B1 (advantage screen).

## Benchmarking honesty

- **Bowles, Ahmed, Schuld — "Better than classical? The subtle art of
  benchmarking quantum machine learning models" (2024).**
  https://arxiv.org/abs/2403.07059
  12 QML models × 160 datasets: out-of-the-box classical models usually win,
  and *removing entanglement often helps*. ~40% of papers claim quantum
  outperformance; 4% report the opposite. → Tasks A4 (entanglement is not
  free), B2 (classical-baseline honesty): the benchmark rewards models that
  report an honest comparison, and penalises hype.
- **"Quantum kernel methods under scrutiny: a benchmarking study", Quantum
  Machine Intelligence (2025)** — 64 datasets, 9 encodings, up to 15 qubits;
  and **"Benchmarking Quantum Kernel SVMs Against Classical Baselines on
  Tabular Data" (2026, 970 experiments with hardware validation).**
  → Task F1 (industrial wide-data pipeline) uses real tabular data and
  demands a classical-competitive threshold.

## LLM + quantum code

- **QuanBench / QuanBench+ (2026), Qiskit HumanEval (2024/25).** Best models
  score below 40% Pass@1 on quantum code; functional-correctness grading
  with semantic checks. → Code-execution grading mode throughout: the model's
  code is run, never trusted.
- **QBugLM (2026):** LLM quantum *debugging* is "not yet systematically
  investigated". → Task F3 (repair a broken qmlf pipeline with five planted,
  literature-documented bugs).

## Error mitigation

- **Digital ZNE best practices (Mitiq authors, 2023) and readout-error
  mitigation via calibration-matrix inversion (2022–24).** Combine readout
  correction with ZNE; inversion can produce negative quasi-probabilities
  that must be renormalised; extrapolation needs ≥2 noise scales.
  → Task D1 (mitigation pipeline) with exactly those traps.

## Applications

- Finance (credit scoring, fraud: Miyabe et al. 2023), HEP (Belis et al.
  2021, QSVM on trapped ions), drug discovery (Q²SAR 2025, multiple-kernel
  learning). → Task C1 (hardware circuit budget) models the NISQ constraint
  every one of these groups reports; F1/F2 use tabular/physical targets in
  that spirit.
