"""CLI contract tests: 429 lines that previously had zero coverage.

Everything runs headless (--no-show / --no-visualize) on tiny synthetic data so
the whole file stays in single-digit seconds.

Run with:  python -m pytest tests/ -q
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import warnings

import numpy as np
import pandas as pd
import pytest

from qmlf.cli import build_parser, main


def run_cli(argv):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main(argv)


def test_no_command_prints_help(capsys):
    main([])
    assert "usage: qmlf" in capsys.readouterr().out


def test_version_flag():
    import qmlf

    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])

    assert excinfo.value.code == 0


def test_benchmark_prints_both_models(capsys):
    run_cli(["benchmark", "--n-qubits", "3", "--n-samples", "24"])
    out = capsys.readouterr().out

    assert "XGBoost" in out and "Quantum SVC" in out


def test_benchmark_accepts_kernel_tuning_flags(capsys):
    run_cli(["benchmark", "--n-qubits", "3", "--n-samples", "24",
             "--mode", "mahalanobis", "--normalize", "maxabs",
             "--bandwidth", "0.25", "--feature-map", "z"])

    assert "Quantum SVC" in capsys.readouterr().out


def test_visualize_saves_html_headless(tmp_path, capsys):
    target = tmp_path / "kernel.html"
    run_cli(["visualize", "--n-qubits", "3", "--n-samples", "12",
             "--no-show", "--save", str(target)])

    assert target.exists() and target.stat().st_size > 0
    assert "Saved visualization" in capsys.readouterr().out


def test_pipeline_reports_shapes(capsys):
    run_cli(["pipeline", "--n-qubits", "3", "--n-samples", "12", "--reps", "1",
             "--output-dim", "4", "--no-visualize", "--no-show"])
    out = capsys.readouterr().out

    assert "kernel_matrix: (12, 12)" in out
    assert "qnn_output: (12, 4)" in out


def test_optimize_circuit_reports_estimate(capsys):
    run_cli(["optimize-circuit", "--depth", "20", "--gate-count", "40",
             "--two-qubit-gates", "10"])
    out = capsys.readouterr().out

    assert "mode: estimate" in out
    assert "mitigated_fidelity" in out


def test_csv_loading_with_named_target(tmp_path, capsys):
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "a": rng.normal(size=24), "b": rng.normal(size=24),
        "c": rng.normal(size=24), "label": rng.integers(0, 2, size=24),
    })
    csv = tmp_path / "data.csv"
    frame.to_csv(csv, index=False)

    run_cli(["benchmark", "--csv", str(csv), "--target", "label",
             "--n-qubits", "3"])

    assert "Quantum SVC" in capsys.readouterr().out


def test_csv_missing_target_is_a_clear_error(tmp_path):
    csv = tmp_path / "data.csv"
    pd.DataFrame({"a": [1.0, 2.0], "b": [0, 1]}).to_csv(csv, index=False)

    with pytest.raises(ValueError, match="not found"):
        main(["benchmark", "--csv", str(csv), "--target", "nope",
              "--n-qubits", "1"])


def test_invalid_mode_rejected_by_parser(capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["benchmark", "--mode", "banana"])
