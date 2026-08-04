import argparse

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from qmlf import __version__
from qmlf.integration.q_pipeline import create_quantum_pipeline
from qmlf.nisq.q_nisq import create_advanced_nisq_optimizer
from qmlf.ops.q_ops_core import QuantumKernel, run_quantum_benchmark
from qmlf.viz.q_viz_pro import QVizPro


def _load_dataset(csv_path, target, n_samples, n_features):
    # Load features/labels from a CSV, or fall back to a synthetic dataset so
    # the CLI works out of the box with nothing but a fresh install.
    if csv_path is not None:
        frame = pd.read_csv(csv_path)

        if target is None:
            target = frame.columns[-1]

        if target not in frame.columns:
            raise ValueError(
                f"Target column '{target}' not found in {csv_path}"
            )

        y = frame[target].to_numpy()
        X = frame.drop(columns=[target]).to_numpy(dtype=float)

        return X, y

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, n_features // 2),
        n_redundant=0,
        random_state=42
    )

    return X, y


def _run_benchmark(args):
    X, y = _load_dataset(
        args.csv,
        args.target,
        args.n_samples,
        args.n_qubits
    )

    results = run_quantum_benchmark(
        X,
        y,
        n_qubits=args.n_qubits,
        reps=args.reps,
        test_size=args.test_size
    )

    print(results.to_string(index=False))


def _run_visualize(args):
    X, y = _load_dataset(
        args.csv,
        args.target,
        args.n_samples,
        args.n_qubits
    )

    n_features = X.shape[1]

    kernel = QuantumKernel(
        n_qubits=n_features,
        mode=args.mode,
        reps=args.reps,
        bandwidth=args.bandwidth,
        normalize=args.normalize
    )

    kernel.fit(X)

    kernel_matrix = kernel.compute_kernel_matrix(
        X,
        batch_size=args.batch_size
    )

    QVizPro.plot_hilbert_space(
        kernel_matrix,
        labels=y,
        show=not args.no_show,
        save_path=args.save
    )

    if args.save is not None:
        print(f"Saved visualization to {args.save}")


def _run_pipeline(args):
    X, y = _load_dataset(
        args.csv,
        args.target,
        args.n_samples,
        args.n_qubits
    )

    pipeline = create_quantum_pipeline(
        n_qubits=args.n_qubits,
        mode=args.mode,
        reps=args.reps,
        output_dim=args.output_dim,
        bandwidth=args.bandwidth,
        normalize=args.normalize
    )

    result = pipeline.run(
        X,
        labels=y,
        visualize=not args.no_visualize,
        show=not args.no_show,
        save_path=args.save,
        batch_size=args.batch_size
    )

    print(f"kernel_matrix: {result['kernel_matrix'].shape}")
    print(f"qnn_output: {result['qnn_output'].shape}")

    if args.save is not None:
        print(f"Saved visualization to {args.save}")


def _run_optimize_circuit(args):
    optimizer = create_advanced_nisq_optimizer(
        max_depth=args.max_depth,
        optimization_strength=args.optimization_strength
    )

    report = optimizer.optimize_circuit(
        depth=args.depth,
        gate_count=args.gate_count,
        two_qubit_gates=args.two_qubit_gates
    )

    for key, value in report.items():
        print(f"{key}: {value}")


def _add_kernel_tuning_arguments(subparser):
    # Fidelity kernels concentrate as the encoding angles widen, so bandwidth is
    # usually the single most important knob to sweep; see QuantumKernel's
    # docstring. Exposed here so the CLI can reach it, not just Python callers.
    subparser.add_argument(
        "--bandwidth",
        type=float,
        default=1.0,
        help=(
            "Scale factor on the encoding angles. Lower values widen the kernel "
            "and usually help; try 0.5 / 0.25 / 0.1. Default 1.0."
        )
    )

    subparser.add_argument(
        "--normalize",
        default=None,
        choices=["maxabs", "std"],
        help=(
            "Rescale prepared angles by a train-fitted scale before bandwidth. "
            "Required for covariant/mahalanobis mode to respond to --bandwidth."
        )
    )


def _add_dataset_arguments(subparser):
    subparser.add_argument(
        "--csv",
        default=None,
        help="CSV file to load (defaults to a synthetic dataset)."
    )

    subparser.add_argument(
        "--target",
        default=None,
        help="Target column name (defaults to the last column)."
    )

    subparser.add_argument(
        "--n-qubits",
        type=int,
        default=4,
        dest="n_qubits",
        help="Number of qubits / features."
    )

    subparser.add_argument(
        "--reps",
        type=int,
        default=2,
        help="Feature-map repetitions."
    )

    subparser.add_argument(
        "--n-samples",
        type=int,
        default=60,
        dest="n_samples",
        help="Samples for the synthetic dataset."
    )


def build_parser():
    parser = argparse.ArgumentParser(
        prog="qmlf",
        description="Command line interface for the qmlf quantum machine learning framework."
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"qmlf {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command")

    benchmark = subparsers.add_parser(
        "benchmark",
        help="Benchmark a quantum SVC against XGBoost on a dataset."
    )

    _add_dataset_arguments(benchmark)

    benchmark.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        dest="test_size",
        help="Test split fraction."
    )

    benchmark.set_defaults(func=_run_benchmark)

    visualize = subparsers.add_parser(
        "visualize",
        help="Plot the quantum kernel of a dataset in Hilbert space."
    )

    _add_dataset_arguments(visualize)
    _add_kernel_tuning_arguments(visualize)

    visualize.add_argument(
        "--mode",
        default="ZZ",
        choices=["ZZ", "covariant", "mahalanobis"],
        help="Quantum kernel mode ('mahalanobis' is the accurate alias of 'covariant')."
    )

    visualize.add_argument(
        "--batch-size",
        type=int,
        default=None,
        dest="batch_size",
        help="Evaluate the kernel in row-chunks of this size to bound peak memory."
    )

    visualize.add_argument(
        "--save",
        default=None,
        help="Write the figure to an HTML file instead of only showing it."
    )

    visualize.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the figure in a browser."
    )

    visualize.set_defaults(func=_run_visualize)

    pipeline = subparsers.add_parser(
        "pipeline",
        help="Run the kernel -> QNN -> visualization QuantumPipeline on a dataset."
    )

    _add_dataset_arguments(pipeline)
    _add_kernel_tuning_arguments(pipeline)

    pipeline.add_argument(
        "--mode",
        default="ZZ",
        choices=["ZZ", "covariant", "mahalanobis"],
        help="Quantum kernel mode ('mahalanobis' is the accurate alias of 'covariant')."
    )

    pipeline.add_argument(
        "--output-dim",
        type=int,
        default=16,
        dest="output_dim",
        help="QNN output dimension."
    )

    pipeline.add_argument(
        "--batch-size",
        type=int,
        default=None,
        dest="batch_size",
        help="Evaluate the kernel in row-chunks of this size to bound peak memory."
    )

    pipeline.add_argument(
        "--save",
        default=None,
        help="Write the figure to an HTML file instead of only showing it."
    )

    pipeline.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the figure in a browser."
    )

    pipeline.add_argument(
        "--no-visualize",
        action="store_true",
        dest="no_visualize",
        help="Skip the Hilbert-space plot and only report shapes."
    )

    pipeline.set_defaults(func=_run_pipeline)

    optimize_circuit = subparsers.add_parser(
        "optimize-circuit",
        help="Project depth/gate-count/fidelity gains from NISQ circuit resource optimization."
    )

    optimize_circuit.add_argument(
        "--depth",
        type=int,
        required=True,
        help="Original circuit depth."
    )

    optimize_circuit.add_argument(
        "--gate-count",
        type=int,
        required=True,
        dest="gate_count",
        help="Original total gate count."
    )

    optimize_circuit.add_argument(
        "--two-qubit-gates",
        type=int,
        required=True,
        dest="two_qubit_gates",
        help="Original two-qubit gate count."
    )

    optimize_circuit.add_argument(
        "--max-depth",
        type=int,
        default=16,
        dest="max_depth",
        help="Depth cap for the optimizer."
    )

    optimize_circuit.add_argument(
        "--optimization-strength",
        type=float,
        default=0.65,
        dest="optimization_strength",
        help="Optimization strength in [0, 1]."
    )

    optimize_circuit.set_defaults(func=_run_optimize_circuit)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
