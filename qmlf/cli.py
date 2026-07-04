import argparse

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from qmlf import __version__
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
            raise ValueError(f"Target column '{target}' not found in {csv_path}")

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
    X, y = _load_dataset(args.csv, args.target, args.n_samples, args.n_qubits)

    results = run_quantum_benchmark(
        X,
        y,
        n_qubits=args.n_qubits,
        reps=args.reps,
        test_size=args.test_size
    )

    print(results.to_string(index=False))


def _run_visualize(args):
    X, y = _load_dataset(args.csv, args.target, args.n_samples, args.n_qubits)

    n_features = X.shape[1]
    kernel = QuantumKernel(n_qubits=n_features, mode=args.mode, reps=args.reps)
    kernel.fit(X)

    kernel_matrix = kernel.compute_kernel_matrix(X)

    QVizPro.plot_hilbert_space(
        kernel_matrix,
        labels=y,
        show=not args.no_show,
        save_path=args.save
    )

    if args.save is not None:
        print(f"Saved visualization to {args.save}")


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
    benchmark.add_argument("--csv", default=None, help="CSV file to load (defaults to a synthetic dataset).")
    benchmark.add_argument("--target", default=None, help="Target column name (defaults to the last column).")
    benchmark.add_argument("--n-qubits", type=int, default=4, dest="n_qubits", help="Number of qubits / features.")
    benchmark.add_argument("--reps", type=int, default=2, help="Feature-map repetitions.")
    benchmark.add_argument("--test-size", type=float, default=0.2, dest="test_size", help="Test split fraction.")
    benchmark.add_argument("--n-samples", type=int, default=60, dest="n_samples", help="Samples for the synthetic dataset.")
    benchmark.set_defaults(func=_run_benchmark)

    visualize = subparsers.add_parser(
        "visualize",
        help="Plot the quantum kernel of a dataset in Hilbert space."
    )
    visualize.add_argument("--csv", default=None, help="CSV file to load (defaults to a synthetic dataset).")
    visualize.add_argument("--target", default=None, help="Target column name (defaults to the last column).")
    visualize.add_argument("--n-qubits", type=int, default=4, dest="n_qubits", help="Number of qubits / features.")
    visualize.add_argument("--reps", type=int, default=2, help="Feature-map repetitions.")
    visualize.add_argument("--mode", default="ZZ", choices=["ZZ", "covariant"], help="Quantum kernel mode.")
    visualize.add_argument("--n-samples", type=int, default=60, dest="n_samples", help="Samples for the synthetic dataset.")
    visualize.add_argument("--save", default=None, help="Write the figure to an HTML file instead of only showing it.")
    visualize.add_argument("--no-show", action="store_true", help="Do not open the figure in a browser.")
    visualize.set_defaults(func=_run_visualize)

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
