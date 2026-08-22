"""Contract tests for the NISQ optimizer and the graph-diffusion kernel.

Both modules previously had zero coverage.

Run with:  python -m pytest tests/ -q
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pytest
from qiskit.circuit.library import zz_feature_map

import qmlf

# --------------------------------------------------------------------------
# NISQ
# --------------------------------------------------------------------------

def test_estimate_mode_labels_itself():
    report = qmlf.create_advanced_nisq_optimizer().optimize_circuit(20, 40, 10)

    assert report["mode"] == "estimate"


def test_estimate_respects_caps():
    optimizer = qmlf.create_advanced_nisq_optimizer(
        max_depth=5, optimization_strength=0.9
    )
    report = optimizer.optimize_circuit(100, 200, 80)

    assert report["optimized_depth"] <= 5
    assert report["optimized_two_qubit_gates"] <= optimizer.max_two_qubit_gates


def test_fidelity_model_is_bounded_and_monotone():
    optimizer = qmlf.create_advanced_nisq_optimizer()

    shallow = optimizer.estimate_fidelity(5, 2)
    deep = optimizer.estimate_fidelity(50, 20)

    assert 0.0 < deep < shallow <= 1.0
    assert optimizer.apply_error_mitigation(deep) > deep
    assert optimizer.apply_error_mitigation(1.0) == 1.0


def test_transpile_mode_runs_a_real_transpiler():
    report = qmlf.create_advanced_nisq_optimizer() \
        .optimize_transpile(zz_feature_map(4, reps=2))

    assert report["mode"] == "transpiled"
    assert report["optimized_depth"] > 0
    assert report["optimized_gate_count"] > 0


def test_transform_dispatches_on_input_type():
    optimizer = qmlf.create_advanced_nisq_optimizer()

    assert optimizer.transform({"depth": 10})["mode"] == "estimate"
    assert optimizer.transform(zz_feature_map(3))["mode"] == "transpiled"
    assert isinstance(optimizer.transform(10), int)


def test_optimization_strength_maps_to_transpiler_level():
    assert qmlf.create_advanced_nisq_optimizer(
        optimization_strength=0.0).optimization_level == 0
    assert qmlf.create_advanced_nisq_optimizer(
        optimization_strength=1.0).optimization_level == 3


# --------------------------------------------------------------------------
# Graph
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def X():
    return np.random.default_rng(3).normal(size=(16, 4))


def test_classical_fit_produces_full_gram(X):
    kernel = qmlf.create_advanced_graph_kernel(n_qubits=4, n_neighbors=3).fit(X)
    gram = kernel.get_kernel_matrix()

    assert gram.shape == (16, 16)
    assert np.allclose(gram, gram.T, atol=1e-10)


def test_diffusion_operator_rows_are_stochastic_mixtures(X):
    kernel = qmlf.create_advanced_graph_kernel(n_qubits=4, n_neighbors=3).fit(X)
    diffusion = kernel.get_diffused_graph()

    # Average of I and k powers of a row-stochastic transition matrix: every
    # row still sums to 1.
    assert np.allclose(diffusion.sum(axis=1), 1.0, atol=1e-8)


def test_quantum_backend_agrees_in_shape_and_symmetry(X):
    kernel = qmlf.create_advanced_graph_kernel(
        n_qubits=4, n_neighbors=3, use_quantum=True
    ).fit(X)
    gram = kernel.get_kernel_matrix()

    assert gram.shape == (16, 16)
    assert np.allclose(gram, gram.T, atol=1e-8)


def test_transform_is_similarity_to_training_nodes(X):
    kernel = qmlf.create_advanced_graph_kernel(n_qubits=4, n_neighbors=3).fit(X)

    assert kernel.transform(X[:5]).shape == (5, 16)


def test_transform_before_fit_raises(X):
    with pytest.raises(ValueError, match="not been fitted"):
        qmlf.create_advanced_graph_kernel().transform(X)


def test_unknown_kernel_type_raises(X):
    kernel = qmlf.AdvancedQuantumGraphKernel(kernel_type="banana")

    with pytest.raises(ValueError, match="banana"):
        kernel.fit(X)


def test_too_many_neighbors_is_a_clear_error():
    tiny = np.random.default_rng(0).normal(size=(4, 3))

    with pytest.raises(ValueError, match="n_neighbors=5"):
        qmlf.create_advanced_graph_kernel(n_neighbors=5).fit(tiny)
