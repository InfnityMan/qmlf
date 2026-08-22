"""Reproducibility tests for the QNN-backed layers and the pipeline.

These guard the fix for the silent nondeterminism that made every
EstimatorQNN-backed module unusable for anything that asserts on a value.
qiskit-machine-learning defaults EstimatorQNN to default_precision=0.015625,
which samples the expectation instead of reading it exactly off the
statevector. On a circuit whose exact expectation was 0.0322, five successive
forward passes spanned 0.0374 -- drift larger than the signal, and training
that would not reproduce under a fixed seed.

The layers now default to precision=0.0 (exact) and take the old sampling
behaviour as an explicit opt-in.

Run with:  python -m pytest tests/ -q
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pytest

import qmlf
from qmlf.qnn.qnn_layers import AdvancedQuantumNNLayer

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def X():
    return np.random.default_rng(0).uniform(-1.0, 1.0, size=(6, 4))


@pytest.fixture(scope="module")
def X_torch(X):
    return torch.tensor(X, dtype=torch.float32)


# --------------------------------------------------------------------------
# Exactness
# --------------------------------------------------------------------------

def test_qnn_forward_is_exact_by_default(X_torch):
    """The default read-out must equal the statevector expectation, not sample it."""
    from qiskit.quantum_info import Statevector

    layer = AdvancedQuantumNNLayer(n_qubits=4, reps=2, output_dim=4)
    row = X_torch.numpy()[:1]

    got = layer.qnn.forward(row, layer.initial_weights)[0]

    bound = layer.circuit.assign_parameters(
        np.concatenate([row[0], layer.initial_weights])
    )
    state = Statevector(bound)
    expected = [state.expectation_value(obs).real for obs in layer.observables]

    assert np.allclose(got, expected, atol=1e-9)


def test_qnn_forward_repeats_exactly(X_torch):
    layer = AdvancedQuantumNNLayer(n_qubits=4, reps=2, output_dim=4)
    row = X_torch.numpy()[:1]

    values = [
        float(layer.qnn.forward(row, layer.initial_weights)[0, 0])
        for _ in range(5)
    ]

    assert max(values) - min(values) == 0.0


def test_training_reproduces_under_a_fixed_seed(X_torch):
    """Two identically seeded runs must produce identical losses."""
    from torch import nn

    y = torch.tensor(
        np.random.default_rng(1).integers(0, 2, size=X_torch.shape[0]),
        dtype=torch.long
    )
    loss_fn = nn.CrossEntropyLoss()

    def run():
        torch.manual_seed(7)
        np.random.seed(7)

        layer = qmlf.create_advanced_qnn_layer(n_qubits=4, reps=2, output_dim=2)
        layer.eval()
        optimizer = torch.optim.Adam(layer.parameters(), lr=0.05)

        losses = []

        for _ in range(3):
            optimizer.zero_grad()
            loss = loss_fn(layer(X_torch), y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))

        return losses

    assert run() == run()


# --------------------------------------------------------------------------
# The old sampling behaviour is still reachable, just no longer the default
# --------------------------------------------------------------------------

def test_positive_precision_still_samples(X_torch):
    layer = AdvancedQuantumNNLayer(
        n_qubits=4, reps=2, output_dim=4, precision=0.015625
    )
    row = X_torch.numpy()[:1]

    values = [
        float(layer.qnn.forward(row, layer.initial_weights)[0, 0])
        for _ in range(5)
    ]

    assert max(values) - min(values) > 0.0


def test_negative_precision_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        AdvancedQuantumNNLayer(n_qubits=4, precision=-0.1)


@pytest.mark.parametrize("factory,kwargs", [
    (qmlf.create_advanced_qnn_layer, {"n_qubits": 4, "reps": 1, "output_dim": 4}),
    (qmlf.create_advanced_hybrid_layer,
     {"input_dim": 4, "n_qubits": 4, "reps": 1, "output_dim": 4}),
    (qmlf.create_advanced_quantum_transformer,
     {"n_qubits": 4, "heads": 2, "reps": 1, "embed_dim": 8}),
])
def test_every_qnn_layer_defaults_to_exact(factory, kwargs):
    assert factory(**kwargs).precision == 0.0


# --------------------------------------------------------------------------
# Inference mode: no_grad() is not eval()
# --------------------------------------------------------------------------

def test_pipeline_run_is_reproducible(X):
    """QuantumPipeline.run() evaluates under no_grad(), which does not disable
    dropout -- without .eval() the QNN read-out differed on every call."""
    pipeline = qmlf.create_quantum_pipeline(n_qubits=4, reps=1, output_dim=4)

    first = pipeline.run(X, visualize=False)
    second = pipeline.run(X, visualize=False)

    assert np.allclose(first["qnn_output"], second["qnn_output"])
    assert np.allclose(first["kernel_matrix"], second["kernel_matrix"])


def test_pipeline_run_restores_the_layers_training_mode(X):
    """A layer the caller is mid-training must be handed back untouched."""
    pipeline = qmlf.create_quantum_pipeline(n_qubits=4, reps=1, output_dim=4)

    pipeline.run(X, visualize=False)
    assert pipeline.qnn_layer.training is True

    pipeline.qnn_layer.eval()
    pipeline.run(X, visualize=False)
    assert pipeline.qnn_layer.training is False


# --------------------------------------------------------------------------
# The kernel path was already exact; keep it that way
# --------------------------------------------------------------------------

def test_kernel_matrix_is_bit_identical_across_rebuilds(X):
    def gram():
        return qmlf.QuantumKernel(4, mode="ZZ", bandwidth=0.5).fit(X) \
            .compute_kernel_matrix(X)

    assert gram().tobytes() == gram().tobytes()
