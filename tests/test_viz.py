"""Headless smoke tests for every public QVizPro plot.

The viz module was the largest untested surface in the package (966 lines,
zero tests). These do not judge aesthetics; they pin the contract that matters
for automation: every plot builds a Figure with show=False, honours save_path,
and the module-level wrappers forward keyword arguments instead of dropping
them.

Run with:  python -m pytest tests/ -q
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pytest
import plotly.graph_objects as go

import qmlf.viz.q_viz_pro as viz
from qmlf.viz.q_viz_pro import QVizPro

RNG = np.random.default_rng(0)
GRAM = (lambda a: a @ a.T / a.shape[1])(RNG.normal(size=(10, 4)))
GRADS = RNG.normal(scale=0.1, size=(6, 12))


def _is_fig(obj):
    assert isinstance(obj, go.Figure)
    return True


def test_plot_hilbert_space_tsne():
    _is_fig(QVizPro.plot_hilbert_space(GRAM, labels=np.arange(10) % 2, show=False))


def test_plot_hilbert_space_pca():
    _is_fig(QVizPro.plot_hilbert_space(GRAM, method="pca", show=False))


def test_plot_hilbert_space_tiny_input_falls_back():
    _is_fig(QVizPro.plot_hilbert_space(np.eye(3), show=False))


def test_plot_hilbert_space_rejects_1d():
    with pytest.raises(ValueError, match="2D"):
        QVizPro.plot_hilbert_space(np.arange(4.0), show=False)


def test_plot_kernel_matrix():
    _is_fig(QVizPro.plot_kernel_matrix(GRAM, show=False))


def test_plot_kernel_matrix_rejects_rectangular():
    with pytest.raises(ValueError, match="square"):
        QVizPro.plot_kernel_matrix(GRAM[:4], show=False)


def test_plot_kernel_eigenvalues():
    _is_fig(QVizPro.plot_kernel_eigenvalues(GRAM, show=False))


def test_gradient_heatmap():
    _is_fig(QVizPro.gradient_heatmap(GRADS, show=False))


def test_plot_gradient_variance():
    _is_fig(QVizPro.plot_gradient_variance(GRADS, show=False))


def test_plot_gradient_distribution():
    _is_fig(QVizPro.plot_gradient_distribution(GRADS, show=False))


def test_plot_barren_plateau():
    _is_fig(QVizPro.plot_barren_plateau(
        [0.1, 0.03, 0.008, 0.001], depths=[1, 2, 3, 4], show=False
    ))


def test_plot_barren_plateau_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        QVizPro.plot_barren_plateau([0.1, 0.03], depths=[1, 2, 3], show=False)


def test_plot_bloch_sphere_from_amplitudes():
    states = np.array([[1, 0], [0, 1], [1 / np.sqrt(2), 1j / np.sqrt(2)]])
    _is_fig(QVizPro.plot_bloch_sphere(states, show=False))


def test_plot_bloch_sphere_empty():
    _is_fig(QVizPro.plot_bloch_sphere(show=False))


def test_plot_parameter_landscape():
    _is_fig(QVizPro.plot_parameter_landscape(RNG.normal(size=(7, 9)), show=False))


def test_plot_training_history_full():
    _is_fig(QVizPro.plot_training_history(
        [1.0, 0.6, 0.4], accuracies=[0.5, 0.7, 0.8],
        val_losses=[1.1, 0.7, 0.5], val_accuracies=[0.4, 0.6, 0.75],
        show=False
    ))


def test_plot_metric_comparison():
    _is_fig(QVizPro.plot_metric_comparison(
        {"XGBoost": {"acc": 0.9, "f1": 0.88}, "QSVC": {"acc": 0.8, "f1": 0.79}},
        show=False
    ))


def test_plot_confusion_matrix():
    _is_fig(QVizPro.plot_confusion_matrix(
        [0, 1, 1, 0, 1], [0, 1, 0, 0, 1], normalize=True, show=False
    ))


def test_plot_noise_comparison():
    _is_fig(QVizPro.plot_noise_comparison(
        [0.9, 0.8], [0.7, 0.6], mitigated_values=[0.85, 0.75], show=False
    ))


def test_plot_circuit_depths():
    _is_fig(QVizPro.plot_circuit_depths(["a", "b"], [12, 7], show=False))


def test_plot_probability_distribution():
    _is_fig(QVizPro.plot_probability_distribution(
        np.array([0.5, 0.25, 0.125, 0.125]), show=False
    ))


def test_plot_research_dashboard_with_kernel():
    _is_fig(QVizPro.plot_research_dashboard(
        [1.0, 0.5], GRADS, kernel_matrix=GRAM, show=False
    ))


def test_plot_research_dashboard_without_kernel():
    _is_fig(QVizPro.plot_research_dashboard([1.0, 0.5], GRADS, show=False))


def test_save_path_writes_html(tmp_path):
    target = tmp_path / "gram.html"
    QVizPro.plot_kernel_matrix(GRAM, show=False, save_path=str(target))

    assert target.exists() and target.stat().st_size > 0


# ---- module-level wrappers must forward kwargs, not drop them -----------

def test_wrapper_plot_hilbert_space_forwards_kwargs(tmp_path):
    target = tmp_path / "w.html"
    _is_fig(viz.plot_hilbert_space(GRAM, show=False, save_path=str(target)))
    assert target.exists()


def test_wrapper_gradient_heatmap_forwards_kwargs():
    _is_fig(viz.gradient_heatmap(GRADS, show=False))


def test_wrapper_plot_bloch_sphere_forwards_kwargs():
    _is_fig(viz.plot_bloch_sphere(show=False))


def test_wrapper_plot_parameter_landscape_forwards_kwargs():
    _is_fig(viz.plot_parameter_landscape(RNG.normal(size=(5, 5)), show=False))
