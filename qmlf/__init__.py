__version__ = "1.2.0"

from .ops.q_ops_core import QuantumKernel, run_quantum_benchmark, plot_hilbert_space

# Everything below is resolved lazily (PEP 562). The QNN, hybrid, transformer,
# chemistry and pipeline modules all import torch, and pulling torch in just to
# use QuantumKernel is both slow and, on macOS, dangerous: torch and
# scikit-learn/xgboost each vendor their own libomp.dylib, and importing torch
# first can segfault the interpreter with no traceback. Resolving these on first
# attribute access keeps `import qmlf` cheap and torch-free until it is needed.
#
# `from qmlf import QVizPro` and `qmlf.QVizPro` both still work exactly as
# before -- only the import *timing* changes.
_LAZY_ATTRS = {
    "AdvancedIBIInitializer": ".qnn.qnn_layers",
    "AdvancedQuantumNNLayer": ".qnn.qnn_layers",
    "create_advanced_qnn_layer": ".qnn.qnn_layers",
    "QVizPro": ".viz.q_viz_pro",
    "AdvancedQIGASelector": ".optimizers.q_opt",
    "create_advanced_qiga_selector": ".optimizers.q_opt",
    "AdvancedNoiseMitigator": ".noise.q_noise",
    "create_advanced_noise_mitigator": ".noise.q_noise",
    "AdvancedHybridLayer": ".hybrid.q_hybrid",
    "create_advanced_hybrid_layer": ".hybrid.q_hybrid",
    "AdvancedQuantumGraphKernel": ".graph.q_graph",
    "create_advanced_graph_kernel": ".graph.q_graph",
    "FederatedQML": ".federated.q_fed",
    "create_federated_qml": ".federated.q_fed",
    "AdvancedQuantumTransformerLayer": ".transformer.q_transformer",
    "create_advanced_quantum_transformer": ".transformer.q_transformer",
    "AdvancedQuantumChemistryLayer": ".chem.q_chem",
    "create_advanced_chem_layer": ".chem.q_chem",
    "AdvancedNISQOptimizer": ".nisq.q_nisq",
    "create_advanced_nisq_optimizer": ".nisq.q_nisq",
    "QMLFPipeline": ".integration.q_integration",
    "create_full_pipeline": ".integration.q_integration",
    "QuantumPipeline": ".integration.q_pipeline",
    "create_quantum_pipeline": ".integration.q_pipeline",
}

__all__ = [
    "QuantumKernel",
    "run_quantum_benchmark",
    "plot_hilbert_space",
    *sorted(_LAZY_ATTRS),
]


def __getattr__(name):
    module_path = _LAZY_ATTRS.get(name)

    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    value = getattr(import_module(module_path, __name__), name)
    # Cache on the module so later lookups skip this path entirely.
    globals()[name] = value

    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_ATTRS))
