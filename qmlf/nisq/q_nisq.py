import numpy as np

from qiskit import transpile
from qiskit import QuantumCircuit


class AdvancedNISQOptimizer:
    """NISQ circuit resource analysis and real transpiler-based optimization.

    The integer/dict paths (``optimize_circuit`` and the ``transform`` fallback)
    return an analytic resource and fidelity PROJECTION from circuit statistics
    — a planning estimate, not a hardware-verified result. ``optimize_transpile``
    performs REAL Qiskit transpilation on an actual ``QuantumCircuit`` and reports
    the measured depth/gate reductions. All fidelity numbers are simple analytic
    models, not hardware measurements.
    """

    MITIGATION_FACTOR = 0.25

    def __init__(
        self,
        max_depth=16,
        optimization_strength=0.65,
        max_two_qubit_gates=50
    ):
        self.max_depth = max_depth
        self.optimization_strength = (
            optimization_strength
        )

        self.max_two_qubit_gates = (
            max_two_qubit_gates
        )

        # optimization_strength in [0, 1] maps to a transpiler level in [0, 3].
        self.optimization_level = max(
            0,
            min(
                3,
                int(round(optimization_strength * 3))
            )
        )

        self.original_stats = None
        self.optimized_stats = None

    def optimize_depth(
        self,
        original_depth
    ):
        """Projected depth under the configured optimization strength (estimate)."""
        optimized = int(
            original_depth
            * self.optimization_strength
        )

        return min(
            optimized,
            self.max_depth
        )

    def optimize_gate_count(
        self,
        gate_count
    ):
        """Projected gate count under the configured optimization strength (estimate)."""
        return max(
            1,
            int(
                gate_count
                * self.optimization_strength
            )
        )

    def optimize_two_qubit_gates(
        self,
        gate_count
    ):
        """Projected two-qubit gate count under the optimization strength (estimate)."""
        optimized = int(
            gate_count
            * self.optimization_strength
        )

        return min(
            optimized,
            self.max_two_qubit_gates
        )

    def estimate_fidelity(
        self,
        depth,
        two_qubit_gates,
        noise_rate=0.01
    ):
        """Simple analytic fidelity MODEL (not a hardware measurement).

        Assumes independent per-gate error: fidelity decays exponentially with
        circuit depth and (twice as steeply) with the two-qubit gate count.
        """
        error = (
            depth * noise_rate
            + two_qubit_gates
            * noise_rate
            * 2
        )

        fidelity = np.exp(
            -error
        )

        return float(fidelity)

    def apply_error_mitigation(
        self,
        fidelity
    ):
        """Optimistic heuristic model of error-mitigation gains.

        Closes a fixed fraction (``MITIGATION_FACTOR``) of the gap to unit
        fidelity. This is a modelling assumption, not a measured or guaranteed
        improvement.
        """
        improved = (
            fidelity
            + (1 - fidelity)
            * self.MITIGATION_FACTOR
        )

        return min(
            improved,
            1.0
        )

    def optimize_circuit(
        self,
        depth,
        gate_count,
        two_qubit_gates
    ):
        """Analytic resource + fidelity projection from circuit statistics.

        A planning estimate based on ``optimization_strength``; it does NOT
        inspect or transpile a real circuit. For genuine optimization pass a
        QuantumCircuit to ``optimize_transpile``.
        """
        optimized_depth = (
            self.optimize_depth(
                depth
            )
        )

        optimized_gate_count = (
            self.optimize_gate_count(
                gate_count
            )
        )

        optimized_two_qubit = (
            self.optimize_two_qubit_gates(
                two_qubit_gates
            )
        )

        before_fidelity = (
            self.estimate_fidelity(
                depth,
                two_qubit_gates
            )
        )

        after_fidelity = (
            self.estimate_fidelity(
                optimized_depth,
                optimized_two_qubit
            )
        )

        mitigated_fidelity = (
            self.apply_error_mitigation(
                after_fidelity
            )
        )

        return {
            "mode": "estimate",
            "original_depth": depth,
            "optimized_depth": optimized_depth,
            "original_gate_count": gate_count,
            "optimized_gate_count": optimized_gate_count,
            "original_two_qubit_gates": two_qubit_gates,
            "optimized_two_qubit_gates": optimized_two_qubit,
            "original_fidelity": before_fidelity,
            "optimized_fidelity": after_fidelity,
            "mitigated_fidelity": mitigated_fidelity
        }

    def _circuit_stats(
        self,
        circuit
    ):
        return {
            "depth": circuit.depth(),
            "gate_count": sum(
                circuit.count_ops().values()
            ),
            "two_qubit_gates": circuit.num_nonlocal_gates()
        }

    def optimize_transpile(
        self,
        circuit
    ):
        """Real Qiskit transpiler optimization of an actual QuantumCircuit."""
        before = self._circuit_stats(
            circuit
        )

        optimized = transpile(
            circuit,
            optimization_level=self.optimization_level
        )

        after = self._circuit_stats(
            optimized
        )

        before_fidelity = self.estimate_fidelity(
            before["depth"],
            before["two_qubit_gates"]
        )

        after_fidelity = self.estimate_fidelity(
            after["depth"],
            after["two_qubit_gates"]
        )

        mitigated_fidelity = self.apply_error_mitigation(
            after_fidelity
        )

        self.original_stats = before
        self.optimized_stats = after

        return {
            "mode": "transpiled",
            "original_depth": before["depth"],
            "optimized_depth": after["depth"],
            "original_gate_count": before["gate_count"],
            "optimized_gate_count": after["gate_count"],
            "original_two_qubit_gates": before["two_qubit_gates"],
            "optimized_two_qubit_gates": after["two_qubit_gates"],
            "original_fidelity": before_fidelity,
            "optimized_fidelity": after_fidelity,
            "mitigated_fidelity": mitigated_fidelity
        }

    def fit(
        self,
        circuit_stats
    ):
        if isinstance(
            circuit_stats,
            QuantumCircuit
        ):
            self.original_stats = self._circuit_stats(
                circuit_stats
            )

        else:
            self.original_stats = circuit_stats

        return self

    def transform(
        self,
        circuit_stats
    ):
        if isinstance(
            circuit_stats,
            QuantumCircuit
        ):
            return self.optimize_transpile(
                circuit_stats
            )

        if isinstance(
            circuit_stats,
            dict
        ):
            return self.optimize_circuit(
                depth=circuit_stats.get(
                    "depth",
                    10
                ),
                gate_count=circuit_stats.get(
                    "gate_count",
                    20
                ),
                two_qubit_gates=circuit_stats.get(
                    "two_qubit_gates",
                    5
                )
            )

        return self.optimize_depth(
            circuit_stats
        )

    def fit_transform(
        self,
        circuit_stats
    ):
        return self.fit(
            circuit_stats
        ).transform(
            circuit_stats
        )

    def get_resource_report(
        self,
        circuit_stats
    ):
        return self.transform(
            circuit_stats
        )


def create_advanced_nisq_optimizer(
    max_depth=16,
    optimization_strength=0.65
):
    return AdvancedNISQOptimizer(
        max_depth=max_depth,
        optimization_strength=optimization_strength
    )
