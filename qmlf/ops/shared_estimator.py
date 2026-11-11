"""An estimator that simulates each parameter set once, not once per observable.

Qiskit's :class:`~qiskit.primitives.StatevectorEstimator` loops over the
broadcast of (parameter sets x observables) and builds a fresh statevector for
every pair. The quantum layers here read one Z expectation per qubit off the
*same* state, so an 8-qubit layer simulated every sample eight times on every
forward pass -- and, because parameter-shift gradients go through the estimator
too, roughly ``2 * n_parameters * n_observables`` times on every backward pass.

Sharing the state across the observables of a pub is exact: the observables do
not touch the circuit. Measured against qmlf 1.4.0's layers (stock estimator)
on a Kaggle CPU: one forward pass of the default 8-qubit layer on a batch of 4
went from 32 simulations to 4, forward 6.4x and parameter-shift backward 6.9x
faster; at 4 qubits 2.8x and 3.4x. Outputs and gradients bit-identical.

The shortcut is skipped -- falling back to the stock implementation -- whenever
it would not be exact:

* ``precision != 0`` (shot emulation), where each pair draws its own Gaussian
  noise from a shared generator, so the number of draws and their order are
  part of the result;
* circuits containing ``reset`` or ``measure``, whose simulation is itself
  stochastic and consumes that generator;
* a Qiskit build that does not expose the statevector helper the stock
  estimator uses, since matching it bit-for-bit is the point.
"""
import numpy as np
from qiskit.circuit import Barrier, Delay, Gate
from qiskit.primitives import StatevectorEstimator
from qiskit.primitives.containers import DataBin, PrimitiveResult, PubResult
from qiskit.quantum_info import SparsePauliOp

try:
    from qiskit.primitives.utils import _statevector_from_circuit
except ImportError:  # pragma: no cover - only on a Qiskit that moved the helper
    _statevector_from_circuit = None


def _is_deterministic(circuit):
    """True only when simulating this circuit cannot consume randomness.

    Checked by TYPE rather than by name, and so it also catches what a name
    check misses: a reset hidden inside a composite instruction's definition,
    which every ``initialize`` on a subset of qubits contains. A Gate is
    unitary by contract, whatever it is built from; Reset, Measure, Initialize
    and control-flow ops are Instructions, not Gates.
    """
    return all(
        isinstance(instruction.operation, (Gate, Barrier, Delay))
        for instruction in circuit.data
    )


class SharedStateEstimator(StatevectorEstimator):
    """Drop-in :class:`StatevectorEstimator` that reuses simulations.

    Identical results, fewer simulations. See the module docstring for the
    cases where it defers to the stock implementation.
    """

    def _run(self, pubs):
        if _statevector_from_circuit is None or not all(
            pub.precision == 0 and _is_deterministic(pub.circuit) for pub in pubs
        ):
            return super()._run(pubs)

        # Keyed by the parameter values themselves, so the same sample shared
        # between two pubs (the usual case: one pub per observable) is
        # simulated once for all of them.
        states = {}
        # A QNN layer has one observable per qubit and they repeat across every
        # row of every pub, so rebuilding the operator per element is thousands
        # of identical constructions per backward pass.
        operators = {}
        results = []

        for pub in pubs:
            num_parameters = pub.circuit.num_parameters
            # Flattened the same way as `bound`, so row `flat` is the parameter
            # set of circuit `flat`. A parameter-free circuit has one state.
            values = (
                pub.parameter_values.as_array(pub.circuit.parameters)
                .reshape(-1, num_parameters)
                if num_parameters
                else None
            )
            count = 1 if values is None else len(values)
            indices = np.arange(count).reshape(pub.parameter_values.shape)
            broadcast_indices, broadcast_observables = np.broadcast_arrays(
                indices, pub.observables
            )
            evs = np.zeros(broadcast_indices.shape, dtype=np.float64)
            rng = np.random.default_rng(self._seed)

            for index in np.ndindex(*broadcast_indices.shape):
                flat = int(broadcast_indices[index])
                key = (id(pub.circuit),
                       values[flat].tobytes() if values is not None else b"")
                state = states.get(key)

                if state is None:
                    # Bound on the miss, not up front: with one pub per
                    # observable, binding every row of every pub eagerly built
                    # (observables - 1) / observables of the circuits purely to
                    # throw them away.
                    circuit = (
                        pub.circuit if values is None
                        else pub.circuit.assign_parameters(values[flat])
                    )
                    state = states[key] = _statevector_from_circuit(circuit, rng)

                items = tuple(broadcast_observables[index].items())
                operator = operators.get(items)

                if operator is None:
                    paulis, coeffs = zip(*items)
                    operator = operators[items] = SparsePauliOp(paulis, coeffs)

                evs[index] = np.real_if_close(state.expectation_value(operator))

            results.append(
                PubResult(
                    DataBin(evs=evs, stds=np.zeros_like(evs), shape=evs.shape),
                    metadata={
                        "target_precision": pub.precision,
                        "circuit_metadata": pub.circuit.metadata,
                    },
                )
            )

        return PrimitiveResult(results, metadata={"version": 2})
