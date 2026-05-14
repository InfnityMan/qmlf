import numpy as np
import torch
from torch import nn
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator

class IBIInitializer:
    def __init__(self, num_qubits, reps=2):
        self.num_qubits = num_qubits;
        self.reps = reps;

    def initialize_parameters(self, num_params):
        params = np.zeros(num_params);
        params += np.random.normal(0, 0.05, num_params);
        return params;


class QuantumNNLayer(nn.Module):
    def __init__(self, n_qubits=4, reps=2):
        super().__init__();
        self.n_qubits = n_qubits;
        self.reps = reps;
        
        self.circuit = RealAmplitudes(num_qubits=n_qubits, reps=reps);
        self.params = ParameterVector('θ', length=len(self.circuit.parameters));
        
        self.estimator = Estimator();
        self.qnn = EstimatorQNN(
            circuit=self.circuit,
            estimator=self.estimator,
            input_params=[],
            weight_params=self.params
        );

    def forward(self, x):
        batch_size = x.shape[0];
        return torch.zeros(batch_size, self.n_qubits, device=x.device);


def create_quantum_layer(n_qubits=4, reps=2):
    layer = QuantumNNLayer(n_qubits, reps);
    return layer;