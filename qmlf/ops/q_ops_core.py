import numpy as np
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel

class QuantumKernel:
    fidelity_quantum_kernel = None
    feature_map = None
    kernel_matrix = None
    cov_matrix = None
    num_quibits = None
    mode = None
    reps = None

    def __init__(self, num_quibits, mode, reps):
        self.num_quibits = num_quibits;
        self.mode = mode
        self.reps = reps
        self.cov_matrix = None
        
        if mode == "ZZ":
            self.feature_map = ZZFeatureMap(num_quibits, reps=reps)
        elif mode == "covariant":
            # TODO: Implement covariant feature map later
            self.feature_map = ZZFeatureMap(num_quibits, reps=reps)

        self.fidelity_quantum_kernel = FidelityQuantumKernel(feature_map)
    
    def fit(self, X_train, y=None):
        if self.mode == "covariant":
            X_np = np.array(X_train)
            self.cov_matrix = np.cov(X_np.T)

        return self
    
    def compute_kernel_matrix(self, X):
        if(self.mode == "ZZ"):
            kernel_matrix = self.fidelity_quantum_kernel.evaluate(X)
        elif(self.mode == "covariant"):
            variances = np.diag(self.cov_matrix)

            scaling = np.sqrt(variances + 1e-8)
            X_scaled = X_np / scaling

            kernel_matrix = self.fidelity_quantum_kernel.evaluate(X_scaled)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        return self.kernel_matrix
