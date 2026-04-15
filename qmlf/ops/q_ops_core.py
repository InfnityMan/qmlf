import numpy as np
import pandas as pd
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb
from qiskit_machine_learning.algorithms import QSVC

class QuantumKernel:
    fidelity_quantum_kernel = None
    feature_map = None
    kernel_matrix = None
    cov_matrix = None
    num_quibits = None
    mode = None
    reps = None

    def __init__(self, n_qubits, mode="ZZ", reps=2):
        self.n_qubits = n_qubits
        self.mode = mode
        self.reps = reps
        self.cov_matrix = None
        self.feature_map = None
        self.fidelity_quantum_kernel = None
        
        if mode == "ZZ" or mode == "covariant":
            self.feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=reps)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        self.fidelity_quantum_kernel = FidelityQuantumKernel(feature_map=self.feature_map)
    
    def fit(self, X_train, y=None):
        if self.mode == "covariant":
            X_np = np.asarray(X_train)
            self.cov_matrix = np.cov(X_np.T)

        return self
    
    def compute_kernel_matrix(self, X):
        X_np = np.asarray(X)
        
        if self.mode == "ZZ":
            kernel_matrix = self.fidelity_quantum_kernel.evaluate(X_np)
        elif self.mode == "covariant":
            if self.cov_matrix is None:
                raise ValueError("Must call .fit() before using covariant mode")
            variances = np.diag(self.cov_matrix)
            scaling = np.sqrt(variances + 1e-8)
            X_scaled = X_np / scaling
            kernel_matrix = self.fidelity_quantum_kernel.evaluate(X_scaled)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        return kernel_matrix

def run_quantum_benchmark(X, y, n_qubits=4, reps=2, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)

    quantum_kernel = QuantumKernel(n_qubits=n_qubits, mode="ZZ", reps=reps)
    quantum_kernel.fit(X_train)

    qsvc = QSVC(quantum_kernel=quantum_kernel.fidelity_quantum_kernel)
    qsvc.fit(X_train, y_train)
    y_pred_qsvc = qsvc.predict(X_test)

    results = {
        "Model": ["XGBoost", "Quantum SVC"],
        "Accuracy": [
            accuracy_score(y_test, y_pred_xgb),
            accuracy_score(y_test, y_pred_qsvc)
        ],
        "F1 Score": [
            f1_score(y_test, y_pred_xgb, average='weighted'),
            f1_score(y_test, y_pred_qsvc, average='weighted')
        ]
    }
    
    return pd.DataFrame(results)
