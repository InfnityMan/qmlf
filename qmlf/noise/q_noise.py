import numpy as np

class NoiseAwareTrainer:
    def __init__(self, noise_level=0.01):
        self.noise_level = noise_level
        self.mitigation_factor = 1.0

    def apply_error_mitigation(self, probabilities):
        probabilities = np.asarray(probabilities)
        # Simple error mitigation (amplitude damping style)
        mitigated = probabilities * (1 - self.noise_level)
        return mitigated / (mitigated.sum() + 1e-8)

    def fit(self, circuit_output):
        return self.apply_error_mitigation(circuit_output)


def create_noise_aware_trainer(noise_level=0.01):
    return NoiseAwareTrainer(noise_level=noise_level)