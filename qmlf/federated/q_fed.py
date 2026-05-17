import numpy as np

class FederatedQML:
    def __init__(self, num_clients=5):
        self.num_clients = num_clients
        self.global_params = None

    def aggregate(self, client_params_list):
        client_params_list = np.asarray(client_params_list)
        self.global_params = np.mean(client_params_list, axis=0)
        return self.global_params

    def fit(self, client_updates):
        return self.aggregate(client_updates)


def create_federated_qml(num_clients=5):
    return FederatedQML(num_clients=num_clients)