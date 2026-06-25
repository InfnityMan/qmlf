import numpy as np


class FederatedQML:
    """Federated averaging (FedAvg) for model parameter vectors.

    Aggregates per-client parameter vectors into a single global vector using a
    sample-size-weighted mean (the standard FedAvg update), and supports
    repeated aggregation rounds.
    """

    def __init__(
        self,
        num_clients=5
    ):
        self.num_clients = num_clients
        self.global_params = None
        self.round = 0
        self.history = []

    def aggregate(
        self,
        client_params_list,
        client_weights=None
    ):
        client_params = np.asarray(
            client_params_list,
            dtype=float
        )

        if client_params.ndim < 2:
            raise ValueError(
                "client_params_list must hold one parameter vector per client"
            )

        num_reporting = client_params.shape[0]

        if client_weights is None:
            weights = np.ones(num_reporting)
        else:
            weights = np.asarray(
                client_weights,
                dtype=float
            )

            if weights.shape[0] != num_reporting:
                raise ValueError(
                    "client_weights must have one weight per client"
                )

        weight_sum = weights.sum()

        if weight_sum <= 0:
            raise ValueError(
                "client_weights must sum to a positive value"
            )

        weights = weights / weight_sum

        # Sample-weighted FedAvg: weighted mean of the client parameter vectors.
        self.global_params = np.tensordot(
            weights,
            client_params,
            axes=(0, 0)
        )

        return self.global_params

    def fit(
        self,
        client_updates,
        client_weights=None
    ):
        global_params = self.aggregate(
            client_updates,
            client_weights=client_weights
        )

        self.round += 1
        self.history.append(global_params)

        return global_params

    def get_global_params(self):
        return self.global_params


def create_federated_qml(
    num_clients=5
):
    return FederatedQML(
        num_clients=num_clients
    )
