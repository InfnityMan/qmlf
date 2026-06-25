import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier


class AdvancedQIGASelector(BaseEstimator, TransformerMixin):
    """Quantum-Inspired Genetic Algorithm feature selector.

    A genuine quantum-inspired evolutionary algorithm (Han & Kim): each feature
    is a qubit whose selection probability is ``sin(theta) ** 2``. Every
    generation measures (collapses) the qubit chromosomes into binary feature
    masks, scores them with a RandomForest cross-validation, and updates the
    angles with a quantum rotation gate toward the best-so-far solution. Cost is
    population_size * generations * cv RandomForest fits.
    """

    def __init__(
        self,
        n_features=12,
        population_size=100,
        generations=50,
        mutation_rate=0.18,
        elite_fraction=0.1,
        scoring="accuracy"
    ):
        self.n_features = n_features
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_fraction = elite_fraction
        self.scoring = scoring

        self.selected_features = None
        self.feature_importances_ = None
        self.best_score = None
        self.convergence_history = []

    def _initialize_angles(self, n_features):
        # Equal superposition: P(selected) = sin(pi/4)**2 = 0.5 for every qubit.
        angles = np.full(
            (self.population_size, n_features),
            np.pi / 4
        )

        return angles

    def _measure(self, angles):
        probabilities = np.sin(angles) ** 2
        draws = np.random.rand(*angles.shape)
        masks = draws < probabilities

        for i in range(self.population_size):
            if np.sum(masks[i]) == 0:
                masks[i, np.argmax(probabilities[i])] = True

        return masks

    def _evaluate_population(self, masks, X, y):
        fitness = np.zeros(self.population_size)

        for i in range(self.population_size):
            mask = masks[i]

            X_selected = X[:, mask]

            try:
                model = RandomForestClassifier(
                    n_estimators=50,
                    random_state=42,
                    n_jobs=-1
                )

                score = cross_val_score(
                    model,
                    X_selected,
                    y,
                    cv=3,
                    scoring=self.scoring
                ).mean()

                fitness[i] = score

            except Exception:
                fitness[i] = 0.0

        return fitness

    def _rotate(self, angles, best_mask):
        # Quantum rotation gate: rotate each qubit toward the best-so-far bit.
        delta = 0.05 * np.pi
        direction = np.where(best_mask, 1.0, -1.0)
        angles = angles + delta * direction
        angles = np.clip(angles, 0.0, np.pi / 2)

        return angles

    def _mutate(self, angles):
        mutation_mask = (
            np.random.rand(*angles.shape)
            < self.mutation_rate
        )

        noise = np.random.normal(
            0,
            0.05,
            size=angles.shape
        )

        angles = angles + mutation_mask * noise
        angles = np.clip(angles, 0.0, np.pi / 2)

        return angles

    def fit(self, X, y):
        X_np = np.asarray(X)

        if y is None:
            raise ValueError(
                "y cannot be None for feature selection"
            )

        n_samples, n_features = X_np.shape

        angles = self._initialize_angles(
            n_features
        )

        best_score = -np.inf
        best_mask = None

        for generation in range(self.generations):
            masks = self._measure(
                angles
            )

            fitness = self._evaluate_population(
                masks,
                X_np,
                y
            )

            generation_best_idx = np.argmax(fitness)
            generation_best_score = fitness[generation_best_idx]

            if generation_best_score > best_score:
                best_score = generation_best_score
                best_mask = masks[generation_best_idx].copy()

            self.convergence_history.append(
                generation_best_score
            )

            angles = self._rotate(
                angles,
                best_mask
            )

            angles = self._mutate(
                angles
            )

        self.selected_features = best_mask
        self.best_score = best_score

        # Graded importance: final per-feature selection probability.
        self.feature_importances_ = np.mean(
            np.sin(angles) ** 2,
            axis=0
        )

        return self

    def transform(self, X):
        if self.selected_features is None:
            raise ValueError(
                "Selector has not been fitted yet"
            )

        X_np = np.asarray(X)

        return X_np[:, self.selected_features]

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_support(self):
        if self.selected_features is None:
            raise ValueError(
                "Selector has not been fitted yet"
            )

        return self.selected_features

    def get_selected_indices(self):
        if self.selected_features is None:
            raise ValueError(
                "Selector has not been fitted yet"
            )

        return np.where(
            self.selected_features
        )[0]

    def get_feature_ranking(self):
        if self.feature_importances_ is None:
            raise ValueError(
                "Selector has not been fitted yet"
            )

        return np.argsort(
            self.feature_importances_
        )[::-1]


def create_advanced_qiga_selector(
    n_features=12,
    population_size=100,
    generations=50,
    mutation_rate=0.18
):
    return AdvancedQIGASelector(
        n_features=n_features,
        population_size=population_size,
        generations=generations,
        mutation_rate=mutation_rate
    )
