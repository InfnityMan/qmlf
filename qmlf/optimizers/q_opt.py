import warnings

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Distinguishes "caller did not pass n_features" from "caller explicitly asked
# for 12", so the no-op can be reported only to the people it misleads.
_UNSET = object()


class AdvancedQIGASelector(BaseEstimator, TransformerMixin):
    """Quantum-Inspired Genetic Algorithm feature selector.

    A genuine quantum-inspired evolutionary algorithm (Han & Kim): each feature
    is a qubit whose selection probability is ``sin(theta) ** 2``. Every
    generation measures (collapses) the qubit chromosomes into binary feature
    masks, scores them with a RandomForest cross-validation, and updates the
    angles with a quantum rotation gate toward the best-so-far solution. Cost is
    population_size * generations * cv RandomForest fits.

    NOTE ON ``n_features``: this does NOT request a subset of that size. The
    number of features is taken from ``X`` in :meth:`fit`, and how many get
    selected is whatever scores best -- asking for 2 on 6-column data has been
    observed selecting 5. The parameter is retained for backwards compatibility
    and warns when set explicitly. To bound the subset size, filter
    :meth:`get_feature_ranking` yourself.
    """

    def __init__(
        self,
        n_features=_UNSET,
        population_size=100,
        generations=50,
        mutation_rate=0.18,
        elite_fraction=0.1,
        scoring="accuracy",
        random_state=None
    ):
        if n_features is not _UNSET:
            warnings.warn(
                f"n_features={n_features} has no effect: the feature count is "
                "read from X in fit(), and the number selected is whatever "
                "scores best. Slice get_feature_ranking() to bound the subset "
                "size.",
                UserWarning,
                stacklevel=2
            )

        self.n_features = 12 if n_features is _UNSET else n_features
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_fraction = elite_fraction
        self.scoring = scoring
        # None keeps the historical behaviour (module-level np.random, seedable
        # only globally); an int makes this fit self-contained and reproducible
        # without touching global state.
        self.random_state = random_state

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

        if self._rng is not None:
            draws = self._rng.random(angles.shape)
        else:
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
        if self._rng is not None:
            mutation_mask = self._rng.random(angles.shape) < self.mutation_rate
        else:
            mutation_mask = (
                np.random.rand(*angles.shape)
                < self.mutation_rate
            )

        if self._rng is not None:
            noise = self._rng.normal(0, 0.05, size=angles.shape)
        else:
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

        # Derived here, per sklearn convention, so two fits of the same seeded
        # selector give the same answer instead of continuing one stream.
        self._rng = (
            np.random.default_rng(self.random_state)
            if self.random_state is not None
            else None
        )

        # Reset rather than append-forever: refitting used to keep the previous
        # fit's curve, so the history silently mixed two different datasets.
        self.convergence_history = []

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
    n_features=_UNSET,
    population_size=100,
    generations=50,
    mutation_rate=0.18,
    random_state=None
):
    return AdvancedQIGASelector(
        n_features=n_features,
        population_size=population_size,
        generations=generations,
        mutation_rate=mutation_rate,
        random_state=random_state
    )
