import warnings

import numpy as np

_VALID_STRATEGIES = ("readout", "depolarizing", "zne")


class AdvancedNoiseMitigator:
    """Readout, depolarizing and zero-noise-extrapolation mitigation.

    ``strategy`` selects the method and is validated at construction: an
    unrecognised value now raises instead of silently returning the input
    unmitigated, which made a mis-cased ``"ZNE"`` indistinguishable from a
    successful mitigation.

    Input shapes differ by strategy, because the strategies need different data:

    - ``readout`` / ``depolarizing`` mitigate a single distribution. They accept
      a 1D ``(n_states,)`` vector or a 2D ``(n_rows, n_states)`` stack.
    - ``zne`` extrapolates *across noise scales*, so it needs at least two
      measurements: a 2D ``(n_scales, n_states)`` stack whose rows are the
      distributions observed at increasing noise. A 1D input is an error rather
      than something to guess at.
    """

    def __init__(
        self,
        noise_level=0.015,
        strategy="readout"
    ):
        if strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"Unknown strategy: {strategy!r} (expected one of "
                f"{_VALID_STRATEGIES}). Note these are lower-case."
            )

        if not 0.0 <= noise_level < 1.0:
            raise ValueError(
                f"noise_level must be in [0, 1), got {noise_level}. The "
                "depolarizing inverse divides by (1 - noise_level)."
            )

        self.noise_level = noise_level
        self.strategy = strategy

        self.calibration_matrix = None
        self.inverse_calibration_matrix = None
        self.scale_factors = None

    def _normalize(self, probabilities):
        probabilities = np.maximum(
            probabilities,
            0
        )

        row_sums = (
            probabilities.sum(
                axis=-1,
                keepdims=True
            ) + 1e-12
        )

        return probabilities / row_sums

    def _readout_mitigation(self, probabilities):
        if self.inverse_calibration_matrix is None:
            # Returning the input unchanged looks exactly like a successful
            # mitigation from the caller's side, so say what happened.
            warnings.warn(
                "readout mitigation was requested but no calibration matrix "
                "has been fitted, so the input is being returned unmitigated. "
                "Call .fit(confusion_matrix) first.",
                UserWarning,
                stacklevel=3
            )
            return probabilities

        mitigated = (
            probabilities
            @ self.inverse_calibration_matrix.T
        )

        return self._normalize(mitigated)

    def _depolarizing_mitigation(self, probabilities):
        """Invert a global depolarizing channel exactly.

        A global depolarizing channel produces the noisy distribution
        ``q = (1 - p) * ideal + p * uniform`` where ``p`` is the depolarizing
        probability (``self.noise_level``) and ``uniform`` is the maximally
        mixed distribution ``1 / n_states``. The exact inverse recovers
        ``ideal = (q - p * uniform) / (1 - p)``. The recovered distribution is
        then normalized to remain a valid probability vector.
        """
        n_states = probabilities.shape[-1]

        p = self.noise_level

        uniform = 1.0 / n_states

        ideal = (
            (probabilities - p * uniform)
            / (1.0 - p)
        )

        return self._normalize(ideal)

    def zero_noise_extrapolation(
        self,
        values,
        scale_factors
    ):
        """Richardson-style zero-noise extrapolation over measured data.

        ``values`` holds measurements taken at the corresponding noise
        ``scale_factors`` (for example ``[1, 2, 3]``). A least-squares
        polynomial of degree ``min(2, len(scale_factors) - 1)`` is fit to the
        data and evaluated at scale ``0`` to obtain the zero-noise estimate.

        ``values`` may be 1D (a single observable across the scales) or 2D with
        shape ``(n_scales, n_states)``, in which case each column is
        extrapolated independently and the resulting length ``n_states`` vector
        is normalized to a probability distribution.
        """
        values = np.asarray(
            values,
            dtype=float
        )

        scale_factors = np.asarray(
            scale_factors,
            dtype=float
        )

        degree = min(
            2,
            len(scale_factors) - 1
        )

        if values.ndim == 2:
            n_states = values.shape[1]

            extrapolated = np.empty(n_states)

            for state in range(n_states):
                coeffs = np.polyfit(
                    scale_factors,
                    values[:, state],
                    degree
                )

                extrapolated[state] = np.polyval(
                    coeffs,
                    0.0
                )

            return self._normalize(extrapolated)

        coeffs = np.polyfit(
            scale_factors,
            values,
            degree
        )

        return np.polyval(
            coeffs,
            0.0
        )

    def _zne_mitigation(self, probabilities):
        """Extrapolate a distribution measured at increasing noise scales.

        Unlike the single-shot strategies, ZNE needs real data taken at several
        noise scales. ``probabilities`` must therefore be a 2D stack with shape
        ``(n_scales, n_states)`` whose rows are the distributions observed at
        increasing noise. ``self.scale_factors`` supplies the corresponding
        scales; when it is ``None`` they default to ``[1, 2, ..., n_scales]``.

        With only a single row there is nothing to extrapolate from, so the
        input is returned unchanged rather than faking a result.
        """
        if probabilities.ndim != 2:
            # A 1D vector here used to be read as one measurement per *noise
            # scale* rather than one per state: mitigate([0.7, 0.3]) fitted a
            # polynomial through the two probabilities and returned the scalar
            # 1.0, with no error.
            raise ValueError(
                f"zne needs a 2D (n_scales, n_states) array whose rows are the "
                f"distributions measured at increasing noise, got a "
                f"{probabilities.ndim}D array of shape {probabilities.shape}. "
                "To mitigate a single distribution use strategy='readout' or "
                "strategy='depolarizing'."
            )

        n_scales = probabilities.shape[0]

        if n_scales < 2:
            warnings.warn(
                "zne received a single noise scale, so there is nothing to "
                "extrapolate from and the input is returned unchanged. Pass "
                "measurements at two or more scales.",
                UserWarning,
                stacklevel=3
            )
            return probabilities

        if self.scale_factors is None:
            scale_factors = np.arange(
                1,
                n_scales + 1
            )

        else:
            scale_factors = self.scale_factors

        extrapolated = self.zero_noise_extrapolation(
            probabilities,
            scale_factors
        )

        return self._normalize(extrapolated)

    def mitigate(self, probabilities):
        probabilities = np.asarray(
            probabilities,
            dtype=float
        )

        if (
            probabilities.ndim == 1
            and self.strategy in ("readout", "depolarizing")
        ):
            probabilities = probabilities.reshape(
                1,
                -1
            )

        if self.strategy == "readout":
            mitigated = self._readout_mitigation(
                probabilities
            )

        elif self.strategy == "depolarizing":
            mitigated = self._depolarizing_mitigation(
                probabilities
            )

        else:
            mitigated = self._zne_mitigation(
                probabilities
            )

        return mitigated

    def fit(self, calibration_data):
        """Build the readout calibration matrix directly from measured data.

        ``calibration_data`` is a square ``(n_states, n_states)`` confusion
        matrix whose row ``k`` is the measured probability distribution observed
        when basis state ``k`` was prepared (rows = true prepared state,
        columns = observed outcome). Each row is normalized to sum to 1 so the
        matrix is a valid confusion matrix, and its pseudo-inverse is stored for
        readout mitigation. This is genuine, data-driven calibration: the matrix
        comes from the data, not from ``noise_level``.

        If ``calibration_data`` is not 2D or not square the mitigator is left
        unchanged (both matrices stay ``None``) -- and now says so, because a
        silently skipped fit() looks identical to a successful one right up
        until mitigate() passes data through unmitigated.
        """
        calibration_data = np.asarray(
            calibration_data,
            dtype=float
        )

        if (calibration_data.ndim != 2
                or calibration_data.shape[0] != calibration_data.shape[1]):
            warnings.warn(
                f"fit() expected a square (n_states, n_states) confusion "
                f"matrix, got shape {calibration_data.shape}; no calibration "
                "was stored and readout mitigation will pass data through "
                "unchanged.",
                UserWarning,
                stacklevel=2
            )
            return self

        calibration = self._normalize(calibration_data)

        self.calibration_matrix = calibration

        self.inverse_calibration_matrix = (
            np.linalg.pinv(calibration)
        )

        return self

    def transform(self, noisy_data):
        return self.mitigate(
            noisy_data
        )

    def fit_transform(self, noisy_data):
        return self.fit(
            noisy_data
        ).transform(
            noisy_data
        )

    def estimate_noise(self, probabilities):
        """Return an entropy-based noise proxy in ``[0, 1]``.

        The mean Shannon entropy across rows is divided by ``log(n_states)`` so
        the result lies in ``[0, 1]``, with higher values indicating more
        mixing. This is only a proxy for how noisy the distributions look; it is
        NOT a calibrated physical noise estimate.
        """
        probabilities = np.asarray(
            probabilities,
            dtype=float
        )

        if probabilities.ndim == 1:
            probabilities = probabilities.reshape(
                1,
                -1
            )

        n_states = probabilities.shape[-1]

        entropy = -np.sum(
            probabilities
            * np.log(
                probabilities + 1e-12
            ),
            axis=-1
        )

        max_entropy = np.log(
            max(2, n_states)
        )

        return np.mean(entropy) / max_entropy

    def get_calibration_matrix(self):
        return self.calibration_matrix

    def get_inverse_calibration_matrix(self):
        return self.inverse_calibration_matrix


def create_advanced_noise_mitigator(
    noise_level=0.015,
    strategy="readout"
):
    return AdvancedNoiseMitigator(
        noise_level=noise_level,
        strategy=strategy
    )
