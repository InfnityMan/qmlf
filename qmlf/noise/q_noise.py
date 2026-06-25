import numpy as np


class AdvancedNoiseMitigator:
    def __init__(
        self,
        noise_level=0.015,
        strategy="readout"
    ):
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
        n_scales = probabilities.shape[0]

        if n_scales < 2:
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

        elif self.strategy == "zne":
            mitigated = self._zne_mitigation(
                probabilities
            )

        else:
            mitigated = probabilities

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
        unchanged (both matrices stay ``None``).
        """
        calibration_data = np.asarray(
            calibration_data,
            dtype=float
        )

        if calibration_data.ndim != 2:
            return self

        if calibration_data.shape[0] != calibration_data.shape[1]:
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
