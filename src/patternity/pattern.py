from fractions import Fraction

import matplotlib.pyplot as plt
import numpy as np


class Pattern:
    def __init__(self, history: np.array, min_length: int = 50):
        """
        Initialize the pattern detector.

        :param history: Historical data to be analyzed. Usually, the closing prices of an asset.
        :param min_length: Minimum length of the pattern to be considered.
        """

        self.history = history
        self.horizon = len(history)
        self.min_length = min_length
        self.predictions = np.array([])

    @staticmethod
    def fraction(plot1: np.array, plot2: np.array) -> tuple[int, int]:
        """Determines the ratio of the lengths of two plots."""

        fraction = Fraction(len(plot2) - 1, len(plot1) - 1)
        return fraction.numerator, fraction.denominator

    @staticmethod
    def scale(plot: np.array, factor: int) -> np.array:
        """Scale the plot by a factor."""

        diff = np.repeat(np.diff(plot), factor)
        return np.cumsum(np.insert(diff, 0, plot[0]))

    def similarity(self, plot1: np.array, plot2: np.array) -> float:
        """Determines the similarity factor between two plots."""

        plot2 = plot2 - (plot2.min() - plot1.min())
        numerator, denominator = self.fraction(plot1, plot2)

        if numerator > 20 or denominator > 20:  # prevent large scale factors
            return float("inf")

        plot1 = self.scale(plot1, numerator)
        plot2 = self.scale(plot2, denominator)

        min_value = min(plot1.min(), plot2.min())
        plot1_shifted = plot1 - min_value
        plot2_shifted = plot2 - min_value
        return np.mean(np.abs(plot1_shifted - plot2_shifted) /
                       np.maximum(plot1_shifted, plot2_shifted))

    def slices(self, min_slice_length: int = 20) -> tuple[tuple, np.array]:  # TODO: use self.min_length instead
        """Generates all possible slices of the historical data."""

        for i in range(self.horizon):
            for j in range(i + min_slice_length, self.horizon + 1):
                yield (i, j), self.history[i:j]

    def patterns(self) -> list[tuple[tuple, np.array]]:
        """Generates all possible patterns from the historical data."""

        def valid(x) -> bool:
            return all([
                self.horizon == x[0][1],
                len(x[1]) >= self.min_length,
                len(x[1]) <= self.horizon - self.min_length,
            ])

        return sorted([x for x in self.slices() if valid(x)], key=lambda x: (len(x[1]), x[0][1]), reverse=True)

    def matching_patterns(self) -> list[tuple[tuple, np.array]]:
        """Finds the best matching pattern in the historical data."""

        best_rate = float("inf")
        best_fractal = tuple()
        best_pattern = tuple()
        for (x1, x2), pattern in self.patterns():
            for subplot in self.slices():
                if subplot[0][1] >= x1:
                    continue

                similarity_rate = self.similarity(pattern, subplot[1])
                if similarity_rate < best_rate:
                    best_rate = similarity_rate
                    best_fractal = subplot
                    best_pattern = ((x1, x2), pattern)

        if not best_pattern:
            return []

        return [best_fractal, best_pattern]

    def predict(self) -> np.array:
        """Predicts the future values of the historical data."""

        matching_patterns = self.matching_patterns()
        if len(matching_patterns) < 2:
            return np.array([])

        _, last = matching_patterns[-1]
        prev_indices, prev = matching_patterns[-2]
        prev_end = prev_indices[1] - 1
        scale_factor = max(1, round((len(last) - 1) / (len(prev) - 1)))
        prediction = self.scale(self.history[prev_end:], scale_factor)
        self.predictions = np.array([i - (self.history[prev_end] - self.history[-1]) for i in prediction])
        return self.predictions

    def plot(self, background: str = "#1e1e20", history_color: str = "#0079cc", prediction_color: str = "#99999f"):
        """Plots the historical data and the predictions."""

        if not self.predictions.any():
            raise ValueError("No predictions available. Try on larger historical data.")

        plt.figure(figsize=(20, 10), facecolor=background)
        plt.gca().set_facecolor(background)
        history_range = range(0, self.horizon)
        prediction_range = range(self.horizon - 1, self.horizon + len(self.predictions) - 1)
        plt.plot(history_range, self.history, color=history_color)
        plt.plot(prediction_range, self.predictions, color=prediction_color, linestyle="--")
        plt.tight_layout()
        plt.show()
