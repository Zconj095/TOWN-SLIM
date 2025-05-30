from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

import numpy as np

logger = logging.getLogger("Recall")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(message)s [%(asctime)s]"))
    logger.addHandler(_h)


@dataclass
class RecallEntry:
    """Container for single recall datapoint."""

    input_data: List[float]
    is_recalled: bool

    def as_array(self) -> np.ndarray:
        return np.asarray(self.input_data, dtype=float)


class RecallManager:
    """Simple recall tracker with a tiny neural network."""

    entries: List[RecallEntry]
    weights1: np.ndarray
    bias1: np.ndarray
    weights2: np.ndarray
    bias2: float
    lr: float

    def __init__(self) -> None:
        self.entries = []
        self.lr = 0.1

    # ------------------------------------------------------------------
    def initialize_network(
        self, input_size: int, hidden_neurons: int, output_size: int
    ) -> None:
        self.weights1 = np.random.randn(input_size, hidden_neurons)
        self.bias1 = np.zeros(hidden_neurons)
        self.weights2 = np.random.randn(hidden_neurons, output_size)
        self.bias2 = 0.0
        logger.info(
            "Recall network initialised (%d → %d → %d)",
            input_size,
            hidden_neurons,
            output_size,
        )

    # ------------------------------------------------------------------
    def add_recall_data(self, input_data: List[float], is_recalled: bool) -> None:
        self.entries.append(RecallEntry(input_data, is_recalled))

    # ------------------------------------------------------------------
    def calculate_recall_rate(self) -> float:
        if not self.entries:
            return 0.0
        recalled = sum(1 for e in self.entries if e.is_recalled)
        return recalled / len(self.entries)

    # ------------------------------------------------------------------
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    # ------------------------------------------------------------------
    def train_network(self, epochs: int = 1000) -> None:
        if not self.entries:
            return
        X = np.vstack([e.as_array() for e in self.entries])
        y = np.array([1.0 if e.is_recalled else 0.0 for e in self.entries]).reshape(
            -1, 1
        )

        for epoch in range(epochs):
            # forward
            hidden = self._sigmoid(X.dot(self.weights1) + self.bias1)
            output = self._sigmoid(hidden.dot(self.weights2) + self.bias2)

            # backprop
            error = output - y
            d_weights2 = hidden.T.dot(error) / len(X)
            d_bias2 = np.mean(error, axis=0)

            hidden_error = error.dot(self.weights2.T) * hidden * (1 - hidden)
            d_weights1 = X.T.dot(hidden_error) / len(X)
            d_bias1 = np.mean(hidden_error, axis=0)

            # update
            self.weights2 -= self.lr * d_weights2
            self.bias2 -= self.lr * d_bias2
            self.weights1 -= self.lr * d_weights1
            self.bias1 -= self.lr * d_bias1

            if epoch % 100 == 0:
                loss = np.mean((error) ** 2)
                logger.info("Epoch %d: Loss %.6f", epoch, loss)

    # ------------------------------------------------------------------
    def predict_recall(self, input_data: List[float]) -> float:
        x = np.asarray(input_data, dtype=float)
        hidden = self._sigmoid(x.dot(self.weights1) + self.bias1)
        output = self._sigmoid(hidden.dot(self.weights2) + self.bias2)
        return float(output[0])
