from __future__ import annotations

"""Small demo that mimics the Unity RecallRate behaviour."""

import logging

from importlib import import_module

RecallManager = import_module("town_core.recall_manager").RecallManager

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main() -> None:
    recall = RecallManager()
    recall.initialize_network(input_size=2, hidden_neurons=5, output_size=1)

    # Example data
    recall.add_recall_data([1.0, 2.0], True)
    recall.add_recall_data([2.0, 3.0], True)
    recall.add_recall_data([4.0, 5.0], False)
    recall.add_recall_data([6.0, 7.0], True)
    recall.add_recall_data([8.0, 9.0], False)

    initial_rate = recall.calculate_recall_rate()
    logging.info("Initial Recall Rate: %.1f%%", initial_rate * 100)

    recall.train_network(epochs=1000)

    test_input = [2.0, 3.0]
    confidence = recall.predict_recall(test_input)
    logging.info(
        "Recall Confidence for %s: %.1f%%",
        ", ".join(map(str, test_input)),
        confidence * 100,
    )

    recall.add_recall_data(test_input, confidence >= 0.5)
    updated_rate = recall.calculate_recall_rate()
    logging.info("Updated Recall Rate: %.1f%%", updated_rate * 100)


if __name__ == "__main__":
    main()
