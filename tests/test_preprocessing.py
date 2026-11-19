import json
from pathlib import Path

import numpy as np

from Preprocessing import pad_and_truncate, preprocess_dataset, tokenize
from spam_detection import DEFAULT_DATASET


def test_unknown_words_and_truncation_are_safe():
    word_to_idx = {"known": 2, "word": 3}
    assert tokenize("known unseenword", word_to_idx) == [2, 1]
    padded = pad_and_truncate([[2, 3, 4, 5]], max_length=3)
    assert padded.tolist() == [[3, 4, 5]]
    assert pad_and_truncate([[]], max_length=3).tolist() == [[0, 0, 0]]


def test_preprocessing_writes_train_then_test_arrays(tmp_path: Path):
    metadata = preprocess_dataset(
        dataset_path=DEFAULT_DATASET,
        output_dir=tmp_path,
        max_length=40,
        test_size=0.2,
        seed=42,
    )
    inputs = np.load(tmp_path / "inputs.npy")
    labels = np.load(tmp_path / "labels.npy")
    vocabulary = json.loads((tmp_path / "word_to_idx.json").read_text(encoding="utf-8"))
    assert inputs.shape == (850, 40)
    assert labels.shape == (850,)
    assert metadata["train_samples"] == 680
    assert metadata["test_samples"] == 170
    assert min(vocabulary.values()) >= 2
    assert set(np.unique(labels)) == {0, 1}
