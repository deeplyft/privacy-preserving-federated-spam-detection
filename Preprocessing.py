"""Leakage-safe preprocessing for the repository's legacy custom GRU script.

The new recommended classifier in spam_detection.py performs preprocessing
inside its own pipeline. This script exists for the older PyTorch GRU demo.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from spam_detection import DEFAULT_DATASET, load_dataset

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "data"
UNKNOWN_TOKEN_ID = 1
PADDING_TOKEN_ID = 0


def clean_text(text: str) -> str:
    """Normalize an email while retaining digits that may signal spam."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str, word_to_idx: dict[str, int]) -> list[int]:
    """Map tokens to IDs; unseen words safely map to the UNK token."""
    return [word_to_idx.get(word, UNKNOWN_TOKEN_ID) for word in clean_text(text).split()]


def pad_and_truncate(messages: list[list[int]], max_length: int = 100) -> np.ndarray:
    """Pre-pad sequences and keep the last max_length tokens."""
    if max_length < 1:
        raise ValueError("max_length must be at least 1.")
    features = np.full((len(messages), max_length), PADDING_TOKEN_ID, dtype=np.int64)
    for row_index, message in enumerate(messages):
        if not message:
            continue
        kept_tokens = message[-max_length:]
        features[row_index, -len(kept_tokens):] = kept_tokens
    return features


def preprocess_dataset(
    dataset_path: str | Path = DEFAULT_DATASET,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    max_length: int = 100,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict[str, int | str]:
    emails, labels = load_dataset(dataset_path)
    cleaned_emails = [clean_text(email) for email in emails]
    train_emails, test_emails, train_labels, test_labels = train_test_split(
        cleaned_emails,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )

    # Build the vocabulary from training emails only to avoid test-set leakage.
    word_counts = Counter(" ".join(train_emails).split())
    vocabulary = sorted(word_counts, key=lambda word: (-word_counts[word], word))
    word_to_idx = {word: index for index, word in enumerate(vocabulary, start=2)}

    train_tokens = [tokenize(email, word_to_idx) for email in train_emails]
    test_tokens = [tokenize(email, word_to_idx) for email in test_emails]
    train_inputs = pad_and_truncate(train_tokens, max_length=max_length)
    test_inputs = pad_and_truncate(test_tokens, max_length=max_length)

    # The legacy GRU file expects the final 20% of arrays to be its test split.
    all_inputs = np.concatenate([train_inputs, test_inputs], axis=0)
    all_labels = np.concatenate([train_labels, test_labels]).astype(np.int64)

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    np.save(output_path / "inputs.npy", all_inputs)
    np.save(output_path / "labels.npy", all_labels)
    (output_path / "word_to_idx.json").write_text(
        json.dumps(word_to_idx, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    resolved_dataset = Path(dataset_path).expanduser().resolve()
    try:
        dataset_label = resolved_dataset.relative_to(ROOT).as_posix()
    except ValueError:
        dataset_label = str(resolved_dataset)

    metadata: dict[str, int | str] = {
        "dataset": dataset_label,
        "samples": len(all_labels),
        "train_samples": len(train_labels),
        "test_samples": len(test_labels),
        "vocabulary_size_including_reserved_tokens": len(word_to_idx) + 2,
        "max_sequence_length": max_length,
        "padding_token_id": PADDING_TOKEN_ID,
        "unknown_token_id": UNKNOWN_TOKEN_ID,
        "seed": seed,
    }
    (output_path / "preprocessing_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare leakage-safe arrays for the legacy GRU experiment.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-length", type=int, default=100)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    metadata = preprocess_dataset(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        max_length=args.max_length,
        test_size=args.test_size,
        seed=args.seed,
    )
    print("Preprocessing completed:")
    for key, value in metadata.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
