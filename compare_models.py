"""Compare a few simple spam classifiers using a leakage-safe train/test split."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from spam_detection import DEFAULT_DATASET, load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/model_comparison.csv")
    args = parser.parse_args()

    emails, labels = load_dataset(args.dataset)
    x_train, x_test, y_train, y_test = train_test_split(
        emails,
        labels,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels,
    )

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=args.seed),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Linear SVM": LinearSVC(class_weight="balanced", random_state=args.seed),
    }
    rows = []
    for name, classifier in classifiers.items():
        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=30000,
                    sublinear_tf=True,
                )),
                ("classifier", classifier),
            ]
        )
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        if hasattr(pipeline, "predict_proba"):
            scores = pipeline.predict_proba(x_test)[:, 1]
        else:
            scores = pipeline.decision_function(x_test)
        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, predictions),
                "precision_spam": precision_score(y_test, predictions, zero_division=0),
                "recall_spam": recall_score(y_test, predictions, zero_division=0),
                "f1_spam": f1_score(y_test, predictions, zero_division=0),
                "roc_auc": roc_auc_score(y_test, scores),
            }
        )

    results = pd.DataFrame(rows).sort_values("f1_spam", ascending=False)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved comparison: {output}")


if __name__ == "__main__":
    main()
