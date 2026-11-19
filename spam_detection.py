"""Reliable baseline and simulated FedAvg email-spam detector.

This is a single-machine simulation of federated learning. It is useful for
academic experiments, but it is not a networked deployment and does not by
itself provide differential privacy or secure aggregation.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "Data" / "EmailSpamCollection-mini.csv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs"
DEFAULT_ARTIFACT = ROOT / "models" / "spam_model.joblib"


def load_dataset(path: str | Path) -> tuple[list[str], np.ndarray]:
    """Load a label/email CSV, including the repository's wide legacy CSV.

    The first two columns are used because EmailSpamCollection.csv contains
    many trailing empty columns. Labels other than 'ham'/'spam' are dropped.
    """
    csv_path = Path(path).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    try:
        frame = pd.read_csv(csv_path, usecols=[0, 1], encoding="latin-1")
    except (pd.errors.ParserError, ValueError) as exc:
        raise ValueError(
            "Could not read the dataset. Expected a CSV whose first two "
            "columns are label and email."
        ) from exc

    if frame.shape[1] != 2:
        raise ValueError("Dataset must contain at least two columns: label and email.")

    frame.columns = ["label", "email"]
    frame["label"] = frame["label"].astype(str).str.strip().str.lower()
    frame["email"] = frame["email"].fillna("").astype(str).str.strip()
    frame = frame[frame["label"].isin(["ham", "spam"]) & frame["email"].ne("")]

    if frame.empty:
        raise ValueError("No valid rows found. Labels must be 'ham' or 'spam'.")
    if frame["label"].nunique() != 2:
        raise ValueError("The dataset must contain both ham and spam examples.")

    labels = frame["label"].map({"ham": 0, "spam": 1}).to_numpy(dtype=np.int64)
    emails = frame["email"].tolist()
    return emails, labels


def split_dataset(
    emails: list[str], labels: np.ndarray, test_size: float = 0.2, seed: int = 42
):
    """Create a reproducible stratified split before fitting any vocabulary."""
    if not 0.05 <= test_size <= 0.5:
        raise ValueError("test_size must be between 0.05 and 0.5.")
    return train_test_split(
        emails,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )


def build_centralized_model(seed: int = 42) -> Pipeline:
    """Build a train-only TF-IDF + logistic-regression pipeline."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=30000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )


def _model_scores(model: Any, features: Any) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    return model.decision_function(features)


def compute_metrics(model: Any, features: Any, labels: np.ndarray) -> dict[str, Any]:
    predictions = model.predict(features)
    scores = _model_scores(model, features)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_spam": float(precision_score(labels, predictions, zero_division=0)),
        "recall_spam": float(recall_score(labels, predictions, zero_division=0)),
        "f1_spam": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


def train_centralized(
    train_emails: list[str], train_labels: np.ndarray, seed: int = 42
) -> Pipeline:
    model = build_centralized_model(seed)
    model.fit(train_emails, train_labels)
    return model


def train_federated_fedavg(
    train_emails: list[str],
    train_labels: np.ndarray,
    clients: int = 3,
    rounds: int = 8,
    local_epochs: int = 1,
    seed: int = 42,
) -> tuple[HashingVectorizer, SGDClassifier, list[int]]:
    """Run weighted FedAvg across simulated clients on one machine.

    HashingVectorizer keeps the feature space fixed without fitting a shared
    vocabulary. Each client starts each round from the same global weights.
    The server averages client weights by the number of local training rows.
    """
    if clients < 2:
        raise ValueError("Federated mode requires at least two clients.")
    if rounds < 1 or local_epochs < 1:
        raise ValueError("rounds and local_epochs must both be at least 1.")
    if len(train_labels) < clients * 2:
        raise ValueError("Not enough training examples to create the requested clients.")

    vectorizer = HashingVectorizer(
        n_features=2**14,
        alternate_sign=False,
        norm="l2",
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
    )
    train_matrix = vectorizer.transform(train_emails)

    rng = np.random.default_rng(seed)
    shard_indices = np.array_split(rng.permutation(len(train_labels)), clients)
    client_sizes = [int(len(indices)) for indices in shard_indices]
    if any(size == 0 for size in client_sizes):
        raise ValueError("Every client must receive at least one training example.")

    global_model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-5,
        learning_rate="constant",
        eta0=0.03,
        random_state=seed,
        average=False,
    )
    # Initialize sklearn's model attributes, then begin from zero global weights.
    global_model.partial_fit(train_matrix[:1], train_labels[:1], classes=np.array([0, 1]))
    global_model.coef_.fill(0.0)
    global_model.intercept_.fill(0.0)

    for _round in range(rounds):
        client_models: list[SGDClassifier] = []
        for indices in shard_indices:
            local_model = copy.deepcopy(global_model)
            client_matrix = train_matrix[indices]
            client_y = train_labels[indices]
            local_counts = np.bincount(client_y, minlength=2)
            local_class_weight = {
                class_id: len(client_y) / (2.0 * max(int(local_counts[class_id]), 1))
                for class_id in (0, 1)
            }
            local_model.class_weight = local_class_weight
            for _epoch in range(local_epochs):
                order = rng.permutation(len(indices))
                local_model.partial_fit(client_matrix[order], client_y[order])
            client_models.append(local_model)

        total_rows = sum(client_sizes)
        averaged_coef = sum(
            model.coef_ * size for model, size in zip(client_models, client_sizes)
        ) / total_rows
        averaged_intercept = sum(
            model.intercept_ * size for model, size in zip(client_models, client_sizes)
        ) / total_rows

        global_model.coef_ = averaged_coef
        global_model.intercept_ = averaged_intercept
        # Constant learning rate is used; retain the progressed counter for consistency.
        global_model.t_ = max(model.t_ for model in client_models)

    return vectorizer, global_model, client_sizes


def predict_with_artifact(artifact: dict[str, Any], emails: list[str]) -> tuple[np.ndarray, np.ndarray]:
    kind = artifact.get("kind")
    if kind == "centralized":
        model = artifact["model"]
        predictions = model.predict(emails)
        scores = _model_scores(model, emails)
        return predictions, scores
    if kind == "federated_fedavg":
        vectorizer = artifact["vectorizer"]
        model = artifact["model"]
        matrix = vectorizer.transform(emails)
        return model.predict(matrix), _model_scores(model, matrix)
    raise ValueError(f"Unsupported model artifact kind: {kind!r}")


def _write_outputs(
    output_dir: Path,
    metrics: dict[str, Any],
    labels: np.ndarray,
    predictions: np.ndarray,
    report: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    serializable = {key: value for key, value in metrics.items()}
    (output_dir / "metrics.json").write_text(
        json.dumps(serializable, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        confusion_matrix(labels, predictions, labels=[0, 1]),
        index=["actual_ham", "actual_spam"],
        columns=["predicted_ham", "predicted_spam"],
    ).to_csv(output_dir / "confusion_matrix.csv")
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")


def _portable_project_path(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def run_training(args: argparse.Namespace) -> None:
    emails, labels = load_dataset(args.dataset)
    train_emails, test_emails, train_labels, test_labels = split_dataset(
        emails, labels, test_size=args.test_size, seed=args.seed
    )

    if args.mode == "centralized":
        model = train_centralized(train_emails, train_labels, seed=args.seed)
        artifact = {"kind": "centralized", "model": model, "seed": args.seed}
        test_predictions = model.predict(test_emails)
        test_scores = _model_scores(model, test_emails)
        training_details = {"training_samples": len(train_labels)}
    else:
        vectorizer, model, client_sizes = train_federated_fedavg(
            train_emails,
            train_labels,
            clients=args.clients,
            rounds=args.rounds,
            local_epochs=args.local_epochs,
            seed=args.seed,
        )
        test_matrix = vectorizer.transform(test_emails)
        test_predictions = model.predict(test_matrix)
        test_scores = _model_scores(model, test_matrix)
        artifact = {
            "kind": "federated_fedavg",
            "vectorizer": vectorizer,
            "model": model,
            "seed": args.seed,
            "clients": args.clients,
            "rounds": args.rounds,
            "local_epochs": args.local_epochs,
            "client_training_sizes": client_sizes,
        }
        training_details = {
            "training_samples": len(train_labels),
            "clients": args.clients,
            "rounds": args.rounds,
            "local_epochs": args.local_epochs,
            "client_training_sizes": client_sizes,
            "privacy_note": "Simulated FedAvg only; no differential privacy or secure aggregation.",
        }

    metrics = {
        "mode": args.mode,
        "dataset": _portable_project_path(args.dataset),
        "total_valid_samples": len(labels),
        "test_samples": len(test_labels),
        **compute_metrics_from_predictions(test_labels, test_predictions, test_scores),
        **training_details,
    }
    report = classification_report(
        test_labels,
        test_predictions,
        labels=[0, 1],
        target_names=["ham", "spam"],
        zero_division=0,
    )

    artifact_path = Path(args.artifact).expanduser().resolve()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_path)
    output_dir = Path(args.output_dir).expanduser().resolve()
    _write_outputs(output_dir, metrics, test_labels, test_predictions, report)

    print(f"Mode: {args.mode}")
    print(f"Dataset: {len(labels)} valid emails ({int((labels == 0).sum())} ham, {int((labels == 1).sum())} spam)")
    print(f"Train/test: {len(train_labels)}/{len(test_labels)}")
    for name in ("accuracy", "precision_spam", "recall_spam", "f1_spam", "roc_auc"):
        value = metrics[name]
        print(f"{name}: {value:.4f}" if value is not None else f"{name}: n/a")
    print("\n", report, sep="")
    print(f"Saved model: {artifact_path}")
    print(f"Saved metrics and confusion matrix: {output_dir}")
    if args.mode == "federated":
        print("Note: clients are simulated in this process; this is not a privacy guarantee.")


def compute_metrics_from_predictions(
    labels: np.ndarray, predictions: np.ndarray, scores: np.ndarray
) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_spam": float(precision_score(labels, predictions, zero_division=0)),
        "recall_spam": float(recall_score(labels, predictions, zero_division=0)),
        "f1_spam": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


def run_prediction(args: argparse.Namespace) -> None:
    if not args.model_path:
        raise ValueError("Prediction mode requires --model-path.")
    artifact_path = Path(args.model_path).expanduser().resolve()
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Model file not found: {artifact_path}")
    artifact = joblib.load(artifact_path)
    text = args.predict_text
    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8", errors="replace")
    if not text or not text.strip():
        raise ValueError("Provide non-empty text using --predict-text or --text-file.")

    prediction, score = predict_with_artifact(artifact, [text])
    label = "SPAM" if int(prediction[0]) == 1 else "HAM"
    print(f"Prediction: {label}")
    if artifact.get("kind") == "centralized":
        spam_probability = float(artifact["model"].predict_proba([text])[0, 1])
    else:
        matrix = artifact["vectorizer"].transform([text])
        spam_probability = float(artifact["model"].predict_proba(matrix)[0, 1])
    print(f"Estimated spam probability: {spam_probability:.2%}")
    print("This is a model estimate, not a guarantee that an email is safe or malicious.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Email spam detection: centralized baseline or simulated FedAvg.")
    parser.add_argument("--mode", choices=["centralized", "federated", "predict"], default="centralized")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="CSV with label,email columns.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--artifact", default=str(DEFAULT_ARTIFACT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model-path", help="Saved model artifact used with --mode predict.")
    parser.add_argument("--predict-text", help="Classify one email directly.")
    parser.add_argument("--text-file", help="Read one email from a UTF-8 text file.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "predict":
        run_prediction(args)
    else:
        run_training(args)


if __name__ == "__main__":
    main()
