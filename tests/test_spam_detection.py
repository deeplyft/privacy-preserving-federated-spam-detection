from pathlib import Path

import joblib
import numpy as np

from spam_detection import (
    DEFAULT_DATASET,
    load_dataset,
    predict_with_artifact,
    split_dataset,
    train_centralized,
    train_federated_fedavg,
)


def test_default_dataset_loads_two_classes():
    emails, labels = load_dataset(DEFAULT_DATASET)
    assert len(emails) == len(labels)
    assert len(emails) > 100
    assert set(np.unique(labels)) == {0, 1}
    assert all(isinstance(email, str) and email for email in emails)


def test_split_is_stratified_and_reproducible():
    emails, labels = load_dataset(DEFAULT_DATASET)
    first = split_dataset(emails, labels, test_size=0.2, seed=19)
    second = split_dataset(emails, labels, test_size=0.2, seed=19)
    assert first[0] == second[0]
    assert np.array_equal(first[2], second[2])
    assert set(np.unique(first[2])) == {0, 1}
    assert set(np.unique(first[3])) == {0, 1}


def test_centralized_model_trains_and_predicts():
    emails, labels = load_dataset(DEFAULT_DATASET)
    x_train, x_test, y_train, y_test = split_dataset(emails, labels, seed=11)
    model = train_centralized(x_train, y_train, seed=11)
    predictions = model.predict(x_test[:8])
    probabilities = model.predict_proba(x_test[:8])[:, 1]
    assert len(predictions) == 8
    assert set(np.unique(predictions)).issubset({0, 1})
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_federated_model_trains_and_predicts_unknown_words():
    emails, labels = load_dataset(DEFAULT_DATASET)
    x_train, x_test, y_train, _ = split_dataset(emails, labels, seed=13)
    vectorizer, model, client_sizes = train_federated_fedavg(
        x_train, y_train, clients=3, rounds=2, local_epochs=1, seed=13
    )
    sample_matrix = vectorizer.transform([x_test[0], "unseenword xyzzy qwerty"])
    predictions = model.predict(sample_matrix)
    probabilities = model.predict_proba(sample_matrix)[:, 1]
    assert len(client_sizes) == 3
    assert sum(client_sizes) == len(y_train)
    assert len(predictions) == 2
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_saved_federated_artifact_predicts(tmp_path: Path):
    emails, labels = load_dataset(DEFAULT_DATASET)
    x_train, _, y_train, _ = split_dataset(emails, labels, seed=17)
    vectorizer, model, _ = train_federated_fedavg(
        x_train, y_train, clients=3, rounds=1, local_epochs=1, seed=17
    )
    artifact = {"kind": "federated_fedavg", "vectorizer": vectorizer, "model": model}
    path = tmp_path / "spam_model.joblib"
    joblib.dump(artifact, path)
    loaded = joblib.load(path)
    predictions, scores = predict_with_artifact(loaded, ["free offer claim your prize"])
    assert len(predictions) == len(scores) == 1
    assert int(predictions[0]) in (0, 1)
    assert 0.0 <= float(scores[0]) <= 1.0


def test_larger_legacy_csv_loads():
    larger_dataset = DEFAULT_DATASET.parent / "EmailSpamCollection.csv"
    emails, labels = load_dataset(larger_dataset)
    assert len(emails) > 5000
    assert set(np.unique(labels)) == {0, 1}
