# Upgrade notes

## Work completed in this copy

1. Added `spam_detection.py` as the new recommended entry point.
   - Resolves data paths relative to the project directory.
   - Reads both the 850-row mini CSV and the larger CSV (uses the first two columns and filters invalid labels).
   - Uses a stratified train/test split before text vectorization.
   - Provides a centralized TF-IDF + Logistic Regression baseline.
   - Provides a single-machine, three-client-capable FedAvg simulation using fixed HashingVectorizer features, local SGD updates, and sample-weighted model aggregation.
   - Applies class balancing locally on each simulated client.
   - Saves real metrics, a confusion matrix, and a reusable model artifact.
   - Supports classifying a new text or a text file with the saved model.
2. Added `compare_models.py` for Logistic Regression, Multinomial Naive Bayes, and Linear SVM. AUC uses probabilities or decision scores rather than hard class labels.
3. Replaced the old `Multiple Models.py` implementation with a compatibility wrapper for the updated comparison script. This removes its obsolete pandas APIs and deprecated plotting call.
4. Reworked `Preprocessing.py` for the legacy GRU experiment.
   - Corrects the `Data/` versus `data/` path issue.
   - Reads the existing CSV header correctly through the shared loader.
   - Splits the dataset before vocabulary creation.
   - Uses an unknown-word ID and safe sequence truncation/padding.
   - Writes the training rows first and test rows last, matching the old GRU script's split convention.
5. Updated `Selfmade_GRU.py` to use current PyTorch activation functions and skip padding tokens when updating its hidden state.
6. Added tests, setup requirements, `.gitignore`, and updated run instructions.

## Verification completed

The following checks were run successfully in the provided environment:

- `python -m pytest -q` — 8 tests passed.
- `python spam_detection.py --mode centralized` — completed on the mini CSV.
- `python spam_detection.py --mode federated --clients 3 --rounds 8 --local-epochs 1` — completed on the mini CSV.
- `python spam_detection.py --mode predict ...` — loaded a saved artifact and classified a new email.
- `python compare_models.py` — completed all three classifiers.
- `python 'Multiple Models.py'` — compatibility entry point completed.
- `python Preprocessing.py` — generated the legacy GRU arrays and metadata.
- The larger `Data/EmailSpamCollection.csv` loaded 5,726 valid examples and completed centralized training.
- `Selfmade_GRU.py` forward pass was checked with PyTorch, including padded input.

### Example held-out results

These are results from one stratified 80/20 split with random seed 42. They are demonstration results on the supplied historical dataset, not a guarantee of performance on modern email traffic.

| Dataset / mode | Accuracy | Spam precision | Spam recall | Spam F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Mini CSV, centralized | 0.9824 | 0.9434 | 1.0000 | 0.9709 | 0.9990 |
| Mini CSV, simulated FedAvg (3 clients, 8 rounds) | 0.9529 | 0.8621 | 1.0000 | 0.9259 | 0.9895 |
| Larger CSV, centralized | 0.9869 | 0.9576 | 0.9891 | 0.9731 | 0.9992 |

## What remains a legacy experiment

The following original experiments are preserved but were not claimed as fully upgraded or verified end to end:

- `Email Spam Detection - GRU (with FL).py` still depends on the historical PySyft `TorchHook` / `VirtualWorker` API. The updated `Preprocessing.py` prepares its arrays, but the training script itself needs a separate PySyft migration. Use the new `spam_detection.py --mode federated` workflow for a runnable FedAvg demonstration.
- `Email Spam Detection - Neural Networks (without FL).py` still uses the old standalone Keras workflow. TensorFlow was not available in the execution environment, so its training path could not be safely tested or represented as verified.
- The Paillier scripts still require the optional `phe` package. The original encrypted-inference demonstration is not federated training; no claim of secure federated training is made.

## Privacy boundary

The new FedAvg mode simulates clients in one Python process for reproducibility. It does not provide network isolation, encryption in transit, secure aggregation, differential privacy, or a formal privacy budget. It is appropriate as an educational baseline and a starting point for a later distributed implementation, not as a production email-security service.
