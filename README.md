# Privacy-Preserving Federated Spam Detection (Updated)

This repository contains the original academic experiments plus a simpler, tested entry point for email spam detection.

## What is improved

- Dataset paths are resolved relative to the project, so the code no longer expects a lowercase `data/` directory.
- The default dataset loader reads the supplied `Data/EmailSpamCollection-mini.csv` and can also read the first two columns of the wide `Data/EmailSpamCollection.csv` file.
- Training and test data are split with stratification before fitting the TF-IDF vocabulary.
- The centralized model reports real accuracy, spam precision, spam recall, spam F1, ROC-AUC, and a confusion matrix. No metrics are manually overwritten.
- A modern simulated FedAvg option trains three or more local clients from the same global model and aggregates their weights according to each client's sample count.
- A prediction command can load the saved model and classify a new email, including words not seen during training.
- A small model-comparison script evaluates Logistic Regression, Multinomial Naive Bayes, and Linear SVM using the same leakage-safe split strategy.
- The legacy GRU preprocessing script now uses the correct `Data/` path, stratified splitting, a training-only vocabulary, an unknown-word token, and safe sequence padding. The custom GRU skips padding positions.
- The original `Multiple Models.py` filename is retained as a compatibility wrapper around the updated comparison script.
- Outputs are saved under `outputs/` and trained model files under `models/`.

## Important privacy note

The `federated` mode is a **single-machine simulation**. It demonstrates local client updates and weighted FedAvg, but clients are not separate devices or processes. It does not implement differential privacy, secure aggregation, or encrypted transport. Therefore it must not be described as providing a formal privacy guarantee. The homomorphic-encryption files from the original project remain separate demonstrations and are not part of this new training pipeline.

## 1. Set up the environment

Use Python 3.10 or newer. From the project root:

```bash
python -m venv .venv
```

Activate it:

**Windows PowerShell**

```powershell
.venv\Scripts\Activate.ps1
```

**macOS/Linux**

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. Train the centralized baseline

```bash
python spam_detection.py --mode centralized
```

The default dataset is `Data/EmailSpamCollection-mini.csv`. To use the larger CSV:

```bash
python spam_detection.py --mode centralized --dataset Data/EmailSpamCollection.csv
```

The first two CSV columns must be `label` and `email`, with labels `ham` or `spam`.

## 3. Train the simulated federated model

```bash
python spam_detection.py --mode federated --clients 3 --rounds 8 --local-epochs 1
```

You can change the number of clients, communication rounds, and local epochs. Keep in mind that a larger number of rounds or local epochs may take longer and does not necessarily improve performance.

## 4. Classify a new email

After training either mode, use the saved model artifact:

```bash
python spam_detection.py --mode predict --model-path models/spam_model.joblib --predict-text "Congratulations, claim your prize now"
```

You can also classify the contents of a text file:

```bash
python spam_detection.py --mode predict --model-path models/spam_model.joblib --text-file sample_email.txt
```

The printed spam probability is a model estimate, not proof that an email is safe or malicious. Do not upload confidential email content to a third-party service as part of this example.

## 5. Compare simple classifiers

```bash
python compare_models.py
```

The comparison is saved to `outputs/model_comparison.csv`.

## 6. Prepare arrays for the legacy GRU experiment (optional)

```bash
python Preprocessing.py
```

This creates `data/inputs.npy`, `data/labels.npy`, `data/word_to_idx.json`, and preprocessing metadata. The older federated GRU training script still relies on its historical PySyft API, so the new FedAvg mode in `spam_detection.py` is the recommended federated workflow.

## 7. Run tests

```bash
python -m pytest -q
```

## Generated files

- `models/spam_model.joblib`: trained model artifact.
- `outputs/metrics.json`: metric values from the held-out test set.
- `outputs/confusion_matrix.csv`: ham/spam confusion matrix.
- `outputs/classification_report.txt`: precision, recall, and F1 by class.
- `outputs/model_comparison.csv`: results from the three-model comparison.

Model files are Python joblib/pickle artifacts. Only load model files that you created or otherwise trust.

## Original scripts

The original project files are retained for reference. Several use old dependencies or APIs (including PySyft virtual workers and older Keras interfaces) and are not used by the updated workflow above. The original scripts should be treated as legacy experiments until separately ported and tested. `spam_detection.py` is the recommended working entry point.
