The updated and recommended instructions are in README.md.

Start with:
  python -m pip install -r requirements.txt
  python spam_detection.py --mode centralized
  python spam_detection.py --mode federated --clients 3 --rounds 8
  python compare_models.py
  python -m pytest -q

Important: federated mode is a single-machine FedAvg simulation. It is not a real multi-device deployment and does not provide differential privacy or secure aggregation. The old scripts are retained as legacy experiments and may require older dependencies.
