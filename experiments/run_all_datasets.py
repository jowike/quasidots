import pandas as pd
import numpy as np
from aeon.datasets import load_classification
from sklearn.metrics import accuracy_score
from sktime.classification.kernel_based import RocketClassifier

from situ import SITUClassifier

import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning
)

DATASETS = [
    "ECG200", "ECG5000", "Coffee", "GunPoint", "ItalyPowerDemand",
    "FordA", "FordB", "Wafer", "ElectricDevices", "TwoLeadECG",
    # 🔴 TODO: wkleić pełną listę 100+ z main.txt
]

def run_single(dataset, operation, num_kernels, seed):
    try:
        X_train, y_train = load_classification(dataset, split="train", load_equal_length=True, load_no_missing=True)
        X_test, y_test = load_classification(dataset, split="test", load_equal_length=True, load_no_missing=True)

        situ = SITUClassifier(
            num_kernels=num_kernels,
            operation=operation,
            random_state=seed
        )

        rocket = RocketClassifier(
            num_kernels=num_kernels,
            random_state=seed
        )

        situ.fit(X_train, y_train)
        rocket.fit(X_train, y_train)

        y_pred_situ = situ.predict(X_test)
        y_pred_rocket = rocket.predict(X_test)

        acc_situ = accuracy_score(y_test, y_pred_situ)
        acc_rocket = accuracy_score(y_test, y_pred_rocket)

        return acc_situ, acc_rocket

    except Exception as e:
        print(f"[ERROR] {dataset}: {e}")
        return None, None

def run_all(operation="AT", num_kernels=1000, seed=42, output_file="results.csv"):
    results = []

    for i, dataset in enumerate(DATASETS):
        print(f"[{i+1}/{len(DATASETS)}] Running {dataset}...")

        acc_situ, acc_rocket = run_single(
            dataset, operation, num_kernels, seed
        )

        results.append({
            "dataset": dataset,
            "situ_accuracy": acc_situ,
            "rocket_accuracy": acc_rocket,
            "difference": None if acc_situ is None else acc_situ - acc_rocket
        })

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)

    print("\nSummary:")
    valid = df.dropna()

    wins = (valid["difference"] > 0).sum()
    losses = (valid["difference"] < 0).sum()
    draws = (valid["difference"] == 0).sum()

    print(f"SITU wins: {wins}")
    print(f"ROCKET wins: {losses}")
    print(f"Draws: {draws}")

    print("\n✅ Finished. Results saved to:", output_file)

# Entry point for running all datasets with specified parameters.
if __name__ == "__main__":
    run_all(
        operation="AT",
        num_kernels=1000,
        seed=42,
        output_file="results_AT.csv"
    )