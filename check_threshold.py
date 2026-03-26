"""
check_threshold.py – Read the Run ID from model_info.txt,
query MLflow for accuracy, and fail if it is below 0.85.
"""

import os
import sys
import mlflow


THRESHOLD = 0.85


def main():
    # ------------------------------------------------------------------
    # 1. Read the Run ID
    # ------------------------------------------------------------------
    if not os.path.exists("model_info.txt"):
        print("ERROR: model_info.txt not found.")
        sys.exit(1)

    with open("model_info.txt", "r") as f:
        run_id = f.read().strip()

    if not run_id:
        print("ERROR: model_info.txt is empty.")
        sys.exit(1)

    print(f"Run ID: {run_id}")

    # ------------------------------------------------------------------
    # 2. Query MLflow for the accuracy metric
    # ------------------------------------------------------------------
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    accuracy = run.data.metrics.get("accuracy")

    if accuracy is None:
        print("ERROR: 'accuracy' metric not found for this run.")
        sys.exit(1)

    print(f"Accuracy: {accuracy:.4f}")

    # ------------------------------------------------------------------
    # 3. Threshold check
    # ------------------------------------------------------------------
    if accuracy < THRESHOLD:
        print(f"FAIL: Accuracy {accuracy:.4f} is below threshold {THRESHOLD}.")
        sys.exit(1)
    else:
        print(f"PASS: Accuracy {accuracy:.4f} meets threshold {THRESHOLD}.")


if __name__ == "__main__":
    main()
