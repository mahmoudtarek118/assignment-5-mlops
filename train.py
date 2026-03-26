"""
train.py – Train a classifier, log to MLflow, and export the Run ID.
"""

import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():
    # ------------------------------------------------------------------
    # 1. Load data (in a real project this would come from DVC-tracked files)
    # ------------------------------------------------------------------
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ------------------------------------------------------------------
    # 2. Configure MLflow
    # ------------------------------------------------------------------
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("iris-classification")

    # ------------------------------------------------------------------
    # 3. Train & log
    # ------------------------------------------------------------------
    with mlflow.start_run() as run:
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(clf, "model")

        print(f"Run ID : {run.info.run_id}")
        print(f"Accuracy: {accuracy:.4f}")

        # --------------------------------------------------------------
        # 4. Export Run ID to model_info.txt
        # --------------------------------------------------------------
        with open("model_info.txt", "w") as f:
            f.write(run.info.run_id)

        print("model_info.txt written successfully.")


if __name__ == "__main__":
    main()
