import pandas as pd
import pickle
import json
import os
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

# ==========================================
# DagsHub + MLflow Authentication
# ==========================================

dagshub_token = os.getenv("DAGSHUB_PAT")

if not dagshub_token:
    raise ValueError(
        "DAGSHUB_PAT not found. Check GitHub Secrets."
    )

print("DAGSHUB_PAT found")
print("Token length:", len(dagshub_token))

os.environ["MLFLOW_TRACKING_USERNAME"] = "akashkangule18"
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

mlflow.set_tracking_uri(
    "https://dagshub.com/akashkangule18/ml-ops-project.mlflow"
)

print("MLflow Version:", mlflow.__version__)
print("Tracking URI:", mlflow.get_tracking_uri())

# ==========================================
# Experiment
# ==========================================

mlflow.set_experiment("final model registerd-staging")

with mlflow.start_run(run_name="model_registering"):

    # Load model
    with open("models/model.pkl", "rb") as file:
        model = pickle.load(file)

    # Load training data
    X_train = pd.read_csv(
        "./data/processed/train_tr_processed.csv"
    )

    # Load metrics
    with open("reports/metrics.json", "r") as file:
        metrics = json.load(file)

    # ==========================================
    # Log Params
    # ==========================================

    mlflow.log_params(model.get_params())
    mlflow.log_param("test_size", 0.4)

    # ==========================================
    # Log Artifacts
    # ==========================================

    mlflow.log_artifact(__file__)
    mlflow.log_artifact("reports/metrics.json")

    # ==========================================
    # Log Metrics
    # ==========================================

    mlflow.log_metric(
        "accuracy_score",
        metrics["accuracy_score"]
    )

    mlflow.log_metric(
        "precision_score",
        metrics["precision_score"]
    )

    mlflow.log_metric(
        "recall_score",
        metrics["recall_score"]
    )

    # ==========================================
    # Infer Signature
    # ==========================================

    predictions = model.predict(X_train)

    signature = infer_signature(
        X_train,
        predictions
    )

    # ==========================================
    # Register Model
    # ==========================================

    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="Random_forest_model",
        signature=signature,
        registered_model_name="Random_forest_classifier"
    )

    print("Model URI:", model_info.model_uri)

    # ==========================================
    # Model Registry
    # ==========================================

    client = MlflowClient()

    versions = client.search_model_versions(
        "name='Random_forest_classifier'"
    )

    print("\nAvailable Versions:")

    for v in versions:
        print(f"Version={v.version}")

    latest_version = max(
        int(v.version)
        for v in versions
    )

    print(
        f"\nLatest Version: {latest_version}"
    )

    print(
        "Version Type:",
        type(latest_version)
    )

    # ==========================================
    # Update Staging Alias
    # ==========================================

    try:
        client.set_registered_model_alias(
            "Random_forest_classifier",
            "Staging",
            str(latest_version)
        )

        print(
            f"Staging moved to Version {latest_version}"
        )

    except Exception as e:
        print(
            f"Alias update failed: {e}"
        )

    # ==========================================
    # Show Versions
    # ==========================================

    print("\nRegistered Versions:")

    for v in client.search_model_versions(
        "name='Random_forest_classifier'"
    ):
        print(
            f"Version={v.version}, "
            f"Aliases={getattr(v, 'aliases', [])}"
        )

    print(
        "\nModel registration successful"
    )