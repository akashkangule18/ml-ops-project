import os
import json
import pickle
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub

from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

# ==========================================
# DagsHub Configuration
# ==========================================

repo_owner = "akashkangule18"
repo_name = "ml-ops-project"

dagshub_token = os.getenv("DAGSHUB_PAT")

if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub.init(
    repo_owner=repo_owner,
    repo_name=repo_name,
    mlflow=True
)

print("MLflow Version:", mlflow.__version__)
print("Tracking URI:", mlflow.get_tracking_uri())

# ==========================================
# Experiment
# ==========================================

mlflow.set_experiment("final model registerd-staging")

# ==========================================
# Start Run
# ==========================================

with mlflow.start_run(run_name="model_registering"):

    # Load Model
    with open("models/model.pkl", "rb") as file:
        model = pickle.load(file)

    # Load Training Data
    X_train = pd.read_csv(
        "data/processed/train_tr_processed.csv"
    )

    # Load Metrics
    with open("reports/metrics.json", "r") as file:
        metrics = json.load(file)

    # ==========================================
    # Log Parameters
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
    # Signature
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

    print("\nModel Logged Successfully")
    print(model_info)

    # ==========================================
    # Set Staging Alias
    # ==========================================

    client = MlflowClient()

    versions = client.search_model_versions(
        "name='Random_forest_classifier'"
    )

    latest_version = max(
        int(v.version)
        for v in versions
    )

    client.set_registered_model_alias(
        name="Random_forest_classifier",
        alias="Staging",
        version=latest_version
    )

    print(
        f"\nStaging Alias Assigned "
        f"to Version {latest_version}"
    )

    # ==========================================
    # Verify Registry
    # ==========================================

    print("\nRegistered Models:")

    for model_obj in client.search_registered_models():
        print(model_obj.name)

    print("\nModel Versions:")

    for version in client.search_model_versions(
        "name='Random_forest_classifier'"
    ):
        print(
            f"Version={version.version}, "
            f"Aliases={getattr(version, 'aliases', [])}"
        )

print("\nModel registration and staging successful")