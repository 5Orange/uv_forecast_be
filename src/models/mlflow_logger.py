import os
import json
import tempfile
from typing import  Dict, Any, Optional
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
import torch
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

from src.config import PROJECT_ROOT
from src.models.mlflow_wrapper import (
    DeepLearningModelWrapper, ProphetLGBWrapper, get_model_wrapper_class
)

def setup_mlflow():
    databricks_host = os.getenv("DATABRICKS_HOST")
    databricks_token = os.getenv("DATABRICKS_TOKEN")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "databricks")
    experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID")

    if not databricks_host or not databricks_token:
        mlflow.set_tracking_uri("file://" + str(PROJECT_ROOT) + "/mlruns")
        mlflow.set_experiment("uv-forecasting-local")
        return False

    mlflow.set_tracking_uri(tracking_uri)
    if experiment_id:
        mlflow.set_experiment(experiment_id=experiment_id)
    else:
        mlflow.set_experiment("UV-Forecasting")

    return True
def log_sklearn_model(
        model: Any,
        model_name: str,
        metrics_df: pd.DataFrame,
        feature_cols: list,
        hyperparameter: Optional[Dict] = None
):
    with mlflow.start_run(run_name=f"uv_forecast-{model_name}"):
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("feature_list", json.dumps(feature_cols))

        if hyperparameter:
            for key, value in hyperparameter.items():
                if isinstance(value, (int, float, str, bool)):
                    mlflow.log_param(f"hp_{key}", value)

        for _, row in metrics_df.iterrows():
            split = row['split']
            mlflow.log_metric(f"{split}_r2", row["r2"])
            mlflow.log_metric(f"{split}_rmse", row["rmse"])
            mlflow.log_metric(f"{split}_mae", row["mae"])
            if 'mape' in row:
                mlflow.log_metric(f"{split}_mape", row["mape"])
            if 'mbe' in row:
                mlflow.log_metric(f"{split}_mbe", row["mbe"])

        if hasattr(model, "feature_importances_"):
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                importance_df.to_csv(f, index=False)
                f.flush()
            mlflow.log_artifact(f.name, 'feature_importance')
            os.unlink(f.name)

        registered_model_name = f"main.uv_models.uv_forecast-{model_name}"
        input_example = pd.DataFrame(
            np.zeros((1, len(feature_cols))),
            columns=feature_cols
        )

        mlflow.sklearn.log_model(
            model,
            'model',
            registered_model_name=registered_model_name,
            input_example=input_example
        )
        print(f" Logged {model_name} to ml flow (registered as {registered_model_name})")

def log_pytorch_model(
        wrapper: Any,
        model_name: str,
        metrics_df: pd.DataFrame,
        feature_cols: list,
        hyperparameter: Optional[Dict] = None
):
    with mlflow.start_run(run_name=f"uv_forecast-{model_name}"):
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("feature_list", json.dumps(feature_cols))
        mlflow.log_param("sequence_length", 48)

        if hyperparameter:
            for key, value in hyperparameter.items():
                if isinstance(value, (int, float, str, bool)):
                    mlflow.log_param(f"hp_{key}", value)
        for _, row in metrics_df.iterrows():
            split = row['split']
            mlflow.log_metric(f"{split}_r2", row["r2"])
            mlflow.log_metric(f"{split}_rmse", row["rmse"])
            mlflow.log_metric(f"{split}_mae", row["mae"])
            if 'mape' in row:
                mlflow.log_metric(f"{split}_mape", row["mape"])
            if 'mbe' in row:
                mlflow.log_metric(f"{split}_mbe", row["mbe"])

        wrapper_class, model_class = get_model_wrapper_class(model_name)

        if wrapper_class is None:
            return
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            state_dict_path = f.name
        torch.save(wrapper.model.state_dict(), f.name)

        model_params = {
            "input_dim": len(feature_cols),
        }

        if hasattr(wrapper.model, 'lstm'):
            if hasattr(wrapper.model.lstm, 'hidden_size'):
                model_params['hidden_dim'] = wrapper.model.lstm.hidden_size
            if hasattr(wrapper.model.lstm, 'num_layers'):
                model_params['num_layers'] = wrapper.model.lstm.num_layers
        elif hasattr(wrapper.model, 'gru'):
            if hasattr(wrapper.model.gru, 'hidden_size'):
                model_params['hidden_dim'] = wrapper.model.gru.hidden_size
            if hasattr(wrapper.model.gru, 'num_layers'):
                model_params['num_layers'] = wrapper.model.gru.num_layers
        elif hasattr(wrapper.model, 'bilstm'):
            if hasattr(wrapper.model.bilstm, 'hidden_size'):
                model_params['hidden_dim'] = wrapper.model.bilstm.hidden_size
            if hasattr(wrapper.model.bilstm, 'num_layers'):
                model_params['num_layers'] = wrapper.model.bilstm.num_layers
                
        if hyperparameter and "dropout" in hyperparameter:
            model_params["dropout"] = hyperparameter["dropout"]

        pyfunc_wrapper = wrapper_class(
            model_class=model_class,
            model_params=model_params,
            scaler=wrapper.scaler,
            feature_cols=feature_cols,
            device="cpu"
        )

        registered_model_name = f"main.uv_models.uv_forecast-{model_name}"
        artifacts = {
            "model_state": state_dict_path,
        }

        input_shema = Schema([
            ColSpec('double', name=col) for col in feature_cols
        ])
        output_schema = Schema([ColSpec("double")])
        signature = ModelSignature(inputs=input_shema, outputs=output_schema)
        mlflow.pyfunc.log_model(
            "model",
            python_model=pyfunc_wrapper,
            artifacts=artifacts,
            registered_model_name=registered_model_name,
            signature=signature,
        )

        os.unlink(state_dict_path)
        print(f"Logged {model_name} to mlflow (registered as {registered_model_name})")

def log_prophet_model(
        wrapper: Any,
        model_name: str,
        metrics_df: pd.DataFrame,
        feature_cols: list,
        hyperparameter: Optional[Dict] = None
):
    with mlflow.start_run(run_name=f"uv_forecast-{model_name}"):
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("feature_list", json.dumps(feature_cols))
        mlflow.log_param("n_locations", len(wrapper.prophet_models) if hasattr(wrapper, 'prophet_models') else 0)

        if hyperparameter:
            for key, value in hyperparameter.items():
                if isinstance(value, (int, float, str, bool)):
                    mlflow.log_param(f"hp_{key}", value)

        for _, row in metrics_df.iterrows():
            split = row['split']
            mlflow.log_metric(f"{split}_r2", row["r2"])
            mlflow.log_metric(f"{split}_rmse", row["rmse"])
            mlflow.log_metric(f"{split}_mae", row["mae"])
            if 'mape' in row:
                mlflow.log_metric(f"{split}_mape", row["mape"])
            if 'mbe' in row:
                mlflow.log_metric(f"{split}_mbe", row["mbe"])

        prophet_models = wrapper.prophet_models if hasattr(wrapper, 'prophet_models') else []
        lgb_model = wrapper.lgb_model if hasattr(wrapper, 'lgb_model') else None
        locations = list(prophet_models.keys())

        pyfunc_wrapper = ProphetLGBWrapper(
            prophet_models=prophet_models,
            lgb_model=lgb_model,
            feature_cols=feature_cols,
            locations=locations,
        )

        registered_model_name = f"main.uv_models.uv_forecast-{model_name}"
        input_schema = Schema([
            ColSpec("double", name=col) for col in feature_cols
        ])
        output_schema = Schema([ColSpec("double")])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        mlflow.pyfunc.log_model(
            "model",
            python_model=pyfunc_wrapper,
            registered_model_name=registered_model_name,
            signature=signature,
        )
        print(f"Logged {model_name} to mlflow")

def load_hyperparameter(model_name: str) -> Optional[Dict]:
    metadata_dir = PROJECT_ROOT / "models" / "metadata"
    param_file = metadata_dir / f"best_params_{model_name}.json"
    if param_file.exists():
        with open(param_file) as f:
            return json.load(f)
    return None