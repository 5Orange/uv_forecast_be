import numpy as np
import pandas as pd
import torch
import mlflow
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any
from src.models.deep_learning_models import (
    UVIndexLSTM, UVIndexGRU, UVIndexBiLSTM, CNNLSTM, AttentionLSTM
)
from src.models.sequence_utils import create_sequences_per_location

class DeepLearningModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model_class, model_params: Dict[str, Any],
                 scaler: StandardScaler, feature_cols: list, device: str = "cpu"):
        self.model_class = model_class
        self.model_params = model_params
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.device = device
        self.model = None

    def load_context(self, context):
        model_path = context.artifacts.get("model_state")
        if model_path:
            self.model = self.model_class(**self.model_params)
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

    def predict(self, context, model_input):
        if self.model is None:
            raise RuntimeError("model not loaded")

        if isinstance(model_input, pd.DataFrame):
            df = model_input
        else:
            df = pd.DataFrame(model_input)

        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0.0
        X_seq, seq_indices = create_sequences_per_location(
            df, self.feature_cols, None, self.scaler, return_indices=True
        )
        if len(X_seq) == 0:
            return np.zeros(len(df))

        batch_size = 512
        predictions_seq = []

        with torch.no_grad():
            for i in range(0, len(X_seq), batch_size):
                batch = torch.FloatTensor(X_seq[i:i + batch_size]).to(self.device)
                preds = self.model(batch).cpu().numpy()
                predictions_seq.append(preds)
        predictions = np.zeros(len(df))

        for idx, pred in zip(seq_indices, predictions_seq):
            predictions[df.index.get_loc(idx)] = pred

        return np.maximum(predictions, 0)

class ProphetLGBWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, prophet_models: Dict[str, Any], lgb_model: Any,
                 feature_cols: list, locations: list):
        self.prophet_models = prophet_models
        self.lgb_model = lgb_model
        self.feature_cols = feature_cols
        self.locations = locations

    def load_context(self, context):
        pass

    def predict(self, context, model_input):
        if isinstance(model_input, pd.DataFrame):
            df = model_input.copy()
        else:
            df = pd.DataFrame(model_input)

        predictions = []

        for loc in self.locations:
            loc_data = df[df['location_id'] == loc].copy()
            if len(loc_data) == 0:
                continue

            if loc in self.prophet_models:
                prophet_model = self.prophet_models[loc]
                prophet_df = pd.DataFrame({
                    'ds': loc_data['timestamp']
                })
                prophet_preds = prophet_model.predict(prophet_df)
                loc_data['prophet_pred'] = prophet_preds['yhat'].values
            else:
                loc_data['prophet_pred'] = 0.0

            lgb_features = []
            for col in self.feature_cols:
                if col in loc_data.columns:
                    lgb_features.append(loc_data[col].values)
                else:
                    lgb_features.append(np.zeros(len(loc_data)))

            if 'prophet_pred' in loc_data.columns:
                lgb_features.append(loc_data['prophet_pred'].values)

            X_lgb = np.column_stack(lgb_features)
            lgb_preds = self.lgb_model.predict(X_lgb)
            predictions.append(lgb_preds)
        return np.maximum(0, np.array(predictions))

def get_model_wrapper_class(model_name: str):
    if model_name == "lstm":
        return DeepLearningModelWrapper, UVIndexLSTM
    elif model_name == "gru":
        return DeepLearningModelWrapper, UVIndexGRU
    elif model_name == "bilstm":
        return DeepLearningModelWrapper, UVIndexBiLSTM
    elif model_name == "cnn_lstm":
        return DeepLearningModelWrapper, CNNLSTM
    elif model_name == "attention_lstm":
        return DeepLearningModelWrapper, AttentionLSTM
    elif model_name == "prophet_lgb":
        return DeepLearningModelWrapper, None
    else:
        return None, None