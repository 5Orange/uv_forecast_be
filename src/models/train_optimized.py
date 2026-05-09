import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from prophet import Prophet
import json
import joblib
from pathlib import Path
import gc

from src.models.deep_learning_models import UVIndexLSTM, UVIndexGRU, UVIndexBiLSTM, CNNLSTM, AttentionLSTM
from src.models.sequence_utils import create_sequences_per_location

class SequenceModelWrapper:
    def __init__(self, model, scaler, feature_cols, device):
        self.model = model
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.device = device

    def get_params(self, deep=True):
        return {
            'model': self.model,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'device': self.device
        }
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def predict(self, df):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"SequenceModelWrapper.predict() requireds DataFrame with 'timestamp' and"
            )
        required = ['timestamp', 'location_id']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

        X_seq, seq_indices = create_sequences_per_location(
            df, self.feature_cols, target_col=None, scaler_X=self.scaler, return_indices=True)
        batch_size = 512
        predictions_seq = []

        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(X_seq), batch_size):
                batch_X = torch.FloatTensor(X_seq[i:i + batch_size]).to(self.device)
                batch_pred = self.model(batch_X).cpu().numpy()
                predictions_seq.append(batch_pred)
                del batch_X
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        predictions_seq = np.concatenate(predictions_seq)

        predictions = np.zeros(len(df))
        df_indices = df.index.values
        for seq_idx, df_idx in enumerate(seq_indices):
            pos = np.where(df_indices == df_idx)[0][0]
            predictions[pos] = predictions_seq[seq_idx]
        return predictions

def load_best_params(model_name):
    model_dir = Path(__file__).parent.parent.parent / 'models' / 'metadata'
    params_file = model_dir / f'best_params_{model_name}.json'

    if not params_file.exists():
        print(f"waring: {params_file} not found")
        return {}

    with open(str(params_file), 'r') as f:
        return json.load(f)

def _train_traditional_ml(model_class, param_key, display_name, X_train, y_train, X_val, y_val, extra_params=None):
    print(f"\nTraining {display_name}")
    params = load_best_params(param_key)
    if extra_params:
        params.update(extra_params)
    model = model_class(**params)
    fit_params = {}

    if hasattr(model, 'fit') and 'eval_set' in model.fit.__code__.co_varnames:
        fit_params['eval_set'] = [(X_val, y_val)]
        if 'verbose' in model.fit.__code__.co_varnames:
            fit_params['verbose'] = False

    model.fit(X_train, y_train, **fit_params)

    val_pred = model.predict(X_val)
    val_r2 = r2_score(y_val, val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

    print(f" Val R2: {val_r2:.4f}, RMSE: {val_rmse:.4f}")
    return model

def train_random_forest(X_train, y_train, X_val, y_val):
    return _train_traditional_ml(RandomForestRegressor, 'rf', 'Random Forest', X_train, y_train, X_val, y_val,
                                 {'random_state': 42, 'n_jobs': -1})

def train_decision_tree(X_train, y_train, X_val, y_val):
    return _train_traditional_ml(DecisionTreeRegressor, 'dt', 'Decision Tree', X_train, y_train, X_val, y_val,
                                 {'random_state': 42})

def train_xgboost(X_train, y_train, X_val, y_val):
    return _train_traditional_ml(xgb.XGBRegressor, 'xgb', 'XGBoost', X_train, y_train, X_val, y_val,
                                 {'random_state': 42, 'tree_method': 'hist'})

def train_lightgbm(X_train, y_train, X_val, y_val):
    return _train_traditional_ml(lgb.LGBMRegressor, 'lgb', 'LightGBM', X_train, y_train, X_val, y_val,
                                 {'random_state': 42, 'verbose': -1})


def train_catboost(X_train, y_train, X_val, y_val):
    return _train_traditional_ml(CatBoostRegressor, 'catboost', 'CatBoost', X_train, y_train, X_val, y_val,
                                 {'random_state': 42, 'verbose': False})

def _train_sequence_model(model_class, model_name, df_train, df_val, feature_cols, max_epochs=80, patience=10):
    print(f"\nTraining {model_name}")
    params = load_best_params(model_name.lower().replace('-', '_'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    scaler_X = StandardScaler().fit(df_train[feature_cols])
    X_seq_train, y_seq_train = create_sequences_per_location(df_train, feature_cols, 'uv_index', scaler_X)

    df_train_tail = df_train.groupby('location_id').tail(48 -1)
    df_val_with_context = pd.concat([df_train_tail, df_val], ignore_index=False)

    X_seq_val, y_seq_val, val_indices = create_sequences_per_location(
        df_val_with_context, feature_cols, 'uv_index', scaler_X, return_indices=True
    )

    val_mask = np.isin(val_indices, df_val.index)
    X_seq_val = X_seq_val[val_mask]
    y_seq_val = y_seq_val[val_mask]

    batch_size = 512

    if model_name in ['LSTM', 'GRU', 'BiLSTM']:
        model = model_class(X_seq_train.shape[2], params['hidden_dim'],
                            params['num_layers'], params['dropout']).to(device)
        criterion = nn.HuberLoss(delta=1.0)
        patience_val = 10
        scheduler_patience = 5
    elif model_name == 'CNN-LSTM':
        model = model_class(X_seq_train.shape[2], params['conv_out'], params['hidden_dim']).to(device)
        criterion = nn.MSELoss()
        patience_val = 8
        scheduler_patience = 3
    else:
        model = model_class(X_seq_train.shape[2], params['hidden_dim']).to(device)
        criterion = nn.MSELoss()
        patience_val = 8
        scheduler_patience = 3

    optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=scheduler_patience)

    train_dataset = TensorDataset(
        torch.FloatTensor(X_seq_train),
        torch.FloatTensor(y_seq_train)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True if torch.cuda.is_available() else False,
        num_workers=2,
        persistent_workers=True if torch.cuda.is_available() else False
    )

    val_dataset = TensorDataset(
        torch.FloatTensor(X_seq_val),
        torch.FloatTensor(y_seq_val)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True if torch.cuda.is_available() else False,
        num_workers=2,
    )
    scaler = GradScaler() if torch.cuda.is_available() else None

    best_val_loss = float('inf')
    patience_count = 0

    for epoch in range(max_epochs):
        model.train()
        epoch_losses = []

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad()

            if scaler is not None:
                with autocast():
                    y_pred = model(batch_X)
                    loss = criterion(y_pred, batch_y)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                y_pred = model(batch_X)
                loss = criterion(y_pred, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            epoch_losses.append(loss.item())

        model.eval()
        val_preds = []
        val_losses = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)

                if scaler is not None:
                    with autocast():
                        batch_pred = model(batch_X)
                        batch_loss = criterion(batch_pred, batch_y)
                else:
                    batch_pred = model(batch_X)
                    batch_loss = criterion(batch_pred, batch_y)

                val_preds.append(batch_pred.cpu().numpy())
                val_losses.append(batch_loss.item())

        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= patience_val:
            break

    model.eval()
    final_val_preds = []
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device, non_blocking=True)
            if scaler is not None:
                with autocast():
                    batch_pred = model(batch_X)
            else:
                batch_pred = model(batch_X)
            final_val_preds.append(batch_pred.cpu().numpy())

    y_pred_val = np.concatenate(final_val_preds)
    val_r2 = r2_score(y_seq_val, y_pred_val)
    val_rmse = np.sqrt(mean_squared_error(y_seq_val, y_pred_val))

    print(f" Val R2: {val_r2:.4f}, RMSE: {val_rmse:.4f}")

    return SequenceModelWrapper(model, scaler_X, feature_cols, device)

def train_lstm(df_train, df_val, feature_cols):
    return _train_sequence_model(UVIndexLSTM, 'LSTM', df_train, df_val, feature_cols, max_epochs=80, patience=10)

def train_gru(df_train, df_val, feature_cols):
    return _train_sequence_model(UVIndexGRU, 'GRU', df_train, df_val, feature_cols, max_epochs=80, patience=10)

def train_bilstm(df_train, df_val, feature_cols):
    return _train_sequence_model(UVIndexBiLSTM, 'BiLSTM', df_train, df_val, feature_cols, max_epochs=80, patience=10)

def train_cnn_lstm(df_train, df_val, feature_cols):
    return _train_sequence_model(CNNLSTM, 'CNN-LSTM', df_train, df_val, feature_cols, max_epochs=40, patience=8)

def train_attention_lstm(df_train, df_val, feature_cols):
    return _train_sequence_model(AttentionLSTM, 'Attention-LSTM', df_train, df_val, feature_cols, max_epochs=40, patience=8)

class ProphetLGBWrapper:
    def __init__(self, prophet_models, lgb_models, scalers, feature_cols):
        self.prophet_models = prophet_models
        self.lgb_models = lgb_models
        self.scalers = scalers
        self.feature_cols = feature_cols

    def get_params(self, deep=True):
        return {
            'prophet_models': self.prophet_models,
            'lgb_models': self.lgb_models,
            'scalers': self.scalers,
            'feature_cols': self.feature_cols,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def predict(self, df):
        results = []

        for loc in df['location_id'].unique():
            if loc not in self.prophet_models:
                continue

            loc_data = df[df['location_id'] == loc].copy().sort_values('timestamp')

            test_pr = pd.DataFrame({'ds': loc_data['timestamp']})
            for col in self.feature_cols:
                test_pr[col] = loc_data[col].ffill().bfill().values

            if loc in self.scalers:
                test_pr[self.feature_cols] = self.scalers[loc].transform(test_pr[self.feature_cols].values)

            prophet_pred = self.prophet_models[loc].predict(test_pr)['yhat'].values
            lgb_pred = self.lgb_models[loc].predict(test_pr[self.feature_cols].values)
            final_pred = (prophet_pred + lgb_pred).clip(min=0)

            loc_data['prediction'] = final_pred
            results.append(loc_data[['prediction']])
        combined = pd.concat(results)
        predictions = combined.loc[df.index, 'prediction'].values
        return predictions

def train_prophet_lgb(df_train, df_val, feature_cols):
    print(f"\nTrain Prophet LGB")
    params = load_best_params('prophet_lgb')

    models_by_loc = {}
    lgb_models_by_loc = {}
    scalers_by_loc = {}
    failed_locations = []
    total_locations = len(df_train['location_id'].unique())

    for loc in df_train['location_id'].unique():
        loc_train = df_train[df_train['location_id'] == loc].sort_values('timestamp')

        fill_loc = loc_train[feature_cols].median()
        train_clean = loc_train[['timestamp', 'uv_index'] + feature_cols].fillna(fill_loc)

        loc_scaler = StandardScaler()
        train_clean[feature_cols] = loc_scaler.fit_transform(train_clean[feature_cols].values)

        train_pr = pd.DataFrame({
            'ds': train_clean['timestamp'],
            'y': train_clean['uv_index']
        })
        for col in feature_cols:
            train_pr[col] = train_clean[col].values

        try:
            m = Prophet(growth='linear',
                        daily_seasonality=params.get('daily_seasonality', True),
                        yearly_seasonality=params.get('yearly_seasonality', True),
                        weekly_seasonality=False)
            for col in feature_cols:
                m.add_regressor(col)
            m.fit(train_pr)

            in_pred = m.predict(train_pr)['yhat'].values
            resids = train_pr['y'].values - in_pred

            lgb_model = lgb.LGBMRegressor(
                n_estimators=params.get('lgb_n_estimators', 100),
                max_depth=params.get('lgb_max_depth', 5),
                learning_rate=params.get('lgb_lr', 0.05),
                random_state=42, verbose=-1
            )
            lgb_model.fit(train_pr[feature_cols].values, resids)
            models_by_loc[loc] = m
            lgb_models_by_loc[loc] = lgb_model
            scalers_by_loc[loc] = loc_scaler
        except Exception as e:
            print(f" Prophet failed for location {loc}: {e}")
            failed_locations.append(loc)

    failure_rate = len(failed_locations) / total_locations
    if failure_rate > 0.3:
        raise RuntimeError(
            f"prophet training failed for {len(failed_locations)}/{total_locations} locations"
            f"({failure_rate:.2f}%). to many failures to proceed"
        )

    wrapper = ProphetLGBWrapper(models_by_loc, lgb_models_by_loc, scalers_by_loc, feature_cols)
    print(f"Trained ProphetLGB for {len(models_by_loc)}/{total_locations} locations")
    if failed_locations:
        print(f"failed locations: {failed_locations[:5]}{'...' if len(failed_locations) > 5 else ''}")
    y_val = df_val['uv_index'].values
    val_pred = wrapper.predict(df_val)
    val_r2 = r2_score(y_val, val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

    print(f" Val R2: {val_r2:.4f}, RMSE: {val_rmse:.4f}")
    return wrapper

def train_all_models(X_train, y_train, X_val, y_val, df_train=None, df_val=None, feature_cols=None, df_full=None):
    print("="*70)
    print("Training OPTIMIZED MODELS")
    print("="*70)

    models_dir = Path(__file__).parent.parent.parent / 'models' / 'optimized'

    models_dir.mkdir(exist_ok=True, parents=True)

    models = {}

    models['rf'] = train_random_forest(X_train, y_train, X_val, y_val)
    joblib.dump(models['rf'], models_dir / 'rf_optimized.joblib')
    gc.collect()

    models['dt'] = train_decision_tree(X_train, y_train, X_val, y_val)
    joblib.dump(models['dt'], models_dir / 'dt_optimized.joblib')
    gc.collect()

    models['xgb'] = train_xgboost(X_train, y_train, X_val, y_val)
    joblib.dump(models['xgb'], models_dir / 'xgb_optimized.joblib')
    gc.collect()

    models['lgb'] = train_lightgbm(X_train, y_train, X_val, y_val)
    joblib.dump(models['lgb'], models_dir / 'lgb_optimized.joblib')
    gc.collect()

    models['catboost'] = train_catboost(X_train, y_train, X_val, y_val)
    joblib.dump(models['catboost'], models_dir / 'catboost_optimized.joblib')
    gc.collect()

    if df_train is not None and df_val is not None and feature_cols is not None:

        if df_full is not None:
            from .train_pipeline import split_data_per_location
            prophet_train_df, prophet_val_df, prophet_test_df = split_data_per_location(df_full)
            models['prophet_lgb'] = train_prophet_lgb(prophet_train_df, prophet_val_df, feature_cols)
        else:
            models['prophet_lgb'] = train_prophet_lgb(df_train, df_val, feature_cols)
        joblib.dump(models['prophet_lgb'], models_dir / 'prophet_lgb_optimized.joblib')
        gc.collect()

        models['lstm'] = train_lstm(df_train, df_val, feature_cols)
        joblib.dump(models['lstm'], models_dir / 'lstm_optimized.joblib')
        gc.collect()

        models['gru'] = train_gru(df_train, df_val, feature_cols)
        joblib.dump(models['gru'], models_dir / 'gru_optimized.joblib')
        gc.collect()

        models['bilstm'] = train_bilstm(df_train, df_val, feature_cols)
        joblib.dump(models['bilstm'], models_dir / 'bilstm_optimized.joblib')
        gc.collect()


        models['cnn_lstm'] = train_cnn_lstm(df_train, df_val, feature_cols)
        joblib.dump(models['cnn_lstm'], models_dir / 'cnn_lstm_optimized.joblib')
        gc.collect()

        models['attention_lstm'] = train_attention_lstm(df_train, df_val, feature_cols)
        joblib.dump(models['attention_lstm'], models_dir / 'attention_lstm_optimized.joblib')
        gc.collect()

    else:
        print("SKIPPPPPPPPPPPPPPPPED")

    print("="*70)
    print("All models trained and saved")
    print("="*70)

    return models

