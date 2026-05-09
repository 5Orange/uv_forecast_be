import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
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
from pathlib import Path
import gc
import warnings
import signal
from contextlib import contextmanager

@contextmanager
def time_limit(seconds):
    def signal_handler(sig, frame):
        raise TimeoutError("Timed out after {} seconds.".format(seconds))
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


from src.models.deep_learning_models import UVIndexLSTM, UVIndexGRU, UVIndexBiLSTM, CNNLSTM, AttentionLSTM
from src.models.sequence_utils import create_sequences_per_location

def objective_rf(trial, X_train, y_train):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 800),
        'max_depth': trial.suggest_int('max_depth', 10, 25),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 15),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, 0.7]),
        'random_state': 42,
        'n_jobs': -1
    }

    tscv = TimeSeriesSplit(n_splits=3)
    scores = []
    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        model = RandomForestRegressor(**params)
        model.fit(X_tr, y_tr)
        score = r2_score(y_val, model.predict(X_val))
        scores.append(score)

    return np.mean(scores)

def objective_dt(trial, X_train, y_train):
    params = {
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 15),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'random_state': 42
    }
    tscv = TimeSeriesSplit(n_splits=3)
    scores = []
    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        model = DecisionTreeRegressor(**params)
        model.fit(X_tr, y_tr)
        score = r2_score(y_val, model.predict(X_val))
        scores.append(score)

    return np.mean(scores)


def objective_xgb(trial, X_train, y_train):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
        'random_state': 42,
        'tree_method': 'hist',
        'early_stopping_rounds': 50
    }

    tscv = TimeSeriesSplit(n_splits=3)
    scores = []

    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        score = r2_score(y_val, model.predict(X_val))
        scores.append(score)

    return np.mean(scores)

def objective_lstm(trial, df_train, feature_cols):
    hidden_dim = trial.suggest_int('hidden_dim', 128, 512, step=64)
    num_layers = trial.suggest_int('num_layers', 2, 4)
    dropout = trial.suggest_float('dropout', 0.2, 0.4)
    lr = trial.suggest_float('lr', 1e-4, 1e-3, log=True)

    scaler_X = StandardScaler().fit(df_train[feature_cols])
    X_seq, y_seq = create_sequences_per_location(df_train, feature_cols, 'uv_index', scaler_X)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tscv = TimeSeriesSplit(n_splits=3)
    batch_size = 2048
    scores = []

    for train_idx, val_idx in tscv.split(X_seq):
        X_tr_np = X_seq[train_idx]
        X_val_np = X_seq[val_idx]
        y_tr_np = y_seq[train_idx]
        y_val_np = y_seq[val_idx]

        train_dataset = TensorDataset(
            torch.FloatTensor(X_tr_np),
            torch.FloatTensor(y_tr_np)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True if torch.cuda.is_available() else False
        )

        val_dataset = TensorDataset(
            torch.FloatTensor(X_val_np),
            torch.FloatTensor(y_val_np)
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True if torch.cuda.is_available() else False
        )

        model = UVIndexLSTM(X_seq.shape[2], hidden_dim, num_layers, dropout).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.HuberLoss(delta=1.0)
        scaler = GradScaler() if torch.cuda.is_available() else None

        for epoch in range(20):
            model.train()
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

        model.eval()
        val_preds = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device, non_blocking=True)
                if scaler is not None:
                    with autocast():
                        batch_pred = model(batch_X)
                else:
                    batch_pred = model(batch_X)
                val_preds.append(batch_pred.cpu().numpy())

        y_pred_val = np.concatenate(val_preds)
        score = r2_score(y_val_np, y_pred_val)
        scores.append(score)

        del model, optimizer
        gc.collect()
    return np.mean(scores)

def objective_gru(trial, df_train, feature_cols):
    hidden_dim = trial.suggest_int('hidden_dim', 64, 256, step=32)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    dropout = trial.suggest_float('dropout', 0.1, 0.3)
    lr = trial.suggest_float('lr', 1e-4, 1e-3, log=True)

    scaler_X = StandardScaler().fit(df_train[feature_cols])
    X_seq, y_seq = create_sequences_per_location(df_train, feature_cols, 'uv_index', scaler_X)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tscv = TimeSeriesSplit(n_splits=3)
    scores = []
    batch_size = 2048

    for train_idx, val_idx in tscv.split(X_seq):
        X_tr_np = X_seq[train_idx]
        X_val_np = X_seq[val_idx]
        y_tr_np = y_seq[train_idx]
        y_val_np = y_seq[val_idx]

        train_dataset = TensorDataset(
            torch.FloatTensor(X_tr_np),
            torch.FloatTensor(y_tr_np)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True if torch.cuda.is_available() else False
        )

        val_dataset = TensorDataset(
            torch.FloatTensor(X_val_np),
            torch.FloatTensor(y_val_np)
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True if torch.cuda.is_available() else False
        )

        model = UVIndexGRU(X_seq.shape[2], hidden_dim, num_layers, dropout).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.HuberLoss(delta=1.0)
        scaler = GradScaler() if torch.cuda.is_available() else None

        for epoch in range(20):
            model.train()
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

        model.eval()
        val_preds = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device, non_blocking=True)
                if scaler is not None:
                    with autocast():
                        batch_pred = model(batch_X)
                else:
                    batch_pred = model(batch_X)
                val_preds.append(batch_pred.cpu().numpy())

        y_pred_val = np.concatenate(val_preds)
        score = r2_score(y_val_np, y_pred_val)
        scores.append(score)

        del model, optimizer
        gc.collect()
    return np.mean(scores)


def objective_bilstm(trial, df_train, feature_cols):
    hidden_dim = trial.suggest_int('hidden_dim', 32, 128, step=16)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    dropout = trial.suggest_float('dropout', 0.1, 0.3)
    lr = trial.suggest_float('lr', 1e-4, 1e-3, log=True)

    scaler_X = StandardScaler().fit(df_train[feature_cols])
    X_seq, y_seq = create_sequences_per_location(df_train, feature_cols, 'uv_index', scaler_X)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tscv = TimeSeriesSplit(n_splits=3)
    scores = []
    batch_size = 2048

    for train_idx, val_idx in tscv.split(X_seq):
        X_tr_np = X_seq[train_idx]
        X_val_np = X_seq[val_idx]
        y_tr_np = y_seq[train_idx]
        y_val_np = y_seq[val_idx]

        train_dataset = TensorDataset(
            torch.FloatTensor(X_tr_np),
            torch.FloatTensor(y_tr_np)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True if torch.cuda.is_available() else False
        )

        val_dataset = TensorDataset(
            torch.FloatTensor(X_val_np),
            torch.FloatTensor(y_val_np)
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True if torch.cuda.is_available() else False
        )

        model = UVIndexBiLSTM(X_seq.shape[2], hidden_dim, num_layers, dropout).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.HuberLoss(delta=1.0)
        scaler = GradScaler() if torch.cuda.is_available() else None

        for epoch in range(20):
            model.train()
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

        model.eval()
        val_preds = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device, non_blocking=True)
                if scaler is not None:
                    with autocast():
                        batch_pred = model(batch_X)
                else:
                    batch_pred = model(batch_X)
                val_preds.append(batch_pred.cpu().numpy())

        y_pred_val = np.concatenate(val_preds)
        score = r2_score(y_val_np, y_pred_val)
        scores.append(score)

        del model, optimizer
        gc.collect()
    return np.mean(scores)


def objective_lgb(trial, X_train, y_train):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
        'random_state': 42,
        'verbose': -1,
        'early_stopping_rounds': 50
    }

    tscv = TimeSeriesSplit(n_splits=3)
    scores = []

    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
        score = r2_score(y_val, model.predict(X_val))
        scores.append(score)

    return np.mean(scores)

def _optimize_sequence_model(trial, df_train, feature_cols, model_class, params):
    scaler_X = StandardScaler().fit(df_train[feature_cols])
    X_seq, y_seq = create_sequences_per_location(df_train, feature_cols, 'uv_index', scaler_X)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tscv = TimeSeriesSplit(n_splits=3)
    scores = []
    batch_size = 2048

    for train_idx, val_idx in tscv.split(X_seq):
        X_tr_np = X_seq[train_idx]
        X_val_np = X_seq[val_idx]
        y_tr_np = y_seq[train_idx]
        y_val_np = y_seq[val_idx]

        train_dataset = TensorDataset(
            torch.FloatTensor(X_tr_np),
            torch.FloatTensor(y_tr_np)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True if torch.cuda.is_available() else False
        )

        val_dataset = TensorDataset(
            torch.FloatTensor(X_val_np),
            torch.FloatTensor(y_val_np)
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True if torch.cuda.is_available() else False
        )

        model = model_class(X_seq.shape[2], **{k: v for k, v in params.items() if k != 'lr'}).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=1e-4)
        criterion = nn.HuberLoss(delta=1.0)
        scaler = GradScaler() if torch.cuda.is_available() else None

        for epoch in range(20):
            model.train()
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

        model.eval()
        val_preds = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device, non_blocking=True)
                if scaler is not None:
                    with autocast():
                        batch_pred = model(batch_X)
                else:
                    batch_pred = model(batch_X)
                val_preds.append(batch_pred.cpu().numpy())

        y_pred_val = np.concatenate(val_preds)
        score = r2_score(y_val_np, y_pred_val)
        scores.append(score)

        del model, optimizer
        gc.collect()
    return np.mean(scores)

def objective_cnn_lstm(trial, df_train, feature_cols):
    params = {
        'conv_out': trial.suggest_int('conv_out', 32, 128, step=16),
        'hidden_dim': trial.suggest_int('hidden_dim', 128, 512, step=64),
        'lr': trial.suggest_float('lr', 1e-4, 1e-3, log=True)
    }
    return _optimize_sequence_model(trial, df_train, feature_cols, CNNLSTM, params)

def objective_attention_lstm(trial, df_train, feature_cols):
    params = {
        'hidden_dim': trial.suggest_int('hidden_dim', 128, 512, step=64),
        'lr': trial.suggest_float('lr', 1e-4, 1e-3, log=True)
    }
    return _optimize_sequence_model(trial, df_train, feature_cols, AttentionLSTM, params)

def _optimize_hybrid_prophet_lgb(trial, df_train, feature_cols, params):
    scores = []
    locations = df_train['location_id'].unique()
    for loc_idx, loc in enumerate(locations):
        loc_data = df_train[df_train['location_id'] == loc].sort_values('timestamp')

        train_pr = pd.DataFrame({
            'ds': loc_data['timestamp'],
            'y':loc_data['uv_index'],
        })

        for col in feature_cols:
            train_pr[col] = loc_data[col].ffill().bfill().values

        split_idx = int(len(train_pr) * 0.8)
        train_pr_tr, train_pr_val = train_pr[:split_idx].copy(), train_pr[split_idx:].copy()

        try:
            with time_limit(300):
                m = Prophet(growth='linear',
                            daily_seasonality=params['daily_seasonality'],
                            yearly_seasonality=params['yearly_seasonality'],
                            weekly_seasonality=False)
                for col in feature_cols:
                    m.add_regressor(col)
                m.fit(train_pr_tr)
                warnings.filterwarnings('default')

                in_pred = m.predict(train_pr_tr)['yhat'].values
                resids = train_pr_tr['y'].values - in_pred

                lgb_model = lgb.LGBMRegressor(
                    n_estimators=params['lgb_n_estimators'],
                    max_depth=params['lgb_max_depth'],
                    learning_rate=params['lgb_lr'],
                    random_state=42, verbose=-1)
                lgb_model.fit(train_pr_tr[feature_cols].values, resids)

                val_pred_prophet = m.predict(train_pr_val)['yhat'].values
                val_pred_lgb = lgb_model.predict(train_pr_val[feature_cols].values)
                final_pred = (val_pred_prophet + val_pred_lgb).clip(min=0)

                score = r2_score(train_pr_val['y'].values, final_pred)
                scores.append(score)

                if (loc_idx + 1) % 5 == 0:
                    print(f"Progress: {loc_idx + 1} / {len(locations)} locations")

        except TimeoutError:
            print(f"Timeout on location {loc_idx + 1}/{len(locations)}, [Prophet+LGB] skipping..")
            continue
        except Exception as e:
            print(f" Error on location {loc_idx + 1}/{len(locations)}, skipping.. {type(e).__name__}: {str(e)[:80]}")
            continue
    if len(scores) < len(locations) * 0.3:
        print("TOO MANY FAILURES")
        return -999
    return np.mean(scores) if scores else -999

def objective_prophet_lgb(trial, df_train, feature_cols):
    params = {
        'daily_seasonality': trial.suggest_categorical('daily_seasonality', [True, False]),
        'yearly_seasonality': trial.suggest_categorical('yearly_seasonality', [True, False]),
        'lgb_n_estimators': trial.suggest_int('lgb_n_estimators', 50, 200),
        'lgb_max_depth': trial.suggest_int('lgb_max_depth', 3, 7),
        'lgb_lr': trial.suggest_float('lgb_lr', 0.01, 0.1)
    }
    return _optimize_hybrid_prophet_lgb(trial, df_train, feature_cols, params)

def objective_catboost(trial, X_train, y_train):
    params = {
        'iterations': trial.suggest_int('iterations', 200, 1000),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'random_seed': 42,
        'verbose': False,
        'early_stopping_rounds': 50
    }

    tscv = TimeSeriesSplit(n_splits=3)
    scores = []

    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        model = CatBoostRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
        score = r2_score(y_val, model.predict(X_val))
        scores.append(score)

    return np.mean(scores)

def tune_model(model_name, X_train=None, y_train=None, df_train=None, feature_cols=None):
    print(f"=" * 70)
    print(f"Tuning {model_name.upper()} - full optimization (10 trials)")
    print(f"=" * 70)

    objectives = {
        'rf': objective_rf,
        'dt': objective_dt,
        'xgb': objective_xgb,
        'lgb': objective_lgb,
        'catboost': objective_catboost,
        'lstm': objective_lstm,
        'gru': objective_gru,
        'bilstm': objective_bilstm,
        'cnn_lstm': objective_cnn_lstm,
        'attention_lstm': objective_attention_lstm,
        'prophet_lgb': objective_prophet_lgb,
    }

    if model_name not in objectives:
        raise ValueError(f"unknown model: {model_name}")

    needs_dataframe = model_name in ['lstm', 'gru', 'bilstm',
                                     'cnn_lstm', 'attention_lstm', 'prophet_lgb']

    if needs_dataframe:
        if df_train is None or feature_cols is None:
            raise ValueError(f"df_train and feature_cols cannot be None for {model_name}")
        objective = lambda trial: objectives[model_name](trial, df_train, feature_cols)
    else:
        if X_train is None or y_train is None:
            raise ValueError(f"X_train and y_train cannot be None for {model_name}")
        objective = lambda trial: objectives[model_name](trial, X_train, y_train)

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    study.optimize(objective, n_trials=30, show_progress_bar=True)

    print(f"\n{model_name.upper()} optimization results:")
    print(f" best CV R2: {study.best_value:.4f}")
    print(f" best params: {study.best_params}")

    models_dir = Path(__file__).parent.parent.parent / 'models' / 'metadata'
    models_dir.mkdir(exist_ok=True, parents=True)

    params_file = models_dir / f'best_params_{model_name}.json'
    with open(params_file, 'w') as f:
        json.dump(study.best_params, f, indent=2)

    print(f" saved to: {params_file}")

    return study.best_params