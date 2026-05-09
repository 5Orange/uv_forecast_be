import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

def mbe(y_true, y_pred):
    return np.mean(y_true - y_pred)

def mape(y_true, y_pred, epsilon=0.1):
    return np.mean(np.abs((y_true - y_pred) / np.abs(y_true + epsilon))) * 100

def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    bias = mbe(y_test, y_pred)
    mape_score = mape(y_test, y_pred)

    return {
        'model': model_name,
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'mbe': bias,
        'mape': mape_score,
    }

def stratified_evaluation(y_true, y_pred, model_name="Model"):
    uv_ranges = [
        (0, 2, 'Low'),
        (3, 5, 'Moderate'),
        (6, 7, 'High'),
        (8, 10, 'Very High'),
        (11, 20, 'Extreme'),
    ]

    results = []

    for uv_min, uv_max, label in uv_ranges:
        mask = (y_true >= uv_min) & (y_true < uv_max)

        if mask.sum() > 10:
            y_t = y_true[mask]
            y_p = y_pred[mask]

            mae = mean_absolute_error(y_t, y_p)
            rmse = np.sqrt(mean_squared_error(y_t, y_p))
            bias = mbe(y_t, y_p)

            results.append({
                'model': model_name,
                'uv_range': label,
                'n_samples': mask.sum(),
                'mae': mae,
                'rmse': rmse,
                'mbe': bias,
            })

    return pd.DataFrame(results)

def cross_validate_model(model, X, y, cv=5):
    tscv = TimeSeriesSplit(n_splits=cv)
    scores = []

    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model_clone = model.__class__(**model.get_params())
        model_clone.fit(X_tr, y_tr)

        y_pred = model_clone.predict(X_val)
        score = r2_score(y_val, y_pred)
        scores.append(score)

    return {
        'mean_r2': np.mean(scores),
        'std_r2': np.std(scores),
        'cv_scores': scores,
    }

def compare_models(models_dict, X_test, y_test):
    results = []

    for name, model in models_dict.items():
        metrics = evaluate_model(model, X_test, y_test, model_name=name)
        results.append(metrics)

    df = pd.DataFrame(results)
    df = df.sort_values('r2', ascending=False)
    return df