import numpy as np
import pandas as pd
import gc
from sklearn.metrics import r2_score

from src.config import FINAL_FEATURES, FEATURES_DIR, PROJECT_ROOT
from src.models.train_optimized import train_all_models
from src.models.evaluation import evaluate_model, stratified_evaluation, mbe, mape
from src.models.mlflow_logger import (
    setup_mlflow, log_sklearn_model, log_pytorch_model,
    log_prophet_model, load_hyperparameter
)

def get_model_type(model_name):
    mapping = {
        'rf': 'Traditional ML',
        'xgb': 'Traditional ML',
        'lgb': 'Traditional ML',
        'catboost': 'Traditional ML',
        'dt': 'Traditional ML',
        'lr': 'Traditional ML',
        'lstm': 'Deep Learning',
        'gru': 'Deep Learning',
        'bilstm': 'Deep Learning',
        'cnn_lstm': 'Hybrid',
        'attention_lstm': 'Hybrid',
        'prophet_lgb': 'Hybrid',
        'stacking': 'Ensemble',
        'weighted': 'Ensemble',
        'quantile_median': 'Uncertainty Quantification'
    }

    return mapping.get(model_name, 'Other')

def load_data_optimized():
    print("=" * 70)
    print("LOAD DATA")
    print("=" * 70)

    features_path = FEATURES_DIR / 'features_regression.csv'

    needed_cols = ['timestamp', 'location_id'] + FINAL_FEATURES + ['uv_index']

    df = pd.read_csv(features_path, usecols=needed_cols, parse_dates=['timestamp'])

    print(f"loaded {df.shape[0]} rows x {df.shape[1]} columns")

    numeric_cols = [c for c in df.columns if c not in ['timestamp', 'location_id']]
    for col in numeric_cols:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')

    return df

def apply_physics_constraint(X, feature_cols, cos_zenith_col_idx):
    X_physics = X.copy()
    cos_z = X[:, cos_zenith_col_idx].clip(min=0)
    X_physics = X_physics * cos_z[:, np.newaxis]
    return X_physics

def split_data(df):
    print("\nSplitting data (70/15/15) temporal...")

    n = len(df)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    print(f" Train: {len(train_df):,} rows ({train_df['timestamp'].min()} to {train_df['timestamp'].max()})")
    print(f" Test: {len(test_df):,} rows ({test_df['timestamp'].min()} to {test_df['timestamp'].max()})")
    print(f" Val: {len(val_df):,} rows ({val_df['timestamp'].min()} to {val_df['timestamp'].max()})")

    feature_cols = [c for c in FINAL_FEATURES if c in df.columns]

    X_train = train_df[feature_cols].values
    y_train = train_df['uv_index'].values

    X_val = val_df[feature_cols].values
    y_val = val_df['uv_index'].values

    X_test = test_df[feature_cols].values
    y_test = test_df['uv_index'].values

    cos_zenith_idx = feature_cols.index('cos_solar_zenith') if 'cos_solar_zenith' in feature_cols else None

    if cos_zenith_idx is not None:
        X_train = apply_physics_constraint(X_train, feature_cols, cos_zenith_idx)
        X_val = apply_physics_constraint(X_val, feature_cols, cos_zenith_idx)
        X_test = apply_physics_constraint(X_test, feature_cols, cos_zenith_idx)

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, train_df, val_df, test_df

def split_data_per_location(df):
    print(f"\nSplitting data (70/15/15) temporal...")
    df_sorted = df.sort_values('timestamp')
    train_date = df_sorted['timestamp'].quantile(0.70)
    val_date = df_sorted['timestamp'].quantile(0.85)

    train_df = df[df['timestamp'] <= train_date].copy()
    val_df = df[(df['timestamp'] > train_date) & (df['timestamp'] <= val_date)].copy()
    test_df = df[df['timestamp'] > val_date].copy()

    print('\n per location distribution')
    for loc in sorted(df['location_id'].unique()):
        tr_count = (train_df['location_id'] == loc).sum()
        v_count = (val_df['location_id'] == loc).sum()
        te_count = (test_df['location_id'] == loc).sum()
        print(f" {loc:8s} train={tr_count:,} val={v_count:,} test={te_count:,}")

    return train_df, val_df, test_df


def evaluate_prophet_per_location(wrapper, df_split):
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    locations_metrics = []
    for loc in sorted(df_split['location_id'].unique()):
        if loc not in wrapper.prophet_models:
            print(f" warning: no prophet model for location {loc}")
            continue

        loc_data = df_split[df_split['location_id'] == loc].copy()
        y_true = loc_data['uv_index'].values

        y_pred = wrapper.predict(loc_data)
        locations_metrics.append({
            'location': loc,
            'r2': r2_score(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mape': mape(y_true, y_pred),
            'mbe': mbe(y_true, y_pred)
        })
    df_locs = pd.DataFrame(locations_metrics)
    return {
         'r2': df_locs['r2'].mean(),
         'mae': df_locs['mae'].mean(),
         'rmse': df_locs['rmse'].mean(),
         'mape': df_locs['mape'].mean(),
         'mbe': df_locs['mbe'].mean(),
         'r2_std': df_locs['r2'].std(),
         'location_breakdown': locations_metrics
     }

def main():
    print("\n" + "=" * 70)
    print("UV FORECASTING: TRAINING PIPELINE TO ACHIEVE R2 ")
    print("=" * 70)

    print("MLFlow Configuration")
    print("=" * 70)
    mlflow_enabled = setup_mlflow()
    # mlflow_enabled = False
    if mlflow_enabled:
        print("MlFlow tracking to databricks enabled")
    else:
        print("MlFlow not ready")

    df = load_data_optimized()

    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, train_df, val_df, test_df = split_data(df)
    df_full = df.copy()
    del df
    gc.collect()

    print(f"\nFeatures used: {len(feature_cols)}")
    print(f"Features: {', '.join(feature_cols)}")

    print("\n" + "=" * 70)
    print("Train Optimized base Models")
    print("=" * 70)

    base_models = train_all_models(X_train, y_train, X_val, y_val, train_df, val_df, feature_cols, df_full=df_full)

    print("\n" + "=" * 70)
    print("CONSOLIDATED EVALUATION: TRAIN / VAL / TEST")
    print("=" * 70)

    all_models = {
        'rf': base_models['rf'],
        'dt': base_models['dt'],
        'xgb': base_models['xgb'],
        'lgb': base_models['lgb'],
        'catboost': base_models['catboost'],
        'lstm': base_models['lstm'],
        'gru': base_models['gru'],
        'bilstm': base_models['bilstm'],
        'cnn_lstm': base_models['cnn_lstm'],
        'attention_lstm': base_models['attention_lstm'],
        'prophet_lgb': base_models['prophet_lgb'],
    }

    all_models = {k: v for k, v in all_models.items() if v is not None}
    prophet_per_location_data = {}
    prophet_train_df, prophet_val_df, prophet_test_df = split_data_per_location(df_full)

    all_results = []

    for split_name, (X, y, df_split) in [('train', (X_train, y_train, train_df)),
                               ('val', (X_val, y_val, val_df)),
                               ('test', (X_test, y_test, test_df))]:
        print(f"\nEvaluating on {split_name} set...")

        prophet_splits = {'train': prophet_train_df, 'val': prophet_val_df, 'test': prophet_test_df}
        prophet_df_split = prophet_splits[split_name]

        for model_name, model in all_models.items():
            if model_name == 'prophet_lgb':
                eval_result = evaluate_prophet_per_location(model, prophet_df_split)
                metrics = {
                    'model': model_name,
                    'r2': eval_result['r2'],
                    'rmse': eval_result['rmse'],
                    'mae': eval_result['mae'],
                    'mbe': eval_result['mbe'],
                    'mape': eval_result['mape']
                }
                if model_name not in prophet_per_location_data:
                    prophet_per_location_data[model_name] = {}
                prophet_per_location_data[model_name][split_name] = eval_result['location_breakdown']
            elif model_name in ['lstm', 'gru', 'bilstm',
                              'cnn_lstm', 'attention_lstm']:
                y_pred = model.predict(df_split)
                from sklearn.metrics import mean_squared_error, mean_absolute_error
                metrics = {
                    'model': model_name,
                    'r2': r2_score(y, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                    'mae': mean_absolute_error(y, y_pred),
                    'mbe': mbe(y, y_pred),
                    'mape': mape(y, y_pred),
                }
            else:
                metrics = evaluate_model(model, X, y, model_name=model_name)
            metrics['split'] = split_name
            metrics['type'] = get_model_type(model_name)
            all_results.append(metrics)

    consolidated_df = pd.DataFrame(all_results)
    consolidated_df = consolidated_df[['type','model','split', 'mae', 'rmse', 'r2', 'mape', 'mbe']]

    results_dir = PROJECT_ROOT / 'results' / 'optimized'
    results_dir.mkdir(parents=True, exist_ok=True)

    consolidated_df.to_csv(results_dir / 'consolidated_results.csv', index=False)

    if mlflow_enabled:
        print("=" * 70)
        print("Logging models to MlFlow")
        print("=" * 70)
        for model_name, model in all_models.items():
            try:
                model_metrics = consolidated_df[consolidated_df['model'] == model_name]

                hyperparameter = load_hyperparameter(model_name)
                if model_name in ['rf', 'dt', 'xgb', 'lgb', 'catboost']:
                    log_sklearn_model(model, model_name, model_metrics, feature_cols, hyperparameter)
                elif model_name in ['lstm', 'gru', 'bilstm', 'cnn_lstm', 'attention_lstm']:
                    log_pytorch_model(model, model_name, model_metrics, feature_cols, hyperparameter)
                elif model_name == 'prophet_lgb':
                    log_prophet_model(model, model_name, model_metrics, feature_cols, hyperparameter)
                else:
                    print("Unknown model type for {model_name}")
            except Exception as e:
                print(f"Failed to log {model_name} to mlFlow : {e}")

    print("\n" + "=" * 70)
    print("Consolidated Results saved to:")
    print(f"{results_dir / 'consolidated_results.csv'}")
    print("=" * 70)
    print(f"\nSample:")
    print(consolidated_df.head(12).to_string(index=False))

    print("\n" + "=" * 70)
    print("BEST MODEL PER SPLIT (by R2)")
    print("=" * 70)

    for split in ['train', 'val', 'test']:
        split_df = consolidated_df[consolidated_df['split'] == split].sort_values('r2', ascending=False)
        best = split_df.iloc[0]
        print(f"{split.upper()}: {best['model']} R2={best['r2']:.4f} RMSE={best['rmse']:.4f} MAE={best['mae']:.4f} MAPE={best['mape']:.4f}%")

    print("\n" + "=" * 70)
    print("STRATIFIED EVALUATION (best test model")
    print("=" * 70)

    best_test = consolidated_df[consolidated_df['split'] == 'test'].sort_values('r2', ascending=False).iloc[0]
    best_model_name = best_test['model']
    best_model = all_models[best_model_name]
    if best_model_name in ['lstm', 'gru', 'bilstm',
                              'cnn_lstm', 'attention_lstm', 'prophet_lgb']:
        y_pred_best = best_model.predict(test_df)
    else:
        y_pred_best = best_model.predict(X_test)

    stratified = stratified_evaluation(y_test, y_pred_best, model_name=best_model_name)
    print(stratified.to_string(index=False))

    stratified.to_csv(results_dir / 'stratified_results.csv', index=False)

    if 'prophet_lgb' in all_models and prophet_per_location_data:
        prophet_location_results = []
        for split_name in ['train', 'val', 'test']:
            if split_name in prophet_per_location_data.get('prophet_lgb', {}):
                for loc_metric in prophet_per_location_data['prophet_lgb'][split_name]:
                    prophet_location_results.append({
                        'split': split_name,
                        **loc_metric,
                    })
        if prophet_location_results:
            pd.DataFrame(prophet_location_results).to_csv(results_dir / 'prophet_lgb_location_results.csv', index=False)

    best_r2 =best_test['r2']
    best_rmse = best_test['rmse']

    print(f"\nBest Model: {best_model_name}")
    print(f"Best R2: {best_r2}\n")
    print(f"Best RMSE: {best_rmse}")


if __name__ == "__main__":
    main()
