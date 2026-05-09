import numpy as np
import pandas as pd
from pathlib import Path
import gc

from src.config import FINAL_FEATURES, FEATURES_DIR
from src.models.hyperparameter_tuning import tune_model

def main():
    print("=" * 70)
    print("UV FORECASTING - HYPERPARAMETERS TUNING")
    print("\nTuning all models with 30 trials each: ")
    print("=" * 70)

    features_path = FEATURES_DIR / 'features_regression.csv'

    needed_cols = ['timestamp', 'location_id'] + FINAL_FEATURES + ['uv_index']

    df = pd.read_csv(features_path, usecols=needed_cols, parse_dates=['timestamp'])

    print(f"Loaded {df.shape[0]:,} rows")

    numeric_cols = [c for c in df.columns if c not in ['timestamp', 'location_id']]
    for col in numeric_cols:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')

    n = len(df)
    train_end = int(0.7 * n)

    train_df = df.iloc[:train_end].copy()

    feature_cols = [c for c in FINAL_FEATURES if c in train_df.columns]

    X_train = train_df[feature_cols].values
    y_train = train_df['uv_index'].values


    print(f"Training data: {X_train.shape[0]:,} rows, {X_train.shape[1]} features")
    print(f"Locations: {train_df['location_id'].nunique()} \n")

    # traditional_ml = ['rf', 'dt', 'xgb', 'lgb', 'catboost']
    # print("=" * 70)
    # print("TUNING TRADITIONAL ML MODELS")
    # print("=" * 70)
    #
    # for model_name in traditional_ml:
    #     print("=" * 70)
    #     print(f"TUNING {model_name}")
    #     print("=" * 70)
    #     tune_model(model_name, X_train=X_train, y_train=y_train)
    #     gc.collect()

    deep_learning = ['lstm', 'gru', 'bilstm']

    print("=" * 70)
    print("TUNING DEEP LEARNING MODELS")
    print("=" * 70)

    for model_name in deep_learning:
        print("=" * 70)
        print(f"TUNING {model_name}")
        print("=" * 70)
        tune_model(model_name, df_train=train_df, feature_cols=feature_cols)
        gc.collect()


    hybrid = ['cnn_lstm', 'attention_lstm', 'prophet_lgb']
    print("=" * 70)
    print("TUNING HYBRID MODELS")
    print("=" * 70)

    for model_name in hybrid:
        print("=" * 70)
        print(f"TUNING {model_name}")
        print("=" * 70)
        tune_model(model_name, df_train=train_df, feature_cols=feature_cols)
        gc.collect()

    print("=" * 70)
    print("TUNING COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
