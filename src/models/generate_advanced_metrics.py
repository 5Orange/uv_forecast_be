"""
Script to generate Advanced Analysis metrics for the Streamlit UI:
1. Ablation Study: Compare LSTM, BiLSTM, and Attention-LSTM.
2. Peak Hour Error: Calculate errors specifically for 10:00 - 14:00.
"""
import sys
import gc
from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.train_pipeline import load_data_optimized, split_data
from src.models.evaluation import mbe, mape

def main():
    print("=" * 60)
    print("GENERATING ADVANCED METRICS (Ablation & Peak Hour)")
    print("=" * 60)

    # 1. Load and split data
    df = load_data_optimized()
    _, _, _, _, _, y_test, feature_cols, _, val_df, test_df = split_data(df)
    
    # 2. Define the target models for the study
    models_dir = ROOT / 'models' / 'optimized'
    results_dir = ROOT / 'results' / 'optimized'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    target_models = ['lstm', 'bilstm', 'attention_lstm']
    
    results = []
    
    # Identify peak hours mask
    # Assuming test_df has a 'timestamp' column and it's a datetime object
    if not pd.api.types.is_datetime64_any_dtype(test_df['timestamp']):
        test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
        
    peak_mask = test_df['timestamp'].dt.hour.between(10, 14).to_numpy()
    y_test_peak = y_test[peak_mask]

    print(f"Total Test Samples: {len(test_df)}")
    print(f"Peak Hour Samples (10:00 - 14:00): {len(y_test_peak)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_name in target_models:
        model_path = models_dir / f"{model_name}_optimized.joblib"
        if not model_path.exists():
            print(f"Warning: {model_path} not found. Skipping {model_name}.")
            continue
            
        print(f"\nEvaluating {model_name}...")
        try:
            # Patch torch.load to handle device mapping if models were trained on GPU but evaluated on CPU
            _original_torch_load = torch.load
            def patched_torch_load(*args, **kwargs):
                kwargs["map_location"] = device
                kwargs.setdefault("weights_only", False)
                return _original_torch_load(*args, **kwargs)

            torch.load = patched_torch_load
            model = joblib.load(model_path)
            torch.load = _original_torch_load
            
            if hasattr(model, "model") and hasattr(model, "device"):
                model.model = model.model.to(device)
                model.device = device
                
            # Inject 47-hour context from val_df to fix the sequence boundary issue
            val_df_tail = val_df.groupby('location_id').tail(47)
            test_df_with_context = pd.concat([val_df_tail, test_df], ignore_index=False)
            
            # Predict with context (models will now have enough history for the first test day)
            y_pred_with_context = model.predict(test_df_with_context)
            
            # Slice out the predictions belonging to the context to match y_test shape exactly
            y_pred_all = y_pred_with_context[len(val_df_tail):]
            
            # Peak hour metrics must be computed after full-context prediction.
            # Calling predict(test_df_peak) breaks 48-step sequence context.
            y_pred_peak = y_pred_all[peak_mask]
            
            # Overall Metrics
            results.append({
                'model': model_name,
                'condition': 'Overall (0h-24h)',
                'mae': mean_absolute_error(y_test, y_pred_all),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_all)),
                'r2': r2_score(y_test, y_pred_all),
                'mape': mape(y_test, y_pred_all),
                'mbe': mbe(y_test, y_pred_all)
            })
            
            # Peak Hour Metrics
            results.append({
                'model': model_name,
                'condition': 'Peak Hours (10h-14h)',
                'mae': mean_absolute_error(y_test_peak, y_pred_peak),
                'rmse': np.sqrt(mean_squared_error(y_test_peak, y_pred_peak)),
                'r2': r2_score(y_test_peak, y_pred_peak),
                'mape': mape(y_test_peak, y_pred_peak),
                'mbe': mbe(y_test_peak, y_pred_peak)
            })
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")

    # Save to CSV
    if results:
        results_df = pd.DataFrame(results)
        out_path = results_dir / 'advanced_metrics.csv'
        results_df.to_csv(out_path, index=False)
        print(f"\nSaved advanced metrics to {out_path}")
        print(results_df)
        
        # Do not copy to UI here. This backend script only writes backend results.

if __name__ == "__main__":
    main()
