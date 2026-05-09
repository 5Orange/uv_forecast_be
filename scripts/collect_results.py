import pandas as pd
from pathlib import Path
from src.config import PROJECT_ROOT, FEATURES_DIR
from src.models.train_pipeline import load_data_optimized, split_data_per_location
import sys
def get_model_type(model_name):
    model_lower = model_name.lower()

    if any(x in model_lower for x in ['lstm', 'gru', 'bilstm']):
        return 'Deep learning'
    elif any(x in model_lower for x in ['prophet+lgb', 'cnn-lstm', 'stacking' , 'attention-lstm']):
        return 'Hybrid'
    elif any(x in model_lower for x in ['xgboost', 'lightgbm', 'catboost', 'randomforest', 'decisiontree', 'linearregression']):
        return 'Traditional'
    elif any(x in model_lower for x in ['prophet']):
        return 'Modern'
    else:
        return 'Other'

def collect_results():
    results_dir = PROJECT_ROOT / 'results'

    csv_files = [
        'regression_results.csv',
        'timeseries_results.csv',
        'hybrid_results.csv',
        'modern_results.csv',
    ]

    all_results = []
    for csv_file in csv_files:
        csv_path = results_dir / csv_file
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            all_results.append(df)
        else:
            print(f'File {csv_file} not found.')

    if not all_results:
        print('No results found.')
        return None

    consolidated = pd.concat(all_results, ignore_index=True)

    consolidated['type'] = consolidated['model'].apply(get_model_type)

    wide_data = []
    for model_name in consolidated['model'].unique():
        model_df = consolidated[consolidated['model'] == model_name]
        model_type = model_df['type'].iloc[0]

        row = {'type': model_type, 'model_name': model_name}

        for split in ['train', 'val', 'test']:
            split_data = model_df[model_df['split'] == split]
            if len(split_data) > 0:
                row[f'{split}_MAE'] = split_data['MAE'].iloc[0]
                row[f'{split}_RMSE'] = split_data['RMSE'].iloc[0]
                row[f'{split}_MAPE'] = split_data['MAPE'].iloc[0]
                row[f'{split}_R2'] = split_data['R2'].iloc[0]
            else:
                row[f'{split}_MAE'] = None
                row[f'{split}_RMSE'] = None
                row[f'{split}_MAPE'] = None
                row[f'{split}_R2'] = None
        wide_data.append(row)

    wide_df = pd.DataFrame(wide_data)

    column_order = ['type', 'model_name']
    for split in ['train', 'val', 'test']:
        column_order.extend([f'{split}_MAE', f'{split}_RMSE', f'{split}_MAPE', f'{split}_R2'])

    wide_df = wide_df[column_order]
    wide_df = wide_df.sort_values(by=['type', 'model_name'])

    excel_path = results_dir / 'consolidated_results.xlsx'
    wide_df.to_excel(excel_path, index=False, sheet_name='consolidated')

    return  wide_df

if __name__ == '__main__':

    try:
        df = collect_results()
        if df is not None:
            print("COMPLETE")
            sys.exit(0)
        else:
            print("FAILED")
            sys.exit(1)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # optimez_path = PROJECT_ROOT / 'results' / 'optimized'
    # a = optimez_path / 'consolidated_results.csv'
    # b = optimez_path / 'prophet_lgb_location_results.csv'
    #
    #
    # df_a = pd.read_csv(a)
    # df_b = pd.read_csv(b)
    # with pd.ExcelWriter(optimez_path / 'consolidated_results.xlsx') as writer:
    #     # df_a.to_excel(writer, index=False, sheet_name='consolidated')
    #     df_b.to_excel(writer, index=False, sheet_name='prophet_lgb_location_results')