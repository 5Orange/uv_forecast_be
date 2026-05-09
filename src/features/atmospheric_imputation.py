import pandas as pd

def impute_atmnostpheric_location_month(df: pd.DataFrame) -> pd.DataFrame:

    atmospheric_cols = ['ozone', 'aerosol_optical_depth', 'pm2_5']

    print("=" * 70)
    print("ATMOSPHERIC DATA IMPUTATION")
    print("=" * 70)

    for col in atmospheric_cols:
        if col not in df.columns:
            continue
        missing_before = df[col].isna().sum()       

        real_data_mask = (df['timestamp'] >= '2023-01-01') & df[col].notna()

        monthly_medians = (
            df[real_data_mask]
            .groupby(['location_id', df.loc[real_data_mask, 'timestamp'].dt.month])
            [col]
            .median()
        )

        missing_mask = df[col].isna()

        for loc_id in df['location_id'].unique():
            for month in range(1, 13):
                fill_mask = (
                    missing_mask &
                    (df['location_id'] == loc_id) &
                    (df['timestamp'].dt.month == month)
                )

                if fill_mask.sum() == 0:
                    continue

                if (loc_id, month) in monthly_medians.index:
                    fill_value = monthly_medians[loc_id, month]
                else:
                    loc_mask = (df['location_id'] == loc_id) & df[col].notna()
                    if loc_mask.sum() > 0:
                        fill_value = df.loc[loc_mask, col].median()
                    else:
                        fill_value = df[col].median()
                
                df.loc[fill_mask, col] = fill_value

        missing_after = df[col].isna().sum()
        filled = missing_before - missing_after

        print(f"\n{col}:")
        print(f"before: {missing_before:,} missing ({missing_before/len(df)*100:.1f})")
        print(f"after: {missing_after:,} missing ({missing_after/len(df)*100:.1f})")
        print(f"filled: {filled:,} values")

    return df