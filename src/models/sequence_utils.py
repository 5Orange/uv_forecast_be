import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

SEQ_LEN = 48

def create_sequences_per_location(df, feature_cols, target_col=None, scaler_X=None, return_indices=False):
    df_sorted = df.sort_values(['location_id', 'timestamp']).copy()
    X_seqs, y_seqs, indices = [], [], []

    for loc in df['location_id'].unique():
        mask = df_sorted['location_id'] == loc
        loc_data = df_sorted[mask]
        loc_X = loc_data[feature_cols].values
        # loc_y = loc_data[target_col].values
        loc_idx = loc_data.index.values
        if target_col is not None:
            loc_y = loc_data[target_col].values

        if scaler_X is not None:
            loc_X = scaler_X.transform(loc_X)

        for i in range(SEQ_LEN - 1, len(loc_X)):
            X_seqs.append(loc_X[i - SEQ_LEN + 1:i + 1])
            if target_col is not None:
                y_seqs.append(loc_y[i])
            indices.append(loc_idx[i])

    X_seqs_array = np.array(X_seqs, dtype='float32')

    if target_col is None:
        if return_indices:
            return X_seqs_array, np.array(indices)
        return X_seqs_array, None

    y_seqs_array = np.array(y_seqs, dtype='float32')
    if return_indices:
        return X_seqs_array, y_seqs_array,  np.array(indices)
    return X_seqs_array, y_seqs_array


def create_dataloaders(X_seq, y_seq, batch_size=256, shuffle=False):
    dataset = TensorDataset(
        torch.FloatTensor(X_seq),
        torch.FloatTensor(y_seq)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
