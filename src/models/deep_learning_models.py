import torch
import torch.nn as nn

class UVIndexLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)

class UVIndexGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :]).squeeze(-1)

class UVIndexBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        out, _ = self.bilstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)

class CNNLSTM(nn.Module):
    def __init__(self, input_dim, conv_out=64, hidden_dim=256, num_layers=2, dropout=0.2):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, conv_out, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(conv_out)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(
            conv_out, hidden_dim, num_layers=2,
            batch_first=True, dropout=0.2
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.relu(self.bn(self.conv(x)))
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)

class AttentionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.attn_w = nn.Linear(hidden_dim, 1)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        attn_scores = torch.softmax(self.attn_w(out), dim=1)
        context = torch.sum(attn_scores * out, dim=1)
        return self.head(context).squeeze(-1)

# ── DLinear (AAAI 2023) ────────────────────────────────────────────────────
# Reference: Zeng et al. "Are Transformers Effective for Time Series Forecasting?" (2023)
# Decomposes the sequence into Trend (moving avg) + Seasonality (residual),
# then applies one Linear layer per component. Extremely fast and competitive.
class DLinear(nn.Module):
    def __init__(self, input_dim: int, seq_len: int = 48, ma_window: int = 13):
        """
        Args:
            input_dim:  Number of input features (F).
            seq_len:    Lookback window length (T).
            ma_window:  Moving-average kernel size for trend decomposition.
                        Must be odd. Optuna will tune this.
        """
        super().__init__()
        # Ensure kernel is odd for symmetric padding
        self.ma_window = ma_window if ma_window % 2 == 1 else ma_window + 1
        self.avg = nn.AvgPool1d(kernel_size=self.ma_window,
                                stride=1,
                                padding=self.ma_window // 2)
        # Each feature channel has its own linear map: (T,) -> (1,)
        in_features = seq_len * input_dim
        self.trend_proj = nn.Linear(in_features, 1)
        self.season_proj = nn.Linear(in_features, 1)

    def forward(self, x):
        # x: (B, T, F)
        # Trend via per-feature moving average
        x_t = x.transpose(1, 2)                    # (B, F, T)
        trend = self.avg(x_t).transpose(1, 2)       # (B, T, F)  — handles edge padding
        season = x - trend
        # Flatten to (B, T*F)
        B = x.size(0)
        trend_flat  = trend.reshape(B, -1)
        season_flat = season.reshape(B, -1)
        return (self.trend_proj(trend_flat) + self.season_proj(season_flat)).squeeze(-1)


# ── TimesNet (ICLR 2023) ───────────────────────────────────────────────────
# Reference: Wu et al. "TimesNet: Temporal 2D-Variation Modeling for General
#            Time Series Analysis" (2023)
# FFT identifies top-k dominant periods. The 1D sequence is reshaped into
# k 2D (period × frequency) tensors and processed with 2D Inception blocks.

class _InceptionBlock(nn.Module):
    """Simplified 2D Inception block with 3 parallel convolution scales."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        mid = max(out_ch // 4, 1)
        self.branch1 = nn.Conv2d(in_ch, mid,  kernel_size=1)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_ch, mid, kernel_size=1),
            nn.Conv2d(mid, mid, kernel_size=3, padding=1),
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_ch, mid, kernel_size=1),
            nn.Conv2d(mid, mid, kernel_size=5, padding=2),
        )
        self.pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_ch, mid, kernel_size=1),
        )
        self.proj = nn.Conv2d(mid * 4, out_ch, kernel_size=1)
        self.norm = nn.BatchNorm2d(out_ch)
        self.act  = nn.GELU()

    def forward(self, x):
        b1 = self.branch1(x)
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        bp = self.pool(x)
        out = torch.cat([b1, b3, b5, bp], dim=1)
        return self.act(self.norm(self.proj(out)))


class TimesNet(nn.Module):
    def __init__(self, input_dim: int, seq_len: int = 48,
                 top_k: int = 3, d_model: int = 64, num_kernels: int = 4):
        """
        Args:
            input_dim:   Number of input features (F).
            seq_len:     Lookback window length (T).
            top_k:       Number of dominant periods extracted by FFT.
            d_model:     Internal channel size after projection.
            num_kernels: Number of stacked Inception blocks.
        """
        super().__init__()
        self.seq_len    = seq_len
        self.top_k      = top_k
        self.d_model    = d_model

        # Project F features to d_model channels
        self.input_proj = nn.Linear(input_dim, d_model)

        # Stack of Inception blocks operating in 2D (period × freq)
        blocks = []
        for i in range(num_kernels):
            blocks.append(_InceptionBlock(d_model, d_model))
        self.inception = nn.ModuleList(blocks)
        self.layer_norm = nn.LayerNorm(d_model)

        # Output head: flatten and regress to scalar
        self.head = nn.Sequential(
            nn.Linear(d_model * seq_len, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def _period_reshape(self, x: torch.Tensor, period: int):
        """Pad x to be divisible by period then reshape to 2D."""
        B, T, C = x.shape
        # How many rows of `period` length we need
        n_rows = (T + period - 1) // period
        pad_len = n_rows * period - T
        if pad_len > 0:
            x = torch.cat([x, x[:, :pad_len, :]], dim=1)   # wrap-pad
        # (B, n_rows * period, C)  ->  (B, C, n_rows, period)
        x2d = x.reshape(B, n_rows, period, C).permute(0, 3, 1, 2)
        return x2d, n_rows

    def forward(self, x):
        # x: (B, T, F)
        B, T, _ = x.shape

        # 1. Project features
        x = self.input_proj(x)                  # (B, T, d_model)

        # 2. FFT to find top-k dominant periods
        # NOTE: cuFFT in half-precision requires power-of-2 sizes.
        # We cast to float32 here to avoid the restriction regardless of AMP context.
        x_f32  = x.float()
        x_freq = torch.fft.rfft(x_f32.permute(0, 2, 1), dim=-1)  # (B, d_model, T//2+1)
        amp    = x_freq.abs().mean(dim=1)                          # (B, T//2+1)
        amp[:, 0] = 0                                          # remove DC component
        topk_amp, topk_idx = torch.topk(amp, self.top_k, dim=-1)  # (B, top_k)

        # Periods are T / freq_index; clip to [2, T]
        periods = (T / topk_idx.float()).clamp(2, T).long()    # (B, top_k)
        # Use the most common period across the batch for reshape
        period_vals = periods.mode(dim=0).values               # (top_k,)  CPU-safe

        # 3. Process each dominant period with Inception blocks
        agg = torch.zeros(B, T, self.d_model, device=x.device)
        for p in period_vals:
            p = int(p.item())
            x2d, n_rows = self._period_reshape(x, p)    # (B, d_model, n_rows, p)
            for block in self.inception:
                x2d = block(x2d)                         # same shape
            # Collapse 2D back to 1D: (B, d_model, n_rows, p) -> (B, n_rows*p, d_model)
            x1d = x2d.permute(0, 2, 3, 1).reshape(B, n_rows * p, self.d_model)
            agg = agg + x1d[:, :T, :]                   # trim back to T
        agg = agg / self.top_k                           # mean over periods
        agg = self.layer_norm(agg + x)                   # residual + norm

        # 4. Flatten and regress
        out = agg.reshape(B, -1)                         # (B, T * d_model)
        return self.head(out).squeeze(-1)
