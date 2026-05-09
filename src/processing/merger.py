import numpy as np
import pandas as pd

import pvlib

from src.config import LOCATIONS, PROCESSED_DIR, RAW_AQ_DIR, RAW_OM_DIR, RAW_WB_DIR, TIMEZONE, RAW_UV_DIR


class DataMerger:
    def __init__(self):
        self.hourly: pd.DataFrame = pd.DataFrame()
        self.daily: pd.DataFrame = pd.DataFrame()
        self.wb_current: pd.DataFrame = pd.DataFrame()
        self.openuv_realtime: pd.DataFrame = pd.DataFrame()
        self.forecast_hourly: pd.DataFrame = pd.DataFrame()
        self.forecast_daily: pd.DataFrame = pd.DataFrame()
        self.air_quality: pd.DataFrame = pd.DataFrame()

    def load_raw(self, locations_to_keep: list = None) -> None:
        path = RAW_OM_DIR / 'historical_hourly.csv'
        if path.exists():
            if locations_to_keep:
                chunks = []
                for chunk in pd.read_csv(path, parse_dates=['timestamp'], chunksize=100000):
                    if 'location_id' in chunk.columns:
                        chunk = chunk[chunk.location_id.isin(locations_to_keep)]
                    if not chunk.empty:
                        chunks.append(chunk)
                self.hourly = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
            else:
                self.hourly = pd.read_csv(path, parse_dates=['timestamp'])
            if not self.hourly.empty:
                self.hourly['source'] = 'open_meteo'
            print(f"loaded {path.name} ({len(self.hourly)} rows)")

        path = RAW_OM_DIR / 'historical_daily.csv'
        if path.exists():
            if locations_to_keep:
                chunks = []
                for chunk in pd.read_csv(path, chunksize=100000):
                    if 'location_id' in chunk.columns:
                        chunk = chunk[chunk.location_id.isin(locations_to_keep)]
                    if not chunk.empty:
                        chunks.append(chunk)
                self.daily = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
            else:
                self.hourly = pd.read_csv(path)
            if not self.daily.empty:
                self.daily['date'] = pd.to_datetime(self.daily['date'], errors='coerce').dt.date
                self.daily = self.daily[self.daily['date'].notna()]
            print(f"loaded {path.name} ({len(self.daily)} rows)")

        wb_frames = []
        path = RAW_WB_DIR / 'current.csv'
        if path.exists():
            if locations_to_keep:
                chunks = []
                for chunk in pd.read_csv(path, chunksize=100000):
                    if 'location_id' in chunk.columns:
                        chunk = chunk[chunk.location_id.isin(locations_to_keep)]
                    if not chunk.empty:
                        chunks.append(chunk)
                if chunks:
                    wb_frames.append(pd.concat(chunks, ignore_index=True))
            else:
                wb_frames.append(pd.read_csv(path))
            if wb_frames:
                print(f"loaded {path.name} ({len(wb_frames[-1])} rows)")
        path = RAW_WB_DIR / 'historical.csv'
        if path.exists():
            if locations_to_keep:
                chunks = []
                for chunk in pd.read_csv(path, chunksize=100000):
                    if 'location_id' in chunk.columns:
                        chunk = chunk[chunk.location_id.isin(locations_to_keep)]
                    if not chunk.empty:
                        chunks.append(chunk)
                if chunks:
                    wb_frames.append(pd.concat(chunks, ignore_index=True))
            else:
                wb_frames.append(pd.read_csv(path))
            if wb_frames and len(wb_frames) > (1 if RAW_WB_DIR / 'current.csv' in [RAW_WB_DIR / 'current.csv'] and (RAW_WB_DIR / 'current.csv').exists() else 0):
                print(f"loaded {path.name} ({len(wb_frames[-1])} rows)")
        if wb_frames:
            self.wb_current = pd.concat(wb_frames, ignore_index=True)

        ouv_frames = []
        path = RAW_UV_DIR / 'realtime.csv'
        if path.exists():
            if locations_to_keep:
                chunks = []
                for chunk in pd.read_csv(path, chunksize=100000):
                    if 'location_id' in chunk.columns:
                        chunk = chunk[chunk.location_id.isin(locations_to_keep)]
                    if not chunk.empty:
                        chunks.append(chunk)
                if chunks:
                    ouv_frames.append(pd.concat(chunks, ignore_index=True))
            else:
                ouv_frames.append(pd.read_csv(path))
            if ouv_frames:
                print(f"loaded {path.name} ({len(ouv_frames[-1])} rows)")

        path = RAW_UV_DIR / 'historical.csv'
        if path.exists():
            if locations_to_keep:
                chunks = []
                for chunk in pd.read_csv(path, chunksize=100000):
                    if 'location_id' in chunk.columns:
                        chunk = chunk[chunk.location_id.isin(locations_to_keep)]
                    if not chunk.empty:
                        chunks.append(chunk)
                if chunks:
                    ouv_frames.append(pd.concat(chunks, ignore_index=True))
            else:
                ouv_frames.append(pd.read_csv(path))
            print(f"loaded {path.name} ({len(ouv_frames[-1])} rows)")
        if ouv_frames:
            self.openuv_realtime = pd.concat(ouv_frames, ignore_index=True)

        path = RAW_OM_DIR / 'recent_uv_hourly.csv'
        if path.exists():
            if locations_to_keep:
                chunks = []
                for chunk in pd.read_csv(path, parse_dates=['timestamp'], chunksize=100000):
                    if 'location_id' in chunk.columns:
                        chunk = chunk[chunk.location_id.isin(locations_to_keep)]
                    if not chunk.empty:
                        chunks.append(chunk)
                self.forecast_hourly = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
            else:
                self.forecast_hourly = pd.read_csv(path, parse_dates=['timestamp'])
            print(f"loaded {path.name} ({len(self.forecast_hourly)} rows)")

        path = RAW_OM_DIR / 'recent_uv_daily.csv'
        if path.exists():
            if locations_to_keep:
                chunks = []
                for chunk in pd.read_csv(path, parse_dates=['date'], chunksize=100000):
                    if 'location_id' in chunk.columns:
                        chunk = chunk[chunk.location_id.isin(locations_to_keep)]
                    if not chunk.empty:
                        chunks.append(chunk)
                self.forecast_daily = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
            else:
                self.forecast_daily = pd.read_csv(path, parse_dates=['date'])
            print(f"loaded {path.name} ({len(self.forecast_daily)} rows)")

        aq_frames = []
        for aq_name in ['historical.csv', 'air_quality.csv']:
            path = RAW_AQ_DIR / aq_name
            if path.exists():
                if locations_to_keep:
                    chunks = []
                    for chunk in pd.read_csv(path, parse_dates=['timestamp'] ,chunksize=100000):
                        if 'location_id' in chunk.columns:
                            chunk = chunk[chunk.location_id.isin(locations_to_keep)]
                        if not chunk.empty:
                            chunks.append(chunk)
                    if chunks:
                        aq_frames.append(pd.concat(chunks, ignore_index=True))
                else:
                    aq_frames.append(pd.read_csv(path, parse_dates=['timestamp']))
                if aq_frames:
                    print(f"loaded {path.name} ({len(aq_frames[-1])} rows)")
        if aq_frames:
            self.air_quality = pd.concat(aq_frames, ignore_index=True)
            self.air_quality.drop_duplicates(subset=['timestamp', 'location_id'], keep='last', inplace=True)

    def broadcast_daily(self)-> None:
        if self.daily.empty or self.hourly.empty:
            return
        overlap_rename = {}
        for c in self.daily.columns:
            if c in self.hourly.columns and c not in ('date', 'location_id'):
                overlap_rename[c] = f"{c}_daily"
        self.daily.rename(columns=overlap_rename, inplace=True)
        self.hourly['timestamp'] = pd.to_datetime(self.hourly['timestamp'], errors='coerce')
        self.hourly['date'] = self.hourly['timestamp'].dt.date
        daily_cols = [c for c in self.daily.columns if c not in self.hourly.columns]
        self.hourly = self.hourly.merge(self.daily[['date', 'location_id'] + daily_cols], on=['date','location_id'], how='left')

    def merge_weatherbit(self) -> None:
        if self.wb_current.empty or self.hourly.empty:
            return
        rename = {
            'uv': 'wb_uv', 'ozone': 'wb_ozone',
            'ghi': 'wb_ghi', 'dni': 'wb_dni', 'dhi': 'wb_dhi',
            'solar_rad': 'wb_solar_rad',
            'elev_angle': 'wb_elev_angle', 'h_angle': 'wb_h_angle'
        }

        wb = self.wb_current.rename(columns=rename)
        if 'ob_time' in wb.columns:
            # Use format='mixed' because current API uses "YYYY-MM-DD HH:MM"
            # while historical API returns ISO-8601 "YYYY-MM-DDTHH:MM:SS"
            ts = pd.to_datetime(wb['ob_time'], format='mixed', errors='coerce')
            if getattr(ts.dt, 'tz', None) is not None:
                ts = ts.dt.tz_convert(TIMEZONE).dt.tz_localize(None)
            wb['timestamp'] = ts.dt.floor('h')
            wb = wb[wb['timestamp'].notna()]
            merge_cols = [c for c in wb.columns if c.startswith('wb_')]
            self.hourly = self.hourly.merge(wb[['timestamp', 'location_id'] + merge_cols], on=['timestamp', 'location_id'], how='left')
            print(f'Weather bit merged - {self.hourly[merge_cols[0]].notna().sum()} rows')

    def merge_openuv(self) -> None:
        if self.openuv_realtime.empty or self.hourly.empty:
            return
        rename = {'uv': 'openuv_uv', 'uv_max': 'openuv_uv_max', 'ozone': 'openuv_ozone'}
        ouv = self.openuv_realtime.rename(columns=rename)
        if "uv_time" in ouv.columns:
            ts = pd.to_datetime(ouv['uv_time'], format='mixed', errors='coerce')
            if getattr(ts.dt, 'tz', None) is not None:
                ts = ts.dt.tz_convert(TIMEZONE).dt.tz_localize(None)
            ouv['timestamp'] = ts.dt.floor('h')
            safe_cols = [c for c in ouv.columns if c.startswith('safe_exposure_')]
            merge_cols = [c for c in ouv.columns if c .startswith('openuv_')] + safe_cols + ['sun_altitude', 'sun_azimuth']
            merge_cols = [c for c in merge_cols if c in ouv.columns]
            self.hourly = self.hourly.merge(ouv[['timestamp', 'location_id'] + merge_cols], on=['timestamp', 'location_id'], how='left')
            print(f" OpenUv merged - {self.hourly['openuv_uv'].notna().sum() if 'openuv_uv' in self.hourly.columns else 0} rows")

    def merge_forecast_uv(self) -> None:
        if self.forecast_hourly.empty or self.hourly.empty:
            return
        # Rename so forecast UV doesn't collide with archive's uv_index column
        rename = {'uv_index': 'om_uv_index', 'uv_index_clear_sky': 'om_uv_clear_sky'}
        fh = self.forecast_hourly.rename(columns=rename)
        merge_cols = [c for c in fh.columns if c.startswith('om_')] + ['visibility']
        merge_cols = [c for c in merge_cols if c in fh.columns]
        self.hourly = self.hourly.merge(
            fh[['timestamp', 'location_id'] + merge_cols],
            on=['timestamp', 'location_id'], how='left'
        )
        matched = self.hourly['om_uv_index'].notna().sum() if 'om_uv_index' in self.hourly.columns else 0
        print(f" Forecast merged - {matched} rows")

    def merge_air_quality(self) -> None:
        if self.air_quality.empty or self.hourly.empty:
            return
        aq_cols = [c for c in self.air_quality.columns if c not in ('timestamp', 'location_id')]
        existing = [c for c in aq_cols if c not in self.hourly.columns]
        if not existing:
            return
        self.hourly = self.hourly.merge(
            self.air_quality[['timestamp', 'location_id'] + existing],
            on=['timestamp', 'location_id'], how='left'
        )
        matched = self.hourly['ozone'].notna().sum() if 'ozone' in self.hourly.columns else 0
        print(f" Air quality merged - {matched} rows")

    def calculate_solar_position(self) -> None:
        if self.hourly.empty:
            return
        for loc_id, loc in LOCATIONS.items():
            mask = self.hourly['location_id'] == loc_id
            timestamp = pd.DatetimeIndex(self.hourly.loc[mask, 'timestamp']).tz_localize(TIMEZONE, ambiguous='NaT', nonexistent='shift_forward')
            if timestamp.empty:
                continue
            solpos = pvlib.solarposition.get_solarposition(timestamp, latitude=loc['lat'], longitude=loc['lon'], altitude=loc.get('altitude_m', 0))
            self.hourly.loc[mask, 'solar_elevation'] = solpos['apparent_elevation'].values
            self.hourly.loc[mask, 'solar_azimuth'] = solpos['azimuth'].values
            self.hourly.loc[mask, 'solar_zenith'] = solpos['apparent_zenith'].values
        self.hourly['cos_solar_zenith'] = np.cos(np.radians(self.hourly['solar_zenith']))
        self.hourly.loc[self.hourly['solar_elevation'] < 0, 'cos_solar_zenith'] = 0

    def estimate_uv(self) -> None:
        if self.hourly.empty:
            return

        # Start with existing uv_index from archive/forecast (if any)
        if 'uv_index' not in self.hourly.columns:
            self.hourly['uv_index'] = np.nan

        self.hourly['uv_source'] = 'none'

        # 1. Patch in WeatherBit UV
        if 'wb_uv' in self.hourly.columns:
            has_wb = self.hourly['wb_uv'].notna()
            self.hourly.loc[has_wb, 'uv_index'] = self.hourly.loc[has_wb, 'wb_uv']
            self.hourly.loc[has_wb, 'uv_source'] = 'weatherbit'

        # 2. Patch in OpenUV
        if 'openuv_uv' in self.hourly.columns:
            self.hourly['uv_daily_max_openuv'] = self.hourly['openuv_uv']

        # 3. Patch in Open-Meteo Forecast (recent/realtime, higher accuracy than archive)
        if 'om_uv_index' in self.hourly.columns:
            has_om = self.hourly['om_uv_index'].notna()
            self.hourly.loc[has_om, 'uv_index'] = self.hourly.loc[has_om, 'om_uv_index']
            self.hourly.loc[has_om, 'uv_source'] = 'open_meteo'

        real_count = (self.hourly['uv_source'] != 'none').sum()
        total_count = len(self.hourly)
        print('=' * 60)
        print(f"UV merged: {real_count}/{total_count} rows have real observations ({100*real_count/total_count:.1f}%)")
        print(f"WeatherBit: {(self.hourly['uv_source'] == 'weatherbit').sum()} rows")
        print(f"Open-Meteo: {(self.hourly['uv_source'] == 'open_meteo').sum()} rows")
        print('=' * 60)

    def derive_is_day(self) -> None:
        if "solar_elevation" in self.hourly.columns:
            self.hourly['is_day'] = (self.hourly['solar_elevation'] > 0).astype(int)
        else:
            self.hourly['is_day'] = np.nan

    def calculate_heat_index(self) -> None:
        if self.hourly.empty:
            return

        temp_f = self.hourly.get('temperature_2m', pd.Series()) * 9/5 + 32
        rh = self.hourly.get('relative_humidity_2m', pd.Series())

        hi = self._calculate_heat_index_noaa(temp_f, rh)
        self.hourly['heat_index'] = (hi - 32) * 5/9

        self.hourly['heat_stress_level'] = pd.cut(
            self.hourly['heat_index'],
            bins=[-float('inf'), 27, 32, 38, 46, float('inf')],
            labels=['comfortable', 'caution', 'extreme_caution', 'danger', 'extreme_danger']
        )

        print(f"Heat index calculated for {len(self.hourly)} rows")

    @staticmethod
    def _calculate_heat_index_noaa(temp_f: pd.Series, rh: pd.Series) -> pd.Series:
        simple = 0.5 * (temp_f + 61.0 + (temp_f - 68.0) * 1.2 + rh * 0.094 )

        use_full = ((simple + temp_f) / 2) >= 80
        c1, c2, c3, c4 = -42.379, 2.04901523, 10.14333127, -0.22475541
        c5, c6, c7, c8, c9 = -0.00683783, -0.05481717, 0.00122874, 0.00085282, -0.00000199
        full = (c1 + c2 * temp_f + c3 * rh + c4 * temp_f * rh
                + c5 * temp_f ** 2 + c6 * rh **2
                + c7 * temp_f ** 2 * rh + c8 * temp_f * rh ** 2
                + c9 * temp_f ** 2 * rh ** 2)
        low_rh = use_full & (rh < 13) & temp_f.between(80, 112)
        adj_low = ((13 - rh) / 4) * np.sqrt((17 - np.abs(temp_f - 95)) / 17)
        full = full.where(~low_rh, full - adj_low)

        high_rh = use_full & (rh > 85) & temp_f.between(80, 87)
        adj_high = ((rh - 85) / 10) * ((87 - temp_f) / 5)
        full = full.where(~high_rh, full + adj_high)
        return simple.where(~use_full, full)


    def clean(self)->None:
        if self.hourly.empty:
            return

        before = len(self.hourly)
        self.hourly.drop_duplicates(subset=['timestamp', 'location_id'], keep='last', inplace=True)
        print(f'Deduplication: {before} rows -> {len(self.hourly)}')
        self.hourly.sort_values(['location_id', 'timestamp'], inplace=True)
        self.hourly.reset_index(drop=True, inplace=True)
        ffill_exclude = {
            'timestamp', 'location_id', 'source', 'date',
            'sunrise', 'sunset', 'location_type',
            'heat_stress_level', 'weather_code', 'weather_code_daily'
        }
        ffill_cols = [c for c in self.hourly.columns
                      if c not in ffill_exclude and self.hourly[c].dtype.kind in ('f', 'i')]
        nan_before = self.hourly[ffill_cols].isna().sum()
        self.hourly[ffill_cols] = self.hourly.groupby('location_id', group_keys=False)[ffill_cols].ffill(limit=3)
        nan_after = self.hourly[ffill_cols].isna().sum()
        filled = nan_before - nan_after
        if filled.sum() > 0:
            top_filled = filled[filled > 0].sort_values(ascending=False).head(10)
            print(f"Forward-fill summary (top 10 columns):")
            for col, count in top_filled.items():
                pct = 100 * count / len(self.hourly)
                print(f"    {col}; {count} values filled ({pct:.2f}%)")
        self.hourly['location_type'] = self.hourly['location_id'].map({lid: info['type'] for lid, info in LOCATIONS.items()})
        print(f"Cleaning done - final shape: {self.hourly.shape}")

    def drop_empty_columns(self) -> None:
        if self.hourly.empty:
            return
        all_null = [c for c in self.hourly.columns if self.hourly[c].isna().all()]
        if all_null:
            self.hourly.drop(columns=all_null, inplace=True)
            print(f"Dropped {len(all_null)} null values: {all_null}")
    @staticmethod
    def clear_output_files() -> None:
        path = PROCESSED_DIR / 'merged_hourly.csv'
        if path.exists():
            path.unlink()
            print(f"Cleared existing file: {path.name}")

    def run(self, locations_to_keep: list = None) -> pd.DataFrame:
        self.load_raw(locations_to_keep=locations_to_keep)
        self.broadcast_daily()
        self.merge_weatherbit()
        self.merge_openuv()
        self.merge_forecast_uv()
        self.merge_air_quality()
        self.calculate_solar_position()
        self.calculate_heat_index()
        self.estimate_uv()
        self.derive_is_day()
        self.clean()
        self.drop_empty_columns()
        
        out_path = PROCESSED_DIR / 'merged_hourly.csv'
        mode = 'a' if out_path.exists() else 'w'
        header = not out_path.exists()
        self.hourly.to_csv(out_path, mode=mode, header=header, index=False)
        action = "Appended to" if mode == 'a' else "Saved"
        print(f" {action} {out_path.name}")
        
        return self.hourly
