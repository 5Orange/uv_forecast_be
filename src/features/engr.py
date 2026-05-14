import numpy as np
import pandas as pd

from src.config import (
    FEATURES_DIR, FINAL_FEATURES,
    LOCATIONS, PROCESSED_DIR,
)
from src.recommendation.safe_time_policy import SKIN_TYPE_MULTIPLIER

class FeatureEngineer:

    def __init__(self, df: pd.DataFrame | None = None):
        self.df = df.copy() if df is not None else pd.DataFrame()

    def load(self, path: str | None = None) -> "FeatureEngineer":
        path = path or str(PROCESSED_DIR / 'merged_hourly.csv')
        self.df = pd.read_csv(path, parse_dates=['timestamp'])
        print(f"Loaded merged data: {self.df.shape}")
        return self

    def add_temporal(self) -> "FeatureEngineer":
        print("Adding temporal features")
        ts  = self.df["timestamp"]
        self.df['hour'] = ts.dt.hour
        self.df['month'] = ts.dt.month
        self.df['day_of_year'] = ts.dt.dayofyear
        self.df['day_of_week'] = ts.dt.dayofweek
        self.df['hour_sin'] = np.sin(2 * np.pi * self.df['hour'] / 24)
        self.df['hour_cos'] = np.cos(2 * np.pi * self.df['hour'] / 24)
        self.df['month_sin'] = np.sin(2 * np.pi * self.df['month'] / 12)
        self.df['month_cos'] = np.cos(2 * np.pi * self.df['month'] / 12)
        self.df['doy_sin'] = np.sin(2 * np.pi * self.df['day_of_year'] / 365.25)
        self.df['doy_cos'] = np.cos(2 * np.pi * self.df['day_of_year'] / 365.25)
        if "sunrise" in self.df.columns:
            sunrise = pd.to_datetime(self.df['sunrise'], errors='coerce')
            sunset = pd.to_datetime(self.df['sunset'], errors='coerce')
            self.df['hours_since_sunrise'] = ((ts - sunrise).dt.total_seconds() / 3600).clip(lower=0)
            self.df['hours_until_sunset'] = ((sunset  - ts).dt.total_seconds() / 3600).clip(lower=0)
            daylight_h = (sunset - sunrise).dt.total_seconds() / 3600
            self.df['day_fraction'] = (self.df['hours_since_sunrise'] / daylight_h.replace(0, np.nan)).clip(0, 1)
        return self

    def add_solar_geometry(self) -> "FeatureEngineer":
        print("Adding solar geometry")
        if "solar_zenith" in self.df.columns:
            zenith_rad = np.radians(self.df['solar_zenith'].clip(lower=0))
            cos_z = np.cos(zenith_rad)
            self.df['cos_solar_zenith'] = cos_z.clip(lower=0)
            safe_cos = cos_z.replace(0, np.nan)
            self.df['air_mass'] = (1.0 / safe_cos).clip(upper=40)
            self.df.loc[self.df['solar_elevation'] <= 0, 'air_mass'] = np.nan
        if "shortwave_radiation" in self.df.columns:
            ext_rad = 1361 * self.df.get("cos_solar_zenith", 0)
            ext_rad = ext_rad.replace(0, np.nan)
            self.df['clearness_index'] = (self.df['shortwave_radiation'] / ext_rad).clip(0, 1)
        return self

    def add_uv_lag_rolling(self) -> "FeatureEngineer":
        print("Adding uv_lag_rolling features")
        if "uv_index" not in self.df.columns:
            return self
        g =self.df.groupby('location_id')['uv_index']
        self.df['uv_lag_1h'] = g.shift(1)
        self.df['uv_lag_2h'] = g.shift(2)
        self.df['uv_lag_3h'] = g.shift(3)
        self.df['uv_lag_24h'] = g.shift(24)
        self.df['uv_rolling_mean_3h'] = g.transform(
            lambda x: x.rolling(3, min_periods=1).mean()
        )

        self.df['uv_rolling_mean_6h'] = g.transform(
            lambda x: x.rolling(6, min_periods=1).mean()
        )

        self.df['uv_rolling_std_6h'] = g.transform(
            lambda x: x.rolling(6, min_periods=2).std()
        )
        self.df['uv_diff_1h'] = g.diff(1)
        self.df['uv_max_today_so_far'] = self.df.groupby(
            ['location_id', self.df['timestamp'].dt.date]
        )['uv_index'].cummax()
        return self

    def add_atmospheric(self) -> "FeatureEngineer":
        print("Adding atmospheric features")

        from src.features.atmospheric_imputation import impute_atmnostpheric_location_month

        if any(col in self.df.columns for col in ['ozone', 'aerosol_optical_depth', 'pm2_5']):
            if 'timestamp' in self.df.columns and 'location_id' in self.df.columns:
                self.df = impute_atmnostpheric_location_month(self.df)
                
        cc_low = self.df.get("cloud_cover_low", pd.Series(0, index = self.df.index))
        cc_mid = self.df.get("cloud_cover_mid", pd.Series(0, index = self.df.index))
        cc_high = self.df.get("cloud_cover_high", pd.Series(0, index = self.df.index))
        self.df['cloud_opacity'] = 0.5 * cc_low.fillna(0) + 0.3 * cc_mid.fillna(0) + 0.2 * cc_high.fillna(0)
        cc = self.df.get('cloud_cover', pd.Series(np.nan, index = self.df.index))
        self.df['cloud_group'] = pd.cut(cc, bins=[-1, 20, 60, 100], labels=['clear', 'partly', 'overcast'])
        self.df['is_raining'] = (self.df.get('precipitation', 0) > 0).astype(int)
        sw = self.df.get("shortwave_radiation", pd.Series(0, index = self.df.index)).fillna(0)
        dr = self.df.get("direct_radiation", pd.Series(0, index = self.df.index)).fillna(0)
        self.df['radiation_ratio'] = dr / (sw + 1)
        ozone_col = None
        for c in ['ozone', 'wb_ozone', 'openuv_ozone']:
            if c in self.df.columns and self.df[c].notna().any():
                ozone_col = c
                break
        if ozone_col:
            df = self.df
            df['_ozone_month'] = df['timestamp'].dt.to_period('M')
            monthly_agg = (
                df.groupby(['location_id', '_ozone_month'])[ozone_col]
                .mean()
                .rename('_ozone_monthly_mean')
            )
            shiffed = monthly_agg.groupby(level='location_id').shift(1)
            loc_global_mean = df.groupby('location_id')[ozone_col].transform('mean')
            df = df.join(
                shiffed,
                on=['location_id', '_ozone_month'],
            )
            df['_ozone_monthly_mean'] = df['_ozone_monthly_mean'].fillna(loc_global_mean)
            df['ozone_anomaly'] = df[ozone_col] - df['_ozone_monthly_mean']
            df.drop(columns=['_ozone_month', '_ozone_monthly_mean'], inplace=True)
            self.df = df
        if 'aerosol_optical_depth' in self.df.columns and self.df['aerosol_optical_depth'].notna().any():
            self.df['aerosol_uv_attenuation'] = 1 - self.df['aerosol_optical_depth'].fillna(0).clip(0, 1)
        temp = self.df.get('temperature_2m', pd.Series(np.nan, index = self.df.index))
        rh = self.df.get('relative_humidity_2m', pd.Series(np.nan, index = self.df.index))
        self.df['humidity_temp_index'] = temp * rh / 100
        return self

    def add_enhanced_atmospheric(self) -> "FeatureEngineer":
        print("Adding enhanced atmospheric features")
        
        if 'heat_index' in self.df.columns:
            self.df['heat_index_anomaly'] = (
                self.df['heat_index'] - 
                self.df.groupby([self.df['location_id'], self.df['timestamp'].dt.to_period('M')])['heat_index']
                .transform('mean')
            )

            is_stressed = self.df['heat_stress_level'] != 'comfortable'
            for loc_id in self.df['location_id'].unique():
                mask = self.df['location_id'] == loc_id
                stressed_loc = is_stressed.loc[mask]
                groups = stressed_loc.cumsum()
                hours = groups.groupby(groups).cumcount()
                hours[stressed_loc] = 0
                self.df.loc[mask, 'hours_since_heat_stress'] = hours

        return self

    def add_enhanced_solar(self) -> "FeatureEngineer":
        print("Adding enhanced solar features")
        
        if all(col in self.df.columns for col in ['shortwave_radiation', 'cos_solar_zenith']):
            theoretical_max = 1361 * self.df['cos_solar_zenith'].clip(lower=0)
            self.df['solar_efficiency'] = (
                self.df['shortwave_radiation'] / (theoretical_max + 1)
            ).clip(0, 1)
        
        if 'uv_index' in self.df.columns:
            
            if 'cloud_cover' in self.df.columns:
                self.df['uv_clear_sky_enhancement'] = (
                    self.df['uv_index'] / (self.df['cloud_cover'].fillna(50) / 100 + 0.5)
                )
        
        if 'solar_elevation' in self.df.columns:
            self.df['solar_angle_category'] = pd.cut(
                self.df['solar_elevation'],
                bins=[-90, 0, 15, 30, 45, 90],
                labels=['night', 'sunrise_sunset', 'low', 'medium', 'high']
            )
        
        return self

    def add_trend_features(self):
        if "cloud_cover" in self.df.columns:
            self.df['cloud_cover_change_1h'] = (
                self.df.groupby('location_id')['cloud_cover']
                .diff(1)
                .fillna(0)
            )

        if 'temperature_2m' in self.df.columns:
            self.df['temp_change_3h'] = (
                self.df.groupby('location_id')['temperature_2m']
                .diff(3)
                .fillna(0)
            )
        if 'relative_humidity_2m' in self.df.columns:
            self.df['humidity_change_1h'] = (
                self.df.groupby('location_id')['relative_humidity_2m']
                .diff(1)
                .fillna(0)
            )
        if 'pressure_msl' in self.df.columns:
            self.df['pressure_change_3h'] = (
                self.df.groupby('location_id')['pressure_msl']
                .diff(3)
                .fillna(0)
            )

        if 'hour' in self.df.columns:
            self.df['solar_noon_proximity'] = 1 - np.abs(self.df['hour'] - 12) / 12

    def add_tourism_features(self) -> "FeatureEngineer":
        print("Adding tourism-specific features")
        
        if all(col in self.df.columns for col in ['temperature_2m', 'uv_index', 'wind_speed_10m']):
            beach_score = (
                (1 - np.minimum(np.abs(self.df['temperature_2m'] - 28) / 15, 1)) * 0.4 +
                (1 - np.minimum(self.df['uv_index'] / 8, 1)) * 0.3 +
                (1 - np.minimum(np.maximum(self.df['wind_speed_10m'] - 15, 0) / 20, 1)) * 0.3
            )
            self.df['beach_suitability'] = beach_score.clip(0, 1)
        
        if 'outdoor_suitability' in self.df.columns:
            if 'heat_index' in self.df.columns:
                heat_penalty = np.maximum(0, (self.df['heat_index'] - 32) / 15)
                self.df['outdoor_suitability_enhanced'] = (
                    self.df['outdoor_suitability'] * (1 - heat_penalty)
                ).clip(0, 1)
        
        if all(col in self.df.columns for col in ['weather_code', 'precipitation']):
            wc = self.df['weather_code']
            weather_score = np.where(wc <= 3, 1.0, np.where(wc <= 48, 0.6, np.where(wc <= 67, 0.2, 0.0)))
            rain_score = 1 - np.minimum(self.df['precipitation'] / 5, 1)
            if 'visibility' in self.df.columns and self.df['visibility'].notna().any():
                visibility_score = np.minimum(self.df['visibility'].fillna(10) / 10 , 1)
                self.df['sightseeing_suitability'] = (
                        visibility_score * 0.4 + weather_score * 0.3 + rain_score * 0.3
                ).clip(0, 1)
            else:
                self.df['sightseeing_suitability'] = (
                    weather_score * 0.5 + rain_score * 0.5
                ).clip(0, 1)
        
        return self

    def add_health_safety_features(self) -> "FeatureEngineer":
        print("Adding health and safety features")
        
        if 'uv_index' in self.df.columns:
            self.df['uv_risk_level'] = pd.cut(
                self.df['uv_index'],
                bins=[-0.1, 2, 5, 7, 10, float('inf')],
                labels=['low', 'moderate', 'high', 'very_high', 'extreme']
            )
            
            uv_factor = np.maximum(1.0, self.df['uv_index'])
            for skin_type in [1, 2, 3, 4, 5, 6]:
                multiplier = SKIN_TYPE_MULTIPLIER[skin_type]
                api_col = f'safe_exposure_st{skin_type}'
                est_col = f'estimated_safe_st{skin_type}'
                if api_col in self.df.columns and self.df[api_col].notna().any():
                    self.df[est_col] = self.df[api_col]
                    missing = self.df[est_col].isna()
                    self.df.loc[missing, est_col] = (200.0 * multiplier) / (3.0 * uv_factor.loc[missing])
                else:
                    self.df[est_col] = (200.0 * multiplier) / (3.0 * uv_factor)

        if 'heat_index' in self.df.columns:
            self.df['heat_warning'] = (self.df['heat_index'] > 32).astype(int)
            self.df['extreme_heat_warning'] = (self.df['heat_index'] > 38).astype(int)
        
        if all(col in self.df.columns for col in ['uv_index', 'heat_index']):
            uv_risk = self.df['uv_index'] / 11
            heat_risk = np.maximum(0, (self.df['heat_index'] - 27) / 20)
            self.df['combined_health_risk'] = np.maximum(uv_risk, heat_risk)
        
        return self

    def add_interactions(self) -> "FeatureEngineer":
        print("Adding interaction features")
        cos_z = self.df.get('cos_solar_zenith', pd.Series(0, index = self.df.index))
        cc = self.df.get('cloud_cover', pd.Series(0, index = self.df.index)).fillna(0)/100
        self.df['solar_cloud_interaction'] = cos_z * (1 - cc)
        uv = self.df.get('uv_index', pd.Series(np.nan, index = self.df.index))
        sw = self.df.get('shortwave_radiation', pd.Series(0, index = self.df.index)).fillna(0)
        uv_clear_sky = (sw / 25.0).clip(lower=0)
        self.df['uv_clear_sky_ratio'] = uv / (uv_clear_sky + 0.01)
        temp = self.df.get('temperature_2m', pd.Series(np.nan, index = self.df.index))
        sw = self.df.get('shortwave_radiation', pd.Series(0, index = self.df.index)).fillna(0)
        self.df['temp_radiation_ratio'] = temp / (sw + 1)
        return self

    def add_targets(self) -> "FeatureEngineer":
        print("Adding targets")
        uv = self.df.get('uv_index', pd.Series(np.nan, index = self.df.index))
        self.df['uv_category'] = pd.cut(uv, bins=[-0.1, 2, 5, 7, 10, float('inf')], labels=[0, 1, 2, 3, 4]).astype("Int64")
        self.df['uv_category_name'] = pd.cut(uv, bins=[-0.1, 2, 5, 7, 10, float('inf')], labels=['low', 'moderate', 'high', 'very_high', 'extreme'])
        temp = self.df.get('temperature_2m', pd.Series(np.nan, index = self.df.index))
        self.df['comfort_temp_score'] = (1- np.minimum(np.abs(temp - 27) / 10, 1.0)).clip(0, 1)
        uv_penalty = (uv.fillna(0) / 11).clip(0, 1)
        rain_penalty = self.df.get('is_raining', pd.Series(0, index = self.df.index))
        self.df['outdoor_suitability'] = ((1 - uv_penalty) * (1 - rain_penalty) * self.df['comfort_temp_score']).clip(0, 1)
        self.df['indoor_preference'] = 1 - self.df['outdoor_suitability']
        hour = self.df.get('hour', self.df['timestamp'].dt.hour)
        conditions = [hour.between(5, 7), hour.between(8, 10), hour.between(11, 14), hour.between(15, 17), hour.between(18, 20)]
        labels = ['early_morning', 'morning', 'midday', 'afternoon', 'evening']
        self.df['time_slot'] = np.select(conditions, labels, default='night')
        return self

    def add_location(self)->"FeatureEngineer":
        print("Adding location")
        self.df['altitude_m'] = self.df['location_id'].map({lid: info['altitude_m'] for lid, info in LOCATIONS.items()})
        if 'location_type' not in self.df.columns:
            self.df['location_type'] = self.df['location_id'].map({lid: info['type'] for lid, info in LOCATIONS.items()})
        return self
    
    def add_advanced_features(self)-> "FeatureEngineer":
        print("adding  advanced features")
        
        cos_z = self.df.get('cos_solar_zenith', pd.Series(0, index=self.df.index))
        self.df['cos_zenith_squared'] = cos_z ** 2

        cc = self.df.get('cloud_cover', pd.Series(50, index=self.df.index)).fillna(50)
        self.df['cloud_attenuation_exp'] = np.exp(-cc / 50)

        temp = self.df.get('temperature_2m', pd.Series(np.nan, index=self.df.index))
        rh = self.df.get('relative_humidity_2m', pd.Series(np.nan, index=self.df.index))
        self.df['temp_humidity_product'] = temp * rh / 100

        pressure = self.df.get('pressure_msl', pd.Series(1013, index=self.df.index))
        self.df['pressure_cloud_interaction'] = pressure * (1 - cc/100)

        for col in ['temperature_2m', 'cloud_cover', 'ozone']:
            if col in self.df.columns:
                self.df[f'{col}_ema'] = self.df.groupby('location_id')[col].transform(
                    lambda x: x.ewm(alpha=0.3, adjust=False).mean()
                )

        altitude = self.df.get('altitude_m', pd.Series(0, index=self.df.index))
        self.df['altitude_solar_interaction'] = altitude * cos_z / 1000

        if 'ozone' in self.df.columns:
            ozone_q25 = self.df['ozone'].quantile(0.25)
            self.df['ozone_depletion_risk'] = (self.df['ozone'] < ozone_q25).astype(int)

        aod = self.df.get('aerosol_optical_depth', pd.Series(0, index=self.df.index)).fillna(0)
        pm25 = self.df.get('pm2_5', pd.Series(0, index=self.df.index)).fillna(0)
        self.df['air_quality_combined'] = 0.6 * aod + 0.4 * (pm25 / 100)
        return self

    def run_all(self) -> pd.DataFrame:
        self.df = self.df.copy()  # defragment before bulk column additions (avoids PerformanceWarning)
        self.add_temporal()
        self.add_solar_geometry()
        self.add_enhanced_solar()
        self.add_uv_lag_rolling()
        self.add_atmospheric()
        self.add_location()
        self.add_advanced_features()
        self.add_trend_features()
        self.add_enhanced_atmospheric()
        self.add_interactions()
        self.add_targets()
        self.add_tourism_features()
        self.add_health_safety_features()

        daytime = self.df[self.df.get('is_day', pd.Series(1, index=self.df.index)) == 1].copy()

        if 'uv_source' in daytime.columns:
            before_count = len(daytime)
            real_data = daytime[
                (daytime['uv_source'].isin(['weatherbit', 'open_meteo'])) &
                (daytime['uv_index'].notna())
            ].copy()
            print('=' * 60)
            print(f"Filtered to ral UV data: {len(real_data)}/{before_count} rows" +
                  f"({100*len(real_data)/before_count:.1f}% of daytime data)")
            print(f'WeatherBit: {(real_data["uv_source"] == "weatherbit").sum()} rows')
            print(f'Open_Meteo: {(real_data["uv_source"] == "open_meteo").sum()} rows')
            print('=' * 60)
        else:
            print("WARNING: uv_source column not  found, filtering by uv_index non-null only")
            real_data = daytime[daytime['uv_index'].notna()].copy()

        self._save(real_data[self._select_columns('regression')], 'features_regression.csv')
        self._save(real_data[self._select_columns('classification')], 'features_classification.csv')
        self._save(real_data[self._select_columns('recommendation')], 'features_recommendation.csv')
        self._save(real_data[self._select_columns('tourism')], 'features_tourism.csv')
        self._save(real_data[self._select_columns('health')], 'features_health.csv')

        self._save(self.df, 'features_full.csv')
        return self.df

    def _select_columns(self, task: str) -> list[str]:
        base = ['timestamp', 'location_id', 'uv_source']
        temporal = ['hour', 'month', 'day_of_year', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'doy_sin', 'doy_cos', 'day_fraction', 'hours_since_sunrise', 'hours_until_sunset']
        
        solar_enhanced = [
            'solar_elevation', 'solar_azimuth', 'solar_zenith', 'cos_solar_zenith', 'air_mass' , 'solar_angle_category'
        ]
        
        atmospheric_enhanced = [
            'temperature_2m', 'relative_humidity_2m', 'heat_index', 'heat_index_anomaly',
            'cloud_cover', 'cloud_cover_low', 'cloud_cover_mid', 'cloud_cover_high', 'cloud_opacity',
            'pressure_msl',
            'precipitation', 'wind_speed_10m', 'wind_direction_10m',
            'is_raining', 'humidity_temp_index', 'ozone_anomaly', 'aerosol_uv_attenuation',
            'aerosol_optical_depth', 'pm2_5'
        ]

        interactions = ['solar_cloud_interaction']
        
        health_features = [
            'uv_index', 'uv_risk_level', 'heat_index', 'heat_stress_level', 'heat_warning', 'extreme_heat_warning',
            'combined_health_risk', 'hours_since_heat_stress', 'estimated_safe_st1', 'estimated_safe_st2',
            'estimated_safe_st3', 'estimated_safe_st4', 'estimated_safe_st5', 'estimated_safe_st6'
        ]
        
        tourism_features = [
            'beach_suitability', 'outdoor_suitability_enhanced', 'sightseeing_suitability',
            'comfort_temp_score', 'outdoor_suitability', 'indoor_preference', 'time_slot'
        ]
        
        if task == 'regression':

            core = [c for c in FINAL_FEATURES if c in self.df.columns]
            return self._filter_existing(base + temporal + core + ['uv_index'])
        elif task == 'classification':
            return self._filter_existing(base + temporal + solar_enhanced + atmospheric_enhanced +  interactions + ['uv_category', 'uv_category_name'])
        elif task == 'recommendation':
            return self._filter_existing(
                base + temporal + solar_enhanced + atmospheric_enhanced
                + interactions + tourism_features + ['uv_index']
            )
        elif task == 'tourism':
            return self._filter_existing(
                base + temporal + solar_enhanced + atmospheric_enhanced
                 + interactions + tourism_features + health_features
                )
        elif task == 'health':
            return self._filter_existing(base + temporal + solar_enhanced + atmospheric_enhanced + health_features)
        
        return list(self.df.columns)

    def _filter_existing(self, cols: list[str]) -> list[str]:
        seen = set()
        result = []
        for c in cols:
            if c in self.df.columns and c not in seen:
                result.append(c)
                seen.add(c)
        return result

    @staticmethod
    def _save(df: pd.DataFrame, name: str) -> None:
        path = FEATURES_DIR / name
        mode = 'a' if path.exists() else 'w'
        header = not path.exists()
        df.to_csv(path, mode=mode, header=header, index=False)
        action = "Appended to" if mode == 'a' else "Saved"
        print(f" {action} {name} -> {path} ({df.shape[0]} rows x {df.shape[1]} columns)")

    @staticmethod
    def clear_output_files() -> None:
        names = [
            'features_regression.csv', 'features_classification.csv',
            'features_recommendation.csv', 'features_tourism.csv',
            'features_health.csv', 'features_full.csv'
        ]
        for name in names:
            path = FEATURES_DIR / name
            if path.exists():
                path.unlink()
                print(f"Cleared existing file: {path.name}")
