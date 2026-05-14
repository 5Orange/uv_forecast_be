import os
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

now = datetime.now()
target_date = datetime(2026, 5, 15)

load_dotenv()

def _parse_keys(multi_var: str, single_var: str) -> list[str]:
    raw = os.getenv(multi_var, "") or os.getenv(single_var, "")
    return [k.strip() for k in raw.split(",") if k.strip()]

WEATHERBIT_API_KEYS = _parse_keys("WEATHERBIT_API_KEYS", "WEATHERBIT_API_KEY")
OPENUV_API_KEYS = _parse_keys("OPENUV_API_KEYS", "OPENUV_API_KEY")

LOCATIONS = {
    "hcm": {"lat": 10.7769, "lon": 106.7009, "name": "Ho Chi Minh City", "type": "urban", "altitude_m": 19},
    "vungtau": {"lat": 10.4113, "lon": 107.1365, "name": "Vung Tau", "type": "coastal", "altitude_m": 4},
    "cuchi": {"lat": 11.0020, "lon": 106.5090, "name": "Cu Chi", "type": "suburban", "altitude_m": 26},
    "nhabe": {"lat": 10.6940, "lon": 106.7360, "name": "Nha Be", "type": "suburban", "altitude_m": 5},
    "thuduc": {"lat": 10.8506, "lon": 106.7710, "name": "Thu Duc City", "type": "urban", "altitude_m": 12},
    "cangio": {"lat": 10.4114, "lon": 106.9547, "name": "Can Gio Mangrove Forest", "type": "coastal_ecosystem", "altitude_m": 2},
    "longhai": {"lat": 10.4070, "lon": 107.2430, "name": "Long Hai Beach", "type": "coastal", "altitude_m": 6},
}

TIMEZONE = "Asia/Ho_Chi_Minh"
HISTORICAL_START = "2015-01-01"
HISTORICAL_END = min(now, target_date).strftime("%Y-%m-%d")

OM_HISTORICAL_HOURLY = [
    'temperature_2m', 'relative_humidity_2m', 'apparent_temperature',
    'precipitation', 'weather_code',
    'cloud_cover', 'cloud_cover_low', 'cloud_cover_mid', 'cloud_cover_high',
    'shortwave_radiation', 'direct_radiation', 'diffuse_radiation',
    'sunshine_duration',
    'wind_speed_10m', 'wind_direction_10m',
    'pressure_msl',
    'uv_index'
]
OM_HISTORICAL_DAILY = [
    'temperature_2m_max', 'temperature_2m_mean', 'temperature_2m_min',
    'apparent_temperature_max', 'apparent_temperature_min',
    'sunrise', 'sunset', 'daylight_duration', 'sunshine_duration',
    'precipitation_sum', 'precipitation_hours',
    'wind_speed_10m_max', 'wind_direction_10m_dominant',
    'weather_code', 'shortwave_radiation_sum'
]

OM_FORECAST_HOURLY = [
    'uv_index', 'uv_index_clear_sky',
    'temperature_2m', 'relative_humidity_2m', 'apparent_temperature',
    'cloud_cover', 'visibility',
    'shortwave_radiation', 'direct_radiation', 'diffuse_radiation'
]

OM_FORECAST_DAILY = [
    'uv_index_max', 'uv_index_clear_sky_max',
    'sunrise', 'sunset'
]

OM_AIR_QUALITY_HOURLY = [
    'ozone', 'pm2_5', 'pm10', 'aerosol_optical_depth',
    'dust', 'us_aqi'
]

# WHO UV risk thresholds — used in feature engineering and classification targets
UV_CATEGORIES = {
    "low":       (0,  2),
    "moderate":  (3,  5),
    "high":      (6,  7),
    "very_high": (8,  10),
    "extreme":   (11, float("inf"))
}
UV_CATEGORY_BINS   = [-0.1, 2, 5, 7, 10, float("inf")]
UV_CATEGORY_LABELS = [0, 1, 2, 3, 4]  # int labels matching uv_category target
UV_CATEGORY_NAMES  = ["low", "moderate", "high", "very_high", "extreme"]

# ── Anti-leakage feature registry ────────────────────────────────────────────
# CORE_19_FEATURES is the single source of truth for both the data-engineering
# pipeline (src/features/engr.py) and every training notebook.
# DO NOT add UV-observed or radiation-observed variables here.
CORE_19_FEATURES = [
    # Solar geometry (proxy for sun position without using measured radiation)
    "cos_solar_zenith",        # |r|=0.725 with UV — best solar geometry proxy
    "doy_sin",                 # |r|=0.202 — encodes seasonal cycle
    # Thermal / humidity
    "temperature_2m",          # |r|=0.539
    "relative_humidity_2m",    # |r|=0.504
    # Cloud cover (low vs high altitude behave differently for UV)
    "cloud_cover",             # |r|=0.272 — total cloud amount
    "cloud_cover_high",        # |r|=0.286 — cirrus (different UV impact)
    "solar_cloud_interaction", # |r|=0.554 — engineered interaction feature
    # Atmospheric composition
    "ozone_anomaly",           # |r|=0.323 — key UV absorber anomaly
    "aerosol_optical_depth",   # |r|=0.063 — aerosol loading
    "pm2_5",                   # |r|=0.132 — air quality proxy
    # General weather state
    "pressure_msl",            # |r|=0.192 — weather system proxy
    "precipitation",           # |r|=0.067 — non-linear UV suppressor
    "wind_speed_10m",          # |r|=0.117 — aerosol dispersion proxy
    # Spatial
    "altitude_m",              # key for Da Lat (1500m) vs sea-level UV
    # trend/momentum features
    "cloud_cover_change_1h",
    "temp_change_3h",
    "humidity_change_1h",
    "pressure_change_3h",
    "solar_noon_proximity"
]

CORE_14_FEATURES = CORE_19_FEATURES[:14]

FINAL_22_FEATURES = [
    "cos_solar_zenith",
    "doy_sin", 
    "temperature_2m",      
    "relative_humidity_2m",  
    "cloud_cover",             
    "solar_cloud_interaction",
    "ozone_anomaly",           
    "pressure_msl",           
    "wind_speed_10m",         
    "altitude_m",
    "cos_zenith_squared",
    "cloud_attenuation_exp",
    "temp_humidity_product",
    "pressure_cloud_interaction",
    "temperature_2m_ema",
    "cloud_cover_ema",
    "ozone_ema",
    "altitude_solar_interaction",
    "ozone_depletion_risk",
    "air_quality_combined",
    "is_raining",
    "cloud_cover_change_1h"
]

FINAL_FEATURES = FINAL_22_FEATURES

# Features that cause data leakage (observed UV energy / UV memory)
LEAKAGE_COLS = [
    "shortwave_radiation", "direct_radiation", "diffuse_radiation",
    "clearness_index", "solar_efficiency", "radiation_ratio",
    "temp_radiation_ratio", "uv_clear_sky_ratio",
    "uv_clear_sky_enhancement"
]

# UV lag/rolling pattern prefixes — excluded from regression features
UV_LAG_PREFIXES = ("uv_lag", "uv_rolling", "uv_diff", "uv_max_today")

USE_PYSICS_CONSTRAINT = True
PHYSICS_CONSTRAINT_EXPONENT = 1.0

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_OM_DIR = RAW_DIR / "open_meteo"
RAW_WB_DIR = RAW_DIR / "weatherbit"
RAW_UV_DIR = RAW_DIR / "openuv"
RAW_AQ_DIR = RAW_DIR / "air_quality"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = PROCESSED_DIR / "features"
STATIC_DIR = PROCESSED_DIR / "static"

CHECKPOINTS_DIR = PROCESSED_DIR / ".checkpoints"

for d in [RAW_OM_DIR, RAW_WB_DIR, RAW_UV_DIR, RAW_AQ_DIR, PROCESSED_DIR, FEATURES_DIR, STATIC_DIR, CHECKPOINTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)