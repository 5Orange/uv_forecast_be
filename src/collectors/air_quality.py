import time

import pandas as pd
import requests

from src.config import LOCATIONS, OM_AIR_QUALITY_HOURLY, RAW_AQ_DIR, TIMEZONE
from src.utils.checkpoint import Checkpoint, append_csv

AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

class AirQualityCollector:
    def __init__(self, retry: int =3, pause: float = 1.0):
        self.retry = retry
        self.pause = pause
        self.session = requests.Session()

    def _get(self, url: str, params: dict) -> dict:
        for attempt in range(1, self.retry + 1):
            try:
                resp = self.session.get(url, params=params, timeout=120)
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                print(e)
                if attempt < self.retry:
                    time.sleep(self.pause * attempt)
        raise RuntimeError(f"failed after {self.retry} attempts")

    def collect(self, past_days: int = 92, forecast_days: int = 5, locations: dict | None = None) -> pd.DataFrame:
        locations = locations or LOCATIONS
        all_rows = []
        for loc_id, loc in locations.items():
            print(f" air quality ({past_days}d: {loc['name']})")
            params = {
                'latitude': loc['lat'], 'longitude': loc['lon'],
                'timezone': TIMEZONE, 'past_days': past_days, 'forecast_days': forecast_days,
                'hourly': ",".join(OM_AIR_QUALITY_HOURLY),
            }
            data = self._get(AQ_URL, params)
            hourly = data.get("hourly", {})

            if not hourly:
                continue
            df = pd.DataFrame(hourly)
            df['timestamp'] = pd.to_datetime(df['time'])
            df.drop(columns=['time'], inplace=True)
            df['location_id'] = loc_id
            all_rows.append(df)
            time.sleep(self.pause)
        return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()

    def collect_historical(self, start_date: str, end_date: str, locations: dict | None = None) -> pd.DataFrame:
        locations = locations or LOCATIONS
        # Open-Meteo allows large requests, so we don't chunk as aggressively unless needed
        
        all_keys = [f"{loc_id}|{start_date}|{end_date}" for loc_id in locations]
        ckpt = Checkpoint("air_quality_historical")
        out_path = RAW_AQ_DIR / 'historical.csv'
        
        pending = ckpt.pending(all_keys)
        done = len(all_keys) - len(pending)
        print(f" air quality historical: {done}/{len(all_keys)} done, {len(pending)} pending")
        
        for key in pending:
            loc_id, sd, ed = key.split("|")
            loc = locations[loc_id]
            print(f" air quality historical: {loc['name']} [{sd} -> {ed}]")
            
            params = {
                'latitude': loc['lat'], 'longitude': loc['lon'],
                'timezone': TIMEZONE, 'start_date': sd, 'end_date': ed,
                'hourly': ",".join(OM_AIR_QUALITY_HOURLY),
            }
            
            data = self._get(AQ_URL, params)
            hourly = data.get("hourly", {})
            if hourly:
                df = pd.DataFrame(hourly)
                df['timestamp'] = pd.to_datetime(df['time'])
                df.drop(columns=['time'], inplace=True)
                df['location_id'] = loc_id
                append_csv(df, out_path)
            
            ckpt.mark_done(key)
            time.sleep(self.pause)
            
        return pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()

    def save(self, df: pd.DataFrame, name: str = 'air_quality') -> None:
        if df.empty:
            return
        path = RAW_AQ_DIR / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"{name} saved")