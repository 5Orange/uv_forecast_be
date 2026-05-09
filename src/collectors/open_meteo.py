import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from src.config import (
    LOCATIONS, OM_FORECAST_DAILY, OM_FORECAST_HOURLY,
    OM_HISTORICAL_DAILY, OM_HISTORICAL_HOURLY, RAW_OM_DIR, TIMEZONE
)

from src.utils.checkpoint import Checkpoint, append_csv

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


class OpenMeteoCollector:
    def __init__(self, retry: int = 3, pause: float = 1.0):
        self.retry = retry
        self.pause = pause
        self.session = requests.Session()

    def get(self, url: str, params: dict) -> dict:
        for attempt in range(1, self.retry + 1):
            try:
                resp = self.session.get(url, params=params, timeout=120)
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                print(f"Attempt #{attempt} failed. Retrying.... {e}")
                if attempt <= self.retry:
                    time.sleep(self.pause * attempt + 60)
        raise RuntimeError(f"failed after {self.retry} attempts: {url}")

    @staticmethod
    def _json_to_hourly_df(data: dict, location_id: str) -> pd.DataFrame:
        hourly = data.get("hourly", {})
        if not hourly:
            return pd.DataFrame()
        df = pd.DataFrame(hourly)
        df['timestamp'] = pd.to_datetime(df['time'])
        df.drop(columns=['time'], inplace=True)
        df['location_id'] = location_id
        return df

    @staticmethod
    def _json_to_daily_df(data:dict, location_id: str) -> pd.DataFrame:
        daily = data.get("daily", {})
        if not daily:
            return pd.DataFrame()
        df = pd.DataFrame(daily)
        df['date'] = pd.to_datetime(df['time'])
        df.drop(columns=['time'], inplace=True)
        df['location_id'] = location_id
        return df

    def collect_historical(
            self,
            start_date:str,
            end_date:str,
            locations: dict | None = None,
            chunk_days: int =365
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        locations = locations or LOCATIONS
        chunks = self._date_chunks(start_date, end_date, chunk_days)

        all_keys = [
            f"{loc_id}|{cs}|{ce}"
            for loc_id in locations
            for cs, ce in chunks
        ]

        ckpt = Checkpoint("open_meteo_historical")
        hourly_path = RAW_OM_DIR / "historical_hourly.csv"
        daily_path = RAW_OM_DIR / "historical_daily.csv"

        pending = ckpt.pending(all_keys)
        total = len(all_keys)
        print(f" open-meteo historical: {total - len(pending)} / {total} chunks already done. "
              f"{len(pending)} pending")

        for key in pending:
            loc_id, chunk_start, chunk_end = key.split("|")
            loc = locations[loc_id]
            print(f"open-meteo historical: {loc['name']} [{chunk_start} -> {chunk_end}]")

            params = {
                'latitude': loc['lat'], 'longitude': loc['lon'],
                'timezone': TIMEZONE, 'start_date': chunk_start, 'end_date': chunk_end,
                'hourly': ",".join(OM_HISTORICAL_HOURLY),
                'daily': ",".join(OM_HISTORICAL_DAILY)
            }

            data = self.get(ARCHIVE_URL, params)
            append_csv(self._json_to_hourly_df(data, loc_id), hourly_path)
            append_csv(self._json_to_daily_df(data, loc_id), daily_path)
            ckpt.mark_done(key)
            time.sleep(self.pause)

        print(f" Open-meteo historical: all {total} chunks collected")

        hourly_df = pd.read_csv(hourly_path) if hourly_path.exists() else pd.DataFrame()
        daily_df = pd.read_csv(daily_path) if daily_path.exists() else pd.DataFrame()
        return hourly_df, daily_df

    def collect_recent_uv(
            self,
            past_days: int =92,
            forecast_days: int = 7,
            locations: dict | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        locations = locations or LOCATIONS
        all_hourly, all_daily = [], []
        for loc_id, loc in locations.items():
            print(f" open meteo forecast (past{past_days}): {loc['name']} [{loc_id}]")
            params = {
                'latitude': loc['lat'], 'longitude': loc['lon'],
                'timezone': TIMEZONE, 'past_days': past_days, 'forecast_days': forecast_days,
                'hourly': ",".join(OM_FORECAST_HOURLY), 'daily': ",".join(OM_FORECAST_DAILY),
                'models': "gfs_seamless"
            }
            data = self.get(FORECAST_URL, params)
            all_hourly.append(self._json_to_hourly_df(data, loc_id))
            all_daily.append(self._json_to_daily_df(data, loc_id))
            time.sleep(self.pause)
        hourly_df = pd.concat(all_hourly, ignore_index=True) if all_hourly else pd.DataFrame()
        daily_df = pd.concat(all_daily, ignore_index=True) if all_daily else pd.DataFrame()
        return hourly_df, daily_df

    def save(self, hourly_df: pd.DataFrame, daily_df: pd.DataFrame, prefix: str = "historical") -> None:
        if not hourly_df.empty:
            path = RAW_OM_DIR / f"{prefix}_hourly.csv"
            hourly_df.to_csv(path, index=False)
            print(f"saved {len(hourly_df)} hourly data")
        if not daily_df.empty:
            path = RAW_OM_DIR / f"{prefix}_daily.csv"
            daily_df.to_csv(path, index=False)
            print(f"saved {len(daily_df)} daily data")

    @staticmethod
    def _date_chunks(start: str, end: str, chunk_days: int) -> list[tuple[str, str]]:
        s = datetime.strptime(start, '%Y-%m-%d')
        e = datetime.strptime(end, '%Y-%m-%d')
        chunks = []
        while s <= e:
            chunk_end = min(s + timedelta(days=chunk_days - 1), e)
            chunks.append((s.strftime('%Y-%m-%d'), chunk_end.strftime('%Y-%m-%d')))
            s = chunk_end + timedelta(days=1)
        return chunks

