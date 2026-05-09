import time
from datetime import date, timedelta
import pandas as pd
import requests

from src.config import LOCATIONS, RAW_WB_DIR, WEATHERBIT_API_KEYS
from src.utils.checkpoint import  Checkpoint, append_csv
from src.utils.key_pool import AllKeyExhaustedError, KeyPool

BASE = "https://api.weatherbit.io/v2.0"

QUOTA_CODES = {403, 429}

class WeatherBitCollector:

    def __init__(
            self,
            api_keys: list[str] | None = None,
            retry_per_key: int = 3,
            pause: float = 2.0
    ):
        keys = api_keys or WEATHERBIT_API_KEYS
        self._has_keys = bool(keys)
        self.pool = KeyPool(keys, service="WeatherBit") if keys else None
        self.retry_per_key = retry_per_key
        self.pause = pause
        self.session = requests.Session()

        if not self._has_keys:
            print("No API key provided - will be skipped")

    @property
    def available(self) -> bool:
        return self._has_keys

    def _get(self, endpoint: str, params: dict) -> dict:
        url = f'{BASE}/{endpoint}'
        while self.pool and self.pool.available:
            key = self.pool.next_key()
            params['key'] = key
            for attempt in range(1, self.retry_per_key + 1):
                try:
                    resp = self.session.get(url, params=params, timeout=60)
                    if resp.status_code in QUOTA_CODES:
                        self.pool.mark_exhausted(key)
                        break
                    resp.raise_for_status()
                    return resp.json()
                except requests.RequestException as e:
                    print(e)
                    if attempt < self.retry_per_key:
                        time.sleep(self.pause * attempt + 90)
            else:
                raise RuntimeError(f"WeatherBit: All retries failed for {endpoint}")
        raise AllKeyExhaustedError(
            "WeatherBit: All keys exhausted"
        )

    @staticmethod
    def _parse_entry(entry:dict, loc_id: str) -> dict:
        return {
            'location_id': loc_id,
            'temp': entry.get("temp"), 'app_temp': entry.get("app_temp"),
            'rh': entry.get("rh"), 'dewpt': entry.get("dewpt"),
            'wind_spd': entry.get("wind_spd"), 'wind_gust_spd': entry.get("gust"),
            'wind_dir': entry.get("wind_dir"), 'clouds': entry.get("clouds"),
            'vis': entry.get("vis"), 'precip': entry.get("precip"),
            'uv': entry.get("uv"), 'solar_rad': entry.get("solar_rad"),
            'ghi': entry.get("ghi"), 'dni': entry.get("dni"), 'dhi': entry.get("dhi"),
            'elev_angle': entry.get("elev_angle"), 'h_angle': entry.get("h_angle"),
            'aqi': entry.get("aqi"), 'slp': entry.get("slp"), 'pres': entry.get("pres"),
            'ozone': entry.get("ozone"),
            'weather_code': entry.get("weather", {}).get("code"),
            'weather_desc': entry.get("weather", {}).get("description"),
            'sunrise': entry.get("sunrise"), 'sunset': entry.get("sunset"),
            'pod': entry.get("pod"), 
            'ob_time': entry.get("ob_time") or entry.get("timestamp_local") or entry.get("datetime"),
        }

    def collect_current(self, locations: dict | None = None) -> pd.DataFrame:
        if not self.available:
            return pd.DataFrame()

        locations = locations or LOCATIONS
        all_loc_ids = list(locations.keys())
        ckpt = Checkpoint("WeatherBit")
        out_path = RAW_WB_DIR / "current.csv"

        pending = ckpt.pending(all_loc_ids)
        total = len(all_loc_ids)
        print(f" WeatherBit: {total - len(pending)}/{total} locations collected,"
              f"{len(pending)} pending")

        for loc_id in pending:
            loc = locations[loc_id]
            print(f" WeatherBit: {loc['name']}")
            data = self._get("current", {"lat": loc["lat"], "lon": loc["lon"]})
            rows = [self._parse_entry(e, loc_id) for e in data.get("data", [])]
            if rows:
                append_csv(pd.DataFrame(rows), out_path)
            ckpt.mark_done(loc_id)
            time.sleep(self.pause)
        ckpt.clear()
        print(f" WeatherBit: all {total} locations collected -> {out_path}")
        return pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()

    def collect_historical(self, max_days_back: int = 730,
                           locations: dict | None = None) -> int:
        """Crawl backward from yesterday. Returns count of new date-location pairs."""
        if not self.available:
            return 0

        locations = locations or LOCATIONS
        ckpt = Checkpoint("weatherbit_historical")
        out_path = RAW_WB_DIR / 'historical.csv'

        all_keys = [
            f"{(date.today() - timedelta(days=d)).isoformat()}|{lid}"
            for d in range(1, max_days_back + 1)
            for lid in locations
        ]

        pending = ckpt.pending(all_keys)
        done_before = len(all_keys) - len(pending)
        print(f"  WB hist: {done_before}/{len(all_keys)} done, {len(pending)} pending")

        collected = 0
        try:
            for key in pending:
                dt_str, loc_id = key.split("|")
                loc = locations[loc_id]
                end_date = (date.fromisoformat(dt_str) + timedelta(days=1)).isoformat()
                print(f"  WB hist: {loc['name']} @ {dt_str}")

                try:
                    data = self._get("history/hourly", {
                        "lat": loc["lat"], "lon": loc["lon"],
                        "start_date": dt_str, "end_date": end_date,
                    })
                except RuntimeError as e:
                    print(f"  WB hist: skipping {key} — {e}")
                    ckpt.mark_done(key)
                    continue

                rows = [self._parse_entry(e, loc_id) for e in data.get("data", [])]
                if rows:
                    df = pd.DataFrame(rows)
                    df['date'] = dt_str
                    append_csv(df, out_path)
                ckpt.mark_done(key)
                collected += 1
                time.sleep(self.pause)
        except AllKeyExhaustedError:
            print(f"  WB hist: keys exhausted after {collected} new pairs. "
                  f"Total: {done_before + collected}/{len(all_keys)}. Resumable.")
        return collected