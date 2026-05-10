import time
from datetime import date, timedelta

import pandas as pd
import requests

from src.config import LOCATIONS, OPENUV_API_KEYS, RAW_UV_DIR
from src.utils.checkpoint import Checkpoint, append_csv
from src.utils.key_pool import AllKeyExhaustedError, KeyPool

BASE = "https://api.openuv.io/api/v1"
QUOTA_CODES = {403, 429}

class OpenUVCollector:

    def __init__(self,
                 api_keys: list[str] | None = None,
                 retry_per_key: int = 3,
                 pause: float = 2.0
    ):
        keys = api_keys or OPENUV_API_KEYS
        self._has_keys = bool(keys)
        self.pool = KeyPool(keys, service="OpenUV") if keys else None
        self.retry_per_key = retry_per_key
        self.pause = pause
        self.session = requests.Session()
        if not self._has_keys:
            print("API key not provided - will be skipped")

    @property
    def available(self) -> bool:
        return self._has_keys

    def _get(self, endpoint: str, params: dict) -> dict:
        url = f'{BASE}/{endpoint}'
        while self.pool and self.pool.available:
            key = self.pool.next_key()
            headers = {"x-access-token": key}

            for attempt in range(1, self.retry_per_key + 1):
                try:
                    resp = self.session.get(
                        url, params=params, headers=headers, timeout= 30
                    )
                    if resp.status_code in QUOTA_CODES:
                        self.pool.mark_exhausted(key)
                        break
                    resp.raise_for_status()
                    return resp.json()
                except requests.RequestException as e:
                    print(f"Request error: {e}")
                    if attempt < self.retry_per_key:
                        time.sleep(self.pause * attempt + 90)
            else:
                return {}
        raise AllKeyExhaustedError(
            "OpenUV: all api keys exhausted - hard stop"
        )

    @staticmethod
    def _flatten_uv_response(data: dict, location_id: str) -> dict:
        result = data.get("result", {})
        if not result:
            return {}

        row = {
            'location_id': location_id,
            'uv': result.get('uv'), 'uv_time': result.get('uv_time'),
            'uv_max': result.get('uv_max'), 'uv_max_time': result.get('uv_max_time'),
            'ozone': result.get('ozone'), 'ozone_time': result.get('ozone_time')
        }
        safe = result.get('safe_exposure_time', {})
        for st in range(1, 7):
            row[f'safe_exposure_st{st}'] = safe.get(f'st{st}')
        sun_info = result.get('sun_info', {})
        sun_pos = sun_info.get('sun_position', {})
        row['sun_altitude'] = sun_pos.get('altitude')
        row['sun_azimuth'] = sun_pos.get('azimuth')
        for event in ['sunrise', 'sunset', 'solarNoon', 'dawn', 'dusk', 'goldenHour', 'goldenHourEnd']:
            row[f"sun_{event}"] = sun_info.get("sun_times", {}).get(event)

        return row

    def collect_realtime(self, locations: dict| None = None) -> pd.DataFrame:
        if not self.available:
            return pd.DataFrame()

        locations = locations or LOCATIONS
        all_loc_ids = list(locations.keys())
        ckpt = Checkpoint("openuv_realtime")
        out_path = RAW_UV_DIR / 'realtime.csv'

        pending = ckpt.pending(all_loc_ids)
        total = len(all_loc_ids)
        print(f" OpenUV : {total - len(pending)}/{total} locations already done,"
              f"{len(pending)} pending")

        for loc_id in pending:
            loc = locations[loc_id]
            print(f" OpenUV realtime: {loc['name']}")
            data = self._get('uv', {'lat': loc['lat'], 'lng': loc['lon']})
            row = self._flatten_uv_response(data, loc_id)

            if row:
                append_csv(pd.DataFrame([row]),out_path)
            ckpt.mark_done(loc_id)
            time.sleep(self.pause)

        ckpt.clear()
        print(f"    OpenUV: all {total} locations collected -> {out_path}")
        return pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()

    def collect_historical(self, start_date: str, end_date: str,
                           locations: dict | None = None) -> int:
        """Crawl within a specific date range. Returns count of new pairs collected."""
        if not self.available:
            return 0

        locations = locations or LOCATIONS
        ckpt = Checkpoint("openuv_historical")
        out_path = RAW_UV_DIR / 'historical.csv'

        s = date.fromisoformat(start_date)
        e = date.fromisoformat(end_date)
        delta = (e - s).days

        all_keys = [
            f"{(s + timedelta(days=d)).isoformat()}|{lid}"
            for d in range(delta + 1)
            for lid in locations
        ]

        pending = ckpt.pending(all_keys)
        done_before = len(all_keys) - len(pending)
        print(f"  OpenUV hist: {done_before}/{len(all_keys)} done, {len(pending)} pending")

        collected = 0
        try:
            for key in pending:
                dt_str, loc_id = key.split("|")
                loc = locations[loc_id]
                print(f"  OpenUV hist: {loc['name']} @ {dt_str}")

                try:
                    data = self._get('uv', {
                        'lat': loc['lat'], 'lng': loc['lon'],
                        'dt': f"{dt_str}T05:00:00Z"
                    })
                except RuntimeError as e:
                    print(f"  OpenUV hist: skipping {key} — {e}")
                    ckpt.mark_done(key)
                    continue

                row = self._flatten_uv_response(data, loc_id)
                if row:
                    row['date'] = dt_str
                    if pd.isna(row.get('uv_time')) or not row.get('uv_time'):
                        row['uv_time'] = f"{dt_str}T05:00:00Z"

                    append_csv(pd.DataFrame([row]), out_path)
                ckpt.mark_done(key)
                collected += 1
                time.sleep(self.pause)
        except AllKeyExhaustedError:
            print(f"  OpenUV hist: keys exhausted after {collected} new pairs. "
                  f"Total: {done_before + collected}/{len(all_keys)}. Resumable.")
        return collected
