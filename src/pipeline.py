import gc
from src.collectors.air_quality import AirQualityCollector
from src.collectors.open_meteo import OpenMeteoCollector
from src.collectors.openuv import OpenUVCollector
from src.collectors.weatherbit import WeatherBitCollector
from src.config import HISTORICAL_END, HISTORICAL_START
from src.features.engr import FeatureEngineer
from src.processing.merger import DataMerger
from src.utils.key_pool import AllKeyExhaustedError

_COLLECTION_ERRORS = (AllKeyExhaustedError, RuntimeError)

def collect_historical():
    print(f"Step 1a: collecting historical data (OPEN_METEO) [{HISTORICAL_START} - {HISTORICAL_END}]")
    try:
        om = OpenMeteoCollector()
        hourly, daily = om.collect_historical(HISTORICAL_START, HISTORICAL_END)
        print(f'Historical done - hourly: {len(hourly)}, daily: {len(daily)}')
    except _COLLECTION_ERRORS as e:
        print(e)
        return ["Open-Meteo historical"]
    return []

def collect_recent_uv():
    print(f"Step 1b: collecting recent uv data past 92 days")
    try:
        om = OpenMeteoCollector()
        hourly, daily = om.collect_recent_uv(past_days=92, forecast_days=7)
        om.save(hourly, daily, prefix='recent_uv')
        print(f"recent uv done - hourly: {len(hourly)}, daily: {len(daily)}")
    except _COLLECTION_ERRORS as e:
        print(e)
        return ["Open-Meteo recent UV"]
    return []

def collect_air_quality():
    print(f"Step 1c: collecting air quality data (ozone, aerosol, PM)")
    try:
        aq = AirQualityCollector()
        hist_df = aq.collect_historical(HISTORICAL_START, HISTORICAL_END)
        print(f"Air quality historical done - {len(hist_df)}")
        df = aq.collect(past_days=92, forecast_days=7)
        aq.save(df)
        print(f"Air quality recent done - {len(df)}")
    except RuntimeError as e:
        print(e)
        return ["Air Quality"]
    return []

def collect_current(wb: WeatherBitCollector, ouv: OpenUVCollector):
    print(f"Step 1d: collecting current data (WeatherBit + openUV) ")
    failed = []

    if wb.available:
        try:
            wb.collect_current()
        except AllKeyExhaustedError as e:
            print(f" WeatherBit keys exhausted: {e}")
            failed.append("WeatherBit")
    if ouv.available:
        try:
            ouv.collect_realtime()
        except AllKeyExhaustedError as e:
            print(f" OpenUV keys exhausted: {e}")
            failed.append("OpenUV")
    return failed

def crawl_historical_openuv(ouv: OpenUVCollector):
    """Priority 2: backward crawl with remaining OpenUV quota."""
    print("Step 1e: OpenUV backward historical crawl")
    if not ouv.available:
        return []
    n = ouv.collect_historical(start_date=HISTORICAL_START, end_date=HISTORICAL_END)
    print(f"  OpenUV hist: {n} new date-location pairs this session")
    return []

def crawl_historical_wb(wb: WeatherBitCollector):
    """Priority 2: backward crawl with remaining WeatherBit quota."""
    print("Step 1f: WeatherBit backward historical crawl")
    if not wb.available:
        return []
    n = wb.collect_historical(start_date=HISTORICAL_START, end_date=HISTORICAL_END)
    print(f"  WB hist: {n} new date-location pairs this session")
    return []

def process_and_engineer():
    print("STEP 2 & 3: Merging & Feature Engineering (Per Location)")
    locations = ['hcm', 'vungtau', 'cuchi', 'nhabe', 'cangio', 'longhai', 'thuduc']
    
    DataMerger.clear_output_files()
    FeatureEngineer.clear_output_files()

    for loc in locations:
        print(f"\n--- Processing Location: {loc} ---")
        
        merger = DataMerger()
        hourly_df = merger.run(locations_to_keep=[loc])
        
        fe = FeatureEngineer(df=hourly_df)
        fe.run_all()
        
        del merger
        del hourly_df
        del fe
        gc.collect()

def main():
    print("PIPELINE STARTED")
    failed: list[str] = []

    # --- Open-Meteo (unlimited, no key needed) ---
    failed.extend(collect_historical())
    failed.extend(collect_recent_uv())
    failed.extend(collect_air_quality())

    # Create shared instances so KeyPool state (exhausted keys) is maintained across steps
    wb = WeatherBitCollector()
    ouv = OpenUVCollector()

    # --- Priority 1: Today's realtime snapshot (costs ~10 req each) ---
    failed.extend(collect_current(wb, ouv))

    # --- Priority 2: Backward crawl (uses remaining API quota) ---
    failed.extend(crawl_historical_openuv(ouv))
    failed.extend(crawl_historical_wb(wb))

    # --- Processing ---
    process_and_engineer()

    if failed:
        print(f"\nPipeline completed with quota limits hit: {', '.join(failed)}")
        print("Partial data saved with checkpoints. Re-run to continue collection.")
    else:
        print("\nPipeline completed!!!")

if __name__ == "__main__":
    main()
