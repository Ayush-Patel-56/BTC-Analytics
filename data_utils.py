import requests
import pandas as pd
from datetime import datetime, timedelta

def get_binance_klines(symbol='BTCUSDT', interval='1h', limit=1000):
    """
    Fetch the MOST RECENT OHLCV data from Binance Public API.
    URL: https://data-api.binance.vision/api/v3/klines
    If limit > 1000, it will perform multiple requests moving backwards in time.
    """
    url = "https://data-api.binance.vision/api/v3/klines"
    
    all_data = []
    last_end_time = None
    
    remaining = limit
    while remaining > 0:
        fetch_limit = min(remaining, 1000)
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': fetch_limit
        }
        if last_end_time:
            params['endTime'] = last_end_time
        
        response = requests.get(url, params=params)
        if response.status_code != 200:
            raise RuntimeError(f"Binance API Error {response.status_code}: {response.text}")
        
        data = response.json()
        if not data:
            break
            
        # If we have multiple batches, the first bar of the current batch 
        # is the same as the last bar of the previous batch if we use endTime = last_start_time.
        # But Binance endTime is inclusive. So we set endTime = last_start_time - 1.
        
        if last_end_time:
            # Avoid overlap
            data = data[:-1]
            if not data: break
            
        all_data = data + all_data # Prepend since we are moving backwards
        remaining -= len(data)
        
        if len(data) == 0:
            break
            
        # Set end_time for next batch to the open_time of the first bar of current batch - 1ms
        last_end_time = data[0][0] - 1
        
        if len(data) < fetch_limit and last_end_time is None: # Only for the first call
            break
            
    df = pd.DataFrame(all_data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    
    # Convert to numeric
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])
    
    # Convert time to datetime
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('open_time', inplace=True)
    
    return df

if __name__ == "__main__":
    # Test fetching 1500 bars
    df = get_binance_klines(limit=1500)
    print(df.head())
    print(f"Fetched {len(df)} bars.")
    print(f"First bar: {df.index[0]}")
    print(f"Last bar: {df.index[-1]}")
