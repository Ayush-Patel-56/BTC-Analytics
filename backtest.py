import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from data_utils import get_binance_klines
from model import predict_next_hour, evaluate_prediction

def run_backtest(symbol='BTCUSDT', test_bars=720, train_window=500):
    print(f"Fetching data for backtest...")
    # Fetch enough data: test_bars + train_window
    total_bars = test_bars + train_window + 60 # Extra buffer for rolling metrics
    df = get_binance_klines(symbol=symbol, limit=total_bars)
    
    prices = df['close']
    results = []
    
    print(f"Starting backtest (720 predictions)... This may take a while.")
    
    # Windowed backtest ensuring no look-ahead bias
    start_idx = len(prices) - test_bars
    
    for i in tqdm(range(start_idx, len(prices))):
        # Data available up to i-1
        train_data = prices.iloc[i - train_window : i]
        actual_price = prices.iloc[i]
        prediction_time = prices.index[i]
        
        try:
            # Predict T+1 horizon
            pred = predict_next_hour(train_data, n_sims=10000)
            
            low = pred['predicted_low']
            high = pred['predicted_high']
            
            coverage, winkler, width = evaluate_prediction(actual_price, low, high)
            
            results.append({
                'date': str(prediction_time),
                'actual': float(actual_price),
                'low_95': float(low),
                'high_95': float(high),
                'coverage_95': int(coverage),
                'width_95': float(width),
                'winkler_95': float(winkler)
            })
        except Exception as e:
            print(f"Error at {prediction_time}: {e}")
            continue

    # Save to jsonl
    with open('backtest_results.jsonl', 'w') as f:
        for res in results:
            f.write(json.dumps(res) + '\n')
            
    # Calculate aggregate metrics
    if results:
        df_res = pd.DataFrame(results)
        coverage = df_res['coverage_95'].mean()
        avg_width = df_res['width_95'].mean()
        mean_winkler = df_res['winkler_95'].mean()
        
        print("\n--- Backtest Results ---")
        print(f"Coverage: {coverage:.4f}")
        print(f"Average Width: {avg_width:.2f}")
        print(f"Mean Winkler Score: {mean_winkler:.2f}")
        print(f"Results saved to backtest_results.jsonl")
        
        return coverage, avg_width, mean_winkler
    else:
        print("No results generated.")
        return None

if __name__ == "__main__":
    run_backtest()
