#!/usr/bin/env python3
"""
Quick test script for volume bar compression.
Usage: Load tick data and compress to volume bars.
"""

import sys
import pandas as pd

try:
    from orderflow.compressor.compressor import compress_to_volume_bars
    
    # Load the tick data
    print("Loading tick data...", flush=True)
    tick_data = pd.read_csv(
        "data/tbt/2023_06_29.txt",
        sep=";",
        usecols=["Date", "Time", "Price", "Volume", "TradeType"]
    )
    
    print(f"Loaded {len(tick_data)} ticks", flush=True)
    print(f"Date range: {tick_data['Date'].min()} to {tick_data['Date'].max()}", flush=True)
    print(f"Price range: {tick_data['Price'].min()} to {tick_data['Price'].max()}", flush=True)
    print(f"Total volume: {tick_data['Volume'].sum()}", flush=True)
    
    # Compress to volume bars (e.g., 1000 contracts per bar)
    print("Compressing to volume bars...", flush=True)
    volume_bars = compress_to_volume_bars(tick_data, volume_amount=1000)
    
    print(f"\n{'='*60}", flush=True)
    print(f"Generated {len(volume_bars)} volume bars (1000 contract bars)", flush=True)
    print(f"{'='*60}", flush=True)
    print(volume_bars.head(10), flush=True)
    
    print(f"\nBar statistics:", flush=True)
    print(volume_bars[['Open', 'High', 'Low', 'Close', 'Volume']].describe(), flush=True)
    
    print(f"\nDetailed bar info:", flush=True)
    for i, row in volume_bars.iterrows():
        print(f"\nBar {i+1}:", flush=True)
        print(f"  Time: {row['OpenTime']} -> {row['CloseTime']}", flush=True)
        print(f"  OHLC: {row['Open']} / {row['High']} / {row['Low']} / {row['Close']}", flush=True)
        print(f"  Volume: {row['Volume']} (Bid: {row['BidVolume']}, Ask: {row['AskVolume']})", flush=True)
        print(f"  Trades: {row['NumberOfTrades']}", flush=True)
    
except Exception as e:
    print(f"ERROR: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)
