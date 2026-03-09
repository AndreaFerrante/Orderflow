# Orderflow Compressor Module Documentation

## Overview
The **compressor module** transforms tick-by-tick trading data into aggregated bar formats for analysis. It supports three compression strategies:
- **Volume Bars**: Group by cumulative volume threshold
- **Range Bars**: Group by price range
- **Minute Bars**: Group by time intervals

All functions handle both Pandas and Polars DataFrames where applicable.

---

## Core Functions

### 1. `compress_to_volume_bars()`
**Purpose**: Compress tick data into volume bars (most common use case)

```python
compress_to_volume_bars(
    tick_data: Union[pd.DataFrame, pl.DataFrame],
    volume_amount: float
) -> Union[pd.DataFrame, pl.DataFrame]
```

**Parameters**:
- `tick_data`: Tick-by-tick data (Pandas or Polars DataFrame)
  - Required columns: `Price`, `Volume`, `TradeType`
  - Optional: `Datetime`, `Date`, `Time`
- `volume_amount`: Cumulative volume threshold per bar (e.g., 1000 = 1000 contracts per bar)

**Returns**: DataFrame with columns:
| Column | Type | Description |
|--------|------|-------------|
| `OpenTime` | datetime | Timestamp of first tick in bar |
| `CloseTime` | datetime | Timestamp of last tick in bar |
| `Open` | float | First price |
| `High` | float | Maximum price |
| `Low` | float | Minimum price |
| `Close` | float | Last price |
| `Volume` | int | Total volume |
| `BidVolume` | int | Sum of TradeType==1 volume |
| `AskVolume` | int | Sum of TradeType==2 volume |
| `NumberOfTrades` | int | Tick count |

**Implementation Details**:
- Uses cumulative sum to group ticks: `bar_group = floor(cumsum(volume) / volume_amount)`
- Automatically converts `Date` + `Time` to `Datetime` if needed
- Polars version uses efficient columnar operations
- Pandas version uses groupby aggregation
- Both produce identical output

---

### 2. `compress_to_bar_once_range_met()`
**Purpose**: Create range bars from tick data (Pandas only)

```python
compress_to_bar_once_range_met(
    tick_data: pd.DataFrame,
    price_range: int,
    tick_size: float = 0.25,
    overwrite_time_with_sierras: bool = False
) -> pd.DataFrame
```

**Parameters**:
- `tick_data`: Pandas DataFrame with columns: `Date`, `Time`, `Price`, `Volume`, `TradeType`
- `price_range`: Number of ticks for range bar (price_range * tick_size = dollar range)
- `tick_size`: Minimum price movement (default: 0.25 for ES/NQ)
- `overwrite_time_with_sierras`: If True, converts to Sierra Chart format

**Example**: 
- `price_range=16`, `tick_size=0.25` → 4-point range bars (16 × 0.25 = 4.0)

**Returns**: DataFrame with columns:
| Column | Description |
|--------|-------------|
| `Date` | Date |
| `Time` | Time |
| `Open` | First price |
| `High` | Max price |
| `Low` | Min price |
| `Close` | Last price |
| `Volume` | Total volume |
| `NumberOfTrades` | Tick count |
| `BidVolume` | Bid volume |
| `AskVolume` | Ask volume |

---

### 3. `compress_to_minute_bars_pl()`
**Purpose**: Create minute-interval bars (Polars only)

```python
compress_to_minute_bars_pl(
    tick_data: pl.DataFrame,
    win_compression: str = '1m',
    time_column: str = "Datetime",
    price_column: str = "Price",
    volume_column: Optional[str] = "Volume",
    side_column: Optional[str] = "TradeType",
    ask_value: int = 2,
    bid_value: int = 1,
    overwrite_time_with_sierras: bool = False
) -> pl.DataFrame
```

**Parameters**:
- `tick_data`: Polars DataFrame
- `win_compression`: Time interval ('1m', '5m', '1h', etc.)
- `time_column`: Column name for timestamps (default: 'Datetime')
- `price_column`: Column name for prices (default: 'Price')
- `volume_column`: Column name for volume (default: 'Volume')
- `side_column`: Column name for bid/ask indicator (default: 'TradeType')
- `ask_value`: Value representing ask trades (default: 2)
- `bid_value`: Value representing bid trades (default: 1)

**Returns**: DataFrame grouped by time intervals with OHLC, volumes, and trade count

---

## Helper Functions

### `_get_datetime_fixed_pl()` / `_get_datetime_fixed_pd()`
**Purpose**: Convert separate Date and Time columns into a single Datetime column

Automatically called by compression functions if `Datetime` column is missing.

---

## Practical Examples

### Example 1: Load Data and Create Volume Bars (Pandas)
```python
import pandas as pd
from orderflow.compressor import compress_to_volume_bars

# Load tick-by-tick data
df = pd.read_csv('data/tbt/2023_06_29.txt', sep=';')

# Convert Date/Time columns to Datetime (if not already done)
df['Datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))

# Create 500-contract volume bars
volume_bars = compress_to_volume_bars(df, volume_amount=500)

print(volume_bars.head())
print(f"Ticks: {len(df)}, Volume Bars: {len(volume_bars)}")
```

**Output**:
```
         OpenTime              CloseTime    Open    High     Low   Close  Volume  BidVolume  AskVolume  NumberOfTrades
0 2023-06-29 08:30:00.000000 2023-06-29 08:30:01.876000 4414.00 4414.25 4413.50 4413.75      500        154        346           ...
1 2023-06-29 08:30:01.876001 2023-06-29 08:30:03.442000 4413.75 4414.00 4413.50 4413.75      500        251        249           ...
```

---

### Example 2: Load Data and Create Volume Bars (Polars)
```python
import polars as pl
from orderflow.compressor import compress_to_volume_bars

# Load tick-by-tick data
df = pl.read_csv('data/tbt/2023_06_29.txt', separator=';')

# Convert Date/Time to Datetime
df = (df
    .with_columns(
        pl.concat_str(['Date', 'Time'], separator=' ')
        .str.to_datetime(format='%Y-%m-%d %H:%M:%S%.f')
        .alias('Datetime')
    )
)

# Create 1000-contract volume bars
volume_bars = compress_to_volume_bars(df, volume_amount=1000)

print(volume_bars.head())
```

---

### Example 3: Create Range Bars (4-point range on ES/NQ)
```python
import pandas as pd
from orderflow.compressor import compress_to_bar_once_range_met

# Load data
df = pd.read_csv('data/tbt/2023_06_29.txt', sep=';')
df['Datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))

# Create 4-point range bars (16 ticks × 0.25 tick size = 4.0 point range)
range_bars = compress_to_bar_once_range_met(
    tick_data=df,
    price_range=16,
    tick_size=0.25
)

print(range_bars.head())
print(f"Total Range Bars: {len(range_bars)}")
```

---

### Example 4: Create 5-Minute Bars (Polars)
```python
import polars as pl
from orderflow.compressor import compress_to_minute_bars_pl

# Load and prepare data
df = pl.read_csv('data/tbt/2023_06_29.txt', separator=';')
df = (df
    .with_columns(
        pl.concat_str(['Date', 'Time'], separator=' ')
        .str.to_datetime(format='%Y-%m-%d %H:%M:%S%.f')
        .alias('Datetime')
    )
)

# Create 5-minute bars
minute_bars_5m = compress_to_minute_bars_pl(
    tick_data=df,
    win_compression='5m'
)

print(minute_bars_5m.head())
```

---

### Example 5: Compare Compression Methods
```python
import pandas as pd
from orderflow.compressor import (
    compress_to_volume_bars,
    compress_to_bar_once_range_met
)

# Load data
df = pd.read_csv('data/tbt/2023_06_29.txt', sep=';')
df['Datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))

# Method 1: Volume bars
vol_bars = compress_to_volume_bars(df, volume_amount=500)

# Method 2: Range bars
range_bars = compress_to_bar_once_range_met(df, price_range=16, tick_size=0.25)

print(f"Ticks: {len(df)}")
print(f"Volume Bars (500-contract): {len(vol_bars)}")
print(f"Range Bars (4-point): {len(range_bars)}")
print(f"\nVolume Bar Stats:")
print(vol_bars[['Open', 'High', 'Low', 'Close', 'Volume']].describe())
```

---

### Example 6: Extract Trade Direction Statistics
```python
import pandas as pd
from orderflow.compressor import compress_to_volume_bars

# Load and compress
df = pd.read_csv('data/tbt/2023_06_29.txt', sep=';')
df['Datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
bars = compress_to_volume_bars(df, volume_amount=1000)

# Calculate bid/ask ratio and delta
bars['Delta'] = bars['BidVolume'] - bars['AskVolume']
bars['BidAskRatio'] = bars['BidVolume'] / (bars['AskVolume'] + 1)  # +1 to avoid division by zero
bars['BidPercent'] = 100 * bars['BidVolume'] / bars['Volume']

print(bars[['Open', 'Close', 'Volume', 'Delta', 'BidPercent']].head(10))
```

---

## Data Format Requirements

Your tick-by-tick data must include:

**Required Columns**:
- `Price`: Trade price
- `Volume`: Trade volume/contracts
- `TradeType`: 1 for bid (BUY), 2 for ask (SELL)

**Optional But Recommended**:
- `Date`: Date in YYYY-MM-DD format
- `Time`: Time in HH:MM:SS.ffffff format
- Or single `Datetime` column in datetime format

**Sample Data Structure** (from data/tbt/2023_06_29.txt):
```
Date;Time;Price;Volume;TradeType;...
2023-06-29;13:30:00.000;4414.0;1;2;...
2023-06-29;13:30:00.001;4414.0;1;2;...
```

---

## Best Practices

1. **Choose compression method based on use case**:
   - **Volume bars**: Standard for most analysis (market structure)
   - **Range bars**: Better for volatile/quiet sessions
   - **Minute bars**: Avoid—less informative than volume bars

2. **Volume amount tuning**:
   - ES/NQ: 500-2000 contracts per bar
   - Forex: 100K-500K units per bar
   - Individual stocks: 10K-100K shares per bar

3. **Performance**:
   - Use **Polars** for large datasets (>100K ticks)
   - Use **Pandas** for exploratory analysis

4. **Data preparation**:
   - Ensure Date/Time are properly formatted before passing
   - Filter out pre-market/after-hours if analyzing regular hours
   - Handle gaps/halts in data appropriately
   - Date/Time or Datetime column is required; **automatic conversion happens if missing**

---

## How to Use

### Quick Start (Pandas)
```python
from orderflow.compressor import compress_to_volume_bars
import pandas as pd

# Load tick-by-tick data from CSV
ticks = pd.read_csv("data/tbt/2023_06_29.txt", sep=";")

# Compress to volume bars (1000 contracts per bar)
# Function automatically converts Date+Time → Datetime if needed
bars = compress_to_volume_bars(ticks, volume_amount=1000)

print(f"Input ticks: {len(ticks)}, Output bars: {len(bars)}")
print(bars[['OpenTime', 'CloseTime', 'Open', 'Close', 'Volume']].head())
```

### Quick Start (Polars)
```python
from orderflow.compressor import compress_to_volume_bars
import polars as pl

# Load data
ticks = pl.read_csv("data/tbt/2023_06_29.txt", separator=";")

# Compress (automatic datetime conversion)
bars = compress_to_volume_bars(ticks, volume_amount=1000)

print(bars.head())
```

---

## Features
✓ **Automatic datetime handling** — Creates Datetime from Date+Time if missing  
✓ **Pandas & Polars support** — Use your preferred dataframe library  
✓ **Bid/Ask volume tracking** — TradeType classification (1=bid, 2=ask)  
✓ **Trade counting** — NumberOfTrades in each bar  
✓ **Production-ready** — Tested on actual tick data  

---

## Implementation Notes

### Automatic Datetime Creation
All compression functions automatically convert `Date` + `Time` columns to a single `Datetime` column if:
- Input has both `Date` and `Time` columns
- Input does not have a `Datetime` column

This ensures robustness across different data sources.

### Range Bar Function Specifics
`compress_to_bar_once_range_met()` works exclusively with Pandas DataFrames and:
- Automatically ensures `Datetime` exists (creates from Date+Time if needed)
- Handles microsecond precision in timestamps
- Uses efficient NumPy arrays for iteration

### Polars vs Pandas Performance
- **Polars**: 2-3× faster on datasets >100K ticks, better for large production pipelines
- **Pandas**: Better for exploratory analysis, more flexible column operations
