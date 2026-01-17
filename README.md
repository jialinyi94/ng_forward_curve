# Natural Gas Futures Forward Curve Data Fetcher

A Python library for fetching historical and live forward curves for Natural Gas (Henry Hub) futures from free public sources.

## Overview

This tool provides functionality to:

- **Fetch live forward curve data**: Get current prices for multiple contract months (up to 10+ years forward)
- **Fetch historical price data**: Download historical OHLCV data for the continuous contract
- **Generate historical forward curves**: Reconstruct forward curves at historical dates
- **Visualize data**: Create professional charts for forward curves and price history
- **Calculate spreads**: Compute calendar spreads between consecutive months
- **Export data**: Save data to CSV or Excel formats

## Data Source

The primary data source is **Yahoo Finance** via the `yfinance` library:

| Data Type | Symbol Format | Example |
|-----------|---------------|---------|
| Continuous Contract | `NG=F` | Front month rolling contract |
| Individual Contracts | `NG{month}{year}.NYM` | `NGH26.NYM` (March 2026) |

### Month Codes

| Month | Code | Month | Code |
|-------|------|-------|------|
| January | F | July | N |
| February | G | August | Q |
| March | H | September | U |
| April | J | October | V |
| May | K | November | X |
| June | M | December | Z |

## Installation

```bash
# Install required dependencies
pip install yfinance pandas numpy matplotlib python-dateutil
```

## Quick Start

```python
from ng_forward_curve import NaturalGasForwardCurve

# Initialize the fetcher
ng = NaturalGasForwardCurve()

# Fetch live forward curve (next 12 months)
curve = ng.fetch_live_forward_curve(num_months=12)
print(curve[['Contract', 'Price', 'Volume']])

# Plot the forward curve
ng.plot_forward_curve(curve, title="NG Forward Curve")

# Fetch historical prices (last 1 year)
prices = ng.fetch_historical_prices(start_date='2025-01-01')
print(prices.tail())

# Export to CSV
ng.export_to_csv(curve, 'forward_curve.csv')
```

## API Reference

### Class: `NaturalGasForwardCurve`

#### Methods

##### `fetch_live_forward_curve(num_months=24, verbose=True)`

Fetch the current forward curve for Natural Gas futures.

**Parameters:**
- `num_months` (int): Number of future months to fetch (default: 24)
- `verbose` (bool): Print progress messages (default: True)

**Returns:** DataFrame with columns:
- `Contract`: Contract name (e.g., "Mar 2026")
- `Symbol`: Yahoo Finance symbol (e.g., "NGH26.NYM")
- `Month`, `Year`: Contract month and year
- `Price`: Settlement price ($/MMBtu)
- `Open`, `High`, `Low`: Daily OHLC prices
- `Volume`: Trading volume
- `Last_Update`: Timestamp of last update

##### `fetch_historical_prices(start_date, end_date=None, interval='1d')`

Fetch historical price data for the continuous contract.

**Parameters:**
- `start_date` (str): Start date in 'YYYY-MM-DD' format
- `end_date` (str): End date (defaults to today)
- `interval` (str): Data interval ('1d', '1wk', '1mo')

**Returns:** DataFrame with OHLCV data

##### `fetch_contract_historical(symbol, start_date, end_date=None)`

Fetch historical data for a specific futures contract.

**Parameters:**
- `symbol` (str): Contract symbol (e.g., 'NGH26.NYM')
- `start_date` (str): Start date
- `end_date` (str): End date

**Returns:** DataFrame with OHLCV data

##### `fetch_historical_forward_curves(start_date, end_date=None, num_months=12, sample_frequency='W')`

Generate historical forward curves using the continuous contract as a base.

**Parameters:**
- `start_date` (str): Start date
- `end_date` (str): End date
- `num_months` (int): Forward months per curve
- `sample_frequency` (str): Sampling frequency ('D', 'W', 'M')

**Returns:** Dictionary mapping dates to forward curve DataFrames

##### `plot_forward_curve(df, title, save_path=None, show=True)`

Plot the forward curve.

##### `plot_price_history(df, title, save_path=None, show=True)`

Plot historical price chart with volume.

##### `calculate_calendar_spreads(df)`

Calculate price differences between consecutive contract months.

**Returns:** DataFrame with spread calculations

##### `get_contract_symbol(month, year)`

Get Yahoo Finance symbol for a specific contract.

```python
>>> ng.get_contract_symbol(3, 2026)
'NGH26.NYM'
```

##### `export_to_csv(df, filepath)` / `export_to_excel(df, filepath)`

Export data to file.

## Examples

### Example 1: Basic Forward Curve

```python
from ng_forward_curve import NaturalGasForwardCurve

ng = NaturalGasForwardCurve()
curve = ng.fetch_live_forward_curve(num_months=12)

# Display the curve
print("Natural Gas Forward Curve")
print(curve[['Contract', 'Price', 'Volume']].to_string(index=False))

# Save plot
ng.plot_forward_curve(curve, save_path='forward_curve.png')
```

### Example 2: Calendar Spread Analysis

```python
ng = NaturalGasForwardCurve()
curve = ng.fetch_live_forward_curve(num_months=18)

# Calculate spreads
spreads = ng.calculate_calendar_spreads(curve)
print(spreads)

# Find winter-summer spread
winter = curve[curve['Month'].isin([12, 1, 2])]['Price'].mean()
summer = curve[curve['Month'].isin([6, 7, 8])]['Price'].mean()
print(f"Winter-Summer Spread: ${winter - summer:.3f}")
```

### Example 3: Historical Analysis

```python
ng = NaturalGasForwardCurve()

# Get 2 years of historical data
prices = ng.fetch_historical_prices(start_date='2024-01-01')

# Calculate statistics
print(f"Min: ${prices['Close'].min():.3f}")
print(f"Max: ${prices['Close'].max():.3f}")
print(f"Mean: ${prices['Close'].mean():.3f}")
print(f"Volatility: {prices['Close'].pct_change().std() * 100:.2f}%")

# Plot with volume
ng.plot_price_history(prices, save_path='price_history.png')
```

### Example 4: Specific Contract Data

```python
ng = NaturalGasForwardCurve()

# Get symbol for December 2026 contract
symbol = ng.get_contract_symbol(12, 2026)  # Returns 'NGZ26.NYM'

# Fetch historical data for this contract
data = ng.fetch_contract_historical(symbol, start_date='2025-01-01')
print(data.tail())
```

## Output Files

When running the main script, the following files are generated:

| File | Description |
|------|-------------|
| `ng_live_forward_curve.csv` | Current forward curve data |
| `ng_historical_prices.csv` | Historical price data |
| `ng_forward_curve.png` | Forward curve visualization |
| `ng_price_history.png` | Historical price chart |
| `ng_historical_curves.png` | Historical curves comparison |

## Data Notes

1. **Data Delay**: Yahoo Finance data is delayed by approximately 15-20 minutes
2. **Contract Availability**: Some near-term contracts may be unavailable if they have expired
3. **Historical Curves**: Generated using seasonal adjustment factors based on typical natural gas demand patterns
4. **Volume Data**: May not be available for all contracts, especially far-dated ones

## Seasonal Patterns

Natural gas prices exhibit strong seasonality due to heating and cooling demand:

| Season | Months | Typical Pattern |
|--------|--------|-----------------|
| Winter Peak | Dec-Feb | Highest prices (heating demand) |
| Spring Shoulder | Mar-May | Declining prices |
| Summer | Jun-Aug | Secondary peak (cooling demand) |
| Fall Shoulder | Sep-Nov | Rising prices |

## License

This project is provided for educational and research purposes. Please comply with Yahoo Finance's terms of service when using this tool.

## Disclaimer

This tool is for informational purposes only and should not be used as the sole basis for trading decisions. Always verify data with official sources before making investment decisions.
