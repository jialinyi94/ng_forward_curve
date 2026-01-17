#!/usr/bin/env python3
"""
Natural Gas Forward Curve - Quick Usage Examples

This script demonstrates how to use the NaturalGasForwardCurve class
to fetch and visualize natural gas futures data.

Run this script to see examples of:
1. Fetching live forward curve
2. Fetching historical prices
3. Exporting data to files
4. Calendar spread analysis
"""

from ng_forward_curve import NaturalGasForwardCurve
from datetime import datetime, timedelta


def example_1_live_forward_curve():
    """Example 1: Fetch and display the current forward curve."""
    print("\n" + "=" * 60)
    print("Example 1: Live Forward Curve")
    print("=" * 60)
    
    ng = NaturalGasForwardCurve()
    
    # Fetch live forward curve for the next 12 months
    curve = ng.fetch_live_forward_curve(num_months=12)
    
    if not curve.empty:
        print("\nCurrent Natural Gas Forward Curve:")
        print(curve[['Contract', 'Symbol', 'Price', 'Volume']].to_string(index=False))
        
        # Plot the curve
        ng.plot_forward_curve(curve, 
                             title="Natural Gas Forward Curve (12 Months)",
                             save_path="example_forward_curve.png",
                             show=False)
        print("\nPlot saved to: example_forward_curve.png")
    
    return curve


def example_2_historical_prices():
    """Example 2: Fetch historical price data."""
    print("\n" + "=" * 60)
    print("Example 2: Historical Prices")
    print("=" * 60)
    
    ng = NaturalGasForwardCurve()
    
    # Fetch 1 year of historical data
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    prices = ng.fetch_historical_prices(start_date=start_date)
    
    if not prices.empty:
        print(f"\nHistorical prices from {start_date}:")
        print(f"Total trading days: {len(prices)}")
        print(f"\nPrice Statistics:")
        print(f"  Min:        ${prices['Close'].min():.3f}")
        print(f"  Max:        ${prices['Close'].max():.3f}")
        print(f"  Mean:       ${prices['Close'].mean():.3f}")
        print(f"  Current:    ${prices['Close'].iloc[-1]:.3f}")
        print(f"  Volatility: {prices['Close'].pct_change().std() * 100:.2f}% (daily)")
        
        # Plot price history
        ng.plot_price_history(prices,
                             title="Natural Gas - 1 Year Price History",
                             save_path="example_price_history.png",
                             show=False)
        print("\nPlot saved to: example_price_history.png")
    
    return prices


def example_3_export_data():
    """Example 3: Export data to files."""
    print("\n" + "=" * 60)
    print("Example 3: Export Data to CSV")
    print("=" * 60)
    
    ng = NaturalGasForwardCurve()
    
    # Fetch and export forward curve
    curve = ng.fetch_live_forward_curve(num_months=24, verbose=False)
    if not curve.empty:
        ng.export_to_csv(curve, "ng_forward_curve_export.csv")
    
    # Fetch and export historical prices
    start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
    prices = ng.fetch_historical_prices(start_date=start_date)
    if not prices.empty:
        ng.export_to_csv(prices.reset_index(), "ng_historical_export.csv")
    
    print("\nData exported successfully!")


def example_4_calendar_spreads():
    """Example 4: Calculate and analyze calendar spreads."""
    print("\n" + "=" * 60)
    print("Example 4: Calendar Spread Analysis")
    print("=" * 60)
    
    ng = NaturalGasForwardCurve()
    curve = ng.fetch_live_forward_curve(num_months=18, verbose=False)
    
    if not curve.empty and len(curve) >= 2:
        # Calculate spreads
        spreads = ng.calculate_calendar_spreads(curve)
        
        if not spreads.empty:
            print("\nCalendar Spreads (Month-over-Month):")
            print("-" * 70)
            print(spreads.to_string(index=False))
        
        # Winter-Summer spread analysis
        winter_contracts = curve[curve['Month'].isin([12, 1, 2])]
        summer_contracts = curve[curve['Month'].isin([6, 7, 8])]
        
        if not winter_contracts.empty and not summer_contracts.empty:
            winter_avg = winter_contracts['Price'].mean()
            summer_avg = summer_contracts['Price'].mean()
            ws_spread = winter_avg - summer_avg
            
            print(f"\n{'='*40}")
            print("Seasonal Spread Analysis:")
            print(f"{'='*40}")
            print(f"  Winter Avg (Dec-Feb): ${winter_avg:.3f}")
            print(f"  Summer Avg (Jun-Aug): ${summer_avg:.3f}")
            print(f"  Winter-Summer Spread: ${ws_spread:.3f}")
            print(f"  Premium:              {(ws_spread/summer_avg)*100:.1f}%")


def example_5_specific_contract():
    """Example 5: Fetch data for a specific contract."""
    print("\n" + "=" * 60)
    print("Example 5: Specific Contract Data")
    print("=" * 60)
    
    ng = NaturalGasForwardCurve()
    
    # Get symbol for a specific contract
    month, year = 12, 2026  # December 2026
    symbol = ng.get_contract_symbol(month, year)
    print(f"\nContract: December 2026")
    print(f"Symbol: {symbol}")
    
    # Fetch historical data for this contract
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    data = ng.fetch_contract_historical(symbol, start_date=start_date)
    
    if not data.empty:
        print(f"\nHistorical data (last 5 days):")
        print(data.tail().to_string())
    else:
        print("No historical data available for this contract yet.")


if __name__ == "__main__":
    print("=" * 60)
    print("Natural Gas Forward Curve - Usage Examples")
    print("=" * 60)
    
    # Run all examples
    example_1_live_forward_curve()
    example_2_historical_prices()
    example_3_export_data()
    example_4_calendar_spreads()
    example_5_specific_contract()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
