import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# --- CONFIGURATION ---
LATITUDE = 19.0760  # Mumbai
LONGITUDE = 72.8777
START_YEAR = 1985
PROJECT_LIFESPAN = 25    # Years
SYSTEM_SIZE_KW = 5       # 5 kW Rooftop System

# --- FINANCIAL INPUTS ---
CAPEX_PER_KW = 40000     # Cost to build per kW (e.g., ₹40,000 or $1000)
OPEX_PER_KW_YEAR = 500   # Maintenance cost per kW per year
ELECTRICITY_RATE = 8     # Current value of 1 unit (e.g., ₹8)
INFLATION_RATE = 0.05    # 5% Annual increase in electricity prices

# --- TECHNICAL INPUTS ---
PERFORMANCE_RATIO = 0.75 # Efficiency losses (heat, dirt, wires)
DEGRADATION_RATE = 0.005 # 0.5% loss per year

def fetch_nasa_data(lat, lon, start_year):
    """
    Fetches clean annual solar data (GHI).
    """
    print(f"📡 Fetching Data for Financial Analysis...")
    end_year = datetime.now().year - 1
    url = "https://power.larc.nasa.gov/api/temporal/monthly/point"
    params = {
        "parameters": "ALLSKY_SFC_SW_DWN",
        "community": "RE",
        "longitude": lon,
        "latitude": lat,
        "format": "JSON",
        "start": start_year,
        "end": end_year
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        parameter_data = data["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"]
        
        # Extract Annual Averages (Month 13 in NASA API)
        annual_values = []
        for key, value in parameter_data.items():
            if key.endswith("13") and value > -999: # Month 13 is annual avg
                annual_values.append(value)

        return np.array(annual_values)

    except Exception as e:
        print(f"❌ Error: {e}")
        return np.array([])

def calculate_roi(ghi_history):
    if len(ghi_history) == 0: return None

    # 1. Determine Solar Resource (P50 vs P90)
    mean_ghi = np.mean(ghi_history)
    std_dev_ghi = np.std(ghi_history)
    
    p50_ghi = mean_ghi
    p90_ghi = mean_ghi - (1.282 * std_dev_ghi)
    
    # 2. Initial Costs (Year 0)
    initial_investment = SYSTEM_SIZE_KW * CAPEX_PER_KW
    
    print(f"\n💼 --- PROJECT P&L STATEMENT (Inflation: {INFLATION_RATE*100}%) ---")
    print(f"Initial Investment (CAPEX): {initial_investment:,.0f}")
    print(f"P50 Solar Irradiance: {p50_ghi:.4f} kWh/m²/day")
    print(f"P90 Solar Irradiance: {p90_ghi:.4f} kWh/m²/day")
    
    # 3. Year-by-Year Cash Flow
    years = np.arange(1, PROJECT_LIFESPAN + 1)
    
    cumulative_cash_p50 = -initial_investment
    cumulative_cash_p90 = -initial_investment
    
    results = []

    for year in years:
        # Technical Factors
        degradation_factor = (1 - DEGRADATION_RATE) ** (year - 1)
        
        # Financial Factors (Inflation)
        # We assume the utility rate goes UP every year
        current_tariff = ELECTRICITY_RATE * ((1 + INFLATION_RATE) ** (year - 1))
        
        annual_opex = SYSTEM_SIZE_KW * OPEX_PER_KW_YEAR
        
        # --- P50 Scenario ---
        p50_gen = p50_ghi * SYSTEM_SIZE_KW * PERFORMANCE_RATIO * degradation_factor * 365
        p50_revenue = p50_gen * current_tariff # Using inflated tariff
        p50_net_profit = p50_revenue - annual_opex
        cumulative_cash_p50 += p50_net_profit
        
        # --- P90 Scenario ---
        p90_gen = p90_ghi * SYSTEM_SIZE_KW * PERFORMANCE_RATIO * degradation_factor * 365
        p90_revenue = p90_gen * current_tariff # Using inflated tariff
        p90_net_profit = p90_revenue - annual_opex
        cumulative_cash_p90 += p90_net_profit
        
        results.append({
            "Year": year,
            "Tariff_Rate": current_tariff, # Track the changing rate
            "P50_Net_Cash_Flow": p50_net_profit,
            "P90_Net_Cash_Flow": p90_net_profit,
            "P50_Cumulative": cumulative_cash_p50,
            "P90_Cumulative": cumulative_cash_p90
        })

    df = pd.DataFrame(results)
    return df, initial_investment

def plot_roi(df, initial_investment):
    plt.figure(figsize=(12, 6))

    # Plot Cumulative Cash Flow (The "J-Curve")
    plt.plot(df["Year"], df["P50_Cumulative"], color='#2ca02c', linewidth=3, label='P50 (Expected) Cumulative Cash')
    plt.plot(df["Year"], df["P90_Cumulative"], color='#1f77b4', linewidth=3, linestyle='--', label='P90 (Conservative) Cumulative Cash')
    
    # Zero Line (Break Even Point)
    plt.axhline(y=0, color='black', linewidth=1, linestyle='-')
    plt.axhline(y=-initial_investment, color='red', linestyle=':', label='Initial Investment')

    # Find Payback Period (Where line crosses 0)
    try:
        payback_year_p50 = df[df["P50_Cumulative"] >= 0]["Year"].iloc[0]
        plt.scatter(payback_year_p50, 0, color='gold', s=150, zorder=5, edgecolors='black', label=f'Break Even: Year {payback_year_p50}')
    except IndexError:
        pass 

    plt.title(f"ROI with {INFLATION_RATE*100}% Utility Inflation", fontsize=14)
    plt.xlabel("Years after Installation", fontsize=12)
    plt.ylabel("Net Profit / Loss (Currency)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    # Add Summary Text
    final_profit_p50 = df["P50_Cumulative"].iloc[-1]
    final_profit_p90 = df["P90_Cumulative"].iloc[-1]
    
    # Calculate simple ROI multiple
    roi_multiple = (final_profit_p50 + initial_investment) / initial_investment
    
    plt.figtext(0.15, 0.70, 
                f"FINAL PROFIT (Year 25):\n"
                f"P50: {final_profit_p50:,.0f}\n"
                f"ROI Multiple: {roi_multiple:.1f}x",
                fontsize=10, bbox=dict(facecolor='white', alpha=0.9))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ghi_values = fetch_nasa_data(LATITUDE, LONGITUDE, START_YEAR)
    
    if len(ghi_values) > 0:
        df_roi, initial_cost = calculate_roi(ghi_values)
        
        # Show first few years to see the Tariff rising
        print("\n💰 Annual Cash Flow Forecast (First 7 Years):")
        print(df_roi[["Year", "Tariff_Rate", "P50_Net_Cash_Flow", "P50_Cumulative"]].head(7).to_string(index=False))
        
        plot_roi(df_roi, initial_cost)