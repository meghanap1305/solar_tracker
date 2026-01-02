import streamlit as st
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Solar Pro Max", layout="wide", page_icon="⚡")

# --- 1. CORE LOGIC (INTEGRATED) ---
# We keep the backend logic here so we don't need 'main.py'

@st.cache_data(ttl=3600)
def fetch_nasa_data(lat, lon, start_year=1985):
    """Fetches clean annual solar data (GHI) from NASA POWER API."""
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
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        parameter_data = data["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"]
        
        annual_values = []
        for key, value in parameter_data.items():
            if key.endswith("13") and value > -999:
                annual_values.append(value)
        return np.array(annual_values)
    except Exception:
        return np.array([])

def calculate_single_scenario(ghi_history, system_size_kw, capex_per_kw, opex_per_kw, tariff, inflation, degradation, lifespan):
    """Calculates Financials for ONE scenario."""
    if len(ghi_history) == 0: return None

    # Solar Resource (P90 Conservative)
    p90_ghi = np.mean(ghi_history) - (1.282 * np.std(ghi_history))
    
    # Costs
    initial_investment = system_size_kw * capex_per_kw
    
    years = np.arange(1, lifespan + 1)
    cumulative_cash = -initial_investment
    cash_flows = []
    
    for year in years:
        # Physics
        degradation_factor = (1 - degradation) ** (year - 1)
        annual_gen = p90_ghi * system_size_kw * 0.75 * degradation_factor * 365
        
        # Economics (with Inflation)
        current_tariff = tariff * ((1 + inflation) ** (year - 1))
        annual_revenue = annual_gen * current_tariff
        annual_opex = system_size_kw * opex_per_kw
        
        net_profit = annual_revenue - annual_opex
        cumulative_cash += net_profit
        
        cash_flows.append(cumulative_cash)
        
    return {
        "investment": initial_investment,
        "final_profit": cumulative_cash,
        "cash_flows": cash_flows,
        "roi_multiple": (cumulative_cash + initial_investment) / initial_investment if initial_investment > 0 else 0
    }

# --- 2. FRONTEND UI ---

st.title("⚡ Solar Planner Tool: Multi-Scenario Analysis")

# --- Location Section ---
col_loc1, col_loc2 = st.columns([1, 2])
with col_loc1:
    st.subheader("📍 Location")
    # Simplified Location Input (Removed external dependency for stability)
    lat = st.number_input("Latitude", value=19.0760, format="%.4f")
    lon = st.number_input("Longitude", value=72.8777, format="%.4f")
with col_loc2:
    st.info("💡 **Pro Tip:** Compare 'Basic' vs 'Premium' systems to see if higher efficiency pays off in the long run.")

st.divider()

# --- Sidebar: Global & Scenario Settings ---
st.sidebar.header("🌍 Market Conditions")
ELECTRICITY_RATE = st.sidebar.number_input("Current Electricity Rate (Unit Cost)", value=8.0)
INFLATION_RATE = st.sidebar.slider("Energy Inflation Rate (%)", 0.0, 10.0, 5.0) / 100
PROJECT_LIFESPAN = st.sidebar.slider("Project Lifespan (Years)", 10, 30, 25)

st.sidebar.divider()
st.sidebar.header("⚙️ Compare 3 Systems")

scenarios = ["Basic", "Premium", "Max Power"]
configs = {}

# User's Original Loop (Enhanced with specific financial inputs)
for scenario in scenarios:
    with st.sidebar.expander(f"{scenario} System", expanded=(scenario=="Basic")):
        # Defaults tailored to scenario type
        if scenario == "Basic":
            def_size, def_cost = 5.0, 40000
        elif scenario == "Premium":
            def_size, def_cost = 10.0, 45000
        else:
            def_size, def_cost = 20.0, 38000 # Bulk discount?
            
        s_size = st.number_input(f"{scenario} Size (kW)", value=def_size, key=f"sz_{scenario}")
        s_cost = st.number_input(f"{scenario} Cost/kW", value=def_cost, key=f"cost_{scenario}")
        s_opex = st.number_input(f"{scenario} O&M/Year", value=500, key=f"opex_{scenario}")
        
        configs[scenario] = {
            "size": s_size, 
            "capex": s_cost, 
            "opex": s_opex,
            "color": {"Basic": "#1f77b4", "Premium": "#ff7f0e", "Max Power": "#2ca02c"}[scenario]
        }

# --- 3. EXECUTION ---

if st.button("🚀 Run Comparative Analysis", type="primary"):
    with st.spinner("Fetching NASA Weather Data & Calculating Financial Models..."):
        # 1. Fetch Data Once
        ghi_data = fetch_nasa_data(lat, lon)
        
        if len(ghi_data) > 0:
            results = {}
            
            # 2. Run Model for EACH Scenario
            for name, cfg in configs.items():
                res = calculate_single_scenario(
                    ghi_data, 
                    cfg['size'], 
                    cfg['capex'], 
                    cfg['opex'], 
                    ELECTRICITY_RATE, 
                    INFLATION_RATE, 
                    0.005, # Standard Degradation
                    PROJECT_LIFESPAN
                )
                results[name] = res

            # 3. Display Comparison Table
            st.subheader("📊 Financial Showdown (25 Years)")
            
            # Create a nice dataframe for metrics
            metrics = []
            for name, res in results.items():
                # Determine status
                is_profit = res['final_profit'] > 0
                status_icon = "✅ PROFIT" if is_profit else "❌ LOSS"
                
                metrics.append({
                    "Scenario": name,
                    "Initial Cost": f"{res['investment']:,.0f}",
                    "Net Profit": f"{res['final_profit']:,.0f}",
                    "Outcome": status_icon,
                    "ROI Multiple": f"{res['roi_multiple']:.2f}x"
                })
            
            st.dataframe(pd.DataFrame(metrics), use_container_width='stretch')

            # 4. Plot Comparative Graph
            st.subheader("📈 Cumulative Cash Flow Comparison")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            years = np.arange(1, PROJECT_LIFESPAN + 1)
            
            for name, res in results.items():
                ax.plot(years, res['cash_flows'], label=f"{name} ({configs[name]['size']}kW)", 
                        linewidth=3, color=configs[name]['color'])
            
            ax.axhline(0, color='black', linewidth=1, linestyle='--')
            ax.set_ylabel("Net Profit / Loss")
            ax.set_xlabel("Years")
            ax.set_title(f"Profitability Timeline (Inflation: {INFLATION_RATE*100}%)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
        else:
            st.error("Could not fetch solar data. Please check coordinates.")