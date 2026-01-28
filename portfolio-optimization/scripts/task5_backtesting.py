import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. SETUP & DATA LOADING ---
# Path to the cleaned data from Task 1
data_path = r"C:\Users\hp\Desktop\Kifya\Time-Series-Forecasting-for-Portfolio-Management-Optimization\portfolio-optimization\data\processed\cleaned_financial_data.csv"
data = pd.read_csv(data_path, index_col=0, parse_dates=True)

# --- 2. DEFINE OUT-OF-SAMPLE WINDOW ---
# We use the last year (2025 onwards) which the model never saw during training
backtest_start = '2025-01-01'
backtest_data = data.loc[backtest_start:]
daily_returns = backtest_data.pct_change().dropna()

print(f"Backtest Period: {daily_returns.index[0].date()} to {daily_returns.index[-1].date()}")

# --- 3. IMPLEMENT STRATEGY VS BENCHMARK ---

# AI Strategy Weights (from Task 4 Optimization)
# Order: BND, SPY, TSLA
strat_weights = np.array([0.5927, 0.3137, 0.0936])
strat_returns = daily_returns.dot(strat_weights)
strat_cumulative = (1 + strat_returns).cumprod()

# Benchmark Weights (Standard 60/40 Portfolio)
# Order: BND (40%), SPY (60%), TSLA (0%)
bench_weights = np.array([0.40, 0.60, 0.00])
bench_returns = daily_returns.dot(bench_weights)
bench_cumulative = (1 + bench_returns).cumprod()

# --- 4. COMPUTE RISK & PERFORMANCE METRICS ---
def get_performance_metrics(returns, cumulative):
    total_ret = (cumulative.iloc[-1] - 1)
    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol
    
    # Maximum Drawdown calculation
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_dd = drawdown.min()
    
    return {
        "Total Return": total_ret,
        "Annualized Return": ann_ret,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd
    }

strat_stats = get_performance_metrics(strat_returns, strat_cumulative)
bench_stats = get_performance_metrics(bench_returns, bench_cumulative)

# --- 5. VISUALIZATION ---
plt.figure(figsize=(12, 6))
plt.plot(strat_cumulative, label=f"AI Strategy (Total Ret: {strat_stats['Total Return']:.2%})", color='blue', lw=2)
plt.plot(bench_cumulative, label=f"60/40 Benchmark (Total Ret: {bench_stats['Total Return']:.2%})", color='gray', linestyle='--')
plt.title("Cumulative Returns Comparison: AI Strategy vs. 60/40 Benchmark")
plt.ylabel("Portfolio Value (Starting at $1)")
plt.legend()
plt.grid(True, alpha=0.3)

# Save the evidence to the processed data folder
output_dir = "data/processed"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
plt.savefig(os.path.join(output_dir, "backtest_results.png"))
plt.show()

# --- 6. IN-CODE CONCLUSION & REFLECTION ---
"""
BACKTESTING SUMMARY & CONCLUSION:
----------------------------------
1. PERFORMANCE: The 60/40 Benchmark achieved a slightly higher total return (15.07%) 
   compared to the AI Strategy (14.01%). This is expected in a strong equity market 
   as the benchmark holds more SPY (60%) than the AI strategy (31%).

2. RISK MITIGATION: The AI Strategy was more stable, showing a lower Maximum Drawdown 
   (-10.06%) compared to the Benchmark (-11.29%). This proves the AI-driven 
   heavy allocation to Bonds successfully cushioned the portfolio during dips.

3. VIABILITY: The strategy is viable for risk-averse investors who prioritize 
   capital preservation over raw growth. The LSTM forecast helped justify the 
   9.36% TSLA slice, which added growth without excessive volatility.

4. LIMITATIONS: 
   - This backtest assumes zero transaction costs and no slippage.
   - The window is limited to 1 year; results may vary in a recessionary environment.
   - The weights remained static; monthly rebalancing was not implemented here.
"""

print("\n--- Final Metrics Comparison ---")
print(f"AI Strat Sharpe: {strat_stats['Sharpe Ratio']:.2f} | Bench Sharpe: {bench_stats['Sharpe Ratio']:.2f}")
print(f"AI Strat Max DD: {strat_stats['Max Drawdown']:.2%} | Bench Max DD: {bench_stats['Max Drawdown']:.2%}")