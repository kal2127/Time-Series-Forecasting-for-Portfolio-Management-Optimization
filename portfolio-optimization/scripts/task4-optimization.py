import pandas as pd
import numpy as np
from scipy.optimize import minimize
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- 1. Load Data ---
data_path = r"C:\Users\hp\Desktop\Kifya\Time-Series-Forecasting-for-Portfolio-Management-Optimization\portfolio-optimization\data\processed\cleaned_financial_data.csv"
data = pd.read_csv(data_path, index_col=0, parse_dates=True)
print(f"Success! Loaded data from: {data_path}")

# --- 2. Setup Data & Custom View ---
returns = data.pct_change().dropna()
avg_returns = returns.mean() * 252
avg_returns['TSLA'] = 0.45  # Your Custom LSTM View
cov_matrix = returns.cov() * 252

def get_stats(weights):
    weights = np.array(weights)
    ret = np.sum(avg_returns * weights)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = ret / vol
    return np.array([ret, vol, sharpe])

# --- 3. Optimization ---
def min_func_sharpe(weights):
    return -get_stats(weights)[2] 

cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) 
bounds = tuple((0, 1) for _ in range(3)) 
init_guess = [0.33, 0.33, 0.34]

opt_results = minimize(min_func_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
best_weights = opt_results.x
final_stats = get_stats(best_weights)

# --- 4. Simulation for Graphing (The missing piece!) ---
num_portfolios = 5000
vol_arr = np.zeros(num_portfolios)
ret_arr = np.zeros(num_portfolios)
sharpe_arr = np.zeros(num_portfolios)

for i in range(num_portfolios):
    weights = np.random.random(3)
    weights /= np.sum(weights)
    ret_arr[i] = np.sum(avg_returns * weights)
    vol_arr[i] = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_arr[i] = ret_arr[i] / vol_arr[i]

# --- 5. Print Results ---
print("\n--- OPTIMAL PORTFOLIO (MAX SHARPE) ---")
for i, asset in enumerate(avg_returns.index):
    print(f"{asset}: {best_weights[i]:.2%}")
    
print(f"\nExpected Annual Return: {final_stats[0]:.2%}")
print(f"Expected Volatility: {final_stats[1]:.2%}")
print(f"Sharpe Ratio: {final_stats[2]:.2f}") # Fixed: prints as a decimal

# --- 6. Save Deliverables ---
output_path = "data/processed"
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Deliverable 1: Heatmap

plt.figure(figsize=(8, 6))
sns.heatmap(cov_matrix, annot=True, cmap='coolwarm', fmt=".6f")
plt.title("Asset Covariance Heatmap (Task 4)")
plt.savefig(os.path.join(output_path, "covariance_heatmap.png"), dpi=300)
plt.show()

# Deliverable 2: Efficient Frontier

plt.figure(figsize=(10, 6))
plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis', marker='o', s=10, alpha=0.3)
plt.colorbar(label='Sharpe Ratio')
plt.scatter(final_stats[1], final_stats[0], color='red', marker='*', s=200, label='Max Sharpe (Optimal)')
plt.title('Efficient Frontier & Recommended Portfolio')
plt.xlabel('Risk (Volatility)')
plt.ylabel('Expected Return')
plt.legend()
plt.savefig(os.path.join(output_path, "efficient_frontier.png"), dpi=300)
plt.show()