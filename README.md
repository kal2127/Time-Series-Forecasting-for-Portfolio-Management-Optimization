AI-Driven Portfolio Management & Optimization
📌 Project Overview

This project presents an AI-driven investment decision system that integrates financial time series forecasting, risk analysis, and Modern Portfolio Theory (MPT) to construct and evaluate optimized investment portfolios.

The system combines:

Deep Learning (LSTM) for price forecasting

Statistical modeling (ARIMA) as a baseline

Monte Carlo simulation for portfolio exploration

Forecast-driven optimization using MPT

Backtesting against a traditional benchmark

The objective is to assess whether AI-based forecasts can improve portfolio stability and risk-adjusted performance compared to passive investing strategies.

🎯 Objectives

Analyze historical financial data and asset risk profiles

Forecast future stock prices using ARIMA and LSTM models

Compare statistical and deep learning forecasting approaches

Construct optimal portfolios using both historical and forecasted returns

Evaluate portfolio performance through backtesting

Develop a disciplined, risk-aware investment strategy

🧩 Assets Used
Asset	Ticker	Risk Profile
Tesla Inc.	TSLA	High growth, high volatility
S&P 500 ETF	SPY	Moderate risk, market exposure
Vanguard Total Bond Market ETF	BND	Low risk, capital preservation
🗂️ Project Structure
├── data/
│   ├── raw/                # Original downloaded market data
│   └── processed/          # Cleaned and transformed datasets
│
├── notebooks/
│   ├── 01_data_preprocessing_eda.ipynb
│   ├── 02_stationarity_volatility.ipynb
│   ├── 03_forecasting_arima_lstm.ipynb
│   ├── 04_portfolio_optimization.ipynb
│   └── 05_backtesting.ipynb
│
├── src/
│   ├── data_utils.py
│   ├── forecasting_models.py
│   ├── portfolio_optimization.py
│   └── evaluation_metrics.py
│
├── README.md
└── requirements.txt

🛠️ Tools & Technologies

Python

NumPy, Pandas

Matplotlib, Seaborn

yfinance

statsmodels (ARIMA)

TensorFlow / Keras (LSTM)

SciPy (Optimization)

Scikit-learn (Metrics)

📊 Methodology & Tasks
Task 1: Data Preprocessing & Exploratory Data Analysis

Historical data collected from January 1, 2015 to January 15, 2026

Handled missing market days using forward filling

Transformed prices into daily percentage returns

Conducted EDA to analyze trends, volatility, and asset behavior

Performed ADF stationarity tests to validate modeling assumptions

Computed risk metrics including Sharpe Ratio and Value at Risk (VaR)

Task 2: Time Series Forecasting

Two forecasting models were implemented to predict Tesla’s price:

🔹 ARIMA (Statistical Model)

Captures linear dependencies

Served as a baseline model

Struggled with high volatility and non-linear behavior

🔹 LSTM (Deep Learning Model)

Used a 60-day sliding window

Captured complex, non-linear price dynamics

Achieved significantly lower forecasting error

📐 Model Evaluation
Model	MAPE
ARIMA	22.47%
LSTM	~14–16%

Selected Model: LSTM

Task 3: Portfolio Optimization (Historical)

Conducted Monte Carlo simulation (5,000 portfolios)

Analyzed risk-return trade-offs

Visualized the efficient frontier using historical returns

Task 4: Forecast-Driven Portfolio Optimization

Integrated LSTM forecast (Expected Return: 45%)

Applied Modern Portfolio Theory

Used scipy.optimize to compute:

Maximum Sharpe Ratio Portfolio

Minimum Volatility Portfolio

📌 Recommended Portfolio (Max Sharpe Ratio)
Asset	Weight
BND	59.27%
SPY	31.37%
TSLA	9.36%

Expected Return: 9.91%
Expected Volatility: 10.27%
Sharpe Ratio: 0.96

Task 5: Strategy Backtesting

Backtested over a 1-year period

Compared against a 60/40 Stock-Bond benchmark

Metric	AI Strategy	60/40 Benchmark
Total Return	14.01%	15.07%
Max Drawdown	-10.06%	-11.29%
Sharpe Ratio	1.18	1.22

📌 Result: AI strategy showed stronger downside protection and improved stability.
