# Time-Series-Forecasting-for-Portfolio-Management-Optimization
Project Overview

This project focuses on the analysis, risk assessment, and forecasting of financial time series data using both statistical and deep learning approaches. The goal is to understand asset behavior, quantify risk, and compare classical forecasting models with modern AI-based techniques.

Three assets with different risk profiles were selected to represent a diversified portfolio:

Tesla (TSLA) – High-growth, high-volatility equity

S&P 500 ETF (SPY) – Broad market, moderate risk

Vanguard Total Bond Market ETF (BND) – Low-risk fixed-income asset

The project is structured in progressive tasks, where each task builds on the insights and outputs of the previous one.

🗂️ Project Structure (Up to Task 2)
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_data_preprocessing_and_eda.ipynb
│   ├── 02_stationarity_and_volatility.ipynb
│   └── 03_forecasting_models.ipynb
├── src/
│   ├── data_utils.py
│   ├── visualization.py
│   └── models.py
├── README.md
└── requirements.txt

📊 Task 1: Data Preprocessing and Exploratory Data Analysis
🔹 Data Collection

Historical financial data was retrieved using the yfinance Python library for the period January 1, 2015 to January 15, 2026. Adjusted closing prices were used to ensure accuracy in return calculations.

🔹 Data Cleaning and Preparation

To prepare the data for time series modeling:

Missing dates caused by weekends and holidays were handled using forward filling (ffill) to maintain a continuous time index.

All datasets were verified to ensure no remaining missing values.

Asset prices were transformed into daily percentage returns to support volatility analysis and stationarity testing.

📈 Exploratory Data Analysis (EDA)
🔹 Price Trend Analysis

TSLA displayed strong exponential growth, reflecting its high-growth nature.

SPY showed consistent long-term growth aligned with overall market trends.

BND remained relatively stable, reinforcing its defensive role.

These patterns confirm the complementary behavior of assets in a diversified portfolio.

🔹 Volatility and Daily Returns

Tesla exhibited the highest volatility with large daily price swings.

SPY demonstrated moderate, market-level volatility.

BND showed minimal daily fluctuations, acting as a volatility stabilizer.

🔹 Rolling Statistics

Using a 30-day rolling window:

TSLA showed time-varying volatility with sharp spikes during major market events.

SPY’s volatility evolved smoothly over time.

BND maintained consistently low volatility.

📐 Stationarity Testing

Stationarity was assessed using the Augmented Dickey-Fuller (ADF) Test:

Raw price series were found to be non-stationary (p-value > 0.05).

Daily return series were stationary (p-value < 0.01).

✔️ This confirmed that daily returns are suitable for statistical forecasting models such as ARIMA.

⚠️ Risk and Volatility Analysis
🔹 Sharpe Ratio

TSLA: High return potential with elevated risk

SPY: Balanced risk-adjusted performance

BND: Low-risk asset with capital preservation characteristics

These results highlight the importance of diversification when constructing financial portfolios.

🤖 Task 2: Forecasting Models (In Progress)

Task 2 focuses on forecasting future price movements using two distinct modeling approaches:

🔹 Statistical Model: ARIMA

Captures linear trends and historical dependencies.

Serves as a strong baseline model for time series forecasting.

🔹 Deep Learning Model: LSTM

Utilizes Long Short-Term Memory networks to learn complex, non-linear patterns.

Employs a 60-day sliding window to predict next-day prices.

🔹 Model Evaluation Strategy

Chronological train-test splitting to preserve time order.

Performance measured using:

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

MAPE (Mean Absolute Percentage Error)

🔹 Initial Findings

Preliminary results indicate that LSTM outperforms ARIMA, particularly for high-volatility assets such as Tesla, by capturing non-linear price dynamics more effectively.

🚀 Current Status and Next Steps
✅ Completed

Data extraction and cleaning

Exploratory data analysis and visualization

Stationarity testing

Volatility analysis and risk metric computation

🔄 In Progress / Next Steps

Finalizing ARIMA/SARIMA and LSTM implementations

Hyperparameter tuning

Detailed model comparison and interpretation

Portfolio optimization and backtesting (future tasks)

🧠 Conclusion (Up to Task 2)

This phase of the project has established a strong analytical foundation by validating the statistical properties of financial time series data and quantifying asset-level risk. The insights gained from exploratory analysis and stationarity testing directly informed the modeling choices in Task 2. With robust preprocessing and evaluation frameworks in place, the project is well-positioned to advance toward accurate forecasting, portfolio optimization, and performance backtesting.
