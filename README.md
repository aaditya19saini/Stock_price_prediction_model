# 📈 Stock Price Prediction & Portfolio Optimization

> 🚧 **This project is currently a work in progress.** Features, models, and documentation are actively being developed and improved.

A machine learning project that predicts stock prices using multiple algorithms and optimizes portfolio allocation using Modern Portfolio Theory.

## Overview

This project builds and compares prediction pipelines for **6 stocks** — NKE, IBM, KO, GS, JNJ, and NVDA — using four different models:

| Model                 | Approach                                 |
| --------------------- | ---------------------------------------- |
| **Linear Regression** | Baseline statistical model               |
| **Random Forest**     | Ensemble tree-based model                |
| **XGBoost**           | Gradient-boosted decision trees          |
| **LSTM**              | Deep learning (recurrent neural network) |

Models are evaluated using MAE, RMSE, MAPE, and R² on both training and validation sets.

### Portfolio Optimization

The project also includes a portfolio optimization module that:

- Constructs a portfolio of **NVDA, IBM, JNJ, and GS** using predicted close prices
- Computes normalized returns, weighted allocations, and position values for a $1M portfolio
- Calculates daily returns, cumulative returns, and the **Annualized Sharpe Ratio**
- Runs a **Monte Carlo simulation** (8,000 random portfolios) to explore the risk-return frontier
- Uses **SciPy's SLSQP optimizer** to find the maximum-Sharpe-ratio portfolio weights
- Visualizes optimal allocation via pie charts and portfolio performance plots

## Project Structure

```
stock_price_prediction_model/
├── data/                        # Raw stock data (AAPL, GOOGL, NFLX)
├── predicition_pipeline/        # Core prediction notebooks & results
│   ├── prediction_pipelines_using_randomforrest_LSTM_LR_Xgboost.ipynb
│   ├── LSTM_and_Lagged_days.ipynb
│   ├── evaluation.csv           # Model comparison metrics
│   └── metrics.txt              # Detailed per-stock results
├── predictions/                 # Standalone prediction notebook
├── portfolio_optimisation/      # Portfolio optimization (Monte Carlo + SLSQP)
│   └── portfolio_optimization.ipynb
└── Feature_extraction/          # Feature engineering (WIP)
```

## Tech Stack

- **Python** — pandas, NumPy, seaborn
- **scikit-learn** — Random Forest, feature selection, preprocessing
- **XGBoost** — gradient boosting
- **TensorFlow / Keras** — LSTM networks
- **SciPy** — constrained optimization (SLSQP)
- **Matplotlib / Seaborn** — visualizations
- **yfinance** — stock data retrieval

## Getting Started

1. **Clone the repo**

   ```bash
   git clone https://github.com/aaditya19saini/Stock_price_prediction_model.git
   cd Stock_price_prediction_model
   ```

2. **Install dependencies**

   ```bash
   pip install pandas numpy scikit-learn xgboost tensorflow matplotlib seaborn scipy yfinance
   ```

3. **Run the notebooks** — open any `.ipynb` file in Jupyter Notebook or VS Code.

## License

This project is for educational and research purposes.
