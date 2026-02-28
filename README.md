# 📈 Stock Price Prediction Model

> 🚧 **This project is currently a work in progress.** Features, models, and documentation are actively being developed and improved.

A machine learning project that predicts stock prices using multiple algorithms and includes portfolio optimization capabilities.

## Overview

This project builds and compares prediction pipelines for **6 stocks** — NKE, IBM, KO, GS, JNJ, and NVDA — using four different models:

| Model                 | Approach                                 |
| --------------------- | ---------------------------------------- |
| **Linear Regression** | Baseline statistical model               |
| **Random Forest**     | Ensemble tree-based model                |
| **XGBoost**           | Gradient-boosted decision trees          |
| **LSTM**              | Deep learning (recurrent neural network) |

Models are evaluated using MAE, RMSE, MAPE, and R² on both training and validation sets.

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
├── portfolio_optimisation/      # Portfolio optimization module
└── Feature_extraction/          # Feature engineering (WIP)
```

## Tech Stack

- **Python** — pandas, NumPy, scikit-learn
- **XGBoost** — gradient boosting
- **TensorFlow / Keras** — LSTM networks
- **Matplotlib** — visualizations
- **yfinance** — stock data retrieval

## Getting Started

1. **Clone the repo**

   ```bash
   git clone https://github.com/aaditya19saini/Stock_price_prediction_model.git
   cd Stock_price_prediction_model
   ```

2. **Install dependencies**

   ```bash
   pip install pandas numpy scikit-learn xgboost tensorflow matplotlib yfinance
   ```

3. **Run the notebooks** — open any `.ipynb` file in Jupyter Notebook or VS Code.

## License

This project is for educational and research purposes.
