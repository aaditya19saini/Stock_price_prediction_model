# 📈 Stock Price Prediction & Portfolio Optimization

A machine learning project that predicts stock prices using multiple algorithms and optimizes portfolio allocation using Modern Portfolio Theory.

---

## Table of Contents

- [Overview](#overview)
- [Models](#models)
- [Results](#results)
- [Portfolio Optimization](#portfolio-optimization)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Key Learnings](#key-learnings)
- [License](#license)

---

## Overview

This project builds and compares end-to-end prediction pipelines for **6 stocks** — **NKE, IBM, KO, GS, JNJ, and NVDA** — using four different machine learning models. It then takes the predicted close prices and feeds them into a **portfolio optimization** module to find the maximum-Sharpe-ratio allocation.

### Data

- **Historical stock data** is sourced via the [yfinance](https://pypi.org/project/yfinance/) API.
- The raw CSVs in `data/` contain OHLCV data for AAPL, GOOGL, and NFLX (used during initial exploration).
- The prediction pipeline CSVs in `predicition_pipeline/` contain **pre-processed data with 63 columns** of technical indicators and lagged index data for the 6 target stocks.

### Feature Engineering

Each stock's dataset includes:

| Feature Category         | Examples                                                |
| ------------------------ | ------------------------------------------------------- |
| **Price data**           | Open, High, Low, Close                                  |
| **Technical indicators** | Bollinger Bands (Upper/Lower), moving averages          |
| **Market indices**       | QQQ, S&P 500, DJIA close prices                         |
| **Time-lagged features** | Close(t-1) through Close(t-20), index lags up to 5 days |
| **Derived features**     | `Diff` (Close − Open), `High-Low` (daily range)         |

---

## Models

| Model                 | Approach                                   | Type          |
| --------------------- | ------------------------------------------ | ------------- |
| **Linear Regression** | Baseline statistical model                 | Supervised    |
| **Random Forest**     | Ensemble of decision trees                 | Supervised    |
| **XGBoost**           | Gradient-boosted decision trees            | Supervised    |
| **LSTM**              | Recurrent neural network with memory cells | Deep Learning |

### How Each Model Works

**Linear Regression** — Fits a linear relationship between the engineered features and the target close price. Serves as the baseline for comparison.

**Random Forest** — Builds multiple decision trees on random subsets of data and features, then averages their predictions. Reduces variance and overfitting compared to a single tree.

**XGBoost** — Sequentially builds trees where each new tree corrects the errors of the previous ones. Uses gradient descent to minimize the loss function.

**LSTM** — Uses a 60-day lookback window of normalized close prices as input sequences. The architecture is a 2-layer stacked LSTM (50 units each) + Dense(1), trained for 25 epochs with Adam optimizer and MSE loss. Total parameters: ~30,651.

---

## Results

### Validation Performance (Best Model per Stock)

| Stock    | Best Model        | Val R² | Val MAE | Val MAPE (%) |
| -------- | ----------------- | ------ | ------- | ------------ |
| **NKE**  | Linear Regression | 0.99   | 1.19    | 1.52         |
| **IBM**  | Linear Regression | 0.99   | 2.56    | 1.23         |
| **KO**   | Linear Regression | 0.99   | 0.42    | 0.67         |
| **GS**   | Linear Regression | 0.99   | 5.94    | 1.21         |
| **JNJ**  | XGBoost           | 0.96   | 0.91    | 0.61         |
| **NVDA** | Linear Regression | 0.99   | 2.67    | 2.53         |

### Full Model Comparison (Val R² across all stocks)

| Model                 | NKE   | IBM   | KO    | GS    | JNJ  | NVDA  |
| --------------------- | ----- | ----- | ----- | ----- | ---- | ----- |
| **Linear Regression** | 0.99  | 0.99  | 0.99  | 0.99  | 0.95 | 0.99  |
| **Random Forest**     | 0.98  | −2.30 | −0.27 | −0.89 | 0.94 | −2.91 |
| **XGBoost**           | 0.98  | −2.71 | −0.32 | −0.92 | 0.96 | −2.97 |
| **LSTM**              | −7.01 | −3.52 | −1.16 | −1.43 | 0.18 | −3.99 |

> **Key takeaway:** Linear Regression consistently outperformed the more complex models on the validation set. Random Forest and XGBoost showed significant overfitting on several stocks (negative R² on validation). The LSTM was the most challenging to tune and performed poorly with the univariate setup used.

### LSTM Standalone (IBM — Dedicated Notebook)

The `LSTM_and_Lagged_days.ipynb` notebook contains a more focused LSTM experiment on IBM stock, achieving better results:

| Metric   | Value |
| -------- | ----- |
| **R²**   | 0.88  |
| **MAPE** | 3.27% |
| **RMSE** | 0.42  |
| **MAE**  | 0.34  |

---

## Portfolio Optimization

The portfolio optimization module takes the predicted close prices and applies Modern Portfolio Theory to find the optimal allocation.

### Methodology

1. **Stock universe**: NVDA, IBM, JNJ, GS
2. **Portfolio size**: $1,000,000
3. **Returns calculation**: Normalized returns, weighted allocations, and position values
4. **Risk metrics**: Daily returns, cumulative returns, annualized Sharpe Ratio

### Optimization Techniques

| Technique                  | Description                                                                               |
| -------------------------- | ----------------------------------------------------------------------------------------- |
| **Monte Carlo Simulation** | 8,000 random portfolio weight combinations to map the risk-return frontier                |
| **SLSQP Optimizer**        | SciPy's Sequential Least Squares Programming finds the exact maximum-Sharpe-ratio weights |

### Outputs

- Efficient frontier visualization (risk vs. return scatter)
- Optimal portfolio weights pie chart
- Portfolio performance over time
- Sharpe Ratio comparison

---

## Project Structure

```
stock_price_prediction_model/
├── data/                               # Raw stock data (AAPL, GOOGL, NFLX)
│   ├── AAPL.csv
│   ├── GOOGL.csv
│   └── NFLX.csv
├── predicition_pipeline/               # Core prediction notebooks & results
│   ├── prediction_pipelines_using_randomforrest_LSTM_LR_Xgboost.ipynb
│   ├── LSTM_and_Lagged_days.ipynb      # Dedicated LSTM experiment (IBM)
│   ├── evaluation.csv                  # Model comparison metrics (all stocks)
│   ├── metrics.txt                     # Detailed per-stock JSON results
│   ├── notebook_walkthrough.md         # Cell-by-cell LSTM notebook docs
│   ├── NKE.csv, IBM.csv, KO.csv       # Pre-processed stock data
│   ├── GS.csv, JNJ.csv, NVDA.csv      #   with 63 feature columns
│   └── __pycache__/
├── predictions/                        # Standalone prediction notebook
│   └── predictions.ipynb
├── portfolio_optimisation/             # Portfolio optimization module
│   └── portfolio_optimization.ipynb
├── Feature_extraction/                 # Feature engineering (placeholder)
└── README.md
```

---

## Tech Stack

| Category          | Tools                                                    |
| ----------------- | -------------------------------------------------------- |
| **Language**      | Python 3                                                 |
| **Data**          | pandas, NumPy, yfinance                                  |
| **ML Models**     | scikit-learn (Linear Regression, Random Forest), XGBoost |
| **Deep Learning** | TensorFlow / Keras (LSTM)                                |
| **Optimization**  | SciPy (SLSQP)                                            |
| **Visualization** | Matplotlib, Seaborn                                      |

---

## Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook or VS Code with Jupyter extension

### Installation

1. **Clone the repo**

   ```bash
   git clone https://github.com/aaditya19saini/Stock_price_prediction_model.git
   cd Stock_price_prediction_model
   ```

2. **Install dependencies**

   ```bash
   pip install pandas numpy scikit-learn xgboost tensorflow matplotlib seaborn scipy yfinance
   ```

---

## Usage

### Running the Prediction Pipeline

1. Open `predicition_pipeline/prediction_pipelines_using_randomforrest_LSTM_LR_Xgboost.ipynb` in Jupyter
2. Run all cells — the notebook will:
   - Load pre-processed stock CSVs
   - Train Linear Regression, Random Forest, XGBoost, and LSTM models
   - Evaluate each model and save results to `evaluation.csv` and `metrics.txt`
   - Generate comparison visualizations

### Running the LSTM Experiment

1. Open `predicition_pipeline/LSTM_and_Lagged_days.ipynb`
2. This notebook focuses on IBM stock with a dedicated LSTM architecture
3. See `predicition_pipeline/notebook_walkthrough.md` for a detailed cell-by-cell explanation

### Running Portfolio Optimization

1. Open `portfolio_optimisation/portfolio_optimization.ipynb`
2. The notebook uses predicted prices to:
   - Simulate 8,000 random portfolios (Monte Carlo)
   - Optimize weights for maximum Sharpe Ratio (SLSQP)
   - Visualize the efficient frontier and optimal allocation

---

## Key Learnings

- **Simpler models can outperform complex ones** — Linear Regression achieved the best validation R² across most stocks, while tree-based models overfit significantly.
- **LSTM needs careful tuning** — The univariate LSTM setup (using only close price) underperformed. Feeding all 37 engineered features as multivariate input could improve results.
- **Feature engineering matters** — The 63-column pre-processed datasets with technical indicators and lagged cross-market data provided a strong foundation.
- **Portfolio optimization adds practical value** — Moving beyond single-stock prediction to multi-asset allocation with risk-adjusted metrics (Sharpe Ratio) bridges the gap between ML predictions and real-world investment decisions.

---

## License

This project is for educational and research purposes.
