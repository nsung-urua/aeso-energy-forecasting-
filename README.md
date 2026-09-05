# AESO Load and Price Forecasting Project
>This project includes both an exploratory notebook and a production-style Streamlit dashboard for visualization.
> 
## Overview
A machine-learning project exploring short-term electricity demand and pool-price forecasting in Alberta using AESO market data.

The project compares Linear Regression, XGBoost and simple forecasting baselines using leakage-safe time-series features. It also explores whether high-price periods can be identified when predicting the exact magnitude of price spikes becomes difficult.

Key findings

* Alberta electricity load was highly predictable, with the final XGBoost model achieving an R² of approximately 0.98.
* Pool prices were considerably more difficult to forecast, particularly during extreme price events.
* Historical load and price behaviour were important predictors, but model performance highlighted the limitations of forecasting without information such as outages and changing supply conditions.
* Reframing the price problem as high-price risk classification provided another way to extract useful information from the data.

Tools: Python · Pandas · Scikit-learn · XGBoost · Plotly · Streamlit




