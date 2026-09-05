# AESO Load and Price Forecasting Project
>This project includes both an exploratory notebook and a production-style Streamlit dashboard for visualization.
> 
## Overview
A machine-learning project exploring short-term electricity demand and pool-price forecasting in Alberta using AESO market data.

The project compares Linear Regression, XGBoost and simple forecasting baselines using leakage-safe time-series features. It also explores whether high-price periods can be identified when predicting the exact magnitude of price spikes becomes difficult.

**Key findings**

* Load forecasting performed strongly: both Linear Regression and XGBoost achieved an R² of about 0.98, with XGBoost reaching an MAE of roughly 55 MW.
* Electricity prices were much harder to predict: XGBoost improved overall price-fit metrics versus simple baselines, but extreme price spikes remained difficult to forecast accurately.
* Recent history was highly important: previous-hour load and price were among the strongest predictors.
* High-price risk classification was promising: Because predicting the exact size of electricity price spikes was difficult, the analysis also tested whether machine learning could instead identify periods at higher risk of prices reaching $100/MWh or more.
  ** The classifier achieved a ROC-AUC of ~0.96. ROC-AUC measures how well a model distinguishes between two groups—in this case, high-price and normal-price hours across different decision thresholds. A score of 0.5 represents essentially random ranking, while 1.0 represents perfect separation, so 0.96 indicates strong ability to rank high-price conditions above normal-price conditions. This does not mean the model was 96% accurate; it should be interpreted as a risk-identification tool rather than an exact price predictor.

* Main takeaway: machine learning can identify useful patterns, but model performance depends heavily on the quality of the inputs. Domain knowledge and human judgement remain important for interpreting results and understanding real-world market conditions.


Tools: Python · Pandas · Scikit-learn · XGBoost · Plotly · Streamlit




