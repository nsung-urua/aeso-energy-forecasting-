# AESO Load Forecasting Project
>This project includes both an exploratory notebook and a production-style Streamlit dashboard for visualization.
> 
## Overview
This project builds a machine learning model to forecast electricity demand in Alberta using AESO data.

## Objective
The goal is to understand how demand responds to key drivers such as temperature and recent load trends.

## Models Used
- Linear Regression
- XGBoost

## Results
- XGBoost achieved higher accuracy compared to Linear Regression
- Captured short-term demand fluctuations effectively

## Tools & Technologies
- Python
- Pandas
- Scikit-learn
- XGBoost
- Streamlit

## How to Run

### 1. Install dependencies
pip install -r requirements.txt

### 2. Run notebook
jupyter notebook

### 3. Run Streamlit dashboard
streamlit run app.py

## Project Structure
- app.py → Streamlit dashboard
- metrics.csv → Model performance
- forecast_plot.png → Visualization of predictions

## Future Improvements
- Add weather forecast inputs
- Improve seasonality modeling
- Deploy live dashboard