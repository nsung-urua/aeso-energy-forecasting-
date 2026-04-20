import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="AESO Load Forecast Dashboard", layout="wide")

st.title("AESO Load Forecast Dashboard")
st.write(
    """
    This dashboard compares a Linear Regression model and an XGBoost model
    for short-term electricity load forecasting using AESO hourly data.
    """
)

# Load data
results_df = pd.read_csv("model_outputs.csv")
metrics_df = pd.read_csv("metrics.csv")

# Convert timestamp column to datetime
results_df["timestamp"] = pd.to_datetime(results_df["timestamp"])

# Sidebar controls
model_choice = st.sidebar.selectbox(
    "Choose model to visualize",
    ["Linear Regression", "XGBoost"]
)

num_points = st.sidebar.slider(
    "Number of points to display",
    min_value=24,
    max_value=500,
    value=200,
    step=24
)

# Metrics section
st.subheader("Model Performance")
st.dataframe(metrics_df, use_container_width=True)

# Pick prediction column based on selection
if model_choice == "Linear Regression":
    pred_col = "pred_lr"
else:
    pred_col = "pred_xgb"

# Filter recent rows
plot_df = results_df.tail(num_points)

# Plot section
st.subheader(f"Actual vs Predicted Load: {model_choice}")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(plot_df["timestamp"], plot_df["actual_load"], label="Actual Load")
ax.plot(plot_df["timestamp"], plot_df[pred_col], label=f"Predicted Load ({model_choice})")

ax.set_title(f"Actual vs Predicted Electricity Load ({model_choice})")
ax.set_xlabel("Time")
ax.set_ylabel("Load (MW)")
ax.legend()
ax.grid(True)

plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)

# Show data table
st.subheader("Recent Forecast Data")
st.dataframe(plot_df, use_container_width=True)


# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
