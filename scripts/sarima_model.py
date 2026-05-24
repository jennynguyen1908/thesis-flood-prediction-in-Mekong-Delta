import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_predict
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Read datasets
train = pd.read_csv("../data/04_processed/train.csv")
test = pd.read_csv("../data/04_processed/sarima_test.csv")

train = train[["x_coord", "y_coord", "water_percentage", "date_time"]]
test = test[["x_coord", "y_coord", "water_percentage", "date_time"]]

def extract_grid_data(df, grid):
    grid_data = df[(df["x_coord"] == grid[0]) & (df["y_coord"] == grid[1])]
    grid_data = grid_data[["date_time", "water_percentage"]].copy()
    grid_data["date_time"] = pd.to_datetime(grid_data["date_time"])
    grid_data = grid_data.sort_values("date_time")

    # SARIMA only uses ordered water values
    # date_time is not used as SARIMA index because monsoon data is not continuous
    grid_data = grid_data[["water_percentage"]].reset_index(drop=True)

    return grid_data

def train_grid_SARIMA(data, order, seasonal_order):
    # Train model
    model = SARIMAX(data, 
                    order=order, 
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False, 
                    enforce_invertibility=False)
    result = model.fit(disp=False)
    return result

def generate_forecast(model_result, test_grid_data):
    test_grid_data = test_grid_data.copy()
    test_grid_data["date_time"] = pd.to_datetime(test_grid_data["date_time"])
    test_grid_data = test_grid_data.sort_values("date_time").reset_index(drop=True)

    rolling_result = model_result
    forecast_results = []

    for _, row in test_grid_data.iterrows():

        # Forecast next available monsoon observation
        forecast_value = rolling_result.get_forecast(steps=1).predicted_mean.iloc[0]
        forecast_value = max(forecast_value, 0)

        # Save prediction using the real test timestamp
        forecast_results.append({
            "date_time": row["date_time"],
            "predicted_water": forecast_value
        })

        # After prediction, actual value becomes available
        new_observation = pd.Series(
            [row["water_percentage"]],
            index=[rolling_result.nobs]
        )

        rolling_result = rolling_result.append(
            new_observation,
            refit=False
        )

    forecast_values = pd.DataFrame(forecast_results)
    forecast_values = forecast_values.set_index("date_time")["predicted_water"]

    return forecast_values


import warnings
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ValueWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

grid_parameter = pd.read_csv("sarima_parameters.csv", index_col=0)

import ast
grid_parameter["grid"] = grid_parameter["grid"].apply(ast.literal_eval)
grid_parameter["order"] = grid_parameter["order"].apply(ast.literal_eval)
grid_parameter["seasonal_order"] = grid_parameter["seasonal_order"].apply(ast.literal_eval)

forecast_results = []

for i in range(len(grid_parameter)):
    grid = grid_parameter["grid"][i]
    
    # Extract data for the current grid
    data = extract_grid_data(train,grid)
    
    # Train model with the best hyperparameters for the current grid
    model = train_grid_SARIMA(data,grid_parameter["order"][i],grid_parameter["seasonal_order"][i])
    
    # Get test data for the current grid
    test_grid_data = test.loc[
        (test["x_coord"] == grid[0]) & 
        (test["y_coord"] == grid[1]),
        ["date_time", "water_percentage"]
    ].copy()
    
    # Get rolling one-step-ahead forecasts
    forecast_values = generate_forecast(model, test_grid_data)

    # Save results
    for timestamp, value in forecast_values.items():
        forecast_results.append({
            "x_coord": grid[0],
            "y_coord": grid[1],
            "date_time": timestamp,
            "predicted_water": np.maximum(value, 0)
        })
        
df_forecasts = pd.DataFrame(forecast_results)

# Align the forecast values on grid location and datetime
df_forecasts["date_time"] = pd.to_datetime(df_forecasts["date_time"])
test["date_time"] = pd.to_datetime(test["date_time"])

performance_df = test.merge(
    df_forecasts, 
    on=["x_coord", "y_coord", "date_time"]
)

performance_df.to_csv("./performance/sarima_test_predictions.csv")

print("SARIMA model performance on test set is saved ✅")