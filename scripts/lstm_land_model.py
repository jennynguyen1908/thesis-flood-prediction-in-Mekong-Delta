import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import keras_tuner as kt

# Read datasets
train_val = pd.read_csv("../data/04_processed/train.csv")
test = pd.read_csv("../data/04_processed/dl_test.csv")

train = train_val[train_val["year"] < 2013].copy()
val = train_val[train_val["year"] >= 2012].copy()

seed = 123
np.random.seed(seed)
tf.random.set_seed(seed)

# Normalize x,y coords (0 to 11)
train["x_norm"] = train["x_coord"] / train["x_coord"].max()
train["y_norm"] = train["y_coord"] / train["y_coord"].max()

val["x_norm"] = val["x_coord"] / train["x_coord"].max()
val["y_norm"] = val["y_coord"] / train["y_coord"].max()

test["x_norm"] = test["x_coord"] / train["x_coord"].max()
test["y_norm"] = test["y_coord"] / train["y_coord"].max()

train = train.sort_values(by=["x_coord", "y_coord", "date_time"])
val = val.sort_values(by=["x_coord", "y_coord", "date_time"])
test = test.sort_values(by=["x_coord", "y_coord", "date_time"])

# Create lag feature
train["lag_27"] = train.groupby(["x_coord","y_coord"])["water_percentage"].shift(27)
val["lag_27"] = val.groupby(["x_coord","y_coord"])["water_percentage"].shift(27)
test["lag_27"] = test.groupby(["x_coord","y_coord"])["water_percentage"].shift(27)

train = train.dropna(subset=["lag_27"])
val = val.dropna(subset=["lag_27"])
test = test.dropna(subset=["lag_27"])

def create_lookback_data(data, lookback):
    dataX, dataY = [], []
    coords = []

    # group data per grid and per year
    group_data = data.groupby(["x_coord", "y_coord", "year"])

    for (x_coord, y_coord, year), grid_data_per_year in group_data:
        grid_data_per_year = grid_data_per_year.sort_values("date_time")
        data_values = grid_data_per_year[["water_percentage", "lag_27", "x_norm", "y_norm",
                                "inland_aquaculture", "shrimp_rice_farming", 
                                "single_rice_cropping", "triple_rice_cropping",
                                "double_rice_cropping_dry", "double_rice_cropping_rain",
                                "others"]].values
        datetime_values = grid_data_per_year["date_time"].values

        for i in range(len(data_values) - lookback):
            dataX.append(data_values[i : i + lookback, :])
            dataY.append(data_values[i + lookback, 0])

            # save coordinate/date information for the target prediction
            coords.append([
                x_coord,
                y_coord,
                year,
                datetime_values[i + lookback]
            ])

    coords = pd.DataFrame(
        coords,
        columns=["x_coord", "y_coord", "year", "date_time"])

    return np.array(dataX), np.array(dataY), coords

# Drop irrelevant columns for training/validation
train = train[["year","water_percentage","lag_27","x_coord","y_coord",
                "inland_aquaculture", "shrimp_rice_farming", 
                "single_rice_cropping", "triple_rice_cropping",
                "double_rice_cropping_dry", "double_rice_cropping_rain",
                "others", "date_time", "x_norm", "y_norm"]]
val = val[["year","water_percentage","lag_27","x_coord","y_coord",
                "inland_aquaculture", "shrimp_rice_farming", 
                "single_rice_cropping", "triple_rice_cropping",
                "double_rice_cropping_dry", "double_rice_cropping_rain",
                "others", "date_time", "x_norm", "y_norm"]]
test = test[["year","water_percentage","lag_27","x_coord","y_coord",
                "inland_aquaculture", "shrimp_rice_farming", 
                "single_rice_cropping", "triple_rice_cropping",
                "double_rice_cropping_dry", "double_rice_cropping_rain",
                "others", "date_time", "x_norm", "y_norm"]]


# Create data with lookback=4
lookback = 4
X_train, y_train, coords_train = create_lookback_data(train, lookback)
X_val, y_val, coords_val = create_lookback_data(val, lookback)
X_test, y_test, coords_test = create_lookback_data(test, lookback)

# Instantiate the model with optimal params
lstm_model_land = Sequential()
lstm_model_land.add(Input(shape=(4, 11))) 
lstm_model_land.add(LSTM(units=160, return_sequences=True))
lstm_model_land.add(Dropout(0.4)) 
lstm_model_land.add(LSTM(units=80, return_sequences=False))
lstm_model_land.add(Dropout(0.4))
lstm_model_land.add(Dense(1))
lstm_model_land.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")
# Add EarlyStopping
callbacks = [
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    ModelCheckpoint("best_lstm_land_model.keras", save_best_only=True)
]

# Train the model
lstm_land_training = lstm_model_land.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=callbacks
)
# make predictions
train_predict = lstm_model_land.predict(X_train)
val_predict = lstm_model_land.predict(X_val)
test_predict = lstm_model_land.predict(X_test)

# val_rmse = np.sqrt(mean_squared_error(y_val.reshape(-1), val_predict.reshape(-1)))
# test_rmse = np.sqrt(mean_squared_error(y_test.reshape(-1), test_predict.reshape(-1)))
# val_r2 = r2_score(y_val.reshape(-1), val_predict.reshape(-1))
# test_r2 = r2_score(y_test.reshape(-1), test_predict.reshape(-1))
# print("Validation RMSE: %.4f" % val_rmse)
# print("Test RMSE: %.4f" % test_rmse)
# print("Validation R2:", val_r2)
# print("Test R2:", test_r2)

# save results
train_results = coords_train.copy()
train_results["actual"] = y_train.reshape(-1)
train_results["predicted"] = train_predict.reshape(-1)
train_results.to_csv("../performance/lstm_land_train_predictions.csv", index=False)

val_results = coords_val.copy()
val_results["actual"] = y_val.reshape(-1)
val_results["predicted"] = val_predict.reshape(-1)
val_results.to_csv("../performance/lstm_land_validation_predictions.csv", index=False)

test_results = coords_test.copy()
test_results["actual"] = y_test.reshape(-1)
test_results["predicted"] = test_predict.reshape(-1)
test_results.to_csv("../performance/lstm_land_test_predictions.csv", index=False)

print("LSTM + land-use model performance is saved ✅")