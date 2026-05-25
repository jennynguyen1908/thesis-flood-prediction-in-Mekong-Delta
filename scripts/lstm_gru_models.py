import numpy as np
import pandas as pd
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import keras_tuner as kt
from tensorflow.keras.layers import GRU

seed = 123
np.random.seed(seed)
tf.random.set_seed(seed)

# Read datasets
train_val = pd.read_csv("../data/04_processed/train.csv")
test = pd.read_csv("../data/04_processed/dl_test.csv")

train = train_val[train_val["year"] < 2013].copy()
val = train_val[train_val["year"] >= 2012].copy()

# Normalize your coords (0 to 11)
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
        data_values = grid_data_per_year[["water_percentage", "lag_27", "x_norm", "y_norm"]].values
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


train_subset = train[["year","water_percentage","lag_27","x_coord","y_coord","date_time","x_norm","y_norm"]]
val_subset = val[["year","water_percentage","lag_27","x_coord","y_coord","date_time","x_norm","y_norm"]]
test_subset = test[["year","water_percentage","lag_27","x_coord","y_coord","date_time","x_norm","y_norm"]]


# Create data with lookback (whole data)
lookback = 4
X_train, y_train, coords_train = create_lookback_data(train, lookback)
X_val, y_val, coords_val = create_lookback_data(val, lookback)
X_test, y_test, coords_test = create_lookback_data(test, lookback)


# Instantiate the model with the most optimal params
lstm_model = Sequential()
lstm_model.add(Input(shape=(4, 4))) 
lstm_model.add(LSTM(units=160, return_sequences=True))
lstm_model.add(Dropout(0.1)) 
lstm_model.add(LSTM(units=16, return_sequences=False))
lstm_model.add(Dropout(0.1))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer=Adam(learning_rate=0.01), loss="mean_squared_error")

# Add EarlyStopping
callbacks = [
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    ModelCheckpoint("best_lstm_water_model.keras", save_best_only=True)
]

# Train the model
lstm_train = lstm_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100, 
    batch_size=32,
    callbacks=callbacks
)

# make predictions
lstm_train_predict = lstm_model.predict(X_train)
lstm_val_predict = lstm_model.predict(X_val)
lstm_test_predict = lstm_model.predict(X_test)

# metrics
# lstm_train_rmse = np.sqrt(mean_squared_error(y_train.reshape(-1), lstm_train_predict.reshape(-1)))
# lstm_val_rmse = np.sqrt(mean_squared_error(y_val.reshape(-1), lstm_val_predict.reshape(-1)))
# lstm_test_rmse = np.sqrt(mean_squared_error(y_test.reshape(-1), lstm_test_predict.reshape(-1)))
# lstm_val_r2 = r2_score(y_val.reshape(-1), lstm_val_predict.reshape(-1))
# lstm_test_r2 = r2_score(y_test.reshape(-1), lstm_test_predict.reshape(-1))

# save results
lstm_train_results = coords_train.copy()
lstm_train_results["actual"] = y_train.reshape(-1)
lstm_train_results["predicted"] = lstm_train_predict.reshape(-1)
lstm_train_results.to_csv("../performance/lstm_train_predictions.csv", index=False)

lstm_val_results = coords_val.copy()
lstm_val_results["actual"] = y_val.reshape(-1)
lstm_val_results["predicted"] = lstm_val_predict.reshape(-1)
lstm_val_results.to_csv("../performance/lstm_validation_predictions.csv", index=False)

lstm_test_results = coords_test.copy()
lstm_test_results["actual"] = y_test.reshape(-1)
lstm_test_results["predicted"] = lstm_test_predict.reshape(-1)
lstm_test_results.to_csv("../performance/lstm_test_predictions.csv", index=False)

# GRU model

gru_model = Sequential()
gru_model.add(Input(shape=(4, 4))) 
gru_model.add(GRU(units=153, return_sequences=True))
gru_model.add(Dropout(0.1)) 
gru_model.add(GRU(units=17, return_sequences=False))
gru_model.add(Dropout(0.1))
gru_model.add(Dense(1))
gru_model.compile(optimizer=Adam(learning_rate=0.01), loss="mean_squared_error")
# Add EarlyStopping
callbacks = [
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    ModelCheckpoint("best_gru_water_model.keras", save_best_only=True)
]

# Train the model
gru_train = gru_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100, # Increased epochs because EarlyStopping will stop it
    batch_size=32,
    callbacks=callbacks
)

# make predictions
gru_train_predict = gru_model.predict(X_train)
gru_val_predict = gru_model.predict(X_val)
gru_test_predict = gru_model.predict(X_test)
# metrics
gru_train_rmse = np.sqrt(mean_squared_error(y_train.reshape(-1), gru_train_predict.reshape(-1)))
gru_val_rmse = np.sqrt(mean_squared_error(y_val.reshape(-1), gru_val_predict.reshape(-1)))
gru_test_rmse = np.sqrt(mean_squared_error(y_test.reshape(-1), gru_test_predict.reshape(-1)))
gru_val_r2 = r2_score(y_val.reshape(-1), gru_val_predict.reshape(-1))
gru_test_r2 = r2_score(y_test.reshape(-1), gru_test_predict.reshape(-1))

# print("Train RMSE: %.4f" % gru_train_rmse)
# print("Validation RMSE: %.4f" % gru_val_rmse)
# print("Test RMSE: %.4f" % gru_test_rmse)
# print("Validation R2:", gru_val_r2)
# print("Test R2:", gru_test_r2)


# save results
gru_train_results = coords_train.copy()
gru_train_results["actual"] = y_train.reshape(-1)
gru_train_results["predicted"] = gru_train_predict.reshape(-1)
gru_train_results.to_csv("../performance/gru_train_predictions.csv", index=False)

gru_val_results = coords_val.copy()
gru_val_results["actual"] = y_val.reshape(-1)
gru_val_results["predicted"] = gru_val_predict.reshape(-1)
gru_val_results.to_csv("../performance/gru_validation_predictions.csv", index=False)

gru_test_results = coords_test.copy()
gru_test_results["actual"] = y_test.reshape(-1)
gru_test_results["predicted"] = gru_test_predict.reshape(-1)
gru_test_results.to_csv("../performance/gru_test_predictions.csv", index=False)