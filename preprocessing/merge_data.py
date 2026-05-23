import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df_water = pd.read_csv("./data/04_processed/flood_data.csv")
df_land = pd.read_csv("./data/04_processed/land_data.csv")

df_water = df_water.sort_values("x_coord")
df_land = df_land.sort_values("x_coord")

# Merge water and land data
df = pd.merge(df_water, df_land, on=["x_coord", "y_coord", "year"], how="left")
# Align the date value for year 2017
df.loc[df["date"]==28, "date"] = 289
# Create a datetime column
df["date_time"] = pd.to_datetime(df["year"].astype(str) + df["date"].astype(str), format="%Y%j")
df = df.sort_values("date_time")
# Save the dataset and splitting into train, validation and test sets
df.to_csv("./data/04_processed/flood_land_data.csv")
# Split data into training, validation and testing set
train_df = df[(df["year"] >= 2000) & (df["year"] < 2017)]
sarima_test_df = df[df["year"] >= 2017]
dl_test_df = df[df["year"] >= 2016]

# Save as separate csv files
train_df.to_csv("train.csv", index=False)
sarima_test_df.to_csv("sarima_test.csv", index=False)
dl_test_df.to_csv("dl_test.csv", index=False)
