import os
from skimage import io
import numpy as np
from skimage.util import view_as_windows
import pandas as pd

# Define function to tile images
def tiling(image, tile_size, shape):
    pad_h = tile_size - (shape[0] % tile_size) % tile_size
    pad_w = tile_size - (shape[1] % tile_size) % tile_size
    # Perform padding 
    padded_img = np.pad(image, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=3)
    # Perform tiling
    tiles = view_as_windows(padded_img, (tile_size, tile_size), step=tile_size)
    return tiles

# Define function to calculate water percentage
def image_processing(image_path, image_file, tile_size, shape):
    # Read images as array
    image = os.path.join(image_path, image_file) # get image path
    image_arr = io.imread(image)
    
    # Create mask to filter the non-study areas
    ref_image_path = "./data/01_raw/raw_land_map/Landuse_2000.tif"
    ref_image = io.imread(ref_image_path)
    mask = ref_image == 0
    image_arr[mask] = 3 # convert non-study area to pixel 3

    # Perform tiling image
    tiled_image = tiling(image_arr, tile_size, shape)

    # Create empty lists to save data
    x_coord = []
    y_coord = []
    image_source = []
    water_percentage = []
    
    # Calculate water percentage:
    for row in range(tiled_image.shape[0]):
        for col in range(tiled_image.shape[1]):
            current_tile = tiled_image[row, col] # Get the location of current tile
            # Calculate number no_water, mix and water pixels
            no_water = np.sum(current_tile == 0)
            water    = np.sum(current_tile == 1)
            mix      = np.sum(current_tile == 2)
            non_study= np.sum(current_tile == 3)
            # Calculate total pixels
            total_pixels = current_tile.size
            # Calculate the coverage of the study areas
            valid_pixels = total_pixels - non_study
            valid_coverage = valid_pixels / total_pixels
            # Calculate water percentage for cells in the study areas
            if valid_coverage >= 0.5:
                percentage = round(((water + mix * 0.5) / valid_pixels), 2)
            else:
                continue # put water percentage in non-study areas as NaN
            # Append to lists
            x_coord.append(int(col))
            y_coord.append(int(row))
            water_percentage.append(percentage)
            image_source.append(image_file)
    
    data = pd.DataFrame({
    "x_coord": x_coord,
    "y_coord": y_coord,
    "image_source": image_source,
    "water_percentage": water_percentage
    })
    return data

# Perform data extraction with grid size 40x40 pixels
# Read water maps
flood_map_path = "./data/03_aligned/flood_map_aligned/"
flood_images = sorted(os.listdir(flood_map_path)) 

# Define image shape and grid size
shape = (548, 516) # image shape
tile_size = 40 # grid size

data = []

for image in flood_images:
    data_per_image = image_processing(flood_map_path, image, tile_size, shape)
    data.append(data_per_image)

# Save data into a DataFrame
df = pd.concat(data, ignore_index=True)

# Data manipulation
df["image_source"] = df["image_source"].astype(str)
df["date"] = df["image_source"].str[6:14]
df[["year", "date"]] = df["date"].str.split("_", expand = True)
df["year"] = pd.to_numeric(df["year"])
df["date"] = pd.to_numeric(df["date"])

# Create function to impute missing data using linear temporal interpolation
def impute_missing_data(missing_date, missing_year, df): 
    # Create the new rows for the missing data
    new_rows = df[["x_coord", "y_coord"]].drop_duplicates()
    new_rows["date"] = missing_date
    new_rows["year"] = missing_year

    # Add the water_percentage data of the previous week
    previous_week = df[(df["date"] == missing_date-8) & (df["year"] == missing_year)]
    previous_week_values = previous_week[["x_coord", "y_coord", "water_percentage"]]
    previous_week_values = previous_week_values.rename(columns={"water_percentage": "prev_water_percentage"})
    new_rows = pd.merge(previous_week_values, new_rows, on=["x_coord", "y_coord"], how="left")
    
    # Add the water_percentage data of the next week
    next_week = df[(df["date"] == missing_date+8) & (df["year"] == missing_year)]
    next_week_values = next_week[["x_coord", "y_coord", "water_percentage"]]
    next_week_values = next_week_values.rename(columns={"water_percentage": "next_water_percentage"})
    new_rows = pd.merge(new_rows, next_week_values, on=["x_coord", "y_coord"], how="left")
    
    # Calculate the water_percentage of the current year
    new_rows["water_percentage"] = round((new_rows["prev_water_percentage"] + new_rows["next_water_percentage"])/2, 2)
    
    # Add the new rows to the original dataset
    new_rows = new_rows.drop(columns=["prev_water_percentage", "next_water_percentage"])

    return new_rows

# Perform linear imputation on the missing data
new_rows_233_2013 = impute_missing_data(233, 2013, df)
new_rows_257_2019 = impute_missing_data(257, 2019, df)

# Add the missing rows to the original data
df = pd.concat([df, new_rows_233_2013, new_rows_257_2019], ignore_index=True)

# Save to csv file
df.to_csv("data/04_processed/flood_data.csv", index = False)