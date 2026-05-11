import numpy as np
from skimage import io
import os
from skimage.util import view_as_windows
import pandas as pd

water_df = pd.read_csv("./data/04_processed/flood_data.csv")
master_coords_df = water_df[["x_coord", "y_coord"]].drop_duplicates().sort_values(by=["y_coord", "x_coord"])
master_coords = master_coords_df[["x_coord", "y_coord"]].values.tolist()

# Define function to tile images
def tiling(image, tile_size, shape):
    pad_h = (tile_size - (shape[0] % tile_size)) % tile_size
    pad_w = (tile_size - (shape[1] % tile_size)) % tile_size
    # Perform padding 
    padded_img = np.pad(image, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
    # Perform tiling
    tiles = view_as_windows(padded_img, (tile_size, tile_size), step=tile_size)
    return tiles

# Define function to calculate water percentage
# Define function to calculate water percentage
def land_data_extraction(image_path, image_file, tile_size, shape, master_coords):
    # Read images as array
    image = os.path.join(image_path, image_file) # get image path
    image_arr = io.imread(image)

    # Create a blank array of the target shape (filled with 0 for non-study)
    target_h, target_w = 548, 516
    standardized_arr = np.zeros((target_h, target_w), dtype=image_arr.dtype)
    
    # Determine the overlapping area (crop if larger, pad if smaller)
    h = min(image_arr.shape[0], target_h)
    w = min(image_arr.shape[1], target_w)
    
    # Paste the source image into the standardized template
    standardized_arr[:h, :w] = image_arr[:h, :w]

    # Perform tiling image
    tiled_image = tiling(standardized_arr, tile_size, shape)

    # Create empty lists to save data
    results = []

    # Calculate percentage for each land type:
    for location in master_coords:
        current_tile = tiled_image[location[1], location[0]] # Get the location of current tile
        # Calculate number no_water, mix and water pixels
        non_study = np.sum(current_tile == 0)
        
        # Calculate total pixels
        total_pixels = current_tile.size
        # Calculate the coverage of the study areas
        valid_pixels = total_pixels - non_study

        # Calculate water percentage for cells in the study areas
        res = {
            "x_coord": location[0],
            "y_coord": location[1],
            "image_source": image_file,
            "inland_aquaculture": round(np.sum(current_tile == 1) / valid_pixels, 2),
            "shrimp_rice_farming": round(np.sum(current_tile == 2) / valid_pixels, 2),
            "single_rice_cropping": round(np.sum(current_tile == 6) / valid_pixels, 2),
            "triple_rice_cropping": round(np.sum(current_tile == 7) / valid_pixels, 2),
            "double_rice_cropping_dry": round(np.sum(current_tile == 8) / valid_pixels, 2),
            "double_rice_cropping_rain": round(np.sum(current_tile == 9) / valid_pixels, 2),
            "others": round(np.sum(current_tile == 5) / valid_pixels, 2)
        }
        results.append(res)

    return pd.DataFrame(results)


# Read land images
land_path = "./data/01_raw/raw_land_map/"
land_images = sorted(os.listdir(land_path))

# Perform data extraction
shape = (548, 516) 
tile_size = 40

data = []

for image in land_images:
    data_per_image = land_data_extraction(land_path, image, tile_size, shape, master_coords)
    data.append(data_per_image)

# Save data into a DataFrame
df = pd.concat(data, ignore_index=True)

# Data manipulation
df["image_source"] = df["image_source"].astype(str)
df["year"] = df["image_source"].str[8:12]

# Save dataframe
df.to_csv("data/04_processed/land_data.csv", index = False)