import os
import rasterio
import numpy as np
from rasterio.warp import reproject, Resampling

# Transform all images to match the first image of the landmap
# 1. Open the first image as the Master image
def transform_coor(ref_image, image_path, image, save_path):
    ref_path = "./dataset/land_map/"
    with rasterio.open(ref_path + ref_image) as master:
        master_profile = master.profile
        master_kwarg = master_profile.copy()
        dst_crs = master.crs
        dst_transform = master.transform
        dst_width = master.width
        dst_height = master.height
    
    # 2. Open all other images
    with rasterio.open(image_path + image) as src:
        # Prepare the output array based on Master's dimensions
        destination_data = np.zeros((src.count, dst_height, dst_width))
    
        # 3. Perform the Reprojection
        reproject(
            source=rasterio.band(src, range(1, src.count + 1)),
            destination=destination_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest # Use nearest for classification (0, 1, 2)
        )
    
    # 4. Save the result to a new folder using the Master's spatial profile
    with rasterio.open(save_path + image, 'w', **master_kwarg) as dst:
        dst.write(destination_data)

# Make new folder 
new_path = "./data/03_aligned/flood_map_aligned/"

if not os.path.exists(new_path):
    os.makedirs(new_path)

# Read flood maps
water_path = "./02_interim/flood_map_clean/"
water_images = sorted(os.listdir(water_path))

# Read land images
land_path = "./01_raw/raw_land_map/"
land_images = sorted(os.listdir(land_path))
ref_image = land_images[0]

# Perform the transformation
for image in water_images:
    transform_coor(land_images[0], water_path, image, new_path)

# ------------------------------------------------
# Test if all files match with the reference image
ref_path = land_path + land_images[0]

with rasterio.open(ref_path) as master:
    m_res = master.res
    m_shape = master.shape
    m_crs = master.crs
    m_transform = master.transform

print(f"Comparing all files in folder to: {ref_path}")
print(f"Target Shape: {m_shape} | Target Res: {m_res}\n" + "-"*50)

mismatched = []

for filename in os.listdir(new_path):
    if filename.endswith((".tif", ".tiff")):
        file_path = os.path.join(new_path, filename)
        
        try:
            with rasterio.open(file_path) as ds:
                # 1. Check Shape 
                shape_match = (ds.shape == m_shape)
                
                # 2. Check Resolution (Allow for tiny rounding differences)
                res_match = np.all(np.isclose(ds.res, m_res, atol=1e-5))
                
                # 3. Check Transform (Coordinates must align)
                trans_match = all(np.isclose(ds.transform, m_transform, atol=1e-5))
                
                if not (shape_match and res_match and trans_match):
                    mismatched.append((filename, ds.shape, ds.res))
                    print(f"❌ {filename} -> Shape: {ds.shape}, Res: {ds.res}")
        
        except Exception as e:
            print(f"⚠️ Error reading {filename}: {e}")

if not mismatched:
    print("\n✅ ALL CLEAR: All images match the master spatial profile.")
else:
    print(f"\nSummary: {len(mismatched)} files do not match the master template.")