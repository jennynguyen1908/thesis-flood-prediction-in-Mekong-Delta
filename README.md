# 🌊 Flood Prediction in the Vietnamese Mekong Delta

## 📌 Project overview:
- This repository contains the code and data used for my master's thesis on flood prediction.
- This project aims to explore time-series models in predicting flood extent in the Vietnamese Mekong Delta using grid-based approach.

## Dataset description
- This project uses datasets from the research titled ["Datasets of land use change and flood dynamics in the Vietnamese Mekong Delta"](https://www.sciencedirect.com/science/article/pii/S235234092200470X).
- The datasets can be accessed through this [link](https://data.mendeley.com/datasets/kpftzmsyyz/2).
- There are two type of maps available in the datasets: flood maps and land maps. 
- The datasets are licensed by [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). For this project, the datasets have been modified: aligning and cropping images, and extracting information into tabular format. The original owner of the data retains the ownership of the data during and after the completion of this project.

## ⚙️ Data extraction processed
1. Maps are aligned and cropped using the same coordinate system (longitude and latitude).
2. Each flood map is divided into smaller grid-cells (size: 40x40 pixels).
3. For each grid cell, the number of water pixels are counted and from there, calculate the water percentage per grid-cell.
4. The same process: dividing grid-cells and extracting land-use percentages for each land class is applied to the land maps.
5. Saving the data into tabular formats.

## 🎯 Research questions
1. Comparing `SARIMA`, `LSTM` and `GRU` using historical flood data as input.
2. Comparing `LSTM with time-series input` and `LSTM with timeseries + land-use inputs`.

## Repository structure
