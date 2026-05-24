# 🌊 Flood Prediction in the Vietnamese Mekong Delta

## 📌 Project overview:
- This repository contains the code and data used for my master's thesis on flood prediction.
- This project aims to explore time-series models in predicting flood extent in the Vietnamese Mekong Delta using grid-based approach.

## Repository structure
```
├── data/                         # Input and processed datasets
│   ├── 01_raw/                   # Original images
│   ├── 02_interim/               # Fixing inconsistent image format
│   ├── 03_aligned/               # Images after alignment
│   └── 04_processed/             # Data after extraction
│
├── eda/                          # Jupyter notebooks for data exploration
│   ├── eda.ipynb
│   ├── sarima_eda.ipynb
│
├── hyperparameter_tuning/        # Jupyter notebooks for hyperparameter tuning
│   ├── sarima_hyperparameter_tuning.ipynb 
│   ├── sarima_parameters.csv     # SARIMA final hyperparameters
│   ├── lstm_gru_hyperparameter_tuning.ipynb 
│   ├── lstm_land_hyperparameter_tuning.ipynb 
│ 
├── scripts/                      # Python scripts for running the models
│   ├── preprocess_data.py
│   ├── train_sarima.py
│   ├── train_lstm_gru.py
│   └── evaluate_models.py
│
├── performance/                  # Saved model performance
│
├── evaluation/                   # Evaluation results 
│
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
```

## 🗂️ Dataset description
- This project uses datasets from the research titled ["Datasets of land use change and flood dynamics in the Vietnamese Mekong Delta"](https://www.sciencedirect.com/science/article/pii/S235234092200470X)(Vu et al., 2022).
- The datasets can be accessed through this [link](https://data.mendeley.com/datasets/kpftzmsyyz/2).
- There are two type of maps available in the datasets: flood maps and land maps. 
- The datasets are licensed by [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). For this project, the datasets have been modified: aligning and cropping images, and extracting information into tabular format. The original owner of the data retains the ownership of the data during and after the completion of this project.

## 🧩 Project setup
1. Flood extent is presented as water percentage.
2. Study area is divided into grid cells to include spatial features.
3. Historical water percentage data is used as historical flood information.

## ⚙️ Data extraction processed
1. Maps are aligned and cropped using the same coordinate system (longitude and latitude).
2. Each flood map is divided into smaller grid-cells (size: 40x40 pixels).
3. For each grid cell, the number of water pixels are counted and from there, calculate the water percentage per grid-cell.
4. The same process: dividing grid-cells and extracting land-use percentages for each land class is applied to the land maps.
5. Saving the data into tabular formats.

## 🤖 Model Implementation
1. Data split
- Training set: 2000-2012.
- Validation set: 2013-2016.
- Test set: 2017-2020.
2. `SARIMA` as local model
- Hyperparameter tuning for each grid cell.
- Input: historical water percentage data.
- Output: water percentage data forecast.
3. Deep learning `LSTM` `GRU` as global model
- Hyperparameter tuning using Bayesian optimzation.
- Input: `4-week lookback window`, `seasonal lag variable`, `grid coordinates`.
- Output: water percentage prediction.

## 🎯 Research questions
1. Comparing `SARIMA`, `LSTM` and `GRU` using historical flood data as input.
2. Comparing `LSTM with time-series input` and `LSTM with timeseries + land-use inputs`.
3. Error analysis across temporal and spatial patterns.

## 📝 Conclusion
1. No single model clearly outperformed the others. All models showed potential for the prediction task.
2. Adding `land-use` variables did not clearly improve performance. `Historical flood data` remains the key input.
3. Models had higher prediction errors in October and November, and in the northern and western parts of the study area, where there were greater water coverage.

## Reference
Vu, H. T. D., Vu, H. L., Oberle, P., Andreas, S., Nguyen, P. C., & Tran, D. D. (2022). Datasets of land use change and flood dynamics in the vietnamese mekong delta. Data in Brief, 42, 108268. https://doi.org/10.1016/j.dib.2022.108268
