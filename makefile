# Makefile for reproducing project results
install: pip install -r requirements.txt

# Extract data from images
extract:
@echo "Starting grid-based data extraction from images..."
python preprocessing/convert_image_type.py
python preprocessing/align_map_coordinates.py
python preprocessing/data_extraction_flood_map.py
python preprocessing/data_extraction_land_map.py
@echo "Grid-based data extraction completed successfully ✅"
@echo "Starting data preprocessing..."
python preprocessing/merge_data.py
@echo "Data preprocessing completed successfully ✅"

# Train models
train-sarima: 
@echo "Starting SARIMA model training..."
python scripts/sarima_model.py
@echo "SARIMA model training completed successfully ✅"

train-lstm-gru:
@echo "Starting LSTM and GRU models training..."
python scripts/lstm_gru_models.py
@echo "LSTM and GRU models training completed successfully ✅"

train-lstm-land:
@echo "Starting LSTM + land-use model training..."
python scripts/lstm_land_model.py
@echo "LSTM + land-use model training completed successfully ✅"

@echo "All training completed and results are saved 🎉"

# Full workflow: 
all: install extract train-sarima train-lstm-gru train-lstm-land
models: install train-sarima train-lstm-gru train-lstm-land