# Makefile for reproducing project results

.PHONY: install extract train-sarima train-lstm-gru train-lstm-land train all models clean

install:
	pip install -r requirements.txt

# Extract data from images and preprocess data
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

# Train SARIMA model
train-sarima:
	@echo "Starting SARIMA model training..."
	python scripts/sarima_model.py
	@echo "SARIMA model training completed successfully ✅"

# Train LSTM and GRU models
train-lstm-gru:
	@echo "Starting LSTM and GRU models training..."
	python scripts/lstm_gru_models.py
	@echo "LSTM and GRU models training completed successfully ✅"

# Train LSTM + land-use model
train-lstm-land:
	@echo "Starting LSTM + land-use model training..."
	python scripts/lstm_land_model.py
	@echo "LSTM + land-use model training completed successfully ✅"

# Train all models
train: train-sarima train-lstm-gru train-lstm-land
	@echo "All training completed and results are saved 🎉"

# Full workflow including image extraction
all: install extract train
	@echo "Full workflow completed successfully 🎉"

# Workflow without image extraction
models: install train
	@echo "Model workflow completed successfully 🎉"
