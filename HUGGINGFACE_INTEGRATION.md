# Hugging Face Integration

## Project Structure

This project uses a hybrid approach combining GitHub and Hugging Face:

### GitHub Repository (This Repository)
- Core algorithm implementation
- Documentation
- Configuration files
- CI/CD pipelines
- Test suites
- Small test data samples

### Hugging Face Resources
- Large datasets
- Model weights
- Training artifacts
- Experiment tracking

## Setup Instructions

### 1. Dependencies
```bash
# Install Hugging Face Hub
pip install huggingface_hub

# Login to Hugging Face (required for private repos)
huggingface-cli login
```

### 2. Dataset Access
The full dataset is hosted on Hugging Face at [URL_TO_BE_ADDED]. To use it:

```python
from huggingface_hub import snapshot_download

# Download dataset
dataset_path = snapshot_download(
    repo_id="AllProAi/bpc-dataset",
    repo_type="dataset"
)
```

### 3. Model Weights
Pre-trained models are available at [URL_TO_BE_ADDED]:

```python
from huggingface_hub import snapshot_download

# Download model weights
model_path = snapshot_download(
    repo_id="AllProAi/bpc-models",
    repo_type="model"
)
```

## Directory Structure
```
.
├── src/                  # Core algorithm implementation
├── tests/                # Test suite
├── configs/              # Configuration files
├── scripts/             
│   ├── train.py         # Training script
│   └── evaluate.py      # Evaluation script
└── data/
    └── samples/         # Small test samples
```

## Configuration

The `.env` file should contain:
```
HF_TOKEN=your_token_here
HF_DATASET_REPO=AllProAi/bpc-dataset
HF_MODEL_REPO=AllProAi/bpc-models
```

## Next Steps
1. [ ] Create Hugging Face dataset repository
2. [ ] Upload initial dataset
3. [ ] Create model repository
4. [ ] Set up CI/CD pipeline for model training
5. [ ] Add experiment tracking integration 