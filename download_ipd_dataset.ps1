# PowerShell script to download the IPD dataset from Hugging Face
# Based on instructions from https://huggingface.co/datasets/bop-benchmark/ipd

$SRC = "https://huggingface.co/datasets/bop-benchmark/ipd/resolve/main"
$OUTPUT_DIR = "ipd_data"

# Create output directory if it doesn't exist
if (-not (Test-Path $OUTPUT_DIR)) {
    New-Item -ItemType Directory -Path $OUTPUT_DIR | Out-Null
    Write-Host "Created directory: $OUTPUT_DIR"
}

# Download the dataset files
Write-Host "Downloading IPD dataset files..."

# Base archive with camera parameters, etc
Write-Host "Downloading ipd_base.zip..."
Invoke-WebRequest -Uri "$SRC/ipd_base.zip" -OutFile "$OUTPUT_DIR/ipd_base.zip"

# 3D object models
Write-Host "Downloading ipd_models.zip..."
Invoke-WebRequest -Uri "$SRC/ipd_models.zip" -OutFile "$OUTPUT_DIR/ipd_models.zip"

# Validation images
Write-Host "Downloading ipd_val.zip..."
Invoke-WebRequest -Uri "$SRC/ipd_val.zip" -OutFile "$OUTPUT_DIR/ipd_val.zip"

Write-Host "Downloads completed. Now extracting files..."

# Extract the files
Write-Host "Extracting ipd_base.zip..."
Expand-Archive -Path "$OUTPUT_DIR/ipd_base.zip" -DestinationPath "$OUTPUT_DIR" -Force

Write-Host "Extracting ipd_models.zip..."
Expand-Archive -Path "$OUTPUT_DIR/ipd_models.zip" -DestinationPath "$OUTPUT_DIR/ipd" -Force

Write-Host "Extracting ipd_val.zip..."
Expand-Archive -Path "$OUTPUT_DIR/ipd_val.zip" -DestinationPath "$OUTPUT_DIR/ipd" -Force

Write-Host "Dataset extraction completed."
Write-Host "The IPD dataset is now available in the $OUTPUT_DIR/ipd directory." 