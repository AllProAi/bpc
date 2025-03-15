# Bin-Picking Challenge Setup Summary

## Completed Steps

1. **Environment Setup**
   - Cloned the official BPC repository: `git clone https://github.com/opencv/bpc.git`
   - Created a Python virtual environment: `python -m venv bpc_ws/bpc_env`
   - Activated the virtual environment: `.\bpc_env\Scripts\Activate.ps1`
   - Installed the ibpc package: `pip install ibpc`

2. **Dataset Download**
   - Created a PowerShell script to download the IPD dataset from Hugging Face
   - Downloaded and extracted the following dataset components:
     - `ipd_base.zip`: Base archive with camera parameters
     - `ipd_models.zip`: 3D object models
     - `ipd_val.zip`: Validation images

3. **Repository Exploration**
   - Examined the baseline solution branch (`baseline_solution`)
   - Reviewed the pose estimator implementation in `ibpc_pose_estimator_py/ibpc_pose_estimator_py/ibpc_pose_estimator.py`

## Dataset Structure

The IPD dataset contains:
- 10 object models in PLY format
- Multiple validation scenes with RGB, depth, AOLP, and DOLP data
- Camera calibration information
- Ground truth pose annotations

Each scene contains data from multiple cameras:
- Three IPS cameras (cam1, cam2, cam3) with RGB, depth, AOLP, and DOLP data
- One Photoneo depth camera with RGB and depth data

## Next Steps

1. **Build the Docker Container**
   ```bash
   cd bpc
   docker buildx build -t bpc_pose_estimator:example \
       --file ./Dockerfile.estimator \
       --build-arg="MODEL_DIR=models" \
       .
   ```

2. **Run the Tester**
   ```bash
   bpc test bpc_pose_estimator:example ipd
   ```
   
   Note: On Windows, you may need to run the Docker containers manually:
   
   a. Start the Zenoh router:
   ```bash
   docker run --init --rm --net host eclipse/zenoh:1.2.1 --no-multicast-scouting
   ```
   
   b. Run the pose estimator:
   ```bash
   docker run --network=host bpc_pose_estimator:example
   ```
   
   c. Run the tester:
   ```bash
   docker run --network=host -e BOP_PATH=/opt/ros/underlay/install/datasets -e SPLIT_TYPE=val -v<PATH_TO_DATASET>:/opt/ros/underlay/install/datasets -v<PATH_TO_OUTPUT_DIR>:/submission -it bpc_tester:latest
   ```

3. **Implement Your Solution**
   - Modify the `get_pose_estimates` function in `ibpc_pose_estimator_py/ibpc_pose_estimator_py/ibpc_pose_estimator.py`
   - Build and test your custom Docker image
   - Submit your solution according to the challenge requirements

## Resources

- [Official BPC Website](https://bpc.opencv.org/)
- [BPC GitHub Repository](https://github.com/opencv/bpc)
- [IPD Dataset on Hugging Face](https://huggingface.co/datasets/bop-benchmark/ipd)
- [BOP Toolkit](https://github.com/thodan/bop_toolkit) 