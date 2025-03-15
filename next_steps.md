# Next Steps for Bin-Picking Challenge

## Environment Setup Completed
- ✅ Cloned the official BPC repository
- ✅ Created and activated a Python virtual environment
- ✅ Installed the necessary Python packages
- ✅ Downloaded and extracted the IPD dataset
- ✅ Explored the baseline solution code

## Docker Setup Required
Before proceeding with building and testing the solution, you need to:

1. **Start Docker Desktop**
   - Make sure Docker Desktop is installed and running
   - Verify with `docker ps` to check if the Docker daemon is accessible

2. **Build the Docker Container**
   ```bash
   cd bpc
   docker buildx build -t bpc_pose_estimator:example \
       --file ./Dockerfile.estimator \
       --build-arg="MODEL_DIR=models" \
       .
   ```

3. **Run the Tester**
   For Windows, you'll need to run the Docker containers manually:
   
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

## Implementing Your Solution

1. **Modify the Pose Estimator Code**
   - Edit the `get_pose_estimates` function in `ibpc_pose_estimator_py/ibpc_pose_estimator_py/ibpc_pose_estimator.py`
   - The baseline solution branch (`baseline_solution`) provides a reference implementation

2. **Test Your Solution**
   - Build your custom Docker image with your implementation
   - Run the tester to validate your solution
   - Check the results in the output CSV file

3. **Submit Your Solution**
   - Follow the submission instructions on the [official BPC website](https://bpc.opencv.org/web/challenges/challenge-page/1/submission)

## Dataset Structure Reference

The IPD dataset is organized as follows:
- `models/`: Contains 10 object models in PLY format
- `val/`: Contains validation scenes with multiple cameras
  - Each scene has RGB, depth, AOLP, and DOLP data from three IPS cameras
  - Each scene also has RGB and depth data from one Photoneo camera
  - Ground truth pose annotations are provided in JSON format

## Useful Resources

- [Official BPC Website](https://bpc.opencv.org/)
- [BPC GitHub Repository](https://github.com/opencv/bpc)
- [IPD Dataset on Hugging Face](https://huggingface.co/datasets/bop-benchmark/ipd)
- [BOP Toolkit](https://github.com/thodan/bop_toolkit) 