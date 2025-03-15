# Product Requirements Document (PRD)
# Bin-Picking Perception Challenge

**Date**: March 4, 2025  
**Version**: 1.0  
**Author**: [Your Name]

## 1. Product Overview

### 1.1 Problem Statement
The Bin-Picking Perception Challenge requires developing a computer vision system capable of accurately estimating 6DoF poses (position and orientation) of industrial objects in bin-picking scenarios. The system must be robust to occlusions, varying lighting conditions, and different object arrangements.

### 1.2 Competition Structure
The challenge is structured in two phases:
1. **Phase 1:** Initial evaluation on known objects from the Industrial Plenoptic Dataset (IPD)
2. **Phase 2:** Final evaluation on new, unseen objects (to be released in April)

Participation in Phase 1 is mandatory even for teams focusing primarily on the one-shot solution for Phase 2.

### 1.3 Solution Categories
Two solution categories are supported:
1. **Standard Solution:** Specialized models trained for each phase's specific objects
2. **One-Shot Solution:** A generalized approach that can handle unseen objects without specific training on Phase 2 data

### 1.4 Objectives
- Develop accurate 6DoF pose estimation for industrial objects
- Implement solutions optimized for the provided hardware constraints
- Create a system that generalizes well to previously unseen objects
- Achieve high scores on the official MSSD and mAP metrics

### 1.5 Target Users
- Robotics engineers and researchers
- Industrial automation systems integrators
- Manufacturing companies implementing bin-picking solutions

### 1.6 Success Criteria
- Successful submission to the Bin-Picking Perception Challenge
- Placement in the top 10 teams on the leaderboard
- Potential consideration for the one-shot solution prize
- Successful real-world evaluation in the Intrinsic Flowstate robotics cell

## 2. Requirements

### 2.1 Functional Requirements

#### 2.1.1 Core Functionality
- FR-1: Process multi-view and multimodal data from the Industrial Plenoptic Dataset (IPD)
- FR-2: Accurately estimate 6DoF pose (position and orientation) of objects in bin environments
- FR-3: Handle occlusions and challenging lighting conditions
- FR-4: Process objects of varying shapes, sizes, and materials
- FR-5: Operate within the performance constraints of the challenge evaluation system

#### 2.1.2 Integration Requirements
- IR-1: Package solution as a Docker container per competition requirements
- IR-2: Implement required API endpoints for the evaluation system
- IR-3: Follow the data input/output formats specified in the challenge
- IR-4: Ensure compatibility with the challenge's evaluation framework

### 2.2 Non-Functional Requirements

#### 2.2.1 Performance
- NFR-1: Achieve pose estimation accuracy meeting or exceeding baseline requirements
- NFR-2: Process images within time constraints specified by the challenge
- NFR-3: Optimize for both accuracy and speed

#### 2.2.2 Reliability
- NFR-4: Handle edge cases (reflective objects, similar objects, partial visibility)
- NFR-5: Provide confidence scores with pose estimations
- NFR-6: Gracefully handle error conditions

## 3. Technical Specifications

### 3.1 Data Requirements
- Access to the full Industrial Plenoptic Dataset (IPD)
- CAD models of the objects (if provided)
- Training, validation, and test data splits

### 3.2 Model Architecture
- Deep learning-based pose estimation
- Potential multi-view fusion techniques
- Consideration for one-shot learning approaches

### 3.3 Evaluation Metrics
- **MSSD (Maximum Symmetry-Aware Surface Distance):** Measures the maximum distance between matched object surfaces after applying the predicted pose
- **mAP (mean Average Precision):** Calculated across varying MSSD thresholds [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
- Both metrics account for object symmetries when applicable

### 3.4 Hardware Constraints
- RAM: 16GB
- GPU: NVIDIA L4-class with 24GB VRAM
- Processing time: Evaluations must complete within 12 hours
- Average inference time: Approximately 35 seconds per scene recommended

## 4. Development Timeline

### 4.1 Phase 1: Dataset Exploration and Baseline (2 weeks)
- Download and analyze the IPD dataset
- Implement baseline solution
- Set up Docker development environment

### 4.2 Phase 2: Model Development (4 weeks)
- Develop core pose estimation algorithms
- Implement multi-view integration
- Train and validate models

### 4.3 Phase 3: Optimization and Testing (3 weeks)
- Optimize for performance and accuracy
- Comprehensive testing on validation data
- Performance profiling and improvements

### 4.4 Phase 4: Packaging and Submission (1 week)
- Finalize Docker container
- Document solution approach
- Prepare submission materials

## 5. Risk Analysis

### 5.1 Technical Risks
- TR-1: Insufficient accuracy for challenging objects/scenes
- TR-2: Performance bottlenecks in real-time processing
- TR-3: Overfitting to training data

### 5.2 Mitigation Strategies
- MS-1: Implement ensemble methods and multi-view approaches
- MS-2: Optimize code and use efficient algorithms
- MS-3: Employ robust validation techniques and data augmentation

## 6. Success Metrics

### 6.1 Primary Metrics
- Performance on challenge leaderboard
- Accuracy of pose estimation (using BOP metrics)
- Processing time per scene

### 6.2 Secondary Metrics
- Robustness to different lighting conditions
- Adaptability to unseen objects (for one-shot approaches)
- Memory and computational efficiency

## 7. References
- Official challenge repository
- BOP benchmark documentation
- Industrial Plenoptic Dataset (IPD) specifications 

## Important GitHub Actions Implementation Details

## 1. Setting Up the GitHub Workflow File

- **File Location**: Create directory `.github/workflows/` in your repository and add a file named `docker-build.yml` with the workflow configuration.

- **Basic Structure**:
  ```yaml
  name: Build BPC Docker Image
  
  on:
    push:
      branches: [ main ]
    pull_request:
      branches: [ main ]
    workflow_dispatch:
  
  jobs:
    build:
      runs-on: ubuntu-latest
      steps:
        - name: Checkout code
          uses: actions/checkout@v3
        
        # Additional steps here
  ```

## 2. Repository Access & Secrets

- **For Private Repositories**: 
  - Ensure your GitHub token has sufficient permissions to checkout code
  - For Docker Hub access, add secrets:
    - Go to repository → Settings → Secrets and variables → Actions
    - Add `DOCKER_USERNAME` and `DOCKER_PASSWORD` as secrets

- **Docker Registry Authentication**:
  ```yaml
  - name: Login to Docker Hub
    uses: docker/login-action@v2
    with:
      username: ${{ secrets.DOCKER_USERNAME }}
      password: ${{ secrets.DOCKER_PASSWORD }}
  ```

## 3. Optimize Build Performance

- **Enable Buildx for Better Caching**:
  ```yaml
  - name: Set up Docker Buildx
    uses: docker/setup-buildx-action@v2
  ```

- **Implement Layer Caching**:
  ```yaml
  - name: Build Docker image
    uses: docker/build-push-action@v4
    with:
      context: .
      file: ./Dockerfile.estimator
      build-args: |
        MODEL_DIR=models
      push: false
      tags: bpc_pose_estimator:latest
      cache-from: type=gha
      cache-to: type=gha,mode=max
  ```

## 4. Build Triggers & Controls

- **Branch Control**: Limit builds to specific branches
  ```yaml
  on:
    push:
      branches: [ main, develop ]
      paths:
        - 'ibpc_pose_estimator_py/**'
        - 'Dockerfile.estimator'
  ```

- **Manual Triggering**: Enable workflow_dispatch for manual runs

## 5. Testing & Validation

- **Add Validation Step**: Test your Docker image works correctly
  ```yaml
  - name: Test Docker image
    run: |
      docker run --rm bpc_pose_estimator:latest python -c "import ibpc_pose_estimator_py; print('Import successful')"
  ```

## 6. Artifact Management

- **Save Build Logs**:
  ```yaml
  - name: Archive build logs
    uses: actions/upload-artifact@v3
    with:
      name: build-logs
      path: logs/
      retention-days: 5
  ```

## 7. Notifications & Monitoring

- **Add Build Status Notifications**:
  ```yaml
  - name: Notify on failure
    if: failure()
    uses: actions/github-script@v6
    with:
      script: |
        github.rest.issues.createComment({
          issue_number: context.issue.number,
          owner: context.repo.owner,
          repo: context.repo.repo,
          body: '⚠️ Docker build failed! Check the logs.'
        })
  ```

## 8. Complete Workflow Example

```yaml
name: Build BPC Docker Image

on:
  push:
    branches: [ main ]
    paths:
      - 'ibpc_pose_estimator_py/**'
      - 'Dockerfile.estimator'
  pull_request:
    branches: [ main ]
  workflow_dispatch:

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2
      
      - name: Login to Docker Hub
        if: github.event_name != 'pull_request'
        uses: docker/login-action@v2
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}
      
      - name: Build Docker image
        uses: docker/build-push-action@v4
        with:
          context: .
          file: ./Dockerfile.estimator
          build-args: |
            MODEL_DIR=models
          push: ${{ github.event_name != 'pull_request' }}
          tags: ${{ secrets.DOCKER_USERNAME }}/bpc_pose_estimator:latest
          cache-from: type=gha
          cache-to: type=gha,mode=max
      
      - name: Test Docker image
        run: |
          docker run --rm ${{ secrets.DOCKER_USERNAME }}/bpc_pose_estimator:latest python -c "import ibpc_pose_estimator_py; print('Import successful')" 