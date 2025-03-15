# General Development Rules for Bin-Picking Challenge
Copyright 2025, Daniel Hill dba Novus Nexum Labs (Daniel@allpro.enterprises)

## Project Structure

1. **Repository Organization**
   - Maintain the existing directory structure of the cloned BPC repository
   - Keep all custom implementations within their designated directories
   - Add supporting documentation to the `docs/` directory
   - Store configuration files in the `configs/` directory

2. **File Naming Conventions**
   - Use snake_case for Python files and directories
   - Use kebab-case for documentation files
   - Prefix custom utility files with `bpc_`
   - All filenames should clearly indicate their purpose

3. **Code Organization**
   - Each module should have a single, well-defined responsibility
   - Group related functionality together
   - Maintain logical separation of:
     - Data processing
     - Pose estimation algorithms
     - Utility functions
     - Visualization tools

## Development Process

1. **Pre-Implementation Requirements**
   - Ensure all planning documents are in place before coding
   - Review the challenge requirements thoroughly
   - Understand the evaluation metrics before implementation

2. **Implementation Workflow**
   - Develop code locally without Docker
   - Push changes to GitHub for Docker builds via Actions
   - Monitor build progress through GitHub Actions
   - Document all major decisions and approaches

3. **GitHub Workflow**
   - Regularly commit changes with descriptive messages
   - Push to GitHub to trigger automated builds
   - Use feature branches for experimental implementations
   - Merge only tested and validated code to main

4. **Code Quality Standards**
   - Follow PEP 8 style guide for Python code
   - Keep functions small and focused on a single task
   - Document all functions with docstrings
   - Use type hints for function parameters and return values

## Risk Management

1. **Error Handling**
   - Implement comprehensive error handling
   - Log errors with sufficient context for debugging
   - Ensure the pose estimator fails gracefully
   - Catch and handle exceptions at appropriate levels

2. **Performance Considerations**
   - Optimize for the 35-second processing time requirement
   - Balance accuracy and speed in the implementation
   - Profile code to identify and address bottlenecks
   - Consider using parallel processing where appropriate

3. **Data Handling**
   - Validate all input data before processing
   - Handle edge cases in image processing
   - Account for missing or corrupted data
   - Ensure compatibility with the evaluation pipeline 