# Python Development Rules for Bin-Picking Challenge
Copyright 2025, Daniel Hill dba Novus Nexum Labs (Daniel@allpro.enterprises)

## Code Style and Formatting

1. **PEP 8 Compliance**
   - Follow PEP 8 style guide for all Python code
   - Maintain line length of 100 characters maximum
   - Use 4 spaces for indentation (not tabs)
   - Separate top-level functions and classes with two blank lines

2. **Naming Conventions**
   - Use `snake_case` for variables, functions, and methods
   - Use `PascalCase` for class names
   - Use `UPPER_CASE` for constants
   - Use descriptive names that clearly indicate purpose
   - Prefix private methods and variables with a single underscore

3. **Imports**
   - Organize imports in three groups:
     1. Standard library imports
     2. Third-party package imports
     3. Local application imports
   - Sort imports alphabetically within each group
   - Use absolute imports where possible

## Documentation

1. **Docstrings**
   - All modules, classes, and functions must have docstrings
   - Follow Google-style docstring format
   - Include descriptions for all parameters, return values, and raised exceptions
   - Document type hints in both annotations and docstrings

2. **Comments**
   - Use comments sparingly to explain complex algorithms
   - Comment on "why" not "what" (the code should be self-explanatory)
   - Update comments when code changes
   - Remove commented-out code before finalizing

## Implementation Guidelines

1. **Data Processing**
   - Use NumPy for array operations
   - Use OpenCV for image processing
   - Implement efficient algorithms for point cloud processing
   - Consider using PyTorch for deep learning components

2. **Error Handling**
   - Use specific exceptions rather than generic ones
   - Handle exceptions at the appropriate level
   - Log errors with sufficient context
   - Provide meaningful error messages

3. **Performance Optimization**
   - Optimize for the 35-second processing time requirement
   - Avoid unnecessary copying of large arrays
   - Use vectorized operations where possible
   - Profile code to identify bottlenecks
   - Consider using multiprocessing for parallel tasks

4. **Testing**
   - Write unit tests for critical functions
   - Test edge cases and error conditions
   - Create validation tests against the training dataset
   - Document testing results and metrics

## Pose Estimator Specific Guidelines

1. **Core Implementation**
   - Focus on the `get_pose_estimates` function in `ibpc_pose_estimator.py`
   - Return results in the format expected by the evaluation pipeline
   - Handle all supported object types

2. **Algorithm Design**
   - Balance accuracy and speed
   - Consider different approaches for different object types
   - Leverage all available camera data (RGB, depth, AOLP, DOLP)
   - Document the rationale for algorithmic choices

3. **Integration Requirements**
   - Ensure compatibility with the tester module
   - Follow the specified interface for input and output
   - Respect timing constraints
   - Validate against the provided dataset 