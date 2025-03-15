# CI/CD Pipeline for Bin-Picking Challenge

This repository uses GitHub Actions for continuous integration and Docker image building during development, with plans to transition to AWS for production deployment.

## GitHub Actions Setup

The workflow automatically builds a Docker image for the Bin-Picking Challenge pose estimator whenever:
- Code is pushed to the `main` or `develop` branches
- Changes are made to the pose estimator code or Dockerfile
- A pull request is opened against the `main` branch
- The workflow is manually triggered

### Prerequisites

1. **Docker Hub Account**:
   - Create an account at [Docker Hub](https://hub.docker.com/)
   - Create a repository named `bpc_pose_estimator`

2. **GitHub Secrets**:
   - In your GitHub repository, go to Settings → Secrets and variables → Actions
   - Add the following secrets:
     - `DOCKER_USERNAME`: Your Docker Hub username
     - `DOCKER_PASSWORD`: Your Docker Hub access token (create one in Docker Hub → Account Settings → Security)

### How It Works

1. **Automated Builds**:
   - Triggered automatically on code changes
   - Uses Docker Buildx for efficient layer caching
   - Tags images with branch name, commit SHA, and latest
   - Pushes to Docker Hub (except for PR builds)

2. **Testing**:
   - Basic validation to ensure the Docker image can load the pose estimator module
   - Add more comprehensive tests by modifying the workflow file

3. **Logging and Notifications**:
   - Logs are saved as artifacts for 5 days
   - Failed builds add comments to associated PRs

### Manual Triggering

You can manually trigger a build by:
1. Going to the Actions tab in your repository
2. Selecting the "Build BPC Docker Image" workflow
3. Clicking "Run workflow"
4. Selecting the branch to build from
5. Clicking "Run workflow"

## Using the Docker Image

After a successful build, the Docker image will be available at:
```
docker pull {your-username}/bpc_pose_estimator:latest
```

Or with a specific tag:
```
docker pull {your-username}/bpc_pose_estimator:{tag}
```

## Future AWS Integration

For production deployment, we plan to transition to AWS:
1. Create an ECR repository for the Docker image
2. Set up AWS CodeBuild for production builds
3. Implement a full CI/CD pipeline with CodePipeline

## Troubleshooting

Common issues and solutions:
- **Docker build fails**: Check the build logs for specific errors
- **Push to Docker Hub fails**: Verify your Docker Hub credentials are correct
- **Tests fail**: Ensure your code is properly structured and dependencies are correct

For more help, open an issue on GitHub or contact the repository maintainers. 