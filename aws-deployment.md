# AWS Deployment Guide for Bin-Picking Challenge

This guide outlines the steps to transition from GitHub Actions to AWS for production deployment of your Bin-Picking Challenge solution.

## Prerequisites

1. **AWS Account**:
   - Create an AWS account if you don't have one
   - Set up IAM users with appropriate permissions
   - Install AWS CLI and configure it with your credentials

2. **Docker Hub Access**:
   - Ensure your Docker images are accessible from AWS

## Step 1: Create an Amazon ECR Repository

```bash
# Create a repository for your Docker images
aws ecr create-repository --repository-name bpc-pose-estimator --image-scanning-configuration scanOnPush=true
```

Take note of the repository URI in the output.

## Step 2: Set Up AWS CodeBuild

1. **Create a buildspec.yml file** in your repository root:

```yaml
version: 0.2

phases:
  pre_build:
    commands:
      - echo Logging in to Amazon ECR...
      - aws ecr get-login-password --region $AWS_DEFAULT_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com
      - REPOSITORY_URI=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/bpc-pose-estimator
      - COMMIT_HASH=$(echo $CODEBUILD_RESOLVED_SOURCE_VERSION | cut -c 1-7)
      - IMAGE_TAG=${COMMIT_HASH:=latest}
  
  build:
    commands:
      - echo Building the Docker image...
      - docker buildx build -t $REPOSITORY_URI:latest -t $REPOSITORY_URI:$IMAGE_TAG --file ./Dockerfile.estimator --build-arg="MODEL_DIR=models" .
  
  post_build:
    commands:
      - echo Pushing the Docker image...
      - docker push $REPOSITORY_URI:latest
      - docker push $REPOSITORY_URI:$IMAGE_TAG
      - echo Writing image definitions file...
      - echo '{"ImageURI":"'$REPOSITORY_URI:$IMAGE_TAG'"}' > imageDefinitions.json

artifacts:
  files:
    - imageDefinitions.json
    - appspec.yml
```

2. **Create a CodeBuild project**:
   - Go to AWS CodeBuild console
   - Click "Create build project"
   - Connect to your GitHub repository
   - Set environment to Linux, managed image with privileged mode enabled
   - Set service role with ECR permissions
   - Use the buildspec.yml from your repository
   - Configure build triggers as needed

## Step 3: Set Up AWS CodePipeline (Optional)

For a complete CI/CD pipeline:

1. Go to AWS CodePipeline console
2. Click "Create pipeline"
3. Set up source stage (GitHub)
4. Add build stage (CodeBuild project from Step 2)
5. Add deploy stage if needed (ECS, EKS, etc.)

## Step 4: Test and Validate

1. Trigger a build manually in CodeBuild
2. Verify the image is pushed to ECR
3. Test the image functionality

## Step 5: Production Deployment Options

### Option 1: Amazon ECS

For containerized deployment:

```bash
# Create an ECS cluster
aws ecs create-cluster --cluster-name bpc-cluster

# Register a task definition (create a task-definition.json file first)
aws ecs register-task-definition --cli-input-json file://task-definition.json

# Create a service
aws ecs create-service --cluster bpc-cluster --service-name bpc-service --task-definition bpc-task:1 --desired-count 1
```

### Option 2: Amazon EC2

For direct deployment on EC2 instances:

1. Create an EC2 instance with Docker installed
2. Pull your image from ECR
3. Run the container with appropriate configuration

### Option 3: AWS Batch

For batch processing jobs:

1. Create compute environment
2. Create job queue
3. Define job definitions using your ECR image
4. Submit jobs as needed

## Monitoring and Scaling

1. Set up CloudWatch Alarms for resource utilization
2. Configure Auto Scaling for your deployment
3. Implement CloudWatch Logs for log monitoring

## Cost Optimization

1. Use EC2 Spot Instances for non-critical workloads
2. Implement lifecycle policies for ECR images
3. Use CloudWatch to monitor costs

## Security Best Practices

1. Implement least privilege IAM policies
2. Enable ECR image scanning
3. Use Secrets Manager for sensitive data
4. Enable VPC isolation for your services

## Troubleshooting

- **Image pull failures**: Check ECR permissions
- **Build failures**: Check CodeBuild logs
- **Deployment issues**: Verify task definitions and service configurations

For more information, consult the AWS documentation or open a support case. 