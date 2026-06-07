"""
Amazon SageMaker deployment script.

Steps
-----
1. Authenticate to ECR and push the inference Docker image.
2. Create a SageMaker Model pointing to the ECR image and S3 model artifacts.
3. Create (or update) a SageMaker endpoint configuration.
4. Deploy the endpoint, then optionally configure Application Auto Scaling.

Prerequisites
-------------
- AWS credentials configured (IAM role with SageMaker + ECR + S3 permissions).
- Docker daemon running locally.
- The inference image built:  docker build -t fraud-classifier:latest -f deployment/Dockerfile .
- Model artifact uploaded to S3:
    aws s3 cp models/fraud_classifier.pkl s3://<BUCKET>/fraud-classifier/model.tar.gz

Environment variables
---------------------
  AWS_REGION              (default: us-east-1)
  AWS_ACCOUNT_ID          (required)
  ECR_REPO_NAME           (default: fraud-classifier)
  SAGEMAKER_ROLE_ARN      (required)
  S3_MODEL_ARTIFACT_URI   e.g. s3://my-bucket/fraud-classifier/model.tar.gz
  ENDPOINT_NAME           (default: fraud-classifier-endpoint)
  INSTANCE_TYPE           (default: ml.m5.large)
  INITIAL_INSTANCE_COUNT  (default: 1)
"""

import logging
import os
import subprocess
import sys
import time

import boto3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_ACCOUNT_ID = os.getenv("AWS_ACCOUNT_ID")
ECR_REPO_NAME = os.getenv("ECR_REPO_NAME", "fraud-classifier")
SAGEMAKER_ROLE_ARN = os.getenv("SAGEMAKER_ROLE_ARN")
S3_MODEL_ARTIFACT_URI = os.getenv("S3_MODEL_ARTIFACT_URI", "")
ENDPOINT_NAME = os.getenv("ENDPOINT_NAME", "fraud-classifier-endpoint")
INSTANCE_TYPE = os.getenv("INSTANCE_TYPE", "ml.m5.large")
INITIAL_INSTANCE_COUNT = int(os.getenv("INITIAL_INSTANCE_COUNT", "1"))
IMAGE_TAG = os.getenv("IMAGE_TAG", "latest")

_REQUIRED = ["AWS_ACCOUNT_ID", "SAGEMAKER_ROLE_ARN", "S3_MODEL_ARTIFACT_URI"]


def _check_env():
    missing = [k for k in _REQUIRED if not os.getenv(k)]
    if missing:
        logger.error("Missing required environment variables: %s", missing)
        sys.exit(1)


# ── ECR helpers ───────────────────────────────────────────────────────────────

def get_ecr_image_uri() -> str:
    return f"{AWS_ACCOUNT_ID}.dkr.ecr.{AWS_REGION}.amazonaws.com/{ECR_REPO_NAME}:{IMAGE_TAG}"


def ensure_ecr_repo(ecr_client) -> None:
    try:
        ecr_client.describe_repositories(repositoryNames=[ECR_REPO_NAME])
        logger.info("ECR repository '%s' already exists.", ECR_REPO_NAME)
    except ecr_client.exceptions.RepositoryNotFoundException:
        ecr_client.create_repository(repositoryName=ECR_REPO_NAME)
        logger.info("Created ECR repository '%s'.", ECR_REPO_NAME)


def docker_login_ecr(ecr_client) -> None:
    token = ecr_client.get_authorization_token()
    registry = token["authorizationData"][0]["proxyEndpoint"]
    import base64
    creds = base64.b64decode(token["authorizationData"][0]["authorizationToken"]).decode()
    username, password = creds.split(":", 1)
    _run(["docker", "login", "--username", username, "--password-stdin", registry],
         input_data=password.encode())


def _run(cmd: list, input_data: bytes = None) -> None:
    result = subprocess.run(cmd, input=input_data, capture_output=True)
    if result.returncode != 0:
        logger.error("Command failed: %s\n%s", " ".join(cmd), result.stderr.decode())
        sys.exit(1)


def push_image(ecr_client) -> str:
    image_uri = get_ecr_image_uri()
    local_tag = f"{ECR_REPO_NAME}:latest"

    logger.info("Tagging %s → %s", local_tag, image_uri)
    _run(["docker", "tag", local_tag, image_uri])

    logger.info("Authenticating to ECR…")
    docker_login_ecr(ecr_client)

    logger.info("Pushing %s…", image_uri)
    _run(["docker", "push", image_uri])

    logger.info("Image pushed: %s", image_uri)
    return image_uri


# ── SageMaker helpers ─────────────────────────────────────────────────────────

def create_or_update_model(sm_client, model_name: str, image_uri: str) -> None:
    model_data_url = S3_MODEL_ARTIFACT_URI
    try:
        sm_client.delete_model(ModelName=model_name)
        logger.info("Deleted existing SageMaker model '%s'.", model_name)
    except sm_client.exceptions.ClientError:
        pass

    sm_client.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": image_uri,
            "ModelDataUrl": model_data_url,
            "Environment": {
                "MODEL_PATH": "/opt/ml/model/fraud_classifier.pkl",
                "FRAUD_THRESHOLD": "0.5",
            },
        },
        ExecutionRoleArn=SAGEMAKER_ROLE_ARN,
    )
    logger.info("SageMaker model '%s' created.", model_name)


def create_endpoint_config(sm_client, config_name: str, model_name: str) -> None:
    try:
        sm_client.delete_endpoint_config(EndpointConfigName=config_name)
    except sm_client.exceptions.ClientError:
        pass

    sm_client.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[{
            "VariantName": "AllTraffic",
            "ModelName": model_name,
            "InstanceType": INSTANCE_TYPE,
            "InitialInstanceCount": INITIAL_INSTANCE_COUNT,
            "InitialVariantWeight": 1,
        }],
    )
    logger.info("Endpoint config '%s' created.", config_name)


def deploy_endpoint(sm_client, config_name: str) -> None:
    existing = None
    try:
        existing = sm_client.describe_endpoint(EndpointName=ENDPOINT_NAME)
    except sm_client.exceptions.ClientError:
        pass

    if existing:
        logger.info("Updating existing endpoint '%s'…", ENDPOINT_NAME)
        sm_client.update_endpoint(
            EndpointName=ENDPOINT_NAME,
            EndpointConfigName=config_name,
        )
    else:
        logger.info("Creating endpoint '%s'…", ENDPOINT_NAME)
        sm_client.create_endpoint(
            EndpointName=ENDPOINT_NAME,
            EndpointConfigName=config_name,
        )

    # Wait for endpoint to become InService
    logger.info("Waiting for endpoint to reach InService state (may take ~5 min)…")
    waiter = sm_client.get_waiter("endpoint_in_service")
    waiter.wait(EndpointName=ENDPOINT_NAME, WaiterConfig={"Delay": 30, "MaxAttempts": 20})
    logger.info("Endpoint '%s' is InService.", ENDPOINT_NAME)


def configure_autoscaling(sm_client) -> None:
    """Add Application Auto Scaling for the endpoint variant."""
    aas = boto3.client("application-autoscaling", region_name=AWS_REGION)
    resource_id = f"endpoint/{ENDPOINT_NAME}/variant/AllTraffic"

    aas.register_scalable_target(
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        MinCapacity=1,
        MaxCapacity=4,
    )
    aas.put_scaling_policy(
        PolicyName=f"{ENDPOINT_NAME}-scaling-policy",
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        PolicyType="TargetTrackingScaling",
        TargetTrackingScalingPolicyConfiguration={
            "TargetValue": 70.0,
            "PredefinedMetricSpecification": {
                "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance"
            },
            "ScaleInCooldown": 300,
            "ScaleOutCooldown": 60,
        },
    )
    logger.info("Auto Scaling configured for endpoint '%s'.", ENDPOINT_NAME)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    _check_env()

    session = boto3.Session(region_name=AWS_REGION)
    ecr = session.client("ecr")
    sm = session.client("sagemaker")

    model_name = f"fraud-classifier-{int(time.time())}"
    config_name = f"{model_name}-config"

    ensure_ecr_repo(ecr)
    image_uri = push_image(ecr)
    create_or_update_model(sm, model_name, image_uri)
    create_endpoint_config(sm, config_name, model_name)
    deploy_endpoint(sm, config_name)
    configure_autoscaling(sm)

    logger.info("Deployment complete. Endpoint: %s", ENDPOINT_NAME)


if __name__ == "__main__":
    main()
