"""
Please fill in your credentials here.
"""
import boto3, os

ISENGARD_PRODUCTION_ACCOUNT="ISENGARD_PRODUCTION_ACCOUNT"
AWS_ACCESS_KEY_ID="AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY="AWS_SECRET_ACCESS_KEY"
AWS_SESSION_TOKEN="AWS_SESSION_TOKEN"


def setup():
    os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
    os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
    os.environ["AWS_SESSION_TOKEN"] = AWS_SESSION_TOKEN
    os.environ["ISENGARD_PRODUCTION_ACCOUNT"] = ISENGARD_PRODUCTION_ACCOUNT

    # Create a low-level client representing Amazon SageMaker Runtime, <DO NOT Change>
    role_arn = "arn:aws:iam::339569644176:role/AllowTranslateScienceInvokeSageMakerEndpoints"
    # role_arn = "arn:aws:sts::324012739285:assumed-role/PowerUser/yuwzhan-Isengard"
    
    # replace the RoleSessionName placeholder with your own alias and project name
    credentials = boto3.client("sts").assume_role(
        RoleArn=role_arn,
        RoleSessionName="yuwzhan_intern",
    )["Credentials"]

    sagemaker_runtime = boto3.client(
        "sagemaker-runtime",
        region_name='us-east-1',
        aws_access_key_id=credentials["AccessKeyId"],
        aws_secret_access_key=credentials["SecretAccessKey"],
        aws_session_token=credentials["SessionToken"],
    )

    return sagemaker_runtime

MODEL2ENDPOINTS = {
        "mpt-30b-chat": "mpt-30b-chat-ml-g5-48xlarge-2023-07-21-00-13-13-634",
        "mpt-7b-instruct": "mpt-7b-instruct-ml-g5-2xlarge-2023-06-27-21-55-10-527",
        "mpt-7b-chat": "mpt-7b-chat-ml-g5-2xlarge-2023-07-16-22-01-23-092",
        "open-llama-13b-open-instruct":"open-llama-13b-open-instruct-ml-g5-12xl-2023-06-29-02-11-10-956",
        "falcon-40b-instruct":"falcon-40b-instruct-ml-g5-48xlarge-2023-08-08-01-29-21-845",
        "falcon-7b-instruct": "falcon-7b-instruct-ml-g5-2xlarge-2023-06-16-23-26-45-967",
        "koala-13b": "koala-13B-HF-ml-g5-12xlarge-2023-06-07-15-33-59-410",
        "vicuna-13b": "vicuna-13b-hf-ml-g5-12xlarge-2023-06-07-16-25-06-065",
        "llama-13b": "llama-13b-ml-g5-12xlarge-2023-06-06-22-37-52-080",
        "openllama": "openllama-ml-g5-4xlarge-2023-05-19-16-44-58-937",
        "redpajama-instruct-7b": "RedPajama-instruct-7b-g5-2xlarge-2023-05-15-18-02-20-451",
        "redpajama-instruct-3b": "RedPajama-instruct-3b-g5-2xlarge-2023-05-15-17-15-00-587",
        "mpt-7b": "mpt-7b-g5-12xlarge-2023-05-11-21-41-56-398",
        "santacoder": "santacoder-ml-g5-2xlarge-2023-06-29-19-29-50-616"
}