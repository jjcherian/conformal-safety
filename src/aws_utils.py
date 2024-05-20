import boto3
import io

def s3_open(bucket_name, key):
    # Create a session using your AWS credentials
    session = boto3.Session()
    # Create an S3 client
    s3 = session.client('s3')

    # Download the file object
    response = s3.get_object(Bucket=bucket_name, Key=key)
    file_content = response['Body'].read()

    # Return a BytesIO object to mimic a file object
    return io.BytesIO(file_content)