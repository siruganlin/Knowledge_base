import json

from minio import Minio

from config.minio_config import minio_config

try:
    minio_client = Minio(
        endpoint=minio_config.endpoint,
        access_key=minio_config.access_key,
        secret_key=minio_config.secret_key,
        secure=False
    )

    # exists = minio_client.bucket_exists('a-bucket')
    # print(exists)
    if not minio_client.bucket_exists(minio_config.bucket_name):
        minio_client.make_bucket(minio_config.bucket_name)

    # 新建的桶是私有的不能直接使用需要修改一下，用到下面的策略语言
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"AWS": ["*"]},
                "Action": ["s3:GetObject"],
                "Resource": [f"arn:aws:s3:::{minio_config.bucket_name}/*"]
            }
        ]
    }
    minio_client.set_bucket_policy(minio_config.bucket_name, json.dumps(policy))

except Exception as e:
    print(e)
    minio_client=None

# 这里声明一个方法将来给外面去调用minio的客户端
def get_minio_client():
    return minio_client
