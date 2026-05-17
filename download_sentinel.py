import boto3

s3 = boto3.client("s3")

bucket = "sentinel-s2-l2a"
key = "tiles/36/R/UJ/2024/5/1/0/R10m/B02.jp2"

print("Downloading...")

s3.download_file(bucket, key, "B02.jp2")

print("Done!")