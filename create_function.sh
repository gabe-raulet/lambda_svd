#!/bin/bash

make aws-lambda-package-svd_aws_bench && zip -ur svd_aws/svd_aws_bench.zip ../fmi.json

aws lambda create-function \
--function-name  svd_aws_bench_func \
--role arn:aws:iam::183425780977:role/lambda-vpc-role \
--runtime provided.al2 \
--timeout 10 \
--memory-size 1024 \
--architectures arm64 \
--handler svd_aws_bench \
--zip-file fileb://build/svd_aws/svd_aws_bench.zip

