#!/bin/bash

mkdir -p build
cd build
rm -rf *
cmake .. -DDOCKER_VERSION=ON
make -j12
make aws-lambda-package-svd_aws_bench && zip -ur svd_aws/svd_aws_bench.zip ../fmi.json
aws lambda update-function-code --function-name svd_aws_bench_func --zip-file fileb://svd_aws/svd_aws_bench.zip
