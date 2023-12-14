#!/bin/bash
set -x

#nprocs=$1
#nrows=$2
#ncols=$3
#trunc=$4
timestamp=$(date +%s)

aws lambda invoke --cli-read-timeout 600 --function-name svd_aws_test_func2 --cli-binary-format raw-in-base64-out \
    --payload '{"timestamp": "'$timestamp'", "nprocs": '2', "myrank":'0', "number": '23'}' "out1.json" &

aws lambda invoke --cli-read-timeout 600 --function-name svd_aws_test_func2 --cli-binary-format raw-in-base64-out \
    --payload '{"timestamp": "'$timestamp'", "nprocs": '2', "myrank":'1', "number": '1245'}' "out2.json" &


