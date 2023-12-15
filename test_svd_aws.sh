#!/bin/bash
set -x

#nprocs=$1
#nrows=$2
#ncols=$3
#trunc=$4
timestamp=$(date +%s)

aws lambda invoke --cli-read-timeout 600 --function-name svd_aws_bench_func --cli-binary-format raw-in-base64-out \
    --payload '{"timestamp": "'$timestamp'", "nprocs": '2', "myrank":'0', "matrank":'128', "trunc":'10',"seed":'1', "mattype":"tall"}' "out1.json" &

aws lambda invoke --cli-read-timeout 600 --function-name svd_aws_bench_func --cli-binary-format raw-in-base64-out \
    --payload '{"timestamp": "'$timestamp'", "nprocs": '2', "myrank":'1', "matrank":'128', "trunc":'10',"seed":'2', "mattype":"tall"}' "out2.json" &


