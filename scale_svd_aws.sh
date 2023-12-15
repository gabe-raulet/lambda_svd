#!/bin/bash
set -x

nprocs=$1
matrank=$2
trunc=$3
mattype=$4
timestamp=$(date +%s)

for i in $(seq 1 $nprocs);
do
     myrank=$(($i - 1))
	aws lambda invoke --cli-read-timeout 600 --function-name svd_aws_bench_func --cli-binary-format raw-in-base64-out \
	    --payload '{"timestamp": '"$timestamp"', "nprocs": '"$nprocs"', "myrank": '"$myrank"', "matrank":'"$matrank"', "trunc":'"$trunc"',"seed":'"$myrank"', "mattype":'\""$mattype"\"'}' "out$i.json" &
done


