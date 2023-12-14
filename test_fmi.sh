#!/bin/bash
set -x

nprocs=$1
nrows=$2
ncols=$3
trunc=$4
timestamp=$(date +%s)

./fmi/extern/TCPunch/server/build/tcpunchd 5000 > /dev/null &
./build/svd_fmi/svd_fmi 0 $nprocs $nrows $ncols $trunc $timestamp &

rank_pid=$!

for i in $(seq 2 $nprocs);
do
     myrank=$(($i - 1))
     ./build/svd_fmi/svd_fmi $myrank $nprocs $nrows $ncols $trunc $timestamp &
done

wait $rank_pid

pkill -f tcpunchd
pkill -f svd_fmi

python check_fmi_results.py $nprocs

rm Uh.mtx Sh.txt Vth.mtx
rm A_*.mtx

