#!/bin/bash
set -x

nprocs=$1
nrows=$2
ncols=$3
trunc=$4
timestamp=$(date +%s)

mpirun -np $nprocs ./build/svd_mpi/svd_mpi $nrows $ncols $trunc output

python scripts/check_mpi_results.py $nprocs

rm output_*.txt output_*.mtx

