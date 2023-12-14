import numpy as np
import subprocess as sp
from pathlib import Path
import pandas as pd
import shutil
import glob
import json
from scipy.io import mmread, mmwrite

def parse_test_output(output):
    lines = output.rstrip().split("\n")
    assert len(lines) == 3
    assert lines[0].startswith("[params]")
    assert lines[1].startswith("[init]")
    assert lines[2].startswith("[dist_mpi_tree]")
    params = json.loads(lines[0].split("[params]")[1])
    init = json.loads(lines[1].split("[init]")[1])
    dist_mpi_tree = json.loads(lines[2].split("[dist_mpi_tree]")[1])
    return params | dist_mpi_tree

def run_test(p, r, mscale, nprocs, mattype):
    cmd = ["mpirun", "-np", str(nprocs), "./build/svd_mpi/svd_mpi_bench", "random", str(p), str(r), str(mscale), mattype]
    proc = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
    out, err = proc.communicate()
    return err.decode("utf-8")

output = run_test(10, 256, 2, 8, "tall")

if __name__ == "__main__":

    p = 10
    proc_counts = [2,4,8]
    # ranks = [256]
    ranks = [512,1024]
    # mscales = [1,2]
    mscales = [1,2,4]

    runs = []

    for r in ranks:
        for mscale in mscales:
            for nprocs in proc_counts:
                mattypes = ["tall"] if mscale == 1 else ["tall", "wide"]
                for mattype in mattypes:
                    output = run_test(p, r, mscale, nprocs, mattype)
                    runs.append(parse_test_output(output))

    table = pd.DataFrame(runs)
    table.to_csv("test.tsv", sep="\t", index=False)

