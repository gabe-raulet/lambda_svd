import sys
import numpy as np
import subprocess as sp
import time

# rowcnts = [256]
# colcnts = [128]
# pvals = [10]
# bvals = [8]

rowcnts = [128, 256, 512]
colcnts = [128, 256, 512]
pvals = [2, 5, 10]
bvals = [4, 8]

for m in rowcnts:
    for n in colcnts:
        for p in pvals:
            for nprocs in bvals:
                failed = False
                t1 = -time.perf_counter()
                tcpunch = sp.Popen(["fmi/extern/TCPunch/server/build/tcpunchd"], stdout=sp.PIPE, stderr=sp.PIPE)
                procs = [None]*nprocs
                for rank in range(nprocs):
                    cmd = f"./build/svd_fmi/svd_fmi {rank} {nprocs} {m} {n} {p}"
                    procs[rank] = sp.Popen(cmd.split(), stdout=sp.PIPE, stderr=sp.PIPE)
                for rank in range(nprocs):
                    out, err = procs[rank].communicate()
                    if procs[rank].returncode != 0:
                        failed = True
                        print(err.decode("utf-8"))
                        for i in range(rank+1, nprocs):
                            procs[rank].terminate()
                        break
                tcpunch.terminate()
                t1 += time.perf_counter()
                if not failed:
                    sys.stdout.write(f"[nprocs={nprocs},m={m},n={n},p={p}] finished in {t1:.5f} seconds\n")
                    sys.stdout.flush()
                else:
                    sys.stdout.write(f"[nprocs={nprocs},m={m},n={n},p={p}] failed\n")
                    sys.stdout.flush()
