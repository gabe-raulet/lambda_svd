import sys
import numpy as np
from scipy.io import mmread
from fang_svd_redux import svd_serial

def vecread(fname):
    return np.array([np.double(line.rstrip()) for line in open(fname, "r")])

nprocs = int(sys.argv[1])

A = np.hstack([mmread(f"A_{i+1}_{nprocs}.mtx") for i in range(nprocs)])
print(A.shape)

Stest = vecread("Sh.txt")
Utest = mmread("Uh.mtx")
Vttest = mmread("Vth.mtx")

q = int(np.log(nprocs) / np.log(2))
p = len(Stest)

Ufang, Sfang, Vtfang = svd_serial(A, p, q)

print(f"Uerr = {np.linalg.norm(Utest - Ufang):.10e}")
print(f"Verr = {np.linalg.norm(Vttest - Vtfang):.10e}")
print(f"Serr = {np.linalg.norm(Stest - Sfang):.10e}")
