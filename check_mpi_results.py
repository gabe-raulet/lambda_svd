import sys
import numpy as np
from scipy.io import mmread
from fang_svd_redux import svd_serial

def vecread(fname):
    return np.array([np.double(line.rstrip()) for line in open(fname, "r")])

nprocs = int(sys.argv[1])

A = mmread(f"output_A.mtx")
print(A.shape)

Stest = vecread("output_Sp.txt")
Utest = mmread("output_Up.mtx")
Vttest = mmread("output_Vtp.mtx")

q = int(np.log(nprocs) / np.log(2))
p = len(Stest)

Ufang, Sfang, Vtfang = svd_serial(A, p, q)

print(f"Uerr = {np.linalg.norm(Utest - Ufang):.10e}")
print(f"Verr = {np.linalg.norm(Vttest - Vtfang):.10e}")
print(f"Serr = {np.linalg.norm(Stest - Sfang):.10e}")
