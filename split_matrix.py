import sys
import numpy as np
from scipy.io import mmwrite, mmread

def split_matrix(A, splits):
    m, n = A.shape
    s = n // splits
    for i in range(splits):
        yield A[:,i*s:(i+1)*s]

if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.stderr.write(f"Usage: {sys.argv[0]} <A.mtx> <splits> <oprefix>\n")
        sys.stderr.flush()
        sys.exit(1)

    mtxfname = sys.argv[1]
    splits = int(sys.argv[2])
    oprefix = sys.argv[3]

    A = mmread(mtxfname)

    for i, Ai in enumerate(split_matrix(A, splits))
        mmwrite(f"{oprefix}_{i+1}_{splits}.mtx", Ai)
