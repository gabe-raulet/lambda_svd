import sys
import numpy as np
from scipy.io import mmwrite, mmread

if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.stderr.write(f"Usage: {sys.argv[0]} <A.mtx> <splits> <oprefix>\n")
        sys.stderr.flush()
        sys.exit(1)

    mtxfname = sys.argv[1]
    splits = int(sys.argv[2])
    oprefix = sys.argv[3]

    A = mmread(mtxfname)

    m, n = A.shape
    s = n // splits

    for i in range(splits):
        Ai = A[:,i*s:(i+1)*s]
        mmwrite(f"{oprefix}_{i+1}_{splits}.mtx", Ai)
