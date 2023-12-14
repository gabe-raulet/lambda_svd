import sys
import numpy as np
from scipy.io import mmwrite, mmread

if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.stderr.write(f"Usage: {sys.argv[0]} <iprefix> <splits> <A.mtx>\n")
        sys.stderr.flush()
        sys.exit(1)

    iprefix = sys.argv[1]
    splits = int(sys.argv[2])
    mtxfname = sys.argv[3]

    A = np.hstack([mmread(f"{iprefix}_{i+1}_{splits}.mtx") for i in range(splits)])
    mmwrite(mtxfname, A)
