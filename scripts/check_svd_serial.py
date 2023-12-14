import sys
import numpy as np
from scipy.io import mmread

def vecread(fname):
    return np.array([np.double(line.rstrip()) for line in open(fname)])

def main(oprefix):
    A = mmread(f"{oprefix}_A.mtx")
    Up = mmread(f"{oprefix}_Up.mtx")
    Vtp = mmread(f"{oprefix}_Vtp.mtx")
    Sp = vecread(f"{oprefix}_Sp.txt")

    p = Up.shape[1]

    assert p == Vtp.shape[0] and p == len(Sp)

    Ap = Up@np.diag(Sp)@Vtp
    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    Up2 = U[:,:p]
    Sp2 = S[:p]
    Vtp2 = Vt[:p,:]

    Ap2 = Up2@np.diag(Sp2)@Vtp2

    print(f"Aerr = {np.linalg.norm(Ap-Ap2):.10e}")
    print(f"Serr = {np.linalg.norm(Sp-Sp2):.10e}")
    print(f"Uerr = {np.linalg.norm(Up@Up.T - Up2@Up2.T):.10e}")
    print(f"Verr = {np.linalg.norm(Vtp.T@Vtp - Vtp2.T@Vtp2):.10e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write(f"Usage: {sys.argv[0]} <outprefix>\n")
        sys.stderr.flush()
        sys.exit(1)
    main(sys.argv[1])
