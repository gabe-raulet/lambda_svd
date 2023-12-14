import sys
import numpy as np
from scipy.io import mmwrite

def damped_matrix(m, n, cond, alpha):
    Q = np.random.random((m,m))
    U, R = np.linalg.qr(Q)

    Q = np.random.random((n,n))
    Vt, R = np.linalg.qr(Q)

    S = np.zeros((m,n))

    r = min(m, n)

    S[0,0] = cond
    for i in range(1, r):
        S[i,i] = S[i-1,i-1] / alpha

    A = U@S@Vt
    return A, U, np.diag(S), Vt

if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.stderr.write(f"Usage: {sys.argv[0]} <m> <n> <cond> <alpha> <oprefix>\n")
        sys.stderr.flush()
        sys.exit(1)

    m = int(sys.argv[1])
    n = int(sys.argv[2])
    cond = float(sys.argv[3])
    alpha = float(sys.argv[4])
    oprefix = sys.argv[5]

    A, U, S, Vt = damped_matrix(m, n, cond, alpha)

    mmwrite(f"{oprefix}_A.mtx", A)
    mmwrite(f"{oprefix}_U.mtx", U)
    mmwrite(f"{oprefix}_Vt.mtx", Vt)

    with open(f"{oprefix}_S.txt", "w") as f:
        for i in range(min(m,n)): f.write(f"{S[i]:.18e}\n")

