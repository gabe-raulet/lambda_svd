import numpy as np
from scipy.io import mmwrite

def combination_node(At1, At2, Vt1, Vt2, p):
    At = np.concatenate((At1, At2), axis=1)
    Vtt1 = np.concatenate((Vt1, np.zeros(Vt1.shape)), axis=1)
    Vtt2 = np.concatenate((np.zeros(Vt2.shape), Vt2), axis=1)
    Vtt = np.concatenate((Vtt1, Vtt2))
    Ut, St, Vt = np.linalg.svd(At, full_matrices=False)
    Ut = Ut[:,:p]
    St = St[:p]
    Vt = Vt.T[:,:p]
    Qt, Rt = np.linalg.qr(Vtt@Vt)
    AV = Ut@np.diag(St)@np.linalg.inv(Rt)
    V = Qt
    return AV, V

def extraction_node(At1, At2, Vt1, Vt2, p):
    AV, V = combination_node(At1, At2, Vt1, Vt2, p)
    U, S, Vbar = np.linalg.svd(AV, full_matrices=False)
    V = V@Vbar.T
    S = np.diag(S)
    return U, S, V

def seed_node(At, p):
    Ut, St, Vt = np.linalg.svd(At, full_matrices=False)
    Ut = Ut[:,:p]
    St = St[:p]
    Vt = Vt.T[:,:p]
    Aseed = Ut@np.diag(St)
    Vseed = Vt
    return Aseed, Vseed

def computeSVD(A, nlevel, m, n, p):
    num_seed_nodes = 2 ** (nlevel-1)
    Adict, Vdict = {}, {}
    for i in range(num_seed_nodes):
        Aseed, Vseed = seed_node(A[:,i*n//num_seed_nodes:(i+1)*n//num_seed_nodes], p)
        Adict[i] = Aseed
        Vdict[i] = Vseed

    for i in range(nlevel - 2):
        Adictnew = Adict
        Vdictnew = Vdict
        for j in range(2**(nlevel-2-i)):
            At1 = Adictnew[2*j]
            Vt1 = Vdictnew[2*j]
            At2 = Adictnew[2*j+1]
            Vt2 = Vdictnew[2*j+1]
            Acomb, Vcomb = combination_node(At1, At2, Vt1, Vt2, p)
            Adict[j] = Acomb
            Vdict[j] = Vcomb

        At1 = Adict[0]
        Vt1 = Vdict[0]
        At2 = Adict[1]
        Vt2 = Vdict[1]

        U, S, V = extraction_node(At1, At2, Vt1, Vt2, p)

    return U, S, V

def damped_matrix(m, n, r, alpha):
    Q = np.random.random((m,m))
    U, R = np.linalg.qr(Q)

    Q = np.random.random((n,n))
    V, R = np.linalg.qr(Q)

    S = np.zeros((m,n))
    S[0,0] = 100
    for i in range(1, r):
        S[i,i] = S[i-1,i-1] / alpha

    A = U@S@V.T
    return A, U, S, V

def regular_matrix(m, n):
    A = np.random.random((m,n))
    U, S, Vt = np.linalg.svd(A, full_matrices=True)
    return A, U, np.diag(S), Vt.T

if __name__ == "__main__":
    m = 1000
    n = 1024
    nlevel = 4
    p = 5
    r = 1000
    alpha = 5

    # A, U, S, V = regular_matrix(m, n)
    A, U, S, V = damped_matrix(m, n, r, alpha)

    Uout, Sout, Vout = computeSVD(A, nlevel, m, n, p)
    Aout = Uout@Sout@Vout.T

    mmwrite("fang_A.mtx", A)
    mmwrite("fang_Up.mtx", Uout)
    mmwrite("fang_Vtp.mtx", Vout.T)

    with open("fang_Sp.txt", "w") as f:
        for i in range(p):
            f.write(f"{Sout[i,i]:.18e}\n")

    Up = U[:,:p]
    Vp = V[:,:p]
    Sp = S[:p,:p]
    Ap = Up@Sp@Vp.T

    print(f"Aerr = {np.linalg.norm(Ap - Aout):.10e}")
    print(f"Serr = {np.linalg.norm(Sp - Sout):.10e}")
    print(f"Uerr = {np.linalg.norm(Up@Up.T - Uout@Uout.T):.10e}")
    print(f"Verr = {np.linalg.norm(Vp@Vp.T - Vout@Vout.T):.10e}")

    for i in range(p):
        print(f"S[{i}]_rel_err = {np.abs(Sout[i,i] - Sp[i,i]):.10e}")
        # print(f"S[{i}]_rel_err = {np.abs((Sout[i,i] - Sp[i,i]) / Sp[i,i]):.10e}")

