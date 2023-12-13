import numpy as np
from scipy.io import mmread, mmwrite

def svds(A, p):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    return U[:,:p], S[:p], Vt[:p,:]

def seed_node(Ai, m, n, q, p):
    b = 1 << q
    s = n // b
    assert Ai.shape == (m, s)

    U1i, S1i, Vt1i = svds(Ai, p)
    assert U1i.shape == (m, p) and S1i.shape == (p,) and Vt1i.shape == (p, s)

    A1i = U1i@np.diag(S1i)
    return A1i, Vt1i

def combine_node(Ak_2i_0, Vtk_2i_0, Ak_2i_1, Vtk_2i_1, m, n, k, q, p):
    b = 1 << q
    s = n // b
    d = (1 << (k-1)) * s

    assert Ak_2i_0.shape == (m, p) and Ak_2i_1.shape == (m, p)
    assert Vtk_2i_0.shape == (p, d) and Vtk_2i_1.shape == (p, d)

    Aki = np.hstack((Ak_2i_0, Ak_2i_1))
    assert Aki.shape == (m, 2*p)

    Uki, Ski, Vtki = svds(Aki, p)
    assert Uki.shape == (m, p) and Ski.shape == (p,) and Vtki.shape == (p, 2*p)

    USki = Uki@np.diag(Ski)
    assert USki.shape == (m, p)

    Vhtki = np.vstack((np.hstack((Vtk_2i_0, np.zeros((p,d)))), np.hstack((np.zeros((p,d)), Vtk_2i_1))))
    assert Vhtki.shape == (2*p, 2*d)

    W = Vhtki.T@Vtki.T
    assert W.shape == (2*d, p)
    assert 2*d >= p

    Qki, Rki = np.linalg.qr(W)
    assert Qki.shape == (2*d, p) and Rki.shape == (p, p)

    Ak1_lj = USki @ np.linalg.inv(Rki)
    assert Ak1_lj.shape == (m, p)

    Vtk1_lj = Qki.T
    assert Vtk1_lj.shape == (p, 2*d)

    return Ak1_lj, Vtk1_lj

def extract_node(Aq1_11, Vtq1_11, Aq1_12, Vtq1_12, m, n, q, p):
    Ah, Qtq = combine_node(Aq1_11, Vtq1_11, Aq1_12, Vtq1_12, m, n, q, q, p)
    Uq, Sq, Vtq = svds(Ah, p)
    Vtq = Vtq@Qtq
    return Uq, Sq, Vtq

def svd_serial(A, p, q):
    m,n = A.shape
    b = 1 << q
    s = n // b
    r = min(m, n)
    assert n % b == 0 and p <= r and p <= s

    Amem = {}
    Vtmem = {}

    for i in range(b):
        Ai = A[:,s*i:s*(i+1)]
        A1i, Vt1i = seed_node(Ai, m, n, q, p)
        Amem[i] = A1i
        Vtmem[i] = Vt1i

    for k in range(1, q):
        c = 1 << (q-k)
        d = s * (1 << (k-1))
        for i in range(c):
            Ak_2i_0 = Amem[2*i]
            Ak_2i_1 = Amem[2*i+1]
            Vtk_2i_0 = Vtmem[2*i]
            Vtk_2i_1 = Vtmem[2*i+1]
            Ak1_lj, Vtk1_lj = combine_node(Ak_2i_0, Vtk_2i_0, Ak_2i_1, Vtk_2i_1, m, n, k, q, p)
            Amem[i] = Ak1_lj
            Vtmem[i] = Vtk1_lj

    Aq1_11 = Amem[0]
    Aq1_12 = Amem[1]
    Vtq1_11 = Vtmem[0]
    Vtq1_12 = Vtmem[1]

    Uq, Sq, Vtq = extract_node(Aq1_11, Vtq1_11, Aq1_12, Vtq1_12, m, n, q, p)

    return Uq, Sq, Vtq

def damped_matrix(m, n, r, alpha):
    Q = np.random.random((m,m))
    U, R = np.linalg.qr(Q)

    Q = np.random.random((n,n))
    V, R = np.linalg.qr(Q)

    Vt = V.T

    S = np.zeros((m,n))

    S[0,0] = 100
    for i in range(1, r):
        S[i,i] = S[i-1,i-1] / alpha

    A = U@S@Vt
    return A, U, np.diag(S), Vt

def regular_matrix(m, n):
    A = np.random.random((m,n))
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    return A, U, S, Vt

if __name__ == "__main__":
    m = 256
    n = 128
    p = 5
    q = 3
    r = min(m,n)
    alpha = 1.1

    # A, U, S, V = regular_matrix(m, n)
    A, U, S, Vt = damped_matrix(m, n, r, alpha)

    Uq, Sq, Vtq = svd_serial(A, p, q)
    Aq = Uq@np.diag(Sq)@Vtq

    mmwrite("fang_A.mtx", A)
    mmwrite("fang_Uq.mtx", Uq)
    mmwrite("fang_Vtq.mtx", Vtq)

    with open("fang_Sq.txt", "w") as f:
        for i in range(p):
            f.write(f"{Sq[i]:.18e}\n")

    Up = U[:,:p]
    Sp = S[:p]
    Vtp = Vt[:p,:]
    Ap = Up@np.diag(Sp)@Vtp

    print(f"Aerr = {np.linalg.norm(Ap - Aq):.10e}")
    print(f"Serr = {np.linalg.norm(Sp - Sq):.10e}")
    print(f"Uerr = {np.linalg.norm(Up@Up.T - Uq@Uq.T):.10e}")
    print(f"Verr = {np.linalg.norm(Vtp.T@Vtp - Vtq.T@Vtq):.10e}")

    for i in range(p):
        print(f"S[{i}]_abs_err = {np.abs(Sq[i] - Sp[i]):.10e}")
        print(f"S[{i}]_rel_err = {np.abs((Sq[i] - Sp[i]) / Sp[i]):.10e}")

