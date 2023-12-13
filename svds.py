import numpy as np

def random_matrix(m, n, S):
    r = len(S)
    U, _ = np.linalg.qr(np.random.random((m,m)))
    V, _ = np.linalg.qr(np.random.random((n,n)))
    A = U[:,:r]@np.diag(S)@V.T[:r,:]
    return A, U, S, V

def damped_matrix(m, n, alpha, cond):
    U, _ = np.linalg.qr(np.random.random((m,m)))
    V, _ = np.linalg.qr(np.random.random((n,n)))
    S = np.zeros((m,n))
    S[0,0] = cond
    for i in range(1, min(m,n)):
        S[i,i] = S[i-1,i-1] / alpha
    A = U@S@V.T
    return A, U, np.array([S[i,i] for i in range(min(m,n))]), V

def svds(A, p):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    return U[:,:p], S[:p], Vt.T[:,:p]

def seed(AQ_in, Q_in, p):
    _, _, P = svds(AQ_in, p)
    Q_out = Q_in@P
    AQ_out = AQ_in@P
    return AQ_out, Q_out

def combine(AQ_in, Q_in, p):
    U, S, V = svds(AQ_in, p)
    Q_out, R = np.linalg.qr(Q_in@V)
    AQ_out = AQ_in@V@np.linalg.inv(R)
    return AQ_out, Q_out

def extract(AQ_in, Q_in):
    U, S, P = svds(AQ_in, p)
    V = Q_in@P
    AV = AQ_in@P
    return U, S, V, AV

m = 512
n = 256
p = 2

# S = np.array([2.]*min(m>>1,n>>1) + [1.]*min(m>>1,n>>1))
# A, U, S, V = random_matrix(m, n, S)

A, U, S, V = damped_matrix(m, n, 1.5, 50)

Up = U[:,:p]
Sp = S[:p]
Vp = V[:,:p]


AQ, Q = seed(A, np.eye(n), p)
AQ, Q = combine(AQ, Q, p)
Uh, Sh, Vh, AVh = extract(AQ, Q)
Ah = Uh@np.diag(Sh)@Vh.T
Ap = Up@np.diag(Sp)@Vp.T

print(f"Aerr = {np.linalg.norm(Ap-Ah):.10e}")
print(f"Serr = {np.linalg.norm(Sp-Sh):.10e}")
print(f"Uerr = {np.linalg.norm(Up@Up.T - Uh@Uh.T):.10e}")
print(f"Verr = {np.linalg.norm(Vp@Vp.T - Vh@Vh.T):.10e}")
print("\n")

# Q, _ = np.linalg.qr(A.T@Uh)
# AQ = A@Q
AQ, Q = AVh, Vh

AQ, Q = combine(AQ, Q, p)
Uh, Sh, Vh, AVh = extract(AQ, Q)
Ah = Uh@np.diag(Sh)@Vh.T
Ap = Up@np.diag(Sp)@Vp.T

print(f"Aerr = {np.linalg.norm(Ap-Ah):.10e}")
print(f"Serr = {np.linalg.norm(Sp-Sh):.10e}")
print(f"Uerr = {np.linalg.norm(Up@Up.T - Uh@Uh.T):.10e}")
print(f"Verr = {np.linalg.norm(Vp@Vp.T - Vh@Vh.T):.10e}")
