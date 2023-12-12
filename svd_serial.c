#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include "kiss.h"
#include "cblas.h"
#include "lapacke.h"

int log2i(int v);

int svds_naive(double *A, double *Up, double *Sp, double *Vpt, int m, int n, int p);
int seed_node(double *Ai, double *A1i, double *Vt1i, int m, int n, int q, int p);
int combine_node(double *Ak_2i_0, double *Vtk_2i_0, double *Ak_2i_1, double *Vtk_2i_1, double *Ak1_lj, double *Vtk1_lj, int m, int n, int k, int q, int p);
int extract_node(double *Aq1_11, double *Vtq1_11, double *Aq1_12, double *Vtq1_12, double *U, double *S, double *Vt, int m, int n, int q, int p);

int svd_serial
(
    double *A, /* input m-by-n matrix */
    double *Up, /* output m-by-p matrix */
    double *Sp, /* output p-by-p diagonal matrix */
    double *Vtp, /* output p-by-n matrix */
    int m, /* rows of A */
    int n, /* columns of A */
    int p, /* rank approximation */
    int b  /* number of seed nodes in binary topology */
);

int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        fprintf(stderr, "usage: %s <m:nrows> <n:ncols> <p:trunc> <b:nprocs>\n", argv[0]);
        return 1;
    }

    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int p = atoi(argv[3]);
    int b = atoi(argv[4]);

    int r = m < n? m : n;
    int s = n / b;

    if (!(m >= 1 && n >= 1) || (m&(m-1)) || (n&(n-1)))
    {
        fprintf(stderr, "[main::error::param_check][m=%d,n=%d] must have m,n >= 1 with m and n both powers of 2\n", m, n);
        return 1;
    }

    if ((b&(b-1)) || b <= 0)
    {
        fprintf(stderr, "[main::error::param_check][b=%d] must have b >= 1 with b being a power of 2\n", b);
        return 1;
    }

    if (!(p <= r && p <= s))
    {
        fprintf(stderr, "[main::error::param_check][p=%d,r=%d,s=%d] must have p <= r and p <= s\n", p, r, s);
        return 1;
    }

    return 0;
}

int log2i(int v)
{
    int x = 0;
    while (v >>= 1) ++x;
    return x;
}

int svds_naive(double *A, double *Up, double *Sp, double *Vpt, int m, int n, int p)
{
    int r = m < n? m : n;

    assert(A != NULL && Up != NULL && Sp != NULL && Vpt != NULL && r >= p && p >= 1);

    double *S, *U, *Vt, *work;

    work = malloc(5*r*sizeof(double));
    S = malloc(r*sizeof(double));
    U = malloc(m*r*sizeof(double));
    Vt = malloc(r*n*sizeof(double));

    LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'S', 'S', m, n, A, m, S, U, m, Vt, r, work);

    memcpy(Up, U, p*m*sizeof(double));
    memcpy(Sp, S, p*sizeof(double));

    double *Vt_ptr = Vt;
    double *Vpt_ptr = Vpt;

    for (int j = 0; j < r; ++j)
    {
        memcpy(Vpt_ptr, Vt_ptr, p*sizeof(double));

        Vpt_ptr += p;
        Vt_ptr += r;
    }

    free(S);
    free(U);
    free(Vt);

    return 0;
}

double* combine_routine(double *Ak_2i_0, double *Vtk_2i_0, double *Ak_2i_1, double *Vtk_2i_1, double *USki, int m, int n, int k, int q, int p)
{
    int b, s, d;
    double *Aki, *Vhtki, *Ski, *Vtki, *W, *tau;

    b = 1 << q;
    s = n / b;
    d = (1 << (k-1)) * s;

    Aki = malloc(m*(2*p)*sizeof(double));

    memcpy(&Aki[0],   Ak_2i_0, m*p*sizeof(double));
    memcpy(&Aki[m*p], Ak_2i_1, m*p*sizeof(double));

    Vhtki = calloc((2*p)*(2*d), sizeof(double));

    for (int j = 0; j < d; ++j)
    {
        memcpy(&Vhtki[j*(2*p)],       &Vtk_2i_0[j*p], p*sizeof(double));
        memcpy(&Vhtki[(j+d)*(2*p)+p], &Vtk_2i_1[j*p], p*sizeof(double));
    }

    Ski = malloc(p*sizeof(double));
    Vtki = malloc(p*(2*p)*sizeof(double));

    svds_naive(Aki, USki, Ski, Vtki, m, 2*p, p);

    for (int j = 0; j < p; ++j)
        for (int i = 0; i < m; ++i)
            USki[i + j*m] *= Ski[j];

    free(Ski);

    W = malloc((2*d)*p*sizeof(double));

    cblas_dgemm(CblasColMajor, CblasTrans, CblasTrans, 2*d, p, 2*p, 1.0, Vhtki, 2*p, Vtki, p, 0.0, W, 2*d);

    free(Vtki);
    free(Vhtki);

    assert(2*d >= p);
    tau = malloc(p*sizeof(double));

    LAPACKE_dgeqrf(LAPACK_COL_MAJOR, 2*d, p, W, 2*d, tau);
    LAPACKE_dtrtri(LAPACK_COL_MAJOR, 'U', 'N', p, W, 2*d);

    cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, m, p, 1.0, W, 2*d, USki, m);

    LAPACKE_dorgqr(LAPACK_COL_MAJOR, 2*d, p, p, W, 2*d, tau);

    free(tau);
    free(Aki);

    return W;
}

int combine_node(double *Ak_2i_0, double *Vtk_2i_0, double *Ak_2i_1, double *Vtk_2i_1, double *Ak1_lj, double *Vtk1_lj, int m, int n, int k, int q, int p)
{
    int b, s, d;
    double *W;

    b = 1 << q;
    s = n / b;
    d = (1 << (k-1)) * s;

    W = combine_routine(Ak_2i_0, Vtk_2i_0, Ak_2i_1, Vtk_2i_1, Ak1_lj, m, n, k, q, p);

    for (int j = 0; j < p; ++j)
        for (int i = 0; i < 2*d; ++i)
            Vtk1_lj[j + i*p] = W[i + j*2*d]; /* Vtk1_lj[j,i] = W[i,j]; W is 2d-by-p */

    free(W);

    return 0;
}

int seed_node(double *Ai, double *A1i, double *Vt1i, int m, int n, int q, int p)
{
    int b = 1 << q;
    int s = n / b;

    double *Sp = malloc(p*sizeof(double));

    svds_naive(Ai, A1i, Sp, Vt1i, m, s, p);

    for (int j = 0; j < p; ++j)
        for (int i = 0; i < m; ++i)
            A1i[i + m*j] *= Sp[j];

    free(Sp);
    return 0;
}

int extract_node(double *Aq1_11, double *Vtq1_11, double *Aq1_12, double *Vtq1_12, double *U, double *S, double *Vt, int m, int n, int q, int p)
{
    double *USq, *Qq, *Vtp;

    USq = malloc(m*p*sizeof(double));
    Vtp = malloc(p*n*sizeof(double));

    Qq = combine_routine(Aq1_11, Vtq1_11, Aq1_12, Vtq1_12, USq, m, n, q, q, p);

    svds_naive(USq, U, S, Vtp, m, p, p);

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, p, n, p, 1.0, Vtp, p, Qq, n, 0.0, Vt, p );

    free(USq);
    free(Qq);
    free(Vtp);

    return 0;
}

int svd_serial
(
    double *A, /* input m-by-n matrix */
    double *Up, /* output m-by-p matrix */
    double *Sp, /* output p-by-p diagonal matrix */
    double *Vtp, /* output p-by-n matrix */
    int m, /* rows of A */
    int n, /* columns of A */
    int p, /* rank approximation */
    int b  /* number of seed nodes in binary topology */
)
{
    int r = m < n? m : n;
    int s = n / b;

    assert(n % b == 0);
    assert(p <= r && p <= s);

    int q = log2i(b);

    double *Al, *Ai, *A1i, *Vt1i, *Acat, *Vtcat;

    Al = malloc(m*n*sizeof(double));
    memcpy(Al, A, m*n*sizeof(double));

    Acat = malloc(m*p*b*sizeof(double));
    Vtcat = malloc(p*s*b*sizeof(double));

    for (int i = 0; i < b; ++i)
    {
        Ai = &Al[i*m*s];
        A1i = &Acat[i*m*p];
        Vt1i = &Vtcat[i*p*s];

        seed_node(Ai, A1i, Vt1i, m, n, q, p);
    }

    double *Ak_2i_0, *Vtk_2i_0, *Ak_2i_1, *Vtk_2i_1, *Ak1_lj, *Vtk1_lj;

    for (int k = 1; k < q; ++k)
    {
        int c = 1 << (q-k); /* nodes on this level */
        int d = s * (1 << (k-1)); /* column count of incoming Vtk_2i_j matrices */

        for (int i = 0; i < c; ++i)
        {
            Ak_2i_0 = &Acat[(2*i)*m*p];
            Ak_2i_1 = &Acat[(2*i+1)*m*p];
            Vtk_2i_0 = &Vtcat[(2*i)*p*d];
            Vtk_2i_1 = &Vtcat[(2*i+1)*p*d];

            Ak1_lj = &Acat[i*m*p];
            Vtk1_lj = &Vtcat[(2*i)*p*d];

            combine_node(Ak_2i_0, Vtk_2i_0, Ak_2i_1, Vtk_2i_1, Ak1_lj, Vtk1_lj, m, n, k, q, p);
        }
    }

    double *Aq1_11, *Aq1_12, *Vtq1_11, *Vtq1_12;

    Aq1_11 = &Acat[0];
    Aq1_12 = &Acat[m*p];
    Vtq1_11 = &Vtcat[0];
    Vtq1_12 = &Vtcat[(n*p)>>1];

    extract_node(Aq1_11, Vtq1_11, Aq1_12, Vtq1_12, Up, Sp, Vtp, m, n, q, p);

    free(Acat);
    free(Vtcat);
    free(Al);

    return 0;
}
