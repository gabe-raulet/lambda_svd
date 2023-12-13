#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include "kiss.h"
#include "mmiodense.h"
#include "cblas.h"
#include "lapacke.h"

int log2i(int v);
int param_check(int m, int n, int p, int b, int r, int s);

int svds_naive(double *A, double *Up, double *Sp, double *Vpt, int m, int n, int p);
int seed_node(double *Ai, double *A1i, double *Vt1i, int m, int n, int q, int p);
int combine_node(double *Ak_2i_0, double *Vtk_2i_0, double *Ak_2i_1, double *Vtk_2i_1, double *Ak1_lj, double *Vtk1_lj, int m, int n, int k, int q, int p);
int extract_node(double *Aq1_11, double *Vtq1_11, double *Aq1_12, double *Vtq1_12, double *U, double *S, double *Vt, int m, int n, int q, int p);
int get_file_handlers(FILE **handlers, char const *oprefix);

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

    if (argc != 6)
    {
        fprintf(stderr, "usage: %s <m:nrows> <n:ncols> <p:trunc> <b:nprocs> <oprefix>\n", argv[0]);
        return 1;
    }

    kiss_init();

    int m = atoi(argv[1]), n = atoi(argv[2]), p = atoi(argv[3]), b = atoi(argv[4]);
    int r = m < n? m : n;
    int s = n / b;

    if (param_check(m, n, p, b, r, s) != 0)
        return 1;

    char const *oprefix = argv[5];

    double *A, *Up, *Sp, *Vtp;

    A = (double*)malloc(m*n*sizeof(double));
    Up = (double*)malloc(m*p*sizeof(double));
    Sp = (double*)malloc(p*sizeof(double));
    Vtp = (double*)malloc(p*n*sizeof(double));

    /*
     * Generate random matrix A.
     */

    for (int i = 0; i < m*n; ++i)
        A[i] = kiss_unirandf();

    svd_serial(A, Up, Sp, Vtp, m, n, p, b);

    FILE *handlers[4];

    get_file_handlers(handlers, oprefix);

    FILE *Afh = handlers[0], *Ufh = handlers[1], *Sfh = handlers[2], *Vfh = handlers[3];

    mmio_write_dense(Afh, A, m, n);
    mmio_write_dense(Ufh, Up, m, p);
    mmio_write_dense(Vfh, Vtp, p, n);

    for (int i = 0; i < p; ++i)
    {
        fprintf(Sfh, "%.18e\n", Sp[i]);
    }

    for (int i = 0; i < 4; ++i)
        fclose(handlers[i]);

    free(A);
    free(Up);
    free(Sp);
    free(Vtp);

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

    work = (double*)malloc(5*r*sizeof(double));
    S = (double*)malloc(r*sizeof(double));
    U = (double*)malloc(m*r*sizeof(double));
    Vt = (double*)malloc(r*n*sizeof(double));

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

    Aki = (double*)malloc(m*(2*p)*sizeof(double));

    memcpy(&Aki[0],   Ak_2i_0, m*p*sizeof(double));
    memcpy(&Aki[m*p], Ak_2i_1, m*p*sizeof(double));

    Vhtki = (double*)calloc((2*p)*(2*d), sizeof(double));

    for (int j = 0; j < d; ++j)
    {
        memcpy(&Vhtki[j*(2*p)],       &Vtk_2i_0[j*p], p*sizeof(double));
        memcpy(&Vhtki[(j+d)*(2*p)+p], &Vtk_2i_1[j*p], p*sizeof(double));
    }

    Ski = (double*)malloc(p*sizeof(double));
    Vtki = (double*)malloc(p*(2*p)*sizeof(double));

    svds_naive(Aki, USki, Ski, Vtki, m, 2*p, p);

    for (int j = 0; j < p; ++j)
        for (int i = 0; i < m; ++i)
            USki[i + j*m] *= Ski[j];

    free(Ski);

    W = (double*)malloc((2*d)*p*sizeof(double));

    cblas_dgemm(CblasColMajor, CblasTrans, CblasTrans, 2*d, p, 2*p, 1.0, Vhtki, 2*p, Vtki, p, 0.0, W, 2*d);

    free(Vtki);
    free(Vhtki);

    assert(2*d >= p);
    tau = (double*)malloc(p*sizeof(double));

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

    double *Sp = (double*)malloc(p*sizeof(double));

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

    USq = (double*)malloc(m*p*sizeof(double));
    Vtp = (double*)malloc(p*n*sizeof(double));

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

    Al = (double*)malloc(m*n*sizeof(double));
    memcpy(Al, A, m*n*sizeof(double));

    Acat = (double*)malloc(m*p*b*sizeof(double));
    Vtcat = (double*)malloc(p*s*b*sizeof(double));

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


int param_check(int m, int n, int p, int b, int r, int s)
{
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

int get_file_handlers(FILE **handlers, char const *oprefix)
{
    char fname[1024];

    snprintf(fname, 1024, "%s_A.mtx", oprefix);
    handlers[0] = fopen(fname, "w");

    snprintf(fname, 1024, "%s_Up.mtx", oprefix);
    handlers[1] = fopen(fname, "w");

    snprintf(fname, 1024, "%s_Sp.txt", oprefix);
    handlers[2] = fopen(fname, "w");

    snprintf(fname, 1024, "%s_Vtp.mtx", oprefix);
    handlers[3] = fopen(fname, "w");

    return 0;
}
