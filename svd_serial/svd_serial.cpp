#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include "kiss.h"
#include "mmiodense.h"
#include "svdalgs.h"
#include "serial_tree.h"
#include "cblas.h"
#include "lapacke.h"

int param_check(int m, int n, int p, int b, int r, int s);
int get_file_handlers(FILE **handlers, char const *oprefix);

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

    serial_tree(A, Up, Sp, Vtp, m, n, p, b);

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
