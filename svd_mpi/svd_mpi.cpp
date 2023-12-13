#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <mpi.h>
#include "kiss.h"
#include "mmiodense.h"
#include "svdalgs.h"
#include "dist_mpi_tree.h"
#include "cblas.h"
#include "lapacke.h"

int param_check(int m, int n, int p, int b, int r, int s, int isroot);
int get_file_handlers(FILE **handlers, char const *oprefix);

int main(int argc, char *argv[])
{
    int myrank, nprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);


    if (argc != 5)
    {
        if (!myrank) fprintf(stderr, "usage: %s <m:nrows> <n:ncols> <p:trunc> <oprefix>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int p = atoi(argv[3]);

    int r = m < n? m : n;
    int s = n / nprocs;


    if (param_check(m, n, p, nprocs, r, s, (myrank==0)) != 0)
    {
        MPI_Finalize();
        return 1;
    }

    kiss_init();

    char const *oprefix = argv[4];

    double *A, *Aloc, *Up, *Sp, *Vtp;

    Aloc = (double*)malloc(m*s*sizeof(double));

    if (!myrank)
    {
        Up = (double*)malloc(m*p*sizeof(double));
        Sp = (double*)malloc(p*sizeof(double));
        Vtp = (double*)malloc(p*n*sizeof(double));
    }
    else
    {
        Up = Sp = Vtp = NULL;
    }

    /*
     * Generate matrix A in parallel.
     */

    for (int i = 0; i < m*s; ++i)
        Aloc[i] = kiss_unirandf();

    if (dist_mpi_tree(Aloc, Up, Sp, Vtp, m, n, p, 0, MPI_COMM_WORLD) != 0)
    {
        MPI_Finalize();
        return 1;
    }

    if (!myrank)
    {
        A = (double*)malloc(m*n*sizeof(double));
    }

    MPI_Gather(Aloc, m*s, MPI_DOUBLE, A, m*s, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    free(Aloc);

    if (!myrank)
    {
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
    }

    MPI_Finalize();

    return 0;
}

int param_check(int m, int n, int p, int b, int r, int s, int isroot)
{
    if (!(m >= 1 && n >= 1) || (m&(m-1)) || (n&(n-1)))
    {
        if (isroot) fprintf(stderr, "[main::error::param_check][m=%d,n=%d] must have m,n >= 1 with m and n both powers of 2\n", m, n);
        return 1;
    }

    if ((b&(b-1)) || b <= 0)
    {
        if (isroot) fprintf(stderr, "[main::error::param_check][b=%d] must have b >= 1 with b being a power of 2\n", b);
        return 1;
    }

    if (!(p <= r && p <= s))
    {
        if (isroot) fprintf(stderr, "[main::error::param_check][p=%d,r=%d,s=%d] must have p <= r and p <= s\n", p, r, s);
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
