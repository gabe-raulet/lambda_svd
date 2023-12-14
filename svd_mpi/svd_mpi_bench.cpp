#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <mpi.h>
#include "kiss.h"
#include "mmiodense.h"
#include "svdalgs.h"
#include "mpitimer.h"
#include "dist_mpi_tree.h"
#include "cblas.h"
#include "lapacke.h"

#define BUFSIZE 1024

int random_benchmark(int argc, char *argv[], MPI_Comm comm);
int stored_benchmark(int argc, char *argv[], MPI_Comm comm);
void usage(char *argv[]);

int main(int argc, char *argv[])
{
    int myrank, nprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if (argc < 2)
    {
        if (!myrank) usage(argv);
        MPI_Finalize();
        return 1;
    }


    if (!strcmp(argv[1], "random"))
    {
        if (random_benchmark(argc, argv, MPI_COMM_WORLD))
        {
            MPI_Finalize();
            return 1;
        }
    }
    else if (!strcmp(argv[1], "stored"))
    {
        if (stored_benchmark(argc, argv, MPI_COMM_WORLD))
        {
            MPI_Finalize();
            return 1;
        }
    }
    else
    {
        if (!myrank) usage(argv);
        MPI_Finalize();
        return 1;
    }

    MPI_Finalize();
    return 0;
}

void usage(char *argv[])
{
    fprintf(stderr, "Usage: mpirun -np $nprocs %s <random:stored> ...\n", argv[0]);
    fprintf(stderr, "random: <p:trunc> <r:rank> <a:mscale> <tall|short>\n");
    fprintf(stderr, "stored: <p:trunc> <iprefix> <oprefix>\n");
}

int random_benchmark(int argc, char *argv[], MPI_Comm comm)
{

    int myrank, nprocs;
    char const *mattype;

    int m, n, p, r, s, a;
    double *Aloc, *Utest, *Stest, *Vttest;

    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &myrank);

    if (argc != 6)
    {
        if (!myrank) fprintf(stderr, "Usage: mpirun -np $nprocs %s [random] <p:trunc> <r:rank> <a:mscale> <tall:wide>\n", argv[0]);
        return 1;
    }

    p = atoi(argv[2]);
    r = atoi(argv[3]);
    a = atof(argv[4]);
    mattype = argv[5];

    if (!strcmp(mattype, "tall"))
    {
        n = r;
        m = a*r;
    }
    else if (!strcmp(mattype, "wide"))
    {
        n = a*r;
        m = r;
    }
    else
    {
        m = n = r;
    }

    s = n / nprocs;

    assert(n % nprocs == 0);
    assert(a >= 1);
    assert(nprocs >= 2);
    assert(!(nprocs&(nprocs-1)));
    assert(m >= 1 && n >= 1);
    assert(!(m&(m-1)));
    assert(!(n&(n-1)));
    assert(p <= r);
    assert(p <= s);

    if (!myrank) fprintf(stderr, "[svd_mpi_bench][random][m=%d,n=%d,r=%d,p=%d,s=%d,nprocs=%d]\n", m, n, r, p, s, nprocs);

    double maxtime, proctime;

    mpi_timer_t timer;
    mpi_timer_init(&timer, comm);

    mpi_timer_start(&timer);

    Aloc = (double*)malloc(m*s*sizeof(double));

    kiss_init();

    for (int i = 0; i < m*s; ++i)
        Aloc[i] = kiss_unirandf();

    if (!myrank)
    {
        Utest = (double*)malloc(m*p*sizeof(double));
        Stest = (double*)malloc(p*sizeof(double));
        Vttest = (double*)malloc(p*n*sizeof(double));
    }
    else
    {
        Utest = Stest = Vttest = NULL;
    }

    mpi_timer_stop(&timer);
    mpi_timer_query(&timer, &maxtime, &proctime);

    if (!myrank) fprintf(stderr, "[initialization][maxtime=%.5f(s),proctime=%.5f(s),meantime=%.5f(s)]\n", maxtime, proctime, proctime / nprocs);

    mpi_timer_start(&timer);

    if (dist_mpi_tree(Aloc, Utest, Stest, Vttest, m, n, p, 0, comm))
        return 1;

    mpi_timer_stop(&timer);
    mpi_timer_query(&timer, &maxtime, &proctime);

    if (!myrank) fprintf(stderr, "[dist_mpi_tree][maxtime=%.5f(s),proctime=%.5f(s),meantime=%.5f(s)]\n", maxtime, proctime, proctime / nprocs);

    free(Aloc);

    if (!myrank)
    {
        free(Utest);
        free(Stest);
        free(Vttest);
    }

    return 0;
}

int stored_benchmark(int argc, char *argv[], MPI_Comm comm)
{
    int myrank, nprocs;
    char const *iprefix;
    char const *oprefix;
    char fname[BUFSIZE];
    FILE *f;

    int m, n, p, r, s;
    double *Aloc, *Utest, *Stest, *Vttest;

    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &myrank);

    if (argc != 5)
    {
        if (!myrank) fprintf(stderr, "Usage: mpirun -np $nprocs %s [stored] <p:trunc> <iprefix> <oprefix>\n", argv[0]);
        return 1;
    }

    p = atoi(argv[2]);
    iprefix = argv[3];
    oprefix = argv[4];

    snprintf(fname, BUFSIZE, "%s_A_%d_%d.mtx", iprefix, myrank+1, nprocs);
    Aloc = mmread(fname, &m, &s);

    if (!Aloc)
    {
        if (!myrank) fprintf(stderr, "error: unable to open file named '%s'\n", fname);
        return 1;
    }

    n = s*nprocs;
    r = m < n? m : n;

    assert(nprocs >= 2);
    assert(!(nprocs&(nprocs-1)));
    assert(m >= 1 && n >= 1);
    assert(!(m&(m-1)));
    assert(!(n&(n-1)));
    assert(p <= r);
    assert(p <= s);

    if (!myrank)
    {
        Utest = (double*)malloc(m*p*sizeof(double));
        Stest = (double*)malloc(p*sizeof(double));
        Vttest = (double*)malloc(p*n*sizeof(double));
    }
    else
    {
        Utest = Stest = Vttest = NULL;
    }

    if (dist_mpi_tree(Aloc, Utest, Stest, Vttest, m, n, p, 0, comm))
        return 1;

    free(Aloc);

    if (!myrank)
    {
        snprintf(fname, BUFSIZE, "%s_Utest.mtx", oprefix);
        mmwrite(fname, Utest, m, p);
        free(Utest);

        snprintf(fname, BUFSIZE, "%s_Vttest.mtx", oprefix);
        mmwrite(fname, Vttest, p, n);
        free(Vttest);

        snprintf(fname, BUFSIZE, "%s_Stest.txt", oprefix);

        f = fopen(fname, "w");

        for (int i = 0; i < p; ++i)
        {
            fprintf(f, "%.18e\n", Stest[i]);
        }

        fclose(f);
        free(Stest);
    }

    return 0;
}
