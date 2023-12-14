#include "dist_mpi_tree.h"
#include "svdalgs.h"
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

int dist_mpi_tree
(
    double *Aloc, /* (rank[myrank]) input m-by-(n/nprocs) matrix */
    double *Up, /* (rank[root]) output m-by-p matrix */
    double *Sp, /* (rank[root]) output p-by-p diagonal matrix */
    double *Vtp, /* (rank[root]) output p-by-n matrix */
    int m, /* rows of A */
    int n, /* columns of A */
    int p, /* rank approximation */
    int root, /* root rank */
    MPI_Comm comm /* communicator */
)
{
    int myrank, nprocs;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);

    int r = m < n? m : n;
    int s = n / nprocs;

    assert(n % nprocs == 0);
    assert(p <= r && p <= s);

    double *A1i, *Vt1i;

    A1i = (double*)malloc(m*s*sizeof(double));
    Vt1i = (double*)malloc(s*p*sizeof(double));

    int q = log2i(nprocs);

    seed_node(Aloc, A1i, Vt1i, m, n, q, p);

    double *Amem = (double*)malloc(2*m*p*sizeof(double));
    double *Vtmem = (double*)malloc(n*p*sizeof(double)); /* this should be allocated with less memory depending on what myrank is */

    if (myrank % 2 != 0)
    {
        MPI_Send(A1i, m*p, MPI_DOUBLE, myrank-1, myrank, comm);
        MPI_Send(Vt1i, p*s, MPI_DOUBLE, myrank-1, myrank+nprocs, comm);
    }
    else
    {
        memcpy(Amem, A1i, m*p*sizeof(double));
        memcpy(Vtmem, Vt1i, p*s*sizeof(double));
        MPI_Recv(&Amem[m*p], m*p, MPI_DOUBLE, myrank+1, myrank+1, comm, MPI_STATUS_IGNORE);
        MPI_Recv(&Vtmem[p*s], p*s, MPI_DOUBLE, myrank+1, myrank+1+nprocs, comm, MPI_STATUS_IGNORE);
    }

    double *Ak_2i_0, *Vtk_2i_0, *Ak_2i_1, *Vtk_2i_1, *Ak1_lj, *Vtk1_lj;

    for (int k = 1; k < q; ++k)
    {
        int d = s * (1 << (k-1)); /* column count of incoming Vtk_2i_j matrices */

        Ak_2i_0 = &Amem[0]; /* m-by-p */
        Ak_2i_1 = &Amem[m*p]; /* m-by-p */
        Vtk_2i_0 = &Vtmem[0]; /* p-by-d */
        Vtk_2i_1 = &Vtmem[p*d]; /* p-by-d */

        Ak1_lj = &Amem[0]; /* m-by-p on exit */
        Vtk1_lj = &Vtmem[0]; /* p-by-2d on exit */

        combine_node(Ak_2i_0, Vtk_2i_0, Ak_2i_1, Vtk_2i_1, Ak1_lj, Vtk1_lj, m, n, k, q, p);

        if ((myrank % (1 << (k+1))) == (1 << k))
        {

            int dest = myrank - (1 << k);
            int Atag = myrank;
            int Vtag = myrank + nprocs;

            MPI_Send(Ak1_lj, m*p, MPI_DOUBLE, dest, Atag, comm);
            MPI_Send(Vtk1_lj, p*2*d, MPI_DOUBLE, dest, Vtag, comm);
        }
        else if ((myrank % (1 << (k+1))) == 0)
        {

            int source = myrank + (1 << k);
            int Atag = myrank + (1 << k);
            int Vtag = myrank + (1 << k) + nprocs;

            MPI_Recv(&Amem[m*p], m*p, MPI_DOUBLE, source, Atag, comm, MPI_STATUS_IGNORE);
            MPI_Recv(&Vtmem[p*2*d], p*2*d, MPI_DOUBLE, source, Vtag, comm, MPI_STATUS_IGNORE);
        }

    }

    MPI_Barrier(comm);

    double *Aq1_11, *Aq1_12, *Vtq1_11, *Vtq1_12;

    if (myrank == root)
    {
        Aq1_11 = &Amem[0];
        Aq1_12 = &Amem[m*p];
        Vtq1_11 = &Vtmem[0];
        Vtq1_12 = &Vtmem[(n*p)>>1];

        extract_node(Aq1_11, Vtq1_11, Aq1_12, Vtq1_12, Up, Sp, Vtp, m, n, q, p);
    }

    free(A1i);
    free(Vt1i);

    MPI_Barrier(comm);
    return 0;
}
