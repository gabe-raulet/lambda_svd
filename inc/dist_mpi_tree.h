#ifndef DIST_MPI_TREE_H_
#define DIST_MPI_TREE_H_

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
);

#endif
