#include "mpitimer.h"

int mpi_timer_init(mpi_timer_t *timer, MPI_Comm comm)
{
    int myrank;

    MPI_Comm_rank(comm, &myrank);
    timer->isroot = (myrank == 0);
    timer->telapsed = 0;
    timer->comm = comm;

    return 0;
}

int mpi_timer_start(mpi_timer_t *timer)
{
    MPI_Barrier(timer->comm);
    timer->telapsed = -MPI_Wtime();
    return 0;
}

int mpi_timer_stop(mpi_timer_t *timer)
{
    timer->telapsed += MPI_Wtime();
    return 0;
}

int mpi_timer_query(mpi_timer_t *timer, double *maxtime, double *proctime)
{
    MPI_Reduce(&timer->telapsed, maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, timer->comm);
    MPI_Reduce(&timer->telapsed, proctime, 1, MPI_DOUBLE, MPI_SUM, 0, timer->comm);
    return 0;
}
