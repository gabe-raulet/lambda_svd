#ifndef MPI_TIMER_H_
#define MPI_TIMER_H_

#include <mpi.h>

typedef struct
{
    int isroot;
    double telapsed;
    MPI_Comm comm;
} mpi_timer_t;

int mpi_timer_init(mpi_timer_t *timer, MPI_Comm comm);
int mpi_timer_start(mpi_timer_t *timer);
int mpi_timer_stop(mpi_timer_t *timer);
int mpi_timer_query(mpi_timer_t *timer, double *maxtime, double *proctime);

#endif
