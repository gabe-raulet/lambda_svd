
#ifndef MMIO_DENSE_H_
#define MMIO_DENSE_H_

#include <stdio.h>

double* mmio_read_dense(FILE *f, int *m, int *n);
int mmio_write_dense(FILE *f, double const *A, int m, int n);

double* mmread(char const *fname, int *m, int *n);
int mmwrite(char const *fname, double const *A, int m, int n);

int mmwrite_diagonal(char const *fname, double const *D, int n);
int mmwrite_upper_triangular(char const *fname, double const *A, int n, int lda);

int write_diag(const char *fname, const double *D, int n);

#endif
