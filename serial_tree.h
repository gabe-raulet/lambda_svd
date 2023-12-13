#ifndef SERIAL_TREE_H_
#define SERIAL_TREE_H_

int serial_tree
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

#endif
