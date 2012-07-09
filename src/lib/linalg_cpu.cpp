#include "linalg_cpu.h"

void zcopy(int32_t n, complex16 *zx, int32_t incx, complex16 *zy, int32_t incy)
{
    FORTRAN(zcopy)(&n, zx, &incx, zy, &incy);
}

int dgtsv(int n, int nrhs, double *dl, double *d, double *du, double *b, int ldb)
{
    int info;

    FORTRAN(dgtsv)(&n, &nrhs, dl, d, du, b, &ldb, &info);
    
    return info;
}
