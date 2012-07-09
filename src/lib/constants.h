#ifndef __CONSTANTS_H__
#define __CONSTANTS_H__

const int blas_worker = 1;
const int lapack_worker = 2;
const int cublas_worker = 3;
const int magma_worker = 4;
const int simple_worker = 5;

enum implementation {cpu, gpu};

enum diagonalization {second_variational, full};

enum spin_block {nm, uu, ud, dd};

// NIST value for the inverse fine structure (http://physics.nist.gov/cuu/Constants/index.html)
const double speed_of_light = 137.035999074; 

#endif // __CONSTANTS_H__

