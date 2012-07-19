#ifndef __TYPEDEFS_H__
#define __TYPEDEFS_H__

#include <complex>
#include <stdint.h>

typedef std::complex<double> complex16;

typedef double real8;

typedef int32_t int4;

enum spin_block {nm, uu, ud, dd};

enum implementation {cpu, gpu};

#endif // __TYPEDEFS_H__
