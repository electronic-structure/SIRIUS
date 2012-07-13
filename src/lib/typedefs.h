#ifndef __TYPEDEFS_H__
#define __TYPEDEFS_H__

#include <complex>

typedef std::complex<double> complex16;

enum spin_block {nm, uu, ud, dd};

enum implementation {cpu, gpu};

#endif // __TYPEDEFS_H__
