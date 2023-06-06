/** \file generate_w90_coeffs.hpp
 *
 *  \brief Interface to W90 library.
 */
#ifdef SIRIUS_WANNIER90

#ifndef __GENERATE_W90_COEFFS_HPP__
#define __GENERATE_W90_COEFFS_HPP__

#include "k_point_set.hpp"

namespace sirius {
extern "C" {
void wannier_setup_(const char*, int32_t*, int32_t*, const double*, const double*, double*,
                    int32_t*, // care! arg (4,5) changed with const
                    int32_t*, char (*)[3], double*, bool*, bool*, int32_t*, int32_t*, int32_t*, int32_t*, int32_t*,
                    double*, int32_t*, int32_t*, int32_t*, double*, double*, double*, int32_t*, int32_t*, double*,
                    size_t, size_t);

void wannier_run_(const char*, int32_t*, int32_t*, double*, double*, double*, int32_t*, int32_t*, int32_t*, int32_t*,
                  char (*)[3], double*, bool*, std::complex<double>*, std::complex<double>*, double*,
                  std::complex<double>*, std::complex<double>*, bool*, double*, double*, double*, size_t, size_t);
}

void write_Amn(sddk::mdarray<std::complex<double>, 3>& Amn);

void write_Mmn(sddk::mdarray<std::complex<double>, 4>& M, sddk::mdarray<int, 2>& nnlist,
               sddk::mdarray<int32_t, 3>& nncell);

void write_eig(sddk::mdarray<double, 2>& eigval);
} // namespace sirius

#endif
#endif // SIRIUS_WANNIER90
