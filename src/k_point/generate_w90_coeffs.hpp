/** \file generate_w90_coeffs.hpp
 *
 *  \brief Interface to W90 library.
 */
#ifdef SIRIUS_WANNIER90

#ifndef __GENERATE_W90_COEFFS_HPP__
#define __GENERATE_W90_COEFFS_HPP__

#include "k_point_set.hpp"

namespace sirius {

using fortran_bool = int32_t;//int16_t;

extern "C" {
// TODO: add names of arguments
void wannier_setup_(const char*, int32_t*, int32_t*, const double*, const double*, double*,
                    int32_t*, // care! arg (4,5) changed with const
                    int32_t*, char (*)[3], double*, fortran_bool*, fortran_bool*, int32_t*, int32_t*, int32_t*, int32_t*, int32_t*,
                    double*, int32_t*, int32_t*, int32_t*, double*, double*, double*, int32_t*, int32_t*, double*,
                    size_t, size_t);

void wannier_run_(const char*, int32_t*, int32_t*, double*, double*, double*, int32_t*, int32_t*, int32_t*, int32_t*,
                  char (*)[3], double*, fortran_bool*, std::complex<double>*, std::complex<double>*, double*,
                  std::complex<double>*, std::complex<double>*, fortran_bool*, double*, double*, double*, size_t, size_t);
}



struct k_info{
    r3::vector<double> ibz;
    int ik_ibz;
    r3::vector<double> fbz;
    r3::vector<double> G;
    r3::matrix<int> R;
    r3::matrix<int> invR;
    r3::vector<double> t;
};

void write_Amn(sddk::mdarray<std::complex<double>, 3> const& Amn, 
          int const& num_kpts, int const& num_bands, int const& num_wann);

void write_Mmn(sddk::mdarray<std::complex<double>, 4> const& M, 
          sddk::mdarray<int, 2> const& nnlist, sddk::mdarray<int32_t, 3> const& nncell,
          int const& num_kpts, int const& num_neighbors, int const& num_bands);

void write_eig(sddk::mdarray<double, 2> const& eigval, 
          int const& num_bands, int const& num_kpts);

void from_irreduciblewedge_to_fullbrillouinzone(K_point_set& kset_ibz, 
          K_point_set& kset_fbz, std::vector<k_info>& k_temp);

void rotate_wavefunctions(K_point_set& kset_ibz, K_point_set& kset_fbz, 
          std::vector<k_info> const& k_temp);

void calculate_Amn(K_point_set& kset_fbz, int const& num_bands, 
          int const& num_wann, sddk::mdarray<std::complex<double>, 3>& A);

void send_receive_kpb(std::vector<std::shared_ptr<fft::Gvec>>& gvec_kpb, 
          std::vector<sddk::mdarray<std::complex<double>, 2>>& wf_kpb, K_point_set& kset_fbz, 
          std::vector<int>& ikpb_index, int const& nntot, sddk::mdarray<int,2> const& nnlist, 
          int const& num_bands);

void calculate_Mmn(sddk::mdarray<std::complex<double>,4>& M, K_point_set& kset_fbz, 
          int const& num_bands, std::vector<std::shared_ptr<fft::Gvec>> const& gvec_kpb, 
          std::vector<sddk::mdarray<std::complex<double>, 2>> const& wf_kpb,
          std::vector<int> const& ikpb_index, int const& nntot, sddk::mdarray<int,2> const& nnlist, 
          sddk::mdarray<int,3> const& nncell);

} // namespace sirius



#endif
#endif // SIRIUS_WANNIER90
