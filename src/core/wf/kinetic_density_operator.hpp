/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich. All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef __KINETIC_DENSITY_OPERATOR_HPP__
#define __KINETIC_DENSITY_OPERATOR_HPP__

#include "core/fft/fft.hpp"
#include "core/fft/gvec.hpp"

namespace sirius::wf {

/// Smooth kinetic-energy density and its variational operator on the same FFT grid.
/** For one spin component, tau = 1/2 sum_nk f_nk w_k |grad psi_nk|^2.
 *  Coefficients use the SIRIUS wave-function normalization. add_density() takes
 *  f_nk w_k / Omega, whereas add_potential() applies -1/2 D_k . v_tau D_k
 *  without occupation or cell-volume factors, with D_k = grad + i k.
 *
 *  This host implementation accepts complex k-point wave functions and real
 *  Gamma-point wave functions. All ranks of the FFT communicator must call the
 *  methods for the same band. It neither reduces occupations over k-points nor
 *  supplies the atom-local PAW or muffin-tin/core contributions.
 */
template <typename T>
class Kinetic_density_operator
{
  private:
    fft::spfft_transform_type<T>& fft_;
    fft::Gvec_fft const& gkvec_;
    mdarray<std::complex<T>, 1> work_;

    void
    gradient(std::complex<T> const* phi__, int direction__)
    {
        // i(G+k) preserves Hermitian symmetry for the real Gamma-point FFT.
        #pragma omp parallel for schedule(static)
        for (int ig = 0; ig < gkvec_.count(); ig++) {
            work_[ig] = std::complex<T>(0, static_cast<T>(gkvec_.gkvec_cart(ig)[direction__])) * phi__[ig];
        }
        fft_.backward(reinterpret_cast<T const*>(work_.at(memory_t::host)), SPFFT_PU_HOST);
    }

  public:
    Kinetic_density_operator(fft::spfft_transform_type<T>& fft__, fft::Gvec_fft const& gkvec__)
        : fft_(fft__)
        , gkvec_(gkvec__)
        , work_({gkvec__.count()})
    {
        if (fft_.processing_unit() != SPFFT_PU_HOST) {
            RTE_THROW("kinetic-density operator currently requires a host FFT");
        }
        if (fft_.num_local_elements() != gkvec_.count()) {
            RTE_THROW("kinetic-density operator: inconsistent FFT and G-vector sizes");
        }
        bool real_fft = fft_.type() == SPFFT_TRANS_R2C;
        if (real_fft != gkvec_.gvec().reduced()) {
            RTE_THROW("kinetic-density operator: inconsistent real FFT and G-vector reduction");
        }
        if (real_fft && gkvec_.gvec().vk().length() != 0) {
            RTE_THROW("a real kinetic-density FFT requires the Gamma point");
        }
    }

    /// Accumulate one occupied wave-function contribution into a host real-space array.
    void
    add_density(T weight__, std::complex<T> const* phi__, T* tau__)
    {
        for (int x = 0; x < 3; x++) {
            gradient(phi__, x);
            auto ptr = fft_.space_domain_data(SPFFT_PU_HOST);
            #pragma omp parallel for schedule(static)
            for (int ir = 0; ir < fft_.local_slice_size(); ir++) {
                T square = fft_.type() == SPFFT_TRANS_R2C
                                   ? ptr[ir] * ptr[ir]
                                   : std::norm(reinterpret_cast<std::complex<T> const*>(ptr)[ir]);
                tau__[ir] += T(0.5) * weight__ * square;
            }
        }
    }

    /// Add the generalized-Kohn-Sham contribution of a real v_tau to host PW coefficients.
    /** Input and output coefficient arrays must not overlap. The FFT workspace is overwritten. */
    void
    add_potential(T const* vtau__, std::complex<T> const* phi__, std::complex<T>* hphi__)
    {
        if (gkvec_.count() && phi__ == hphi__) {
            RTE_THROW("kinetic-density operator does not support in-place application");
        }
        for (int x = 0; x < 3; x++) {
            gradient(phi__, x);
            fft::spfft_multiply<T>(fft_, [&](int ir) { return vtau__[ir]; });
            fft_.forward(SPFFT_PU_HOST, reinterpret_cast<T*>(work_.at(memory_t::host)), SPFFT_FULL_SCALING);
            #pragma omp parallel for schedule(static)
            for (int ig = 0; ig < gkvec_.count(); ig++) {
                hphi__[ig] += std::complex<T>(0, static_cast<T>(-0.5 * gkvec_.gkvec_cart(ig)[x])) * work_[ig];
            }
        }
    }
};

} // namespace sirius::wf

#endif
