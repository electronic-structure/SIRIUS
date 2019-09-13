// Copyright (c) 2013-2018 Anton Kozhevnikov, Thomas Schulthess
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that
// the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
//    following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
//    and the following disclaimer in the documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/** \file local_operator.hpp
 *
 *  \brief Contains declaration and implementation of sirius::Local_operator class.
 */

#ifndef __LOCAL_OPERATOR_HPP__
#define __LOCAL_OPERATOR_HPP__

#include "Potential/potential.hpp"
#include "../SDDK/GPU/acc.hpp"

#ifdef __GPU
extern "C" void mul_by_veff_gpu(int ispn__, int size__, double* const* veff__, double_complex* buf__);

extern "C" void add_pw_ekin_gpu(int                   num_gvec__,
                                double                alpha__,
                                double const*         pw_ekin__,
                                double_complex const* phi__,
                                double_complex const* vphi__,
                                double_complex*       hphi__);
#endif

namespace sirius {

/// Representation of the local operator.
/** The following functionality is implementated:
 *    - application of the local part of Hamiltonian (kinetic + potential) to the wave-fucntions in the PP-PW case
 *    - application of the interstitial part of H and O in the case of FP-LAPW
 *    - application of the interstitial part of effective magnetic field to the first-variational functios
 *    - remapping of potential and unit-step functions from fine to coarse mesh of G-vectors
 */
class Local_operator
{
  private:
    /// Common parameters.
    Simulation_context const& ctx_;

    /// Coarse-grid FFT driver for this operator.
    FFT3D& fft_coarse_;

    /// Distribution of the G-vectors for the FFT transformation.
    Gvec_partition const& gvec_coarse_p_;

    Gvec_partition const* gkvec_p_{nullptr};

    /// Kinetic energy of G+k plane-waves.
    mdarray<double, 1> pw_ekin_;

    /// Effective potential components on a coarse FFT grid.
    std::array<Smooth_periodic_function<double>, 4> veff_vec_;

    /// Temporary array to store [V*phi](G)
    mdarray<double_complex, 2> vphi_;

    /// LAPW unit step function on a coarse FFT grid.
    Smooth_periodic_function<double> theta_;

    /// Temporary array to store psi_{up}(r).
    /** The size of the array is equal to the size of FFT buffer. */
    mdarray<double_complex, 1> buf_rg_;

    /// V(G=0) matrix elements.
    double v0_[2];

  public:
    /// Constructor.
    Local_operator(Simulation_context const& ctx__,
                   FFT3D&                    fft_coarse__,
                   Gvec_partition     const& gvec_coarse_p__)
        : ctx_(ctx__)
        , fft_coarse_(fft_coarse__)
        , gvec_coarse_p_(gvec_coarse_p__)

    {
        PROFILE("sirius::Local_operator");

        for (int j = 0; j < ctx_.num_mag_dims() + 1; j++) {
            veff_vec_[j] = Smooth_periodic_function<double>(fft_coarse__, gvec_coarse_p__);
            for (int ir = 0; ir < fft_coarse_.local_size(); ir++) {
                veff_vec_[j].f_rg(ir) = 2.71828;
            }
        }
        if (ctx_.full_potential()) {
            theta_ = Smooth_periodic_function<double>(fft_coarse__, gvec_coarse_p__);
        }

        if (fft_coarse_.pu() == device_t::GPU) {
            for (int j = 0; j < ctx_.num_mag_dims() + 1; j++) {
                veff_vec_[j].f_rg().allocate(memory_t::device).copy_to(memory_t::device);
            }
            buf_rg_.allocate(memory_t::device);
        }
    }

    /// Keep track of the total number of wave-functions to which the local operator was applied.
    static int num_applied(int n = 0)
    {
        static int num_applied_{0};
        num_applied_ += n;
        return num_applied_;
    }

    /// Map effective potential and magnetic field to a coarse FFT mesh.
    /** \param [in] potential      \f$ V_{eff}({\bf r}) \f$ and \f$ {\bf B}_{eff}({\bf r}) \f$ on the fine grid FFT grid.
     *
     *  This function should be called prior to the band diagonalziation. In case of GPU execution all
     *  effective fields on the coarse grid will be copied to the device and will remain there until the
     *  dismiss() method is called after band diagonalization.
     */
    void prepare(Potential& potential__);

    /// Prepare the k-point dependent arrays.
    void prepare(Gvec_partition const& gkvec_p__);

    /// Cleanup the local operator.
    void dismiss();

    /// Apply local part of Hamiltonian to wave-functions.
    /** \param [in]  ispn Index of spin.
     *  \param [in]  phi  Input wave-functions.
     *  \param [out] hphi Hamiltonian applied to wave-function.
     *  \param [in]  idx0 Starting index of wave-functions.
     *  \param [in]  n    Number of wave-functions to which H is applied.
     *
     *  Index of spin can take the following values:
     *    - 0: apply H_{uu} to the up- component of wave-functions
     *    - 1: apply H_{dd} to the dn- component of wave-functions
     *    - 2: apply full Hamiltonian to the spinor wave-functions
     *
     *  In the current implementation for the GPUs sequential FFT is assumed.
     */
    void apply_h(int ispn__, Wave_functions& phi__, Wave_functions& hphi__, int idx0__, int n__);

    void apply_h_o(int             N__,
                   int             n__,
                   Wave_functions& phi__,
                   Wave_functions* hphi__,
                   Wave_functions* ophi__);

    /// Apply magnetic field to the wave-functions.
    /** In case of collinear magnetism only Bz is applied to <tt>phi</tt> and stored in the first component of
     *  <tt>bphi</tt>. In case of non-collinear magnetims Bx-iBy is also applied and stored in the third
     *  component of <tt>bphi</tt>. The second component of <tt>bphi</tt> is used to store -Bz|phi>. */
    void apply_b(int                          N__,
                 int                          n__,
                 Wave_functions&              phi__,
                 std::vector<Wave_functions>& bphi__);

    inline double v0(int ispn__) const
    {
        return v0_[ispn__];
    }
};

} // namespace sirius

#endif // __LOCAL_OPERATOR_H__
