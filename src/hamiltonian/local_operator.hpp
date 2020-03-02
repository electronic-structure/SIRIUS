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
 *  \brief Declaration of sirius::Local_operator class.
 */

#ifndef __LOCAL_OPERATOR_HPP__
#define __LOCAL_OPERATOR_HPP__

#include "SDDK/memory.hpp"
#include "spfft/spfft.hpp"
#include "typedefs.hpp"

/* forward declarations */
namespace sirius {
class Potential;
class Simulation_context;
template <typename T>
class Smooth_periodic_function;
}
namespace sddk {
class FFT3D;
class Gvec_partition;
class Wave_functions;
class spin_range;
}
namespace spfft {
class Transform;
}

#ifdef __GPU
extern "C" void mul_by_veff_real_real_gpu(int nr__, double* buf__, double* veff__);

extern "C" void mul_by_veff_complex_real_gpu(int nr__, double_complex* buf__, double* veff__);

extern "C" void mul_by_veff_complex_complex_gpu(int nr__, double_complex* buf__, double pref__, double* vx__, double* vy__);

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
    spfft::Transform& fft_coarse_;

    /// Distribution of the G-vectors for the FFT transformation.
    sddk::Gvec_partition const& gvec_coarse_p_;

    /// Kinetic energy of G+k plane-waves.
    sddk::mdarray<double, 1> pw_ekin_;

    /// Effective potential components and unit step function on a coarse FFT grid.
    /** The following elements are stored in the array:
         - V(r) + B_z(r) (in PP-PW case) or V(r) (in FP-LAPW case)
         - V(r) - B_z(r) (in PP-PW case) or B_z(r) (in FP-LAPW case)
         - B_x(r)
         - B_y(r)
         - Theta(r) (in FP-LAPW case)
         - inverse of 1 + relative mass (needed for ZORA LAPW)
     */
    std::array<std::unique_ptr<Smooth_periodic_function<double>>, 6> veff_vec_;

    /// Temporary array to store [V*phi](G)
    sddk::mdarray<double_complex, 1> vphi_;

    /// Temporary array to store psi_{up}(r).
    /** The size of the array is equal to the size of FFT buffer. */
    sddk::mdarray<double_complex, 1> buf_rg_;

    /// V(G=0) matrix elements.
    double v0_[2];

  public:
    /// Constructor.
    /** Prepares k-point independent part of the local potential. If potential is provided, it is mapped to the
     *  coarse FFT grid, otherwise the constant potential is assumed for the debug and benchmarking purposes.
     *  In the case of GPU-enabled FFT driver all effective fields on the coarse grid are copied to the device
     *  and remain there until the local operator is destroyed.

     *  \param [in] ctx           Simulation context.
     *  \param [in] fft_coarse    Explicit FFT driver for the coarse mesh.
     *  \param [in] gvec_coarse_p FFT-friendly G-vector distribution for the coarse mesh.
     *  \param [in] potential     Effective potential and magnetic fields \f$ V_{eff}({\bf r}) \f$ and
     *                             \f$ {\bf B}_{eff}({\bf r}) \f$ on the fine FFT grid.
     */
    Local_operator(Simulation_context   const& ctx__,
                   spfft::Transform&           fft_coarse__,
                   sddk::Gvec_partition const& gvec_coarse_p__,
                   Potential*                  potential__ = nullptr);

    /// Prepare the k-point dependent arrays.
    /** \param [in] gkvec_p  FFT-friendly G+k vector partitioning. */
    void prepare_k(sddk::Gvec_partition const& gkvec_p__);

    /// Apply local part of Hamiltonian to pseudopotential wave-functions.
    /** \param [in]  spfftk  SpFFT transform object for G+k vectors.
     *  \param [in]  gkvec_p FFT-friendly G+k vector partitioning.
     *  \param [in]  spins   Range of wave-function spins to which Hloc is applied.
     *  \param [in]  phi     Input wave-functions.
     *  \param [out] hphi    Local hamiltonian applied to wave-function.
     *  \param [in]  idx0    Starting index of wave-functions.
     *  \param [in]  n       Number of wave-functions to which H is applied.
     *
     *  Spin range can take the following values:
     *    - [0, 0]: apply H_{uu} to the up- component of wave-functions
     *    - [1, 1]: apply H_{dd} to the dn- component of wave-functions
     *    - [0, 1]: apply full Hamiltonian to the spinor wave-functions
     *
     *  Local Hamiltonian includes kinetic term and local part of potential.
     */
    void apply_h(spfft::Transform& spfftk__, sddk::Gvec_partition const& gkvec_p__, sddk::spin_range spins__,
                 sddk::Wave_functions& phi__, sddk::Wave_functions& hphi__, int idx0__, int n__);

    /// Apply local part of LAPW Hamiltonian and overlap operators.
    /** \param [in]  spfftk  SpFFT transform object for G+k vectors.
     *  \param [in]  gkvec_p FFT-friendly G+k vector partitioning.
     *  \param [in]  N       Starting index of wave-functions.
     *  \param [in]  n       Number of wave-functions to which H and O are applied.
     *  \param [in]  phi     Input wave-functions [always on CPU].
     *  \param [out] hphi    LAPW Hamiltonian applied to wave-function [CPU || GPU].
     *  \param [out] ophi    LAPW overlap matrix applied to wave-function [CPU || GPU].
     *
     *  Only plane-wave part of output wave-functions is changed.
     */
    void apply_h_o(spfft::Transform& spfftik__, sddk::Gvec_partition const& gkvec_p__, int N__, int n__,
                   sddk::Wave_functions& phi__, sddk::Wave_functions* hphi__, sddk::Wave_functions* ophi__);

    /// Apply magnetic field to the full-potential wave-functions.
    /** In case of collinear magnetism only Bz is applied to <tt>phi</tt> and stored in the first component of
     *  <tt>bphi</tt>. In case of non-collinear magnetims Bx-iBy is also applied and stored in the third
     *  component of <tt>bphi</tt>. The second component of <tt>bphi</tt> is used to store -Bz|phi>. 
     *
     *  \param [in]  spfftk   SpFFT transform object for G+k vectors.
     *  \param [in]  N        Starting index of wave-functions.
     *  \param [in]  n        Number of wave-functions to which H and O are applied.
     *  \param [in]  phi      Input wave-functions.
     *  \param [out] bphi     Output vector of magentic field components, aplied to the wave-functions.
     */
    void apply_b(spfft::Transform& spfftk__, int N__, int n__, sddk::Wave_functions& phi__,
                 std::vector<sddk::Wave_functions>& bphi__); // TODO: align argument order with apply_h()

    inline double v0(int ispn__) const
    {
        return v0_[ispn__];
    }
};

} // namespace sirius

#endif // __LOCAL_OPERATOR_H__
