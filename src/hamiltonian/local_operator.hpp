/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file local_operator.hpp
 *
 *  \brief Declaration of sirius::Local_operator class.
 */

#ifndef __LOCAL_OPERATOR_HPP__
#define __LOCAL_OPERATOR_HPP__

#include "core/typedefs.hpp"
#include "core/fft/fft.hpp"
#include "core/rte/rte.hpp"

/* forward declarations */
namespace sirius {
class Potential;
class Simulation_context;
template <typename T>
class Smooth_periodic_function;
namespace wf {
template <typename T>
class Wave_functions;
class band_range;
class spin_range;
} // namespace wf
namespace fft {
class Gvec_fft;
}
} // namespace sirius
namespace spfft {
class Transform;
}

#ifdef SIRIUS_GPU
extern "C" {

void
add_to_hphi_pw_gpu_float(int num_gvec__, int add_ekin__, void const* pw_ekin__, void const* phi__, void const* vphi__,
                         void* hphi__);

void
add_to_hphi_pw_gpu_double(int num_gvec__, int add_ekin__, void const* pw_ekin__, void const* phi__, void const* vphi__,
                          void* hphi__);

void
add_to_hphi_lapw_gpu_float(int num_gvec__, void const* p__, void const* gkvec_cart__, void* hphi__);

void
add_to_hphi_lapw_gpu_double(int num_gvec__, void const* p__, void const* gkvec_cart__, void* hphi__);

void
grad_phi_lapw_gpu_float(int num_gvec__, void const* p__, void const* gkvec_cart__, void* hphi__);

void
grad_phi_lapw_gpu_double(int num_gvec__, void const* p__, void const* gkvec_cart__, void* hphi__);

void
mul_by_veff_real_real_gpu_float(int nr__, void const* in__, void const* veff__, void* out__);

void
mul_by_veff_real_real_gpu_double(int nr__, void const* in__, void const* veff__, void* out__);

void
mul_by_veff_complex_real_gpu_float(int nr__, void const* in__, void const* veff__, void* out__);

void
mul_by_veff_complex_real_gpu_double(int nr__, void const* in__, void const* veff__, void* out__);

void
mul_by_veff_complex_complex_gpu_float(int nr__, void const* in__, float pref__, void const* vx__, void const* vy__,
                                      void* out__);

void
mul_by_veff_complex_complex_gpu_double(int nr__, void const* in__, double pref__, void const* vx__, void const* vy__,
                                       void* out__);

} // extern C
#endif

template <typename T>
inline void
mul_by_veff_real_real_gpu(int nr__, T const* in__, T const* veff__, T* out__)
{
#ifdef SIRIUS_GPU
    if (std::is_same<T, float>::value) {
        mul_by_veff_real_real_gpu_float(nr__, in__, veff__, out__);
    }
    if (std::is_same<T, double>::value) {
        mul_by_veff_real_real_gpu_double(nr__, in__, veff__, out__);
    }
#else
    RTE_THROW("not compiled with GPU support");
#endif
}

template <typename T>
inline void
mul_by_veff_complex_real_gpu(int nr__, std::complex<T> const* in__, T const* veff__, std::complex<T>* out__)
{
#ifdef SIRIUS_GPU
    if (std::is_same<T, float>::value) {
        mul_by_veff_complex_real_gpu_float(nr__, in__, veff__, out__);
    }
    if (std::is_same<T, double>::value) {
        mul_by_veff_complex_real_gpu_double(nr__, in__, veff__, out__);
    }
#else
    RTE_THROW("not compiled with GPU support");
#endif
}

template <typename T>
inline void
mul_by_veff_complex_complex_gpu(int nr__, std::complex<T> const* in__, T pref__, T const* vx__, T const* vy__,
                                std::complex<T>* out__)
{
#ifdef SIRIUS_GPU
    if (std::is_same<T, float>::value) {
        mul_by_veff_complex_complex_gpu_float(nr__, in__, pref__, vx__, vy__, out__);
    }
    if (std::is_same<T, double>::value) {
        mul_by_veff_complex_complex_gpu_double(nr__, in__, pref__, vx__, vy__, out__);
    }
#else
    RTE_THROW("not compiled with GPU support");
#endif
}

template <typename T>
inline void
add_to_hphi_pw_gpu(int num_gvec__, int add_ekin__, T const* pw_ekin__, std::complex<T> const* phi__,
                   std::complex<T> const* vphi__, std::complex<T>* hphi__)
{
#ifdef SIRIUS_GPU
    if (std::is_same<T, float>::value) {
        add_to_hphi_pw_gpu_float(num_gvec__, add_ekin__, pw_ekin__, phi__, vphi__, hphi__);
    }
    if (std::is_same<T, double>::value) {
        add_to_hphi_pw_gpu_double(num_gvec__, add_ekin__, pw_ekin__, phi__, vphi__, hphi__);
    }
#else
    RTE_THROW("not compiled with GPU support");
#endif
}

template <typename T>
inline void
add_to_hphi_lapw_gpu(int num_gvec__, std::complex<T> const* p__, T const* gkvec_cart__, std::complex<T>* hphi__)
{
#ifdef SIRIUS_GPU
    if (std::is_same<T, float>::value) {
        add_to_hphi_lapw_gpu_float(num_gvec__, p__, gkvec_cart__, hphi__);
    }
    if (std::is_same<T, double>::value) {
        add_to_hphi_lapw_gpu_double(num_gvec__, p__, gkvec_cart__, hphi__);
    }
#else
    RTE_THROW("not compiled with GPU support");
#endif
}

template <typename T>
inline void
grad_phi_lapw_gpu(int num_gvec__, std::complex<T> const* p__, T const* gkvec_cart__, std::complex<T>* hphi__)
{
#ifdef SIRIUS_GPU
    if (std::is_same<T, float>::value) {
        grad_phi_lapw_gpu_float(num_gvec__, p__, gkvec_cart__, hphi__);
    }
    if (std::is_same<T, double>::value) {
        grad_phi_lapw_gpu_double(num_gvec__, p__, gkvec_cart__, hphi__);
    }
#else
    RTE_THROW("not compiled with GPU support");
#endif
}

namespace sirius {

/// Representation of the local operator.
/** The following functionality is implementated:
 *    - application of the local part of Hamiltonian (kinetic + potential) to the wave-functions in the PP-PW case
 *    - application of the interstitial part of H and O in the case of FP-LAPW
 *    - application of the interstitial part of effective magnetic field to the first-variational functios
 *    - remapping of potential and unit-step functions from fine to coarse mesh of G-vectors
 */
template <typename T>
class Local_operator
{
  private:
    /// Common parameters.
    Simulation_context const& ctx_;

    /// Coarse-grid FFT driver for this operator.
    fft::spfft_transform_type<T>& fft_coarse_;

    /// Distribution of the G-vectors for the FFT transformation.
    std::shared_ptr<fft::Gvec_fft> gvec_coarse_p_;

    /// Kinetic energy of G+k plane-waves.
    mdarray<T, 1> pw_ekin_;

    mdarray<T, 2> gkvec_cart_;

    // Names for indices.
    struct v_local_index_t
    {
        static const int v0     = 0;
        static const int v1     = 1;
        static const int vx     = 2;
        static const int vy     = 3;
        static const int theta  = 4;
        static const int rm_inv = 5;
    };

    /// Effective potential components and unit step function on a coarse FFT grid.
    /** The following elements are stored in the array:
         - V(r) + B_z(r) (in PP-PW case) or V(r) (in FP-LAPW case)
         - V(r) - B_z(r) (in PP-PW case) or B_z(r) (in FP-LAPW case)
         - B_x(r)
         - B_y(r)
         - Theta(r) (in FP-LAPW case)
         - inverse of 1 + relative mass (needed for ZORA LAPW)
     */
    std::array<std::unique_ptr<Smooth_periodic_function<T>>, 6> veff_vec_;

    /// Temporary array to store [V*phi](G)
    mdarray<std::complex<T>, 1> vphi_;

    /// Temporary array to store psi_{up}(r).
    /** The size of the array is equal to the size of FFT buffer. */
    mdarray<std::complex<T>, 1> buf_rg_;

    /// V(G=0) matrix elements.
    T v0_[2];

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
    Local_operator(Simulation_context const& ctx__, fft::spfft_transform_type<T>& fft_coarse__,
                   std::shared_ptr<fft::Gvec_fft> gvec_coarse_fft__, Potential* potential__ = nullptr);

    /// Prepare the k-point dependent arrays.
    /** \param [in] gkvec_p  FFT-friendly G+k vector partitioning. */
    void
    prepare_k(fft::Gvec_fft const& gkvec_p__);

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
    void
    apply_h(fft::spfft_transform_type<T>& spfftk__, std::shared_ptr<fft::Gvec_fft> gkvec_fft__, wf::spin_range spins__,
            wf::Wave_functions<T> const& phi__, wf::Wave_functions<T>& hphi__, wf::band_range br__);

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
    void
    apply_fplapw(fft::spfft_transform_type<T>& spfftik__, std::shared_ptr<fft::Gvec_fft> gkvec_fft__,
                 wf::band_range b__, wf::Wave_functions<T>& phi__, wf::Wave_functions<T>* hphi__,
                 wf::Wave_functions<T>* ophi__, wf::Wave_functions<T>* bzphi__, wf::Wave_functions<T>* bxyphi__);

    /// Apply magnetic field to the full-potential wave-functions.
    /** In case of collinear magnetism only Bz is applied to <tt>phi</tt> and stored in the first component of
     *  <tt>bphi</tt>. In case of non-collinear magnetims Bx-iBy is also applied and stored in the third
     *  component of <tt>bphi</tt>. The second component of <tt>bphi</tt> is used to store -Bz|phi>.
     *
     *  \param [in]  spfftk   SpFFT transform object for G+k vectors.
     *  \param [in]  phi      Input wave-functions.
     *  \param [out] bphi     Output vector of magentic field components, applied to the wave-functions.
     *  \param [in]  br       Range of bands to which B is applied.
     */
    // void apply_b(spfft_transform_type<T>& spfftk__, wf::Wave_functions<T> const& phi__,
    //              std::vector<wf::Wave_functions<T>>& bphi__, wf::band_range br__);

    inline T
    v0(int ispn__) const
    {
        return v0_[ispn__];
    }
};

} // namespace sirius

#endif // __LOCAL_OPERATOR_H__
