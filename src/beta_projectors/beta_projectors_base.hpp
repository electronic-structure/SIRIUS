// Copyright (c) 2013-2017 Anton Kozhevnikov, Thomas Schulthess
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

/** \file beta_projectors_base.hpp
 *
 *  \brief Contains declaration and implementation of sirius::Beta_projectors_base class.
 */

#ifndef __BETA_PROJECTORS_BASE_HPP__
#define __BETA_PROJECTORS_BASE_HPP__

#include "context/simulation_context.hpp"
#include "SDDK/wave_functions.hpp"

namespace sirius {

#if defined(SIRIUS_GPU)
extern "C" {

void
create_beta_gk_gpu_float(int num_atoms, int num_gkvec, int const* beta_desc, std::complex<float> const* beta_gk_t,
    double const* gkvec, double const* atom_pos, std::complex<float>* beta_gk);

void
create_beta_gk_gpu_double(int num_atoms, int num_gkvec, int const* beta_desc, std::complex<double> const* beta_gk_t,
    double const* gkvec, double const* atom_pos, std::complex<double>* beta_gk);

}
#endif

/// Named index of a descriptor of beta-projectors. The same order is used by the GPU kernel.
struct beta_desc_idx
{
    /// Number of beta-projector functions for this atom.
    static const int nbf      = 0;
    /// Offset of beta-projectors in this chunk.
    static const int offset   = 1;
    /// Offset of beta-projectors in the array for atom types.
    static const int offset_t = 2;
    /// Global index of atom.
    static const int ia       = 3;
};

struct beta_chunk_t
{
    /// Number of beta-projectors in the current chunk.
    int num_beta_;
    /// Number of atoms in the current chunk.
    int num_atoms_;
    /// Offset in the global index of beta projectors.
    int offset_;
    /// Descriptor of block of beta-projectors for an atom.
    sddk::mdarray<int, 2> desc_;
    /// Positions of atoms.
    sddk::mdarray<double, 2> atom_pos_;
};

/// Base class for beta-projectors, gradient of beta-projectors and strain derivatives of beta-projectors.
/** \tparam T  Precision of beta-projectors (float or double).
 */
template <typename T>
class Beta_projectors_base
{
  protected:

    Simulation_context& ctx_;

    /// List of G+k vectors.
    fft::Gvec const& gkvec_;

    /// Coordinates of G+k vectors used by GPU kernel.
    sddk::mdarray<double, 2> gkvec_coord_;

    /// Number of different components: 1 for beta-projectors, 3 for gradient, 9 for strain derivatives.
    int N_;

    /// Phase-factor independent coefficients of |beta> functions for atom types.
    sddk::mdarray<std::complex<T>, 3> pw_coeffs_t_;

    bool reallocate_pw_coeffs_t_on_gpu_{true};

    /// Set of beta PW coefficients for a chunk of atoms.
    sddk::matrix<std::complex<T>> pw_coeffs_a_;

    //sddk::mdarray<std::complex<T>, 1> pw_coeffs_a_g0_;

    std::vector<beta_chunk_t> beta_chunks_;

    int max_num_beta_;

    /// Total number of beta-projectors among atom types.
    int num_beta_t_;

    /// Split beta-projectors into chunks.
    void split_in_chunks();

  public:
    Beta_projectors_base(Simulation_context& ctx__, fft::Gvec const& gkvec__, int N__);

    /// Calculate inner product between beta-projectors and wave-functions.
    /** The following is matrix computed: <beta|phi>
     *
     *  \tparam F  Type of the resulting inner product matrix (float, double, complex<float> or complex<double>).
     */
    template <typename F>
    std::enable_if_t<std::is_same<T, real_type<F>>::value, sddk::dmatrix<F>>
    inner(sddk::memory_t mem__, int chunk__, wf::Wave_functions<T> const& phi__, wf::spin_index ispn__, wf::band_range br__) const
    {
        int nbeta = chunk(chunk__).num_beta_;

        sddk::dmatrix<F> result(nbeta, br__.size(), get_memory_pool(ctx_.host_memory_t()), "<beta|phi>");
        if (ctx_.processing_unit() == sddk::device_t::GPU) {
            result.allocate(get_memory_pool(sddk::memory_t::device));
        }

        wf::inner<F>(ctx_.spla_context(), mem__, wf::spin_range(ispn__.get()), *this,
                     wf::band_range(0, nbeta), phi__, br__, result, 0, 0);

        return result;
    }

    /// Generate beta-projectors for a chunk of atoms.
    /** Beta-projectors are always generated and stored in the memory of a processing unit.
     *
     *  \param [in] mem     Location of the beta-projectors (host or device memory).
     *  \param [in] ichunk  Index of a chunk of atoms for which beta-projectors are generated.
     *  \param [in] j index of the component (up to 9 components are used for the strain derivative)
     */
    void generate(sddk::memory_t mem__, int ichunk__, int j__);

    void prepare();

    void dismiss();

    inline auto num_gkvec_loc() const
    {
        return gkvec_.count();
    }

    inline auto ld() const
    {
        return this->num_gkvec_loc();
    }

    inline auto const& gkvec() const
    {
        return gkvec_;
    }

    inline auto num_md() const
    {
        return wf::num_mag_dims(0);
    }

    inline auto num_comp() const
    {
        return N_;
    }

    inline auto const& unit_cell() const
    {
        return ctx_.unit_cell();
    }

    auto& pw_coeffs_t(int ig__, int n__, int j__)
    {
        return pw_coeffs_t_(ig__, n__, j__);
    }

    auto pw_coeffs_t(int j__)
    {
        return sddk::matrix<std::complex<T>>(&pw_coeffs_t_(0, 0, j__), num_gkvec_loc(), num_beta_t());
    }

    /// Plane wave coefficients of |beta> projectors for a chunk of atoms.
    auto& pw_coeffs_a()
    {
        return pw_coeffs_a_;
    }

    auto const& pw_coeffs_a() const
    {
        return pw_coeffs_a_;
    }

    inline int num_beta_t() const
    {
        return num_beta_t_;
    }

    inline int num_chunks() const
    {
        return static_cast<int>(beta_chunks_.size());
    }

    inline beta_chunk_t const& chunk(int idx__) const
    {
        return beta_chunks_[idx__];
    }

    inline int max_num_beta() const
    {
        return max_num_beta_;
    }

    inline auto const& comm() const
    {
        return gkvec_.comm();
    }

    inline auto actual_spin_index(wf::spin_index s__) const
    {
        return wf::spin_index(0);
    }

    inline std::complex<T> const* at(sddk::memory_t mem__, int i__, wf::spin_index s__, wf::band_index b__) const
    {
        return pw_coeffs_a_.at(mem__, i__, b__.get());
    }
};

} // namespace

#endif
