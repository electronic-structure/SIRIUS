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
extern "C" void create_beta_gk_gpu_float(int                        num_atoms,
                                         int                        num_gkvec,
                                         int const*                 beta_desc,
                                         std::complex<float> const* beta_gk_t,
                                         float const*               gkvec,
                                         double const*              atom_pos,
                                         std::complex<float>*       beta_gk);

extern "C" void create_beta_gk_gpu_double(int                   num_atoms,
                                          int                   num_gkvec,
                                          int const*            beta_desc,
                                          double_complex const* beta_gk_t,
                                          double const*         gkvec,
                                          double const*         atom_pos,
                                          double_complex*       beta_gk);
#endif

/// Named index of a descriptor of beta-projectors. The same order is used by the GPU kernel.
enum class beta_desc_idx
{
    /// Number of beta-projector functions for this atom.
    nbf      = 0,
    /// Offset of beta-projectors in this chunk.
    offset   = 1,
    /// Offset of beta-projectors in the array for atom types.
    offset_t = 2,
    /// Global index of atom.
    ia       = 3
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
    mdarray<int, 2> desc_;
    /// Positions of atoms.
    mdarray<double, 2> atom_pos_;
};

/// Base class for beta-projectors, gradient of beta-projectors and strain derivatives of beta-projectors.
template <typename T>
class Beta_projectors_base
{
  protected:

    Simulation_context& ctx_;

    /// List of G+k vectors.
    Gvec const& gkvec_;

    /// Mapping between local and global G+k vector index.
    std::vector<int> const& igk_;

    /// Coordinates of G+k vectors used by GPU kernel.
    mdarray<T, 2> gkvec_coord_;

    /// Number of different components: 1 for beta-projectors, 3 for gradient, 9 for strain derivatives.
    int N_;

    /// Phase-factor independent coefficients of |beta> functions for atom types.
    mdarray<std::complex<T>, 3> pw_coeffs_t_;

    bool reallocate_pw_coeffs_t_on_gpu_{true};

    /// Set of beta PW coefficients for a chunk of atoms.
    matrix<std::complex<T>> pw_coeffs_a_;

    mdarray<std::complex<T>, 1> pw_coeffs_a_g0_;

    std::vector<beta_chunk_t> beta_chunks_;

    int max_num_beta_;

    /// Total number of beta-projectors among atom types.
    int num_beta_t_;

    /// Split beta-projectors into chunks.
    void split_in_chunks();

    template<typename F, std::enable_if_t<std::is_same<std::complex<T>, F>::value, bool> = true>
    void local_inner_aux(F* beta_pw_coeffs_a_ptr__, int nbeta__, Wave_functions<T>& phi__,
                         int ispn__, int idx0__, int n__, matrix<F>& beta_phi__) const
    {
        auto pp = utils::get_env<int>("SIRIUS_PRINT_PERFORMANCE");
        if (pp && gkvec_.comm().rank() == 0) {
            PROFILE_START("sirius::Beta_projectors_base::local_inner_aux");
        }

        const auto t1 = std::chrono::high_resolution_clock::now();
        linalg(ctx_.blas_linalg_t())
            .gemm('C', 'N', nbeta__, n__, num_gkvec_loc(), &linalg_const<F>::one(), beta_pw_coeffs_a_ptr__,
                  num_gkvec_loc(), phi__.pw_coeffs(ispn__).prime().at(phi__.preferred_memory_t(), 0, idx0__),
                  phi__.pw_coeffs(ispn__).prime().ld(), &linalg_const<F>::zero(),
                  beta_phi__.at(ctx_.preferred_memory_t()), beta_phi__.ld());

        if (pp && gkvec_.comm().rank() == 0) {
#ifdef SIRIUS_GPU
            if (ctx_.blas_linalg_t() == linalg_t::gpublas) {
                acc::sync_stream(stream_id(-1));
            }
#endif
            std::chrono::duration<double> t = std::chrono::high_resolution_clock::now() - t1;
            PROFILE_STOP("sirius::Beta_projectors_base::local_inner_aux");
            std::printf(
                "Beta_projectors_base::local_inner performance: %12.6f GFlops [m,n,k=%i %i %i, time=%f (sec)]\n",
                8e-9 * nbeta__ * n__ * num_gkvec_loc() / t.count(), nbeta__, n__, num_gkvec_loc(), t.count());
        }
    }

    template<typename F, std::enable_if_t<std::is_same<T, F>::value, bool> = true>
    void local_inner_aux(F* beta_pw_coeffs_a_ptr__, int nbeta__, Wave_functions<T>& phi__,
                         int ispn__, int idx0__, int n__, matrix<F>& beta_phi__) const
    {
        linalg(ctx_.blas_linalg_t())
            .gemm('C', 'N', nbeta__, n__, 2 * num_gkvec_loc(), &linalg_const<F>::two(), beta_pw_coeffs_a_ptr__,
                  2 * num_gkvec_loc(),
                  reinterpret_cast<F const*>(phi__.pw_coeffs(ispn__).prime().at(phi__.preferred_memory_t(), 0, idx0__)),
                  2 * phi__.pw_coeffs(ispn__).prime().ld(), &linalg_const<F>::zero(),
                  beta_phi__.at(ctx_.preferred_memory_t()), beta_phi__.ld());

        /* rank 0 has to do some extra work for Gamma-point case */
        if (gkvec_.comm().rank() == 0) {
            int incx{2 * num_gkvec_loc()};
            linalg_t la{linalg_t::none};
            /* both wave-functions and beta-projectors are on GPU */
            if (is_device_memory(ctx_.preferred_memory_t())) {
                la = linalg_t::gpublas;
            } else { /* wave-functions are on CPU but the beta-projectors are in the memory of main device */
                la = linalg_t::blas;
                switch (ctx_.processing_unit()) {
                    case device_t::GPU: {
                        beta_pw_coeffs_a_ptr__ =
                            reinterpret_cast<F*>(const_cast<std::complex<T>*>(&pw_coeffs_a_g0_(0)));
                        incx = 2;
                        break;
                    }
                    case device_t::CPU:
                        break;
                }
            }
            linalg(la).ger(
                nbeta__, n__, &linalg_const<F>::m_one(), beta_pw_coeffs_a_ptr__, incx,
                reinterpret_cast<F*>(phi__.pw_coeffs(ispn__).prime().at(phi__.preferred_memory_t(), 0, idx0__)),
                2 * phi__.pw_coeffs(ispn__).prime().ld(), beta_phi__.at(ctx_.preferred_memory_t()), beta_phi__.ld());
        }
    }

  public:
    Beta_projectors_base(Simulation_context& ctx__, Gvec const& gkvec__, std::vector<int> const& igk__, int N__);

    /// Calculate inner product between beta-projectors and wave-functions.
    /** The following is computed: <beta|phi> */
    template <typename F, typename = std::enable_if_t<std::is_same<T, real_type<F>>::value>>
    matrix<F> inner(int chunk__, Wave_functions<T>& phi__, int ispn__, int idx0__, int n__);

    /// Generate beta-projectors for a chunk of atoms.
    /** Beta-projectors are always generated and stored in the memory of a processing unit.
     *
     *  \param [in] ichunk Index of a chunk of atoms for which beta-projectors are generated.
     *  \param [in] j index of the component (up to 9 components are used for the strain derivative)
     */
    void generate(int ichunk__, int j__);

    void prepare();

    void dismiss();

    inline int num_gkvec_loc() const
    {
        return static_cast<int>(igk_.size());
    }

    inline int num_comp() const
    {
        return N_;
    }

    inline Unit_cell const& unit_cell() const
    {
        return ctx_.unit_cell();
    }

    std::complex<T>& pw_coeffs_t(int ig__, int n__, int j__)
    {
        return pw_coeffs_t_(ig__, n__, j__);
    }

    matrix<std::complex<T>> pw_coeffs_t(int j__)
    {
        return matrix<std::complex<T>>(&pw_coeffs_t_(0, 0, j__), num_gkvec_loc(), num_beta_t());
    }

    /// Plane wave coefficients of |beta> projectors for a chunk of atoms.
    matrix<std::complex<T>>& pw_coeffs_a()
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
};

} // namespace

#endif
