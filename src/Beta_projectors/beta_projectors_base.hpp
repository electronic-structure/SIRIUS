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

#include "simulation_context.hpp"
#include "SDDK/wave_functions.hpp"

namespace sirius {

#if defined(__GPU)
extern "C" void create_beta_gk_gpu(int                   num_atoms,
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
class Beta_projectors_base
{
  protected:

    Simulation_context& ctx_;

    /// List of G+k vectors.
    Gvec const& gkvec_;

    /// Mapping between local and global G+k vector index.
    std::vector<int> const& igk_;

    /// Coordinates of G+k vectors used by GPU kernel.
    mdarray<double, 2> gkvec_coord_;

    /// Number of different components: 1 for beta-projectors, 3 for gradient, 9 for strain derivatives.
    int N_;

    /// Phase-factor independent coefficients of |beta> functions for atom types.
    mdarray<double_complex, 3> pw_coeffs_t_;

    bool reallocate_pw_coeffs_t_on_gpu_{true};

    /// Set of beta PW coefficients for a chunk of atoms.
    matrix<double_complex> pw_coeffs_a_;

    mdarray<double_complex, 1> pw_coeffs_a_g0_;

    std::vector<beta_chunk_t> beta_chunks_;

    int max_num_beta_;

    /// Total number of beta-projectors among atom types.
    int num_beta_t_;

    /// Split beta-projectors into chunks.
    void split_in_chunks();

    template <typename T>
    void local_inner_aux(T* beta_pw_coeffs_a_ptr__, int nbeta__, Wave_functions& phi__, int ispn__, int idx0__,
                         int n__, matrix<T>& beta_phi__) const;

  public:
    Beta_projectors_base(Simulation_context& ctx__, Gvec const& gkvec__, std::vector<int> const& igk__, int N__);

    /// Calculate inner product between beta-projectors and wave-functions.
    /** The following is computed: <beta|phi> */
    template <typename T>
    matrix<T> inner(int chunk__, Wave_functions& phi__, int ispn__, int idx0__, int n__);

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

    double_complex& pw_coeffs_t(int ig__, int n__, int j__)
    {
        return pw_coeffs_t_(ig__, n__, j__);
    }

    matrix<double_complex> pw_coeffs_t(int j__)
    {
        return matrix<double_complex>(&pw_coeffs_t_(0, 0, j__), num_gkvec_loc(), num_beta_t());
    }

    /// Plane wave coefficients of |beta> projectors for a chunk of atoms.
    matrix<double_complex>& pw_coeffs_a()
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
