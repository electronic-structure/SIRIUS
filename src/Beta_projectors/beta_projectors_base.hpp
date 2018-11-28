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
#include "utils/env.hpp"

namespace sirius {

#ifdef __GPU
extern "C" void create_beta_gk_gpu(int                   num_atoms,
                                   int                   num_gkvec,
                                   int const*            beta_desc,
                                   double_complex const* beta_gk_t,
                                   double const*         gkvec,
                                   double const*         atom_pos,
                                   double_complex*       beta_gk);
#endif

enum beta_desc_idx 
{
    nbf      = 0,
    offset   = 1,
    offset_t = 2,
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
    void split_in_chunks()
    {
        auto& uc = ctx_.unit_cell();
        /* initial chunk size */
        int chunk_size = std::min(uc.num_atoms(), 256);
        /* maximum number of chunks */
        int num_chunks = uc.num_atoms() / chunk_size + std::min(1, uc.num_atoms() % chunk_size);
        /* final maximum chunk size */
        chunk_size = uc.num_atoms() / num_chunks + std::min(1, uc.num_atoms() % num_chunks);

        int offset_in_beta_gk{0};
        beta_chunks_ = std::vector<beta_chunk_t>(num_chunks);

        for (int ib = 0; ib < num_chunks; ib++) {
            /* number of atoms in this chunk */
            int na = std::min(uc.num_atoms(), (ib + 1) * chunk_size) - ib * chunk_size;
            beta_chunks_[ib].num_atoms_ = na;
            beta_chunks_[ib].desc_      = mdarray<int, 2>(4, na);
            beta_chunks_[ib].atom_pos_  = mdarray<double, 2>(3, na);

            int num_beta{0};
            for (int i = 0; i < na; i++) {
                /* global index of atom by local index and chunk */
                int ia     = ib * chunk_size + i;
                auto pos   = uc.atom(ia).position();
                auto& type = uc.atom(ia).type();
                /* atom fractional coordinates */
                for (int x: {0, 1, 2}) {
                    beta_chunks_[ib].atom_pos_(x, i) = pos[x];
                }
                /* number of beta functions for atom */
                beta_chunks_[ib].desc_(beta_desc_idx::nbf, i) = type.mt_basis_size();
                /* offset in beta_gk*/
                beta_chunks_[ib].desc_(beta_desc_idx::offset, i) = num_beta;
                /* offset in beta_gk_t */
                beta_chunks_[ib].desc_(beta_desc_idx::offset_t, i) = type.offset_lo();
                /* global index of atom */
                beta_chunks_[ib].desc_(beta_desc_idx::ia, i) = ia;

                num_beta += type.mt_basis_size();
            }
            /* number of beta-projectors in this chunk */
            beta_chunks_[ib].num_beta_ = num_beta;
            beta_chunks_[ib].offset_ = offset_in_beta_gk;
            offset_in_beta_gk += num_beta;

            if (ctx_.processing_unit() == GPU) {
                beta_chunks_[ib].desc_.allocate(memory_t::device);
                beta_chunks_[ib].desc_.template copy<memory_t::host, memory_t::device>();

                beta_chunks_[ib].atom_pos_.allocate(memory_t::device);
                beta_chunks_[ib].atom_pos_.template copy<memory_t::host, memory_t::device>();
            }
        }

        max_num_beta_ = 0;
        for (auto& e: beta_chunks_) {
            max_num_beta_ = std::max(max_num_beta_, e.num_beta_);
        }

        num_beta_t_ = 0;
        for (int iat = 0; iat < uc.num_atom_types(); iat++) {
            num_beta_t_ += uc.atom_type(iat).mt_lo_basis_size();
        }
    }

    template <typename T>
    inline void local_inner_aux(T* beta_pw_coeffs_a_ptr__, int nbeta__, Wave_functions& phi__, int ispn__, int idx0__,
                                int n__, matrix<T>& beta_phi__) const;

  public:
    Beta_projectors_base(Simulation_context&     ctx__,
                         Gvec const&             gkvec__,
                         std::vector<int> const& igk__,
                         int N__)
        : ctx_(ctx__)
        , gkvec_(gkvec__)
        , igk_(igk__)
        , N_(N__)
    {
        split_in_chunks();

        if (!num_beta_t()) {
            return;
        }

        /* allocate memory */
        pw_coeffs_t_ = mdarray<double_complex, 3>(num_gkvec_loc(), num_beta_t(), N__, memory_t::host, "pw_coeffs_t_");

        if (ctx_.processing_unit() == GPU) {
            gkvec_coord_ = mdarray<double, 2>(3, num_gkvec_loc());
            gkvec_coord_.allocate(memory_t::device);
            /* copy G+k vectors */
            for (int igk_loc = 0; igk_loc < num_gkvec_loc(); igk_loc++) {
                auto vgk = gkvec_.gkvec(igk_[igk_loc]);
                for (auto x: {0, 1, 2}) {
                    gkvec_coord_(x, igk_loc) = vgk[x];
                }
            }
            gkvec_coord_.copy<memory_t::host, memory_t::device>();
        }
    }

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
        return std::move(matrix<double_complex>(&pw_coeffs_t_(0, 0, j__), num_gkvec_loc(), num_beta_t()));
    }

    /// Plane wave coefficients of |beta> projectors for a chunk of atoms.
    matrix<double_complex>& pw_coeffs_a()
    {
        return pw_coeffs_a_;
    }

    /// Calculate inner product between beta-projectors and wave-functions.
    /** The following is computed: <beta|phi> */
    template <typename T>
    inline matrix<T> inner(int             chunk__,
                           Wave_functions& phi__,
                           int             ispn__,
                           int             idx0__,
                           int             n__)
    {
        PROFILE("sirius::Beta_projectors_base::inner");

        assert(num_gkvec_loc() == phi__.pw_coeffs(ispn__).num_rows_loc());

        int nbeta = chunk(chunk__).num_beta_;

        //static_assert(std::is_same<T, double_complex>::value || std::is_same<T, double>::value, "wrong type");

        matrix<T> beta_phi;

        /* location of the beta-projectors is always on the memory of the processing unit being used */
        T* pw_coeffs_a_ptr{nullptr};
        switch (ctx_.processing_unit()) {
            case device_t::CPU: {
                beta_phi = matrix<T>(ctx_.mem_pool(ctx_.host_memory_t()), nbeta, n__);
                pw_coeffs_a_ptr = reinterpret_cast<T*>(pw_coeffs_a().at(memory_t::host));
                break;
            }
            case device_t::GPU: {
                beta_phi = matrix<T>(ctx_.mem_pool(ctx_.host_memory_t()), ctx_.mem_pool(memory_t::device),
                                     nbeta, n__);
                pw_coeffs_a_ptr = reinterpret_cast<T*>(pw_coeffs_a().at(memory_t::device));
                break;
            }
        }

        local_inner_aux<T>(pw_coeffs_a_ptr, nbeta, phi__, ispn__, idx0__, n__, beta_phi);

        if (gkvec_.comm().size() > 1) {
            utils::timer t1("sirius::Beta_projectors_base::inner|comm");
            /* copy to host for MPI reduction */
            if (is_device_memory(ctx_.preferred_memory_t())) {
                beta_phi.template copy<memory_t::device, memory_t::host>();
            }
            /* MPI reduction on the host */
            gkvec_.comm().allreduce(beta_phi.template at(memory_t::host), static_cast<int>(beta_phi.size()));
        }

        switch (ctx_.processing_unit()) {
            case device_t::GPU: {
                /* copy back to device */
                if ((gkvec_.comm().size() > 1 && is_device_memory(ctx_.preferred_memory_t())) ||
                    is_host_memory(ctx_.preferred_memory_t())) {
                    beta_phi.template copy<memory_t::host, memory_t::device>();
                }
                if (is_device_memory(ctx_.preferred_memory_t())) {
                    beta_phi.template copy<memory_t::device, memory_t::host>();
                }
                break;
            }
            case device_t::CPU: break;
        }

        return std::move(beta_phi);
    }

    /// Generate beta-projectors for a chunk of atoms.
    /** Beta-projectors are always generated and stored in the memory of a processing unit.
     *
     *  \param [in] ichunk Index of a chunk of atoms for which beta-projectors are generated.
     *  \param [in] j index of the component (up to 9 components are used for the strain derivative) 
     */
    void generate(int ichunk__, int j__)
    {
        PROFILE("sirius::Beta_projectors_base::generate");

        switch (ctx_.processing_unit()) {
            case device_t::CPU: {
                #pragma omp for
                for (int i = 0; i < chunk(ichunk__).num_atoms_; i++) {
                    int ia = chunk(ichunk__).desc_(beta_desc_idx::ia, i);

                    double phase = twopi * dot(gkvec_.vk(), ctx_.unit_cell().atom(ia).position());
                    double_complex phase_k = std::exp(double_complex(0.0, phase));

                    std::vector<double_complex> phase_gk(num_gkvec_loc());
                    for (int igk_loc = 0; igk_loc < num_gkvec_loc(); igk_loc++) {
                        auto G = gkvec_.gvec(igk_[igk_loc]);
                        /* total phase e^{i(G+k)r_{\alpha}} */
                        phase_gk[igk_loc] = std::conj(ctx_.gvec_phase_factor(G, ia) * phase_k);
                    }
                    for (int xi = 0; xi < chunk(ichunk__).desc_(beta_desc_idx::nbf, i); xi++) {
                        for (int igk_loc = 0; igk_loc < num_gkvec_loc(); igk_loc++) {
                            pw_coeffs_a_(igk_loc, chunk(ichunk__).desc_(beta_desc_idx::offset, i) + xi) = 
                                pw_coeffs_t_(igk_loc, chunk(ichunk__).desc_(beta_desc_idx::offset_t, i) + xi, j__) * phase_gk[igk_loc];
                        }
                    }
                }
                break;
            }
            case device_t::GPU: {
#ifdef __GPU
                auto& desc = chunk(ichunk__).desc_;
                create_beta_gk_gpu(chunk(ichunk__).num_atoms_,
                                   num_gkvec_loc(),
                                   desc.template at<GPU>(),
                                   pw_coeffs_t_.template at<GPU>(0, 0, j__),
                                   gkvec_coord_.template at<GPU>(),
                                   chunk(ichunk__).atom_pos_.template at<GPU>(),
                                   pw_coeffs_a().template at<GPU>());
#endif
                /* wave-functions are on CPU but the beta-projectors are on GPU */
                if (gkvec_.comm().rank() == 0 && is_host_memory(ctx_.preferred_memory_t())) {
                    /* make beta-projectors for G=0 on the CPU */
                    #pragma omp for schedule(static)
                    for (int i = 0; i < chunk(ichunk__).num_atoms_; i++) {
                        for (int xi = 0; xi < chunk(ichunk__).desc_(beta_desc_idx::nbf, i); xi++) {
                            pw_coeffs_a_g0_(chunk(ichunk__).desc_(beta_desc_idx::offset, i) + xi) =
                                pw_coeffs_t_(0, chunk(ichunk__).desc_(beta_desc_idx::offset_t, i) + xi, j__);
                        }
                    }
                }
                break;
            }
        }
    }

    void prepare()
    {
        PROFILE("sirius::Beta_projectors_base::prepare");

        switch (ctx_.processing_unit()) {
            case device_t::CPU: {
                pw_coeffs_a_ = matrix<double_complex>(ctx_.mem_pool(ctx_.host_memory_t()),
                                                      num_gkvec_loc(), max_num_beta());
                pw_coeffs_a_g0_ = mdarray<double_complex, 1>(ctx_.mem_pool(memory_t::host), max_num_beta());
                break;
            }
            case device_t::GPU: {
                //pw_coeffs_a_ = matrix<double_complex>(ctx_.mem_pool(ctx_.host_memory_t()),
                //                                      ctx_.mem_pool(memory_t::device),
                //                                      num_gkvec_loc(), max_num_beta());
                pw_coeffs_a_ = matrix<double_complex>(nullptr,
                                                      ctx_.mem_pool(memory_t::device),
                                                      num_gkvec_loc(), max_num_beta());
                pw_coeffs_a_g0_ = mdarray<double_complex, 1>(ctx_.mem_pool(memory_t::host),
                                                             ctx_.mem_pool(memory_t::device), max_num_beta());
                break;
            }
        }

        if (ctx_.processing_unit() == device_t::GPU && reallocate_pw_coeffs_t_on_gpu_) {
            pw_coeffs_t_.allocate(memory_t::device);
            pw_coeffs_t_.template copy<memory_t::host, memory_t::device>();
        }
    }

    void dismiss()
    {
        PROFILE("sirius::Beta_projectors_base::dismiss");

        if (ctx_.processing_unit() == GPU && reallocate_pw_coeffs_t_on_gpu_) {
            pw_coeffs_t_.deallocate(memory_t::device);
        }
        pw_coeffs_a_.deallocate(memory_t::device);
        pw_coeffs_a_g0_.deallocate(memory_t::device);
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

template<>
inline void Beta_projectors_base::local_inner_aux<double_complex>(double_complex* beta_pw_coeffs_a_ptr__, int nbeta__,
                                                                  Wave_functions& phi__, int ispn__, int idx0__, int n__,
                                                                  matrix<double_complex>& beta_phi__) const
{
    utils::timer t1("sirius::Beta_projectors_base::local_inner_aux");
    linalg2(ctx_.blas_linalg_t()).gemm('C', 'N', nbeta__, n__, num_gkvec_loc(),
            &linalg_const<double_complex>::one(),
            beta_pw_coeffs_a_ptr__,
            num_gkvec_loc(),
            phi__.pw_coeffs(ispn__).prime().at(ctx_.preferred_memory_t(), 0, idx0__),
            phi__.pw_coeffs(ispn__).prime().ld(),
            &linalg_const<double_complex>::zero(),
            beta_phi__.at(ctx_.preferred_memory_t()), beta_phi__.ld());

    auto pp = utils::get_env<int>("SIRIUS_PRINT_PERFORMANCE");
    if (pp && gkvec_.comm().rank() == 0) {
#ifdef __GPU
        if (ctx_.blas_linalg_t() == linalg_t::cublas) {
            acc::sync_stream(-1);
        }
#endif
        double t = t1.stop();
        printf("Beta_projectors_base::local_inner performance: %12.6f GFlops [m,n,k=%i %i %i, time=%f (sec)]\n",
               8e-9 * nbeta__ * n__ * num_gkvec_loc() / t, nbeta__, n__, num_gkvec_loc(), t);
    }
}

template<>
inline void Beta_projectors_base::local_inner_aux<double>(double* beta_pw_coeffs_a_ptr__, int nbeta__,
                                                          Wave_functions& phi__, int ispn__, int idx0__, int n__,
                                                          matrix<double>& beta_phi__) const
{
    linalg2(ctx_.blas_linalg_t()).gemm('C', 'N', nbeta__, n__, 2 * num_gkvec_loc(),
            &linalg_const<double>::two(),
            beta_pw_coeffs_a_ptr__,
            2 * num_gkvec_loc(),
            reinterpret_cast<double const*>(phi__.pw_coeffs(ispn__).prime().at(ctx_.preferred_memory_t(), 0, idx0__)),
            2 * phi__.pw_coeffs(ispn__).prime().ld(),
            &linalg_const<double>::zero(),
            beta_phi__.at(ctx_.preferred_memory_t()), beta_phi__.ld());

    /* rank 0 has to do some extra work for Gamma-point case */
    if (gkvec_.comm().rank() == 0) {
        int incx{2 * num_gkvec_loc()};
        linalg_t la{linalg_t::none};
        /* both wave-functions and beta-projectors are on GPU */
        if (is_device_memory(ctx_.preferred_memory_t())) {
            la = linalg_t::cublas;
        } else { /* wave-functions are on CPU but the beta-projectors are in the memory of main device */
            la = linalg_t::blas;
            switch (ctx_.processing_unit()) {
                case device_t::GPU: {
                    beta_pw_coeffs_a_ptr__ = reinterpret_cast<double*>(const_cast<double_complex*>(&pw_coeffs_a_g0_(0)));
                    incx = 1;
                    break;
                }
                case device_t::CPU: break;
            }
        }
        linalg2(la).ger(nbeta__, n__,
                        &linalg_const<double>::m_one(),
                        beta_pw_coeffs_a_ptr__, incx,
                        reinterpret_cast<double*>(phi__.pw_coeffs(ispn__).prime().at(ctx_.preferred_memory_t(), 0, idx0__)), 
                        2 * phi__.pw_coeffs(ispn__).prime().ld(),
                        beta_phi__.at(ctx_.preferred_memory_t()), beta_phi__.ld());
    }
}

} // namespace

#endif
