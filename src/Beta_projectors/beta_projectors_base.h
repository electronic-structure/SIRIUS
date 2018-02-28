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

/** \file beta_projectors_base.h
 *
 *  \brief Contains declaration and implementation of sirius::Beta_projectors_base class.
 */

#ifndef __BETA_PROJECTORS_BASE_H__
#define __BETA_PROJECTORS_BASE_H__

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
template <int N>
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
    
    /// Phase-factor independent coefficients of |beta> functions for atom types.
    std::array<matrix<double_complex>, N> pw_coeffs_t_;

    bool reallocate_pw_coeffs_t_on_gpu_{true};

    matrix<double_complex> pw_coeffs_a_;

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
                beta_chunks_[ib].desc_.copy<memory_t::host, memory_t::device>();

                beta_chunks_[ib].atom_pos_.allocate(memory_t::device);
                beta_chunks_[ib].atom_pos_.copy<memory_t::host, memory_t::device>();
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

    /// A buffer for <beta|phi> product, shared between instances of Beta_projectors_base class.
    /** Stored as double to handle both gamma- and general k-point cases */
    static mdarray<double, 1>& beta_phi_shared(size_t size__, memory_t mem_type__)
    {
        static mdarray<double, 1> a;
        /* reallocate buffer */
        if (a.size() < size__) {
            a = mdarray<double, 1>(size__, mem_type__, "beta_phi_shared");
        }
        return a;
    }

    /// A buffer for beta projectors for a chunk of atoms.
    static mdarray<double_complex, 1>& pw_coeffs_a_shared(size_t size__, memory_t mem_type__)
    {
        static mdarray<double_complex, 1> a;
        /* reallocate buffer */
        if (a.size() < size__) {
            a = mdarray<double_complex, 1>(size__, mem_type__, "pw_coeffs_a_shared");
        }
        return a;
    }

  public:
    Beta_projectors_base(Simulation_context&     ctx__,
                         Gvec const&             gkvec__,
                         std::vector<int> const& igk__)
        : ctx_(ctx__)
        , gkvec_(gkvec__)
        , igk_(igk__)
    {
        split_in_chunks();

        if (!num_beta_t()) {
            return;
        }

        /* allocate memory */
        for (int i = 0; i < N; i++) {
            pw_coeffs_t_[i] = matrix<double_complex>(num_gkvec_loc(), num_beta_t(), memory_t::host, "pw_coeffs_t_");
        }

        if (ctx_.processing_unit() == GPU) {
            gkvec_coord_ = mdarray<double, 2>(3, num_gkvec_loc(), ctx__.dual_memory_t());
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

    ~Beta_projectors_base()
    {
        beta_phi_shared(0, memory_t::none) = mdarray<double, 1>();
    }

    inline int num_gkvec_loc() const
    {
        return static_cast<int>(igk_.size());
    }

    inline Unit_cell const& unit_cell() const
    {
        return ctx_.unit_cell();
    }

    matrix<double_complex>& pw_coeffs_t(int i__)
    {
        return pw_coeffs_t_[i__];
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

        static_assert(std::is_same<T, double_complex>::value || std::is_same<T, double>::value, "wrong type");

        int tsz = std::is_same<T, double_complex>::value ? 2 : 1;

        auto& buf = beta_phi_shared(tsz * nbeta * n__, ctx_.dual_memory_t());

        matrix<T> beta_phi;

        switch (ctx_.processing_unit()) {
            case CPU: {
                beta_phi = matrix<T>(reinterpret_cast<T*>(buf.template at<CPU>()), nbeta, n__);
                break;
            }
            case GPU: {
                beta_phi = matrix<T>(reinterpret_cast<T*>(buf.template at<CPU>()), reinterpret_cast<T*>(buf.template at<GPU>()), nbeta, n__);
                break;
            }
        }

        if (std::is_same<T, double_complex>::value) {
            switch (ctx_.processing_unit()) {
                case CPU: {
                    /* compute <beta|phi> */
                    linalg<CPU>::gemm(2, 0, nbeta, n__, num_gkvec_loc(),
                                      pw_coeffs_a().template at<CPU>(), num_gkvec_loc(),
                                      phi__.pw_coeffs(ispn__).prime().at<CPU>(0, idx0__), phi__.pw_coeffs(ispn__).prime().ld(),
                                      reinterpret_cast<double_complex*>(beta_phi.template at<CPU>()), nbeta);
                    break;
                }
                case GPU: {
                    #ifdef __GPU
                    linalg<GPU>::gemm(2, 0, nbeta, n__, num_gkvec_loc(),
                                      pw_coeffs_a().template at<GPU>(), num_gkvec_loc(),
                                      phi__.pw_coeffs(ispn__).prime().at<GPU>(0, idx0__), phi__.pw_coeffs(ispn__).prime().ld(),
                                      reinterpret_cast<double_complex*>(beta_phi.template at<GPU>()), nbeta);
                    beta_phi.template copy<memory_t::device, memory_t::host>();
                    #else
                    TERMINATE_NO_GPU
                    #endif
                    break;
                }
            }
        }
        if (std::is_same<T, double>::value) {
            double a{2};
            double a1{-1};
            double b{0};

            switch (ctx_.processing_unit()) {
                case CPU: {
                    /* compute <beta|phi> */
                    linalg<CPU>::gemm(2, 0, nbeta, n__, 2 * num_gkvec_loc(),
                                      a,
                                      reinterpret_cast<double*>(pw_coeffs_a().template at<CPU>()), 2 * num_gkvec_loc(),
                                      reinterpret_cast<double*>(phi__.pw_coeffs(ispn__).prime().at<CPU>(0, idx0__)),
                                      2 * phi__.pw_coeffs(ispn__).prime().ld(),
                                      b,
                                      reinterpret_cast<double*>(beta_phi.template at<CPU>()), nbeta);

                    if (gkvec_.comm().rank() == 0) {
                        /* subtract one extra G=0 contribution */
                        linalg<CPU>::ger(nbeta, n__, a1,
                                         reinterpret_cast<double*>(pw_coeffs_a().template at<CPU>()), 2 * num_gkvec_loc(),
                                         reinterpret_cast<double*>(phi__.pw_coeffs(ispn__).prime().at<CPU>(0, idx0__)), 
                                         2 * phi__.pw_coeffs(ispn__).prime().ld(),
                                         reinterpret_cast<double*>(beta_phi.template at<CPU>()), nbeta);
                    }
                    break;
                }
                case GPU: {
                    #ifdef __GPU
                    linalg<GPU>::gemm(2, 0, nbeta, n__, 2 * num_gkvec_loc(),
                                      &a,
                                      reinterpret_cast<double*>(pw_coeffs_a().template at<GPU>()), 2 * num_gkvec_loc(),
                                      reinterpret_cast<double*>(phi__.pw_coeffs(ispn__).prime().at<GPU>(0, idx0__)),
                                      2 * phi__.pw_coeffs(ispn__).prime().ld(),
                                      &b,
                                      reinterpret_cast<double*>(beta_phi.template at<GPU>()), nbeta);

                    if (gkvec_.comm().rank() == 0) {
                        /* subtract one extra G=0 contribution */
                        linalg<GPU>::ger(nbeta, n__, &a1, 
                                         reinterpret_cast<double*>(pw_coeffs_a().template at<GPU>()), 2 * num_gkvec_loc(),
                                         reinterpret_cast<double*>(phi__.pw_coeffs(ispn__).prime().template at<GPU>(0, idx0__)),
                                         2 * phi__.pw_coeffs(ispn__).prime().ld(),
                                         reinterpret_cast<double*>(beta_phi.template at<GPU>()), nbeta);
                    }
                    beta_phi.template copy<memory_t::device, memory_t::host>();
                    #else
                    TERMINATE_NO_GPU
                    #endif
                    break;
                }
            }
        }
        
        gkvec_.comm().allreduce(beta_phi.template at<CPU>(), static_cast<int>(beta_phi.size()));

        if (ctx_.processing_unit() == GPU) {
            beta_phi.template copy<memory_t::host, memory_t::device>();
        }

        return std::move(beta_phi);
    }

    /// Generate beta-projectors for a chunk of atoms.
    void generate(int ichunk__, int j__)
    {
        PROFILE("sirius::Beta_projectors_base::generate");

        auto& pw_coeffs = pw_coeffs_a();

        switch (ctx_.processing_unit()) {
            case CPU: {
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
                            pw_coeffs(igk_loc, chunk(ichunk__).desc_(beta_desc_idx::offset, i) + xi) = 
                                pw_coeffs_t_[j__](igk_loc, chunk(ichunk__).desc_(beta_desc_idx::offset_t, i) + xi) * phase_gk[igk_loc];
                        }
                    }
                }
                break;
            }
            case GPU: {
                #ifdef __GPU
                auto& desc = chunk(ichunk__).desc_;
                create_beta_gk_gpu(chunk(ichunk__).num_atoms_,
                                   num_gkvec_loc(),
                                   desc.template at<GPU>(),
                                   pw_coeffs_t_[j__].template at<GPU>(),
                                   gkvec_coord_.template at<GPU>(),
                                   chunk(ichunk__).atom_pos_.template at<GPU>(),
                                   pw_coeffs.template at<GPU>());
                #endif
                break;
            }
        }
    }

    void prepare()
    {
        PROFILE("sirius::Beta_projectors_base::prepare");

        auto& buf = pw_coeffs_a_shared(num_gkvec_loc() * max_num_beta(), ctx_.dual_memory_t());

        switch (ctx_.processing_unit()) {
            case CPU: {
                pw_coeffs_a_ = matrix<double_complex>(buf.template at<CPU>(), num_gkvec_loc(), max_num_beta());
                break;
            }
            case GPU: {
                pw_coeffs_a_ = matrix<double_complex>(buf.template at<CPU>(), buf.template at<GPU>(), num_gkvec_loc(),
                                                      max_num_beta());
                break;
            }
        }

        if (ctx_.processing_unit() == GPU && reallocate_pw_coeffs_t_on_gpu_) {
            for (int i = 0; i < N; i++) {
                pw_coeffs_t_[i].allocate(memory_t::device);
                pw_coeffs_t_[i].template copy<memory_t::host, memory_t::device>();
            }
        }
    }

    void dismiss()
    {
        PROFILE("sirius::Beta_projectors_base::dismiss");

        if (ctx_.processing_unit() == GPU && reallocate_pw_coeffs_t_on_gpu_) {
            for (int i = 0; i < N; i++) {
                pw_coeffs_t_[i].deallocate(memory_t::device);
            }
        }
    }

    static void cleanup()
    {
        beta_phi_shared(0, memory_t::host | memory_t::device) = mdarray<double, 1>();
        pw_coeffs_a_shared(0, memory_t::host|memory_t::device) = mdarray<double_complex, 1>();
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
