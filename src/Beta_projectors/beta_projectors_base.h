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

/// Base class for beta-projectors, gradient of beta-projectors and strain derivatives of beta-projectors.
template <int N>
class Beta_projectors_base
{
  protected:

    Simulation_context& ctx_;

    Gvec const& gkvec_;

    mdarray<double, 2> gkvec_coord_;
    
    int num_gkvec_loc_;

    int lmax_beta_;

    /// Phase-factor independent coefficients of |beta> functions for atom types.
    std::array<matrix<double_complex>, N> pw_coeffs_t_;

    /// A buffer for plane-wave coefficients of beta projectors, shared between instances of Beta_projectors_base class.
    static mdarray<double_complex, 1>& pw_coeffs_a_shared(size_t size__, memory_t mem_type__)
    {
        static mdarray<double_complex, 1> a;
        if (a.size() < size__) {
            a = mdarray<double_complex, 1>(size__, mem_type__, "pw_coeffs_a_shared");
        }
        return a;
    }

    /// A buffer for <beta|phi> product, shared between instances of Beta_projectors_base class.
    /** Stored as double to handle both gamma- and general k-point cases */
    static mdarray<double, 1>& beta_phi_shared(size_t size__, memory_t mem_type__)
    {
        static mdarray<double, 1> a;
        if (a.size() < size__) {
            a = mdarray<double, 1>(size__, mem_type__, "beta_phi_shared");
        }
        return a;
    }

  public:
    Beta_projectors_base(Simulation_context& ctx__,
                         Gvec         const& gkvec__)
        : ctx_(ctx__)
        , gkvec_(gkvec__)
        , lmax_beta_(ctx_.unit_cell().lmax())
    {
        num_gkvec_loc_ = gkvec_.count();

        auto& bchunk = ctx_.beta_projector_chunks();
        if (!bchunk.num_beta_t()) {
            return;
        }

        /* allocate memory */
        for (int i = 0; i < N; i++) {
            //pw_coeffs_t_[i] = matrix<double_complex>(num_gkvec_loc(), bchunk.num_beta_t(), ctx_.dual_memory_t(), "pw_coeffs_t_");
            pw_coeffs_t_[i] = matrix<double_complex>(num_gkvec_loc(), bchunk.num_beta_t(), memory_t::host, "pw_coeffs_t_");
        }

        auto& comm = gkvec_.comm();

        if (ctx_.processing_unit() == GPU) {
            gkvec_coord_ = mdarray<double, 2>(3, num_gkvec_loc(), ctx__.dual_memory_t());
            /* copy G+k vectors */
            for (int igk_loc = 0; igk_loc < num_gkvec_loc_; igk_loc++) {
                int igk  = gkvec_.gvec_offset(comm.rank()) + igk_loc;
                auto vgk = gkvec_.gkvec(igk);
                for (auto x: {0, 1, 2}) {
                    gkvec_coord_(x, igk_loc) = vgk[x];
                }
            }
            gkvec_coord_.copy<memory_t::host, memory_t::device>();
        }
    }
    ~Beta_projectors_base()
    {
        //#ifdef __GPU
        pw_coeffs_a_shared(0, memory_t::none) = mdarray<double_complex, 1>(); //.deallocate_on_device();
        beta_phi_shared(0, memory_t::none) = mdarray<double, 1>(); //.deallocate_on_device();
        //#endif
    }

    inline int num_gkvec_loc() const
    {
        return num_gkvec_loc_;
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
        auto& bchunk = ctx_.beta_projector_chunks();
        auto& buf = pw_coeffs_a_shared(num_gkvec_loc() * bchunk.max_num_beta(), ctx_.dual_memory_t());

        static mdarray<double_complex, 2> pw_coeffs_a_;
        switch (ctx_.processing_unit()) {
            case CPU: {
                pw_coeffs_a_ = mdarray<double_complex, 2>(buf.template at<CPU>(), num_gkvec_loc(), bchunk.max_num_beta(), "pw_coeffs_a_");
                break;
            } 
            case GPU: {
                pw_coeffs_a_ = mdarray<double_complex, 2>(buf.template at<CPU>(), buf.template at<GPU>(), num_gkvec_loc(), bchunk.max_num_beta(), "pw_coeffs_a_");
                break;
           }
        }
        return pw_coeffs_a_;
    }

    /// Calculate inner product between beta-projectors and wave-functions.
    /** The following is computed: <beta|phi> */
    template <typename T>
    inline matrix<T> inner(int             chunk__,
                           wave_functions& phi__,
                           int             idx0__,
                           int             n__)
    {
        PROFILE("sirius::Beta_projectors_base::inner");

        assert(num_gkvec_loc_ == phi__.pw_coeffs().num_rows_loc());
        
        auto& bp_chunks = ctx_.beta_projector_chunks();

        int nbeta = bp_chunks(chunk__).num_beta_;

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
                    linalg<CPU>::gemm(2, 0, nbeta, n__, num_gkvec_loc_,
                                      pw_coeffs_a().template at<CPU>(), num_gkvec_loc_,
                                      phi__.pw_coeffs().prime().at<CPU>(0, idx0__), phi__.pw_coeffs().prime().ld(),
                                      reinterpret_cast<double_complex*>(beta_phi.template at<CPU>()), nbeta);
                    break;
                }
                case GPU: {
                    #ifdef __GPU
                    linalg<GPU>::gemm(2, 0, nbeta, n__, num_gkvec_loc_,
                                      pw_coeffs_a().template at<GPU>(), num_gkvec_loc_,
                                      phi__.pw_coeffs().prime().at<GPU>(0, idx0__), phi__.pw_coeffs().prime().ld(),
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
                    linalg<CPU>::gemm(2, 0, nbeta, n__, 2 * num_gkvec_loc_,
                                      a,
                                      reinterpret_cast<double*>(pw_coeffs_a().template at<CPU>()), 2 * num_gkvec_loc_,
                                      reinterpret_cast<double*>(phi__.pw_coeffs().prime().at<CPU>(0, idx0__)), 2 * phi__.pw_coeffs().prime().ld(),
                                      b,
                                      reinterpret_cast<double*>(beta_phi.template at<CPU>()), nbeta);

                    if (gkvec_.comm().rank() == 0) {
                        /* subtract one extra G=0 contribution */
                        linalg<CPU>::ger(nbeta, n__, a1,
                                         reinterpret_cast<double*>(pw_coeffs_a().template at<CPU>()), 2 * num_gkvec_loc_,
                                         reinterpret_cast<double*>(phi__.pw_coeffs().prime().at<CPU>(0, idx0__)), 2 * phi__.pw_coeffs().prime().ld(),
                                         reinterpret_cast<double*>(beta_phi.template at<CPU>()), nbeta);
                    }
                    break;
                }
                case GPU: {
                    #ifdef __GPU
                    linalg<GPU>::gemm(2, 0, nbeta, n__, 2 * num_gkvec_loc_,
                                      &a,
                                      reinterpret_cast<double*>(pw_coeffs_a().template at<GPU>()), 2 * num_gkvec_loc_,
                                      reinterpret_cast<double*>(phi__.pw_coeffs().prime().at<GPU>(0, idx0__)), 2 * phi__.pw_coeffs().prime().ld(),
                                      &b,
                                      reinterpret_cast<double*>(beta_phi.template at<GPU>()), nbeta);

                    if (gkvec_.comm().rank() == 0) {
                        /* subtract one extra G=0 contribution */
                        linalg<GPU>::ger(nbeta, n__, &a1, 
                                         reinterpret_cast<double*>(pw_coeffs_a().template at<GPU>()), 2 * num_gkvec_loc_,
                                         reinterpret_cast<double*>(phi__.pw_coeffs().prime().at<GPU>(0, idx0__)), 2 * phi__.pw_coeffs().prime().ld(),
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

        auto& bchunk = ctx_.beta_projector_chunks();

        auto& pw_coeffs = pw_coeffs_a();

        switch (ctx_.processing_unit()) {
            case CPU: {
                #pragma omp for
                for (int i = 0; i < bchunk(ichunk__).num_atoms_; i++) {
                    int ia = bchunk(ichunk__).desc_(beta_desc_idx::ia, i);

                    double phase = twopi * dot(gkvec_.vk(), ctx_.unit_cell().atom(ia).position());
                    double_complex phase_k = std::exp(double_complex(0.0, phase));

                    std::vector<double_complex> phase_gk(num_gkvec_loc());
                    for (int igk_loc = 0; igk_loc < num_gkvec_loc_; igk_loc++) {
                        int igk = gkvec_.offset() + igk_loc;
                        auto G = gkvec_.gvec(igk);
                        /* total phase e^{i(G+k)r_{\alpha}} */
                        phase_gk[igk_loc] = std::conj(ctx_.gvec_phase_factor(G, ia) * phase_k);
                    }
                    for (int xi = 0; xi < bchunk(ichunk__).desc_(beta_desc_idx::nbf, i); xi++) {
                        for (int igk_loc = 0; igk_loc < num_gkvec_loc(); igk_loc++) {
                            pw_coeffs(igk_loc, bchunk(ichunk__).desc_(beta_desc_idx::offset, i) + xi) = 
                                pw_coeffs_t_[j__](igk_loc, bchunk(ichunk__).desc_(beta_desc_idx::offset_t, i) + xi) * phase_gk[igk_loc];
                        }
                    }
                }
                break;
            }
            case GPU: {
                #ifdef __GPU
                auto& desc = bchunk(ichunk__).desc_;
                create_beta_gk_gpu(bchunk(ichunk__).num_atoms_,
                                   num_gkvec_loc_,
                                   desc.at<GPU>(),
                                   pw_coeffs_t_[j__].template at<GPU>(),
                                   gkvec_coord_.at<GPU>(),
                                   bchunk(ichunk__).atom_pos_.at<GPU>(),
                                   pw_coeffs.template at<GPU>());
                #endif
                break;
            }
        }
    }

    void prepare()
    {
        PROFILE("sirius::Beta_projectors_base::prepare");

        if (ctx_.processing_unit() == GPU) {
            for (int i = 0; i < N; i++) {
                pw_coeffs_t_[i].allocate(memory_t::device);
                pw_coeffs_t_[i].template copy<memory_t::host, memory_t::device>();
            }
        }
    }

    void dismiss()
    {
        PROFILE("sirius::Beta_projectors_base::dismiss");

        #ifdef __GPU
        if (ctx_.processing_unit() == GPU) {
            for (int i = 0; i < N; i++) {
                pw_coeffs_t_[i].deallocate_on_device();
            }
        }
        #endif
    }
};

} // namespace

#endif
