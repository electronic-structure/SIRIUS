#ifndef __BETA_PROJECTORS_BASE_H__
#define __BETA_PROJECTORS_BASE_H__

namespace sirius {

template <int N>
class Beta_projectors_base
{
  protected:

    Simulation_context& ctx_;

    Gvec const& gkvec_;
    
    int num_gkvec_loc_;

    int lmax_beta_;

    /// Inner product between beta-projectors and wave-functions.
    /** Stored as double to handle both gamma- and general k-point cases */
    mdarray<double, 1> beta_phi_;

    /// Phase-factor independent coefficients of |beta> functions for atom types.
    std::array<matrix<double_complex>, N> pw_coeffs_t_;

    /// Phase-factor independent coefficients of |beta> functions for a chunk of atoms.
    std::array<matrix<double_complex>, N> pw_coeffs_a_;

  public:
    Beta_projectors_base(Simulation_context& ctx__,
                         Gvec         const& gkvec__)
        : ctx_(ctx__)
        , gkvec_(gkvec__)
        , lmax_beta_(ctx_.unit_cell().lmax())
    {
        num_gkvec_loc_ = gkvec_.gvec_count(gkvec_.comm().rank());
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

    matrix<double_complex>& pw_coeffs_a(int i__)
    {
        return pw_coeffs_a_[i__];
    }

    /// Calculate inner product between beta-projectors and wave-functions.
    /** The following is computed: <beta|phi> */
    template <typename T>
    inline matrix<T> inner(int             chunk__,
                           wave_functions& phi__,
                           int             idx0__,
                           int             n__,
                           int             idx_bp__)
    {
        assert(num_gkvec_loc_ == phi__.pw_coeffs().num_rows_loc());
        
        auto& bp_chunks = ctx_.beta_projector_chunks();

        int nbeta = bp_chunks(chunk__).num_beta_;

        static_assert(std::is_same<T, double_complex>::value || std::is_same<T, double>::value, "wrong type");

        int fsz = std::is_same<T, double_complex>::value ? 2 : 1;

        if (static_cast<size_t>(fsz * nbeta * n__) > beta_phi_.size()) {
            beta_phi_ = mdarray<double, 1>(nbeta * n__ * fsz);
            if (ctx_.processing_unit() == GPU) {
                beta_phi_.allocate(memory_t::device);
            }
        }

        matrix<T> beta_phi;

        if (ctx_.processing_unit() == GPU) {
            beta_phi = matrix<T>(reinterpret_cast<T*>(beta_phi_.at<CPU>()), reinterpret_cast<T*>(beta_phi_.at<GPU>()), nbeta, n__);
        } else {
            beta_phi = matrix<T>(reinterpret_cast<T*>(beta_phi_.at<CPU>()), nbeta, n__);
        }

        if (std::is_same<T, double_complex>::value) {
            switch (ctx_.processing_unit()) {
                case CPU: {
                    /* compute <beta|phi> */
                    linalg<CPU>::gemm(2, 0, nbeta, n__, num_gkvec_loc_,
                                      pw_coeffs_a_[idx_bp__].template at<CPU>(), num_gkvec_loc_,
                                      phi__.pw_coeffs().prime().at<CPU>(0, idx0__), phi__.pw_coeffs().prime().ld(),
                                      reinterpret_cast<double_complex*>(beta_phi.template at<CPU>()), nbeta);
                    break;
                }
                case GPU: {
                    #ifdef __GPU
                    linalg<GPU>::gemm(2, 0, nbeta, n__, num_gkvec_loc_,
                                      pw_coeffs_a_[idx_bp__].at<GPU>(), num_gkvec_loc_,
                                      phi__.pw_coeffs().prime().at<GPU>(0, idx0__), phi__.pw_coeffs().prime().ld(),
                                      reinterpret_cast<double_complex*>(beta_phi.at<GPU>()), nbeta);
                    beta_phi.copy<memory_t::device, memory_t::host>();
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
                                      reinterpret_cast<double*>(pw_coeffs_a_[idx_bp__].template at<CPU>()), 2 * num_gkvec_loc_,
                                      reinterpret_cast<double*>(phi__.pw_coeffs().prime().at<CPU>(0, idx0__)), 2 * phi__.pw_coeffs().prime().ld(),
                                      b,
                                      reinterpret_cast<double*>(beta_phi.template at<CPU>()), nbeta);

                    if (gkvec_.comm().rank() == 0) {
                        /* subtract one extra G=0 contribution */
                        linalg<CPU>::ger(nbeta, n__, a1,
                                         reinterpret_cast<double*>(pw_coeffs_a_[idx_bp__].template at<CPU>()), 2 * num_gkvec_loc_,
                                         reinterpret_cast<double*>(phi__.pw_coeffs().prime().at<CPU>(0, idx0__)), 2 * phi__.pw_coeffs().prime().ld(),
                                         reinterpret_cast<double*>(beta_phi.template at<CPU>()), nbeta);
                    }
                    break;
                }
                case GPU: {
                    #ifdef __GPU
                    linalg<GPU>::gemm(2, 0, nbeta, n__, 2 * num_gkvec_loc_,
                                      &a,
                                      reinterpret_cast<double*>(pw_coeffs_a_[idx_bp__].at<GPU>()), 2 * num_gkvec_loc_,
                                      reinterpret_cast<double*>(phi__.pw_coeffs().prime().at<GPU>(0, idx0__)), 2 * phi__.pw_coeffs().prime().ld(),
                                      &b,
                                      reinterpret_cast<double*>(beta_phi.at<GPU>()), nbeta);

                    if (comm_.rank() == 0) {
                        /* subtract one extra G=0 contribution */
                        linalg<GPU>::ger(nbeta, n__, &a1, 
                                         reinterpret_cast<double*>(pw_coeffs_a_[idx_bp__].at<GPU>()), 2 * num_gkvec_loc_,
                                         reinterpret_cast<double*>(phi__.pw_coeffs().prime().at<GPU>(0, idx0__)), 2 * phi__.pw_coeffs().prime().ld(),
                                         reinterpret_cast<double*>(beta_phi.at<CPU>()), nbeta);
                    }
                    beta_phi.copy<memory_t::device, memory_t::host>();
                    #else
                    TERMINATE_NO_GPU
                    #endif
                    break;
                }
            }

        }

        return std::move(beta_phi);
    }
};

}


#endif
