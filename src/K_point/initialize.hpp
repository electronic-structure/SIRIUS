// Copyright (c) 2013-2016 Anton Kozhevnikov, Thomas Schulthess
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

/** \file initialize.hpp
 *
 *  \brief Initialize sirius::K_point class.
 */

inline void K_point::initialize()
{
    PROFILE("sirius::K_point::initialize");

    zil_.resize(ctx_.lmax_apw() + 1);
    for (int l = 0; l <= ctx_.lmax_apw(); l++) {
        zil_[l] = std::pow(double_complex(0, 1), l);
    }

    l_by_lm_ = utils::l_by_lm(ctx_.lmax_apw());

    int bs = ctx_.cyclic_block_size();

    if (use_second_variation && ctx_.full_potential()) {
        assert(ctx_.num_fv_states() > 0);
        fv_eigen_values_.resize(ctx_.num_fv_states());
    }

    /* In case of collinear magnetism we store only non-zero spinor components.
     *
     * non magnetic case:
     * .---.
     * |   |
     * .---.
     *
     * collinear case:
     * .---.          .---.
     * |uu | 0        |uu |
     * .---.---.  ->  .---.
     *   0 |dd |      |dd |
     *     .---.      .---.
     *
     * non collinear case:
     * .-------.
     * |       |
     * .-------.
     * |       |
     * .-------.
     */
    int nst = ctx_.num_bands();

    auto mem_type_evp = (ctx_.std_evp_solver_type() == ev_solver_t::magma) ? memory_t::host_pinned : memory_t::host;
    auto mem_type_gevp = (ctx_.gen_evp_solver_type() == ev_solver_t::magma) ? memory_t::host_pinned : memory_t::host;

    /* build a full list of G+k vectors for all MPI ranks */
    generate_gkvec(ctx_.gk_cutoff());
    /* build a list of basis functions */
    generate_gklo_basis();

    if (ctx_.full_potential()) {
        if (use_second_variation) {
            if (ctx_.need_sv()) {
                /* in case of collinear magnetism store pure up and pure dn components, otherwise store the full matrix */
                for (int is = 0; is < ctx_.num_spin_dims(); is++) {
                    sv_eigen_vectors_[is] = dmatrix<double_complex>(nst, nst, ctx_.blacs_grid(), bs, bs, mem_type_evp);
                }
            }
            /* allocate fv eien vectors */
            fv_eigen_vectors_slab_ = std::unique_ptr<Wave_functions>(
                new Wave_functions(gkvec_partition(), unit_cell_.num_atoms(),
                    [this](int ia){return unit_cell_.atom(ia).mt_lo_basis_size();}, ctx_.num_fv_states()));

            fv_eigen_vectors_slab_->pw_coeffs(0).prime().zero();
            fv_eigen_vectors_slab_->mt_coeffs(0).prime().zero();
            /* starting guess for wave-functions */
            for (int i = 0; i < ctx_.num_fv_states(); i++) {
                for (int igloc = 0; igloc < gkvec().gvec_count(comm().rank()); igloc++) {
                    int ig = igloc + gkvec().gvec_offset(comm().rank());
                    if (ig == i) {
                        fv_eigen_vectors_slab_->pw_coeffs(0).prime(igloc, i) = 1.0;
                    }
                    if (ig == i + 1) {
                        fv_eigen_vectors_slab_->pw_coeffs(0).prime(igloc, i) = 0.5;
                    }
                    if (ig == i + 2) {
                        fv_eigen_vectors_slab_->pw_coeffs(0).prime(igloc, i) = 0.125;
                    }
                }
            }
            if (ctx_.iterative_solver_input().type_ == "exact") {
                /* ELPA needs a full matrix of eigen-vectors as it uses it as a work space */
                fv_eigen_vectors_ = dmatrix<double_complex>(gklo_basis_size(), gklo_basis_size(), ctx_.blacs_grid(), bs, bs, mem_type_gevp);
            } else {
                int ncomp = ctx_.iterative_solver_input().num_singular_;
                if (ncomp < 0) {
                    ncomp = ctx_.num_fv_states() / 2;
                }

                singular_components_ = std::unique_ptr<Wave_functions>(new Wave_functions(gkvec_partition(), ncomp));
                singular_components_->pw_coeffs(0).prime().zero();
                /* starting guess for wave-functions */
                for (int i = 0; i < ncomp; i++) {
                    for (int igloc = 0; igloc < gkvec().count(); igloc++) {
                        int ig = igloc + gkvec().offset();
                        if (ig == i) {
                            singular_components_->pw_coeffs(0).prime(igloc, i) = 1.0;
                        }
                        if (ig == i + 1) {
                            singular_components_->pw_coeffs(0).prime(igloc, i) = 0.5;
                        }
                        if (ig == i + 2) {
                            singular_components_->pw_coeffs(0).prime(igloc, i) = 0.125;
                        }
                    }
                }
                if (ctx_.control().print_checksum_) {
                    auto cs = singular_components_->checksum_pw(CPU, 0, 0, ncomp);
                    if (comm().rank() == 0) {
                        utils::print_checksum("singular_components", cs);
                    }
                }
            }

            fv_states_ = std::unique_ptr<Wave_functions>(new Wave_functions(gkvec_partition(),
                                                                            unit_cell_.num_atoms(),
                                                                            [this](int ia)
                                                                            {
                                                                                return unit_cell_.atom(ia).mt_basis_size();
                                                                            },
                                                                            ctx_.num_fv_states()));

            spinor_wave_functions_ = std::unique_ptr<Wave_functions>(new Wave_functions(gkvec_partition(),
                                                                                        unit_cell_.num_atoms(),
                                                                                        [this](int ia)
                                                                                        {
                                                                                            return unit_cell_.atom(ia).mt_basis_size();
                                                                                        },
                                                                                        nst,
                                                                                        ctx_.num_spins()));
        } else {
            TERMINATE_NOT_IMPLEMENTED
        }
    } else {
        spinor_wave_functions_ = std::unique_ptr<Wave_functions>(new Wave_functions(gkvec_partition(), nst, ctx_.num_spins()));
    }

    if (ctx_.processing_unit() == GPU && keep_wf_on_gpu) {
        /* allocate GPU memory */
        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            spinor_wave_functions_->pw_coeffs(ispn).prime().allocate(memory_t::device);
            if (ctx_.full_potential()) {
                spinor_wave_functions_->mt_coeffs(ispn).prime().allocate(memory_t::device);
            }
        }
    }

    update();
}
