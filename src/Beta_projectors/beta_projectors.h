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

/** \file beta_projectors.h
 *
 *  \brief Contains declaration and implementation of sirius::Beta_projectors class.
 */

#ifndef __BETA_PROJECTORS_H__
#define __BETA_PROJECTORS_H__

#include "gpu.h"
#include "communicator.hpp"
#include "unit_cell.h"
#include "wave_functions.hpp"
#include "sbessel.h"
#include "simulation_context.h"
#include "beta_projector_chunks.h"

#ifdef __GPU
extern "C" void create_beta_gk_gpu(int num_atoms,
                                   int num_gkvec,
                                   int const* beta_desc,
                                   cuDoubleComplex const* beta_gk_t,
                                   double const* gkvec,
                                   double const* atom_pos,
                                   cuDoubleComplex* beta_gk);
#endif

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

        if (!(std::is_same<T, double_complex>::value || std::is_same<T, double>::value)) {
            TERMINATE("wrong type");
        }

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

class Beta_projectors_strain_deriv : public Beta_projectors_base<9>
{
  private:

    void generate_pw_coefs_t()
    {
        Radial_integrals_beta beta_ri0(ctx_.unit_cell(), ctx_.gk_cutoff(), 20);
        Radial_integrals_beta_dg beta_ri1(ctx_.unit_cell(), ctx_.gk_cutoff(), 20);

        auto& comm = gkvec_.comm();

        auto& bchunk = ctx_.beta_projector_chunks();

        /* allocate array */
        for (int i = 0; i < 9; i++) {
            pw_coeffs_t_[i] = matrix<double_complex>(num_gkvec_loc(), bchunk.num_beta_t());
        }

        auto dRlm_deps = [this](int lm, vector3d<double>& gvs, int mu, int nu)
        {
            double theta = gvs[1];
            double phi   = gvs[2];
            
            if (lm == 0) {
                return 0.0;
            }

            vector3d<double> q({std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi), std::cos(theta)});
            vector3d<double> dtheta_dq({std::cos(phi) * std::cos(theta), std::cos(theta) * std::sin(phi), -std::sin(theta)});
            vector3d<double> dphi_dq({-std::sin(phi), std::cos(phi), 0.0});

            return -q[mu] * (SHT::dRlm_dtheta(lm, theta, phi) * dtheta_dq[nu] +
                             SHT::dRlm_dphi_sin_theta(lm, theta, phi) * dphi_dq[nu]);
        };

        auto dRlm_deps_v2 = [this](int lm, vector3d<double>& gvc, vector3d<double>& gvs, int mu, int nu)
        {
            int lmax = 4;
            int lmmax = Utils::lmmax(lmax);

            double dg = 1e-6 * gvs[0];

            mdarray<double, 2>drlm(lmmax, 3);

            for (int x = 0; x < 3; x++) {
                vector3d<double> g1 = gvc;
                g1[x] += dg;
                vector3d<double> g2 = gvc;
                g2[x] -= dg;
                
                auto gs1 = SHT::spherical_coordinates(g1);
                auto gs2 = SHT::spherical_coordinates(g2);
                std::vector<double> rlm1(lmmax);
                std::vector<double> rlm2(lmmax);
                
                SHT::spherical_harmonics(lmax, gs1[1], gs1[2], &rlm1[0]);
                SHT::spherical_harmonics(lmax, gs2[1], gs2[2], &rlm2[0]);
                
                for (int lm = 0; lm < lmmax; lm++) {
                    drlm(lm, x) = (rlm1[lm] - rlm2[lm]) / 2 / dg;
                }
            }

            return -gvc[mu] * drlm(lm, nu);
        };

        /* compute d <G+k|beta> / d epsilon_{mu, nu} */
        #pragma omp parallel for
        for (int igkloc = 0; igkloc < num_gkvec_loc(); igkloc++) {
            int igk  = gkvec_.gvec_offset(comm.rank()) + igkloc;
            auto gvc = gkvec_.gkvec_cart(igk);
            /* vs = {r, theta, phi} */
            auto gvs = SHT::spherical_coordinates(gvc);

            if (gvs[0] < 1e-10) {
                for (int nu = 0; nu < 3; nu++) {
                    for (int mu = 0; mu < 3; mu++) {
                        double p = (mu == nu) ? 0.5 : 0;

                        for (int iat = 0; iat < ctx_.unit_cell().num_atom_types(); iat++) {
                            auto& atom_type = ctx_.unit_cell().atom_type(iat);
                            for (int xi = 0; xi < atom_type.mt_basis_size(); xi++) {
                                int l     = atom_type.indexb(xi).l;
                                int idxrf = atom_type.indexb(xi).idxrf;

                                if (l == 0) {
                                    auto z = fourpi / std::sqrt(ctx_.unit_cell().omega());

                                    auto d1 = beta_ri0.value(idxrf, iat, gvs[0]) * (-p * y00);

                                    pw_coeffs_t_[mu + nu * 3](igkloc, atom_type.offset_lo() + xi) = z * d1;
                                } else {
                                    pw_coeffs_t_[mu + nu * 3](igkloc, atom_type.offset_lo() + xi) = 0;
                                }
                            }
                        }
                    }
                }
                continue;
            }

            for (int nu = 0; nu < 3; nu++) {
                for (int mu = 0; mu < 3; mu++) {
                    double p = (mu == nu) ? 0.5 : 0;
                    /* compute real spherical harmonics for G+k vector */
                    std::vector<double> gkvec_rlm(Utils::lmmax(lmax_beta_));
                    std::vector<double> gkvec_drlm(Utils::lmmax(lmax_beta_));

                    SHT::spherical_harmonics(lmax_beta_, gvs[1], gvs[2], &gkvec_rlm[0]);

                    for (int lm = 0; lm < Utils::lmmax(lmax_beta_); lm++) {
                        gkvec_drlm[lm] = dRlm_deps(lm, gvs, mu, nu);
                        //gkvec_drlm[lm] = dRlm_deps_v2(lm, gvc, gvs, mu, nu);
                    }

                    for (int iat = 0; iat < ctx_.unit_cell().num_atom_types(); iat++) {
                        auto& atom_type = ctx_.unit_cell().atom_type(iat);
                        for (int xi = 0; xi < atom_type.mt_basis_size(); xi++) {
                            int l     = atom_type.indexb(xi).l;
                            int lm    = atom_type.indexb(xi).lm;
                            int idxrf = atom_type.indexb(xi).idxrf;

                            auto z = std::pow(double_complex(0, -1), l) * fourpi / std::sqrt(ctx_.unit_cell().omega());

                            auto d1 = beta_ri0.value(idxrf, iat, gvs[0]) * (gkvec_drlm[lm] - p * gkvec_rlm[lm]);

                            auto d2 = beta_ri1.value(idxrf, iat, gvs[0]) * gkvec_rlm[lm];

                            pw_coeffs_t_[mu + nu * 3](igkloc, atom_type.offset_lo() + xi) = z * (d1 - d2 * gvc[mu] * gvc[nu] / gvs[0]);
                        }
                    }
                }
            }
        }
    }

  public:
    Beta_projectors_strain_deriv(Simulation_context& ctx__,
                                 Gvec         const& gkvec__)
        : Beta_projectors_base<9>(ctx__, gkvec__)
    {
        generate_pw_coefs_t();
    }

    void generate(int ichunk__)
    {
        auto& bchunk = ctx_.beta_projector_chunks();

        int num_beta = bchunk(ichunk__).num_beta_;

        auto& comm = gkvec_.comm();

        /* allocate array */
        for (int i = 0; i < 9; i++) {
            pw_coeffs_a_[i] = matrix<double_complex>(num_gkvec_loc(), num_beta);
        }

        #pragma omp for
        for (int i = 0; i < bchunk(ichunk__).num_atoms_; i++) {
            int ia = bchunk(ichunk__).desc_(beta_desc_idx::ia, i);

            double phase = twopi * (gkvec_.vk() * ctx_.unit_cell().atom(ia).position());
            double_complex phase_k = std::exp(double_complex(0.0, phase));

            std::vector<double_complex> phase_gk(num_gkvec_loc());
            for (int igk_loc = 0; igk_loc < num_gkvec_loc_; igk_loc++) {
                int igk = gkvec_.gvec_offset(comm.rank()) + igk_loc;
                auto G = gkvec_.gvec(igk);
                phase_gk[igk_loc] = std::conj(ctx_.gvec_phase_factor(G, ia) * phase_k);
            }

            for (int nu = 0; nu < 3; nu++) {
                for (int mu = 0; mu < 3; mu++) {

                    for (int xi = 0; xi < bchunk(ichunk__).desc_(beta_desc_idx::nbf, i); xi++) {
                        for (int igk_loc = 0; igk_loc < num_gkvec_loc(); igk_loc++) {
                            pw_coeffs_a_[mu + nu * 3](igk_loc, bchunk(ichunk__).desc_(beta_desc_idx::offset, i) + xi) = 
                                pw_coeffs_t_[mu + nu * 3](igk_loc, bchunk(ichunk__).desc_(beta_desc_idx::offset_t, i) + xi) * phase_gk[igk_loc];
                        }
                    }
                }
            }
        }
    }

    inline int num_gkvec_loc() const
    {
        return num_gkvec_loc_;
    }

};

class Beta_projectors_gradient;

/// Stores <G+k | beta> expansion
/** \todo Beta_projectors and Beta_projectors_gradient need some rethinking. Beta_projectors are used in two
 *        places: in application of non-local potential and in generation of density matrix. Beta_projectors_gradient
 *        are used in the calculation of forces. Both are split in chunks, both require an inner product with
 *        wave-functions.
 */
class Beta_projectors
{
    friend class Beta_projectors_gradient;

    protected:

        Simulation_context const& ctx_;

        Communicator const& comm_;

        Unit_cell const& unit_cell_;

        Gvec const& gkvec_;

        mdarray<double, 2> gkvec_coord_;

        int lmax_beta_;

        device_t pu_;

        int num_gkvec_loc_;

        /// Phase-factor independent plane-wave coefficients of |beta> functions for atom types.
        matrix<double_complex> beta_gk_t_;

        /// Plane-wave coefficients of |beta> functions for all atoms.
        matrix<double_complex> beta_gk_a_;

        /// Plane-wave coefficients of |beta> functions for a chunk of atoms.
        matrix<double_complex> beta_gk_;
        
        /// Inner product between beta-projectors and wave-functions.
        /** Stored as double to handle both gamma- and general k-point cases */
        mdarray<double, 1> beta_phi_;

        /// Explicit GPU buffer for beta-projectors.
        matrix<double_complex> beta_gk_gpu_;

        std::unique_ptr<Radial_integrals_beta> beta_radial_integrals_;

        /// Generate plane-wave coefficients for beta-projectors of atom types.
        void generate_beta_gk_t(Simulation_context_base const& ctx__);
                    
        /// Calculate inner product between beta-projectors and wave-functions.
        /** The following is computed: <beta|phi> */
        template <typename T>
        inline matrix <T> inner(int                         chunk__,
                                wave_functions&             phi__,
                                int                         idx0__,
                                int                         n__,
                                mdarray<double_complex, 2>& beta_gk__,
                                mdarray<double, 1>&         beta_phi__);

    public:

        Beta_projectors(Simulation_context const& ctx__,
                        Communicator const& comm__,
                        Gvec const& gkvec__);

        matrix<double_complex>& beta_gk_t()
        {
            return beta_gk_t_;
        }

        matrix<double_complex> const& beta_gk_a()
        {
            return beta_gk_a_;
        }

        matrix<double_complex> const& beta_gk() const
        {
            return beta_gk_;
        }

        Unit_cell const& unit_cell() const
        {
            return unit_cell_;
        }

        Communicator const& comm() const
        {
            return comm_;
        }

        Gvec const& gk_vectors() const
        {
            return gkvec_;
        }

        device_t proc_unit() const
        {
            return pu_;
        }

        int lmax_beta() const
        {
            return lmax_beta_;
        }

        inline int num_gkvec_loc() const
        {
            return num_gkvec_loc_;
        }

        void generate(int chunk__);

        template <typename T>
        inline matrix<T> inner(int chunk__, wave_functions& phi__, int idx0__, int n__)
        {
            return inner<T>(chunk__, phi__, idx0__, n__, beta_gk_, beta_phi_);
        }

        void prepare()
        {
            if (pu_ == GPU) {
                beta_gk_gpu_.allocate(memory_t::device);
                beta_phi_.allocate(memory_t::device);
            }
        }

        void dismiss()
        {
            #ifdef __GPU
            if (pu_ == GPU) {
                beta_gk_gpu_.deallocate_on_device();
                beta_phi_.deallocate_on_device();
            }
            #endif
        }

        inline Beta_projector_chunks const& beta_projector_chunks() const
        {
            return ctx_.beta_projector_chunks();
        }

        //void generate_beta_gk_t_lat_deriv(Simulation_context const& ctx__)
        //{
        //    PROFILE("sirius::Beta_projectors::generate_beta_gk_t_lat_deriv");

        //    auto beta_dG_ri = Radial_integrals_beta_dg(unit_cell_, ctx_.gk_cutoff(), 20);

        //    int mu = 0;
        //    int nu = 0;

        //    /* allocate array */
        //    matrix<double_complex> beta_gk_t_lat_deriv(gkvec_.gvec_count(comm_.rank()), num_beta_t_);
        //    
        //    /* compute dG_tau / da_{mu,nu} */ 
        //    auto dG_da = [this](vector3d<double>& gvc, int tau, int mu, int nu)
        //    {
        //        return -unit_cell_.inverse_lattice_vectors()(nu, tau) * gvc[mu];
        //    };

        //    /* compute derivative of theta angle with respect to lattice vectors d theta / da_{mu,nu} */
        //    auto dtheta_da = [this, dG_da](vector3d<double>& gvc, vector3d<double>& gvs, int mu, int nu)
        //    {
        //        double g     = gvs[0];
        //        double theta = gvs[1];
        //        double phi   = gvs[2];

        //        double result = std::cos(theta) * std::cos(phi) * dG_da(gvc, 0, mu, nu) +
        //                        std::cos(theta) * std::sin(phi) * dG_da(gvc, 1, mu, nu) -
        //                        std::sin(theta) * dG_da(gvc, 2, mu, nu);
        //        return result / g;
        //    };

        //    auto dphi_da = [this, dG_da](vector3d<double>& gvc, vector3d<double>& gvs, int mu, int nu)
        //    {
        //        double g   = gvs[0];
        //        double phi = gvs[2];

        //        double result = -std::sin(phi) * dG_da(gvc, 0, mu, nu) + std::cos(phi) * dG_da(gvc, 0, mu, nu);
        //        return result / g;
        //    };

        //    auto dRlm_da = [this, dtheta_da, dphi_da](int lm, vector3d<double>& gvc, vector3d<double>& gvs, int mu, int nu)
        //    {
        //        double theta = gvs[1];
        //        double phi   = gvs[2];

        //        return SHT::dRlm_dtheta(lm, theta, phi) * dtheta_da(gvc, gvs, mu, nu) +
        //               SHT::dRlm_dphi_sin_theta(lm, theta, phi) * dphi_da(gvc, gvs, mu, nu);
        //    };

        //    auto djl_da = [this, dG_da](vector3d<double>& gvc, vector3d<double>& gvs, int mu, int nu)
        //    {
        //        double theta = gvs[1];
        //        double phi   = gvs[2];

        //        return std::sin(theta) * std::cos(phi) * dG_da(gvc, 0, mu, nu) +
        //               std::sin(theta) * std::sin(phi) * dG_da(gvc, 1, mu, nu) +
        //               std::cos(theta) * dG_da(gvc, 2, mu, nu);
        //    };
        //    
        //    /* compute d <G+k|beta> / a_{mu, nu} */
        //    #pragma omp parallel for
        //    for (int igkloc = 0; igkloc < gkvec_.gvec_count(comm_.rank()); igkloc++) {
        //        int igk  = gkvec_.gvec_offset(comm_.rank()) + igkloc;
        //        auto gvc = gkvec_.gkvec_cart(igk);
        //        /* vs = {r, theta, phi} */
        //        auto gvs = SHT::spherical_coordinates(gvc);

        //        if (gvs[0] < 1e-10) {
        //            for (int i = 0; i < num_beta_t_; i++) {
        //                 beta_gk_t_lat_deriv(igkloc, i) = 0;
        //            }
        //            continue;
        //        }

        //        /* compute real spherical harmonics for G+k vector */
        //        std::vector<double> gkvec_rlm(Utils::lmmax(lmax_beta_));
        //        std::vector<double> gkvec_drlm(Utils::lmmax(lmax_beta_));

        //        SHT::spherical_harmonics(lmax_beta_, gvs[1], gvs[2], &gkvec_rlm[0]);

        //        for (int lm = 0; lm < Utils::lmmax(lmax_beta_); lm++) {
        //            gkvec_drlm[lm] = dRlm_da(lm, gvc, gvs, mu, nu);
        //        }

        //        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
        //            auto& atom_type = unit_cell_.atom_type(iat);
        //            for (int xi = 0; xi < atom_type.mt_basis_size(); xi++) {
        //                int l     = atom_type.indexb(xi).l;
        //                int lm    = atom_type.indexb(xi).lm;
        //                int idxrf = atom_type.indexb(xi).idxrf;

        //                auto z = std::pow(double_complex(0, -1), l) * fourpi / std::sqrt(unit_cell_.omega());

        //                auto d1 = beta_radial_integrals_->value(idxrf, iat, gvs[0]) *
        //                          (gkvec_drlm[lm] - 0.5 * gkvec_rlm[lm] * unit_cell_.inverse_lattice_vectors()(nu, mu)); 

        //                auto d2 = gkvec_rlm[lm] * djl_da(gvc, gvs, mu, nu) * beta_dG_ri.value(idxrf, iat, gvs[0]);

        //                beta_gk_t_lat_deriv(igkloc, atom_type.offset_lo() + xi) = z * (d1 + d2);
        //            }
        //        }
        //    }

        //    //if (unit_cell_.parameters().control().print_checksum_) {
        //    //    auto c1 = beta_gk_t_.checksum();
        //    //    comm_.allreduce(&c1, 1);
        //    //    if (comm_.rank() == 0) {
        //    //        DUMP("checksum(beta_gk_t) : %18.10f %18.10f", c1.real(), c1.imag())
        //    //    }
        //    //}
        //}
};

inline Beta_projectors::Beta_projectors(Simulation_context const& ctx__,
                                        Communicator const& comm__,
                                        Gvec const& gkvec__)
    : ctx_(ctx__)
    , comm_(comm__)
    , unit_cell_(ctx__.unit_cell())
    , gkvec_(gkvec__)
    , lmax_beta_(unit_cell_.lmax())
    , pu_(ctx__.processing_unit())
{
    PROFILE("sirius::Beta_projectors::Beta_projectors");

    num_gkvec_loc_ = gkvec_.gvec_count(comm_.rank());

    beta_radial_integrals_ = std::unique_ptr<Radial_integrals_beta>(new Radial_integrals_beta(unit_cell_, ctx_.gk_cutoff(), 20));

    auto& bp_chunks = ctx_.beta_projector_chunks();

    generate_beta_gk_t(ctx_);

    if (pu_ == GPU) {
        gkvec_coord_ = mdarray<double, 2>(3, num_gkvec_loc_, ctx__.dual_memory_t());
        /* copy G+k vectors */
        for (int igk_loc = 0; igk_loc < num_gkvec_loc_; igk_loc++) {
            int igk  = gkvec_.gvec_offset(comm_.rank()) + igk_loc;
            auto vgk = gkvec_.gkvec(igk);
            for (auto x: {0, 1, 2}) {
                gkvec_coord_(x, igk_loc) = vgk[x];
            }
        }
        gkvec_coord_.copy<memory_t::host, memory_t::device>();

        beta_gk_t_.allocate(memory_t::device);
        beta_gk_t_.copy<memory_t::host, memory_t::device>();
    }
    beta_gk_gpu_ = matrix<double_complex>(num_gkvec_loc_, bp_chunks.max_num_beta(), memory_t::none);

    beta_gk_a_ = matrix<double_complex>(num_gkvec_loc_, unit_cell_.mt_lo_basis_size());

    #pragma omp for
    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
        double phase = twopi * (gkvec_.vk() * unit_cell_.atom(ia).position());
        double_complex phase_k = std::exp(double_complex(0.0, phase));

        std::vector<double_complex> phase_gk(num_gkvec_loc_);
        for (int igk_loc = 0; igk_loc < num_gkvec_loc_; igk_loc++) {
            int igk = gkvec_.gvec_offset(comm_.rank()) + igk_loc;
            auto G = gkvec_.gvec(igk);
            phase_gk[igk_loc] = std::conj(ctx__.gvec_phase_factor(G, ia) * phase_k);
        }

        for (int xi = 0; xi < unit_cell_.atom(ia).mt_lo_basis_size(); xi++) {
            for (int igk_loc = 0; igk_loc < num_gkvec_loc_; igk_loc++) {
                beta_gk_a_(igk_loc, unit_cell_.atom(ia).offset_lo() + xi) =
                    beta_gk_t_(igk_loc, unit_cell_.atom(ia).type().offset_lo() + xi) * phase_gk[igk_loc];
            }
        }
    }
}

inline void Beta_projectors::generate_beta_gk_t(Simulation_context_base const& ctx__)
{
    PROFILE("sirius::Beta_projectors::generate_beta_gk_t");

    if (!beta_projector_chunks().num_beta_t()) {
        return;
    }

    /* allocate array */
    beta_gk_t_ = matrix<double_complex>(gkvec_.gvec_count(comm_.rank()), beta_projector_chunks().num_beta_t());
    
    /* compute <G+k|beta> */
    #pragma omp parallel for
    for (int igkloc = 0; igkloc < gkvec_.gvec_count(comm_.rank()); igkloc++) {
        int igk   = gkvec_.gvec_offset(comm_.rank()) + igkloc;
        double gk = gkvec_.gvec_len(igk);
        /* vs = {r, theta, phi} */
        auto vs = SHT::spherical_coordinates(gkvec_.gkvec_cart(igk));
        /* compute real spherical harmonics for G+k vector */
        std::vector<double> gkvec_rlm(Utils::lmmax(lmax_beta_));
        SHT::spherical_harmonics(lmax_beta_, vs[1], vs[2], &gkvec_rlm[0]);
        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
            auto& atom_type = unit_cell_.atom_type(iat);
            for (int xi = 0; xi < atom_type.mt_basis_size(); xi++) {
                int l     = atom_type.indexb(xi).l;
                int lm    = atom_type.indexb(xi).lm;
                int idxrf = atom_type.indexb(xi).idxrf;

                auto z = std::pow(double_complex(0, -1), l) * fourpi / std::sqrt(unit_cell_.omega());
                beta_gk_t_(igkloc, atom_type.offset_lo() + xi) = z * gkvec_rlm[lm] * beta_radial_integrals_->value(idxrf, iat, gk);
            }
        }
    }

    if (unit_cell_.parameters().control().print_checksum_) {
        auto c1 = beta_gk_t_.checksum();
        comm_.allreduce(&c1, 1);
        if (comm_.rank() == 0) {
            DUMP("checksum(beta_gk_t) : %18.10f %18.10f", c1.real(), c1.imag())
        }
    }
}

inline void Beta_projectors::generate(int chunk__)
{
    PROFILE("sirius::Beta_projectors::generate");

    auto& bp_chunks = ctx_.beta_projector_chunks();

    if (pu_ == CPU) {
        beta_gk_ = mdarray<double_complex, 2>(&beta_gk_a_(0, bp_chunks(chunk__).offset_),
                                              num_gkvec_loc_, bp_chunks(chunk__).num_beta_);
    }
    #ifdef __GPU
    if (pu_ == GPU) {
        beta_gk_ = mdarray<double_complex, 2>(&beta_gk_a_(0, beta_chunk(chunk__).offset_), beta_gk_gpu_.at<GPU>(),
                                              num_gkvec_loc_, beta_chunk(chunk__).num_beta_);

        auto& desc = beta_chunk(chunk__).desc_;
        create_beta_gk_gpu(beta_chunk(chunk__).num_atoms_,
                           num_gkvec_loc_,
                           desc.at<GPU>(),
                           beta_gk_t_.at<GPU>(),
                           gkvec_coord_.at<GPU>(),
                           beta_chunk(chunk__).atom_pos_.at<GPU>(),
                           beta_gk_.at<GPU>());
    }
    #endif
}

template<>
inline matrix<double_complex> Beta_projectors::inner<double_complex>(int                         chunk__,
                                                                     wave_functions&             phi__,
                                                                     int                         idx0__,
                                                                     int                         n__,
                                                                     mdarray<double_complex, 2>& beta_gk__,
                                                                     mdarray<double, 1>&         beta_phi__)
{
    PROFILE("sirius::Beta_projectors::inner");

    assert(num_gkvec_loc_ == phi__.pw_coeffs().num_rows_loc());

    auto& bp_chunks = ctx_.beta_projector_chunks();

    int nbeta = bp_chunks(chunk__).num_beta_;

    if (static_cast<size_t>(nbeta * n__) > beta_phi__.size()) {
        beta_phi__ = mdarray<double, 1>(2 * nbeta * n__);
        if (pu_ == GPU) {
            beta_phi__.allocate(memory_t::device);
        }
    }

    switch (pu_) {
        case CPU: {
            /* compute <beta|phi> */
            linalg<CPU>::gemm(2, 0, nbeta, n__, num_gkvec_loc_,
                              beta_gk__.at<CPU>(), num_gkvec_loc_,
                              phi__.pw_coeffs().prime().at<CPU>(0, idx0__), phi__.pw_coeffs().prime().ld(),
                              (double_complex*)beta_phi__.at<CPU>(), nbeta);
            break;
        }
        case GPU: {
            #ifdef __GPU
            linalg<GPU>::gemm(2, 0, nbeta, n__, num_gkvec_loc_, beta_gk.at<GPU>(), num_gkvec_loc_,
                              phi__.pw_coeffs().prime().at<GPU>(0, idx0__), phi__.pw_coeffs().prime().ld(),
                              (double_complex*)beta_phi__.at<GPU>(), nbeta);
            beta_phi__.copy_to_host(2 * nbeta * n__);
            #else
            TERMINATE_NO_GPU
            #endif
            break;
        }
    }

    comm_.allreduce(beta_phi__.at<CPU>(), 2 * nbeta * n__);

    if (pu_ == GPU) {
        beta_phi__.copy<memory_t::host, memory_t::device>(2 * nbeta * n__);
    }

    if (unit_cell_.parameters().control().print_checksum_) {
        auto cs = mdarray<double, 1>(beta_phi__.at<CPU>(), 2 * nbeta * n__).checksum();
        if (comm_.rank() == 0) {
            DUMP("checksum(beta_phi) : %18.10f", cs);
        }
    }

    if (pu_ == GPU) {
        return std::move(matrix<double_complex>(reinterpret_cast<double_complex*>(beta_phi__.at<CPU>()),
                                                reinterpret_cast<double_complex*>(beta_phi__.at<GPU>()),
                                                nbeta, n__));
    } else {
        return std::move(matrix<double_complex>(reinterpret_cast<double_complex*>(beta_phi__.at<CPU>()),
                                                nbeta, n__));
    }
}

template<>
inline matrix<double> Beta_projectors::inner<double>(int                         chunk__,
                                                     wave_functions&             phi__,
                                                     int                         idx0__,
                                                     int                         n__,
                                                     mdarray<double_complex, 2>& beta_gk__,
                                                     mdarray<double, 1>&         beta_phi__)
{
    PROFILE("sirius::Beta_projectors::inner");

    assert(num_gkvec_loc_ == phi__.pw_coeffs().num_rows_loc());
    
    auto& bp_chunks = ctx_.beta_projector_chunks();

    int nbeta = bp_chunks(chunk__).num_beta_;

    if (static_cast<size_t>(nbeta * n__) > beta_phi__.size())
    {
        beta_phi__ = mdarray<double, 1>(nbeta * n__);
        #ifdef __GPU
        if (pu_ == GPU) {
            beta_phi__.allocate(memory_t::device);
        }
        #endif
    }

    double a = 2;
    double a1 = -1;
    double b = 0;

    switch (pu_) {
        case CPU: {
            /* compute <beta|phi> */
            linalg<CPU>::gemm(2, 0, nbeta, n__, 2 * num_gkvec_loc_,
                              a,
                              (double*)beta_gk__.at<CPU>(), 2 * num_gkvec_loc_,
                              (double*)phi__.pw_coeffs().prime().at<CPU>(0, idx0__), 2 * phi__.pw_coeffs().prime().ld(),
                              b,
                              beta_phi__.at<CPU>(), nbeta);

            if (comm_.rank() == 0) {
                /* subtract one extra G=0 contribution */
                linalg<CPU>::ger(nbeta, n__, a1, (double*)&beta_gk__(0, 0), 2 * num_gkvec_loc_,
                                (double*)phi__.pw_coeffs().prime().at<CPU>(0, idx0__), 2 * phi__.pw_coeffs().prime().ld(),
                                &beta_phi__[0], nbeta);
            }
            break;
        }
        case GPU: {
            #ifdef __GPU
            linalg<GPU>::gemm(2, 0, nbeta, n__, 2 * num_gkvec_loc_,
                              &a,
                              (double*)beta_gk.at<GPU>(), 2 * num_gkvec_loc_,
                              (double*)phi__.pw_coeffs().prime().at<GPU>(0, idx0__), 2 * phi__.pw_coeffs().prime().ld(),
                              &b,
                              beta_phi__.at<GPU>(), nbeta);
            if (comm_.rank() == 0) {
                /* subtract one extra G=0 contribution */
                linalg<GPU>::ger(nbeta, n__, &a1, (double*)beta_gk.at<GPU>(0, 0), 2 * num_gkvec_loc_,
                                (double*)phi__.pw_coeffs().prime().at<GPU>(0, idx0__), 2 * phi__.pw_coeffs().prime().ld(),
                                beta_phi__.at<GPU>(), nbeta);
            }
            beta_phi__.copy_to_host(nbeta * n__);
            #else
            TERMINATE_NO_GPU
            #endif
            break;
        }
    }

    comm_.allreduce(beta_phi__.at<CPU>(), nbeta * n__);

    #ifdef __GPU
    if (pu_ == GPU) {
        beta_phi__.copy_to_device(nbeta * n__);
    }
    #endif

    #ifdef __PRINT_OBJECT_CHECKSUM
    {
        auto cs = mdarray<double, 1>(beta_phi__.at<CPU>(), nbeta * n__).checksum();
        DUMP("checksum(beta_phi) : %18.10f", cs);
    }
    #endif

    if (pu_ == GPU) {
        return std::move(matrix<double>(beta_phi__.at<CPU>(), beta_phi__.at<GPU>(), nbeta, n__));
    } else {
        return std::move(matrix<double>(beta_phi__.at<CPU>(), nbeta, n__));
    }
}

} // namespace

#endif
