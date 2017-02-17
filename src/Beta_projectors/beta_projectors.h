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

enum beta_desc_idx {
    nbf      = 0,
    offset   = 1,
    offset_t = 2,
    ia       = 3
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

        Communicator const& comm_;

        Unit_cell const& unit_cell_;

        Gvec const& gkvec_;

        mdarray<double, 2> gkvec_coord_;

        int lmax_beta_;

        device_t pu_;

        int num_gkvec_loc_;

        /// Total number of beta-projectors among atom types.
        int num_beta_t_;

        /// Phase-factor independent plane-wave coefficients of |beta> functions for atom types.
        matrix<double_complex> beta_gk_t_;

        /// Plane-wave coefficients of |beta> functions for all atoms.
        matrix<double_complex> beta_gk_a_;

        /// Plane-wave coefficients of |beta> functions for a chunk of atoms.
        matrix<double_complex> beta_gk_;
        
        /// Inner product between beta-projectors and wave-functions.
        /** Store as double to handle both gamma- and general k-point cases */
        mdarray<double, 1> beta_phi_;

        /// Explicit GPU buffer for beta-projectors.
        matrix<double_complex> beta_gk_gpu_;

        struct beta_chunk_t
        {
            int num_beta_;
            int num_atoms_;
            int offset_;
            mdarray<int, 2> desc_;
            mdarray<double, 2> atom_pos_;
        };

        mdarray<beta_chunk_t, 1> beta_chunks_;

        int max_num_beta_;

        /// Generate plane-wave coefficients for beta-projectors of atom types.
        void generate_beta_gk_t(Simulation_context const& ctx__);
                    
        void split_in_chunks();

        /// calculates < Beta | Psi > inner product
        template <typename T>
        void inner(int chunk__,
                   wave_functions& phi__,
                   int idx0__,
                   int n__,
                   mdarray<double_complex, 2>& beta_gk__,
                   mdarray<double, 1>& beta_phi__);

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

        template <typename T>
        matrix<T> beta_phi(int chunk__, int n__)
        {
            int nbeta = beta_chunk(chunk__).num_beta_;
            if (pu_ == GPU) {
                return std::move(matrix<T>(reinterpret_cast<T*>(beta_phi_.at<CPU>()),
                                           reinterpret_cast<T*>(beta_phi_.at<GPU>()),
                                           nbeta, n__));
            } else {
                return std::move(matrix<T>(reinterpret_cast<T*>(beta_phi_.at<CPU>()),
                                           nbeta, n__));
            }
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

        inline int num_beta_chunks() const
        {
            return static_cast<int>(beta_chunks_.size());
        }

        inline beta_chunk_t const& beta_chunk(int idx__) const
        {
            return beta_chunks_(idx__);
        }

        inline int num_gkvec_loc() const
        {
            return num_gkvec_loc_;
        }

        void generate(int chunk__);

        template <typename T>
        void inner(int chunk__, wave_functions& phi__, int idx0__, int n__)
        {
            inner<T>(chunk__, phi__, idx0__, n__, beta_gk_, beta_phi_);
        }

        int max_num_beta()
        {
            return max_num_beta_;
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

        void generate_beta_gk_t_lat_deriv(Simulation_context const& ctx__)
        {
            PROFILE("sirius::Beta_projectors::generate_beta_gk_t_lat_deriv");

            int mu = 0;
            int nu = 0;

            /* allocate array */
            matrix<double_complex> beta_gk_t_lat_deriv(gkvec_.gvec_count(comm_.rank()), num_beta_t_);
            
            /* compute dG_tau / da_{mu,nu} */ 
            auto dG_da = [this](vector3d<double>& gvc, int tau, int mu, int nu)
            {
                return -unit_cell_.inverse_lattice_vectors()(nu, tau) * gvc[mu];
            };

            /* compute derivative of theta angle with respect to lattice vectors d theta / da_{mu,nu} */
            auto dtheta_da = [this, dG_da](vector3d<double>& gvc, vector3d<double>& gvs, int mu, int nu)
            {
                double g     = gvs[0];
                double theta = gvs[1];
                double phi   = gvs[2];

                double result = std::cos(theta) * std::cos(phi) * dG_da(gvc, 0, mu, nu) +
                                std::cos(theta) * std::sin(phi) * dG_da(gvc, 1, mu, nu) -
                                std::sin(theta) * dG_da(gvc, 2, mu, nu);
                return result / g;
            };

            auto dphi_da = [this, dG_da](vector3d<double>& gvc, vector3d<double>& gvs, int mu, int nu)
            {
                double g   = gvs[0];
                double phi = gvs[2];

                double result = -std::sin(phi) * dG_da(gvc, 0, mu, nu) + std::cos(phi) * dG_da(gvc, 0, mu, nu);
                return result / g;
            };

            auto dRlm_da = [this, dtheta_da, dphi_da](int lm, vector3d<double>& gvc, vector3d<double>& gvs, int mu, int nu)
            {
                double theta = gvs[1];
                double phi   = gvs[2];

                return SHT::dRlm_dtheta(lm, theta, phi) * dtheta_da(gvc, gvs, mu, nu) +
                       SHT::dRlm_dphi_sin_theta(lm, theta, phi) * dphi_da(gvc, gvs, mu, nu);
            };

            auto djl_da = [this, dG_da](vector3d<double>& gvc, vector3d<double>& gvs, int mu, int nu)
            {
                double theta = gvs[1];
                double phi   = gvs[2];

                return std::sin(theta) * std::cos(phi) * dG_da(gvc, 0, mu, nu) +
                       std::sin(theta) * std::sin(phi) * dG_da(gvc, 1, mu, nu) +
                       std::cos(theta) * dG_da(gvc, 2, mu, nu);
            };
            
            /* compute d <G+k|beta> / a_{mu, nu} */
            #pragma omp parallel for
            for (int igkloc = 0; igkloc < gkvec_.gvec_count(comm_.rank()); igkloc++) {
                int igk  = gkvec_.gvec_offset(comm_.rank()) + igkloc;
                auto gvc = gkvec_.gkvec_cart(igk);
                /* vs = {r, theta, phi} */
                auto gvs = SHT::spherical_coordinates(gvc);


                /* compute real spherical harmonics for G+k vector */
                std::vector<double> gkvec_rlm(Utils::lmmax(lmax_beta_));
                std::vector<double> gkvec_drlm(Utils::lmmax(lmax_beta_));

                SHT::spherical_harmonics(lmax_beta_, gvs[1], gvs[2], &gkvec_rlm[0]);

                for (int lm = 0; lm < Utils::lmmax(lmax_beta_); lm++) {
                    gkvec_drlm[lm] = dRlm_da(lm, gvc, gvs, mu, nu);
                }

                for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
                    auto& atom_type = unit_cell_.atom_type(iat);
                    for (int xi = 0; xi < atom_type.mt_basis_size(); xi++) {
                        int l     = atom_type.indexb(xi).l;
                        int lm    = atom_type.indexb(xi).lm;
                        int idxrf = atom_type.indexb(xi).idxrf;

                        auto z = std::pow(double_complex(0, -1), l) * fourpi / std::sqrt(unit_cell_.omega());

                        auto d1 = ctx__.radial_integrals().beta_radial_integral(idxrf, iat, gvs[0]) *
                                  (gkvec_drlm[lm] - 0.5 * gkvec_rlm[lm] * unit_cell_.inverse_lattice_vectors()(nu, mu)); 

                        auto d2 = gkvec_rlm[lm] * djl_da(gvc, gvs, mu, nu) * ctx__.radial_integrals().beta_djldq_radial_integral(idxrf, iat, gvs[0]);

                        beta_gk_t_lat_deriv(igkloc, atom_type.offset_lo() + xi) = z * (d1 + d2);
                    }
                }
            }

            //if (unit_cell_.parameters().control().print_checksum_) {
            //    auto c1 = beta_gk_t_.checksum();
            //    comm_.allreduce(&c1, 1);
            //    if (comm_.rank() == 0) {
            //        DUMP("checksum(beta_gk_t) : %18.10f %18.10f", c1.real(), c1.imag())
            //    }
            //}


        }
};

inline Beta_projectors::Beta_projectors(Simulation_context const& ctx__,
                                        Communicator const& comm__,
                                        Gvec const& gkvec__)
    : comm_(comm__)
    , unit_cell_(ctx__.unit_cell())
    , gkvec_(gkvec__)
    , lmax_beta_(unit_cell_.lmax())
    , pu_(ctx__.processing_unit())
{
    PROFILE("sirius::Beta_projectors::Beta_projectors");

    num_gkvec_loc_ = gkvec_.gvec_count(comm_.rank());

    split_in_chunks();

    generate_beta_gk_t(ctx__);

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
    beta_gk_gpu_ = matrix<double_complex>(num_gkvec_loc_, max_num_beta_, memory_t::none);

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

inline void Beta_projectors::generate_beta_gk_t(Simulation_context const& ctx__)
{
    PROFILE("sirius::Beta_projectors::generate_beta_gk_t");

    if (!num_beta_t_) {
        return;
    }

    /* allocate array */
    beta_gk_t_ = matrix<double_complex>(gkvec_.gvec_count(comm_.rank()), num_beta_t_);
    
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
                beta_gk_t_(igkloc, atom_type.offset_lo() + xi) = z * gkvec_rlm[lm] * ctx__.radial_integrals().beta_radial_integral(idxrf, iat, gk);
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

inline void Beta_projectors::split_in_chunks()
{
    /* split beta-projectors into chunks */
    int num_atoms_in_chunk = (comm_.size() == 1) ? unit_cell_.num_atoms() : std::min(unit_cell_.num_atoms(), 256);
    int num_beta_chunks = unit_cell_.num_atoms() / num_atoms_in_chunk + std::min(1, unit_cell_.num_atoms() % num_atoms_in_chunk);
    splindex<block> spl_beta_chunks(unit_cell_.num_atoms(), num_beta_chunks, 0);

    beta_chunks_ = mdarray<beta_chunk_t, 1>(num_beta_chunks);

    int offset_in_beta_gk = 0;

    for (int ib = 0; ib < num_beta_chunks; ib++) {
        /* number of atoms in chunk */
        int na = spl_beta_chunks.local_size(ib);
        beta_chunks_(ib).num_atoms_ = na;
        beta_chunks_(ib).desc_      = mdarray<int, 2>(4, na);
        beta_chunks_(ib).atom_pos_  = mdarray<double, 2>(3, na);

        int num_beta{0};

        for (int i = 0; i < na; i++) {
            /* global index of atom by local index and chunk */
            int ia = spl_beta_chunks.global_index(i, ib);
            auto pos = unit_cell_.atom(ia).position();
            auto& type = unit_cell_.atom(ia).type();
            /* atom fractional coordinates */
            for (int x: {0, 1, 2}) {
                beta_chunks_(ib).atom_pos_(x, i) = pos[x];
            }
            /* number of beta functions for atom */
            beta_chunks_(ib).desc_(beta_desc_idx::nbf, i) = type.mt_basis_size();
            /* offset in beta_gk*/
            beta_chunks_(ib).desc_(beta_desc_idx::offset, i) = num_beta;
            /* offset in beta_gk_t */
            beta_chunks_(ib).desc_(beta_desc_idx::offset_t, i) = type.offset_lo();
            /* global index of atom */
            beta_chunks_(ib).desc_(beta_desc_idx::ia, i) = ia;

            num_beta += type.mt_basis_size();
        }
        /* number of beta-projectors in this chunk */
        beta_chunks_(ib).num_beta_ = num_beta;
        beta_chunks_(ib).offset_ = offset_in_beta_gk;
        offset_in_beta_gk += num_beta;

        if (pu_ == GPU) {
            beta_chunks_[ib].desc_.allocate(memory_t::device);
            beta_chunks_[ib].desc_.copy<memory_t::host, memory_t::device>();

            beta_chunks_[ib].atom_pos_.allocate(memory_t::device);
            beta_chunks_[ib].atom_pos_.copy<memory_t::host, memory_t::device>();
        }
    }

    max_num_beta_ = 0;
    for (int ib = 0; ib < num_beta_chunks; ib++) {
        max_num_beta_ = std::max(max_num_beta_, beta_chunks_(ib).num_beta_);
    }

    num_beta_t_ = 0;
    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
        num_beta_t_ += unit_cell_.atom_type(iat).mt_lo_basis_size();
    }
}

inline void Beta_projectors::generate(int chunk__)
{
    PROFILE("sirius::Beta_projectors::generate");

    if (pu_ == CPU) {
        beta_gk_ = mdarray<double_complex, 2>(&beta_gk_a_(0, beta_chunk(chunk__).offset_),
                                              num_gkvec_loc_, beta_chunk(chunk__).num_beta_);
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
inline void Beta_projectors::inner<double_complex>(int chunk__,
                                                   wave_functions& phi__,
                                                   int idx0__,
                                                   int n__,
                                                   mdarray<double_complex, 2> &beta_gk,
                                                   mdarray<double, 1> &beta_phi)
{
    PROFILE("sirius::Beta_projectors::inner");

    assert(num_gkvec_loc_ == phi__.pw_coeffs().num_rows_loc());

    int nbeta = beta_chunk(chunk__).num_beta_;

    if (static_cast<size_t>(nbeta * n__) > beta_phi.size()) {
        beta_phi = mdarray<double, 1>(2 * nbeta * n__);
        if (pu_ == GPU) {
            beta_phi.allocate(memory_t::device);
        }
    }

    switch (pu_) {
        case CPU: {
            /* compute <beta|phi> */
            linalg<CPU>::gemm(2, 0, nbeta, n__, num_gkvec_loc_,
                              beta_gk.at<CPU>(), num_gkvec_loc_,
                              phi__.pw_coeffs().prime().at<CPU>(0, idx0__), phi__.pw_coeffs().prime().ld(),
                              (double_complex*)beta_phi.at<CPU>(), nbeta);
            break;
        }
        case GPU: {
            #ifdef __GPU
            linalg<GPU>::gemm(2, 0, nbeta, n__, num_gkvec_loc_, beta_gk.at<GPU>(), num_gkvec_loc_,
                              phi__.pw_coeffs().prime().at<GPU>(0, idx0__), phi__.pw_coeffs().prime().ld(),
                              (double_complex*)beta_phi.at<GPU>(), nbeta);
            beta_phi.copy_to_host(2 * nbeta * n__);
            #else
            TERMINATE_NO_GPU
            #endif
            break;
        }
    }

    comm_.allreduce(beta_phi.at<CPU>(), 2 * nbeta * n__);

    if (pu_ == GPU) {
        beta_phi.copy<memory_t::host, memory_t::device>(2 * nbeta * n__);
    }

    if (unit_cell_.parameters().control().print_checksum_) {
        auto cs = mdarray<double, 1>(beta_phi.at<CPU>(), 2 * nbeta * n__).checksum();
        if (comm_.rank() == 0) {
            DUMP("checksum(beta_phi) : %18.10f", cs);
        }
    }
}

template<>
inline void Beta_projectors::inner<double>(int chunk__,
                                           wave_functions& phi__,
                                           int idx0__,
                                           int n__,
                                           mdarray<double_complex, 2> &beta_gk,
                                           mdarray<double, 1> &beta_phi)
{
    PROFILE("sirius::Beta_projectors::inner");

    assert(num_gkvec_loc_ == phi__.pw_coeffs().num_rows_loc());

    int nbeta = beta_chunk(chunk__).num_beta_;

    if (static_cast<size_t>(nbeta * n__) > beta_phi.size())
    {
        beta_phi = mdarray<double, 1>(nbeta * n__);
        #ifdef __GPU
        if (pu_ == GPU) {
            beta_phi.allocate(memory_t::device);
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
                              (double*)beta_gk.at<CPU>(), 2 * num_gkvec_loc_,
                              (double*)phi__.pw_coeffs().prime().at<CPU>(0, idx0__), 2 * phi__.pw_coeffs().prime().ld(),
                              b,
                              beta_phi.at<CPU>(), nbeta);

            if (comm_.rank() == 0) {
                /* subtract one extra G=0 contribution */
                linalg<CPU>::ger(nbeta, n__, a1, (double*)&beta_gk(0, 0), 2 * num_gkvec_loc_,
                                (double*)phi__.pw_coeffs().prime().at<CPU>(0, idx0__), 2 * phi__.pw_coeffs().prime().ld(),
                                &beta_phi[0], nbeta);
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
                              beta_phi.at<GPU>(), nbeta);
            if (comm_.rank() == 0) {
                /* subtract one extra G=0 contribution */
                linalg<GPU>::ger(nbeta, n__, &a1, (double*)beta_gk.at<GPU>(0, 0), 2 * num_gkvec_loc_,
                                (double*)phi__.pw_coeffs().prime().at<GPU>(0, idx0__), 2 * phi__.pw_coeffs().prime().ld(),
                                beta_phi.at<GPU>(), nbeta);
            }
            beta_phi.copy_to_host(nbeta * n__);
            #else
            TERMINATE_NO_GPU
            #endif
            break;
        }
    }

    comm_.allreduce(beta_phi.at<CPU>(), nbeta * n__);

    #ifdef __GPU
    if (pu_ == GPU) {
        beta_phi.copy_to_device(nbeta * n__);
    }
    #endif

    #ifdef __PRINT_OBJECT_CHECKSUM
    {
        auto cs = mdarray<double, 1>(beta_phi.at<CPU>(), nbeta * n__).checksum();
        DUMP("checksum(beta_phi) : %18.10f", cs);
    }
    #endif
}

} // namespace

#endif
