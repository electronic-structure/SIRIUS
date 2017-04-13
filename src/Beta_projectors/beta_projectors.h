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
#include "beta_projectors_base.h"

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

class Beta_projectors_gradient;

/// Stores <G+k | beta> expansion
/** \todo Beta_projectors and Beta_projectors_gradient need some rethinking. Beta_projectors are used in two
 *        places: in application of non-local potential and in generation of density matrix. Beta_projectors_gradient
 *        are used in the calculation of forces. Both are split in chunks, both require an inner product with
 *        wave-functions.
 */
class Beta_projectors: public Beta_projectors_base<1>
{
    friend class Beta_projectors_gradient;

    protected:

        mdarray<double, 2> gkvec_coord_;

        /// Plane-wave coefficients of |beta> functions for all atoms.
        matrix<double_complex> beta_gk_a_;

        /// Explicit GPU buffer for beta-projectors.
        matrix<double_complex> beta_gk_gpu_;

        std::unique_ptr<Radial_integrals_beta> beta_radial_integrals_;

        /// Generate plane-wave coefficients for beta-projectors of atom types.
        void generate_pw_coefs_t()
        {
            auto& bchunk = ctx_.beta_projector_chunks();
            if (!bchunk.num_beta_t()) {
                return;
            }
            
            auto& comm = gkvec_.comm();

            pw_coeffs_t_[0] = matrix<double_complex>(num_gkvec_loc(), bchunk.num_beta_t());

            /* compute <G+k|beta> */
            #pragma omp parallel for
            for (int igkloc = 0; igkloc < gkvec_.gvec_count(comm.rank()); igkloc++) {
                int igk = gkvec_.gvec_offset(comm.rank()) + igkloc;
                /* vs = {r, theta, phi} */
                auto vs = SHT::spherical_coordinates(gkvec_.gkvec_cart(igk));
                /* compute real spherical harmonics for G+k vector */
                std::vector<double> gkvec_rlm(Utils::lmmax(lmax_beta_));
                SHT::spherical_harmonics(lmax_beta_, vs[1], vs[2], &gkvec_rlm[0]);
                for (int iat = 0; iat < ctx_.unit_cell().num_atom_types(); iat++) {
                    auto& atom_type = ctx_.unit_cell().atom_type(iat);
                    for (int xi = 0; xi < atom_type.mt_basis_size(); xi++) {
                        int l     = atom_type.indexb(xi).l;
                        int lm    = atom_type.indexb(xi).lm;
                        int idxrf = atom_type.indexb(xi).idxrf;

                        auto z = std::pow(double_complex(0, -1), l) * fourpi / std::sqrt(ctx_.unit_cell().omega());
                        pw_coeffs_t_[0](igkloc, atom_type.offset_lo() + xi) = z * gkvec_rlm[lm] * beta_radial_integrals_->value(idxrf, iat, vs[0]);
                    }
                }
            }

            if (ctx_.control().print_checksum_) {
                auto c1 = pw_coeffs_t_[0].checksum();
                comm.allreduce(&c1, 1);
                if (comm.rank() == 0) {
                    DUMP("checksum(beta_pw_coeffs_t) : %18.10f %18.10f", c1.real(), c1.imag())
                }
            }
        }
                    
    public:

        Beta_projectors(Simulation_context& ctx__,
                        Gvec         const& gkvec__)
            : Beta_projectors_base<1>(ctx__, gkvec__)
        {
            PROFILE("sirius::Beta_projectors::Beta_projectors");

            beta_radial_integrals_ = std::unique_ptr<Radial_integrals_beta>(new Radial_integrals_beta(ctx_.unit_cell(), ctx_.gk_cutoff(), 20));

            auto& bp_chunks = ctx_.beta_projector_chunks();
            auto& comm = gkvec_.comm();

            generate_pw_coefs_t();

            if (ctx_.processing_unit() == GPU) {
                gkvec_coord_ = mdarray<double, 2>(3, num_gkvec_loc_, ctx__.dual_memory_t());
                /* copy G+k vectors */
                for (int igk_loc = 0; igk_loc < num_gkvec_loc_; igk_loc++) {
                    int igk  = gkvec_.gvec_offset(comm.rank()) + igk_loc;
                    auto vgk = gkvec_.gkvec(igk);
                    for (auto x: {0, 1, 2}) {
                        gkvec_coord_(x, igk_loc) = vgk[x];
                    }
                }
                gkvec_coord_.copy<memory_t::host, memory_t::device>();

                pw_coeffs_t_[0].allocate(memory_t::device);
                pw_coeffs_t_[0].copy<memory_t::host, memory_t::device>();
            }
            beta_gk_gpu_ = matrix<double_complex>(num_gkvec_loc(), bp_chunks.max_num_beta(), memory_t::none);

            beta_gk_a_ = matrix<double_complex>(num_gkvec_loc(), ctx_.unit_cell().mt_lo_basis_size());

            #pragma omp for
            for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
                double phase = twopi * (gkvec_.vk() * ctx_.unit_cell().atom(ia).position());
                double_complex phase_k = std::exp(double_complex(0.0, phase));

                std::vector<double_complex> phase_gk(num_gkvec_loc_);
                for (int igk_loc = 0; igk_loc < num_gkvec_loc_; igk_loc++) {
                    int igk = gkvec_.gvec_offset(comm.rank()) + igk_loc;
                    auto G = gkvec_.gvec(igk);
                    phase_gk[igk_loc] = std::conj(ctx_.gvec_phase_factor(G, ia) * phase_k);
                }

                for (int xi = 0; xi < ctx_.unit_cell().atom(ia).mt_lo_basis_size(); xi++) {
                    for (int igk_loc = 0; igk_loc < num_gkvec_loc_; igk_loc++) {
                        beta_gk_a_(igk_loc, ctx_.unit_cell().atom(ia).offset_lo() + xi) =
                            pw_coeffs_t_[0](igk_loc, ctx_.unit_cell().atom(ia).type().offset_lo() + xi) * phase_gk[igk_loc];
                    }
                }
            }

        }


        //matrix<double_complex>& beta_gk_t()
        //{
        //    return beta_gk_t_;
        //}

        matrix<double_complex> const& beta_gk_total()
        {
            return beta_gk_a_;
        }

        //matrix<double_complex> const& beta_gk() const
        //{
        //    return beta_gk_;
        //}

        //Unit_cell const& unit_cell() const
        //{
        //    return unit_cell_;
        //}

        //Communicator const& comm() const
        //{
        //    return comm_;
        //}

        //Gvec const& gk_vectors() const
        //{
        //    return gkvec_;
        //}

        //device_t proc_unit() const
        //{
        //    return pu_;
        //}

        //int lmax_beta() const
        //{
        //    return lmax_beta_;
        //}

        //inline int num_gkvec_loc() const
        //{
        //    return num_gkvec_loc_;
        //}

        void generate(int chunk__);

        //template <typename T>
        //inline matrix<T> inner(int chunk__, wave_functions& phi__, int idx0__, int n__)
        //{
        //    return inner<T>(chunk__, phi__, idx0__, n__, beta_gk_, beta_phi_);
        //}

        void prepare()
        {
            if (ctx_.processing_unit() == GPU) {
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
};

//inline Beta_projectors::Beta_projectors(Simulation_context const& ctx__,
//                                        Communicator const& comm__,
//                                        Gvec const& gkvec__)
//    : ctx_(ctx__)
//    , comm_(comm__)
//    , unit_cell_(ctx__.unit_cell())
//    , gkvec_(gkvec__)
//    , lmax_beta_(unit_cell_.lmax())
//    , pu_(ctx__.processing_unit())
//{
//}

//inline void Beta_projectors::generate_beta_gk_t(Simulation_context_base const& ctx__)
//{
//    PROFILE("sirius::Beta_projectors::generate_beta_gk_t");
//
//    if (!beta_projector_chunks().num_beta_t()) {
//        return;
//    }
//
//    /* allocate array */
//    beta_gk_t_ = matrix<double_complex>(gkvec_.gvec_count(comm_.rank()), beta_projector_chunks().num_beta_t());
//    
//    /* compute <G+k|beta> */
//    #pragma omp parallel for
//    for (int igkloc = 0; igkloc < gkvec_.gvec_count(comm_.rank()); igkloc++) {
//        int igk   = gkvec_.gvec_offset(comm_.rank()) + igkloc;
//        /* vs = {r, theta, phi} */
//        auto vs = SHT::spherical_coordinates(gkvec_.gkvec_cart(igk));
//        /* compute real spherical harmonics for G+k vector */
//        std::vector<double> gkvec_rlm(Utils::lmmax(lmax_beta_));
//        SHT::spherical_harmonics(lmax_beta_, vs[1], vs[2], &gkvec_rlm[0]);
//        for (int iat = 0; iat < ctx_.unit_cell().num_atom_types(); iat++) {
//            auto& atom_type = unit_cell_.atom_type(iat);
//            for (int xi = 0; xi < atom_type.mt_basis_size(); xi++) {
//                int l     = atom_type.indexb(xi).l;
//                int lm    = atom_type.indexb(xi).lm;
//                int idxrf = atom_type.indexb(xi).idxrf;
//
//                auto z = std::pow(double_complex(0, -1), l) * fourpi / std::sqrt(unit_cell_.omega());
//                beta_gk_t_(igkloc, atom_type.offset_lo() + xi) = z * gkvec_rlm[lm] * beta_radial_integrals_->value(idxrf, iat, vs[0]);
//            }
//        }
//    }
//
//    if (unit_cell_.parameters().control().print_checksum_) {
//        auto c1 = beta_gk_t_.checksum();
//        comm_.allreduce(&c1, 1);
//        if (comm_.rank() == 0) {
//            DUMP("checksum(beta_gk_t) : %18.10f %18.10f", c1.real(), c1.imag())
//        }
//    }
//}

inline void Beta_projectors::generate(int chunk__)
{
    PROFILE("sirius::Beta_projectors::generate");

    auto& bp_chunks = ctx_.beta_projector_chunks();

    if (ctx_.processing_unit() == CPU) {
        pw_coeffs_a_[0] = mdarray<double_complex, 2>(&beta_gk_a_(0, bp_chunks(chunk__).offset_),
                                                     num_gkvec_loc_, bp_chunks(chunk__).num_beta_);
    }
    #ifdef __GPU
    if (ctx_.processing_unit() == GPU) {
         pw_coeffs_a_[0] = mdarray<double_complex, 2>(&beta_gk_a_(0, beta_chunk(chunk__).offset_), beta_gk_gpu_.at<GPU>(),
                                                      num_gkvec_loc_, beta_chunk(chunk__).num_beta_);

        auto& desc = beta_chunk(chunk__).desc_;
        create_beta_gk_gpu(beta_chunk(chunk__).num_atoms_,
                           num_gkvec_loc_,
                           desc.at<GPU>(),
                           pw_coeffs_t_[0].at<GPU>(),
                           gkvec_coord_.at<GPU>(),
                           beta_chunk(chunk__).atom_pos_.at<GPU>(),
                           beta_gk_.at<GPU>());
    }
    #endif
}

} // namespace

#endif
