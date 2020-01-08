// Copyright (c) 2013-2019 Anton Kozhevnikov, Ilia Sivkov, Thomas Schulthess
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

/** \file force.cpp
 *
 *  \brief Contains implementation of sirius::Force class.
 */

#include "force.hpp"
#include "K_point/k_point.hpp"
#include "K_point/k_point_set.hpp"
#include "Density/density.hpp"
#include "Density/augmentation_operator.hpp"
#include "Potential/potential.hpp"
#include "Beta_projectors/beta_projectors.hpp"
#include "Beta_projectors/beta_projectors_gradient.hpp"
#include "non_local_functor.hpp"
#include "Hamiltonian/hamiltonian.hpp"

namespace sirius {

//using namespace geometry3d;

Force::Force(Simulation_context& ctx__, Density& density__, Potential& potential__, K_point_set& kset__)
    : ctx_(ctx__)
    , density_(density__)
    , potential_(potential__)
    , kset_(kset__)
{
}

template <typename T>
void Force::add_k_point_contribution(K_point& kpoint, mdarray<double, 2>& forces__) const
{
    /* if there are no beta projectors then get out there */
    if (ctx_.unit_cell().mt_lo_basis_size() == 0) {
        return;
    }

    Beta_projectors_gradient bp_grad(ctx_, kpoint.gkvec(), kpoint.igk_loc(), kpoint.beta_projectors());
    if (is_device_memory(ctx_.preferred_memory_t())) {
        int nbnd = ctx_.num_bands();
        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            /* allocate GPU memory */
            kpoint.spinor_wave_functions().pw_coeffs(ispn).allocate(ctx_.mem_pool(memory_t::device));
            kpoint.spinor_wave_functions().pw_coeffs(ispn).copy_to(memory_t::device, 0, nbnd);
        }
    }

    Non_local_functor<T> nlf(ctx_, bp_grad);

    nlf.add_k_point_contribution(kpoint, forces__);
    if (is_device_memory(ctx_.preferred_memory_t())) {
        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            /* deallocate GPU memory */
            kpoint.spinor_wave_functions().pw_coeffs(ispn).deallocate(memory_t::device);
        }
    }
}

void Force::compute_dmat(K_point* kp__, dmatrix<double_complex>& dm__) const
{
    dm__.zero();

    /* trivial case */
    if (!ctx_.need_sv()) {
        for (int i = 0; i < ctx_.num_fv_states(); i++) {
            dm__.set(i, i, double_complex(kp__->band_occupancy(i, 0), 0));
        }
    } else {
        if (ctx_.num_mag_dims() != 3) {
            dmatrix<double_complex> ev1(ctx_.num_fv_states(), ctx_.num_fv_states(), ctx_.blacs_grid(),
                                        ctx_.cyclic_block_size(), ctx_.cyclic_block_size());
            for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                auto& ev = kp__->sv_eigen_vectors(ispn);
                /* multiply second-variational eigen-vectors with band occupancies */
                for (int j = 0; j < ev.num_cols_local(); j++) {
                    /* up- or dn- band index */
                    int jb = ev.icol(j);
                    for (int i = 0; i < ev.num_rows_local(); i++) {
                        ev1(i, j) = std::conj(ev(i, j)) * kp__->band_occupancy(jb, ispn);
                    }
                }

                linalg(linalg_t::scalapack).gemm('N', 'T', ctx_.num_fv_states(), ctx_.num_fv_states(), ctx_.num_bands(),
                                                  &linalg_const<double_complex>::one(), ev1, 0, 0, ev, 0, 0,
                                                  &linalg_const<double_complex>::one(), dm__, 0, 0);
            }
        } else {
            dmatrix<double_complex> ev1(ctx_.num_bands(), ctx_.num_bands(), ctx_.blacs_grid(), ctx_.cyclic_block_size(),
                                        ctx_.cyclic_block_size());
            auto& ev = kp__->sv_eigen_vectors(0);
            /* multiply second-variational eigen-vectors with band occupancies */
            for (int j = 0; j < ev.num_cols_local(); j++) {
                /* band index */
                int jb = ev.icol(j);
                for (int i = 0; i < ev.num_rows_local(); i++) {
                    ev1(i, j) = std::conj(ev(i, j)) * kp__->band_occupancy(jb, 0);
                }
            }
            for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                int offs = ispn * ctx_.num_fv_states();

                linalg(linalg_t::scalapack).gemm('N', 'T', ctx_.num_fv_states(), ctx_.num_fv_states(), ctx_.num_bands(),
                                                  &linalg_const<double_complex>::one(), ev1, offs, 0, ev, offs, 0,
                                                  &linalg_const<double_complex>::one(), dm__, 0, 0);
            }
        }
    }
}

mdarray<double, 2> const& Force::calc_forces_total()
{
    forces_total_ = mdarray<double, 2>(3, ctx_.unit_cell().num_atoms());
    if (ctx_.full_potential()) {
        calc_forces_rho();
        calc_forces_hf();
        calc_forces_ibs();
        for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
            for (int x : {0, 1, 2}) {
                forces_total_(x, ia) = forces_ibs_(x, ia) + forces_hf_(x, ia) + forces_rho_(x, ia);
            }
        }
    } else {
        calc_forces_vloc();
        calc_forces_us();
        calc_forces_nonloc();
        calc_forces_core();
        calc_forces_ewald();
        calc_forces_scf_corr();
        calc_forces_hubbard();

        forces_total_ = mdarray<double, 2>(3, ctx_.unit_cell().num_atoms());
        for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
            for (int x : {0, 1, 2}) {
                forces_total_(x, ia) = forces_vloc_(x, ia) + forces_us_(x, ia) + forces_nonloc_(x, ia) +
                                       forces_core_(x, ia) + forces_ewald_(x, ia) + forces_scf_corr_(x, ia) +
                                       forces_hubbard_(x, ia);
            }
        }
    }
    return forces_total_;
}

mdarray<double, 2> const& Force::calc_forces_ibs()
{
    forces_ibs_ = mdarray<double, 2>(3, ctx_.unit_cell().num_atoms());
    forces_ibs_.zero();

    mdarray<double, 2> ffac(ctx_.unit_cell().num_atom_types(), ctx_.gvec().num_shells());
    #pragma omp parallel for
    for (int igs = 0; igs < ctx_.gvec().num_shells(); igs++) {
        for (int iat = 0; iat < ctx_.unit_cell().num_atom_types(); iat++) {
            ffac(iat, igs) = unit_step_function_form_factors(ctx_.unit_cell().atom_type(iat).mt_radius(),
                                                             ctx_.gvec().shell_len(igs));
        }
    }

    Hamiltonian0 H0(potential_);
    for (int ikloc = 0; ikloc < kset_.spl_num_kpoints().local_size(); ikloc++) {
        int ik = kset_.spl_num_kpoints(ikloc);
        auto hk = H0(*kset_[ik]);
        add_ibs_force(kset_[ik], hk, ffac, forces_ibs_);
    }
    ctx_.comm().allreduce(&forces_ibs_(0, 0), (int)forces_ibs_.size());
    symmetrize(forces_ibs_);

    if (ctx_.control().verbosity_ > 2 && ctx_.comm().rank() == 0) {
        std::printf("ibs force\n");
        for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
            std::printf("ia : %i, IBS : %12.6f %12.6f %12.6f\n", ia, forces_ibs_(0, ia), forces_ibs_(1, ia),
                   forces_ibs_(2, ia));
        }
    }

    return forces_ibs_;
}

mdarray<double, 2> const& Force::calc_forces_rho()
{

    forces_rho_ = mdarray<double, 2>(3, ctx_.unit_cell().num_atoms());
    forces_rho_.zero();
    for (int ialoc = 0; ialoc < ctx_.unit_cell().spl_num_atoms().local_size(); ialoc++) {
        int ia = ctx_.unit_cell().spl_num_atoms(ialoc);
        auto g = gradient(density_.density_mt(ialoc));
        for (int x = 0; x < 3; x++) {
            forces_rho_(x, ia) = inner(potential_.effective_potential_mt(ialoc), g[x]);
        }
    }
    ctx_.comm().allreduce(&forces_rho_(0, 0), (int)forces_rho_.size());
    symmetrize(forces_rho_);

    if (ctx_.control().verbosity_ > 2 && ctx_.comm().rank() == 0) {
        std::printf("rho force\n");
        for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
            std::printf("ia : %i, density contribution : %12.6f %12.6f %12.6f\n", ia, forces_rho_(0, ia), forces_rho_(1, ia),
                   forces_rho_(2, ia));
        }
    }
    return forces_rho_;
}

mdarray<double, 2> const& Force::calc_forces_hf()
{
    forces_hf_ = mdarray<double, 2>(3, ctx_.unit_cell().num_atoms());
    forces_hf_.zero();

    for (int ialoc = 0; ialoc < ctx_.unit_cell().spl_num_atoms().local_size(); ialoc++) {
        int ia = ctx_.unit_cell().spl_num_atoms(ialoc);
        auto g = gradient(potential_.hartree_potential_mt(ialoc));
        for (int x = 0; x < 3; x++) {
            forces_hf_(x, ia) = ctx_.unit_cell().atom(ia).zn() * g[x](0, 0) * y00;
        }
    }
    ctx_.comm().allreduce(&forces_hf_(0, 0), (int)forces_hf_.size());
    symmetrize(forces_hf_);

    if (ctx_.control().verbosity_ > 2 && ctx_.comm().rank() == 0) {
        std::printf("H-F force\n");
        for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
            std::printf("ia : %i, Hellmann–Feynman : %12.6f %12.6f %12.6f\n", ia, forces_hf_(0, ia), forces_hf_(1, ia),
                   forces_hf_(2, ia));
        }
    }

    return forces_hf_;
}

mdarray<double, 2> const& Force::calc_forces_hubbard()
{
    PROFILE("sirius::Force::hubbard_force");
    forces_hubbard_ = mdarray<double, 2>(3, ctx_.unit_cell().num_atoms());
    forces_hubbard_.zero();

    if (ctx_.hubbard_correction()) {
        /* we can probably task run this in a task fashion */
        Q_operator q_op(ctx_);

        for (int ikloc = 0; ikloc < kset_.spl_num_kpoints().local_size(); ikloc++) {

            int ik  = kset_.spl_num_kpoints(ikloc);
            auto kp = kset_[ik];
            if (ctx_.num_mag_dims() == 3)
                TERMINATE("Hubbard forces are only implemented for the simple hubbard correction.");

            hubbard_force_add_k_contribution_colinear(*kp, q_op, forces_hubbard_);
        }

        /* global reduction */
        kset_.comm().allreduce(forces_hubbard_.at(memory_t::host), 3 * ctx_.unit_cell().num_atoms());
    }

    return forces_hubbard_;
}

mdarray<double, 2> const& Force::calc_forces_ewald()
{
    PROFILE("sirius::Force::calc_forces_ewald");

    forces_ewald_ = mdarray<double, 2>(3, ctx_.unit_cell().num_atoms());
    forces_ewald_.zero();

    Unit_cell& unit_cell = ctx_.unit_cell();

    double alpha = ctx_.ewald_lambda();

    double prefac = (ctx_.gvec().reduced() ? 4.0 : 2.0) * (twopi / unit_cell.omega());

    int ig0{0};
    if (ctx_.comm().rank() == 0) {
        ig0 = 1;
    }

    mdarray<double_complex, 1> rho_tmp(ctx_.gvec().count());
    rho_tmp.zero();
    #pragma omp parallel for schedule(static)
    for (int igloc = ig0; igloc < ctx_.gvec().count(); igloc++) {
        int ig = ctx_.gvec().offset() + igloc;

        double_complex rho(0, 0);

        for (int ja = 0; ja < unit_cell.num_atoms(); ja++) {
            rho += ctx_.gvec_phase_factor(ig, ja) * static_cast<double>(unit_cell.atom(ja).zn());
        }

        rho_tmp[igloc] = std::conj(rho);
    }

    #pragma omp parallel for
    for (int ja = 0; ja < unit_cell.num_atoms(); ja++) {
        for (int igloc = ig0; igloc < ctx_.gvec().count(); igloc++) {
            int ig = ctx_.gvec().offset() + igloc;

            double g2 = std::pow(ctx_.gvec().gvec_len(ig), 2);

            /* cartesian form for getting cartesian force components */
            vector3d<double> gvec_cart = ctx_.gvec().gvec_cart<index_domain_t::local>(igloc);
            double_complex rho(0, 0);

            double scalar_part = prefac * (rho_tmp[igloc] * ctx_.gvec_phase_factor(ig, ja)).imag() *
                                 static_cast<double>(unit_cell.atom(ja).zn()) * std::exp(-g2 / (4 * alpha)) / g2;

            for (int x : {0, 1, 2}) {
                forces_ewald_(x, ja) += scalar_part * gvec_cart[x];
            }
        }
    }

    ctx_.comm().allreduce(&forces_ewald_(0, 0), 3 * ctx_.unit_cell().num_atoms());

    double invpi = 1. / pi;

    #pragma omp parallel for
    for (int ia = 0; ia < unit_cell.num_atoms(); ia++) {
        for (int i = 1; i < unit_cell.num_nearest_neighbours(ia); i++) {
            int ja = unit_cell.nearest_neighbour(i, ia).atom_id;

            double d  = unit_cell.nearest_neighbour(i, ia).distance;
            double d2 = d * d;

            vector3d<double> t =
                unit_cell.lattice_vectors() * vector3d<int>(unit_cell.nearest_neighbour(i, ia).translation);

            double scalar_part =
                static_cast<double>(unit_cell.atom(ia).zn() * unit_cell.atom(ja).zn()) / d2 *
                (std::erfc(std::sqrt(alpha) * d) / d + 2.0 * std::sqrt(alpha * invpi) * std::exp(-d2 * alpha));

            for (int x : {0, 1, 2}) {
                forces_ewald_(x, ia) += scalar_part * t[x];
            }
        }
    }

    return forces_ewald_;
}

mdarray<double, 2> const& Force::calc_forces_us()
{
    PROFILE("sirius::Force::calc_forces_us");

    forces_us_ = mdarray<double, 2>(3, ctx_.unit_cell().num_atoms());
    forces_us_.zero();

    potential_.fft_transform(-1);

    Unit_cell& unit_cell = ctx_.unit_cell();

    double reduce_g_fact = ctx_.gvec().reduced() ? 2.0 : 1.0;

    linalg_t la{linalg_t::none};

    memory_pool* mp{nullptr};
    switch (ctx_.processing_unit()) {
        case device_t::CPU: {
            mp = &ctx_.mem_pool(memory_t::host);
            la = linalg_t::blas;
            break;
        }
        case device_t::GPU: {
            mp = &ctx_.mem_pool(memory_t::host_pinned);
            la = linalg_t::cublasxt;
            break;
        }
    }

    /* over atom types */
    for (int iat = 0; iat < unit_cell.num_atom_types(); iat++) {
        auto& atom_type = unit_cell.atom_type(iat);

        auto aug_op = ctx_.augmentation_op(iat);

        if (!aug_op) {
            continue;
        }

        int nbf = atom_type.mt_basis_size();

        /* get auxiliary density matrix */
        auto dm = density_.density_matrix_aux(iat);

        mdarray<double, 2> v_tmp(atom_type.num_atoms(), ctx_.gvec().count() * 2, *mp);
        mdarray<double, 2> tmp(nbf * (nbf + 1) / 2, atom_type.num_atoms(), *mp);

        /* over spin components, can be from 1 to 4*/
        for (int ispin = 0; ispin < ctx_.num_mag_dims() + 1; ispin++) {
            /* over 3 components of the force/G - vectors */
            for (int ivec = 0; ivec < 3; ivec++) {
                /* over local rank G vectors */
                #pragma omp parallel for schedule(static)
                for (int igloc = 0; igloc < ctx_.gvec().count(); igloc++) {
                    int ig   = ctx_.gvec().offset() + igloc;
                    auto gvc = ctx_.gvec().gvec_cart<index_domain_t::local>(igloc);
                    for (int ia = 0; ia < atom_type.num_atoms(); ia++) {
                        /* here we write in v_tmp  -i * G * exp[ iGRn] Veff(G)
                         * but in formula we have   i * G * exp[-iGRn] Veff*(G)
                         * the differences because we unfold complex array in the real one
                         * and need negative imagine part due to a multiplication law of complex numbers */
                        auto z = double_complex(0, -gvc[ivec]) * ctx_.gvec_phase_factor(ig, atom_type.atom_id(ia)) *
                                 potential_.component(ispin).f_pw_local(igloc);
                        v_tmp(ia, 2 * igloc)     = z.real();
                        v_tmp(ia, 2 * igloc + 1) = z.imag();
                    }
                }

                /* multiply tmp matrices, or sum over G */
                linalg(la).gemm('N', 'T', nbf * (nbf + 1) / 2, atom_type.num_atoms(), 2 * ctx_.gvec().count(),
                    &linalg_const<double>::one(),
                    aug_op->q_pw().at(memory_t::host), aug_op->q_pw().ld(),
                    v_tmp.at(memory_t::host), v_tmp.ld(),
                    &linalg_const<double>::zero(),
                    tmp.at(memory_t::host), tmp.ld());

                #pragma omp parallel for
                for (int ia = 0; ia < atom_type.num_atoms(); ia++) {
                    for (int i = 0; i < nbf * (nbf + 1) / 2; i++) {
                        forces_us_(ivec, atom_type.atom_id(ia)) += ctx_.unit_cell().omega() * reduce_g_fact *
                                                                   dm(i, ia, ispin) * aug_op->sym_weight(i) * tmp(i, ia);
                    }
                }
            }
        }
    }

    ctx_.comm().allreduce(&forces_us_(0, 0), 3 * ctx_.unit_cell().num_atoms());

    return forces_us_;
}

mdarray<double, 2> const& Force::calc_forces_scf_corr()
{
    PROFILE("sirius::Force::calc_forces_scf_corr");

    forces_scf_corr_ = mdarray<double, 2>(3, ctx_.unit_cell().num_atoms());
    forces_scf_corr_.zero();

    /* get main arrays */
    auto& dveff = potential_.dveff();

    Unit_cell& unit_cell = ctx_.unit_cell();

    Gvec const& gvec = ctx_.gvec();

    int gvec_count  = gvec.count();
    int gvec_offset = gvec.offset();

    double fact = gvec.reduced() ? 2.0 : 1.0;

    int ig0 = (ctx_.comm().rank() == 0) ? 1 : 0;

    auto& ri = ctx_.ps_rho_ri();

    #pragma omp parallel for
    for (int ia = 0; ia < unit_cell.num_atoms(); ia++) {
        Atom& atom = unit_cell.atom(ia);

        int iat = atom.type_id();

        for (int igloc = ig0; igloc < gvec_count; igloc++) {
            int ig = gvec_offset + igloc;

            /* cartesian form for getting cartesian force components */
            vector3d<double> gvec_cart = gvec.gvec_cart<index_domain_t::local>(igloc);

            /* scalar part of a force without multipying by G-vector */
            double_complex z = fact * fourpi * ri.value<int>(iat, gvec.gvec_len(ig)) *
                               std::conj(dveff.f_pw_local(igloc) * ctx_.gvec_phase_factor(ig, ia));

            /* get force components multiplying by cartesian G-vector */
            for (int x : {0, 1, 2}) {
                forces_scf_corr_(x, ia) -= (gvec_cart[x] * z).imag();
            }
        }
    }
    ctx_.comm().allreduce(&forces_scf_corr_(0, 0), 3 * ctx_.unit_cell().num_atoms());

    return forces_scf_corr_;
}

mdarray<double, 2> const& Force::calc_forces_core()
{
    PROFILE("sirius::Force::calc_forces_core");

    forces_core_ = mdarray<double, 2>(3, ctx_.unit_cell().num_atoms());
    forces_core_.zero();

    /* get main arrays */
    auto& xc_pot = potential_.xc_potential();

    /* transform from real space to reciprocal */
    xc_pot.fft_transform(-1);

    Unit_cell& unit_cell = ctx_.unit_cell();

    Gvec const& gvecs = ctx_.gvec();

    int gvec_count  = gvecs.count();
    int gvec_offset = gvecs.offset();

    double fact = gvecs.reduced() ? 2.0 : 1.0;

    auto& ri = ctx_.ps_core_ri();

    /* here the calculations are in lattice vectors space */
    #pragma omp parallel for
    for (int ia = 0; ia < unit_cell.num_atoms(); ia++) {
        Atom& atom = unit_cell.atom(ia);
        if (atom.type().ps_core_charge_density().empty()) {
            continue;
        }
        int iat = atom.type_id();

        for (int igloc = 0; igloc < gvec_count; igloc++) {
            int ig = gvec_offset + igloc;

            if (ig == 0) {
                continue;
            }

            /* cartesian form for getting cartesian force components */
            vector3d<double> gvec_cart = gvecs.gvec_cart<index_domain_t::local>(igloc);

            /* scalar part of a force without multipying by G-vector */
            double_complex z = fact * fourpi * ri.value<int>(iat, gvecs.gvec_len(ig)) *
                               std::conj(xc_pot.f_pw_local(igloc) * ctx_.gvec_phase_factor(ig, ia));

            /* get force components multiplying by cartesian G-vector */
            for (int x : {0, 1, 2}) {
                forces_core_(x, ia) -= (gvec_cart[x] * z).imag();
            }
        }
    }
    ctx_.comm().allreduce(&forces_core_(0, 0), 3 * ctx_.unit_cell().num_atoms());

    return forces_core_;
}

void Force::hubbard_force_add_k_contribution_colinear(K_point& kp__, Q_operator& q_op__, mdarray<double, 2>& forceh_)
{
    mdarray<double_complex, 6> dn(potential_.U().max_number_of_orbitals_per_atom(), potential_.U().max_number_of_orbitals_per_atom(), 2,
                                  ctx_.unit_cell().num_atoms(), 3, ctx_.unit_cell().num_atoms());

    potential_.U().compute_occupancies_derivatives(kp__, q_op__, dn);

    #pragma omp parallel for
    for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
        /* compute the derivative of the occupancies numbers */
        for (int dir = 0; dir < 3; dir++) {
            double d{0};
            for (int ia1 = 0; ia1 < ctx_.unit_cell().num_atoms(); ia1++) {
                auto const& atom = ctx_.unit_cell().atom(ia1);
                if (atom.type().hubbard_correction()) {
                    int const lmax_at = 2 * atom.type().hubbard_orbital(0).l + 1;
                    for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                        for (int m1 = 0; m1 < lmax_at; m1++) {
                            for (int m2 = 0; m2 < lmax_at; m2++) {
                                d += std::real(potential_.U().U(m2, m1, ispn, ia1) * dn(m1, m2, ispn, ia1, dir, ia));
                            }
                        }
                    }
                }
            }
            forceh_(dir, ia) -= d;
        }
    }
}

mdarray<double, 2> const& Force::calc_forces_vloc()
{
    PROFILE("sirius::Force::calc_forces_vloc");

    forces_vloc_ = mdarray<double, 2>(3, ctx_.unit_cell().num_atoms());
    forces_vloc_.zero();

    auto& valence_rho = density_.rho();

    auto& ri = ctx_.vloc_ri();

    Unit_cell& unit_cell = ctx_.unit_cell();

    Gvec const& gvecs = ctx_.gvec();

    int gvec_count  = gvecs.gvec_count(ctx_.comm().rank());
    int gvec_offset = gvecs.gvec_offset(ctx_.comm().rank());

    double fact = valence_rho.gvec().reduced() ? 2.0 : 1.0;

    /* here the calculations are in lattice vectors space */
    #pragma omp parallel for
    for (int ia = 0; ia < unit_cell.num_atoms(); ia++) {
        Atom& atom = unit_cell.atom(ia);

        int iat = atom.type_id();

        for (int igloc = 0; igloc < gvec_count; igloc++) {
            int ig = gvec_offset + igloc;

            /* cartesian form for getting cartesian force components */
            vector3d<double> gvec_cart = gvecs.gvec_cart<index_domain_t::local>(igloc);

            /* scalar part of a force without multiplying by G-vector */
            double_complex z = fact * fourpi * ri.value(iat, gvecs.gvec_len(ig)) *
                               std::conj(valence_rho.f_pw_local(igloc)) * std::conj(ctx_.gvec_phase_factor(ig, ia));

            /* get force components multiplying by cartesian G-vector  */
            for (int x : {0, 1, 2}) {
                forces_vloc_(x, ia) -= (gvec_cart[x] * z).imag();
            }
        }
    }

    ctx_.comm().allreduce(&forces_vloc_(0, 0), 3 * ctx_.unit_cell().num_atoms());

    return forces_vloc_;
}

mdarray<double, 2> const& Force::calc_forces_usnl()
{
    calc_forces_us();
    calc_forces_nonloc();

    forces_usnl_ = mdarray<double, 2>(3, ctx_.unit_cell().num_atoms());
    for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
        for (int x : {0, 1, 2}) {
            forces_usnl_(x, ia) = forces_us_(x, ia) + forces_nonloc_(x, ia);
        }
    }

    return forces_usnl_;
}

void Force::add_ibs_force(K_point* kp__, Hamiltonian_k& Hk__, mdarray<double, 2>& ffac__, mdarray<double, 2>& forcek__) const
{
    PROFILE("sirius::Force::ibs_force");

    auto& uc = ctx_.unit_cell();

    auto& bg = ctx_.blacs_grid();

    auto bs = ctx_.cyclic_block_size();

    auto nfv = ctx_.num_fv_states();

    auto ngklo = kp__->gklo_basis_size();

    /* compute density matrix for a k-point */
    dmatrix<double_complex> dm(nfv, nfv, bg, bs, bs);
    compute_dmat(kp__, dm);

    /* first-variational eigen-vectors in scalapack distribution */
    auto& fv_evec = kp__->fv_eigen_vectors();

    dmatrix<double_complex> h(ngklo, ngklo, bg, bs, bs);
    dmatrix<double_complex> o(ngklo, ngklo, bg, bs, bs);

    dmatrix<double_complex> h1(ngklo, ngklo, bg, bs, bs);
    dmatrix<double_complex> o1(ngklo, ngklo, bg, bs, bs);

    dmatrix<double_complex> zm1(ngklo, nfv, bg, bs, bs);
    dmatrix<double_complex> zf(nfv, nfv, bg, bs, bs);

    mdarray<double_complex, 2> alm_row(kp__->num_gkvec_row(), uc.max_mt_aw_basis_size());
    mdarray<double_complex, 2> alm_col(kp__->num_gkvec_col(), uc.max_mt_aw_basis_size());
    mdarray<double_complex, 2> halm_col(kp__->num_gkvec_col(), uc.max_mt_aw_basis_size());

    for (int ia = 0; ia < uc.num_atoms(); ia++) {
        h.zero();
        o.zero();

        auto& atom = uc.atom(ia);
        auto& type = atom.type();

        /* generate matching coefficients for current atom */
        kp__->alm_coeffs_row().generate<true>(atom, alm_row);
        kp__->alm_coeffs_col().generate<false>(atom, alm_col);

        /* setup apw-lo and lo-apw blocks */
        Hk__.set_fv_h_o_apw_lo(atom, ia, alm_row, alm_col, h, o);

        /* apply MT Hamiltonian to column coefficients */
        Hk__.H0().apply_hmt_to_apw<spin_block_t::nm>(atom, kp__->num_gkvec_col(), alm_col, halm_col);

        /* apw-apw block of the overlap matrix */
        linalg(linalg_t::blas).gemm('N', 'T', kp__->num_gkvec_row(), kp__->num_gkvec_col(), type.mt_aw_basis_size(),
            &linalg_const<double_complex>::one(), alm_row.at(memory_t::host), alm_row.ld(),
            alm_col.at(memory_t::host), alm_col.ld(), &linalg_const<double_complex>::zero(),
            o.at(memory_t::host), o.ld());

        /* apw-apw block of the Hamiltonian matrix */
        linalg(linalg_t::blas).gemm('N', 'T', kp__->num_gkvec_row(), kp__->num_gkvec_col(), type.mt_aw_basis_size(),
            &linalg_const<double_complex>::one(), alm_row.at(memory_t::host), alm_row.ld(), halm_col.at(memory_t::host),
            halm_col.ld(), &linalg_const<double_complex>::zero(), h.at(memory_t::host), h.ld());

        int iat = type.id();

        for (int igk_col = 0; igk_col < kp__->num_gkvec_col(); igk_col++) { // loop over columns
            auto gvec_col       = kp__->gkvec().gvec(kp__->igk_col(igk_col));
            auto gkvec_col_cart = kp__->gkvec().gkvec_cart<index_domain_t::global>(kp__->igk_col(igk_col));
            for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++) { // for each column loop over rows
                auto gvec_row       = kp__->gkvec().gvec(kp__->igk_row(igk_row));
                auto gkvec_row_cart = kp__->gkvec().gkvec_cart<index_domain_t::global>(kp__->igk_row(igk_row));

                int ig12 = ctx_.gvec().index_g12(gvec_row, gvec_col);

                int igs = ctx_.gvec().shell(ig12);

                auto zt = std::conj(ctx_.gvec_phase_factor(ig12, ia)) * ffac__(iat, igs) * fourpi / uc.omega();

                double t1 = 0.5 * dot(gkvec_row_cart, gkvec_col_cart);

                h(igk_row, igk_col) -= t1 * zt;
                o(igk_row, igk_col) -= zt;
            }
        }

        for (int x = 0; x < 3; x++) {
            for (int igk_col = 0; igk_col < kp__->num_gkvec_col(); igk_col++) { // loop over columns
                auto gvec_col = kp__->gkvec().gvec(kp__->igk_col(igk_col));
                for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++) { // loop over rows
                    auto gvec_row = kp__->gkvec().gvec(kp__->igk_row(igk_row));
                    /* compute index of G-G' */
                    int ig12 = ctx_.gvec().index_g12(gvec_row, gvec_col);
                    /* get G-G' */
                    auto vg = ctx_.gvec().gvec_cart<index_domain_t::global>(ig12);
                    /* multiply by i(G-G') */
                    h1(igk_row, igk_col) = double_complex(0.0, vg[x]) * h(igk_row, igk_col);
                    /* multiply by i(G-G') */
                    o1(igk_row, igk_col) = double_complex(0.0, vg[x]) * o(igk_row, igk_col);
                }
            }

            for (int icol = 0; icol < kp__->num_lo_col(); icol++) {
                for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++) {
                    auto gkvec_row_cart = kp__->gkvec().gkvec_cart<index_domain_t::global>(kp__->igk_row(igk_row));
                    /* multiply by i(G+k) */
                    h1(igk_row, icol + kp__->num_gkvec_col()) =
                        double_complex(0.0, gkvec_row_cart[x]) * h(igk_row, icol + kp__->num_gkvec_col());
                    /* multiply by i(G+k) */
                    o1(igk_row, icol + kp__->num_gkvec_col()) =
                        double_complex(0.0, gkvec_row_cart[x]) * o(igk_row, icol + kp__->num_gkvec_col());
                }
            }

            for (int irow = 0; irow < kp__->num_lo_row(); irow++) {
                for (int igk_col = 0; igk_col < kp__->num_gkvec_col(); igk_col++) {
                    auto gkvec_col_cart = kp__->gkvec().gkvec_cart<index_domain_t::global>(kp__->igk_col(igk_col));
                    /* multiply by i(G+k) */
                    h1(irow + kp__->num_gkvec_row(), igk_col) =
                        double_complex(0.0, -gkvec_col_cart[x]) * h(irow + kp__->num_gkvec_row(), igk_col);
                    /* multiply by i(G+k) */
                    o1(irow + kp__->num_gkvec_row(), igk_col) =
                        double_complex(0.0, -gkvec_col_cart[x]) * o(irow + kp__->num_gkvec_row(), igk_col);
                }
            }

            /* zm1 = dO * V */
            linalg(linalg_t::scalapack).gemm('N', 'N', ngklo, nfv, ngklo, &linalg_const<double_complex>::one(),
                o1, 0, 0, fv_evec, 0, 0, &linalg_const<double_complex>::zero(), zm1, 0, 0);
            /* multiply by energy: zm1 = E * (dO * V)  */
            for (int i = 0; i < zm1.num_cols_local(); i++) {
                int ist = zm1.icol(i);
                for (int j = 0; j < kp__->gklo_basis_size_row(); j++) {
                    zm1(j, i) *= kp__->fv_eigen_value(ist);
                }
            }
            /* compute zm1 = dH * V - E * (dO * V) */
            linalg(linalg_t::scalapack).gemm('N', 'N', ngklo, nfv, ngklo, &linalg_const<double_complex>::one(),
                h1, 0, 0, fv_evec, 0, 0, &linalg_const<double_complex>::m_one(), zm1, 0, 0);

            /* compute zf = V^{+} * zm1 = V^{+} * (dH * V - E * (dO * V)) */
            linalg(linalg_t::scalapack).gemm('C', 'N', nfv, nfv, ngklo, &linalg_const<double_complex>::one(),
                fv_evec, 0, 0, zm1, 0, 0, &linalg_const<double_complex>::zero(), zf, 0, 0);

            for (int i = 0; i < dm.num_cols_local(); i++) {
                for (int j = 0; j < dm.num_rows_local(); j++) {
                    forcek__(x, ia) += kp__->weight() * std::real(dm(j, i) * zf(j, i));
                }
            }
        }
    } // ia
}

void Force::symmetrize(mdarray<double, 2>& forces__) const
{
    if (!ctx_.use_symmetry()) {
        return;
    }

    mdarray<double, 2> sym_forces(3, ctx_.unit_cell().num_atoms());
    sym_forces.zero();

    auto& lattice_vectors         = ctx_.unit_cell().symmetry().lattice_vectors();
    auto& inverse_lattice_vectors = ctx_.unit_cell().symmetry().inverse_lattice_vectors();

    sym_forces.zero();

    #pragma omp parallel for
    for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
        vector3d<double> cart_force(&forces__(0, ia));
        vector3d<double> lat_force =
            inverse_lattice_vectors * (cart_force / (double)ctx_.unit_cell().symmetry().num_mag_sym());

        for (int isym = 0; isym < ctx_.unit_cell().symmetry().num_mag_sym(); isym++) {
            int ja                     = ctx_.unit_cell().symmetry().sym_table(ia, isym);
            auto& R                    = ctx_.unit_cell().symmetry().magnetic_group_symmetry(isym).spg_op.R;
            vector3d<double> rot_force = lattice_vectors * (R * lat_force);

            #pragma omp atomic update
            sym_forces(0, ja) += rot_force[0];

            #pragma omp atomic update
            sym_forces(1, ja) += rot_force[1];

            #pragma omp atomic update
            sym_forces(2, ja) += rot_force[2];
        }
    }
    sym_forces >> forces__;
}

mdarray<double, 2> const& Force::calc_forces_nonloc()
{
    PROFILE("sirius::Force::calc_forces_nonloc");

    forces_nonloc_ = mdarray<double, 2>(3, ctx_.unit_cell().num_atoms());
    forces_nonloc_.zero();

    auto& spl_num_kp = kset_.spl_num_kpoints();

    for (int ikploc = 0; ikploc < spl_num_kp.local_size(); ikploc++) {
        K_point* kp = kset_[spl_num_kp[ikploc]];

        if (ctx_.gamma_point()) {
            add_k_point_contribution<double>(*kp, forces_nonloc_);
        } else {
            add_k_point_contribution<double_complex>(*kp, forces_nonloc_);
        }
    }

    ctx_.comm().allreduce(&forces_nonloc_(0, 0), 3 * ctx_.unit_cell().num_atoms());

    symmetrize(forces_nonloc_);

    return forces_nonloc_;
}

void Force::print_info()
{
    if (ctx_.comm().rank() == 0) {
        auto print_forces = [&](mdarray<double, 2> const& forces) {
            for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
                std::printf("atom %4i    force = %15.7f  %15.7f  %15.7f \n", ctx_.unit_cell().atom(ia).type_id(),
                       forces(0, ia), forces(1, ia), forces(2, ia));
            }
        };

        std::printf("===== total Forces in Ha/bohr =====\n");
        print_forces(forces_total());

        std::printf("===== ultrasoft contribution from Qij =====\n");
        print_forces(forces_us());

        std::printf("===== non-local contribution from Beta-projectors =====\n");
        print_forces(forces_nonloc());

        std::printf("===== contribution from local potential =====\n");
        print_forces(forces_vloc());

        std::printf("===== contribution from core density =====\n");
        print_forces(forces_core());

        std::printf("===== Ewald forces from ions =====\n");
        print_forces(forces_ewald());

        if (ctx_.hubbard_correction()) {
            std::printf("===== Ewald forces from hubbard correction =====\n");
            print_forces(forces_hubbard());
        }
    }
}

} // namespace sirius
