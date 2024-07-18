/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file force.cpp
 *
 *  \brief Contains implementation of sirius::Force class.
 */

#include "force.hpp"
#include "k_point/k_point.hpp"
#include "k_point/k_point_set.hpp"
#include "density/density.hpp"
#include "density/augmentation_operator.hpp"
#include "potential/potential.hpp"
#include "beta_projectors/beta_projectors.hpp"
#include "beta_projectors/beta_projectors_gradient.hpp"
#include "non_local_functor.hpp"
#include "hamiltonian/hamiltonian.hpp"
#include "symmetry/symmetrize_forces.hpp"
#include "lapw/step_function.hpp"
#include <string>
#include <iomanip>
#include <vector>
#include <complex>
#include <algorithm>
#include <cmath>

namespace sirius {

Force::Force(Simulation_context& ctx__, Density& density__, Potential& potential__, K_point_set& kset__)
    : ctx_(ctx__)
    , density_(density__)
    , potential_(potential__)
    , kset_(kset__)
{
}

template <typename T>
void
Force::calc_forces_nonloc_aux()
{
    forces_nonloc_ = mdarray<double, 2>({3, ctx_.unit_cell().num_atoms()});
    forces_nonloc_.zero();

    auto& spl_num_kp = kset_.spl_num_kpoints();

    for (auto it : spl_num_kp) {
        auto* kp = kset_.get<T>(it.i);

        if (ctx_.gamma_point()) {
            add_k_point_contribution<T, T>(*kp, forces_nonloc_);
        } else {
            add_k_point_contribution<T, std::complex<T>>(*kp, forces_nonloc_);
        }
    }

    ctx_.comm().allreduce(&forces_nonloc_(0, 0), 3 * ctx_.unit_cell().num_atoms());

    symmetrize_forces(ctx_.unit_cell(), forces_nonloc_);
}

mdarray<double, 2> const&
Force::calc_forces_nonloc()
{
    PROFILE("sirius::Force::calc_forces_nonloc");

    if (ctx_.cfg().parameters().precision_wf() == "fp32") {
#if defined(SIRIUS_USE_FP32)
        this->calc_forces_nonloc_aux<float>();
#endif
    } else {
        this->calc_forces_nonloc_aux<double>();
    }
    return forces_nonloc_;
}

template <typename T, typename F>
void
Force::add_k_point_contribution(K_point<T>& kp__, mdarray<double, 2>& forces__) const
{
    /* if there are no beta projectors then get out there */
    if (ctx_.unit_cell().max_mt_basis_size() == 0) {
        return;
    }

    Beta_projectors_gradient<T> bp_grad(ctx_, kp__.gkvec(), kp__.beta_projectors());
    auto mem = ctx_.processing_unit_memory_t();
    auto mg  = kp__.spinor_wave_functions().memory_guard(mem, wf::copy_to::device);

    mdarray<real_type<F>, 2> f({3, ctx_.unit_cell().num_atoms()});
    f.zero();

    add_k_point_contribution_nonlocal<T, F>(ctx_, bp_grad, kp__, f);

    for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
        for (int x : {0, 1, 2}) {
            forces__(x, ia) += f(x, ia);
        }
    }
}

void
Force::compute_dmat(K_point<double>* kp__, la::dmatrix<std::complex<double>>& dm__) const
{
    dm__.zero();

    /* trivial case */
    if (!ctx_.need_sv()) {
        for (int i = 0; i < ctx_.num_fv_states(); i++) {
            dm__.set(i, i, std::complex<double>(kp__->band_occupancy(i, 0), 0));
        }
    } else {
        if (ctx_.num_mag_dims() != 3) {
            la::dmatrix<std::complex<double>> ev1(ctx_.num_fv_states(), ctx_.num_fv_states(), ctx_.blacs_grid(),
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

                la::wrap(la::lib_t::scalapack)
                        .gemm('N', 'T', ctx_.num_fv_states(), ctx_.num_fv_states(), ctx_.num_bands(),
                              &la::constant<std::complex<double>>::one(), ev1, 0, 0, ev, 0, 0,
                              &la::constant<std::complex<double>>::one(), dm__, 0, 0);
            }
        } else {
            la::dmatrix<std::complex<double>> ev1(ctx_.num_bands(), ctx_.num_bands(), ctx_.blacs_grid(),
                                                  ctx_.cyclic_block_size(), ctx_.cyclic_block_size());
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

                la::wrap(la::lib_t::scalapack)
                        .gemm('N', 'T', ctx_.num_fv_states(), ctx_.num_fv_states(), ctx_.num_bands(),
                              &la::constant<std::complex<double>>::one(), ev1, offs, 0, ev, offs, 0,
                              &la::constant<std::complex<double>>::one(), dm__, 0, 0);
            }
        }
    }
}
mdarray<double, 2> const&
Force::calc_forces_total()
{
    return calc_forces_total(true /*add scf correction*/);
}

mdarray<double, 2> const&
Force::calc_forces_total(bool add_scf_corr)
{
    forces_total_ = mdarray<double, 2>({3, ctx_.unit_cell().num_atoms()});
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
        if (add_scf_corr) {
            calc_forces_scf_corr();
        }
        calc_forces_hubbard();

        forces_total_ = mdarray<double, 2>({3, ctx_.unit_cell().num_atoms()});
        for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
            for (int x : {0, 1, 2}) {
                if (add_scf_corr) {
                    forces_total_(x, ia) = forces_vloc_(x, ia) + forces_us_(x, ia) + forces_nonloc_(x, ia) +
                                           forces_core_(x, ia) + forces_ewald_(x, ia) + forces_scf_corr_(x, ia) +
                                           forces_hubbard_(x, ia);
                } else {
                    forces_total_(x, ia) = forces_vloc_(x, ia) + forces_us_(x, ia) + forces_nonloc_(x, ia) +
                                           forces_core_(x, ia) + forces_ewald_(x, ia) + forces_hubbard_(x, ia);
                }
            }
        }
    }
    return forces_total_;
}

mdarray<double, 2> const&
Force::calc_forces_ibs()
{
    forces_ibs_ = mdarray<double, 2>({3, ctx_.unit_cell().num_atoms()});
    forces_ibs_.zero();

    mdarray<double, 2> ffac({ctx_.unit_cell().num_atom_types(), ctx_.gvec().num_shells()});
    #pragma omp parallel for
    for (int igs = 0; igs < ctx_.gvec().num_shells(); igs++) {
        for (int iat = 0; iat < ctx_.unit_cell().num_atom_types(); iat++) {
            ffac(iat, igs) = unit_step_function_form_factors(ctx_.unit_cell().atom_type(iat).mt_radius(),
                                                             ctx_.gvec().shell_len(igs));
        }
    }

    Hamiltonian0<double> H0(potential_, false);
    for (auto it : kset_.spl_num_kpoints()) {
        auto hk = H0(*kset_.get<double>(it.i));
        add_ibs_force(kset_.get<double>(it.i), hk, ffac, forces_ibs_);
    }
    ctx_.comm().allreduce(&forces_ibs_(0, 0), (int)forces_ibs_.size());
    symmetrize_forces(ctx_.unit_cell(), forces_ibs_);

    return forces_ibs_;
}

mdarray<double, 2> const&
Force::calc_forces_rho()
{

    forces_rho_ = mdarray<double, 2>({3, ctx_.unit_cell().num_atoms()});
    forces_rho_.zero();
    for (auto it : ctx_.unit_cell().spl_num_atoms()) {
        auto g = gradient(density_.density_mt(it.li));
        for (int x = 0; x < 3; x++) {
            forces_rho_(x, it.i) = inner(potential_.effective_potential_mt(it.li), g[x]);
        }
    }
    ctx_.comm().allreduce(&forces_rho_(0, 0), (int)forces_rho_.size());
    symmetrize_forces(ctx_.unit_cell(), forces_rho_);

    return forces_rho_;
}

mdarray<double, 2> const&
Force::calc_forces_hf()
{
    forces_hf_ = mdarray<double, 2>({3, ctx_.unit_cell().num_atoms()});
    forces_hf_.zero();

    for (auto it : ctx_.unit_cell().spl_num_atoms()) {
        auto g = gradient(potential_.hartree_potential_mt(it.li));
        for (int x = 0; x < 3; x++) {
            forces_hf_(x, it.i) = ctx_.unit_cell().atom(it.i).zn() * g[x](0, 0) * y00;
        }
    }
    ctx_.comm().allreduce(&forces_hf_(0, 0), (int)forces_hf_.size());
    symmetrize_forces(ctx_.unit_cell(), forces_hf_);

    return forces_hf_;
}

mdarray<double, 2> const&
Force::calc_forces_hubbard()
{
    PROFILE("sirius::Force::hubbard_force");
    forces_hubbard_ = mdarray<double, 2>({3, ctx_.unit_cell().num_atoms()});
    forces_hubbard_.zero();

    if (ctx_.hubbard_correction()) {
        /* recompute the hubbard potential */
        ::sirius::generate_potential(density_.occupation_matrix(), potential_.hubbard_potential());

        Q_operator<double> q_op(ctx_);

        for (auto it : kset_.spl_num_kpoints()) {

            auto kp  = kset_.get<double>(it.i);
            auto mg1 = kp->spinor_wave_functions().memory_guard(ctx_.processing_unit_memory_t(), wf::copy_to::device);
            auto mg2 =
                    kp->hubbard_wave_functions_S().memory_guard(ctx_.processing_unit_memory_t(), wf::copy_to::device);
            auto mg3 = kp->atomic_wave_functions().memory_guard(ctx_.processing_unit_memory_t(), wf::copy_to::device);
            auto mg4 = kp->atomic_wave_functions_S().memory_guard(ctx_.processing_unit_memory_t(), wf::copy_to::device);

            if (ctx_.num_mag_dims() == 3) {
                RTE_THROW("Hubbard forces are only implemented for the simple hubbard correction.");
            }
            hubbard_force_add_k_contribution_collinear(*kp, q_op, forces_hubbard_);
        }

        /* global reduction */
        kset_.comm().allreduce(forces_hubbard_.at(memory_t::host), 3 * ctx_.unit_cell().num_atoms());
    }

    symmetrize_forces(ctx_.unit_cell(), forces_hubbard_);
    return forces_hubbard_;
}

mdarray<double, 2> const&
Force::calc_forces_ewald()
{
    PROFILE("sirius::Force::calc_forces_ewald");

    forces_ewald_ = mdarray<double, 2>({3, ctx_.unit_cell().num_atoms()});
    forces_ewald_.zero();

    Unit_cell& unit_cell = ctx_.unit_cell();

    double alpha = ctx_.ewald_lambda();

    double prefac = (ctx_.gvec().reduced() ? 4.0 : 2.0) * (twopi / unit_cell.omega());

    mdarray<std::complex<double>, 1> rho_tmp({ctx_.gvec().count()});
    rho_tmp.zero();
    #pragma omp parallel for schedule(static)
    for (auto it : skip_g0(ctx_.gvec())) {

        std::complex<double> rho(0, 0);

        for (int ja = 0; ja < unit_cell.num_atoms(); ja++) {
            rho += ctx_.gvec_phase_factor(it.ig, ja) * static_cast<double>(unit_cell.atom(ja).zn());
        }

        rho_tmp[it.igloc] = std::conj(rho);
    }

    #pragma omp parallel for
    for (int ja = 0; ja < unit_cell.num_atoms(); ja++) {
        for (auto it : skip_g0(ctx_.gvec())) {

            double g2 = std::pow(ctx_.gvec().gvec_len(it.igloc), 2);

            /* cartesian form for getting cartesian force components */
            auto gvec_cart = ctx_.gvec().gvec_cart(it.igloc);

            double scalar_part = prefac * (rho_tmp[it.igloc] * ctx_.gvec_phase_factor(it.ig, ja)).imag() *
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

            auto t = dot(unit_cell.lattice_vectors(), r3::vector<int>(unit_cell.nearest_neighbour(i, ia).translation));

            double scalar_part =
                    static_cast<double>(unit_cell.atom(ia).zn() * unit_cell.atom(ja).zn()) / d2 *
                    (std::erfc(std::sqrt(alpha) * d) / d + 2.0 * std::sqrt(alpha * invpi) * std::exp(-d2 * alpha));

            for (int x : {0, 1, 2}) {
                forces_ewald_(x, ia) += scalar_part * t[x];
            }
        }
    }

    symmetrize_forces(ctx_.unit_cell(), forces_ewald_);

    return forces_ewald_;
}

mdarray<double, 2> const&
Force::calc_forces_us()
{
    PROFILE("sirius::Force::calc_forces_us");

    forces_us_ = mdarray<double, 2>({3, ctx_.unit_cell().num_atoms()});
    forces_us_.zero();

    potential_.fft_transform(-1);

    Unit_cell& unit_cell = ctx_.unit_cell();

    double reduce_g_fact = ctx_.gvec().reduced() ? 2.0 : 1.0;

    la::lib_t la{la::lib_t::none};

    memory_pool* mp{nullptr};
    switch (ctx_.processing_unit()) {
        case device_t::CPU: {
            mp = &get_memory_pool(memory_t::host);
            la = la::lib_t::blas;
            break;
        }
        case device_t::GPU: {
            mp = &get_memory_pool(memory_t::host_pinned);
            la = la::lib_t::spla;
            break;
        }
    }

    /* over atom types */
    for (int iat = 0; iat < unit_cell.num_atom_types(); iat++) {
        auto& atom_type = unit_cell.atom_type(iat);

        if (!ctx_.unit_cell().atom_type(iat).augment()) {
            continue;
        }

        auto& aug_op = ctx_.augmentation_op(iat);

        int nbf = atom_type.mt_basis_size();

        /* get auxiliary density matrix */
        auto dm = density_.density_matrix_aux(atom_type);

        mdarray<double, 2> v_tmp({atom_type.num_atoms(), ctx_.gvec().count() * 2}, *mp);
        mdarray<double, 2> tmp({nbf * (nbf + 1) / 2, atom_type.num_atoms()}, *mp);

        /* over spin components, can be from 1 to 4*/
        for (int ispin = 0; ispin < ctx_.num_mag_dims() + 1; ispin++) {
            /* over 3 components of the force/G - vectors */
            for (int ivec = 0; ivec < 3; ivec++) {
                /* over local rank G vectors */
                #pragma omp parallel for schedule(static)
                for (int igloc = 0; igloc < ctx_.gvec().count(); igloc++) {
                    int ig   = ctx_.gvec().offset() + igloc;
                    auto gvc = ctx_.gvec().gvec_cart(gvec_index_t::local(igloc));
                    for (int ia = 0; ia < atom_type.num_atoms(); ia++) {
                        /* here we write in v_tmp  -i * G * exp[ iGRn] Veff(G)
                         * but in formula we have   i * G * exp[-iGRn] Veff*(G)
                         * the differences because we unfold complex array in the real one
                         * and need negative imagine part due to a multiplication law of complex numbers */
                        auto z = std::complex<double>(0, -gvc[ivec]) *
                                 ctx_.gvec_phase_factor(ig, atom_type.atom_id(ia)) *
                                 potential_.component(ispin).rg().f_pw_local(igloc);
                        v_tmp(ia, 2 * igloc)     = z.real();
                        v_tmp(ia, 2 * igloc + 1) = z.imag();
                    }
                }

                /* multiply tmp matrices, or sum over G */
                la::wrap(la).gemm('N', 'T', nbf * (nbf + 1) / 2, atom_type.num_atoms(), 2 * ctx_.gvec().count(),
                                  &la::constant<double>::one(), aug_op.q_pw().at(memory_t::host), aug_op.q_pw().ld(),
                                  v_tmp.at(memory_t::host), v_tmp.ld(), &la::constant<double>::zero(),
                                  tmp.at(memory_t::host), tmp.ld());

                #pragma omp parallel for
                for (int ia = 0; ia < atom_type.num_atoms(); ia++) {
                    for (int i = 0; i < nbf * (nbf + 1) / 2; i++) {
                        forces_us_(ivec, atom_type.atom_id(ia)) += ctx_.unit_cell().omega() * reduce_g_fact *
                                                                   dm(i, ia, ispin) * aug_op.sym_weight(i) * tmp(i, ia);
                    }
                }
            }
        }
    }

    ctx_.comm().allreduce(&forces_us_(0, 0), 3 * ctx_.unit_cell().num_atoms());

    symmetrize_forces(ctx_.unit_cell(), forces_us_);

    return forces_us_;
}

mdarray<double, 2> const&
Force::calc_forces_scf_corr()
{
    PROFILE("sirius::Force::calc_forces_scf_corr");

    auto q = ctx_.gvec().shells_len();
    /* get form-factors for all G shells */
    auto ff = ctx_.ri().ps_rho_->values(q, ctx_.comm());

    forces_scf_corr_ = mdarray<double, 2>({3, ctx_.unit_cell().num_atoms()});
    forces_scf_corr_.zero();

    /* get main arrays */
    auto& dveff = potential_.dveff();

    Unit_cell& unit_cell = ctx_.unit_cell();

    fft::Gvec const& gvec = ctx_.gvec();

    int gvec_count  = gvec.count();
    int gvec_offset = gvec.offset();

    double fact = gvec.reduced() ? 2.0 : 1.0;

    int ig0 = ctx_.gvec().skip_g0();

    #pragma omp parallel for
    for (int ia = 0; ia < unit_cell.num_atoms(); ia++) {
        Atom& atom = unit_cell.atom(ia);

        int iat = atom.type_id();

        for (int igloc = ig0; igloc < gvec_count; igloc++) {
            int ig   = gvec_offset + igloc;
            int igsh = ctx_.gvec().shell(ig);

            /* cartesian form for getting cartesian force components */
            auto gvec_cart = gvec.gvec_cart(gvec_index_t::local(igloc));

            /* scalar part of a force without multipying by G-vector */
            std::complex<double> z =
                    fact * fourpi * ff(igsh, iat) * std::conj(dveff.f_pw_local(igloc) * ctx_.gvec_phase_factor(ig, ia));

            /* get force components multiplying by cartesian G-vector */
            for (int x : {0, 1, 2}) {
                forces_scf_corr_(x, ia) -= (gvec_cart[x] * z).imag();
            }
        }
    }
    ctx_.comm().allreduce(&forces_scf_corr_(0, 0), 3 * ctx_.unit_cell().num_atoms());

    symmetrize_forces(ctx_.unit_cell(), forces_scf_corr_);

    return forces_scf_corr_;
}

mdarray<double, 2> const&
Force::calc_forces_core()
{
    PROFILE("sirius::Force::calc_forces_core");

    auto q = ctx_.gvec().shells_len();
    /* get form-factors for all G shells */
    auto ff = ctx_.ri().ps_core_->values(q, ctx_.comm());

    forces_core_ = mdarray<double, 2>({3, ctx_.unit_cell().num_atoms()});
    forces_core_.zero();

    /* get main arrays */
    auto& xc_pot = potential_.xc_potential();

    /* transform from real space to reciprocal */
    xc_pot.rg().fft_transform(-1);

    Unit_cell& unit_cell = ctx_.unit_cell();

    fft::Gvec const& gvecs = ctx_.gvec();

    int gvec_count  = gvecs.count();
    int gvec_offset = gvecs.offset();

    double fact = gvecs.reduced() ? 2.0 : 1.0;

    /* here the calculations are in lattice vectors space */
    #pragma omp parallel for
    for (int ia = 0; ia < unit_cell.num_atoms(); ia++) {
        Atom& atom = unit_cell.atom(ia);
        if (atom.type().ps_core_charge_density().empty()) {
            continue;
        }
        int iat = atom.type_id();

        for (int igloc = ctx_.gvec().skip_g0(); igloc < gvec_count; igloc++) {
            int ig    = gvec_offset + igloc;
            auto igsh = ctx_.gvec().shell(ig);

            /* cartesian form for getting cartesian force components */
            auto gvec_cart = gvecs.gvec_cart(gvec_index_t::local(igloc));

            /* scalar part of a force without multipying by G-vector */
            std::complex<double> z = fact * fourpi * ff(igsh, iat) *
                                     std::conj(xc_pot.rg().f_pw_local(igloc) * ctx_.gvec_phase_factor(ig, ia));

            /* get force components multiplying by cartesian G-vector */
            for (int x : {0, 1, 2}) {
                forces_core_(x, ia) -= (gvec_cart[x] * z).imag();
            }
        }
    }
    ctx_.comm().allreduce(&forces_core_(0, 0), 3 * ctx_.unit_cell().num_atoms());

    symmetrize_forces(ctx_.unit_cell(), forces_core_);

    return forces_core_;
}

void
Force::hubbard_force_add_k_contribution_collinear(K_point<double>& kp__, Q_operator<double>& q_op__,
                                                  mdarray<double, 2>& forceh__)
{
    auto r = ctx_.unit_cell().num_hubbard_wf();

    mdarray<std::complex<double>, 5> dn({r.first, r.first, ctx_.num_spins(), 3, ctx_.unit_cell().num_atoms()});

    potential_.U().compute_occupancies_derivatives(kp__, q_op__, dn);

    #pragma omp parallel for
    for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
        /* compute the derivative of the occupancies numbers */
        for (int dir = 0; dir < 3; dir++) {
            std::complex<double> d{0.0};
            for (int at_lvl = 0; at_lvl < static_cast<int>(potential_.hubbard_potential().local().size()); at_lvl++) {
                const int ia1    = potential_.hubbard_potential().atomic_orbitals(at_lvl).first;
                const auto& atom = ctx_.unit_cell().atom(ia1);
                const int lo     = potential_.hubbard_potential().atomic_orbitals(at_lvl).second;
                if (atom.type().lo_descriptor_hub(lo).use_for_calculation()) {
                    int const lmax_at = 2 * atom.type().lo_descriptor_hub(lo).l() + 1;
                    const int offset  = potential_.hubbard_potential().offset(at_lvl);
                    for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                        for (int m2 = 0; m2 < lmax_at; m2++) {
                            for (int m1 = 0; m1 < lmax_at; m1++) {
                                d += potential_.hubbard_potential().local(at_lvl)(m1, m2, ispn) *
                                     dn(offset + m2, offset + m1, ispn, dir, ia);
                            }
                        }
                    }
                }
            }

            for (int i = 0; i < ctx_.cfg().hubbard().nonlocal().size(); i++) {
                auto nl = ctx_.cfg().hubbard().nonlocal(i);
                int ia1 = nl.atom_pair()[0];
                int ja  = nl.atom_pair()[1];
                // consider the links that involve atom i
                int il  = nl.l()[0];
                int jl  = nl.l()[1];
                int in  = nl.n()[0];
                int jn  = nl.n()[1];
                auto Tr = nl.T();

                auto z1           = std::exp(std::complex<double>(0, -twopi * dot(r3::vector<int>(Tr), kp__.vk())));
                const int at_lvl1 = potential_.hubbard_potential().find_orbital_index(ia1, in, il);
                const int at_lvl2 = potential_.hubbard_potential().find_orbital_index(ja, jn, jl);
                const int offset1 = potential_.hubbard_potential().offset(at_lvl1);
                const int offset2 = potential_.hubbard_potential().offset(at_lvl2);
                for (int is = 0; is < ctx_.num_spins(); is++) {
                    for (int m2 = 0; m2 < 2 * jl + 1; m2++) {
                        for (int m1 = 0; m1 < 2 * il + 1; m1++) {
                            auto result1_ = z1 * std::conj(dn(offset2 + m2, offset1 + m1, is, dir, ia)) *
                                            potential_.hubbard_potential().nonlocal(i)(m1, m2, is);
                            d += std::real(result1_);
                        }
                    }
                }
            }
            forceh__(dir, ia) -= std::real(d);
        }
    }
}

mdarray<double, 2> const&
Force::calc_forces_vloc()
{
    PROFILE("sirius::Force::calc_forces_vloc");

    auto q = ctx_.gvec().shells_len();
    /* get form-factors for all G shells */
    auto ff = ctx_.ri().vloc_->values(q, ctx_.comm());

    forces_vloc_ = mdarray<double, 2>({3, ctx_.unit_cell().num_atoms()});
    forces_vloc_.zero();

    auto& valence_rho = density_.rho();

    Unit_cell& unit_cell = ctx_.unit_cell();

    auto const& gvecs = ctx_.gvec();

    double fact = valence_rho.rg().gvec().reduced() ? 2.0 : 1.0;

    /* here the calculations are in lattice vectors space */
    #pragma omp parallel for
    for (int ia = 0; ia < unit_cell.num_atoms(); ia++) {
        Atom& atom = unit_cell.atom(ia);

        int iat = atom.type_id();

        for (auto it : gvecs) {
            int igsh = ctx_.gvec().shell(it.ig);

            /* cartesian form for getting cartesian force components */
            auto gvec_cart = gvecs.gvec_cart(it.igloc);

            /* scalar part of a force without multiplying by G-vector */
            std::complex<double> z = fact * fourpi * ff(igsh, iat) * std::conj(valence_rho.rg().f_pw_local(it.igloc)) *
                                     std::conj(ctx_.gvec_phase_factor(it.ig, ia));

            /* get force components multiplying by cartesian G-vector  */
            for (int x : {0, 1, 2}) {
                forces_vloc_(x, ia) -= (gvec_cart[x] * z).imag();
            }
        }
    }

    ctx_.comm().allreduce(&forces_vloc_(0, 0), 3 * ctx_.unit_cell().num_atoms());

    symmetrize_forces(ctx_.unit_cell(), forces_vloc_);

    return forces_vloc_;
}

mdarray<double, 2> const&
Force::calc_forces_usnl()
{
    calc_forces_us();
    calc_forces_nonloc();

    forces_usnl_ = mdarray<double, 2>({3, ctx_.unit_cell().num_atoms()});
    for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
        for (int x : {0, 1, 2}) {
            forces_usnl_(x, ia) = forces_us_(x, ia) + forces_nonloc_(x, ia);
        }
    }

    return forces_usnl_;
}

void
Force::add_ibs_force(K_point<double>* kp__, Hamiltonian_k<double>& Hk__, mdarray<double, 2>& ffac__,
                     mdarray<double, 2>& forcek__) const
{
    PROFILE("sirius::Force::ibs_force");

    auto& uc = ctx_.unit_cell();

    auto& bg = ctx_.blacs_grid();

    auto bs = ctx_.cyclic_block_size();

    auto nfv = ctx_.num_fv_states();

    auto ngklo = kp__->gklo_basis_size();

    /* compute density matrix for a k-point */
    la::dmatrix<std::complex<double>> dm(nfv, nfv, bg, bs, bs);
    compute_dmat(kp__, dm);

    /* first-variational eigen-vectors in scalapack distribution */
    auto& fv_evec = kp__->fv_eigen_vectors();

    la::dmatrix<std::complex<double>> h(ngklo, ngklo, bg, bs, bs);
    la::dmatrix<std::complex<double>> o(ngklo, ngklo, bg, bs, bs);

    la::dmatrix<std::complex<double>> h1(ngklo, ngklo, bg, bs, bs);
    la::dmatrix<std::complex<double>> o1(ngklo, ngklo, bg, bs, bs);

    la::dmatrix<std::complex<double>> zm1(ngklo, nfv, bg, bs, bs);
    la::dmatrix<std::complex<double>> zf(nfv, nfv, bg, bs, bs);

    mdarray<std::complex<double>, 2> alm_row({kp__->num_gkvec_row(), uc.max_mt_aw_basis_size()});
    mdarray<std::complex<double>, 2> alm_col({kp__->num_gkvec_col(), uc.max_mt_aw_basis_size()});
    mdarray<std::complex<double>, 2> halm_col({kp__->num_gkvec_col(), uc.max_mt_aw_basis_size()});

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
        Hk__.H0().apply_hmt_to_apw(atom, spin_block_t::nm, kp__->num_gkvec_col(), alm_col, halm_col);

        /* apw-apw block of the overlap matrix */
        la::wrap(la::lib_t::blas)
                .gemm('N', 'T', kp__->num_gkvec_row(), kp__->num_gkvec_col(), type.mt_aw_basis_size(),
                      &la::constant<std::complex<double>>::one(), alm_row.at(memory_t::host), alm_row.ld(),
                      alm_col.at(memory_t::host), alm_col.ld(), &la::constant<std::complex<double>>::zero(),
                      o.at(memory_t::host), o.ld());

        /* apw-apw block of the Hamiltonian matrix */
        la::wrap(la::lib_t::blas)
                .gemm('N', 'T', kp__->num_gkvec_row(), kp__->num_gkvec_col(), type.mt_aw_basis_size(),
                      &la::constant<std::complex<double>>::one(), alm_row.at(memory_t::host), alm_row.ld(),
                      halm_col.at(memory_t::host), halm_col.ld(), &la::constant<std::complex<double>>::zero(),
                      h.at(memory_t::host), h.ld());

        int iat = type.id();

        for (int igk_col = 0; igk_col < kp__->num_gkvec_col(); igk_col++) { // loop over columns
            auto gvec_col       = kp__->gkvec_col().gvec(gvec_index_t::local(igk_col));
            auto gkvec_col_cart = kp__->gkvec_col().gkvec_cart(gvec_index_t::local(igk_col));
            for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++) { // for each column loop over rows
                auto gvec_row       = kp__->gkvec_row().gvec(gvec_index_t::local(igk_row));
                auto gkvec_row_cart = kp__->gkvec_row().gkvec_cart(gvec_index_t::local(igk_row));

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
                auto gvec_col = kp__->gkvec_col().gvec(gvec_index_t::local(igk_col));
                for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++) { // loop over rows
                    auto gvec_row = kp__->gkvec_row().gvec(gvec_index_t::local(igk_row));
                    /* compute index of G-G' */
                    int ig12 = ctx_.gvec().index_g12(gvec_row, gvec_col);
                    /* get G-G' */
                    auto vg = ctx_.gvec().gvec_cart(gvec_index_t::global(ig12));
                    /* multiply by i(G-G') */
                    h1(igk_row, igk_col) = std::complex<double>(0.0, vg[x]) * h(igk_row, igk_col);
                    /* multiply by i(G-G') */
                    o1(igk_row, igk_col) = std::complex<double>(0.0, vg[x]) * o(igk_row, igk_col);
                }
            }

            for (int icol = 0; icol < kp__->num_lo_col(); icol++) {
                for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++) {
                    auto gkvec_row_cart = kp__->gkvec_row().gkvec_cart(gvec_index_t::local(igk_row));
                    /* multiply by i(G+k) */
                    h1(igk_row, icol + kp__->num_gkvec_col()) =
                            std::complex<double>(0.0, gkvec_row_cart[x]) * h(igk_row, icol + kp__->num_gkvec_col());
                    /* multiply by i(G+k) */
                    o1(igk_row, icol + kp__->num_gkvec_col()) =
                            std::complex<double>(0.0, gkvec_row_cart[x]) * o(igk_row, icol + kp__->num_gkvec_col());
                }
            }

            for (int irow = 0; irow < kp__->num_lo_row(); irow++) {
                for (int igk_col = 0; igk_col < kp__->num_gkvec_col(); igk_col++) {
                    auto gkvec_col_cart = kp__->gkvec_col().gkvec_cart(gvec_index_t::local(igk_col));
                    /* multiply by i(G+k) */
                    h1(irow + kp__->num_gkvec_row(), igk_col) =
                            std::complex<double>(0.0, -gkvec_col_cart[x]) * h(irow + kp__->num_gkvec_row(), igk_col);
                    /* multiply by i(G+k) */
                    o1(irow + kp__->num_gkvec_row(), igk_col) =
                            std::complex<double>(0.0, -gkvec_col_cart[x]) * o(irow + kp__->num_gkvec_row(), igk_col);
                }
            }

            /* zm1 = dO * V */
            la::wrap(la::lib_t::scalapack)
                    .gemm('N', 'N', ngklo, nfv, ngklo, &la::constant<std::complex<double>>::one(), o1, 0, 0, fv_evec, 0,
                          0, &la::constant<std::complex<double>>::zero(), zm1, 0, 0);
            /* multiply by energy: zm1 = E * (dO * V)  */
            for (int i = 0; i < zm1.num_cols_local(); i++) {
                int ist = zm1.icol(i);
                for (int j = 0; j < kp__->gklo_basis_size_row(); j++) {
                    zm1(j, i) *= kp__->fv_eigen_value(ist);
                }
            }
            /* compute zm1 = dH * V - E * (dO * V) */
            la::wrap(la::lib_t::scalapack)
                    .gemm('N', 'N', ngklo, nfv, ngklo, &la::constant<std::complex<double>>::one(), h1, 0, 0, fv_evec, 0,
                          0, &la::constant<std::complex<double>>::m_one(), zm1, 0, 0);

            /* compute zf = V^{+} * zm1 = V^{+} * (dH * V - E * (dO * V)) */
            la::wrap(la::lib_t::scalapack)
                    .gemm('C', 'N', nfv, nfv, ngklo, &la::constant<std::complex<double>>::one(), fv_evec, 0, 0, zm1, 0,
                          0, &la::constant<std::complex<double>>::zero(), zf, 0, 0);

            for (int i = 0; i < dm.num_cols_local(); i++) {
                for (int j = 0; j < dm.num_rows_local(); j++) {
                    forcek__(x, ia) += kp__->weight() * std::real(dm(j, i) * zf(j, i));
                }
            }
        }
    } // ia
}

void
Force::print_info(std::ostream& out__, int verbosity__)
{
    auto print_forces = [&](std::string label__, mdarray<double, 2> const& forces) {
        out__ << "==== " << label__ << " =====" << std::endl;
        for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
            out__ << "atom: " << std::setw(4) << ia << ", force: " << ffmt(15, 7) << forces(0, ia) << ffmt(15, 7)
                  << forces(1, ia) << ffmt(15, 7) << forces(2, ia) << std::endl;
        }
    };

    if (verbosity__ >= 1) {
        out__ << std::endl;
        print_forces("total Forces in Ha/bohr", forces_total());
    }

    if (!ctx_.full_potential() && verbosity__ >= 2) {
        print_forces("ultrasoft contribution from Qij", forces_us());

        print_forces("non-local contribution from Beta-projector", forces_nonloc());

        print_forces("contribution from local potential", forces_vloc());

        print_forces("contribution from core density", forces_core());

        print_forces("Ewald forces from ions", forces_ewald());

        if (ctx_.hubbard_correction()) {
            print_forces("contribution from Hubbard correction", forces_hubbard());
        }
    }
}

template void
Force::calc_forces_nonloc_aux<double>();
#if defined(SIRIUS_USE_FP32)
template void
Force::calc_forces_nonloc_aux<float>();
#endif

} // namespace sirius
