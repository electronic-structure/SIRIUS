// Copyright (c) 2013-2020 Anton Kozhevnikov, Thomas Schulthess
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

/** \file energy.cpp
 *
 *  \brief Total energy terms.
 */

#include "energy.hpp"

namespace sirius {

double
ewald_energy(const Simulation_context& ctx, const Gvec& gvec, const Unit_cell& unit_cell)
{
    double alpha{ctx.ewald_lambda()};
    double ewald_g{0};

    #pragma omp parallel for reduction(+ : ewald_g)
    for (int igloc = 0; igloc < gvec.count(); igloc++) {
        int ig = gvec.offset() + igloc;
        if (!ig) {
            continue;
        }

        double g2 = std::pow(gvec.gvec_len(ig), 2);

        double_complex rho(0, 0);

        for (int ia = 0; ia < unit_cell.num_atoms(); ia++) {
            rho += ctx.gvec_phase_factor(gvec.gvec(ig), ia) * static_cast<double>(unit_cell.atom(ia).zn());
        }

        ewald_g += std::pow(std::abs(rho), 2) * std::exp(-g2 / 4 / alpha) / g2;
    }

    ctx.comm().allreduce(&ewald_g, 1);
    if (gvec.reduced()) {
        ewald_g *= 2;
    }
    /* remaining G=0 contribution */
    ewald_g -= std::pow(unit_cell.num_electrons(), 2) / alpha / 4;
    ewald_g *= (twopi / unit_cell.omega());

    /* remove self-interaction */
    for (int ia = 0; ia < unit_cell.num_atoms(); ia++) {
        ewald_g -= std::sqrt(alpha / pi) * std::pow(unit_cell.atom(ia).zn(), 2);
    }

    double ewald_r{0};
    #pragma omp parallel for reduction(+ : ewald_r)
    for (int ia = 0; ia < unit_cell.num_atoms(); ia++) {
        for (int i = 1; i < unit_cell.num_nearest_neighbours(ia); i++) {
            int ja   = unit_cell.nearest_neighbour(i, ia).atom_id;
            double d = unit_cell.nearest_neighbour(i, ia).distance;
            ewald_r += 0.5 * unit_cell.atom(ia).zn() * unit_cell.atom(ja).zn() * std::erfc(std::sqrt(alpha) * d) / d;
        }
    }

    return (ewald_g + ewald_r);
}

double
energy_vxc(Density const& density, Potential const& potential)
{
    return potential.energy_vxc(density);
}

double
energy_exc(Density const& density, Potential const& potential)
{
    return potential.energy_exc(density);
}

double
energy_vha(Potential const& potential)
{
    return potential.energy_vha();
}

double
energy_bxc(const Density& density, const Potential& potential)
{
    double ebxc{0};
    for (int j = 0; j < density.ctx().num_mag_dims(); j++) {
        ebxc += sirius::inner(density.magnetization(j), potential.effective_magnetic_field(j));
    }
    return ebxc;
}

double
energy_enuc(Simulation_context const& ctx, Potential const& potential)
{
    auto& unit_cell = ctx.unit_cell();
    double enuc{0};
    if (ctx.full_potential()) {
        for (int ialoc = 0; ialoc < unit_cell.spl_num_atoms().local_size(); ialoc++) {
            int ia = unit_cell.spl_num_atoms(ialoc);
            int zn = unit_cell.atom(ia).zn();
            enuc -= 0.5 * zn * potential.vh_el(ia);
        }
        ctx.comm().allreduce(&enuc, 1);
    }
    return enuc;
}

double
energy_vloc(Density const& density, Potential const& potential)
{
    return sirius::inner(potential.local_potential(), density.rho());
}

double
core_eval_sum(Unit_cell const& unit_cell)
{
    double sum{0};
    for (int ic = 0; ic < unit_cell.num_atom_symmetry_classes(); ic++) {
        sum += unit_cell.atom_symmetry_class(ic).core_eval_sum() * unit_cell.atom_symmetry_class(ic).num_atoms();
    }
    return sum;
}

double
eval_sum(Unit_cell const& unit_cell, K_point_set const& kset)
{
    return core_eval_sum(unit_cell) + kset.valence_eval_sum();
}

double
energy_veff(Density const& density, Potential const& potential)
{
    return sirius::inner(density.rho(), potential.effective_potential());
}

double
energy_kin(Simulation_context const& ctx, K_point_set const& kset, Density const& density, Potential const& potential)
{
    return eval_sum(ctx.unit_cell(), kset) - energy_veff(density, potential) - energy_bxc(density, potential);
}

double
total_energy(Simulation_context const& ctx, K_point_set const& kset, Density const& density, Potential const& potential,
             double ewald_energy)
{
    double tot_en{0};

    switch (ctx.electronic_structure_method()) {
        case electronic_structure_method_t::full_potential_lapwlo: {
            tot_en = (energy_kin(ctx, kset, density, potential) + energy_exc(density, potential) +
                      0.5 * energy_vha(potential) + energy_enuc(ctx, potential));
            break;
        }

        case electronic_structure_method_t::pseudopotential: {
            tot_en = (kset.valence_eval_sum() - energy_vxc(density, potential) - energy_bxc(density, potential) -
                      potential.PAW_one_elec_energy(density)) -
                     0.5 * energy_vha(potential) + energy_exc(density, potential) + potential.PAW_total_energy() +
                     ewald_energy + kset.entropy_sum();
            break;
        }
    }

    if (ctx.hubbard_correction()) {
        tot_en += ::sirius::energy(density.occupation_matrix());
        tot_en -= ::sirius::one_electron_energy_hubbard(density, potential);
    }

    return tot_en;
}

double
hubbard_energy(Density const& density)
{
    if (density.ctx().hubbard_correction()) {
        return ::sirius::energy(density.occupation_matrix());
    } else {
        return 0.0;
    }
}

double
one_electron_energy(Density const& density, Potential const& potential)
{
    return energy_vha(potential) + energy_vxc(density, potential) + energy_bxc(density, potential) +
           potential.PAW_one_elec_energy(density) + one_electron_energy_hubbard(density, potential);
}

double
one_electron_energy_hubbard(Density const& density, Potential const& potential)
{
    auto& ctx = density.ctx();
    if (ctx.hubbard_correction()) {
        return ::sirius::one_electron_energy_hubbard(density.occupation_matrix(), potential.hubbard_potential());
    }
    return 0.0;
}

double
energy_potential(Density const& density, Potential const& potential)
{
    double e =
        energy_veff(density, potential) + energy_bxc(density, potential) + potential.PAW_one_elec_energy(density);
    if (potential.ctx().hubbard_correction()) {
        e += ::sirius::energy(density.occupation_matrix());
    }
    return e;
}

} // namespace sirius
