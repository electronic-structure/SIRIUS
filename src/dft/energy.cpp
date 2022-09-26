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
ewald_energy(const Simulation_context& ctx, const sddk::Gvec& gvec, const Unit_cell& unit_cell)
{
    double alpha{ctx.ewald_lambda()};
    double ewald_g{0};

    #pragma omp parallel for reduction(+ : ewald_g)
    for (int igloc = gvec.skip_g0(); igloc < gvec.count(); igloc++) {
        double g2 = std::pow(gvec.gvec_len<sddk::index_domain_t::local>(igloc), 2);

        double_complex rho(0, 0);

        for (int ia = 0; ia < unit_cell.num_atoms(); ia++) {
            rho += ctx.gvec_phase_factor(gvec.gvec<sddk::index_domain_t::local>(igloc), ia) *
                static_cast<double>(unit_cell.atom(ia).zn());
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
ks_energy(Simulation_context const& ctx, const std::map<std::string, double>& energies)
{
    double tot_en{0};

    switch (ctx.electronic_structure_method()) {
        case electronic_structure_method_t::full_potential_lapwlo: {
            tot_en = energies.at("ekin") + energies.at("exc") + 0.5 * energies.at("vha") + energies.at("enuc");
            break;
        }

        case electronic_structure_method_t::pseudopotential: {
            tot_en = energies.at("valence_eval_sum") - energies.at("vxc") - energies.at("bxc") -
                     energies.at("PAW_one_elec") - 0.5 * energies.at("vha") + energies.at("exc") +
                     energies.at("PAW_total_energy") + energies.at("ewald");
            break;
        }
    }

    if (ctx.hubbard_correction()) {
        tot_en += energies.at("hubbard_energy") - energies.at("hubbard_one_el_contribution");
    }

    return tot_en;
}

double
ks_energy(Simulation_context const& ctx, K_point_set const& kset, Density const& density, Potential const& potential,
          double ewald_energy)
{
    return ks_energy(ctx, total_energy_components(ctx, kset, density, potential, ewald_energy));
}

double
total_energy(Simulation_context const& ctx, K_point_set const& kset, Density const& density, Potential const& potential,
             double ewald_energy)
{

    double eks = ks_energy(ctx, kset, density, potential, ewald_energy);
    double tot_en{0};

    switch (ctx.electronic_structure_method()) {
        case electronic_structure_method_t::full_potential_lapwlo: {
            tot_en = eks;
            break;
        }

        case electronic_structure_method_t::pseudopotential: {
            tot_en = eks + kset.entropy_sum();
            break;
        }
        default: {
            RTE_THROW("invalid electronic_structure_method");
        }
    }

    return tot_en;
}

std::map<std::string, double>
total_energy_components(Simulation_context const& ctx, const K_point_set & kset, Density const& density,
                        Potential const& potential, double ewald_energy)
{
    std::map<std::string, double> table;
    switch (ctx.electronic_structure_method()) {
        case electronic_structure_method_t::full_potential_lapwlo: {
            table["ekin"] = energy_kin(ctx, kset, density, potential);
            table["exc"]  = energy_exc(density, potential);
            table["vha"]  = energy_vha(potential);
            table["enuc"] = energy_enuc(ctx, potential);
            break;
            }

        case electronic_structure_method_t::pseudopotential: {
            table["valence_eval_sum"] = kset.valence_eval_sum();
            table["vxc"]              = energy_vxc(density, potential);
            table["bxc"]              = energy_bxc(density, potential);
            table["PAW_one_elec"]     = potential.PAW_one_elec_energy(density);
            table["vha"]              = energy_vha(potential);
            table["exc"]              = energy_exc(density, potential);
            table["PAW_total_energy"] = potential.PAW_total_energy();
            table["ewald"]            = ewald_energy;
            table["PAW_xc_total_energy"] = potential.PAW_xc_total_energy();
            table["PAW_hartree_total_energy"] = potential.PAW_hartree_total_energy();
            break;
        }
    }

    if (ctx.hubbard_correction()) {
        table["hubbard_energy"] = energy(density.occupation_matrix());
        table["hubbard_one_el_contribution"] = one_electron_energy_hubbard(density, potential);
    }

    table["entropy"] = kset.entropy_sum();

    return table;
}

double
hubbard_energy(Density const& density)
{
    if (density.ctx().hubbard_correction()) {
        return energy(density.occupation_matrix());
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
        return one_electron_energy_hubbard(density.occupation_matrix(), potential.hubbard_potential());
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
