#include "energy.hpp"
#include "dft_ground_state.hpp"

namespace sirius {
namespace energy {

double ewald(const Simulation_context& ctx)
{
    double alpha{ctx.ewald_lambda()};
    double ewald_g{0};

    auto const& gvec = ctx.gvec();
    auto const& unit_cell = ctx.unit_cell();

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

double vxc(Density const& density, Potential const& potential)
{
    return potential.energy_vxc(density);
}

double vha(Potential const& potential)
{
    return potential.energy_vha();
}

double exc(Density const& density, Potential const& potential)
{
    return potential.energy_exc(density);
}

double bxc(const Density& density, const Potential& potential)
{
    double ebxc{0};
    for (int j = 0; j < density.ctx().num_mag_dims(); j++) {
        ebxc += sirius::inner(density.magnetization(j), potential.effective_magnetic_field(j));
    }
    return ebxc;
}

double nuc(Simulation_context const& ctx, Potential const& potential)
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

double vloc(Density const& density, Potential const& potential)
{
    return sirius::inner(potential.local_potential(), density.rho());
}

double ecore_sum(Unit_cell const& unit_cell)
{
    double sum{0};
    for (int ic = 0; ic < unit_cell.num_atom_symmetry_classes(); ic++) {
        sum += unit_cell.atom_symmetry_class(ic).core_eval_sum() * unit_cell.atom_symmetry_class(ic).num_atoms();
    }
    return sum;
}

double veff(Density const& density, Potential const& potential)
{
    return sirius::inner(density.rho(), potential.effective_potential());
}

double kin(DFT_ground_state const& dft)
{
    return ecore_sum(dft.ctx().unit_cell()) + dft.k_point_set().valence_eval_sum() -
        veff(dft.density(), dft.potential()) - bxc(dft.density(), dft.potential());
}

double one_electron(Density const& density, Potential const& potential)
{
    return vha(potential) + vxc(density, potential) + bxc(density, potential) +
        potential.PAW_one_elec_energy();
}

double paw(Potential const& potential__)
{
    return potential__.PAW_total_energy();
}

double total(DFT_ground_state const &dft)
{
    double tot_en{0};

    switch (dft.ctx().electronic_structure_method()) {
        case electronic_structure_method_t::full_potential_lapwlo: {
            tot_en = (kin(dft) + exc(dft.density(), dft.potential()) +
                      0.5 * vha(dft.potential()) + nuc(dft.ctx(), dft.potential()));
            break;
        }

        case electronic_structure_method_t::pseudopotential: {
            tot_en = (dft.k_point_set().valence_eval_sum() - vxc(dft.density(), dft.potential()) -
                      bxc(dft.density(), dft.potential()) - dft.potential().PAW_one_elec_energy()) -
                      0.5 * vha(dft.potential()) + exc(dft.density(), dft.potential()) +
                      dft.potential().PAW_total_energy() + dft.ewald_energy();
            break;
        }
    }

    if (dft.ctx().hubbard_correction()) {
        tot_en += dft.potential().U().hubbard_energy();
    }

    return tot_en;
}

} // namespace energy
} // namespace sirius
