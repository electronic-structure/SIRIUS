#ifndef PYTHON_MODULE_ENERGIES_H
#define PYTHON_MODULE_ENERGIES_H

namespace sirius {

double ewald_energy(const Simulation_context& ctx, const Gvec& gvec, const Unit_cell& unit_cell)
{
    double alpha = 1.5;

    double ewald_g = 0;

    #pragma omp parallel
    {
        double ewald_g_pt = 0;

        #pragma omp for
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

            ewald_g_pt += std::pow(std::abs(rho), 2) * std::exp(-g2 / 4 / alpha) / g2;
        }

        #pragma omp critical
        ewald_g += ewald_g_pt;
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

    double ewald_r = 0;
    #pragma omp parallel
    {
        double ewald_r_pt = 0;

        #pragma omp for
        for (int ia = 0; ia < unit_cell.num_atoms(); ia++) {
            for (int i = 1; i < unit_cell.num_nearest_neighbours(ia); i++) {
                int ja = unit_cell.nearest_neighbour(i, ia).atom_id;
                double d = unit_cell.nearest_neighbour(i, ia).distance;
                ewald_r_pt += 0.5 * unit_cell.atom(ia).zn() * unit_cell.atom(ja).zn() *
                              std::erfc(std::sqrt(alpha) * d) / d;
            }
        }

        #pragma omp critical
        ewald_r += ewald_r_pt;
    }

    return (ewald_g + ewald_r);

}

double energy_bxc(const Density& density, const Potential& potential, int num_mag_dims=0)
{
    double ebxc{0};
    for (int j = 0; j < num_mag_dims; j++) {
        ebxc += density.magnetization(j).inner(potential.effective_magnetic_field(j));
    }
    return ebxc;
}

}  // sirius

#endif /* PYTHON_MODULE_ENERGIES_H */
