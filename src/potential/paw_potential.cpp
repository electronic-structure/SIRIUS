// Copyright (c) 2013-2018 Anton Kozhevnikov, Ilia Sivkov, Thomas Schulthess
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

/** \file paw_potential.hpp
 *
 *  \brief Generate PAW potential.
 */

#include "potential.hpp"

namespace sirius {

void Potential::init_PAW()
{
    paw_potential_data_.clear();
    if (!unit_cell_.num_paw_atoms()) {
        return;
    }

    for (int i = 0; i < unit_cell_.spl_num_paw_atoms().local_size(); i++) {
        int ia_paw = unit_cell_.spl_num_paw_atoms(i);
        int ia     = unit_cell_.paw_atom_index(ia_paw);

        auto& atom = unit_cell_.atom(ia);

        auto& atom_type = atom.type();

        int l_max      = 2 * atom_type.indexr().lmax_lo();
        int lm_max_rho = utils::lmmax(l_max);

        paw_potential_data_t ppd;

        ppd.atom_ = &atom;

        ppd.ia = ia;

        ppd.ia_paw = ia_paw;

        /* allocate potential arrays */
        for (int i = 0; i < ctx_.num_mag_dims() + 1; i++) {
            ppd.ae_potential_.push_back(sf(lm_max_rho, ppd.atom_->radial_grid()));
            ppd.ps_potential_.push_back(sf(lm_max_rho, ppd.atom_->radial_grid()));
        }

        ppd.core_energy_ = atom_type.paw_core_energy();

        paw_potential_data_.push_back(std::move(ppd));
    }

    for (int i = 0; i < unit_cell_.num_paw_atoms(); i++) {
        int ia              = unit_cell_.paw_atom_index(i);
        int bs              = unit_cell_.atom(ia).mt_basis_size();
        max_paw_basis_size_ = std::max(max_paw_basis_size_, bs);
    }

    /* initialize dij matrix */
    paw_dij_ = sddk::mdarray<double, 4>(max_paw_basis_size_, max_paw_basis_size_, ctx_.num_mag_dims() + 1,
                                  unit_cell_.num_paw_atoms(), sddk::memory_t::host, "paw_dij_");

    /* allocate PAW energy array */
    paw_hartree_energies_.resize(unit_cell_.num_paw_atoms());
    paw_xc_energies_.resize(unit_cell_.num_paw_atoms());
    paw_core_energies_.resize(unit_cell_.num_paw_atoms());
    paw_one_elec_energies_.resize(unit_cell_.num_paw_atoms());
}

void Potential::generate_PAW_effective_potential(Density const& density)
{
    PROFILE("sirius::Potential::generate_PAW_effective_potential");

    if (!unit_cell_.num_paw_atoms()) {
        return;
    }

    /* zero PAW arrays */
    std::fill(paw_one_elec_energies_.begin(), paw_one_elec_energies_.end(), 0.0);
    std::fill(paw_hartree_energies_.begin(), paw_hartree_energies_.end(), 0.0);
    std::fill(paw_xc_energies_.begin(), paw_xc_energies_.end(), 0.0);

    /* zero Dij */
    paw_dij_.zero();

    /* calculate xc and hartree for atoms */
    for (int i = 0; i < unit_cell_.spl_num_paw_atoms().local_size(); i++) {
        calc_PAW_local_potential(paw_potential_data_[i], density.paw_ae_density(i), density.paw_ps_density(i));
    }

    /* calculate PAW Dij matrix */
    #pragma omp parallel for
    for (int i = 0; i < unit_cell_.spl_num_paw_atoms().local_size(); i++) {
        calc_PAW_local_Dij(paw_potential_data_[i], paw_dij_);

        //calc_PAW_one_elec_energy(paw_potential_data_[i], density.density_matrix(), paw_dij_);
    }

    // collect Dij and add to atom d_mtrx
    comm_.allreduce(&paw_dij_(0, 0, 0, 0), static_cast<int>(paw_dij_.size()));

    if (ctx_.cfg().control().print_checksum() && comm_.rank() == 0) {
        auto cs = paw_dij_.checksum();
        utils::print_checksum("paw_dij", cs);
    }

    // add paw Dij to uspp Dij
    add_paw_Dij_to_atom_Dmtrx();

    // calc total energy
    double energies[] = {0.0, 0.0, 0.0, 0.0};

    for (int ia = 0; ia < unit_cell_.spl_num_paw_atoms().local_size(); ia++) {
        energies[0] += paw_potential_data_[ia].hartree_energy_;
        energies[1] += paw_potential_data_[ia].xc_energy_;
        //energies[2] += paw_potential_data_[ia].one_elec_energy_;
        energies[3] += paw_potential_data_[ia].core_energy_; // it is light operation
    }

    comm_.allreduce(&energies[0], 4);

    paw_hartree_total_energy_ = energies[0];
    paw_xc_total_energy_      = energies[1];
    //paw_one_elec_energy_      = energies[2];
    paw_total_core_energy_    = energies[3];
}

double xc_mt_paw(std::vector<XC_functional> const& xc_func__, int lmax__, int num_mag_dims__, SHT const& sht__,
    Radial_grid<double> const& rgrid__, std::vector<Flm const*> rho__, std::vector<double> const& rho_core__,
                 std::vector<Flm>& vxc__, const std::string label)
{
    int lmmax = utils::lmmax(lmax__);

    /* new array to store core and valence densities */
    Flm rho0(lmmax, rgrid__);

    assert(rho0.size(0) == rho__[0]->size(0));

    rho0.zero();
    rho0 += (*rho__[0]);

    double invY00 = 1.0 / y00;

    /* add core density */
    for (int ir = 0; ir < rgrid__.num_points(); ir++) {
        rho0(0, ir) += invY00 * rho_core__[ir];
    }

    std::vector<Flm const*> rho;
    rho.push_back(&rho0);
    for (int j = 0; j < num_mag_dims__; j++) {
        rho.push_back(rho__[j + 1]);
    }

    std::vector<Flm*> vxc;
    for (int j = 0; j < num_mag_dims__ + 1; j++) {
        vxc.push_back(&vxc__[j]);
    }

    Flm exclm(lmmax, rgrid__);

    sirius::xc_mt(rgrid__, sht__, xc_func__, num_mag_dims__, rho, vxc, &exclm, label);
    return inner(exclm, rho0);
}

double Potential::calc_PAW_hartree_potential(Atom& atom, sf const& full_density, sf& full_potential)
{
    int lmsize_rho = static_cast<int>(full_density.size(0));

    auto& grid = full_density.radial_grid();

    /* array passed to poisson solver */
    Spheric_function<function_domain_t::spectral, double> atom_pot_sf(lmsize_rho, grid);
    atom_pot_sf.zero();

    poisson_vmt<true>(atom, full_density, atom_pot_sf);

    /* add hartree contribution */
    full_potential += atom_pot_sf;

    /* calculate energy */

    auto l_by_lm = utils::l_by_lm(utils::lmax(lmsize_rho));

    /* create array for integration */
    std::vector<double> intdata(grid.num_points(), 0);

    double hartree_energy{0};

    for (int lm = 0; lm < lmsize_rho; lm++) {
        /* fill data to integrate */
        for (int irad = 0; irad < grid.num_points(); irad++) {
            intdata[irad] = full_density(lm, irad) * full_potential(lm, irad) * grid[irad] * grid[irad];
        }

        /* create spline from the data */
        Spline<double> h_spl(grid, intdata);
        hartree_energy += 0.5 * h_spl.integrate(0);
    }

    return hartree_energy;
}

void Potential::calc_PAW_local_potential(paw_potential_data_t& ppd,
    std::vector<Spheric_function<function_domain_t::spectral, double> const*> ae_density,
    std::vector<Spheric_function<function_domain_t::spectral, double> const*> ps_density)
{
    /* calculation of Hartree potential */
    for (int i = 0; i < ctx_.num_mag_dims() + 1; i++) {
        ppd.ae_potential_[i].zero();
        ppd.ps_potential_[i].zero();
    }

    double ae_hartree_energy = calc_PAW_hartree_potential(*ppd.atom_, *ae_density[0], ppd.ae_potential_[0]);

    double ps_hartree_energy = calc_PAW_hartree_potential(*ppd.atom_, *ps_density[0], ppd.ps_potential_[0]);

    ppd.hartree_energy_ = ae_hartree_energy - ps_hartree_energy;

    /* calculation of XC potential */
    auto& ps_core = ppd.atom_->type().ps_core_charge_density();
    auto& ae_core = ppd.atom_->type().paw_ae_core_charge_density();

    auto& rgrid = ppd.atom_->type().radial_grid();
    int l_max = 2 * ppd.atom_->type().indexr().lmax_lo();

    std::vector<Flm> vxc;
    for (int j = 0; j < ctx_.num_mag_dims() + 1; j++) {
        vxc.emplace_back(utils::lmmax(l_max), rgrid);
    }

    auto ae_xc_energy = sirius::xc_mt_paw(xc_func_, l_max, ctx_.num_mag_dims(), *sht_, rgrid, ae_density,
                                          ae_core, vxc, "ae contrib");
    for (int i = 0; i < ctx_.num_mag_dims() + 1; i++) {
        ppd.ae_potential_[i] += vxc[i];
    }

    auto ps_xc_energy = sirius::xc_mt_paw(xc_func_, l_max, ctx_.num_mag_dims(), *sht_, rgrid, ps_density,
                                          ps_core, vxc, "ps contrib");

    for (int i = 0; i < ctx_.num_mag_dims() + 1; i++) {
        ppd.ps_potential_[i] += vxc[i];
    }

    /* save xc energy in pdd structure */
    ppd.xc_energy_ = ae_xc_energy - ps_xc_energy;
}

void Potential::calc_PAW_local_Dij(const paw_potential_data_t& pdd, sddk::mdarray<double, 4>& paw_dij)
{
    int paw_ind = pdd.ia_paw;

    auto& atom_type = pdd.atom_->type();

    auto& paw_ae_wfs = atom_type.ae_paw_wfs_array();
    auto& paw_ps_wfs = atom_type.ps_paw_wfs_array();

    /* get lm size for density */
    int lmax       = atom_type.indexr().lmax_lo();
    int lmsize_rho = utils::lmmax(2 * lmax);

    auto l_by_lm = utils::l_by_lm(2 * lmax);

    Gaunt_coefficients<double> GC(lmax, 2 * lmax, lmax, SHT::gaunt_rrr);

    /* store integrals here */
    sddk::mdarray<double, 3> integrals(
        lmsize_rho, atom_type.num_beta_radial_functions() * (atom_type.num_beta_radial_functions() + 1) / 2,
        ctx_.num_mag_dims() + 1);

    auto& rgrid = atom_type.radial_grid();

    for (int imagn = 0; imagn < ctx_.num_mag_dims() + 1; imagn++) {
        auto& ae_atom_pot = pdd.ae_potential_[imagn];
        auto& ps_atom_pot = pdd.ps_potential_[imagn];

        for (int irb2 = 0; irb2 < atom_type.num_beta_radial_functions(); irb2++) {
            for (int irb1 = 0; irb1 <= irb2; irb1++) {
                int iqij = (irb2 * (irb2 + 1)) / 2 + irb1;

                /* create array for integration */
                std::vector<double> intdata(rgrid.num_points());

                for (int lm3 = 0; lm3 < lmsize_rho; lm3++) {
                    /* fill array */
                    for (int irad = 0; irad < rgrid.num_points(); irad++) {
                        double ae_part = paw_ae_wfs(irad, irb1) * paw_ae_wfs(irad, irb2);
                        double ps_part = paw_ps_wfs(irad, irb1) * paw_ps_wfs(irad, irb2) +
                                         atom_type.q_radial_function(irb1, irb2, l_by_lm[lm3])(irad);

                        intdata[irad] = ae_atom_pot(lm3, irad) * ae_part - ps_atom_pot(lm3, irad) * ps_part;
                    }

                    /* create spline from data arrays */
                    Spline<double> dij_spl(rgrid, intdata);

                    /* integrate */
                    integrals(lm3, iqij, imagn) = dij_spl.integrate(0);
                }
            }
        }
    }

    /* calculate Dij */
    for (int ib2 = 0; ib2 < atom_type.mt_lo_basis_size(); ib2++) {
        for (int ib1 = 0; ib1 <= ib2; ib1++) {

            /* get lm quantum numbers (lm index) of the basis functions */
            int lm1 = atom_type.indexb(ib1).lm;
            int lm2 = atom_type.indexb(ib2).lm;

            /* get radial basis functions indices */
            int irb1 = atom_type.indexb(ib1).idxrf;
            int irb2 = atom_type.indexb(ib2).idxrf;

            /* common index */
            int iqij = (irb2 * (irb2 + 1)) / 2 + irb1;

            /* get num of non-zero GC */
            int num_non_zero_gk = GC.num_gaunt(lm1, lm2);

            for (int imagn = 0; imagn < ctx_.num_mag_dims() + 1; imagn++) {
                /* add nonzero coefficients */
                for (int inz = 0; inz < num_non_zero_gk; inz++) {
                    auto& lm3coef = GC.gaunt(lm1, lm2, inz);

                    /* add to atom Dij an integral of dij array */
                    paw_dij(ib1, ib2, imagn, paw_ind) += lm3coef.coef * integrals(lm3coef.lm3, iqij, imagn);
                }

                if (ib1 != ib2) {
                    paw_dij(ib2, ib1, imagn, paw_ind) = paw_dij(ib1, ib2, imagn, paw_ind);
                }
            }
        }
    }
}

double Potential::calc_PAW_one_elec_energy(paw_potential_data_t const& pdd,
                                           sddk::mdarray<double_complex, 4> const& density_matrix,
                                           sddk::mdarray<double, 4> const& paw_dij) const
{
    int ia      = pdd.ia;
    int paw_ind = pdd.ia_paw;

    double_complex energy = 0.0;

    for (int ib2 = 0; ib2 < pdd.atom_->mt_lo_basis_size(); ib2++) {
        for (int ib1 = 0; ib1 < pdd.atom_->mt_lo_basis_size(); ib1++) {
            double dm[4] = {0, 0, 0, 0};
            switch (ctx_.num_mag_dims()) {
                case 3: {
                    dm[2] = 2 * std::real(density_matrix(ib1, ib2, 2, ia));
                    dm[3] = -2 * std::imag(density_matrix(ib1, ib2, 2, ia));
                }
                case 1: {
                    dm[0] = std::real(density_matrix(ib1, ib2, 0, ia) + density_matrix(ib1, ib2, 1, ia));
                    dm[1] = std::real(density_matrix(ib1, ib2, 0, ia) - density_matrix(ib1, ib2, 1, ia));
                    break;
                }
                case 0: {
                    dm[0] = density_matrix(ib1, ib2, 0, ia).real();
                    break;
                }
                default: {
                    TERMINATE("calc_PAW_one_elec_energy FATAL ERROR!");
                    break;
                }
            }
            for (int imagn = 0; imagn < ctx_.num_mag_dims() + 1; imagn++) {
                energy += dm[imagn] * paw_dij(ib1, ib2, imagn, paw_ind);
            }
        }
    }

    if (std::abs(energy.imag()) > 1e-10) {
        std::stringstream s;
        s << "PAW energy is not real: " << energy;
        TERMINATE(s.str());
    }

    return energy.real();
}

void Potential::add_paw_Dij_to_atom_Dmtrx()
{
    #pragma omp parallel for
    for (int i = 0; i < unit_cell_.num_paw_atoms(); i++) {
        int ia     = unit_cell_.paw_atom_index(i);
        auto& atom = unit_cell_.atom(ia);

        for (int imagn = 0; imagn < ctx_.num_mag_dims() + 1; imagn++) {
            for (int ib2 = 0; ib2 < atom.mt_lo_basis_size(); ib2++) {
                for (int ib1 = 0; ib1 < atom.mt_lo_basis_size(); ib1++) {
                    atom.d_mtrx(ib1, ib2, imagn) += paw_dij_(ib1, ib2, imagn, i);
                }
            }
        }
    }
}

} // namespace sirius
