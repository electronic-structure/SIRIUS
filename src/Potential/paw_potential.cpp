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
    paw_dij_ = mdarray<double, 4>(max_paw_basis_size_, max_paw_basis_size_, ctx_.num_mag_dims() + 1,
                                  unit_cell_.num_paw_atoms(), memory_t::host, "paw_dij_");

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

        calc_PAW_one_elec_energy(paw_potential_data_[i], density.density_matrix(), paw_dij_);
    }

    // collect Dij and add to atom d_mtrx
    comm_.allreduce(&paw_dij_(0, 0, 0, 0), static_cast<int>(paw_dij_.size()));

    if (ctx_.control().print_checksum_ && comm_.rank() == 0) {
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
        energies[2] += paw_potential_data_[ia].one_elec_energy_;
        energies[3] += paw_potential_data_[ia].core_energy_; // it is light operation
    }

    comm_.allreduce(&energies[0], 4);

    paw_hartree_total_energy_ = energies[0];
    paw_xc_total_energy_      = energies[1];
    paw_one_elec_energy_      = energies[2];
    paw_total_core_energy_    = energies[3];
}

double Potential::xc_mt_PAW_nonmagnetic(sf& full_potential, sf const& full_density, std::vector<double> const& rho_core)
{
    int lmmax = static_cast<int>(full_density.size(0));

    Radial_grid<double> const& rgrid = full_density.radial_grid();

    /* new array to store core and valence densities */
    sf full_rho_lm_sf_new(lmmax, rgrid);

    full_rho_lm_sf_new.zero();
    full_rho_lm_sf_new += full_density;

    double invY00 = 1.0 / y00;

    /* adding core part */
    for (int ir = 0; ir < rgrid.num_points(); ir++) {
        full_rho_lm_sf_new(0, ir) += invY00 * rho_core[ir];
    }

    auto full_rho_tp_sf = transform(*sht_, full_rho_lm_sf_new);

    /* create potential in theta phi */
    Spheric_function<function_domain_t::spatial, double> vxc_tp_sf(sht_->num_points(), rgrid);

    /* create energy in theta phi */
    Spheric_function<function_domain_t::spatial, double> exc_tp_sf(sht_->num_points(), rgrid);

    xc_mt_nonmagnetic(rgrid, xc_func_, full_rho_lm_sf_new, full_rho_tp_sf, vxc_tp_sf, exc_tp_sf);

    full_potential += transform(*sht_, vxc_tp_sf);

    /* calculate energy */
    auto exc_lm_sf = transform(*sht_, exc_tp_sf);

    return inner(exc_lm_sf, full_rho_lm_sf_new);
}

double Potential::xc_mt_PAW_collinear(std::vector<sf>& potential, std::vector<sf const*> density,
                                      std::vector<double> const& rho_core)
{
    int lmsize_rho = static_cast<int>(density[0]->size(0));

    Radial_grid<double> const& rgrid = density[0]->radial_grid();

    /* new array to store core and valence densities */
    sf full_rho_lm_sf_new(lmsize_rho, rgrid);

    full_rho_lm_sf_new.zero();
    full_rho_lm_sf_new += (*density[0]);

    double invY00 = 1 / y00;

    /* adding core part */
    for (int ir = 0; ir < rgrid.num_points(); ir++) {
        full_rho_lm_sf_new(0, ir) += invY00 * rho_core[ir];
    }

    /* calculate spin up spin down density components in lm components */
    /* up = 1/2 ( rho + magn );  down = 1/2 ( rho - magn ) */
    auto rho_u_lm_sf = 0.5 * (full_rho_lm_sf_new + (*density[1]));
    auto rho_d_lm_sf = 0.5 * (full_rho_lm_sf_new - (*density[1]));

    // transform density to theta phi components
    auto rho_u_tp_sf = transform(*sht_, rho_u_lm_sf);
    auto rho_d_tp_sf = transform(*sht_, rho_d_lm_sf);

    // create potential in theta phi
    Spheric_function<function_domain_t::spatial, double> vxc_u_tp_sf(sht_->num_points(), rgrid);
    Spheric_function<function_domain_t::spatial, double> vxc_d_tp_sf(sht_->num_points(), rgrid);

    // create energy in theta phi
    Spheric_function<function_domain_t::spatial, double> exc_tp_sf(sht_->num_points(), rgrid);

    // calculate XC
    xc_mt_magnetic(rgrid, xc_func_, rho_u_lm_sf, rho_u_tp_sf, rho_d_lm_sf, rho_d_tp_sf, vxc_u_tp_sf, vxc_d_tp_sf,
                   exc_tp_sf);

    // transform back in lm
    potential[0] += transform(*sht_, 0.5 * (vxc_u_tp_sf + vxc_d_tp_sf));
    potential[1] += transform(*sht_, 0.5 * (vxc_u_tp_sf - vxc_d_tp_sf));

    //------------------------
    //--- calculate energy ---
    //------------------------
    Spheric_function<function_domain_t::spectral, double> exc_lm_sf = transform(*sht_, exc_tp_sf);

    return inner(exc_lm_sf, full_rho_lm_sf_new);
}

double Potential::xc_mt_PAW_noncollinear(std::vector<sf>& potential, std::vector<sf const*> density,
                                         std::vector<double> const& rho_core)
{
    if (density.size() != 4 || potential.size() != 4) {
        TERMINATE("xc_mt_PAW_noncollinear FATAL ERROR!")
    }

    Radial_grid<double> const& rgrid = density[0]->radial_grid();

    /* transform density to theta phi components */
    std::vector<Spheric_function<function_domain_t::spatial, double>> rho_tp(density.size());

    for (size_t i = 0; i < density.size(); i++) {
        rho_tp[i] = transform(*sht_, *density[i]);
    }

    /* transform 4D magnetization to spin-up, spin-down form (correct for LSDA)  rho ± |magn| */
    Spheric_function<function_domain_t::spatial, double> rho_u_tp(sht_->num_points(), rgrid);
    Spheric_function<function_domain_t::spatial, double> rho_d_tp(sht_->num_points(), rgrid);

    for (int ir = 0; ir < rgrid.num_points(); ir++) {
        for (int itp = 0; itp < sht_->num_points(); itp++) {
            vector3d<double> magn({rho_tp[2](itp, ir), rho_tp[3](itp, ir), rho_tp[1](itp, ir)});
            double norm = magn.length();

            rho_u_tp(itp, ir) = 0.5 * (rho_tp[0](itp, ir) + rho_core[ir] + norm);
            rho_d_tp(itp, ir) = 0.5 * (rho_tp[0](itp, ir) + rho_core[ir] - norm);
        }
    }

    /* in lm representation */
    auto rho_u_lm = transform(*sht_, rho_u_tp);
    auto rho_d_lm = transform(*sht_, rho_d_tp);

    /* allocate potential in theta phi */
    Spheric_function<function_domain_t::spatial, double> vxc_u_tp(sht_->num_points(), rgrid);
    Spheric_function<function_domain_t::spatial, double> vxc_d_tp(sht_->num_points(), rgrid);

    // allocate energy in theta phi
    Spheric_function<function_domain_t::spatial, double> exc_tp(sht_->num_points(), rgrid);

    /* calculate XC */
    xc_mt_magnetic(rgrid, xc_func_, rho_u_lm, rho_u_tp, rho_d_lm, rho_d_tp, vxc_u_tp, vxc_d_tp, exc_tp);

    /* allocate 4D potential in theta phi components */
    std::vector<Spheric_function<function_domain_t::spatial, double>> vxc_tp;

    /* allocate the rest */
    for (size_t i = 0; i < potential.size(); i++) {
        vxc_tp.push_back(Spheric_function<function_domain_t::spatial, double>(sht_->num_points(), rgrid));
    }

    /* transform back potential from up/down to 4D form*/
    for (int ir = 0; ir < rgrid.num_points(); ir++) {
        for (int itp = 0; itp < sht_->num_points(); itp++) {
            /* get total potential and field abs value*/
            double pot   = 0.5 * (vxc_u_tp(itp, ir) + vxc_d_tp(itp, ir));
            double field = 0.5 * (vxc_u_tp(itp, ir) - vxc_d_tp(itp, ir));

            /* get unit magnetization vector*/
            vector3d<double> magn({rho_tp[2](itp, ir), rho_tp[3](itp, ir), rho_tp[1](itp, ir)});
            double norm = magn.length();
            magn        = magn * (norm > 0.0 ? field / norm : 0.0);

            /* add total potential and effective field values at current point */
            vxc_tp[0](itp, ir) = pot;
            for (int i : {0, 1, 2}) {
                vxc_tp[i + 1](itp, ir) = magn[i];
            }
        }
    }

    /* transform back to lm- domain */
    for (size_t i = 0; i < density.size(); i++) {
        potential[i] += transform(*sht_, vxc_tp[i]);
    }

    /* transform to lm- domain */
    auto exc_lm = transform(*sht_, exc_tp);

    return inner(exc_lm, rho_u_lm + rho_d_lm);
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

void Potential::calc_PAW_local_potential(
    paw_potential_data_t& ppd, std::vector<Spheric_function<function_domain_t::spectral, double> const*> ae_density,
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

    double ae_xc_energy = 0.0;
    double ps_xc_energy = 0.0;

    switch (ctx_.num_mag_dims()) {
        case 0: {
            ae_xc_energy = xc_mt_PAW_nonmagnetic(ppd.ae_potential_[0], *ae_density[0], ae_core);
            ps_xc_energy = xc_mt_PAW_nonmagnetic(ppd.ps_potential_[0], *ps_density[0], ps_core);
            break;
        }

        case 1: {
            ae_xc_energy = xc_mt_PAW_collinear(ppd.ae_potential_, ae_density, ae_core);
            ps_xc_energy = xc_mt_PAW_collinear(ppd.ps_potential_, ps_density, ps_core);
            break;
        }

        case 3: {
            ae_xc_energy = xc_mt_PAW_noncollinear(ppd.ae_potential_, ae_density, ae_core);
            ps_xc_energy = xc_mt_PAW_noncollinear(ppd.ps_potential_, ps_density, ps_core);
            break;
        }

        default: {
            TERMINATE("PAW local potential error! Wrong number of spins!")
        }
    }

    /* save xc energy in pdd structure */
    ppd.xc_energy_ = ae_xc_energy - ps_xc_energy;
}

void Potential::calc_PAW_local_Dij(paw_potential_data_t& pdd, mdarray<double, 4>& paw_dij)
{
    int paw_ind = pdd.ia_paw;

    auto& atom_type = pdd.atom_->type();

    auto& paw_ae_wfs = atom_type.ae_paw_wfs_array();
    auto& paw_ps_wfs = atom_type.ps_paw_wfs_array();

    /* get lm size for density */
    int lmax       = atom_type.indexr().lmax_lo();
    int lmsize_rho = utils::lmmax(2 * lmax);

    auto l_by_lm = utils::l_by_lm(2 * lmax);

    Gaunt_coefficients<double> GC(lmax, 2 * lmax, lmax, SHT::gaunt_rlm);

    /* store integrals here */
    mdarray<double, 3> integrals(
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

double Potential::calc_PAW_one_elec_energy(paw_potential_data_t& pdd, const mdarray<double_complex, 4>& density_matrix,
                                           const mdarray<double, 4>& paw_dij)
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

    pdd.one_elec_energy_ = energy.real();

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
