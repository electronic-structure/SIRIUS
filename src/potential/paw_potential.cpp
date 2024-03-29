/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file paw_potential.hpp
 *
 *  \brief Generate PAW potential.
 */

#include "potential.hpp"
#include "symmetry/symmetrize_mt_function.hpp"

namespace sirius {

void
Potential::init_PAW()
{
    if (!unit_cell_.num_paw_atoms()) {
        return;
    }

    bool const is_global{true};
    paw_potential_ = std::make_unique<PAW_field4D<double>>("PAW potential", unit_cell_, is_global);

    paw_ae_exc_ = std::make_unique<Spheric_function_set<double, paw_atom_index_t>>(
            "paw_ae_exc_", unit_cell_, unit_cell_.paw_atoms(),
            [this](int ia) { return lmax_t(2 * this->unit_cell_.atom(ia).type().indexr().lmax()); });

    paw_ps_exc_ = std::make_unique<Spheric_function_set<double, paw_atom_index_t>>(
            "paw_ps_exc_", unit_cell_, unit_cell_.paw_atoms(),
            [this](int ia) { return lmax_t(2 * this->unit_cell_.atom(ia).type().indexr().lmax()); });

    /* initialize dij matrix */
    paw_dij_.resize(unit_cell_.num_paw_atoms());
    for (int i = 0; i < unit_cell_.num_paw_atoms(); i++) {
        int ia      = unit_cell_.paw_atom_index(paw_atom_index_t::global(i));
        paw_dij_[i] = mdarray<double, 3>(
                {unit_cell_.atom(ia).mt_basis_size(), unit_cell_.atom(ia).mt_basis_size(), ctx_.num_mag_dims() + 1});
    }
}

void
Potential::generate_PAW_effective_potential(Density const& density)
{
    PROFILE("sirius::Potential::generate_PAW_effective_potential");

    if (!unit_cell_.num_paw_atoms()) {
        return;
    }

    paw_potential_->zero();

    paw_hartree_total_energy_ = 0.0;

    /* calculate xc and hartree for atoms */
    for (auto it : unit_cell_.spl_num_paw_atoms()) {
        auto ia = unit_cell_.paw_atom_index(it.i);
        paw_hartree_total_energy_ +=
                calc_PAW_local_potential(ia, density.paw_ae_density(ia), density.paw_ps_density(ia));
    }
    comm_.allreduce(&paw_hartree_total_energy_, 1);

    paw_potential_->sync();
    std::vector<Spheric_function_set<double, paw_atom_index_t>*> ae_comp;
    std::vector<Spheric_function_set<double, paw_atom_index_t>*> ps_comp;
    for (int j = 0; j < ctx_.num_mag_dims() + 1; j++) {
        ae_comp.push_back(&paw_potential_->ae_component(j));
        ps_comp.push_back(&paw_potential_->ps_component(j));
    }
    sirius::symmetrize_mt_function(unit_cell_.symmetry(), unit_cell_.comm(), ctx_.num_mag_dims(), ae_comp);
    sirius::symmetrize_mt_function(unit_cell_.symmetry(), unit_cell_.comm(), ctx_.num_mag_dims(), ps_comp);

    /* symmetrize ae- component of Exc */
    paw_ae_exc_->sync(unit_cell_.spl_num_paw_atoms());
    ae_comp.clear();
    ae_comp.push_back(paw_ae_exc_.get());
    sirius::symmetrize_mt_function(unit_cell_.symmetry(), unit_cell_.comm(), 0, ae_comp);

    /* symmetrize ps- component of Exc */
    paw_ps_exc_->sync(unit_cell_.spl_num_paw_atoms());
    ps_comp.clear();
    ps_comp.push_back(paw_ps_exc_.get());
    sirius::symmetrize_mt_function(unit_cell_.symmetry(), unit_cell_.comm(), 0, ps_comp);

    /* calculate PAW Dij matrix */
    #pragma omp parallel for
    for (auto it : unit_cell_.spl_num_paw_atoms()) {
        auto ia = unit_cell_.paw_atom_index(it.i);
        calc_PAW_local_Dij(ia, paw_dij_[it.i]);
    }
    for (int i = 0; i < unit_cell_.num_paw_atoms(); i++) {
        auto location = unit_cell_.spl_num_paw_atoms().location(typename paw_atom_index_t::global(i));
        comm_.bcast(paw_dij_[i].at(memory_t::host), paw_dij_[i].size(), location.ib);
    }

    /* add paw Dij to uspp Dij */
    #pragma omp parallel for
    for (int i = 0; i < unit_cell_.num_paw_atoms(); i++) {
        auto ia    = unit_cell_.paw_atom_index(typename paw_atom_index_t::global(i));
        auto& atom = unit_cell_.atom(ia);

        for (int imagn = 0; imagn < ctx_.num_mag_dims() + 1; imagn++) {
            for (int ib2 = 0; ib2 < atom.mt_basis_size(); ib2++) {
                for (int ib1 = 0; ib1 < atom.mt_basis_size(); ib1++) {
                    atom.d_mtrx(ib1, ib2, imagn) += paw_dij_[i](ib1, ib2, imagn);
                }
            }
        }
    }
}

double
xc_mt_paw(std::vector<XC_functional> const& xc_func__, int lmax__, int num_mag_dims__, SHT const& sht__,
          Radial_grid<double> const& rgrid__, std::vector<Flm const*> rho__, std::vector<double> const& rho_core__,
          std::vector<Flm>& vxc__, Flm& exclm__, bool use_lapl__)
{
    int lmmax = sf::lmmax(lmax__);

    /* new array to store core and valence densities */
    Flm rho0(lmmax, rgrid__);

    if (rho0.size(0) != rho__[0]->size(0)) {
        std::stringstream s;
        s << "Sizes of rho0 and rho do not match" << std::endl
          << "  rho0.size(0) : " << rho0.size(0) << std::endl
          << "  rho__[0]->size(0) : " << rho__[0]->size(0) << std::endl
          << "  lmax : " << lmax__;
        RTE_THROW(s);
    }

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

    sirius::xc_mt(rgrid__, sht__, xc_func__, num_mag_dims__, rho, vxc, &exclm__, use_lapl__);
    return inner(exclm__, rho0);
}

double
Potential::calc_PAW_hartree_potential(Atom& atom__, Flm const& rho__, Flm& v_tot__)
{
    auto lmmax = rho__.angular_domain_size();

    auto& grid = rho__.radial_grid();

    /* array passed to poisson solver */
    Flm v_ha(lmmax, grid);
    v_ha.zero();

    poisson_vmt<true>(atom__, rho__, v_ha);

    /* add hartree contribution */
    v_tot__ += v_ha;

    return 0.5 * inner(rho__, v_ha);
}

double
Potential::calc_PAW_local_potential(typename atom_index_t::global ia__, std::vector<Flm const*> ae_density__,
                                    std::vector<Flm const*> ps_density__)
{
    auto& atom = unit_cell_.atom(ia__);

    /* calculation of XC potential */
    auto& ps_core = atom.type().ps_core_charge_density();
    auto& ae_core = atom.type().paw_ae_core_charge_density();

    auto& rgrid = atom.type().radial_grid();
    int l_max   = 2 * atom.type().indexr().lmax();

    std::vector<Flm> vxc;
    for (int j = 0; j < ctx_.num_mag_dims() + 1; j++) {
        vxc.emplace_back(sf::lmmax(l_max), rgrid);
    }

    sirius::xc_mt_paw(xc_func_, l_max, ctx_.num_mag_dims(), *sht_, rgrid, ae_density__, ae_core, vxc,
                      (*paw_ae_exc_)[ia__], ctx_.cfg().settings().xc_use_lapl());
    for (int i = 0; i < ctx_.num_mag_dims() + 1; i++) {
        paw_potential_->ae_component(i)[ia__] += vxc[i];
    }

    sirius::xc_mt_paw(xc_func_, l_max, ctx_.num_mag_dims(), *sht_, rgrid, ps_density__, ps_core, vxc,
                      (*paw_ps_exc_)[ia__], ctx_.cfg().settings().xc_use_lapl());
    for (int i = 0; i < ctx_.num_mag_dims() + 1; i++) {
        paw_potential_->ps_component(i)[ia__] += vxc[i];
    }

    auto eha = calc_PAW_hartree_potential(atom, *ae_density__[0], paw_potential_->ae_component(0)[ia__]) -
               calc_PAW_hartree_potential(atom, *ps_density__[0], paw_potential_->ps_component(0)[ia__]);

    return eha;
}

void
Potential::calc_PAW_local_Dij(typename atom_index_t::global ia__, mdarray<double, 3>& paw_dij__)
{
    paw_dij__.zero();

    auto& atom = unit_cell_.atom(ia__);

    auto& atom_type = atom.type();

    auto& paw_ae_wfs = atom_type.ae_paw_wfs_array();
    auto& paw_ps_wfs = atom_type.ps_paw_wfs_array();

    /* get lm size for density */
    int lmax  = atom_type.indexr().lmax();
    int lmmax = sf::lmmax(2 * lmax);

    auto l_by_lm = sf::l_by_lm(2 * lmax);

    Gaunt_coefficients<double> GC(lmax, 2 * lmax, lmax, SHT::gaunt_rrr);

    auto nbrf = atom_type.num_beta_radial_functions();

    /* store integrals here */
    mdarray<double, 3> integrals({lmmax, nbrf * (nbrf + 1) / 2, ctx_.num_mag_dims() + 1});

    auto& rgrid = atom_type.radial_grid();

    for (int imagn = 0; imagn < ctx_.num_mag_dims() + 1; imagn++) {
        auto& ae_atom_pot = paw_potential_->ae_component(imagn)[ia__];
        auto& ps_atom_pot = paw_potential_->ps_component(imagn)[ia__];

        for (int irb2 = 0; irb2 < nbrf; irb2++) {
            for (int irb1 = 0; irb1 <= irb2; irb1++) {

                /* create array for integration */
                std::vector<double> intdata(rgrid.num_points());

                // TODO: precompute radial integrals of paw_ae_wfs and paw_ps_wfs pair products
                for (int lm3 = 0; lm3 < lmmax; lm3++) {
                    /* fill array */
                    for (int irad = 0; irad < rgrid.num_points(); irad++) {
                        double ae_part = paw_ae_wfs(irad, irb1) * paw_ae_wfs(irad, irb2);
                        double ps_part = paw_ps_wfs(irad, irb1) * paw_ps_wfs(irad, irb2) +
                                         atom_type.q_radial_function(irb1, irb2, l_by_lm[lm3])(irad);

                        intdata[irad] = ae_atom_pot(lm3, irad) * ae_part - ps_atom_pot(lm3, irad) * ps_part;
                    }

                    /* integrate */
                    integrals(lm3, packed_index(irb1, irb2), imagn) = Spline<double>(rgrid, intdata).integrate(0);
                }
            }
        }
    }

    /* calculate Dij */
    for (int ib2 = 0; ib2 < atom_type.mt_basis_size(); ib2++) {
        for (int ib1 = 0; ib1 <= ib2; ib1++) {

            /* get lm quantum numbers (lm index) of the basis functions */
            int lm1 = atom_type.indexb(ib1).lm;
            int lm2 = atom_type.indexb(ib2).lm;

            /* get radial basis functions indices */
            int irb1 = atom_type.indexb(ib1).idxrf;
            int irb2 = atom_type.indexb(ib2).idxrf;

            /* common index */
            int iqij = packed_index(irb1, irb2);

            /* get num of non-zero GC */
            int num_non_zero_gk = GC.num_gaunt(lm1, lm2);

            for (int imagn = 0; imagn < ctx_.num_mag_dims() + 1; imagn++) {
                /* add nonzero coefficients */
                for (int inz = 0; inz < num_non_zero_gk; inz++) {
                    auto& lm3coef = GC.gaunt(lm1, lm2, inz);

                    /* add to atom Dij an integral of dij array */
                    paw_dij__(ib1, ib2, imagn) += lm3coef.coef * integrals(lm3coef.lm3, iqij, imagn);
                }

                if (ib1 != ib2) {
                    paw_dij__(ib2, ib1, imagn) = paw_dij__(ib1, ib2, imagn);
                }
            }
        }
    }
}

double
Potential::calc_PAW_one_elec_energy(Atom const& atom__, mdarray<double, 2> const& density_matrix__,
                                    mdarray<double, 3> const& paw_dij__) const
{
    double energy{0.0};

    for (int ib2 = 0; ib2 < atom__.mt_basis_size(); ib2++) {
        for (int ib1 = 0; ib1 < atom__.mt_basis_size(); ib1++) {
            for (int imagn = 0; imagn < ctx_.num_mag_dims() + 1; imagn++) {
                energy += density_matrix__(packed_index(ib1, ib2), imagn) * paw_dij__(ib1, ib2, imagn);
            }
        }
    }
    return energy;
}

} // namespace sirius
