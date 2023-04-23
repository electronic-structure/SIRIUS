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
#include "symmetry/symmetrize.hpp"

namespace sirius {

void Potential::init_PAW()
{
    if (!unit_cell_.num_paw_atoms()) {
        return;
    }

    bool const is_global{true};
    paw_potential_ = std::make_unique<PAW_field4D<double>>(unit_cell_, is_global);

    paw_ae_exc_ = std::make_unique<Spheric_function_set<double>>(unit_cell_, unit_cell_.paw_atoms(),
                    [this](int ia){return lmax_t(2 * this->unit_cell_.atom(ia).type().indexr().lmax());});

    paw_ps_exc_ = std::make_unique<Spheric_function_set<double>>(unit_cell_, unit_cell_.paw_atoms(),
                    [this](int ia){return lmax_t(2 * this->unit_cell_.atom(ia).type().indexr().lmax());});

    /* initialize dij matrix */
    paw_dij_.resize(unit_cell_.num_paw_atoms());
    for (int i = 0; i < unit_cell_.num_paw_atoms(); i++) {
        int ia = unit_cell_.paw_atom_index(i);
        paw_dij_[i] = sddk::mdarray<double, 3>(unit_cell_.atom(ia).mt_basis_size(), unit_cell_.atom(ia).mt_basis_size(),
                ctx_.num_mag_dims() + 1);
    }
}

void Potential::generate_PAW_effective_potential(Density const& density)
{
    PROFILE("sirius::Potential::generate_PAW_effective_potential");

    if (!unit_cell_.num_paw_atoms()) {
        return;
    }

    paw_potential_->zero();

    paw_hartree_total_energy_ = 0.0;

    /* calculate xc and hartree for atoms */
    for (int i = 0; i < unit_cell_.spl_num_paw_atoms().local_size(); i++) {
        int ia = unit_cell_.paw_atom_index(unit_cell_.spl_num_paw_atoms(i));
        paw_hartree_total_energy_ += calc_PAW_local_potential(ia, density.paw_ae_density(ia),
                                                              density.paw_ps_density(ia));
    }
    comm_.allreduce(&paw_hartree_total_energy_, 1);

    paw_potential_->sync();
    std::vector<Spheric_function_set<double>*> ae_comp;
    std::vector<Spheric_function_set<double>*> ps_comp;
    for (int j = 0; j < ctx_.num_mag_dims() + 1; j++) {
        ae_comp.push_back(&paw_potential_->ae_component(j));
        ps_comp.push_back(&paw_potential_->ps_component(j));
    }
    sirius::symmetrize(unit_cell_.symmetry(), unit_cell_.comm(), ctx_.num_mag_dims(), ae_comp);
    sirius::symmetrize(unit_cell_.symmetry(), unit_cell_.comm(), ctx_.num_mag_dims(), ps_comp);

    /* symmetrize ae- component of Exc */
    paw_ae_exc_->sync(unit_cell_.spl_num_paw_atoms());
    ae_comp.clear();
    ae_comp.push_back(paw_ae_exc_.get());
    sirius::symmetrize(unit_cell_.symmetry(), unit_cell_.comm(), 0, ae_comp);

    /* symmetrize ps- component of Exc */
    paw_ps_exc_->sync(unit_cell_.spl_num_paw_atoms());
    ps_comp.clear();
    ps_comp.push_back(paw_ps_exc_.get());
    sirius::symmetrize(unit_cell_.symmetry(), unit_cell_.comm(), 0, ps_comp);

    /* calculate PAW Dij matrix */
    #pragma omp parallel for
    for (int i = 0; i < unit_cell_.spl_num_paw_atoms().local_size(); i++) {
        int ia_paw = unit_cell_.spl_num_paw_atoms(i);
        int ia = unit_cell_.paw_atom_index(ia_paw);
        calc_PAW_local_Dij(ia, paw_dij_[ia_paw]);
    }
    for (int i = 0; i < unit_cell_.num_paw_atoms(); i++) {
        auto location = unit_cell_.spl_num_paw_atoms().location(i);
        comm_.bcast(paw_dij_[i].at(sddk::memory_t::host), paw_dij_[i].size(), location.rank);
    }

    /* add paw Dij to uspp Dij */
    #pragma omp parallel for
    for (int i = 0; i < unit_cell_.num_paw_atoms(); i++) {
        int ia     = unit_cell_.paw_atom_index(i);
        auto& atom = unit_cell_.atom(ia);

        for (int imagn = 0; imagn < ctx_.num_mag_dims() + 1; imagn++) {
            for (int ib2 = 0; ib2 < atom.mt_lo_basis_size(); ib2++) {
                for (int ib1 = 0; ib1 < atom.mt_lo_basis_size(); ib1++) {
                    atom.d_mtrx(ib1, ib2, imagn) += paw_dij_[i](ib1, ib2, imagn);
                }
            }
        }
    }
}

double xc_mt_paw(std::vector<XC_functional> const& xc_func__, int lmax__, int num_mag_dims__, SHT const& sht__,
    Radial_grid<double> const& rgrid__, std::vector<Flm const*> rho__, std::vector<double> const& rho_core__,
    std::vector<Flm>& vxc__, Flm& exclm__)
{
    int lmmax = utils::lmmax(lmax__);

    /* new array to store core and valence densities */
    Flm rho0(lmmax, rgrid__);

    RTE_ASSERT(rho0.size(0) == rho__[0]->size(0));

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

    sirius::xc_mt(rgrid__, sht__, xc_func__, num_mag_dims__, rho, vxc, &exclm__);
    return inner(exclm__, rho0);
}

double Potential::calc_PAW_hartree_potential(Atom& atom__, Flm const& rho__, Flm& v_tot__)
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
Potential::calc_PAW_local_potential(int ia, std::vector<Flm const*> ae_density, std::vector<Flm const*> ps_density)
{
    auto& atom = unit_cell_.atom(ia);

    /* calculation of XC potential */
    auto& ps_core = atom.type().ps_core_charge_density();
    auto& ae_core = atom.type().paw_ae_core_charge_density();

    auto& rgrid = atom.type().radial_grid();
    int l_max = 2 * atom.type().indexr().lmax_lo();

    std::vector<Flm> vxc;
    for (int j = 0; j < ctx_.num_mag_dims() + 1; j++) {
        vxc.emplace_back(utils::lmmax(l_max), rgrid);
    }

    sirius::xc_mt_paw(xc_func_, l_max, ctx_.num_mag_dims(), *sht_, rgrid, ae_density, ae_core, vxc, (*paw_ae_exc_)[ia]);
    for (int i = 0; i < ctx_.num_mag_dims() + 1; i++) {
        paw_potential_->ae_component(i)[ia] += vxc[i];
    }

    sirius::xc_mt_paw(xc_func_, l_max, ctx_.num_mag_dims(), *sht_, rgrid, ps_density, ps_core, vxc, (*paw_ps_exc_)[ia]);
    for (int i = 0; i < ctx_.num_mag_dims() + 1; i++) {
        paw_potential_->ps_component(i)[ia] += vxc[i];
    }

    auto eha = calc_PAW_hartree_potential(atom, *ae_density[0], paw_potential_->ae_component(0)[ia]) -
               calc_PAW_hartree_potential(atom, *ps_density[0], paw_potential_->ps_component(0)[ia]);

    return eha;
}

void Potential::calc_PAW_local_Dij(int ia__, sddk::mdarray<double, 3>& paw_dij__)
{
    paw_dij__.zero();

    auto& atom = unit_cell_.atom(ia__);

    auto& atom_type = atom.type();

    auto& paw_ae_wfs = atom_type.ae_paw_wfs_array();
    auto& paw_ps_wfs = atom_type.ps_paw_wfs_array();

    /* get lm size for density */
    int lmax  = atom_type.indexr().lmax();
    int lmmax = utils::lmmax(2 * lmax);

    auto l_by_lm = utils::l_by_lm(2 * lmax);

    Gaunt_coefficients<double> GC(lmax, 2 * lmax, lmax, SHT::gaunt_rrr);

    auto nbrf = atom_type.num_beta_radial_functions();

    /* store integrals here */
    sddk::mdarray<double, 3> integrals(lmmax, nbrf * (nbrf + 1) / 2, ctx_.num_mag_dims() + 1);

    auto& rgrid = atom_type.radial_grid();

    for (int imagn = 0; imagn < ctx_.num_mag_dims() + 1; imagn++) {
        auto& ae_atom_pot = paw_potential_->ae_component(imagn)[ia__];
        auto& ps_atom_pot = paw_potential_->ps_component(imagn)[ia__];

        for (int irb2 = 0; irb2 < nbrf; irb2++) {
            for (int irb1 = 0; irb1 <= irb2; irb1++) {

                /* create array for integration */
                std::vector<double> intdata(rgrid.num_points());

                for (int lm3 = 0; lm3 < lmmax; lm3++) {
                    /* fill array */
                    for (int irad = 0; irad < rgrid.num_points(); irad++) {
                        double ae_part = paw_ae_wfs(irad, irb1) * paw_ae_wfs(irad, irb2);
                        double ps_part = paw_ps_wfs(irad, irb1) * paw_ps_wfs(irad, irb2) +
                                         atom_type.q_radial_function(irb1, irb2, l_by_lm[lm3])(irad);

                        intdata[irad] = ae_atom_pot(lm3, irad) * ae_part - ps_atom_pot(lm3, irad) * ps_part;
                    }

                    /* integrate */
                    integrals(lm3, utils::packed_index(irb1, irb2), imagn) = Spline<double>(rgrid, intdata).integrate(0);
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
            int iqij = utils::packed_index(irb1, irb2);

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
Potential::calc_PAW_one_elec_energy(int ia__, sddk::mdarray<std::complex<double>, 4> const& density_matrix__,
        sddk::mdarray<double, 3> const& paw_dij__) const
{
    std::complex<double> energy = 0.0;

    auto& atom = unit_cell_.atom(ia__);

    for (int ib2 = 0; ib2 < atom.mt_lo_basis_size(); ib2++) {
        for (int ib1 = 0; ib1 < atom.mt_lo_basis_size(); ib1++) {
            double dm[4] = {0, 0, 0, 0};
            switch (ctx_.num_mag_dims()) {
                case 3: {
                    dm[2] = 2 * std::real(density_matrix__(ib1, ib2, 2, ia__));
                    dm[3] = -2 * std::imag(density_matrix__(ib1, ib2, 2, ia__));
                }
                case 1: {
                    dm[0] = std::real(density_matrix__(ib1, ib2, 0, ia__) + density_matrix__(ib1, ib2, 1, ia__));
                    dm[1] = std::real(density_matrix__(ib1, ib2, 0, ia__) - density_matrix__(ib1, ib2, 1, ia__));
                    break;
                }
                case 0: {
                    dm[0] = density_matrix__(ib1, ib2, 0, ia__).real();
                    break;
                }
                default: {
                    RTE_THROW("calc_PAW_one_elec_energy FATAL ERROR!");
                    break;
                }
            }
            for (int imagn = 0; imagn < ctx_.num_mag_dims() + 1; imagn++) {
                energy += dm[imagn] * paw_dij__(ib1, ib2, imagn);
            }
        }
    }

    if (std::abs(energy.imag()) > 1e-10) {
        std::stringstream s;
        s << "PAW energy is not real: " << energy;
        RTE_THROW(s.str());
    }

    return energy.real();
}

} // namespace sirius
