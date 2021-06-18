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

/** \file atom_type.cpp
 *
 *  \brief Contains implementation of sirius::Atom_type class.
 */

#include "atom_type.hpp"

namespace sirius {

void Atom_type::init(int offset_lo__)
{
    PROFILE("sirius::Atom_type::init");

    /* check if the class instance was already initialized */
    if (initialized_) {
        TERMINATE("can't initialize twice");
    }

    offset_lo_ = offset_lo__;

    /* read data from file if it exists */
    read_input(file_name_);

    /* check the nuclear charge */
    if (zn_ == 0) {
        TERMINATE("zero atom charge");
    }

    if (parameters_.full_potential()) {
        /* add valence levels to the list of atom's levels */
        for (auto& e : atomic_conf[zn_ - 1]) {
            /* check if this level is already in the list */
            bool in_list{false};
            for (auto& c : atomic_levels_) {
                if (c.n == e.n && c.l == e.l && c.k == e.k) {
                    in_list = true;
                    break;
                }
            }
            if (!in_list) {
                auto level = e;
                level.core = false;
                atomic_levels_.push_back(level);
            }
        }
        /* get the number of core electrons */
        for (auto& e : atomic_levels_) {
            if (e.core) {
                num_core_electrons_ += e.occupancy;
            }
        }

        /* initialize aw descriptors if they were not set manually */
        if (aw_descriptors_.size() == 0) {
            init_aw_descriptors(parameters_.lmax_apw());
        }

        if (static_cast<int>(aw_descriptors_.size()) != (parameters_.lmax_apw() + 1)) {
            TERMINATE("wrong size of augmented wave descriptors");
        }

        max_aw_order_ = 0;
        for (int l = 0; l <= parameters_.lmax_apw(); l++) {
            max_aw_order_ = std::max(max_aw_order_, (int)aw_descriptors_[l].size());
        }

        if (max_aw_order_ > 3) {
            TERMINATE("maximum aw order > 3");
        }
    }

    /* initialize index of radial functions */
    indexr_.init(aw_descriptors_, lo_descriptors_);

    /* initialize index of muffin-tin basis functions */
    indexb_.init(indexr_);

    /* initialize index for wave functions */
    if (ps_atomic_wfs_.size()) {
        for (size_t i = 0; i < ps_atomic_wfs_.size(); i++) {
            if (ps_atomic_wfs_[i].am.s()) {
                if (ps_atomic_wfs_[i].am.l() == 0) {
                    indexr_wfs_.add(ps_atomic_wfs_[i].am);
                } else {
                    indexr_wfs_.add(ps_atomic_wfs_[i].am, ps_atomic_wfs_[i + 1].am);
                    i += 1;
                }
            } else {
                indexr_wfs_.add(ps_atomic_wfs_[i].am);
            }
        }
        indexb_wfs_ = sirius::experimental::basis_functions_index(indexr_wfs_, false);
        if (static_cast<int>(ps_atomic_wfs_.size()) != indexr_wfs_.size()) {
            RTE_THROW("wrong size of atomic orbital list");
        }
    }

    if (hubbard_correction_) {
        indexb_hub_ = sirius::experimental::basis_functions_index(indexr_hub_, false);
    }

    if (!parameters_.full_potential()) {
        assert(mt_radial_basis_size() == num_beta_radial_functions());
        assert(lmax_beta() == indexr().lmax());
    }

    /* get number of valence electrons */
    num_valence_electrons_ = zn_ - num_core_electrons_;

    int lmmax_pot = utils::lmmax(parameters_.lmax_pot());

    if (parameters_.full_potential()) {
        auto l_by_lm = utils::l_by_lm(parameters_.lmax_pot());

        /* index the non-zero radial integrals */
        std::vector<std::pair<int, int>> non_zero_elements;

        for (int lm = 0; lm < lmmax_pot; lm++) {
            int l = l_by_lm[lm];

            for (int i2 = 0; i2 < indexr().size(); i2++) {
                int l2 = indexr(i2).l;
                for (int i1 = 0; i1 <= i2; i1++) {
                    int l1 = indexr(i1).l;
                    if ((l + l1 + l2) % 2 == 0) {
                        if (lm) {
                            non_zero_elements.push_back(std::pair<int, int>(i2, lm + lmmax_pot * i1));
                        }
                        for (int j = 0; j < parameters_.num_mag_dims(); j++) {
                            int offs = (j + 1) * lmmax_pot * indexr().size();
                            non_zero_elements.push_back(std::pair<int, int>(i2, lm + lmmax_pot * i1 + offs));
                        }
                    }
                }
            }
        }
        idx_radial_integrals_ = mdarray<int, 2>(2, non_zero_elements.size());
        for (int j = 0; j < (int)non_zero_elements.size(); j++) {
            idx_radial_integrals_(0, j) = non_zero_elements[j].first;
            idx_radial_integrals_(1, j) = non_zero_elements[j].second;
        }
    }

    if (parameters_.processing_unit() == device_t::GPU && parameters_.full_potential()) {
        idx_radial_integrals_.allocate(memory_t::device).copy_to(memory_t::device);
        rf_coef_  = mdarray<double, 3>(num_mt_points(), 4, indexr().size(), memory_t::host_pinned, "Atom_type::rf_coef_");
        vrf_coef_ = mdarray<double, 3>(num_mt_points(), 4, lmmax_pot * indexr().size() * (parameters_.num_mag_dims() + 1),
                                       memory_t::host_pinned, "Atom_type::vrf_coef_");
        rf_coef_.allocate(memory_t::device);
        vrf_coef_.allocate(memory_t::device);
    }

    if (this->spin_orbit_coupling()) {
        this->generate_f_coefficients();
    }

    if (is_paw()) {
        if (num_beta_radial_functions() != num_ps_paw_wf()) {
            TERMINATE("wrong number of pseudo wave-functions for PAW");
        }
        if (num_beta_radial_functions() != num_ae_paw_wf()) {
            TERMINATE("wrong number of all-electron wave-functions for PAW");
        }
        ae_paw_wfs_array_ = mdarray<double, 2>(num_mt_points(), num_beta_radial_functions());
        ae_paw_wfs_array_.zero();
        ps_paw_wfs_array_ = mdarray<double, 2>(num_mt_points(), num_beta_radial_functions());
        ps_paw_wfs_array_.zero();

        for (int i = 0; i < num_beta_radial_functions(); i++) {
            std::copy(ae_paw_wf(i).begin(), ae_paw_wf(i).end(), &ae_paw_wfs_array_(0, i));
            std::copy(ps_paw_wf(i).begin(), ps_paw_wf(i).end(), &ps_paw_wfs_array_(0, i));
        }
    }

    initialized_ = true;
}

void Atom_type::init_free_atom_density(bool smooth)
{
    if (free_atom_density_.size() == 0) {
        TERMINATE("free atom density is not set");
    }

    free_atom_density_spline_ = Spline<double>(free_atom_radial_grid_, free_atom_density_);

    /* smooth free atom density inside the muffin-tin sphere */
    if (smooth) {
        /* find point on the grid close to the muffin-tin radius */
        int irmt = free_atom_radial_grid_.index_of(mt_radius());
        /* interpolate at this point near MT radius */
        double R = free_atom_radial_grid_[irmt];

        /* make smooth free atom density inside muffin-tin */
        for (int i = 0; i <= irmt; i++) {
            double x = free_atom_radial_grid(i);
            //free_atom_density_spline_(i) = b(0) * std::pow(free_atom_radial_grid(i), 2) + b(1) * std::pow(free_atom_radial_grid(i), 3);
            free_atom_density_spline_(i) = free_atom_density_[i] * 0.5 * (1 + std::erf((x / R - 0.5) * 10));
        }

        /* interpolate new smooth density */
        free_atom_density_spline_.interpolate();

        ///* write smoothed density */
        //sstr.str("");
        //sstr << "free_density_modified_" << id_ << ".dat";
        //fout = fopen(sstr.str().c_str(), "w");

        //for (int ir = 0; ir < free_atom_radial_grid().num_points(); ir++) {
        //    fprintf(fout, "%18.12f %18.12f \n", free_atom_radial_grid(ir), free_atom_density(ir));
        //}
        //fclose(fout);
    }
}

void Atom_type::print_info() const
{
    std::printf("\n");
    std::printf("label          : %s\n", label().c_str());
    for (int i = 0; i < 80; i++) {
        std::printf("-");
    }
    std::printf("\n");
    std::printf("symbol         : %s\n", symbol_.c_str());
    std::printf("name           : %s\n", name_.c_str());
    std::printf("zn             : %i\n", zn_);
    std::printf("mass           : %f\n", mass_);
    std::printf("mt_radius      : %f\n", mt_radius());
    std::printf("num_mt_points  : %i\n", num_mt_points());
    std::printf("grid_origin    : %f\n", radial_grid_.first());
    std::printf("grid_name      : %s\n", radial_grid_.name().c_str());
    std::printf("\n");
    std::printf("number of core electrons    : %f\n", num_core_electrons_);
    std::printf("number of valence electrons : %f\n", num_valence_electrons_);

    if (parameters_.full_potential()) {
        std::printf("\n");
        std::printf("atomic levels (n, l, k, occupancy, core)\n");
        for (int i = 0; i < (int)atomic_levels_.size(); i++) {
            std::printf("%i  %i  %i  %8.4f %i\n", atomic_levels_[i].n, atomic_levels_[i].l, atomic_levels_[i].k,
                   atomic_levels_[i].occupancy, atomic_levels_[i].core);
        }
        std::printf("\n");
        std::printf("local orbitals\n");
        for (int j = 0; j < (int)lo_descriptors_.size(); j++) {
            std::printf("[");
            for (int order = 0; order < (int)lo_descriptors_[j].rsd_set.size(); order++) {
                if (order)
                    std::printf(", ");
                std::printf("{l : %2i, n : %2i, enu : %f, dme : %i, auto : %i}", lo_descriptors_[j].rsd_set[order].l,
                       lo_descriptors_[j].rsd_set[order].n, lo_descriptors_[j].rsd_set[order].enu,
                       lo_descriptors_[j].rsd_set[order].dme, lo_descriptors_[j].rsd_set[order].auto_enu);
            }
            std::printf("]\n");
        }

        std::printf("\n");
        std::printf("augmented wave basis\n");
        for (int j = 0; j < (int)aw_descriptors_.size(); j++) {
            std::printf("[");
            for (int order = 0; order < (int)aw_descriptors_[j].size(); order++) {
                if (order)
                    std::printf(", ");
                std::printf("{l : %2i, n : %2i, enu : %f, dme : %i, auto : %i}", aw_descriptors_[j][order].l,
                       aw_descriptors_[j][order].n, aw_descriptors_[j][order].enu, aw_descriptors_[j][order].dme,
                       aw_descriptors_[j][order].auto_enu);
            }
            std::printf("]\n");
        }
        std::printf("maximum order of aw : %i\n", max_aw_order_);
    }

    std::printf("\n");
    std::printf("total number of radial functions : %i\n", indexr().size());
    std::printf("lmax of radial functions         : %i\n", indexr().lmax());
    std::printf("max. number of radial functions  : %i\n", indexr().max_num_rf());
    std::printf("total number of basis functions  : %i\n", indexb().size());
    std::printf("number of aw basis functions     : %i\n", indexb().size_aw());
    std::printf("number of lo basis functions     : %i\n", indexb().size_lo());
    if (!parameters_.full_potential()) {
        std::printf("lmax of beta-projectors          : %i\n", this->lmax_beta());
        std::printf("number of ps wavefunctions       : %i\n", static_cast<int>(this->indexr_wfs().size()));
        std::printf("charge augmentation              : %s\n", utils::boolstr(this->augment()).c_str());
        std::printf("vloc is set                      : %s\n", utils::boolstr(!this->local_potential().empty()).c_str());
        std::printf("ps_rho_core is set               : %s\n", utils::boolstr(!this->ps_core_charge_density().empty()).c_str());
        std::printf("ps_rho_total is set              : %s\n", utils::boolstr(!this->ps_total_charge_density().empty()).c_str());
    }
    std::printf("Hubbard correction               : %s\n", utils::boolstr(this->hubbard_correction()).c_str());
    if (parameters_.hubbard_correction() && this->hubbard_correction_) {
        std::printf("  angular momentum                   : %i\n", lo_descriptors_hub_[0].l);
        std::printf("  principal quantum number           : %i\n", lo_descriptors_hub_[0].n());
        std::printf("  occupancy                          : %f\n", lo_descriptors_hub_[0].occupancy());
        std::printf("  number of hubbard radial functions : %i\n", static_cast<int>(indexr_hub_.size()));
        std::printf("  number of hubbard basis functions  : %i\n", static_cast<int>(indexb_hub_.size()));
    }
    std::printf("spin-orbit coupling              : %s\n", utils::boolstr(this->spin_orbit_coupling()).c_str());
}

void Atom_type::read_input_core(nlohmann::json const& parser)
{
    std::string core_str = std::string(parser["core"]);
    if (int size = (int)core_str.size()) {
        if (size % 2) {
            std::stringstream s;
            s << "wrong core configuration string : " << core_str;
            RTE_THROW(s);
        }
        int j = 0;
        while (j < size) {
            char c1 = core_str[j++];
            char c2 = core_str[j++];

            int n = -1;
            int l = -1;

            std::istringstream iss(std::string(1, c1));
            iss >> n;

            if (n <= 0 || iss.fail()) {
                std::stringstream s;
                s << "wrong principal quantum number : " << std::string(1, c1);
                TERMINATE(s);
            }

            switch (c2) {
                case 's': {
                    l = 0;
                    break;
                }
                case 'p': {
                    l = 1;
                    break;
                }
                case 'd': {
                    l = 2;
                    break;
                }
                case 'f': {
                    l = 3;
                    break;
                }
                default: {
                    std::stringstream s;
                    s << "wrong angular momentum label : " << std::string(1, c2);
                    TERMINATE(s);
                }
            }

            for (auto& e: atomic_conf[zn_ - 1]) {
                if (e.n == n && e.l == l) {
                    auto level = e;
                    level.core = true;
                    atomic_levels_.push_back(level);
                }
            }
        }
    }
}

void Atom_type::read_input_aw(nlohmann::json const& parser)
{
    radial_solution_descriptor rsd;
    radial_solution_descriptor_set rsd_set;

    /* default augmented wave basis */
    rsd.n = -1;
    rsd.l = -1;
    for (size_t order = 0; order < parser["valence"][0]["basis"].size(); order++) {
        rsd.enu      = parser["valence"][0]["basis"][order]["enu"].get<double>();
        rsd.dme      = parser["valence"][0]["basis"][order]["dme"].get<int>();
        rsd.auto_enu = parser["valence"][0]["basis"][order]["auto"].get<int>();
        aw_default_l_.push_back(rsd);
    }

    for (size_t j = 1; j < parser["valence"].size(); j++) {
        rsd.l = parser["valence"][j]["l"].get<int>();
        rsd.n = parser["valence"][j]["n"].get<int>();
        rsd_set.clear();
        for (size_t order = 0; order < parser["valence"][j]["basis"].size(); order++) {
            rsd.enu      = parser["valence"][j]["basis"][order]["enu"].get<double>();
            rsd.dme      = parser["valence"][j]["basis"][order]["dme"].get<int>();
            rsd.auto_enu = parser["valence"][j]["basis"][order]["auto"].get<int>();
            rsd_set.push_back(rsd);
        }
        aw_specific_l_.push_back(rsd_set);
    }
}

void Atom_type::read_input_lo(nlohmann::json const& parser)
{
    radial_solution_descriptor rsd;
    radial_solution_descriptor_set rsd_set;

    if (!parser.count("lo")) {
        return;
    }

    int l;
    for (size_t j = 0; j < parser["lo"].size(); j++) {
        l = parser["lo"][j]["l"].get<int>();

        local_orbital_descriptor lod;
        lod.l = l;
        rsd.l = l;
        rsd_set.clear();
        for (size_t order = 0; order < parser["lo"][j]["basis"].size(); order++) {
            rsd.n        = parser["lo"][j]["basis"][order]["n"].get<int>();
            rsd.enu      = parser["lo"][j]["basis"][order]["enu"].get<double>();
            rsd.dme      = parser["lo"][j]["basis"][order]["dme"].get<int>();
            rsd.auto_enu = parser["lo"][j]["basis"][order]["auto"].get<int>();
            rsd_set.push_back(rsd);
        }
        lod.rsd_set = rsd_set;
        lo_descriptors_.push_back(lod);
    }
}


void Atom_type::read_pseudo_uspp(nlohmann::json const& parser)
{
    symbol_ = parser["pseudo_potential"]["header"]["element"].get<std::string>();

    double zp;
    zp  = parser["pseudo_potential"]["header"]["z_valence"].get<double>();
    zn_ = int(zp + 1e-10);

    int nmtp = parser["pseudo_potential"]["header"]["mesh_size"].get<int>();

    auto rgrid = parser["pseudo_potential"]["radial_grid"].get<std::vector<double>>();
    if (static_cast<int>(rgrid.size()) != nmtp) {
        TERMINATE("wrong mesh size");
    }
    /* set the radial grid */
    set_radial_grid(nmtp, rgrid.data());

    local_potential(parser["pseudo_potential"]["local_potential"].get<std::vector<double>>());

    ps_core_charge_density(parser["pseudo_potential"].value("core_charge_density", std::vector<double>(rgrid.size(), 0)));

    ps_total_charge_density(parser["pseudo_potential"]["total_charge_density"].get<std::vector<double>>());

    if (local_potential().size() != rgrid.size() || ps_core_charge_density().size() != rgrid.size() ||
        ps_total_charge_density().size() != rgrid.size()) {
        std::cout << local_potential().size() << " " << ps_core_charge_density().size() << " "
                  << ps_total_charge_density().size() << std::endl;
        TERMINATE("wrong array size");
    }

    if (parser["pseudo_potential"]["header"].count("spin_orbit")) {
        spin_orbit_coupling_ = parser["pseudo_potential"]["header"].value("spin_orbit", spin_orbit_coupling_);
    }

    int nbf = parser["pseudo_potential"]["header"]["number_of_proj"].get<int>();

    for (int i = 0; i < nbf; i++) {
        auto beta = parser["pseudo_potential"]["beta_projectors"][i]["radial_function"].get<std::vector<double>>();
        if (static_cast<int>(beta.size()) > num_mt_points()) {
            std::stringstream s;
            s << "wrong size of beta functions for atom type " << symbol_ << " (label: " << label_ << ")" << std::endl
              << "size of beta radial functions in the file: " << beta.size() << std::endl
              << "radial grid size: " << num_mt_points();
            TERMINATE(s);
        }
        int l = parser["pseudo_potential"]["beta_projectors"][i]["angular_momentum"].get<int>();
        if (spin_orbit_coupling_) {
            // we encode the fact that the total angular momentum j = l
            // -1/2 or l + 1/2 by changing the sign of l

            double j = parser["pseudo_potential"]["beta_projectors"][i]["total_angular_momentum"].get<double>();
            if (j < (double)l) {
                l *= -1;
            }
        }
        add_beta_radial_function(l, beta);
    }

    mdarray<double, 2> d_mtrx(nbf, nbf);
    d_mtrx.zero();
    auto v = parser["pseudo_potential"]["D_ion"].get<std::vector<double>>();

    for (int i = 0; i < nbf; i++) {
        for (int j = 0; j < nbf; j++) {
            d_mtrx(i, j) = v[j * nbf + i];
        }
    }
    d_mtrx_ion(d_mtrx);

    if (parser["pseudo_potential"].count("augmentation")) {
        for (size_t k = 0; k < parser["pseudo_potential"]["augmentation"].size(); k++) {
            int i    = parser["pseudo_potential"]["augmentation"][k]["i"].get<int>();
            int j    = parser["pseudo_potential"]["augmentation"][k]["j"].get<int>();
            //int idx  = j * (j + 1) / 2 + i;
            int l    = parser["pseudo_potential"]["augmentation"][k]["angular_momentum"].get<int>();
            auto qij = parser["pseudo_potential"]["augmentation"][k]["radial_function"].get<std::vector<double>>();
            if ((int)qij.size() != num_mt_points()) {
                TERMINATE("wrong size of qij");
            }
            add_q_radial_function(i, j, l, qij);
        }
    }

    /* read starting wave functions ( UPF CHI ) */
    if (parser["pseudo_potential"].count("atomic_wave_functions")) {
        /* total number of pseudo atomic wave-functions */
        size_t nwf = parser["pseudo_potential"]["atomic_wave_functions"].size();
        /* loop over wave-functions */
        for (size_t k = 0; k < nwf; k++) {
            auto v = parser["pseudo_potential"]["atomic_wave_functions"][k]["radial_function"].get<std::vector<double>>();

            if ((int)v.size() != num_mt_points()) {
                std::stringstream s;
                s << "wrong size of atomic functions for atom type " << symbol_ << " (label: " << label_ << ")"
                  << std::endl
                  << "size of atomic radial functions in the file: " << v.size() << std::endl
                  << "radial grid size: " << num_mt_points();
                TERMINATE(s);
            }

            int l = parser["pseudo_potential"]["atomic_wave_functions"][k]["angular_momentum"].get<int>();
            int n = -1;
            double occ{0};
            if (parser["pseudo_potential"]["atomic_wave_functions"][k].count("occupation")) {
                occ = parser["pseudo_potential"]["atomic_wave_functions"][k]["occupation"].get<double>();
            }

            if (parser["pseudo_potential"]["atomic_wave_functions"][k].count("label")) {
                auto c1 = parser["pseudo_potential"]["atomic_wave_functions"][k]["label"].get<std::string>();
                std::istringstream iss(std::string(1, c1[0]));
                iss >> n;
            }

            int s{0};

            if (spin_orbit_coupling() &&
                parser["pseudo_potential"]["atomic_wave_functions"][k].count("total_angular_momentum")) {
                // check if j = l +- 1/2
                if (parser["pseudo_potential"]["atomic_wave_functions"][k]["total_angular_momentum"].get<int>() < l) {
                    s = -1;
                } else {
                    s = 1;
                }
            }
            add_ps_atomic_wf(n, sirius::experimental::angular_momentum(l, s), v, occ);
        }
    }
}

void Atom_type::read_pseudo_paw(nlohmann::json const& parser)
{
    is_paw_ = true;

    auto& header = parser["pseudo_potential"]["header"];
    /* read core energy */
    if (header.count("paw_core_energy")) {
        paw_core_energy(header["paw_core_energy"].get<double>());
    } else {
        paw_core_energy(0);
    }

    /* cutoff index */
    int cutoff_radius_index = parser["pseudo_potential"]["header"]["cutoff_radius_index"].get<int>();

    /* read core density and potential */
    paw_ae_core_charge_density(parser["pseudo_potential"]["paw_data"]["ae_core_charge_density"].get<std::vector<double>>());

    /* read occupations */
    paw_wf_occ(parser["pseudo_potential"]["paw_data"]["occupations"].get<std::vector<double>>());

    /* setups for reading AE and PS basis wave functions */
    int num_wfc = num_beta_radial_functions();

    /* read ae and ps wave functions */
    for (int i = 0; i < num_wfc; i++) {
        /* read ae wave func */
        auto wfc = parser["pseudo_potential"]["paw_data"]["ae_wfc"][i]["radial_function"].get<std::vector<double>>();

        if ((int)wfc.size() > num_mt_points()) {
            std::stringstream s;
            s << "wrong size of ae_wfc functions for atom type " << symbol_ << " (label: " << label_ << ")" << std::endl
              << "size of ae_wfc radial functions in the file: " << wfc.size() << std::endl
              << "radial grid size: " << num_mt_points();
            TERMINATE(s);
        }

        add_ae_paw_wf(std::vector<double>(wfc.begin(), wfc.begin() + cutoff_radius_index));

        wfc = parser["pseudo_potential"]["paw_data"]["ps_wfc"][i]["radial_function"].get<std::vector<double>>();

        if ((int)wfc.size() > num_mt_points()) {
            std::stringstream s;
            s << "wrong size of ps_wfc functions for atom type " << symbol_ << " (label: " << label_ << ")" << std::endl
              << "size of ps_wfc radial functions in the file: " << wfc.size() << std::endl
              << "radial grid size: " << num_mt_points();
            TERMINATE(s);
        }

        add_ps_paw_wf(std::vector<double>(wfc.begin(), wfc.begin() + cutoff_radius_index));
    }
}

void Atom_type::read_input(std::string const& str__)
{
    auto parser = utils::read_json_from_file_or_string(str__);

    if (parser.empty()) {
        return;
    }

    if (!parameters_.full_potential()) {
        read_pseudo_uspp(parser);

        if (parser["pseudo_potential"].count("paw_data")) {
            read_pseudo_paw(parser);
        }
    }

    if (parameters_.full_potential()) {
        name_     = parser["name"].get<std::string>();
        symbol_   = parser["symbol"].get<std::string>();
        mass_     = parser["mass"].get<double>();
        zn_       = parser["number"].get<int>();
        double r0 = parser["rmin"].get<double>();
        double R  = parser["rmt"].get<double>();
        int nmtp  = parser["nrmt"].get<int>();

        auto rg = get_radial_grid_t(parameters_.cfg().settings().radial_grid());

        set_radial_grid(rg.first, nmtp, r0, R, rg.second);

        read_input_core(parser);

        read_input_aw(parser);

        read_input_lo(parser);

        /* create free atom radial grid */
        auto fa_r              = parser["free_atom"]["radial_grid"].get<std::vector<double>>();
        free_atom_radial_grid_ = Radial_grid_ext<double>(static_cast<int>(fa_r.size()), fa_r.data());
        /* read density */
        free_atom_density_ = parser["free_atom"]["density"].get<std::vector<double>>();
    }

    /* it is already done in input.h; here the different constans are initialized */
    read_hubbard_input();
}


void Atom_type::generate_f_coefficients()
{
    // we consider Pseudo potentials with spin orbit couplings

    // First thing, we need to compute the
    // \f[f^{\sigma\sigma^\prime}_{l,j,m;l\prime,j\prime,m\prime}\f]
    // They are defined by Eq.9 of doi:10.1103/PhysRevB.71.115106
    // and correspond to transformations of the
    // spherical harmonics
    if (!this->spin_orbit_coupling()) {
        return;
    }

    // number of beta projectors
    int nbf         = this->mt_basis_size();
    f_coefficients_ = mdarray<double_complex, 4>(nbf, nbf, 2, 2);
    f_coefficients_.zero();

    for (int xi2 = 0; xi2 < nbf; xi2++) {
        const int l2    = this->indexb(xi2).l;
        const double j2 = this->indexb(xi2).j;
        const int m2    = this->indexb(xi2).m;
        for (int xi1 = 0; xi1 < nbf; xi1++) {
            const int l1    = this->indexb(xi1).l;
            const double j1 = this->indexb(xi1).j;
            const int m1    = this->indexb(xi1).m;

            if ((l2 == l1) && (std::abs(j1 - j2) < 1e-8)) {
                // take beta projectors with same l and j
                for (auto sigma2 = 0; sigma2 < 2; sigma2++) {
                    for (auto sigma1 = 0; sigma1 < 2; sigma1++) {
                        double_complex coef = {0.0, 0.0};

                        // yes dirty but loop over double is worst.
                        // since mj is only important for the rotation
                        // of the spherical harmonics the code takes
                        // into account this odd convention.

                        int jj1 = static_cast<int>(2.0 * j1 + 1e-8);
                        for (int mj = -jj1; mj <= jj1; mj += 2) {
                            coef += sht::calculate_U_sigma_m(l1, j1, mj, m1, sigma1) *
                                sht::ClebschGordan(l1, j1, mj / 2.0, sigma1) *
                                std::conj(sht::calculate_U_sigma_m(l2, j2, mj, m2, sigma2)) *
                                sht::ClebschGordan(l2, j2, mj / 2.0, sigma2);
                        }
                        f_coefficients_(xi1, xi2, sigma1, sigma2) = coef;
                    }
                }
            }
        }
    }
}

void Atom_type::read_hubbard_input()
{
    if (!parameters_.cfg().parameters().hubbard_correction()) {
        return;
    }

    this->hubbard_correction_ = false;

    for (int i = 0; i < parameters_.cfg().hubbard().local().size(); i++) {
        auto ho = parameters_.cfg().hubbard().local(i);
        if (ho.atom_type() == this->label()) {
            std::array<double, 6> coeff{0, 0, 0, 0, 0, 0};
            if (ho.contains("U")) {
                coeff[0] = ho.U();
            }
            if (ho.contains("J")) {
                coeff[1] = ho.J();
            }
            if (ho.contains("BE2")) {
                coeff[2] = ho.BE2();
            }
            if (ho.contains("E3")) {
                coeff[3] = ho.E3();
            }
            if (ho.contains("alpha")) {
                coeff[4] = ho.alpha();
            }
            if (ho.contains("beta")) {
                coeff[5] = ho.beta();
            }
            /* now convert eV in Ha */
            for (int s = 0; s < 6; s++) {
                coeff[s] /= ha2ev;
            }
            std::vector<double> initial_occupancy;
            if (ho.contains("initial_occupancy")) {
                initial_occupancy = ho.initial_occupancy();

                int sz = static_cast<int>(initial_occupancy.size());
                int lmmax = 2 * ho.l() + 1;
                if (!(sz == 0 || sz == lmmax || sz == 2 * lmmax)) {
                    std::stringstream s;
                    s << "wrong size of initial occupacies vector (" << sz << ") for l = " << ho.l();
                    RTE_THROW(s);
                }
            }

            add_hubbard_orbital(ho.n(), ho.l(), ho.total_initial_occupancy(), coeff[0], coeff[1], &coeff[0],
                    coeff[4], coeff[5], 0.0, initial_occupancy);

            this->hubbard_correction_ = true;
        }
    }
}

void Atom_type::add_hubbard_orbital(int n__, int l__, double occ__, double U, double J, const double *hub_coef__,
                         double alpha__, double beta__, double J0__, std::vector<double> initial_occupancy__)
{
    // TODO: pass radial function for l or for j=l+1/2 j=l-1/2 and don't rely on the list of pseudoatomic wfs

    if (n__ <= 0) {
        RTE_THROW("negative principal quantum number");
    }

    /* we have to find one (or two in case of spin-orbit) atomic functions and construct hubbard orbital */
    std::vector<int> idx_rf;
    for (int s = 0; s < (int)ps_atomic_wfs_.size(); s++) {
        auto& e = ps_atomic_wfs_[s];
        int n = e.n;
        auto aqn = e.am;
        if (n == n__ && aqn.l() == l__) {
            idx_rf.push_back(s);
            /* in spin orbit case we need to find the second radial function, otherwise we break */
            if (!(aqn.s() && aqn.l() > 0)) {
                break;
            }
        }
    }
    if (idx_rf.size() == 0) {
        std::stringstream s;
        s << "atomic radial function is not found for atom type " << label_ << std::endl
          << "  the following atomic wave-functions are set: " << std::endl;
        for (int k = 0; k < (int)ps_atomic_wfs_.size(); k++) {
            auto& e = ps_atomic_wfs_[k];
            int n = e.n;
            auto aqn = e.am;
            s << "  n=" << n << " l=" << aqn.l() << " j=" << aqn.j() << std::endl;
        }
        s << "  the following atomic orbital is requested for U-correction: n=" << n__ << " l=" << l__;
        RTE_THROW(s);
    }
    if (idx_rf.size() > 2) {
        std::stringstream s;
        s << "number of atomic functions > 2";
        RTE_THROW(s);
    }

    /* create a scalar hubbard wave-function from one or two atomic radial functions */
    Spline<double> s(radial_grid_);
    double f = 1.0 / static_cast<double>(idx_rf.size());
    for (int i: idx_rf) {
        auto& rwf = ps_atomic_wfs_[i].f;
        for (int ir = 0; ir < s.num_points(); ir++) {
            s(ir) += f * rwf(ir);
        }
    }

    /* add a record in radial function index */
    indexr_hub_.add(sirius::experimental::angular_momentum(l__));

    /* add Hubbard orbital descriptor to a list */
    lo_descriptors_hub_.emplace_back(n__, l__, -1, occ__, J, U, hub_coef__, alpha__, beta__, J0__, initial_occupancy__, std::move(s.interpolate()));
}

} // namespace
