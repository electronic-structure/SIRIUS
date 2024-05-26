/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file atom_type.cpp
 *
 *  \brief Contains implementation of sirius::Atom_type class.
 */

#include "atom_type.hpp"
#include "core/ostream_tools.hpp"
#include "core/traits.hpp"
#include <algorithm>

namespace sirius {

void
Atom_type::init()
{
    PROFILE("sirius::Atom_type::init");

    /* check if the class instance was already initialized */
    if (initialized_) {
        RTE_THROW("can't initialize twice");
    }

    /* read data from file if it exists */
    read_input(file_name_);

    /* check the nuclear charge */
    if (zn_ == 0) {
        RTE_THROW("zero atom charge");
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
            init_aw_descriptors();
        }

        if (static_cast<int>(aw_descriptors_.size()) != (this->lmax_apw() + 1)) {
            std::stringstream s;
            s << "wrong size of augmented wave descriptors" << std::endl
              << "  aw_descriptors_.size() = " << aw_descriptors_.size() << std::endl
              << "  lmax_apw = " << this->lmax_apw() << std::endl;
            RTE_THROW(s);
        }

        max_aw_order_ = 0;
        for (int l = 0; l <= this->lmax_apw(); l++) {
            max_aw_order_ = std::max(max_aw_order_, (int)aw_descriptors_[l].size());
        }

        if (max_aw_order_ > 3) {
            RTE_THROW("maximum aw order > 3");
        }
        /* build radial function index */
        for (auto aw : aw_descriptors_) {
            RTE_ASSERT(aw.size() <= 3);
            for (auto e : aw) {
                indexr_.add(angular_momentum(e.l));
            }
        }
        for (auto e : lo_descriptors_) {
            indexr_.add_lo(e.am);
        }
    } else {
        for (int i = 0; i < this->num_beta_radial_functions(); i++) {
            auto idxrf = rf_index(i);
            if (this->spin_orbit_coupling()) {
                if (this->beta_radial_function(idxrf).first.l() == 0) {
                    indexr_.add(this->beta_radial_function(idxrf).first);
                } else {
                    indexr_.add(this->beta_radial_function(idxrf).first,
                                this->beta_radial_function(rf_index(i + 1)).first);
                    i++;
                }
            } else {
                indexr_.add(this->beta_radial_function(idxrf).first);
            }
        }
        /* check inner consistency of the index */
        for (auto e : this->indexr_) {
            if (e.am != this->beta_radial_function(e.idxrf).first) {
                RTE_THROW("wrong order of beta radial functions");
            }
        }
    }

    /* initialize index of muffin-tin basis functions */
    indexb_ = basis_functions_index(indexr_, false);

    /* initialize index for wave functions */
    if (ps_atomic_wfs_.size()) {
        for (auto& e : ps_atomic_wfs_) {
            indexr_wfs_.add(e.am);
        }
        indexb_wfs_ = basis_functions_index(indexr_wfs_, false);
        if (static_cast<int>(ps_atomic_wfs_.size()) != indexr_wfs_.size()) {
            RTE_THROW("wrong size of atomic orbital list");
        }
    }

    if (hubbard_correction_) {
        indexb_hub_ = basis_functions_index(indexr_hub_, false);
    }

    if (!parameters_.full_potential()) {
        RTE_ASSERT(mt_radial_basis_size() == num_beta_radial_functions());
    }

    /* get number of valence electrons */
    num_valence_electrons_ = zn_ - num_core_electrons_;

    int lmmax_pot = sf::lmmax(parameters_.lmax_pot());

    if (parameters_.full_potential()) {
        auto l_by_lm = sf::l_by_lm(parameters_.lmax_pot());

        /* index the non-zero radial integrals */
        std::vector<std::pair<int, int>> non_zero_elements;

        for (int lm = 0; lm < lmmax_pot; lm++) {
            int l = l_by_lm[lm];

            for (int i2 = 0; i2 < indexr().size(); i2++) {
                int l2 = indexr(i2).am.l();
                for (int i1 = 0; i1 <= i2; i1++) {
                    int l1 = indexr(i1).am.l();
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
        idx_radial_integrals_ = mdarray<int, 2>({2, non_zero_elements.size()});
        for (int j = 0; j < (int)non_zero_elements.size(); j++) {
            idx_radial_integrals_(0, j) = non_zero_elements[j].first;
            idx_radial_integrals_(1, j) = non_zero_elements[j].second;
        }
    }

    if (parameters_.processing_unit() == device_t::GPU && parameters_.full_potential()) {
        idx_radial_integrals_.allocate(memory_t::device).copy_to(memory_t::device);
        rf_coef_ = mdarray<double, 3>({num_mt_points(), 4, indexr().size()}, memory_t::host_pinned,
                                      mdarray_label("Atom_type::rf_coef_"));
        vrf_coef_ =
                mdarray<double, 3>({num_mt_points(), 4, lmmax_pot * indexr().size() * (parameters_.num_mag_dims() + 1)},
                                   memory_t::host_pinned, mdarray_label("Atom_type::vrf_coef_"));
        rf_coef_.allocate(memory_t::device);
        vrf_coef_.allocate(memory_t::device);
    }
    if (parameters_.processing_unit() == device_t::GPU) {
        radial_grid_.copy_to_device();
    }

    if (this->spin_orbit_coupling()) {
        this->generate_f_coefficients();
    }

    if (is_paw()) {
        if (num_beta_radial_functions() != num_ps_paw_wf()) {
            RTE_THROW("wrong number of pseudo wave-functions for PAW");
        }
        if (num_beta_radial_functions() != num_ae_paw_wf()) {
            RTE_THROW("wrong number of all-electron wave-functions for PAW");
        }
        ae_paw_wfs_array_ = mdarray<double, 2>({num_mt_points(), num_beta_radial_functions()});
        ae_paw_wfs_array_.zero();
        ps_paw_wfs_array_ = mdarray<double, 2>({num_mt_points(), num_beta_radial_functions()});
        ps_paw_wfs_array_.zero();

        for (int i = 0; i < num_beta_radial_functions(); i++) {
            std::copy(ae_paw_wf(i).begin(), ae_paw_wf(i).end(), &ae_paw_wfs_array_(0, i));
            std::copy(ps_paw_wf(i).begin(), ps_paw_wf(i).end(), &ps_paw_wfs_array_(0, i));
        }
    }

    if (free_atom_radial_grid_.num_points() == 0) {
        free_atom_radial_grid_ = Radial_grid_factory<double>(radial_grid_t::power, 3000, radial_grid_.first(), 10.0, 2);
        free_atom_density_.resize(free_atom_radial_grid_.num_points());
        for (int i = 0; i < free_atom_radial_grid_.num_points(); i++) {
            auto x                = free_atom_radial_grid_.x(i);
            free_atom_density_[i] = std::exp(-x) * zn_ / 8 / pi;
        }
    }

    if (parameters_.full_potential()) {
        using gc_z   = Gaunt_coefficients<std::complex<double>>;
        gaunt_coefs_ = std::make_unique<gc_z>(std::max(this->lmax_apw(), this->lmax_lo()),
                                              std::max(parameters_.lmax_rho(), parameters_.lmax_pot()),
                                              std::max(this->lmax_apw(), this->lmax_lo()), SHT::gaunt_hybrid);
    }

    initialized_ = true;
}

void
Atom_type::init_free_atom_density(bool smooth)
{
    free_atom_density_spline_ = Spline<double>(free_atom_radial_grid_);
    /* smooth free atom density inside the muffin-tin sphere */
    if (smooth) {
        /* find point on the grid close to the muffin-tin radius */
        int irmt = free_atom_radial_grid_.index_of(mt_radius());
        /* interpolate at this point near MT radius */
        double R = free_atom_radial_grid_[irmt];

        /* make smooth free atom density inside muffin-tin */
        for (int i = 0; i <= irmt; i++) {
            double x = free_atom_radial_grid(i);
            // free_atom_density_spline_(i) = b(0) * std::pow(free_atom_radial_grid(i), 2) + b(1) *
            // std::pow(free_atom_radial_grid(i), 3);
            free_atom_density_spline_(i) = free_atom_density_[i] * 0.5 * (1 + std::erf((x / R - 0.5) * 10));
        }

        ///* write smoothed density */
        // sstr.str("");
        // sstr << "free_density_modified_" << id_ << ".dat";
        // fout = fopen(sstr.str().c_str(), "w");

        // for (int ir = 0; ir < free_atom_radial_grid().num_points(); ir++) {
        //    fprintf(fout, "%18.12f %18.12f \n", free_atom_radial_grid(ir), free_atom_density(ir));
        //}
        // fclose(fout);
    } else {
        for (int i = 0; i < free_atom_radial_grid_.num_points(); i++) {
            free_atom_density_spline_(i) = free_atom_density_[i];
        }
    }
    free_atom_density_spline_.interpolate();
}

void
Atom_type::print_info(std::ostream& out__) const
{
    out__ << "label          : " << label() << std::endl
          << hbar(80, '-') << std::endl
          << "symbol         : " << symbol_ << std::endl
          << "name           : " << name_ << std::endl
          << "zn             : " << zn_ << std::endl
          << "mass           : " << mass_ << std::endl
          << "mt_radius      : " << mt_radius() << std::endl
          << "num_mt_points  : " << num_mt_points() << std::endl
          << "grid_origin    : " << radial_grid_.first() << std::endl
          << "grid_name      : " << radial_grid_.name() << std::endl
          << std::endl
          << "number of core electrons    : " << num_core_electrons_ << std::endl
          << "number of valence electrons : " << num_valence_electrons_ << std::endl;

    if (parameters_.full_potential()) {
        out__ << std::endl;
        out__ << "atomic levels" << std::endl;
        for (auto& e : atomic_levels_) {
            out__ << "n: " << e.n << ", l: " << e.l << ", k: " << e.k << ", occ: " << e.occupancy
                  << ", core: " << e.core << std::endl;
        }
        out__ << std::endl;
        out__ << "local orbitals" << std::endl;
        for (auto e : lo_descriptors_) {
            out__ << "[";
            for (int order = 0; order < (int)e.rsd_set.size(); order++) {
                if (order) {
                    out__ << ", ";
                }
                out__ << e.rsd_set[order];
            }
            out__ << "]" << std::endl;
        }

        out__ << std::endl;
        out__ << "augmented wave basis" << std::endl;
        for (int j = 0; j < static_cast<int>(aw_descriptors_.size()); j++) {
            out__ << "[";
            for (int order = 0; order < static_cast<int>(aw_descriptors_[j].size()); order++) {
                if (order) {
                    out__ << ", ";
                }
                out__ << aw_descriptors_[j][order];
            }
            out__ << "]" << std::endl;
        }
        out__ << "maximum order of aw : " << max_aw_order_ << std::endl;
    }

    out__ << std::endl;
    out__ << "total number of radial functions : " << indexr().size() << std::endl
          << "lmax of radial functions         : " << indexr().lmax() << std::endl
          << "max. number of radial functions  : " << indexr().max_order() << std::endl
          << "total number of basis functions  : " << indexb().size() << std::endl
          << "number of aw basis functions     : " << indexb().size_aw() << std::endl
          << "number of lo basis functions     : " << indexb().size_lo() << std::endl
          << "lmax_apw                         : " << this->lmax_apw() << std::endl;
    if (!parameters_.full_potential()) {
        out__ << "lmax of beta-projectors          : " << this->lmax_beta() << std::endl
              << "number of ps wavefunctions       : " << this->indexr_wfs().size() << std::endl
              << "charge augmentation              : " << boolstr(this->augment()) << std::endl
              << "vloc is set                      : " << boolstr(!this->local_potential().empty()) << std::endl
              << "ps_rho_core is set               : " << boolstr(!this->ps_core_charge_density().empty()) << std::endl
              << "ps_rho_total is set              : " << boolstr(!this->ps_total_charge_density().empty())
              << std::endl;
    }
    out__ << "Hubbard correction               : " << boolstr(this->hubbard_correction()) << std::endl;
    if (parameters_.hubbard_correction() && this->hubbard_correction_) {
        out__ << "  angular momentum                   : " << lo_descriptors_hub_[0].l() << std::endl
              << "  principal quantum number           : " << lo_descriptors_hub_[0].n() << std::endl
              << "  occupancy                          : " << lo_descriptors_hub_[0].occupancy() << std::endl
              << "  U                                  : " << lo_descriptors_hub_[0].U() << std::endl
              << "  number of hubbard radial functions : " << indexr_hub_.size() << std::endl
              << "  number of hubbard basis functions  : " << indexb_hub_.size() << std::endl
              << "  Hubbard wave-functions             : ";
        for (int i = 0; i < indexr_hub_.size(); i++) {
            if (i) {
                out__ << ", ";
            }
            out__ << lo_descriptors_hub_[i];
        }

        bool orthogonalize_ = parameters_.cfg().hubbard().hubbard_subspace_method() == "orthogonalize";
        bool full_orthogonalization_ =
                parameters_.cfg().hubbard().hubbard_subspace_method() == "full_orthogonalization";
        bool normalize_ = parameters_.cfg().hubbard().hubbard_subspace_method() == "normalize";
        out__ << std::endl;
        out__ << "  orthogonalize                      : " << boolstr(orthogonalize_) << std::endl
              << "  normalize                          : " << boolstr(normalize_) << std::endl
              << "  full_orthogonalization             : " << boolstr(full_orthogonalization_) << std::endl
              << "  simplified                         : " << boolstr(parameters_.cfg().hubbard().simplified())
              << std::endl;
    }
    out__ << "spin-orbit coupling              : " << boolstr(this->spin_orbit_coupling()) << std::endl;
    out__ << "atomic wave-functions            : ";
    for (auto e : indexr_wfs_) {
        if (e.idxrf) {
            out__ << ", ";
        }
        out__ << e.am;
    }
    out__ << std::endl;
    out__ << std::endl;
}

void
Atom_type::read_input_core(nlohmann::json const& parser)
{
    std::string core_str = std::string(parser["core"]);
    if (int size = static_cast<int>(core_str.size())) {
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
                RTE_THROW(s);
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
                    RTE_THROW(s);
                }
            }

            for (auto& e : atomic_conf[zn_ - 1]) {
                if (e.n == n && e.l == l) {
                    auto level = e;
                    level.core = true;
                    atomic_levels_.push_back(level);
                }
            }
        }
    }
}

void
Atom_type::read_input_aw(nlohmann::json const& parser)
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

void
Atom_type::read_input_lo(nlohmann::json const& parser)
{
    radial_solution_descriptor rsd;
    radial_solution_descriptor_set rsd_set;

    if (!parser.count("lo")) {
        return;
    }

    int l;
    for (size_t j = 0; j < parser["lo"].size(); j++) {
        l = parser["lo"][j]["l"].get<int>();

        angular_momentum am(l);
        local_orbital_descriptor lod(am);
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

void
Atom_type::read_pseudo_uspp(nlohmann::json const& parser)
{
    symbol_ = parser["pseudo_potential"]["header"]["element"].get<std::string>();

    double zp;
    zp  = parser["pseudo_potential"]["header"]["z_valence"].get<double>();
    zn_ = int(zp + 1e-10);

    int nmtp = parser["pseudo_potential"]["header"]["mesh_size"].get<int>();

    auto rgrid = parser["pseudo_potential"]["radial_grid"].get<std::vector<double>>();
    if (static_cast<int>(rgrid.size()) != nmtp) {
        RTE_THROW("wrong mesh size");
    }
    /* set the radial grid */
    set_radial_grid(nmtp, rgrid.data());

    local_potential(parser["pseudo_potential"]["local_potential"].get<std::vector<double>>());

    ps_core_charge_density(
            parser["pseudo_potential"].value("core_charge_density", std::vector<double>(rgrid.size(), 0)));

    ps_total_charge_density(parser["pseudo_potential"]["total_charge_density"].get<std::vector<double>>());

    if (local_potential().size() != rgrid.size() || ps_core_charge_density().size() != rgrid.size() ||
        ps_total_charge_density().size() != rgrid.size()) {
        std::cout << local_potential().size() << " " << ps_core_charge_density().size() << " "
                  << ps_total_charge_density().size() << std::endl;
        RTE_THROW("wrong array size");
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
            RTE_THROW(s);
        }
        int l = parser["pseudo_potential"]["beta_projectors"][i]["angular_momentum"].get<int>();
        if (spin_orbit_coupling_) {
            // we encode the fact that the total angular momentum j = l
            // -1/2 or l + 1/2 by changing the sign of l

            double j = parser["pseudo_potential"]["beta_projectors"][i]["total_angular_momentum"].get<double>();
            if (j < static_cast<double>(l)) {
                l *= -1;
            }
        }
        // add_beta_radial_function(l, beta);
        if (spin_orbit_coupling_) {
            if (l >= 0) {
                add_beta_radial_function(angular_momentum(l, 1), beta);
            } else {
                add_beta_radial_function(angular_momentum(-l, -1), beta);
            }
        } else {
            add_beta_radial_function(angular_momentum(l), beta);
        }
    }

    mdarray<double, 2> d_mtrx({nbf, nbf});
    d_mtrx.zero();
    auto v = parser["pseudo_potential"]["D_ion"].get<std::vector<double>>();

    if (v.size() != nbf * nbf) {
        RTE_THROW("wrong size of D_ion");
    }

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
            int l    = parser["pseudo_potential"]["augmentation"][k]["angular_momentum"].get<int>();
            auto qij = parser["pseudo_potential"]["augmentation"][k]["radial_function"].get<std::vector<double>>();
            if (static_cast<int>(qij.size()) != num_mt_points()) {
                RTE_THROW("wrong size of qij");
            }
            add_q_radial_function(i, j, l, qij);
        }
    }

    /* read starting wave functions ( UPF CHI ) */
    if (parser["pseudo_potential"].count("atomic_wave_functions")) {
        auto& dict = parser["pseudo_potential"]["atomic_wave_functions"];
        /* total number of pseudo atomic wave-functions */
        size_t nwf = dict.size();
        /* loop over wave-functions */
        for (size_t k = 0; k < nwf; k++) {
            auto v = dict[k]["radial_function"].get<std::vector<double>>();

            if (static_cast<int>(v.size()) != num_mt_points()) {
                std::stringstream s;
                s << "wrong size of atomic functions for atom type " << symbol_ << " (label: " << label_ << ")"
                  << std::endl
                  << "size of atomic radial functions in the file: " << v.size() << std::endl
                  << "radial grid size: " << num_mt_points();
                RTE_THROW(s);
            }

            int l = dict[k]["angular_momentum"].get<int>();
            int n = -1;
            double occ{0};
            if (dict[k].count("occupation")) {
                occ = dict[k]["occupation"].get<double>();
            }

            if (dict[k].count("label")) {
                auto c1 = dict[k]["label"].get<std::string>();
                std::istringstream iss(std::string(1, c1[0]));
                iss >> n;
            }

            if (spin_orbit_coupling() && dict[k].count("total_angular_momentum") && l != 0) {

                auto v1 = dict[k + 1]["radial_function"].get<std::vector<double>>();
                double occ1{0};
                if (dict[k + 1].count("occupation")) {
                    occ1 = dict[k + 1]["occupation"].get<double>();
                }
                occ += occ1;
                for (int ir = 0; ir < num_mt_points(); ir++) {
                    v[ir] = 0.5 * v[ir] + 0.5 * v1[ir];
                }
                k += 1;
            }
            add_ps_atomic_wf(n, angular_momentum(l), v, occ);
        }
    }
}

void
Atom_type::read_pseudo_paw(nlohmann::json const& parser)
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
    paw_ae_core_charge_density(
            parser["pseudo_potential"]["paw_data"]["ae_core_charge_density"].get<std::vector<double>>());

    /* read occupations */
    paw_wf_occ(parser["pseudo_potential"]["paw_data"]["occupations"].get<std::vector<double>>());

    /* setups for reading AE and PS basis wave functions */
    int num_wfc = num_beta_radial_functions();

    /* read ae and ps wave functions */
    for (int i = 0; i < num_wfc; i++) {
        /* read ae wave func */
        auto wfc = parser["pseudo_potential"]["paw_data"]["ae_wfc"][i]["radial_function"].get<std::vector<double>>();

        if (static_cast<int>(wfc.size()) > num_mt_points()) {
            std::stringstream s;
            s << "wrong size of ae_wfc functions for atom type " << symbol_ << " (label: " << label_ << ")" << std::endl
              << "size of ae_wfc radial functions in the file: " << wfc.size() << std::endl
              << "radial grid size: " << num_mt_points();
            RTE_THROW(s);
        }

        add_ae_paw_wf(std::vector<double>(wfc.begin(), wfc.begin() + cutoff_radius_index));

        wfc = parser["pseudo_potential"]["paw_data"]["ps_wfc"][i]["radial_function"].get<std::vector<double>>();

        if (static_cast<int>(wfc.size()) > num_mt_points()) {
            std::stringstream s;
            s << "wrong size of ps_wfc functions for atom type " << symbol_ << " (label: " << label_ << ")" << std::endl
              << "size of ps_wfc radial functions in the file: " << wfc.size() << std::endl
              << "radial grid size: " << num_mt_points();
            RTE_THROW(s);
        }

        add_ps_paw_wf(std::vector<double>(wfc.begin(), wfc.begin() + cutoff_radius_index));
    }
}

void
Atom_type::read_input(nlohmann::json const& parser)
{

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
        try { /* overwrite the muffin-tin radius with the value from the inpupt */
            R = parameters_.cfg().unit_cell().atom_type_rmt(label_);
        } catch (...) {
        }

        int nmtp        = parser["nrmt"].get<int>();
        this->lmax_apw_ = parser.value("lmax_apw", this->lmax_apw_);

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
}

bool
is_upf_file(std::string const& str__)
{
    const std::string ftype = ".upf";
    auto lcstr              = str__;
    lcstr                   = trim(lcstr);
    if (lcstr.size() < ftype.size()) {
        return false;
    }
    return std::equal(ftype.rbegin(), ftype.rend(), lcstr.rbegin(),
                      [](char a, char b) { return std::tolower(a) == std::tolower(b); });
}

#ifdef SIRIUS_USE_PUGIXML
template <typename T>
std::vector<T>
vec_from_str(std::string const& str__, identity_t<T> scaling = T{1})
{
    std::string s;
    std::istringstream ss(str__);
    std::vector<T> data;
    while (ss >> s) {
        if constexpr (std::is_same_v<T, double>) {
            data.push_back(std::stod(s) * scaling);
        } else if constexpr (std::is_same_v<T, int>) {
            data.push_back(std::stoi(s) * scaling);
        } else if constexpr (std::is_same_v<T, float>) {
            data.push_back(std::stof(s) * scaling);
        } else {
            static_assert(!std::is_same_v<T, T>, "type not implemented");
        }
    }
    return data;
}

void
Atom_type::read_pseudo_uspp(pugi::xml_node const& upf)
{
    pugi::xml_node header = upf.child("PP_HEADER");
    symbol_               = header.attribute("element").as_string();

    double zp;
    zp  = header.attribute("z_valence").as_double();
    zn_ = int(zp + 1e-10);

    int nmtp = header.attribute("mesh_size").as_int();

    auto rgrid = vec_from_str<double>(upf.child("PP_MESH").child("PP_R").child_value());
    if (static_cast<int>(rgrid.size()) != nmtp) {
        RTE_THROW("wrong mesh size");
    }
    /* set the radial grid */
    set_radial_grid(nmtp, rgrid.data());

    local_potential(vec_from_str<double>(upf.child("PP_LOCAL").child_value(), 0.5));

    ps_core_charge_density(vec_from_str<double>(upf.child("PP_NLCC").child_value()));

    ps_total_charge_density(vec_from_str<double>(upf.child("PP_RHOATOM").child_value()));

    if (local_potential().size() != rgrid.size() || ps_core_charge_density().size() != rgrid.size() ||
        ps_total_charge_density().size() != rgrid.size()) {
        std::cout << local_potential().size() << " " << ps_core_charge_density().size() << " "
                  << ps_total_charge_density().size() << std::endl;
        RTE_THROW("wrong array size");
    }

    spin_orbit_coupling_ = header.attribute("has_so").as_bool();

    int nbf = header.attribute("number_of_proj").as_int();

    for (int i = 0; i < nbf; i++) {
        std::string bstr         = "PP_BETA." + std::to_string(i + 1);
        pugi::xml_node beta_node = upf.child("PP_NONLOCAL").child(bstr.data());
        auto beta_raw            = vec_from_str<double>(beta_node.child_value());

        int nr = beta_node.attribute("cutoff_radius_index").as_int();
        if (nr == 0) {
            for (int j = beta_raw.size() - 1; j >= 0; j--) {
                if (abs(beta_raw[j]) > 1.0e-80) {
                    nr = j + 1;
                    break;
                }
            }
        }
        if (nr == 0) {
            nr = beta_raw.size();
        }

        std::vector<double> beta(nr);
        std::copy(beta_raw.begin(), beta_raw.begin() + nr, beta.begin());

        if (static_cast<int>(beta.size()) > num_mt_points()) {
            std::stringstream s;
            s << "wrong size of beta functions for atom type " << symbol_ << " (label: " << label_ << ")" << std::endl
              << "size of beta radial functions in the file: " << beta.size() << std::endl
              << "radial grid size: " << num_mt_points();
            RTE_THROW(s);
        }
        int l = beta_node.attribute("angular_momentum").as_int();
        if (spin_orbit_coupling_) {
            // we encode the fact that the total angular momentum j = l
            // -1/2 or l + 1/2 by changing the sign of l
            std::string bstr       = "PP_RELBETA." + std::to_string(i + 1);
            pugi::xml_node so_node = upf.child("PP_SPIN_ORB").child(bstr.data());

            double j = so_node.attribute("jjj").as_double();
            if (j < static_cast<double>(l)) {
                l *= -1;
            }
        }
        // add_beta_radial_function(l, beta);
        if (spin_orbit_coupling_) {
            if (l >= 0) {
                add_beta_radial_function(angular_momentum(l, 1), beta);
            } else {
                add_beta_radial_function(angular_momentum(-l, -1), beta);
            }
        } else {
            add_beta_radial_function(angular_momentum(l), beta);
        }
    }

    mdarray<double, 2> d_mtrx({nbf, nbf});
    d_mtrx.zero();
    auto v = vec_from_str<double>(upf.child("PP_NONLOCAL").child("PP_DIJ").child_value(), 0.5);

    for (int i = 0; i < nbf; i++) {
        for (int j = 0; j < nbf; j++) {
            d_mtrx(i, j) = v[j * nbf + i];
        }
    }
    d_mtrx_ion(d_mtrx);

    pugi::xml_node nl_node = upf.child("PP_NONLOCAL");
    if (!nl_node.child("PP_AUGMENTATION").empty()) {
        for (int i = 0; i < nbf; i++) {
            std::string istr     = "PP_BETA." + std::to_string(i + 1);
            pugi::xml_node inode = nl_node.child(istr.data());
            int li               = inode.attribute("angular_momentum").as_int();

            for (int j = i; j < nbf; j++) {
                std::string jstr     = "PP_BETA." + std::to_string(j + 1);
                pugi::xml_node jnode = nl_node.child(jstr.data());
                int lj               = jnode.attribute("angular_momentum").as_int();

                for (int l = abs(li - lj); l < li + lj + 1; l++) {
                    if ((li + lj + l) % 2 != 0) {
                        continue;
                    }

                    std::string ijl_str =
                            "PP_QIJL." + std::to_string(i + 1) + "." + std::to_string(j + 1) + "." + std::to_string(l);
                    pugi::xml_node ijl_node = nl_node.child("PP_AUGMENTATION").child(ijl_str.data());

                    auto qij = vec_from_str<double>(ijl_node.child_value());
                    if (static_cast<int>(qij.size()) != num_mt_points()) {
                        RTE_THROW("wrong size of qij");
                    }
                    add_q_radial_function(i, j, l, qij);
                }
            }
        }
    }

    /* read starting wave functions ( UPF CHI ) */
    if (!header.attribute("number_of_wfc").empty()) {
        /* total number of pseudo atomic wave-functions */
        size_t nwf = header.attribute("number_of_wfc").as_int();
        /* loop over wave-functions */
        for (size_t k = 0; k < nwf; k++) {
            std::string wstr        = "PP_CHI." + std::to_string(k + 1);
            pugi::xml_node wfc_node = upf.child("PP_PSWFC").child(wstr.data());

            auto v = vec_from_str<double>(wfc_node.child_value());

            if (static_cast<int>(v.size()) != num_mt_points()) {
                std::stringstream s;
                s << "wrong size of atomic functions for atom type " << symbol_ << " (label: " << label_ << ")"
                  << std::endl
                  << "size of atomic radial functions in the file: " << v.size() << std::endl
                  << "radial grid size: " << num_mt_points();
                RTE_THROW(s);
            }

            int l = wfc_node.attribute("l").as_int();
            int n = -1;
            double occ{0};
            if (!wfc_node.attribute("occupation").empty()) {
                occ = wfc_node.attribute("occupation").as_double();
            }

            if (!wfc_node.attribute("label").empty()) {
                auto c1 = wfc_node.attribute("label").as_string();
                std::istringstream iss(std::string(1, c1[0]));
                iss >> n;
            }

            if (spin_orbit_coupling()) {
                std::string bstr       = "PP_RELWFC." + std::to_string(k + 1);
                pugi::xml_node so_node = upf.child("PP_SPIN_ORB").child(bstr.data());
                if (!so_node.attribute("jchi").empty() && l != 0) {

                    std::string wstr           = "PP_CHI." + std::to_string(k + 2);
                    pugi::xml_node wfc_node_p1 = upf.child("PP_PSWFC").child(wstr.data());

                    auto v1 = vec_from_str<double>(wfc_node_p1.child_value());
                    double occ1{0};
                    if (!wfc_node_p1.attribute("occupation").empty()) {
                        occ1 = wfc_node_p1.attribute("occupation").as_double();
                    }
                    occ += occ1;
                    for (int ir = 0; ir < num_mt_points(); ir++) {
                        v[ir] = 0.5 * v[ir] + 0.5 * v1[ir];
                    }
                    k += 1;
                }
            }
            add_ps_atomic_wf(n, angular_momentum(l), v, occ);
        }
    }
}

void
Atom_type::read_pseudo_paw(pugi::xml_node const& upf)
{
    is_paw_ = true;

    /* read core energy */
    if (!upf.child("PP_PAW").attribute("core_energy").empty()) {
        paw_core_energy(0.5 * upf.child("PP_PAW").attribute("core_energy").as_double());
    } else {
        paw_core_energy(0);
    }

    /* cutoff index */
    int cutoff_radius_index = upf.child("PP_NONLOCAL").child("PP_AUGMENTATION").attribute("cutoff_r_index").as_int();

    /* read core density and potential */
    paw_ae_core_charge_density(vec_from_str<double>(upf.child("PP_PAW").child("PP_AE_NLCC").child_value()));

    /* read occupations */
    paw_wf_occ(vec_from_str<double>(upf.child("PP_PAW").child("PP_OCCUPATIONS").child_value()));

    /* setups for reading AE and PS basis wave functions */
    int num_wfc = num_beta_radial_functions();

    /* read ae and ps wave functions */
    for (int i = 0; i < num_wfc; i++) {
        /* read ae wave func */
        std::string wstr        = "PP_AEWFC." + std::to_string(i + 1);
        pugi::xml_node wfc_node = upf.child("PP_FULL_WFC").child(wstr.data());
        auto wfc                = vec_from_str<double>(wfc_node.child_value());

        if (static_cast<int>(wfc.size()) > num_mt_points()) {
            std::stringstream s;
            s << "wrong size of ae_wfc functions for atom type " << symbol_ << " (label: " << label_ << ")" << std::endl
              << "size of ae_wfc radial functions in the file: " << wfc.size() << std::endl
              << "radial grid size: " << num_mt_points();
            RTE_THROW(s);
        }

        add_ae_paw_wf(std::vector<double>(wfc.begin(), wfc.begin() + cutoff_radius_index));

        wstr     = "PP_PSWFC." + std::to_string(i + 1);
        wfc_node = upf.child("PP_FULL_WFC").child(wstr.data());
        wfc      = vec_from_str<double>(wfc_node.child_value());

        if (static_cast<int>(wfc.size()) > num_mt_points()) {
            std::stringstream s;
            s << "wrong size of ps_wfc functions for atom type " << symbol_ << " (label: " << label_ << ")" << std::endl
              << "size of ps_wfc radial functions in the file: " << wfc.size() << std::endl
              << "radial grid size: " << num_mt_points();
            RTE_THROW(s);
        }

        add_ps_paw_wf(std::vector<double>(wfc.begin(), wfc.begin() + cutoff_radius_index));
    }
}

void
Atom_type::read_input(pugi::xml_node const& upf)
{
    if (!parameters_.full_potential()) {
        read_pseudo_uspp(upf);

        std::string pseudo_type = upf.child("PP_HEADER").attribute("pseudo_type").as_string();
        if (pseudo_type == "PAW") {
            read_pseudo_paw(upf);
        }
    } else {
        RTE_THROW("Full potential calculations require JSON potential files.");
    }
}
#endif

void
Atom_type::read_input(std::string const& str__)
{
    // Read from SIRIUS json potential file
    if (!is_upf_file(str__)) {
        auto parser = read_json_from_file_or_string(str__);

        if (parser.empty()) {
            return;
        }

        read_input(parser);

        // Read from standard UPF version 2 xml files
    } else {

#ifdef SIRIUS_USE_PUGIXML
        pugi::xml_document doc;
        pugi::xml_parse_result parser = doc.load_file(str__.data());

        if (!parser) {
            return;
        }

        pugi::xml_node upf = doc.child("UPF");
        if (upf.empty()) {
            RTE_THROW("SIRIUS can only read UPF files with version >= 2. Use the upf_to_json tool.");
        }
        read_input(upf);
#else
        RTE_THROW("SIRIUS cannot read UPF files directly without pugixml.")
#endif
    }

    /* it is already done in input.h; here the different constans are initialized */
    read_hubbard_input();
}

void
Atom_type::generate_f_coefficients()
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
    f_coefficients_ = mdarray<std::complex<double>, 4>({nbf, nbf, 2, 2});
    f_coefficients_.zero();

    for (int xi2 = 0; xi2 < nbf; xi2++) {
        const int l2    = this->indexb(xi2).am.l();
        const double j2 = this->indexb(xi2).am.j();
        const int m2    = this->indexb(xi2).m;
        for (int xi1 = 0; xi1 < nbf; xi1++) {
            const int l1    = this->indexb(xi1).am.l();
            const double j1 = this->indexb(xi1).am.j();
            const int m1    = this->indexb(xi1).m;

            if ((l2 == l1) && (std::abs(j1 - j2) < 1e-8)) {
                // take beta projectors with same l and j
                for (auto sigma2 = 0; sigma2 < 2; sigma2++) {
                    for (auto sigma1 = 0; sigma1 < 2; sigma1++) {
                        std::complex<double> coef = {0.0, 0.0};

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

void
Atom_type::read_hubbard_input()
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

                int sz    = static_cast<int>(initial_occupancy.size());
                int lmmax = 2 * ho.l() + 1;
                if (!(sz == 0 || sz == lmmax || sz == 2 * lmmax)) {
                    std::stringstream s;
                    s << "wrong size of initial occupacies vector (" << sz << ") for l = " << ho.l();
                    RTE_THROW(s);
                }
            }

            add_hubbard_orbital(ho.n(), ho.l(), ho.total_initial_occupancy(), coeff[0], coeff[1], &coeff[0], coeff[4],
                                coeff[5], 0.0, initial_occupancy, true);

            this->hubbard_correction_ = true;
        }
    }

    if (parameters_.cfg().hubbard().hubbard_subspace_method() == "full_orthogonalization") {
        this->hubbard_correction_ = true;
        if (lo_descriptors_hub_.empty()) {
            for (int s = 0; s < (int)ps_atomic_wfs_.size(); s++) {
                auto& e  = ps_atomic_wfs_[s];
                int n    = e.n;
                auto aqn = e.am;
                add_hubbard_orbital(n, aqn.l(), 0, 0, 0, nullptr, 0, 0, 0.0, std::vector<double>(2 * aqn.l() + 1, 0),
                                    false);
            }
        } else {
            for (int s = 0; s < (int)ps_atomic_wfs_.size(); s++) {
                auto& e  = ps_atomic_wfs_[s];
                int n    = e.n;
                auto aqn = e.am;

                // check if the orbital is already listed. In that case skip it
                for (int i = 0; i < parameters_.cfg().hubbard().local().size(); i++) {
                    auto ho = parameters_.cfg().hubbard().local(i);
                    if ((ho.atom_type() == this->label()) && ((ho.n() != n) || (ho.l() != aqn.l()))) {
                        // we add it to the list but we only use it for the orthogonalization procedure
                        add_hubbard_orbital(n, aqn.l(), 0, 0, 0, nullptr, 0, 0, 0.0,
                                            std::vector<double>(2 * aqn.l() + 1, 0), false);
                        break;
                    }
                }
            }
        }
    }
}

void
Atom_type::add_hubbard_orbital(int n__, int l__, double occ__, double U, double J, const double* hub_coef__,
                               double alpha__, double beta__, double J0__, std::vector<double> initial_occupancy__,
                               const bool use_for_calculations__)
{
    if (n__ <= 0) {
        RTE_THROW("negative principal quantum number");
    }

    /* we have to find index of the atomic function */
    int idx_rf{-1};
    for (int s = 0; s < static_cast<int>(ps_atomic_wfs_.size()); s++) {
        auto& e  = ps_atomic_wfs_[s];
        int n    = e.n;
        auto aqn = e.am;

        if ((n == n__) && (aqn.l() == l__)) {
            idx_rf = s;
            break;
        }
    }
    if (idx_rf == -1) {
        std::stringstream s;
        s << "atomic radial function is not found for atom type " << label_ << std::endl
          << "  the following atomic wave-functions are set: " << std::endl;
        for (int k = 0; k < (int)ps_atomic_wfs_.size(); k++) {
            auto& e  = ps_atomic_wfs_[k];
            int n    = e.n;
            auto aqn = e.am;
            s << "  n=" << n << " l=" << aqn.l() << " j=" << aqn.j() << std::endl;
        }
        s << "  the following atomic orbital is requested for U-correction: n=" << n__ << " l=" << l__;
        RTE_THROW(s);
    }

    /* create a scalar hubbard wave-function from one or two atomic radial functions */
    Spline<double> s(radial_grid_);
    for (int ir = 0; ir < s.num_points(); ir++) {
        s(ir) = ps_atomic_wfs_[idx_rf].f[ir];
    }

    /* add a record in radial function index */
    indexr_hub_.add(angular_momentum(l__));
    /* add Hubbard orbital descriptor to a list */
    lo_descriptors_hub_.emplace_back(n__, l__, -1, occ__, J, U, hub_coef__, alpha__, beta__, J0__, initial_occupancy__,
                                     std::move(s.interpolate()), use_for_calculations__, idx_rf);
}

} // namespace sirius
