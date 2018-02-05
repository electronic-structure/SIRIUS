// Copyright (c) 2013-2017 Anton Kozhevnikov, Thomas Schulthess
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

/** \file atom_type.h
 *
 *  \brief Contains definition of sirius::radial_functions_index and sirius::basis_functions_index classes
 *         and declaration and partial implementation of sirius::Atom_type class.
 */

#ifndef __ATOM_TYPE_H__
#define __ATOM_TYPE_H__

#include "atomic_conf.h"
#include "descriptors.h"
#include "geometry3d.hpp"
#include "utils.h"
#include "radial_grid.h"
#include "radial_solver.h"
#include "xc_functional.h"
#include "simulation_parameters.h"
#include "sht.h"
#include "radial_functions_index.hpp"
#include "basis_functions_index.hpp"

namespace sirius {

/// Defines the properties of atom type.
/** Atoms wth the same properties are grouped by type. */
class Atom_type
{
  private:
    /// Basic parameters.
    Simulation_parameters const& parameters_;

    /// Unique id of atom type in the range [0, \f$ N_{types} \f$).
    int id_{-1};

    /// Unique string label for the atom type.
    std::string label_;

    /// Chemical element symbol.
    std::string symbol_;

    /// Chemical element name.
    std::string name_;

    /// Nucleus charge or pseudocharge, treated as positive(!) integer.
    int zn_{0};

    /// Atom mass.
    double mass_{0};

    /// List of atomic levels.
    std::vector<atomic_level_descriptor> atomic_levels_;

    /// Number of core electrons.
    double num_core_electrons_{0};

    /// Number of valence electrons.
    double num_valence_electrons_{0};

    /// Default augmented wave configuration.
    radial_solution_descriptor_set aw_default_l_;

    /// Augmented wave configuration for specific l.
    std::vector<radial_solution_descriptor_set> aw_specific_l_;

    /// List of radial descriptor sets used to construct augmented waves.
    std::vector<radial_solution_descriptor_set> aw_descriptors_;

    /// List of radial descriptor sets used to construct local orbitals.
    std::vector<local_orbital_descriptor> lo_descriptors_;

    /// Maximum number of AW radial functions across angular momentums.
    int max_aw_order_{0};

    int offset_lo_{-1}; // TODO: better name

    /// Index of radial basis functions.
    radial_functions_index indexr_;

    /// Index of atomic basis functions (radial function * spherical harmonic).
    basis_functions_index indexb_;

    /// Radial functions of beta-projectors.
    std::vector<std::pair<int, Spline<double>>> beta_radial_functions_;

    /// Radial functions of the Q-operator.
    /** The dimension of this array is fully determined by the number and lmax of beta-projectors.
        Beta-projectors must be loaded before loading the Q radial functions. */
    mdarray<Spline<double>, 2> q_radial_functions_l_;

    /// Atomic wave-functions used to setup the initial subspace and to apply U-correction.
    /** This are the chi wave-function in the USPP file. Pairs of [l, chi_l(r)] are stored. */
    std::vector<std::pair<int, Spline<double>>> ps_atomic_wfs_;

    /// Total occupancy of the (hubbard) wave functions.
    std::vector<double> ps_atomic_wf_occ_;

    /// True if the pseudopotential is soft and charge augmentation is required.
    bool augment_{false};

    /// Local part of pseudopotential.
    std::vector<double> local_potential_;

    /// Pseudo-core charge density (used by PP-PW method in non-linear core correction).
    std::vector<double> ps_core_charge_density_;

    /// Total pseudo-charge density (used by PP-PW method to setup initial density).
    std::vector<double> ps_total_charge_density_;

    /// Ionic part of D-operator matrix.
    mdarray<double, 2> d_mtrx_ion_;

    /// True if the pseudopotential is used for PAW.
    bool is_paw_{false};

    /// Core energy of PAW.
    bool paw_core_energy_{0};

    /// All electron wave functions of the PAW method.
    /** The number of wave functions is equal to the number of beta-projectors. */
    mdarray<double, 2> paw_ae_wfs_;

    /// Pseudo wave functions of the PAW method.
    /** The number of wave functions is equal to the number of beta-projectors. */
    mdarray<double, 2> paw_ps_wfs_;

    /// Occupations of PAW wave-functions.
    /** Length of vector is the same as the number of beta projectors, paw_ae_wfs and paw_ps_wfs */
    std::vector<double> paw_wf_occ_;

    /// Core electron contribution to all electron charge density in PAW method.
    std::vector<double> paw_ae_core_charge_density_;

    /// True if the pseudo potential includes spin orbit coupling.
    bool spin_orbit_coupling_{false};

    /// starting magnetization // TODO: remove that
    double starting_magnetization_{0.0};

    // direction of the starting magnetization // TODO: remove that
    double starting_magnetization_theta_{0.0};
    double starting_magnetization_phi_{0.0};

    /// Hubbard correction
    bool hubbard_correction_{false};

    /// hubbard angular momentum s, p, d, f
    int hubbard_l_{-1};

    /// hubbard orbital
    int hubbard_n_{0};

    // hubbard occupancy
    double hubbard_occupancy_orbital_;

    /// different hubbard coefficients
    //  s: U = hubbard_coefficients_[0]
    //  p: U = hubbard_coefficients_[0], J = hubbard_coefficients_[1]
    //  d: U = hubbard_coefficients_[0], J = hubbard_coefficients_[1],  B  = hubbard_coefficients_[2]
    //  f: U = hubbard_coefficients_[0], J = hubbard_coefficients_[1],  E2 = hubbard_coefficients_[2], E3 = hubbard_coefficients_[3]
    ///   hubbard_coefficients[4] = U_alpha
    ///   hubbard_coefficients[5] = U_beta

    double hubbard_J_{0.0};
    double hubbard_U_{0.0};
    double hubbard_coefficients_[4];

    mdarray<double, 4> hubbard_matrix_;

    // simplifed hubbard theory
    double hubbard_alpha_{0.0};
    double hubbard_beta_{0.0};
    double hubbard_J0_{0.0};

    /// Inverse of (Q_{\xi \xi'j}^{-1} + beta_pw^{H}_{\xi} * beta_pw_{xi'})
    /** Used in Chebyshev iterative solver as a block-diagonal preconditioner */
    matrix<double_complex> p_mtrx_;

    /// f_coefficients defined in Ref. PRB 71 115106 Eq.9 only
    /// valid when SO interactions are on
    mdarray<double_complex, 4> f_coefficients_;

    /// List of atom indices (global) for a given type.
    std::vector<int> atom_id_;

    std::string file_name_;

    mdarray<int, 2> idx_radial_integrals_;

    mutable mdarray<double, 3> rf_coef_;
    mutable mdarray<double, 3> vrf_coef_;

    bool initialized_{false};

    inline void read_hubbard_parameters(json const& parser);

    inline void read_input_core(json const& parser);

    inline void read_input_aw(json const& parser);

    inline void read_input_lo(json const& parser);

    inline void read_pseudo_uspp(json const& parser);

    inline void read_pseudo_paw(json const& parser);

    inline void read_input(const std::string& fname);

    inline void init_aw_descriptors(int lmax)
    {
        assert(lmax >= -1);

        if (lmax >= 0 && aw_default_l_.size() == 0) {
            TERMINATE("default AW descriptor is empty");
        }

        aw_descriptors_.clear();
        for (int l = 0; l <= lmax; l++) {
            aw_descriptors_.push_back(aw_default_l_);
            for (size_t ord = 0; ord < aw_descriptors_[l].size(); ord++) {
                aw_descriptors_[l][ord].n = l + 1;
                aw_descriptors_[l][ord].l = l;
            }
        }

        for (size_t i = 0; i < aw_specific_l_.size(); i++) {
            int l = aw_specific_l_[i][0].l;
            if (l < lmax) {
                aw_descriptors_[l] = aw_specific_l_[i];
            }
        }
    }

    /* forbid copy constructor */
    Atom_type(const Atom_type& src) = delete;

    /* forbid assignment operator */
    Atom_type& operator=(const Atom_type& src) = delete;

  protected:
    /// Radial grid of the muffin-tin sphere.
    Radial_grid<double> radial_grid_;

    /// Density of a free atom.
    Spline<double> free_atom_density_spline_;

    std::vector<double> free_atom_density_;

    /// Radial grid of a free atom.
    Radial_grid<double> free_atom_radial_grid_;

  public:
    Atom_type(Simulation_parameters const& parameters__,
              std::string symbol__,
              std::string name__,
              int zn__,
              double mass__,
              std::vector<atomic_level_descriptor>& levels__,
              radial_grid_t grid_type__)
        : parameters_(parameters__)
        , symbol_(symbol__)
        , name_(name__)
        , zn_(zn__)
        , mass_(mass__)
        , atomic_levels_(levels__)
    {
        radial_grid_ = Radial_grid_factory<double>(grid_type__, 2000 + zn__ * 50, 1e-6 / zn_, 20.0 + 0.25 * zn_);
    }

    Atom_type(Simulation_parameters const& parameters__, int id__, std::string label__, std::string file_name__)
        : parameters_(parameters__)
        , id_(id__)
        , label_(label__)
        , file_name_(file_name__)
    {
    }

    Atom_type(Atom_type&& src) = default;

    inline void init(int offset_lo__);

    inline void set_radial_grid(radial_grid_t grid_type__, int num_points__, double rmin__, double rmax__)
    {
        radial_grid_        = Radial_grid_factory<double>(grid_type__, num_points__, rmin__, rmax__);
        if (parameters_.processing_unit() == GPU) {
            radial_grid_.copy_to_device();
        }
    }

    inline void set_radial_grid(int num_points__, double const* points__)
    {
        radial_grid_        = Radial_grid_ext<double>(num_points__, points__);
        if (parameters_.processing_unit() == GPU) {
            radial_grid_.copy_to_device();
        }
    }

    /// Add augmented-wave descriptor.
    inline void add_aw_descriptor(int n, int l, double enu, int dme, int auto_enu)
    {
        if ((int)aw_descriptors_.size() < (l + 1)) {
            aw_descriptors_.resize(l + 1, radial_solution_descriptor_set());
        }

        radial_solution_descriptor rsd;

        rsd.n = n;
        if (n == -1) {
            /* default principal quantum number value for any l */
            rsd.n = l + 1;
            for (int ist = 0; ist < num_atomic_levels(); ist++) {
                /* take next level after the core */
                if (atomic_level(ist).core && atomic_level(ist).l == l) {
                    rsd.n = atomic_level(ist).n + 1;
                }
            }
        }

        rsd.l        = l;
        rsd.dme      = dme;
        rsd.enu      = enu;
        rsd.auto_enu = auto_enu;
        aw_descriptors_[l].push_back(rsd);
    }

    /// Add local orbital descriptor
    inline void add_lo_descriptor(int ilo, int n, int l, double enu, int dme, int auto_enu)
    {
        if ((int)lo_descriptors_.size() == ilo) {
            lo_descriptors_.push_back(local_orbital_descriptor());
            lo_descriptors_[ilo].l = l;
        } else {
            if (l != lo_descriptors_[ilo].l) {
                std::stringstream s;
                s << "wrong angular quantum number" << std::endl
                  << "atom type id: " << id() << " (" << symbol_ << ")" << std::endl
                  << "idxlo: " << ilo << std::endl
                  << "n: " << l << std::endl
                  << "l: " << n << std::endl
                  << "expected l: " << lo_descriptors_[ilo].l << std::endl;
                TERMINATE(s);
            }
        }

        radial_solution_descriptor rsd;

        rsd.n = n;
        if (n == -1) {
            /* default value for any l */
            rsd.n = l + 1;
            for (int ist = 0; ist < num_atomic_levels(); ist++) {
                if (atomic_level(ist).core && atomic_level(ist).l == l) {
                    /* take next level after the core */
                    rsd.n = atomic_level(ist).n + 1;
                }
            }
        }

        rsd.l        = l;
        rsd.dme      = dme;
        rsd.enu      = enu;
        rsd.auto_enu = auto_enu;
        lo_descriptors_[ilo].rsd_set.push_back(rsd);
    }

    inline void add_lo_descriptor(local_orbital_descriptor const& lod__)
    {
        lo_descriptors_.push_back(lod__);
    }

    inline void add_ps_atomic_wf(int l__, std::vector<double> f__)
    {
        Spline<double> s(radial_grid_, f__);
        ps_atomic_wfs_.push_back(std::move(std::make_pair(l__, std::move(s))));
    }

    inline void add_ps_atomic_wf(int l__, std::vector<double> f__, double occ_)
    {
        Spline<double> s(radial_grid_, f__);
        ps_atomic_wfs_.push_back(std::move(std::make_pair(l__, std::move(s))));
        ps_atomic_wf_occ_.push_back(occ_);
    }

    std::pair<int, Spline<double>> const& ps_atomic_wf(int idx__) const
    {
        return ps_atomic_wfs_[idx__];
    }

    inline int lmax_ps_atomic_wf() const
    {
        int lmax{-1};
        for (auto& e: ps_atomic_wfs_) {
            lmax = std::max(lmax, std::abs(e.first));
        }
        return lmax;
    }

    inline int num_ps_atomic_wf() const
    {
        return static_cast<int>(ps_atomic_wfs_.size());
    }

    inline std::vector<double> const& ps_atomic_wf_occ() const
    {
        return ps_atomic_wf_occ_;
    }

    inline std::vector<double>& ps_atomic_wf_occ(std::vector<double> inp__)
    {
        ps_atomic_wf_occ_.clear();
        ps_atomic_wf_occ_ = inp__;
        return ps_atomic_wf_occ_;
    }

    /// Add a radial function of beta-projector to a list of functions.
    inline void add_beta_radial_function(int l__, std::vector<double> beta__)
    {
        if (augment_) {
            TERMINATE("can't add more beta projectors");
        }
        Spline<double> s(radial_grid_, beta__);
        beta_radial_functions_.push_back(std::move(std::make_pair(l__, std::move(s))));
    }

    /// Return a radial beta functions.
    inline Spline<double> const& beta_radial_function(int idxrf__) const
    {
        return beta_radial_functions_[idxrf__].second;
    }

    /// Maximum orbital quantum number between all beta-projector radial functions.
    inline int lmax_beta() const
    {
        int lmax{-1};

        // need to take |l| since the total angular momentum is encoded
        // in the sign of l
        for (auto& e: beta_radial_functions_) {
            lmax = std::max(lmax, std::abs(e.first));
        }
        return lmax;
    }

    /// Number of beta-radial functions.
    inline int num_beta_radial_functions() const
    {
        return static_cast<int>(beta_radial_functions_.size());
    }

    inline void add_q_radial_function(int idxrf1__, int idxrf2__, int l__, std::vector<double> qrf__)
    {
        if (l__ > 2 * lmax_beta()) {
            TERMINATE("wrong l for Q radial functions");
        }

        if (!augment_) {
            augment_ = true;
            /* number of radial beta-functions */
            int nbrf = num_beta_radial_functions();
            q_radial_functions_l_ = mdarray<Spline<double>, 2>(nbrf * (nbrf + 1) / 2, 2 * lmax_beta() + 1);

            for (int l = 0; l <= 2 * lmax_beta(); l++) {
                for (int idx = 0; idx < nbrf * (nbrf + 1) / 2; idx++) {
                    q_radial_functions_l_(idx, l) = Spline<double>(radial_grid_);
                }
            }
        }

        /* pack Q-radial functions in a triangular matrix (Q_{ij} matrix is symmetric):
               j
           +-------+
           | +     |
          i|   +   |   -> idx = j * (j + 1) / 2 + i  for  i <= j
           |     + |
           +-------+

           i, j are the indices of radial beta-functions
         */

        /* combined index */
        if (idxrf1__ > idxrf2__) {
            std::swap(idxrf1__, idxrf2__);
        }
        int ijv = idxrf2__ * (idxrf2__ + 1) / 2 + idxrf1__;
        q_radial_functions_l_(ijv, l__) = Spline<double>(radial_grid_, qrf__);
    }

    inline bool augment() const
    {
        return augment_;
    }

    inline std::vector<double>& local_potential(std::vector<double> vloc__)
    {
        local_potential_ = vloc__;
        return local_potential_;
    }

    inline std::vector<double> const& local_potential() const
    {
        return local_potential_;
    }

    inline std::vector<double>& ps_core_charge_density(std::vector<double> ps_core__)
    {
        ps_core_charge_density_ = ps_core__;
        return ps_core_charge_density_;
    }

    inline std::vector<double> const& ps_core_charge_density() const
    {
        return ps_core_charge_density_;
    }

    inline std::vector<double>& ps_total_charge_density(std::vector<double> ps_dens__)
    {
        ps_total_charge_density_ = ps_dens__;
        return ps_total_charge_density_;
    }

    inline std::vector<double> const& ps_total_charge_density() const
    {
        return ps_total_charge_density_;
    }

    inline mdarray<double, 2> const& paw_ae_wfs() const
    {
        return paw_ae_wfs_;
    }

    inline void paw_ae_wfs(mdarray<double, 2>& inp__)
    {
        return inp__ >> paw_ae_wfs_;
    }

    inline mdarray<double, 2> const& paw_ps_wfs() const
    {
        return paw_ps_wfs_;
    }

    inline void paw_ps_wfs(mdarray<double, 2>& inp__)
    {
        return inp__ >> paw_ps_wfs_;
    }

    inline std::vector<double> const& paw_ae_core_charge_density() const
    {
        return paw_ae_core_charge_density_;
    }

    inline std::vector<double>& paw_ae_core_charge_density(std::vector<double> inp__)
    {
        paw_ae_core_charge_density_ = inp__;
        return paw_ae_core_charge_density_;
    }

    inline std::vector<double> const& paw_wf_occ() const
    {
        return paw_wf_occ_;
    }

    inline std::vector<double>& paw_wf_occ(std::vector<double> inp__)
    {
        paw_wf_occ_ = inp__;
        return paw_wf_occ_;
    }

    inline void init_free_atom(bool smooth);

    inline void print_info() const;

    inline int id() const
    {
        return id_;
    }

    inline int zn() const
    {
        assert(zn_ > 0);
        return zn_;
    }

    inline int zn(int zn__)
    {
        zn_ = zn__;
        return zn_;
    }

    inline std::string const& symbol() const
    {
        return symbol_;
    }

    inline std::string const& name() const
    {
        return name_;
    }

    inline double mass() const
    {
        return mass_;
    }

    /// Return muffin-tin radius.
    /** This is the last point of the radial grid. */
    inline double mt_radius() const
    {
        return radial_grid_.last();
    }

    /// Return number of muffin-tin radial grid points.
    inline int num_mt_points() const
    {
        assert(radial_grid_.num_points() > 0);
        return radial_grid_.num_points();
    }

    inline Radial_grid<double> const& radial_grid() const
    {
        assert(radial_grid_.num_points() > 0);
        return radial_grid_;
    }

    inline Radial_grid<double> const& free_atom_radial_grid() const
    {
        return free_atom_radial_grid_;
    }

    inline double radial_grid(int ir) const
    {
        return radial_grid_[ir];
    }

    inline double free_atom_radial_grid(int ir) const
    {
        return free_atom_radial_grid_[ir];
    }

    inline int num_atomic_levels() const
    {
        return static_cast<int>(atomic_levels_.size());
    }

    inline atomic_level_descriptor const& atomic_level(int idx) const
    {
        return atomic_levels_[idx];
    }

    inline double num_core_electrons() const
    {
        return num_core_electrons_;
    }

    inline double num_valence_electrons() const
    {
        return num_valence_electrons_;
    }

    inline double free_atom_density(const int idx) const
    {
        return free_atom_density_spline_(idx);
    }

    inline double free_atom_density(double x) const
    {
        return free_atom_density_spline_.at_point(x);
    }

    inline int num_aw_descriptors() const
    {
        return static_cast<int>(aw_descriptors_.size());
    }

    inline radial_solution_descriptor_set const& aw_descriptor(int l) const
    {
        assert(l < (int)aw_descriptors_.size());
        return aw_descriptors_[l];
    }

    inline int num_lo_descriptors() const
    {
        return (int)lo_descriptors_.size();
    }

    inline local_orbital_descriptor const& lo_descriptor(int idx) const
    {
        return lo_descriptors_[idx];
    }

    inline int max_aw_order() const
    {
        return max_aw_order_;
    }

    /// Order of augmented wave radial functions for a given l.
    inline int aw_order(int l__) const
    {
        return static_cast<int>(aw_descriptor(l__).size());
    }

    inline radial_functions_index const& indexr() const
    {
        return indexr_;
    }

    inline radial_function_index_descriptor const& indexr(int i) const
    {
        assert(i >= 0 && i < (int)indexr_.size());
        return indexr_[i];
    }

    inline int indexr_by_l_order(int l, int order) const
    {
        return indexr_.index_by_l_order(l, order);
    }

    inline int indexr_by_idxlo(int idxlo) const
    {
        return indexr_.index_by_idxlo(idxlo);
    }

    inline basis_functions_index const& indexb() const
    {
        return indexb_;
    }

    inline basis_function_index_descriptor const& indexb(int i) const
    {
        assert(i >= 0 && i < (int)indexb_.size());
        return indexb_[i];
    }

    inline int indexb_by_l_m_order(int l, int m, int order) const
    {
        return indexb_.index_by_l_m_order(l, m, order);
    }

    inline int indexb_by_lm_order(int lm, int order) const
    {
        return indexb_.index_by_lm_order(lm, order);
    }

    inline int mt_aw_basis_size() const
    {
        return indexb_.size_aw();
    }

    inline int mt_lo_basis_size() const
    {
        return indexb_.size_lo();
    }

    inline int mt_basis_size() const
    {
        return indexb_.size();
    }

    /// Total number of radial basis functions.
    inline int mt_radial_basis_size() const
    {
        return indexr_.size();
    }

    /// Return index of a free atom grid point close to the muffin-tin radius.
    inline int idx_rmt_free_atom() const
    {
        for (int i = 0; i < free_atom_radial_grid().num_points(); i++) {
            if (free_atom_radial_grid(i) > mt_radius())
                return i - 1;
        }
        return -1;
    }

    inline void set_symbol(const std::string symbol__)
    {
        symbol_ = symbol__;
    }

    inline void set_zn(int zn__)
    {
        zn_ = zn__;
    }

    inline void set_mass(double mass__)
    {
        mass_ = mass__;
    }

    inline void set_configuration(int n, int l, int k, double occupancy, bool core)
    {
        atomic_level_descriptor level;
        level.n         = n;
        level.l         = l;
        level.k         = k;
        level.occupancy = occupancy;
        level.core      = core;
        atomic_levels_.push_back(level);
    }

    /// Return number of atoms of a given type.
    inline int num_atoms() const
    {
        return static_cast<int>(atom_id_.size());
    }

    inline int atom_id(int idx) const
    {
        return atom_id_[idx];
    }

    /// Add global index of atom to this atom type.
    inline void add_atom_id(int atom_id__)
    {
        atom_id_.push_back(atom_id__);
    }

    inline bool initialized() const
    {
        return initialized_;
    }

    inline std::string const& label() const
    {
        return label_;
    }

    inline std::string const& file_name() const
    {
        return file_name_;
    }

    inline int offset_lo() const
    {
        assert(offset_lo_ >= 0);
        return offset_lo_;
    }

    inline void d_mtrx_ion(matrix<double> const& d_mtrx_ion__)
    {
        d_mtrx_ion_ = matrix<double>(num_beta_radial_functions(), num_beta_radial_functions());
        d_mtrx_ion__ >> d_mtrx_ion_;
    }

    inline mdarray<double, 2> const& d_mtrx_ion() const
    {
        return d_mtrx_ion_;
    }

    inline bool is_paw() const
    {
        return is_paw_;
    }

    inline bool is_paw(bool is_paw__)
    {
        is_paw_ = is_paw__;
        return is_paw_;
    }

    double paw_core_energy() const
    {
        return paw_core_energy_;
    }

    double paw_core_energy(double paw_core_energy__)
    {
        paw_core_energy_ = paw_core_energy__;
        return paw_core_energy_;
    }

    inline mdarray<int, 2> const& idx_radial_integrals() const
    {
        return idx_radial_integrals_;
    }

    inline mdarray<double, 3>& rf_coef() const
    {
        return rf_coef_;
    }

    inline mdarray<double, 3>& vrf_coef() const
    {
        return vrf_coef_;
    }

    inline Simulation_parameters const& parameters() const
    {
        return parameters_;
    }

    inline void set_free_atom_radial_grid(int num_points__, double const* points__)
    {
        if (num_points__ <= 0) {
            TERMINATE("wrong number of radial points");
        }
        free_atom_radial_grid_ = Radial_grid_ext<double>(num_points__, points__);
    }

    inline void set_free_atom_density(int num_points__, double const* dens__)
    {
        free_atom_density_.resize(num_points__);
        std::memcpy(free_atom_density_.data(), dens__, num_points__ * sizeof(double));
    }

    inline double_complex f_coefficients(int xi1, int xi2, int s1, int s2) const
    {
        return f_coefficients_(xi1, xi2, s1, s2);
    }

    inline Spline<double> const& q_radial_function(int idxrf1__, int idxrf2__, int l__) const
    {
        if (idxrf1__ > idxrf2__) {
            std::swap(idxrf1__, idxrf2__);
        }
        /* combined index */
        int ijv = idxrf2__ * (idxrf2__ + 1) / 2 + idxrf1__;

        return q_radial_functions_l_(ijv, l__);
    }

    inline bool spin_orbit_coupling() const
    {
        return spin_orbit_coupling_;
    }

    inline bool spin_orbit_coupling(bool so__)
    {
        spin_orbit_coupling_ = so__;
        return spin_orbit_coupling_;
    }

    bool const& hubbard_correction() const
    {
        return hubbard_correction_;
    }

    inline double Hubbard_J0() const
    {
        return hubbard_J0_;
    }

    inline double Hubbard_U() const
    {
        return hubbard_U_;
    }

    inline double Hubbard_J() const
    {
        return hubbard_J_;
    }

    inline void set_hubbard_U(double U__)
    {
        hubbard_U_ = U__;
    }

    inline void set_hubbard_J(double J__)
    {
        hubbard_J_ = J__;
    }

    inline double Hubbard_U_minus_J() const
    {
        return Hubbard_U() - Hubbard_J();
    }

    inline double Hubbard_B() const
    {
        return hubbard_coefficients_[2];
    }

    inline double Hubbard_E2() const
    {
        return hubbard_coefficients_[2];
    }

    inline double Hubbard_E3() const
    {
        return hubbard_coefficients_[3];
    }

    inline double Hubbard_alpha() const
    {
        return hubbard_alpha_;
    }

    inline double Hubbard_beta() const
    {
        return hubbard_beta_;
    }

    inline void set_hubbard_alpha(const double alpha)
    {
        hubbard_alpha_ = alpha;
    }

    inline void set_hubbard_beta(const double beta_)
    {
        hubbard_beta_ = beta_;
    }

    inline void set_starting_magnetization_theta(const double theta_)
    {
        starting_magnetization_theta_ = theta_;
    }

    inline void set_starting_magnetization_phi(const double phi_)
    {
        starting_magnetization_phi_ = phi_;;
    }

    inline void set_hubbard_coefficients(double const * J_)
    {
        this->hubbard_coefficients_[0] = J_[0];
        this->hubbard_coefficients_[1] = J_[1];
        this->hubbard_coefficients_[2] = J_[2];
        this->hubbard_coefficients_[3] = J_[3];
    }

    inline void set_hubbard_J0(double const J0_)
    {
        this->hubbard_J0_ = J0_;
    }

    inline void set_hubbard_correction(const int co_)
    {
        if(co_ > 0)
            this->hubbard_correction_ = true;
        else
            this->hubbard_correction_ = false;
    }

    inline int hubbard_l() const
    {
        return hubbard_l_;
    }

    inline double starting_magnetization_theta() const
    {
        return starting_magnetization_theta_;
    }

    inline double starting_magnetization_phi() const
    {
        return starting_magnetization_phi_;
    }

    inline double starting_magnetization() const
    {
        return starting_magnetization_;
    }

    inline int hubbard_n() const
    {
        return hubbard_n_;
    }

    inline double hubbard_matrix(const int m1, const int m2, const int m3, const int m4) const
    {
        return hubbard_matrix_(m1, m2, m3, m4);
    }

    inline double& hubbard_matrix(const int m1, const int m2, const int m3, const int m4)
    {
        return hubbard_matrix_(m1, m2, m3, m4);
    }

    void hubbard_F_coefficients(double *F)
    {
        F[0] = Hubbard_U();

        switch(hubbard_l_) {
        case 0:
            F[1] = Hubbard_J();
        case 1:
            F[1] = 5.0 * Hubbard_J();
            break;
        case 2:
            F[1] = 5.0 * Hubbard_J() + 31.5 * Hubbard_B();
            F[2] = 9.0 * Hubbard_J() - 31.5 * Hubbard_B();
            break;
        case 3:
            F[1] = (225.0 / 54.0) * Hubbard_J()     + (32175.0 / 42.0) * Hubbard_E2()   + (2475.0 / 42.0) * Hubbard_E3();
            F[2] = 11.0 * Hubbard_J()            - (141570.0 / 77.0) * Hubbard_E2()  + (4356.0 / 77.0) * Hubbard_E3();
            F[3] = (7361.640 / 594.0) * Hubbard_J() + (36808.20 / 66.0) * Hubbard_E2()  - 111.54 * Hubbard_E3();
            break;
        default:
            printf("Hubbard lmax %d\n", hubbard_l_);
            TERMINATE("Hubbard correction not implemented for l > 3");
            break;
        }
    }

    /// compare the angular, total angular momentum and radial part of
    /// the beta projectors, leaving the m index free. Only useful
    /// when spin orbit coupling is included.
    inline bool compare_index_beta_functions(const int xi, const int xj) const
    {
        return ((indexb(xi).l == indexb(xj).l) && (indexb(xi).idxrf == indexb(xj).idxrf) &&
                (std::abs(indexb(xi).j - indexb(xj).j) < 1e-8));
    }

    inline void set_hubbard_occupancy_orbital(double occ)
    {
        if ((occ < 0) && (hubbard_correction_))
            TERMINATE("this atom has hubbard correction but the orbital occupancy is negative\n");
        hubbard_occupancy_orbital_ = occ;
    }

    inline void set_hubbard_l_and_n_orbital()
    {
        std::vector<std::pair<std::string, int>> nl_orb; // TODO: is there a way to get rid of this harcoded values?

        nl_orb.push_back(std::make_pair("Ti", 3));
        nl_orb.push_back(std::make_pair("V", 3));
        nl_orb.push_back(std::make_pair("Cr", 3));
        nl_orb.push_back(std::make_pair("Mn", 3));
        nl_orb.push_back(std::make_pair("Fe", 3));
        nl_orb.push_back(std::make_pair("Co", 3));
        nl_orb.push_back(std::make_pair("Ni", 3));
        nl_orb.push_back(std::make_pair("Cu", 3));
        nl_orb.push_back(std::make_pair("Zn", 3));
        nl_orb.push_back(std::make_pair("As", 3));
        nl_orb.push_back(std::make_pair("Ga", 3));

        nl_orb.push_back(std::make_pair("Zr", 4));
        nl_orb.push_back(std::make_pair("Nb", 4));
        nl_orb.push_back(std::make_pair("Mo", 4));
        nl_orb.push_back(std::make_pair("Tc", 4));
        nl_orb.push_back(std::make_pair("Ru", 4));
        nl_orb.push_back(std::make_pair("Rh", 4));
        nl_orb.push_back(std::make_pair("Pd", 4));
        nl_orb.push_back(std::make_pair("Ag", 4));
        nl_orb.push_back(std::make_pair("Cd", 4));
        nl_orb.push_back(std::make_pair("Ce", 4));
        nl_orb.push_back(std::make_pair("Pr", 4));
        nl_orb.push_back(std::make_pair("Nd", 4));
        nl_orb.push_back(std::make_pair("Pm", 4));
        nl_orb.push_back(std::make_pair("Sm", 4));
        nl_orb.push_back(std::make_pair("Eu", 4));
        nl_orb.push_back(std::make_pair("Gd", 4));
        nl_orb.push_back(std::make_pair("Tb", 4));
        nl_orb.push_back(std::make_pair("Dy", 4));
        nl_orb.push_back(std::make_pair("Ho", 4));
        nl_orb.push_back(std::make_pair("Er", 4));
        nl_orb.push_back(std::make_pair("Tm", 4));
        nl_orb.push_back(std::make_pair("Yb", 4));
        nl_orb.push_back(std::make_pair("Lu", 4));
        nl_orb.push_back(std::make_pair("In", 4));

        nl_orb.push_back(std::make_pair("Th", 5));
        nl_orb.push_back(std::make_pair("Pa", 5));
        nl_orb.push_back(std::make_pair("U", 5));
        nl_orb.push_back(std::make_pair("Np", 5));
        nl_orb.push_back(std::make_pair("Pu", 5));
        nl_orb.push_back(std::make_pair("Am", 5));
        nl_orb.push_back(std::make_pair("Cm", 5));
        nl_orb.push_back(std::make_pair("Bk", 5));
        nl_orb.push_back(std::make_pair("Cf", 5));
        nl_orb.push_back(std::make_pair("Es", 5));
        nl_orb.push_back(std::make_pair("Fm", 5));
        nl_orb.push_back(std::make_pair("Md", 5));
        nl_orb.push_back(std::make_pair("No", 5));
        nl_orb.push_back(std::make_pair("Lr", 5));
        nl_orb.push_back(std::make_pair("Hf", 5));
        nl_orb.push_back(std::make_pair("Ta", 5));
        nl_orb.push_back(std::make_pair("W", 5));
        nl_orb.push_back(std::make_pair("Re", 5));
        nl_orb.push_back(std::make_pair("Os", 5));
        nl_orb.push_back(std::make_pair("Ir", 5));
        nl_orb.push_back(std::make_pair("Pt", 5));
        nl_orb.push_back(std::make_pair("Au", 5));
        nl_orb.push_back(std::make_pair("Hg", 5));

        nl_orb.push_back(std::make_pair("H", 1));
        nl_orb.push_back(std::make_pair("He", 1));

        nl_orb.push_back(std::make_pair("C", 2));
        nl_orb.push_back(std::make_pair("N", 2));
        nl_orb.push_back(std::make_pair("O", 2));

        for(size_t i = 0; i < nl_orb.size(); i++) {
            if(nl_orb[i].first == label_) {
                hubbard_n_ = nl_orb[i].second;
                break;
            }
        }

        if (hubbard_n_ < 0) {
            TERMINATE("The atom %s is not included in the list of atoms with hubbard correction\n");
        }

        // same with orbital momentum
        nl_orb.clear();
        // d orbitals
        nl_orb.push_back(std::make_pair("Ti", 2));
        nl_orb.push_back(std::make_pair("V", 2));
        nl_orb.push_back(std::make_pair("Cr", 2));
        nl_orb.push_back(std::make_pair("Mn", 2));
        nl_orb.push_back(std::make_pair("Fe", 2));
        nl_orb.push_back(std::make_pair("Co", 2));
        nl_orb.push_back(std::make_pair("Ni", 2));
        nl_orb.push_back(std::make_pair("Cu", 2));
        nl_orb.push_back(std::make_pair("Zn", 2));
        nl_orb.push_back(std::make_pair("Zr", 2));
        nl_orb.push_back(std::make_pair("Nb", 2));
        nl_orb.push_back(std::make_pair("Mo", 2));
        nl_orb.push_back(std::make_pair("Tc", 2));
        nl_orb.push_back(std::make_pair("Ru", 2));
        nl_orb.push_back(std::make_pair("Rh", 2));
        nl_orb.push_back(std::make_pair("Pd", 2));
        nl_orb.push_back(std::make_pair("Ag", 2));
        nl_orb.push_back(std::make_pair("Cd", 2));
        nl_orb.push_back(std::make_pair("Hf", 2));
        nl_orb.push_back(std::make_pair("Ta", 2));
        nl_orb.push_back(std::make_pair("W", 2));
        nl_orb.push_back(std::make_pair("Re", 2));
        nl_orb.push_back(std::make_pair("Os", 2));
        nl_orb.push_back(std::make_pair("Ir", 2));
        nl_orb.push_back(std::make_pair("Pt", 2));
        nl_orb.push_back(std::make_pair("Au", 2));
        nl_orb.push_back(std::make_pair("Hg", 2));
        nl_orb.push_back(std::make_pair("As", 2));
        nl_orb.push_back(std::make_pair("Ga", 2));
        nl_orb.push_back(std::make_pair("In", 2));

        // f orbitals
        nl_orb.push_back(std::make_pair("Ce", 3));
        nl_orb.push_back(std::make_pair("Pr", 3));
        nl_orb.push_back(std::make_pair("Nd", 3));
        nl_orb.push_back(std::make_pair("Pm", 3));
        nl_orb.push_back(std::make_pair("Sm", 3));
        nl_orb.push_back(std::make_pair("Eu", 3));
        nl_orb.push_back(std::make_pair("Gd", 3));
        nl_orb.push_back(std::make_pair("Tb", 3));
        nl_orb.push_back(std::make_pair("Dy", 3));
        nl_orb.push_back(std::make_pair("Ho", 3));
        nl_orb.push_back(std::make_pair("Er", 3));
        nl_orb.push_back(std::make_pair("Tm", 3));
        nl_orb.push_back(std::make_pair("Yb", 3));
        nl_orb.push_back(std::make_pair("Lu", 3));
        nl_orb.push_back(std::make_pair("Th", 3));
        nl_orb.push_back(std::make_pair("Pa", 3));
        nl_orb.push_back(std::make_pair("U", 3));
        nl_orb.push_back(std::make_pair("Np", 3));
        nl_orb.push_back(std::make_pair("Pu", 3));
        nl_orb.push_back(std::make_pair("Am", 3));
        nl_orb.push_back(std::make_pair("Cm", 3));
        nl_orb.push_back(std::make_pair("Bk", 3));
        nl_orb.push_back(std::make_pair("Cf", 3));
        nl_orb.push_back(std::make_pair("Es", 3));
        nl_orb.push_back(std::make_pair("Fm", 3));
        nl_orb.push_back(std::make_pair("Md", 3));
        nl_orb.push_back(std::make_pair("No", 3));
        nl_orb.push_back(std::make_pair("Lr", 3));

        // s orbitals
        nl_orb.push_back(std::make_pair("H", 0));
        nl_orb.push_back(std::make_pair("He", 0));

        // p orbitals
        nl_orb.push_back(std::make_pair("C", 1));
        nl_orb.push_back(std::make_pair("N", 1));
        nl_orb.push_back(std::make_pair("O", 1));


        for(size_t i = 0; i < nl_orb.size(); i++) {
            if(nl_orb[i].first == label_) {
                hubbard_l_ = nl_orb[i].second;
                break;
            }
        }

        if (hubbard_l_ < 0) {
            printf("Atom %s\n", symbol_.c_str());
            TERMINATE("The atom is not included in the list of atoms with hubbard correction\n");
        }

    }


    inline double get_occupancy_hubbard_orbital() const
    {
        return hubbard_occupancy_orbital_;
    }

    inline void set_occupancy_hubbard_orbital(double oc)
    {
        if (oc > 0.0) {
            hubbard_occupancy_orbital_ = oc;
            return;
        }

        std::vector<std::pair<std::string, double>> occ;
        occ.clear();

        occ.push_back(std::make_pair("He", 2.0));

        // transition metals
        occ.push_back(std::make_pair("Ti", 2.0));
        occ.push_back(std::make_pair("Zr", 2.0));
        occ.push_back(std::make_pair("Hf", 2.0));
        occ.push_back(std::make_pair("V", 3.0));
        occ.push_back(std::make_pair("Nb", 3.0));
        occ.push_back(std::make_pair("Ta", 3.0));
        occ.push_back(std::make_pair("Cr", 5.0));
        occ.push_back(std::make_pair("Mo", 5.0));
        occ.push_back(std::make_pair("W", 5.0));
        occ.push_back(std::make_pair("Mn", 5.0));
        occ.push_back(std::make_pair("Tc", 5.0));
        occ.push_back(std::make_pair("re", 5.0));
        occ.push_back(std::make_pair("Fe", 6.0));
        occ.push_back(std::make_pair("Ru", 6.0));
        occ.push_back(std::make_pair("Os", 6.0));
        occ.push_back(std::make_pair("Co", 7.0));
        occ.push_back(std::make_pair("Rh", 7.0));
        occ.push_back(std::make_pair("Ir", 7.0));
        occ.push_back(std::make_pair("Ni", 8.0));
        occ.push_back(std::make_pair("Pd", 8.0));
        occ.push_back(std::make_pair("Pt", 8.0));
        occ.push_back(std::make_pair("Cu", 10.0));
        occ.push_back(std::make_pair("Ag", 10.0));
        occ.push_back(std::make_pair("Au", 10.0));
        occ.push_back(std::make_pair("Zn", 10.0));
        occ.push_back(std::make_pair("Cd", 10.0));
        occ.push_back(std::make_pair("Hg", 10.0));
        occ.push_back(std::make_pair("Ce", 2.0));
        occ.push_back(std::make_pair("Th", 2.0));
        occ.push_back(std::make_pair("Pr", 3.0));
        occ.push_back(std::make_pair("Pa", 3.0));
        occ.push_back(std::make_pair("Nd", 4.0));
        occ.push_back(std::make_pair("U", 4.0));
        occ.push_back(std::make_pair("Pm", 5.0));
        occ.push_back(std::make_pair("Np", 5.0));
        occ.push_back(std::make_pair("Sm", 6.0));
        occ.push_back(std::make_pair("Pu", 6.0));
        occ.push_back(std::make_pair("Eu", 6.0));
        occ.push_back(std::make_pair("Am", 6.0));
        occ.push_back(std::make_pair("Gd", 7.0));
        occ.push_back(std::make_pair("Cm", 7.0));
        occ.push_back(std::make_pair("Tb", 8.0));
        occ.push_back(std::make_pair("Bk", 8.0));
        occ.push_back(std::make_pair("Dy", 9.0));
        occ.push_back(std::make_pair("Cf", 9.0));
        occ.push_back(std::make_pair("Ho", 10.0));
        occ.push_back(std::make_pair("Es", 10.0));
        occ.push_back(std::make_pair("Er", 11.0));
        occ.push_back(std::make_pair("Fm", 11.0));
        occ.push_back(std::make_pair("Tm", 12.0));
        occ.push_back(std::make_pair("Md", 12.0));
        occ.push_back(std::make_pair("Yb", 13.0));
        occ.push_back(std::make_pair("No", 13.0));
        occ.push_back(std::make_pair("Lu", 14.0));
        occ.push_back(std::make_pair("Lr", 14.0));
        occ.push_back(std::make_pair("C", 2.0));
        occ.push_back(std::make_pair("N", 3.0));
        occ.push_back(std::make_pair("O", 4.0));
        occ.push_back(std::make_pair("H", 1.0));
        occ.push_back(std::make_pair("Ga", 10.0));
        occ.push_back(std::make_pair("In", 10.0));

        for(size_t i = 0; i < occ.size(); i++) {
            if (occ[i].first == label_) {
                hubbard_occupancy_orbital_  = occ[i].second;
                occ.clear();
                break;
            }
        }

        if (hubbard_occupancy_orbital_ < 0.0) {
            TERMINATE("this atom is not in the list of atoms with hubbard corrections\n");
        }
    }
  private:
    void read_hubbard_input();
    void generate_f_coefficients(void);
    inline double ClebschGordan(const int l, const double j, const double m, const int spin);
    inline double_complex calculate_U_sigma_m(const int l, const double j, const int mj, const int m, const int sigma);
    inline void calculate_ak_coefficients(mdarray<double, 5> &ak);
    inline void compute_hubbard_matrix();
};

inline void Atom_type::init(int offset_lo__)
{
    PROFILE("sirius::Atom_type::init");

    /* check if the class instance was already initialized */
    if (initialized_) {
        TERMINATE("can't initialize twice");
    }

    offset_lo_ = offset_lo__;

    /* read data from file if it exists */
    if (file_name_.length() > 0) {
        if (!Utils::file_exists(file_name_)) {
            std::stringstream s;
            s << "file " + file_name_ + " doesn't exist";
            TERMINATE(s);
        } else {
            read_input(file_name_);
        }
    }

    /* check the nuclear charge */
    if (zn_ == 0) {
        TERMINATE("zero atom charge");
    }

    /* add valence levels to the list of core levels */
    if (parameters_.full_potential()) {
        atomic_level_descriptor level;
        for (int ist = 0; ist < 28; ist++) {
            bool found      = false;
            level.n         = atomic_conf[zn_ - 1][ist][0];
            level.l         = atomic_conf[zn_ - 1][ist][1];
            level.k         = atomic_conf[zn_ - 1][ist][2];
            level.occupancy = double(atomic_conf[zn_ - 1][ist][3]);
            level.core      = false;

            if (level.n != -1) {
                for (size_t jst = 0; jst < atomic_levels_.size(); jst++) {
                    if (atomic_levels_[jst].n == level.n && atomic_levels_[jst].l == level.l &&
                        atomic_levels_[jst].k == level.k) {
                        found = true;
                    }
                }
                if (!found) {
                    atomic_levels_.push_back(level);
                }
            }
        }
        /* get the number of core electrons */
        for (auto& e : atomic_levels_) {
            if (e.core) {
                num_core_electrons_ += e.occupancy;
            }
        }
    }

    if (parameters_.full_potential()) {
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

    if (!parameters_.full_potential()) {
        /* add beta projectors to a list of atom's local orbitals */
        for (auto& e: beta_radial_functions_) {
            /* think of |beta> functions as of local orbitals */
            local_orbital_descriptor lod;
            lod.l = std::abs(e.first);

            // for spin orbit coupling. We can always do that there is
            // no insidence on the reset when calculations exclude SO
            if (e.first < 0) {
                lod.total_angular_momentum = lod.l - 0.5;
            } else {
                lod.total_angular_momentum = lod.l + 0.5;
            }
            lo_descriptors_.push_back(lod);
        }
    }

    /* initialize index of radial functions */
    indexr_.init(aw_descriptors_, lo_descriptors_);

    /* initialize index of muffin-tin basis functions */
    indexb_.init(indexr_);

    if (!parameters_.full_potential()) {
        assert(mt_radial_basis_size() == num_beta_radial_functions());
        assert(lmax_beta() == indexr().lmax());
    }

    /* get number of valence electrons */
    num_valence_electrons_ = zn_ - num_core_electrons_;

    int lmmax_pot = Utils::lmmax(parameters_.lmax_pot());

    if (parameters_.full_potential()) {
        auto l_by_lm = Utils::l_by_lm(parameters_.lmax_pot());

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
        for (size_t j = 0; j < non_zero_elements.size(); j++) {
            idx_radial_integrals_(0, j) = non_zero_elements[j].first;
            idx_radial_integrals_(1, j) = non_zero_elements[j].second;
        }
    }

    if (parameters_.processing_unit() == GPU && parameters_.full_potential()) {
        idx_radial_integrals_.allocate(memory_t::device);
        idx_radial_integrals_.copy<memory_t::host, memory_t::device>();
        rf_coef_  = mdarray<double, 3>(num_mt_points(), 4, indexr().size(), memory_t::host_pinned | memory_t::device,
                                       "Atom_type::rf_coef_");
        vrf_coef_ = mdarray<double, 3>(num_mt_points(), 4, lmmax_pot * indexr().size() * (parameters_.num_mag_dims() + 1),
                                       memory_t::host_pinned | memory_t::device, "Atom_type::vrf_coef_");
    }

    if (this->hubbard_correction_) {
        set_hubbard_l_and_n_orbital();
        set_occupancy_hubbard_orbital(-1.0);
        compute_hubbard_matrix();
    }

    if (this->spin_orbit_coupling()) {
        this->generate_f_coefficients();
    }
    initialized_ = true;
}

inline void Atom_type::init_free_atom(bool smooth)
{
    free_atom_density_spline_ = Spline<double>(free_atom_radial_grid_, free_atom_density_);
    /* smooth free atom density inside the muffin-tin sphere */
    if (smooth) {
        /* find point on the grid close to the muffin-tin radius */
        int irmt = idx_rmt_free_atom();

        mdarray<double, 1> b(2);
        mdarray<double, 2> A(2, 2);
        double R = free_atom_radial_grid_[irmt];
        A(0, 0) = std::pow(R, 2);
        A(0, 1) = std::pow(R, 3);
        A(1, 0) = 2 * R;
        A(1, 1) = 3 * std::pow(R, 2);

        b(0) = free_atom_density_spline_(irmt);
        b(1) = free_atom_density_spline_.deriv(1, irmt);

        linalg<CPU>::gesv<double>(2, 1, A.at<CPU>(), 2, b.at<CPU>(), 2);

        //== /* write initial density */
        //== std::stringstream sstr;
        //== sstr << "free_density_" << id_ << ".dat";
        //== FILE* fout = fopen(sstr.str().c_str(), "w");

        //== for (int ir = 0; ir < free_atom_radial_grid().num_points(); ir++)
        //== {
        //==     fprintf(fout, "%f %f \n", free_atom_radial_grid(ir), free_atom_density_[ir]);
        //== }
        //== fclose(fout);

        /* make smooth free atom density inside muffin-tin */
        for (int i = 0; i <= irmt; i++) {
            free_atom_density_spline_(i) =
                b(0) * std::pow(free_atom_radial_grid(i), 2) + b(1) * std::pow(free_atom_radial_grid(i), 3);
        }

        /* interpolate new smooth density */
        free_atom_density_spline_.interpolate();

        //== /* write smoothed density */
        //== sstr.str("");
        //== sstr << "free_density_modified_" << id_ << ".dat";
        //== fout = fopen(sstr.str().c_str(), "w");

        //== for (int ir = 0; ir < free_atom_radial_grid().num_points(); ir++)
        //== {
        //==     fprintf(fout, "%f %f \n", free_atom_radial_grid(ir), free_atom_density_[ir]);
        //== }
        //== fclose(fout);
    }
}

inline void Atom_type::print_info() const
{
    printf("\n");
    printf("symbol         : %s\n", symbol_.c_str());
    for (int i = 0; i < 80; i++) {
        printf("-");
    }
    printf("\n");
    printf("name           : %s\n", name_.c_str());
    printf("zn             : %i\n", zn_);
    printf("mass           : %f\n", mass_);
    printf("mt_radius      : %f\n", mt_radius());
    printf("num_mt_points  : %i\n", num_mt_points());
    printf("grid_origin    : %f\n", radial_grid_.first());
    printf("grid_name      : %s\n", radial_grid_.name().c_str());
    printf("\n");
    printf("number of core electrons    : %f\n", num_core_electrons_);
    printf("number of valence electrons : %f\n", num_valence_electrons_);

    if (parameters_.hubbard_correction() && this->hubbard_correction()) {
        printf("Hubbard correction is included in the calculations");
        printf("\n");
        printf("Angular momentum : %d\n", hubbard_l());
        printf("principal quantum number : %d\n", hubbard_n());
        printf("Occupancy : %f\n", hubbard_occupancy_orbital_);
    }

    if (parameters_.full_potential()) {
        printf("\n");
        printf("atomic levels (n, l, k, occupancy, core)\n");
        for (int i = 0; i < (int)atomic_levels_.size(); i++) {
            printf("%i  %i  %i  %8.4f %i\n", atomic_levels_[i].n, atomic_levels_[i].l, atomic_levels_[i].k,
                   atomic_levels_[i].occupancy, atomic_levels_[i].core);
        }
        printf("\n");
        printf("local orbitals\n");
        for (int j = 0; j < (int)lo_descriptors_.size(); j++) {
            printf("[");
            for (int order = 0; order < (int)lo_descriptors_[j].rsd_set.size(); order++) {
                if (order)
                    printf(", ");
                printf("{l : %2i, n : %2i, enu : %f, dme : %i, auto : %i}", lo_descriptors_[j].rsd_set[order].l,
                       lo_descriptors_[j].rsd_set[order].n, lo_descriptors_[j].rsd_set[order].enu,
                       lo_descriptors_[j].rsd_set[order].dme, lo_descriptors_[j].rsd_set[order].auto_enu);
            }
            printf("]\n");
        }

        printf("\n");
        printf("augmented wave basis\n");
        for (int j = 0; j < (int)aw_descriptors_.size(); j++) {
            printf("[");
            for (int order = 0; order < (int)aw_descriptors_[j].size(); order++) {
                if (order)
                    printf(", ");
                printf("{l : %2i, n : %2i, enu : %f, dme : %i, auto : %i}", aw_descriptors_[j][order].l,
                       aw_descriptors_[j][order].n, aw_descriptors_[j][order].enu, aw_descriptors_[j][order].dme,
                       aw_descriptors_[j][order].auto_enu);
            }
            printf("]\n");
        }
        printf("maximum order of aw : %i\n", max_aw_order_);
    }

    printf("\n");
    printf("total number of radial functions : %i\n", indexr().size());
    printf("lmax of radial functions : %i\n", indexr().lmax());
    printf("maximum number of radial functions per orbital quantum number: %i\n", indexr().max_num_rf());
    printf("total number of basis functions : %i\n", indexb().size());
    printf("number of aw basis functions : %i\n", indexb().size_aw());
    printf("number of lo basis functions : %i\n", indexb().size_lo());
}

inline void Atom_type::read_input_core(json const& parser)
{
    std::string core_str = parser["core"];
    if (int size = (int)core_str.size()) {
        if (size % 2) {
            std::string s = std::string("wrong core configuration string : ") + core_str;
            TERMINATE(s);
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
                std::string s = std::string("wrong principal quantum number : ") + std::string(1, c1);
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
                    std::string s = std::string("wrong angular momentum label : ") + std::string(1, c2);
                    TERMINATE(s);
                }
            }

            atomic_level_descriptor level;
            level.n    = n;
            level.l    = l;
            level.core = true;
            for (int ist = 0; ist < 28; ist++) {
                if ((level.n == atomic_conf[zn_ - 1][ist][0]) && (level.l == atomic_conf[zn_ - 1][ist][1])) {
                    level.k         = atomic_conf[zn_ - 1][ist][2];
                    level.occupancy = double(atomic_conf[zn_ - 1][ist][3]);
                    atomic_levels_.push_back(level);
                }
            }
        }
    }
}

inline void Atom_type::read_input_aw(json const& parser)
{
    radial_solution_descriptor rsd;
    radial_solution_descriptor_set rsd_set;

    /* default augmented wave basis */
    rsd.n = -1;
    rsd.l = -1;
    for (size_t order = 0; order < parser["valence"][0]["basis"].size(); order++) {
        rsd.enu      = parser["valence"][0]["basis"][order]["enu"];
        rsd.dme      = parser["valence"][0]["basis"][order]["dme"];
        rsd.auto_enu = parser["valence"][0]["basis"][order]["auto"];
        aw_default_l_.push_back(rsd);
    }

    for (size_t j = 1; j < parser["valence"].size(); j++) {
        rsd.l = parser["valence"][j]["l"];
        rsd.n = parser["valence"][j]["n"];
        rsd_set.clear();
        for (size_t order = 0; order < parser["valence"][j]["basis"].size(); order++) {
            rsd.enu      = parser["valence"][j]["basis"][order]["enu"];
            rsd.dme      = parser["valence"][j]["basis"][order]["dme"];
            rsd.auto_enu = parser["valence"][j]["basis"][order]["auto"];
            rsd_set.push_back(rsd);
        }
        aw_specific_l_.push_back(rsd_set);
    }
}

inline void Atom_type::read_input_lo(json const& parser)
{
    radial_solution_descriptor rsd;
    radial_solution_descriptor_set rsd_set;

    if (!parser.count("lo")) {
        return;
    }

    int l;
    for (size_t j = 0; j < parser["lo"].size(); j++) {
        l = parser["lo"][j]["l"];

        local_orbital_descriptor lod;
        lod.l = l;
        rsd.l = l;
        rsd_set.clear();
        for (size_t order = 0; order < parser["lo"][j]["basis"].size(); order++) {
            rsd.n        = parser["lo"][j]["basis"][order]["n"];
            rsd.enu      = parser["lo"][j]["basis"][order]["enu"];
            rsd.dme      = parser["lo"][j]["basis"][order]["dme"];
            rsd.auto_enu = parser["lo"][j]["basis"][order]["auto"];
            rsd_set.push_back(rsd);
        }
        lod.rsd_set = rsd_set;
        lo_descriptors_.push_back(lod);
    }
}


inline void Atom_type::read_pseudo_uspp(json const& parser)
{
    symbol_ = parser["pseudo_potential"]["header"]["element"];

    double zp;
    zp  = parser["pseudo_potential"]["header"]["z_valence"];
    zn_ = int(zp + 1e-10);

    int nmtp = parser["pseudo_potential"]["header"]["mesh_size"];

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
        spin_orbit_coupling(true);
    }

    int nbf = parser["pseudo_potential"]["header"]["number_of_proj"];

    for (int i = 0; i < nbf; i++) {
        auto beta = parser["pseudo_potential"]["beta_projectors"][i]["radial_function"].get<std::vector<double>>();
        if (static_cast<int>(beta.size()) > num_mt_points()) {
            std::stringstream s;
            s << "wrong size of beta functions for atom type " << symbol_ << " (label: " << label_ << ")" << std::endl
              << "size of beta radial functions in the file: " << beta.size() << std::endl
              << "radial grid size: " << num_mt_points();
            TERMINATE(s);
        }
        int l = parser["pseudo_potential"]["beta_projectors"][i]["angular_momentum"];
        if (spin_orbit_coupling_) {
            // we encode the fact that the total angular momentum j = l
            // -1/2 or l + 1/2 by changing the sign of l

            double j = parser["pseudo_potential"]["beta_projectors"][i]["total_angular_momentum"];
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
            int i    = parser["pseudo_potential"]["augmentation"][k]["i"];
            int j    = parser["pseudo_potential"]["augmentation"][k]["j"];
            //int idx  = j * (j + 1) / 2 + i;
            int l    = parser["pseudo_potential"]["augmentation"][k]["angular_momentum"];
            auto qij = parser["pseudo_potential"]["augmentation"][k]["radial_function"].get<std::vector<double>>();
            if ((int)qij.size() != num_mt_points()) {
                TERMINATE("wrong size of qij");
            }
            add_q_radial_function(i, j, l, qij);
        }
    }

    /* read starting wave functions ( UPF CHI ) */
    if (parser["pseudo_potential"].count("atomic_wave_functions")) {
        size_t nwf = parser["pseudo_potential"]["atomic_wave_functions"].size();
        std::vector<double> occupancies;
        for (size_t k = 0; k < nwf; k++) {
            //std::pair<int, std::vector<double>> wf;
            auto v = parser["pseudo_potential"]["atomic_wave_functions"][k]["radial_function"].get<std::vector<double>>();

            if ((int)v.size() != num_mt_points()) {
                std::stringstream s;
                s << "wrong size of atomic functions for atom type " << symbol_ << " (label: " << label_ << ")"
                  << std::endl
                  << "size of atomic radial functions in the file: " << v.size() << std::endl
                  << "radial grid size: " << num_mt_points();
                TERMINATE(s);
            }
            int l = parser["pseudo_potential"]["atomic_wave_functions"][k]["angular_momentum"];
            add_ps_atomic_wf(l, v);

            if (spin_orbit_coupling() &&
                parser["pseudo_potential"]["atomic_wave_functions"][k].count("total_angular_momentum") &&
                parser["pseudo_potential"]["atomic_wave_functions"][k].count("occupation")) {
                //double jchi = parser["pseudo_potential"]["atomic_wave_functions"][k]["total_angular_momentum"];
                occupancies.push_back(parser["pseudo_potential"]["atomic_wave_functions"][k]["occupation"]);
            }
        }
        ps_atomic_wf_occ(occupancies);
    }
}

inline void Atom_type::read_pseudo_paw(json const& parser)
{
    is_paw_ = true;

    /* read core energy */
    paw_core_energy(parser["pseudo_potential"]["header"]["paw_core_energy"]);

    /* cutoff index */
    int cutoff_radius_index = parser["pseudo_potential"]["header"]["cutoff_radius_index"];

    /* read core density and potential */
    paw_ae_core_charge_density(parser["pseudo_potential"]["paw_data"]["ae_core_charge_density"].get<std::vector<double>>());

    /* read occupations */
    paw_wf_occ(parser["pseudo_potential"]["paw_data"]["occupations"].get<std::vector<double>>());

    /* setups for reading AE and PS basis wave functions */
    int num_wfc = num_beta_radial_functions();
    paw_ae_wfs_ = mdarray<double, 2>(num_mt_points(), num_wfc);
    paw_ae_wfs_.zero();

    paw_ps_wfs_ = mdarray<double, 2>(num_mt_points(), num_wfc);
    paw_ps_wfs_.zero();

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

        std::memcpy(&paw_ae_wfs_(0, i), wfc.data(), cutoff_radius_index * sizeof(double));

        wfc = parser["pseudo_potential"]["paw_data"]["ps_wfc"][i]["radial_function"].get<std::vector<double>>();

        if ((int)wfc.size() > num_mt_points()) {
            std::stringstream s;
            s << "wrong size of ps_wfc functions for atom type " << symbol_ << " (label: " << label_ << ")" << std::endl
              << "size of ps_wfc radial functions in the file: " << wfc.size() << std::endl
              << "radial grid size: " << num_mt_points();
            TERMINATE(s);
        }
        std::memcpy(&paw_ps_wfs_(0, i), wfc.data(), cutoff_radius_index * sizeof(double));
    }
}

inline void Atom_type::read_input(const std::string& fname)
{
    json parser;
    std::ifstream(fname) >> parser;

    if (!parameters_.full_potential()) {
        read_pseudo_uspp(parser);

        if (parser["pseudo_potential"].count("paw_data")) {
            read_pseudo_paw(parser);
        }
    }

    if (parameters_.full_potential()) {
        name_     = parser["name"];
        symbol_   = parser["symbol"];
        mass_     = parser["mass"];
        zn_       = parser["number"];
        double r0 = parser["rmin"];
        double R  = parser["rmt"];
        int nmtp  = parser["nrmt"];

        set_radial_grid(radial_grid_t::exponential_grid, nmtp, r0, R);

        read_input_core(parser);

        read_input_aw(parser);

        read_input_lo(parser);

        /* create free atom radial grid */
        auto fa_r              = parser["free_atom"]["radial_grid"].get<std::vector<double>>();
        free_atom_radial_grid_ = Radial_grid_ext<double>(static_cast<int>(fa_r.size()), fa_r.data());
        /* read density */
        free_atom_density_ = parser["free_atom"]["density"].get<std::vector<double>>();
    }

    // it is already done in input.h. I just initialize the
    // different constants

    read_hubbard_input();
}

inline double Atom_type::ClebschGordan(const int l, const double j, const double mj, const int spin)
{
    // l : orbital angular momentum
    // m:  projection of the total angular momentum $m \pm /frac12$
    // spin: Component of the spinor, 0 up, 1 down

    double CG = 0.0; // Clebsch Gordan coeeficient cf PRB 71, 115106 page 3 first column

    if ((spin != 0) && (spin != 1)) {
        printf("Error : unkown spin direction\n");
    }

    const double denom = sqrt(1.0 / (2.0 * l + 1.0));

    if (std::abs(j - l - 0.5) < 1e-8) {
        int m = static_cast<int>(mj - 0.5);
        if (spin == 0) {
            CG = sqrt(l + m + 1.0);
        }
        if (spin == 1) {
            CG = sqrt((l - m));
        }
    } else {
        if (std::abs(j - l + 0.5) < 1e-8) {
            int m = static_cast<int>(mj + 0.5);
            if (m < (1 - l)) {
                CG = 0.0;
            } else {
                if (spin == 0) {
                    CG = sqrt(l - m + 1);
                }
                if (spin == 1) {
                    CG = -sqrt(l + m);
                }
            }
        } else {
            printf("Clebsch gordan coefficients do not exist for this combination of j=%.5lf and l=%d\n", j, l);
            exit(0);
        }
    }
    return (CG * denom);
}

// this function computes the U^sigma_{ljm mj} coefficient that
// rotates the complex spherical harmonics to the real one for the
// spin orbit case

// mj is normally half integer from -j to j but to avoid computation
// error it is considered as integer so mj = 2 mj

inline double_complex
Atom_type::calculate_U_sigma_m(const int l, const double j, const int mj, const int mp, const int sigma)
{

    if ((sigma != 0) && (sigma != 1)) {
        printf("SphericalIndex function : unkown spin direction\n");
        return 0;
    }

    if (std::abs(j - l - 0.5) < 1e-8) {
        // j = l + 1/2
        // m = mj - 1/2

        int m1 = (mj - 1) >> 1;
        if (sigma == 0) { // up spin
            if (m1 < -l) { // convention U^s_{mj,m'} = 0
                return 0.0;
            } else {// U^s_{mj,mp} =
                return SHT::rlm_dot_ylm(l, m1, mp);
            }
        } else { // down spin
            if ((m1 + 1) > l) {
                return 0.0;
            } else {
                return SHT::rlm_dot_ylm(l, m1 + 1, mp);
            }
        }
    } else {
        if (std::abs(j - l + 0.5) < 1e-8) {
            int m1 = (mj + 1) >> 1;
            if (sigma == 0) {
                return SHT::rlm_dot_ylm(l, m1 - 1, mp);
            } else {
                return SHT::rlm_dot_ylm(l, m1, mp);
            }
        } else {
            printf("Spherical Index function : l and j are not compatible\n");
            exit(0);
        }
    }
}

void Atom_type::generate_f_coefficients(void)
{
    // we consider Pseudo potentials with spin orbit couplings

    // First thing, we need to compute the
    // \f[f^{\sigma\sigma^\prime}_{l,j,m;l\prime,j\prime,m\prime}\f]
    // They are defined by Eq.9 of Ref PRB 71, 115106
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

                        // yes durty but loop over double is worst.
                        // since mj is only important for the rotation
                        // of the spherical harmonics the code takes
                        // into account this odd convention.

                        int jj1 = static_cast<int>(2.0 * j1 + 1e-8);
                        for (int mj = -jj1; mj <= jj1; mj += 2) {
                            coef += calculate_U_sigma_m(l1, j1, mj, m1, sigma1) *
                                    this->ClebschGordan(l1, j1, mj / 2.0, sigma1) *
                                    std::conj(calculate_U_sigma_m(l2, j2, mj, m2, sigma2)) *
                                    this->ClebschGordan(l2, j2, mj / 2.0, sigma2);
                        }
                        f_coefficients_(xi1, xi2, sigma1, sigma2) = coef;
                    }
                }
            }
        }
    }
}

void Atom_type::calculate_ak_coefficients(mdarray<double, 5> &ak)
{
    // compute the ak coefficients appearing in the general treatment of
    // hubbard corrections.  expression taken from Liechtenstein {\it et
    // al}, PRB 52, R5467 (1995)

    // Note that for consistency, the ak are calculated with complex
    // harmonics in the gaunt coefficients <R_lm|Y_l'm'|R_l''m''>.
    // we need to keep it that way because of the hubbard potential
  // With a spherical one it does not really matter-
    ak.zero();

    for (int m1 = -this->hubbard_l_; m1 <= this->hubbard_l_; m1++) {
        for (int m2 = -this->hubbard_l_; m2 <= this->hubbard_l_; m2++) {
            for (int m3 = -this->hubbard_l_; m3 <= this->hubbard_l_; m3++) {
                for (int m4 = -this->hubbard_l_; m4 <= this->hubbard_l_; m4++) {
                    for (int k = 0; k < 2*this->hubbard_l_; k += 2) {
                        double sum = 0.0;
                        for (int q = -k; q <= k; q++) {
                            sum += SHT::gaunt_rlm_ylm_rlm(this->hubbard_l_, k, this->hubbard_l_, m1, q, m2) *
                                SHT::gaunt_rlm_ylm_rlm(this->hubbard_l_, k, this->hubbard_l_, m3, q, m4);
                        }
                        // hmmm according to PRB 52, R5467 it is 4
                        // \pi/(2 k + 1) -> 4 \pi / (4 * k + 1) because
                        // I only consider a_{k=0} a_{k=2}, a_{k=4}
                        ak(k/2,
                           m1 + this->hubbard_l_,
                           m2 + this->hubbard_l_,
                           m3 + this->hubbard_l_,
                           m4 + this->hubbard_l_) = 4.0 * sum * M_PI / static_cast<double>(2 * k + 1);
                    }
                }
            }
        }
    }
}

/// this function computes the matrix elements of the orbital part of
/// the electron-electron interactions. we effectively compute

/// \f[ u(m,m'',m',m''') = \left<m,m''|V_{e-e}|m',m'''\right> \sum_k
/// a_k(m,m',m'',m''') F_k \f] where the F_k are calculated for real
/// spherical harmonics



void Atom_type::compute_hubbard_matrix()
{
    this->hubbard_matrix_ = mdarray<double, 4>(2 * this->hubbard_l_ + 1,
                                               2 * this->hubbard_l_ + 1,
                                               2 * this->hubbard_l_ + 1,
                                               2 * this->hubbard_l_ + 1);
    mdarray<double, 5> ak(this->hubbard_l_,
                          2 * this->hubbard_l_ + 1,
                          2 * this->hubbard_l_ + 1,
                          2 * this->hubbard_l_ + 1,
                          2 * this->hubbard_l_ + 1);
    std::vector<double> F(4);
    hubbard_F_coefficients(&F[0]);
    calculate_ak_coefficients(ak);


    // the indices are rotated around

    // <m, m |vee| m'', m'''> = hubbard_matrix(m, m'', m', m''')
    this->hubbard_matrix_.zero();
    for(int m1 = 0; m1 < 2 * this->hubbard_l_ + 1; m1++) {
        for(int m2 = 0; m2 < 2 * this->hubbard_l_ + 1; m2++) {
            for(int m3 = 0; m3 < 2 * this->hubbard_l_ + 1; m3++) {
                for(int m4 = 0; m4 < 2 * this->hubbard_l_ + 1; m4++) {
                    for(int k = 0; k < hubbard_l_; k++)
                        this->hubbard_matrix(m1, m2, m3, m4) += ak (k, m1, m3, m2, m4) * F[k];
                }
            }
        }
    }
}

void Atom_type::read_hubbard_input()
{
    if(!parameters_.Hubbard().hubbard_correction_) {
        return;
    }

    for(auto &d: parameters_.Hubbard().species) {
        if (d.first == symbol_) {
            hubbard_U_ = d.second[0];
            hubbard_J_ = d.second[1];
            hubbard_coefficients_[0] = d.second[0];
            hubbard_coefficients_[1] = d.second[1];
            hubbard_coefficients_[2] = d.second[2];
            hubbard_coefficients_[3] = d.second[3];
            hubbard_alpha_ = d.second[4];
            hubbard_beta_ = d.second[5];
            starting_magnetization_theta_ = d.second[7];
            starting_magnetization_phi_ = d.second[8];
            starting_magnetization_ = d.second[6];
        }
    }
}
} // namespace

#endif // __ATOM_TYPE_H__
