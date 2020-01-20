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

/** \file atom_type.hpp
 *
 *  \brief Contains declaration and implementation of sirius::Atom_type class.
 */

#ifndef __ATOM_TYPE_HPP__
#define __ATOM_TYPE_HPP__

#include "atomic_data.hpp"
#include "geometry3d.hpp"
#include "radial_solver.hpp"
#include "Potential/xc_functional_base.hpp"
#include "simulation_parameters.hpp"
#include "radial_functions_index.hpp"
#include "basis_functions_index.hpp"
#include "hubbard_orbitals_descriptor.hpp"
#include "SHT/sht.hpp"
#include "utils/profiler.hpp"

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
    /** Low-energy levels are core states. Information about core states is defined in the species file. */
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
    /** In case of LAPW this list defines all local orbitals for a given atom type. In case of PP-PW this is
        a list of all beta-projectors. */
    std::vector<local_orbital_descriptor> lo_descriptors_;

    /// Maximum number of AW radial functions across angular momentums.
    int max_aw_order_{0};

    int offset_lo_{-1}; // TODO: better name // TODO: should be moved to Unit_cell.

    /// Index of radial basis functions.
    /** Radial index is build from the list of local orbiatl descriptors Atom_type::lo_descriptors_.
        In LAPW this index is used to iterate ovver combined set of APW and local-orbital radial functions.
        In pseudo_potential case this index is used to iterate over radial part of beta-projectors.
     */
    radial_functions_index indexr_;

    /// Index of atomic basis functions (radial function * spherical harmonic).
    /** This index is used in LAPW to combine APW and local-orbital muffin-tin functions */
    basis_functions_index indexb_;

    /// Index for the radial atomic functions.
    radial_functions_index indexr_wfs_;

    /// Index of atomic wavefunctions (radial function * spherical harmonic).
    basis_functions_index indexb_wfs_;

    /// List of Hubbard orbital descriptors.
    /** List of sirius::hubbard_orbital_descriptor for each orbital. The corresponding radial functions are stored in
        Atom_type::hubbard_radial_functions_ */
    std::vector<hubbard_orbital_descriptor> lo_descriptors_hub_; // TODO: to be removed

    /// Index for the radial hubbard basis functions
    radial_functions_index hubbard_indexr_; // TODO: to be removed

    /// Index of hubbard basis functions (radial function * spherical harmonic).
    basis_functions_index hubbard_indexb_; // TODO: to be removed

    /// Index of radial functions for hubbard orbitals.
    radial_functions_index indexr_hub_;

    /// Index of basis functions for hubbard orbitals.
    basis_functions_index indexb_hub_;

    /// Radial functions of beta-projectors.
    /** This are the beta-function in the USPP file. Pairs of [l, beta_l(r)] are stored. In case of spin-orbit
        coupling orbital quantum numbers in this list can be positive and negative. This is used to derive the
        total angular momentum of the orbitals:
        \f[
            j = \left\{ \begin{array}{ll}
              |\ell| - 0.5 & \ell < 0 \\
              \ell + 0.5 & \ell > 0
            \end{array} \right.
        \f]
     */
    std::vector<std::pair<int, Spline<double>>> beta_radial_functions_;

    /// Atomic wave-functions used to setup the initial subspace and to apply U-correction.
    /** This are the chi wave-function in the USPP file. Tuples of [n, l, occ, chi_l(r)] are stored. In case of spin-orbit
        coupling orbital quantum numbers in this list can be positive and negative. This is used to derive the
        total angular momentum of the orbitals:
        \f[
            j = \left\{ \begin{array}{ll}
              |\ell| - 0.5 & \ell < 0 \\
              \ell + 0.5 & \ell > 0
            \end{array} \right.
        \f]
     */
    std::vector<std::tuple<int, int, double, Spline<double>>> ps_atomic_wfs_;

    /// List of radial functions for hubbard orbitals.
    /** Hubbard orbitals are copied from atomic wave-functions and are independent of spin. This list is compatible
        with Atom_type::lo_descriptors_hub_ */
    std::vector<Spline<double>> hubbard_radial_functions_;

    /// Radial functions of the Q-operator.
    /** The dimension of this array is fully determined by the number and lmax of beta-projectors.
        Beta-projectors must be loaded before loading the Q radial functions. */
    mdarray<Spline<double>, 2> q_radial_functions_l_;

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

    /// List of all-electron wave functions of the PAW method.
    std::vector<std::vector<double>> ae_paw_wfs_;

    /// All-electron wave functions of the PAW method packed in a single array.
    /** The number of wave functions is equal to the number of beta-projectors. */
    mdarray<double, 2> ae_paw_wfs_array_;

    /// List of pseudo wave functions of the PAW method.
    std::vector<std::vector<double>> ps_paw_wfs_;

    /// Pseudo wave functions of the PAW method packed in a single array.
    /** The number of wave functions is equal to the number of beta-projectors. */
    mdarray<double, 2> ps_paw_wfs_array_;

    /// Occupations of PAW wave-functions.
    /** Length of vector is the same as the number of beta projectors */
    std::vector<double> paw_wf_occ_; // TODO: is this ever used? remove if unnecessary

    /// Core electron contribution to all electron charge density in PAW method.
    std::vector<double> paw_ae_core_charge_density_;

    /// True if the pseudo potential includes spin orbit coupling.
    bool spin_orbit_coupling_{false};

    /// Vector containing all orbitals informations that are relevant for the Hubbard correction.
    std::vector<hubbard_orbital_descriptor> hubbard_orbitals_;

    /// List of radial descriptor sets used to construct hubbard orbitals.
    std::vector<local_orbital_descriptor> hubbard_lo_descriptors_;

    /// Hubbard correction.
    bool hubbard_correction_{false};

    /// Inverse of (Q_{\xi \xi'j}^{-1} + beta_pw^{H}_{\xi} * beta_pw_{xi'})
    /** Used in Chebyshev iterative solver as a block-diagonal preconditioner */
    matrix<double_complex> p_mtrx_;

    /// f_coefficients defined in doi:10.1103/PhysRevB.71.115106 Eq.9 only
    /// valid when SO interactions are on
    mdarray<double_complex, 4> f_coefficients_;

    /// List of atom indices (global) for a given type.
    std::vector<int> atom_id_;

    /// Name of the input file for this atom type.
    std::string file_name_;

    mdarray<int, 2> idx_radial_integrals_;

    mutable mdarray<double, 3> rf_coef_;
    mutable mdarray<double, 3> vrf_coef_;

    /// True if the atom type was initialized.
    /** After initialization it is forbidden to modify the parameters of the atom type. */
    bool initialized_{false};

    /// Pass information from the hubbard input section (parsed in input.hpp) to the atom type.
    void read_hubbard_input();

    /// Generate coefficients used in spin-orbit case.
    void generate_f_coefficients();

    inline void read_input_core(json const& parser);

    inline void read_input_aw(json const& parser);

    inline void read_input_lo(json const& parser);

    inline void read_pseudo_uspp(json const& parser);

    inline void read_pseudo_paw(json const& parser);

    /// Read atomic parameters from json file or string.
    inline void read_input(std::string const& str__);

    /// Initialize descriptors of the augmented-wave radial functions.
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
    Atom_type(Atom_type const& src) = delete;

    /* forbid assignment operator */
    Atom_type& operator=(Atom_type const& src) = delete;

  protected:
    /// Radial grid of the muffin-tin sphere.
    Radial_grid<double> radial_grid_;

    /// Density of a free atom.
    Spline<double> free_atom_density_spline_;

    /// Density of a free atom as read from the input file.
    /** Does not contain 4 Pi and r^2 prefactors. */
    std::vector<double> free_atom_density_;

    /// Radial grid of a free atom.
    Radial_grid<double> free_atom_radial_grid_;

  public:
    /// Constructor.
    /** Basic parameters of atom type are passed as contructor arguments. */
    Atom_type(Simulation_parameters const&                parameters__,
              std::string                                 symbol__,
              std::string                                 name__,
              int                                         zn__,
              double                                      mass__,
              std::vector<atomic_level_descriptor> const& levels__)
        : parameters_(parameters__)
        , symbol_(symbol__)
        , name_(name__)
        , zn_(zn__)
        , mass_(mass__)
        , atomic_levels_(levels__)
        {
        }

    /// Constructor.
    /** ID of atom type and label are passed as arguments. The rest of parameters are obtained from the
        species input file. */
    Atom_type(Simulation_parameters const& parameters__, int id__, std::string label__, std::string file_name__)
        : parameters_(parameters__)
        , id_(id__)
        , label_(label__)
        , file_name_(file_name__)
        {
        }

    /// Move constructor.
    Atom_type(Atom_type&& src) = default;

    /// Initialize the atom type.
    /** Once the unit cell is populated with all atom types and atoms, each atom type can be initialized. */
    inline void init(int offset_lo__);

    /// Set the radial grid of the given type.
    inline void set_radial_grid(radial_grid_t grid_type__, int num_points__, double rmin__, double rmax__, double p__)
    {
        radial_grid_ = Radial_grid_factory<double>(grid_type__, num_points__, rmin__, rmax__, p__);
        if (parameters_.processing_unit() == device_t::GPU) {
            radial_grid_.copy_to_device();
        }
    }

    /// Set external radial grid.
    inline void set_radial_grid(int num_points__, double const* points__)
    {
        radial_grid_ = Radial_grid_ext<double>(num_points__, points__);
        if (parameters_.processing_unit() == device_t::GPU) {
            radial_grid_.copy_to_device();
        }
    }

    /// Set radial grid of the free atom.
    /** The grid is extended to effective infinity (usually > 100 a.u.) */
    inline void set_free_atom_radial_grid(int num_points__, double const* points__)
    {
        if (num_points__ <= 0) {
            TERMINATE("wrong number of radial points");
        }
        free_atom_radial_grid_ = Radial_grid_ext<double>(num_points__, points__);
    }

    inline void set_free_atom_radial_grid(Radial_grid<double>&& rgrid__)
    {
        free_atom_radial_grid_ = std::move(rgrid__);
    }

    /// Add augmented-wave descriptor.
    inline void add_aw_descriptor(int n, int l, double enu, int dme, int auto_enu)
    {
        if (static_cast<int>(aw_descriptors_.size()) < (l + 1)) {
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

    /// Add the entire local orbital descriptor.
    inline void add_lo_descriptor(local_orbital_descriptor const& lod__)
    {
        lo_descriptors_.push_back(lod__);
    }

    /// Add atomic radial function to the list.
    inline void add_ps_atomic_wf(int n__, int l__, std::vector<double> f__, double occ__ = 0.0)
    {
        local_orbital_descriptor lod;

        Spline<double> s(radial_grid_, f__);
        ps_atomic_wfs_.push_back(std::move(std::make_tuple(n__, l__, occ__, std::move(s))));
    }

    /// Return a tuple describing a given atomic radial function
    std::tuple<int, int, double, Spline<double>> const& ps_atomic_wf(int idx__) const
    {
        return ps_atomic_wfs_[idx__];
    }

    /// Return maximum orbital quantum number for the atomic wave-functions.
    inline int lmax_ps_atomic_wf() const
    {
        int lmax{-1};
        for (auto& e: ps_atomic_wfs_) {
            auto l = std::get<1>(e);
            /* need to take |l| since the total angular momentum is encoded in the sign of l */
            lmax = std::max(lmax, std::abs(l));
        }
        return lmax;
    }

    /// Add a radial function of beta-projector to a list of functions.
    /** This is the only allowed way to add beta projectors. */
    inline void add_beta_radial_function(int l__, std::vector<double> beta__)
    {
        if (augment_) {
            TERMINATE("can't add more beta projectors");
        }
        Spline<double> s(radial_grid_, beta__);
        beta_radial_functions_.push_back(std::move(std::make_pair(l__, std::move(s))));

        local_orbital_descriptor lod;
        lod.l = std::abs(l__);

        /* for spin orbit coupling; we can always do that there is
           no insidence on the rest when calculations exclude SO */
        if (l__ < 0) {
            lod.total_angular_momentum = lod.l - 0.5;
        } else {
            lod.total_angular_momentum = lod.l + 0.5;
        }
        /* add local orbital descriptor for the current beta-projector */
        lo_descriptors_.push_back(lod);
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

        /* need to take |l| since the total angular momentum is encoded in the sign of l */
        for (auto& e: beta_radial_functions_) {
            lmax = std::max(lmax, std::abs(e.first));
        }
        return lmax;
    }

    /// Number of beta-radial functions.
    inline int num_beta_radial_functions() const
    {
        assert(lo_descriptors_.size() == beta_radial_functions_.size());
        return lo_descriptors_.size();
    }

    /// Add radial function of the augmentation charge.
    /** Radial functions of beta projectors must be added already. Their total number will be used to
        deterimine the storage size for the radial functions of the augmented charge. */
    inline void add_q_radial_function(int idxrf1__, int idxrf2__, int l__, std::vector<double> qrf__)
    {
        /* sanity check */
        if (l__ > 2 * lmax_beta()) {
            std::stringstream s;
            s << "wrong l for Q radial functions of atom type " << label_ << std::endl
              << "current l: " << l__ << std::endl
              << "lmax_beta: " << lmax_beta() << std::endl
              << "maximum allowed l: " << 2 * lmax_beta();

            TERMINATE(s);
        }

        if (!augment_) {
            /* once we add a Q-radial function, we need to augment the charge */
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

        int ijv = utils::packed_index(idxrf1__, idxrf2__);
        q_radial_functions_l_(ijv, l__) = Spline<double>(radial_grid_, qrf__);
    }

    /// Return true if this atom type has an augementation charge.
    inline bool augment() const
    {
        return augment_;
    }

    /// Set the radial function of the local potential.
    inline std::vector<double>& local_potential(std::vector<double> vloc__)
    {
        local_potential_ = vloc__;
        return local_potential_;
    }

    /// Get the radial function of the local potential.
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

    /// Add all-electron PAW wave-function.
    inline void add_ae_paw_wf(std::vector<double> f__)
    {
        ae_paw_wfs_.push_back(f__);
    }

    /// Get all-electron PAW wave-function.
    inline std::vector<double> const& ae_paw_wf(int i__) const
    {
        return ae_paw_wfs_[i__];
    }

    /// Get the number of all-electron PAW wave-functions.
    inline int num_ae_paw_wf() const
    {
        return static_cast<int>(ae_paw_wfs_.size());
    }

    inline void add_ps_paw_wf(std::vector<double> f__)
    {
        ps_paw_wfs_.push_back(f__);
    }

    inline std::vector<double> const& ps_paw_wf(int i__) const
    {
        return ps_paw_wfs_[i__];
    }

    inline int num_ps_paw_wf() const
    {
        return static_cast<int>(ps_paw_wfs_.size());
    }

    inline mdarray<double, 2> const& ae_paw_wfs_array() const
    {
        return ae_paw_wfs_array_;
    }

    inline mdarray<double, 2> const& ps_paw_wfs_array() const
    {
        return ps_paw_wfs_array_;
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

    /// Initialize the free atom density (smooth or true).
    inline void init_free_atom_density(bool smooth);

    /// Add a hubbard orbital to a list.
    /** All atomic functions must already be loaded prior to callinig this function. Atomic wave-functions
        (chi in the uspp file) are used as a definition of "localized orbitals" to which U-correction is applied.
        Full treatment of spin is not considered. In case of spinor wave-functions the are averaged between
        l+1/2 and l-1/2 states. */
    void add_hubbard_orbital(int n__, int l__, double occ__, double U, double J, const double *hub_coef__,
                             double alpha__, double beta__, double J0__)
    {
        /* we have to find one (or two in case of spin-orbit) atomic functions and construct hubbard orbital */
        std::vector<int> idx_rf;
        for (int s = 0; s < (int)ps_atomic_wfs_.size(); s++) {
            auto& e = ps_atomic_wfs_[s];
            int n = std::get<0>(e);
            int l = std::get<1>(e);
            /* for codes which don't provide principal quantum number find the first orbital with a given l */
            if ((n >= 0 && n == n__ && std::abs(l) == l__) || (n < 0 && std::abs(l) == l__)) {
                idx_rf.push_back(s);
                /* in spin orbit case we need to find the second radial function, otherwise we break */
                if (!this->spin_orbit_coupling_) {
                    break;
                }
            }
        }
        if (idx_rf.size() == 0) {
            std::stringstream s;
            s << "[sirius::Atom_type::add_hubbard_orbital] atomic radial function is not found";
            TERMINATE(s);
        }
        if (idx_rf.size() > 2) {
            std::stringstream s;
            s << "[sirius::Atom_type::add_hubbard_orbital] number of atomic functions > 2";
            TERMINATE(s);
        }

        /* create a scalar hubbard wave-function from one or two atomic radial functions */
        Spline<double> s(radial_grid_);
        double f = 1.0 / static_cast<double>(idx_rf.size());
        for (int i: idx_rf) {
            auto& rwf = std::get<3>(ps_atomic_wfs_[i]);
            for (int ir = 0; ir < s.num_points(); ir++) {
                s(ir) += f * rwf(ir);
            }
        }

        /* add orbital to a list */
        hubbard_radial_functions_.push_back(std::move(s));

        hubbard_orbital_descriptor hub(n__, l__, -1, occ__, J, U, hub_coef__, alpha__, beta__, J0__);
        /* add descriptor to a list */
        lo_descriptors_hub_.push_back(std::move(hub));

        for (int s = 0; s < (int)ps_atomic_wfs_.size(); s++) {
            auto& e = ps_atomic_wfs_[s];
            int n = std::get<0>(e);
            int l = std::get<1>(e);
            if (n >= 0) {
                if (n == n__ && std::abs(l) == l__) {
                    hubbard_orbital_descriptor hub(n__, l__, s, occ__, J, U, hub_coef__, alpha__, beta__, J0__);
                    hubbard_orbitals_.push_back(std::move(hub));
                    local_orbital_descriptor lod;
                    lod.l = std::abs(l);

                    if (l < 0) {
                        lod.total_angular_momentum = lod.l - 0.5;
                    } else {
                        lod.total_angular_momentum = lod.l + 0.5;
                    }
                    // we nedd to consider the case where spin orbit
                    // coupling is included. if so is included then we need
                    // to search for its partner with same n, l, but
                    // different j. if not we can stop the for loop
                    hubbard_lo_descriptors_.push_back(lod);
                    if (!this->spin_orbit_coupling_) {
                        break;
                    }
                }
            } else {
                // we do a search per angular momentum
                // we pick the first atomic wave function we
                // find with the right l. It is to deal with
                // codes that do not store all info about wave
                // functions.
                if (std::abs(l) == l__) {
                    hubbard_orbital_descriptor hub(n__, l__, s, occ__, J, U, hub_coef__, alpha__, beta__, J0__);
                    hubbard_orbitals_.push_back(std::move(hub));
                    local_orbital_descriptor lod;
                    lod.l = std::abs(l);
                    hubbard_lo_descriptors_.push_back(lod);
                    if (!this->spin_orbit_coupling_) {
                        break;
                    }
                }
            }
        }
    }

    // TODO: remove in future
    /// Return the total number of radial functions of hubbard orbitals.
    inline int num_hubbard_orbitals() const
    {
        return static_cast<int>(hubbard_orbitals_.size());
    }

    inline hubbard_orbital_descriptor const& hubbard_orbital(const int channel_) const
    {
        assert(hubbard_orbitals_.size() > 0);
        return hubbard_orbitals_[channel_];
    }

    // TODO: this is needed for stress code but should be removed in futre
    inline std::vector<hubbard_orbital_descriptor> const& hubbard_orbitals() const
    {
        return hubbard_orbitals_;
    }

    /// Print basic info to standard output.
    inline void print_info() const;

    /// Return atom type id.
    inline int id() const
    {
        return id_;
    }

    /// Return ionic charge (as positive integer).
    inline int zn() const
    {
        assert(zn_ > 0);
        return zn_;
    }

    /// Set ionic charge.
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

    /// Return atomic mass.
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

    /// Get free atom density at i-th point of radial grid.
    inline double free_atom_density(const int idx) const
    {
        return free_atom_density_spline_(idx);
    }

    /// Get free atom density at point x.
    inline double free_atom_density(double x) const
    {
        return free_atom_density_spline_.at_point(x);
    }

    /// Set the free atom all-electron density.
    inline void free_atom_density(std::vector<double> rho__)
    {
        free_atom_density_ = rho__;
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

    /// Return const reference to the index of radial functions.
    /** The index can be used to determine the total number of radial functions */
    inline radial_functions_index const& indexr() const
    {
        return indexr_;
    }

    inline radial_functions_index const& indexr_wfs() const
    {
        return indexr_wfs_;
    }

    inline radial_functions_index const& indexr_hub() const
    {
        return indexr_hub_;
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

    inline basis_functions_index const& indexb_wfs() const
    {
        return indexb_wfs_;
    }

    /// Return whole index of hubbard basis functions.
    inline basis_functions_index const& indexb_hub() const
    {
        return indexb_hub_;
    }

    inline basis_functions_index const& hubbard_indexb_wfc() const
    {
        return hubbard_indexb_;
    }

    inline radial_functions_index const& hubbard_indexr() const
    {
        return hubbard_indexr_;
    }

    inline radial_function_index_descriptor const& indexr_hub(int i) const
    {
        assert(i >= 0 && i < (int)indexr_hub_.size());
        return indexr_[i];
    }

    inline Spline<double> const& hubbard_radial_function(int i) const
    {
        return hubbard_radial_functions_[i];
    }

    inline radial_function_index_descriptor const& indexr_wfs(int i) const
    {
        assert(i >= 0 && i < (int)indexr_wfs_.size());
        return indexr_wfs_[i];
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

    /// Return atom ID (global index) by the index of atom withing a given type.
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
        d_mtrx_ion_ = matrix<double>(num_beta_radial_functions(), num_beta_radial_functions(),
                                     memory_t::host, "Atom_type::d_mtrx_ion_");
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
        this->spin_orbit_coupling_ = so__;
        return this->spin_orbit_coupling_;
    }

    /// Get the Hubbard correction switch.
    inline bool hubbard_correction() const
    {
        return hubbard_correction_;
    }

    /// Set the Hubbard correction switch.
    inline bool hubbard_correction(bool ldapu__)
    {
        this->hubbard_correction_ = ldapu__;
        return this->hubbard_correction_;
    }

    /// Compare indices of beta projectors.
    /** Compare the angular, total angular momentum and radial part of the beta projectors,
     *  leaving the m index free. Only useful when spin orbit coupling is included. */
    inline bool compare_index_beta_functions(const int xi, const int xj) const
    {
        return ((indexb(xi).l == indexb(xj).l) && (indexb(xi).idxrf == indexb(xj).idxrf) &&
                (std::abs(indexb(xi).j - indexb(xj).j) < 1e-8));
    }
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
        std::vector<local_orbital_descriptor> lo_descriptors_wfs;
        local_orbital_descriptor lod;
        for (auto& e: ps_atomic_wfs_) {
            int l = std::get<1>(e);
            lod.l = std::abs(l);
            if (l < 0) {
                lod.total_angular_momentum = l - 0.5;
            } else {
                lod.total_angular_momentum = l + 0.5;
            }
            /* add corresponding descriptor */
            lo_descriptors_wfs.push_back(lod);
        }
        indexr_wfs_.init(lo_descriptors_wfs);
        indexb_wfs_.init(indexr_wfs_);
        if ((int)ps_atomic_wfs_.size() != indexr_wfs_.size()) {
            TERMINATE("[sirius::Atom_type::init] wrong size of atomic orbital list");
        }
    }

    if (hubbard_correction_) {
        /* circus for the hubbard orbitals */
        hubbard_indexr_.init(hubbard_lo_descriptors_);
        hubbard_indexb_.init(hubbard_indexr_);

        indexr_hub_.init(lo_descriptors_hub_);
        indexb_hub_.init(indexr_hub_);
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

inline void Atom_type::init_free_atom_density(bool smooth)
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

inline void Atom_type::print_info() const
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
        std::printf("number of ps wavefunctions       : %i\n", this->indexr_wfs().size());
    }
    std::printf("Hubbard correction               : %s\n", utils::boolstr(this->hubbard_correction()).c_str());
    if (parameters_.hubbard_correction() && this->hubbard_correction_) {
        std::printf("  angular momentum                   : %i\n", hubbard_orbital(0).l);
        std::printf("  principal quantum number           : %i\n", hubbard_orbital(0).n());
        std::printf("  occupancy                          : %f\n", hubbard_orbital(0).occupancy());
        std::printf("  number of hubbard radial functions : %i\n", indexr_hub_.size());
        std::printf("  number of hubbard basis functions  : %i\n", indexb_hub_.size());
    }
    std::printf("spin-orbit coupling              : %s\n", utils::boolstr(this->spin_orbit_coupling()).c_str());
}

inline void Atom_type::read_input_core(json const& parser)
{
    std::string core_str = parser["core"];
    if (int size = (int)core_str.size()) {
        if (size % 2) {
            std::stringstream s;
            s << "wrong core configuration string : " << core_str;
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
        spin_orbit_coupling_ = parser["pseudo_potential"]["header"].value("spin_orbit", spin_orbit_coupling_);
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
            int n = -1;
            double occ{0};
            if (parser["pseudo_potential"]["atomic_wave_functions"][k].count("occupation")) {
                occ = parser["pseudo_potential"]["atomic_wave_functions"][k]["occupation"];
            }

            if (parser["pseudo_potential"]["atomic_wave_functions"][k].count("label")) {
                std::string c1 = parser["pseudo_potential"]["atomic_wave_functions"][k]["label"];
                std::istringstream iss(std::string(1, c1[0]));
                iss >> n;
            }

            if (spin_orbit_coupling() &&
                parser["pseudo_potential"]["atomic_wave_functions"][k].count("total_angular_momentum")) {
                // check if j = l +- 1/2
                if (parser["pseudo_potential"]["atomic_wave_functions"][k]["total_angular_momentum"] < l) {
                    l = -l;
                }
            }
            add_ps_atomic_wf(n, l, v, occ);
        }
    }
}

inline void Atom_type::read_pseudo_paw(json const& parser)
{
    is_paw_ = true;

    auto& header = parser["pseudo_potential"]["header"];
    /* read core energy */
    if (header.count("paw_core_energy")) {
        paw_core_energy(header["paw_core_energy"]);
    } else {
        paw_core_energy(0);
    }

    /* cutoff index */
    int cutoff_radius_index = parser["pseudo_potential"]["header"]["cutoff_radius_index"];

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

inline void Atom_type::read_input(std::string const& str__)
{
    json parser = utils::read_json_from_file_or_string(str__);

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
        name_     = parser["name"];
        symbol_   = parser["symbol"];
        mass_     = parser["mass"];
        zn_       = parser["number"];
        double r0 = parser["rmin"];
        double R  = parser["rmt"];
        int nmtp  = parser["nrmt"];

        auto rg = get_radial_grid_t(parameters_.settings().radial_grid_);

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


inline void Atom_type::generate_f_coefficients()
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
                            coef += SHT::calculate_U_sigma_m(l1, j1, mj, m1, sigma1) *
                                SHT::ClebschGordan(l1, j1, mj / 2.0, sigma1) *
                                std::conj(SHT::calculate_U_sigma_m(l2, j2, mj, m2, sigma2)) *
                                SHT::ClebschGordan(l2, j2, mj / 2.0, sigma2);
                        }
                        f_coefficients_(xi1, xi2, sigma1, sigma2) = coef;
                    }
                }
            }
        }
    }
}

inline void Atom_type::read_hubbard_input()
{
    if(!parameters_.hubbard_input().hubbard_correction_) {
        return;
    }

    this->hubbard_correction_ = false;

    for(auto &d: parameters_.hubbard_input().species) {
        if (d.first == symbol_) {
            int hubbard_l_ = d.second.l;
            int hubbard_n_ = d.second.n;
            if (hubbard_l_ < 0) {
                std::istringstream iss(std::string(1, d.second.level[0]));
                iss >> hubbard_n_;

                if (hubbard_n_ <= 0 || iss.fail()) {
                    std::stringstream s;
                    s << "wrong principal quantum number : " << std::string(1, d.second.level[0]);
                    TERMINATE(s);
                }

                switch (d.second.level[1]) {
                    case 's': {
                        hubbard_l_ = 0;
                        break;
                    }
                    case 'p': {
                        hubbard_l_ = 1;
                        break;
                    }
                    case 'd': {
                        hubbard_l_ = 2;
                        break;
                    }
                    case 'f': {
                        hubbard_l_ = 3;
                        break;
                    }
                    default: {
                        std::stringstream s;
                        s << "wrong angular momentum label : " << std::string(1, d.second.level[1]);
                        TERMINATE(s);
                    }
                }
            }

            add_hubbard_orbital(hubbard_n_,
                                hubbard_l_,
                                d.second.occupancy_,
                                d.second.coeff_[0],
                                d.second.coeff_[1],
                                &d.second.coeff_[0],
                                d.second.coeff_[4],
                                d.second.coeff_[5],
                                0.0);

            this->hubbard_correction_ = true;
        }
    }
}
} // namespace

#endif // __ATOM_TYPE_HPP__
