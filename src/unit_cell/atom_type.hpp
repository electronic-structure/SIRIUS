/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file atom_type.hpp
 *
 *  \brief Contains declaration and implementation of sirius::Atom_type class.
 */

#ifndef __ATOM_TYPE_HPP__
#define __ATOM_TYPE_HPP__

#include "atomic_data.hpp"
#include "radial/radial_solver.hpp"
#include "context/simulation_parameters.hpp"
#include "radial_functions_index.hpp"
#include "basis_functions_index.hpp"
#include "hubbard_orbitals_descriptor.hpp"
#include "core/sht/sht.hpp"
#include "core/sht/gaunt.hpp"
#include "core/r3/r3.hpp"
#include "core/profiler.hpp"
#include "core/packed_index.hpp"

#ifdef SIRIUS_USE_PUGIXML
#include "pugixml.hpp"
#endif

namespace sirius {

/// Descriptor of a local orbital radial function.
struct local_orbital_descriptor
{
    /// Orbital quantum number \f$ \ell \f$.
    angular_momentum am;

    /// Set of radial solution descriptors.
    /** Local orbital is constructed from at least two radial functions in order to make it zero at the
     *  muffin-tin sphere boundary. */
    radial_solution_descriptor_set rsd_set;

    local_orbital_descriptor(angular_momentum am__)
        : am(am__)
    {
    }
};

/// Store basic information about radial pseudo wave-functions.
struct ps_atomic_wf_descriptor
{
    /// Constructor.
    ps_atomic_wf_descriptor(int n__, angular_momentum am__, double occ__, Spline<double> f__)
        : n(n__)
        , am(am__)
        , occ(occ__)
        , f(std::move(f__))
    {
    }
    /// Principal quantum number.
    int n;
    /// Angular momentum quantum number.
    angular_momentum am;
    /// Shell occupancy
    double occ;
    /// Radial wave-function.
    Spline<double> f;
};

inline std::ostream&
operator<<(std::ostream& out, ps_atomic_wf_descriptor const& wfd)
{
    if (wfd.am.s() == 0) {
        out << "{n: " << wfd.n << ", l: " << wfd.am.l() << "}";
    } else {
        out << "{n: " << wfd.n << ", l: " << wfd.am.l() << ", j: " << wfd.am.j() << "}";
    }
    return out;
}

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

    /// Index of radial basis functions.
    /** In LAPW this index is used to iterate over combined set of APW and local-orbital radial functions.
     *  In pseudo_potential case this index is used to iterate over radial part of beta-projectors. */
    radial_functions_index indexr_;

    /// Index of atomic basis functions (radial function * spherical harmonic).
    /** This index is used in LAPW to combine APW and local-orbital muffin-tin functions */
    basis_functions_index indexb_;

    /// Index for the radial atomic functions.
    radial_functions_index indexr_wfs_;

    /// Index of atomic wavefunctions (radial function * spherical harmonic).
    basis_functions_index indexb_wfs_;

    /// List of Hubbard orbital descriptors.
    /** List of sirius::hubbard_orbital_descriptor for each orbital. Each element of the list contains
     *  information about radial function and U and J parameters for the Hubbard correction. The list is
     *  compatible with the indexr_hub_ radial index. */
    std::vector<hubbard_orbital_descriptor> lo_descriptors_hub_;

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
    std::vector<std::pair<angular_momentum, Spline<double>>> beta_radial_functions_;

    /// Atomic wave-functions used to setup the initial subspace and to apply U-correction.
    /** This are the chi wave-function in the USPP file. Lists of [n, j, occ, chi_l(r)] are stored. In case of
     *  spin-orbit coupling orbital angular quantum number j is equal to l +/- 1/2. Otherwise it is just l. */
    std::vector<ps_atomic_wf_descriptor> ps_atomic_wfs_;

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
    /** Length of vector is the same as the number of beta projectors. This is used for the initial guess of
     *  oribtal occupancies. */
    std::vector<double> paw_wf_occ_;

    /// Core electron contribution to all electron charge density in PAW method.
    std::vector<double> paw_ae_core_charge_density_;

    /// True if the pseudo potential includes spin orbit coupling.
    bool spin_orbit_coupling_{false};

    /// Hubbard correction.
    bool hubbard_correction_{false};

    /// f_coefficients defined in doi:10.1103/PhysRevB.71.115106 Eq.9 only
    /// valid when SO interactions are on
    mdarray<std::complex<double>, 4> f_coefficients_;

    /// List of atom indices (global) for a given type.
    std::vector<int> atom_id_;

    /// Name of the input file for this atom type.
    std::string file_name_;

    mdarray<int, 2> idx_radial_integrals_;

    mutable mdarray<double, 3> rf_coef_;
    mutable mdarray<double, 3> vrf_coef_;

    /// Non-zero Gaunt coefficients.
    std::unique_ptr<Gaunt_coefficients<std::complex<double>>> gaunt_coefs_{nullptr};

    /// Maximul orbital quantum number of LAPW basis functions.
    int lmax_apw_{-1};

    /// True if the atom type was initialized.
    /** After initialization it is forbidden to modify the parameters of the atom type. */
    bool initialized_{false};

    /// Pass information from the hubbard input section (parsed in input.hpp) to the atom type.
    void
    read_hubbard_input();

    /// Generate coefficients used in spin-orbit case.
    void
    generate_f_coefficients();

    inline void
    read_input_core(nlohmann::json const& parser);

    inline void
    read_input_aw(nlohmann::json const& parser);

    inline void
    read_input_lo(nlohmann::json const& parser);

    inline void
    read_pseudo_uspp(nlohmann::json const& parser);

    inline void
    read_pseudo_paw(nlohmann::json const& parser);

    /// Read atomic parameters from json file.
    inline void
    read_input(std::string const& str__);

    inline void
    read_input(nlohmann::json const& parser);

    /// Read atomic parameters directly from UPF v2 files
#ifdef SIRIUS_USE_PUGIXML
    inline void
    read_pseudo_uspp(pugi::xml_node const& upf);

    inline void
    read_pseudo_paw(pugi::xml_node const& upf);

    inline void
    read_input(pugi::xml_node const& upf);
#endif

    /// Initialize descriptors of the augmented-wave radial functions.
    inline void
    init_aw_descriptors()
    {
        RTE_ASSERT(this->lmax_apw() >= -1);

        if (this->lmax_apw() >= 0 && aw_default_l_.size() == 0) {
            RTE_THROW("default AW descriptor is empty");
        }

        aw_descriptors_.clear();
        for (int l = 0; l <= this->lmax_apw(); l++) {
            aw_descriptors_.push_back(aw_default_l_);
            for (size_t ord = 0; ord < aw_descriptors_[l].size(); ord++) {
                aw_descriptors_[l][ord].n = l + 1;
                aw_descriptors_[l][ord].l = l;
            }
        }

        for (size_t i = 0; i < aw_specific_l_.size(); i++) {
            int l = aw_specific_l_[i][0].l;
            if (l < this->lmax_apw()) {
                aw_descriptors_[l] = aw_specific_l_[i];
            }
        }
    }

    /* forbid copy constructor */
    Atom_type(Atom_type const& src) = delete;

    /* forbid assignment operator */
    Atom_type&
    operator=(Atom_type const& src) = delete;

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
    /** Basic parameters of atom type are passed as constructor arguments. */
    Atom_type(Simulation_parameters const& parameters__, std::string symbol__, std::string name__, int zn__,
              double mass__, std::vector<atomic_level_descriptor> const& levels__)
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
    void
    init();

    /// Initialize the free atom density (smooth or true).
    void
    init_free_atom_density(bool smooth);

    /// Add a hubbard orbital to a list.
    /** All atomic functions must already be loaded prior to callinig this function. Atomic wave-functions
        (chi in the uspp file) are used as a definition of "localized orbitals" to which U-correction is applied.
        Full treatment of spin is not considered. In case of spinor wave-functions the are averaged between
        l+1/2 and l-1/2 states. */
    void
    add_hubbard_orbital(int n__, int l__, double occ__, double U, double J, const double* hub_coef__, double alpha__,
                        double beta__, double J0__, std::vector<double> initial_occupancy__,
                        const bool use_for_calculations__);

    /// Print basic info to standard output.
    void
    print_info(std::ostream& out__) const;

    /// Set the radial grid of the given type.
    inline void
    set_radial_grid(radial_grid_t grid_type__, int num_points__, double rmin__, double rmax__, double p__)
    {
        radial_grid_ = Radial_grid_factory<double>(grid_type__, num_points__, rmin__, rmax__, p__);
        if (parameters_.processing_unit() == device_t::GPU) {
            radial_grid_.copy_to_device();
        }
    }

    /// Set external radial grid.
    inline void
    set_radial_grid(int num_points__, double const* points__)
    {
        radial_grid_ = Radial_grid_ext<double>(num_points__, points__);
        if (parameters_.processing_unit() == device_t::GPU) {
            radial_grid_.copy_to_device();
        }
    }

    /// Set radial grid of the free atom.
    /** The grid is extended to effective infinity (usually > 100 a.u.) */
    inline void
    set_free_atom_radial_grid(int num_points__, double const* points__)
    {
        if (num_points__ <= 0) {
            RTE_THROW("wrong number of radial points");
        }
        free_atom_radial_grid_ = Radial_grid_ext<double>(num_points__, points__);
    }

    inline void
    set_free_atom_radial_grid(Radial_grid<double>&& rgrid__)
    {
        free_atom_radial_grid_ = std::move(rgrid__);
    }

    inline auto const&
    atomic_level(int idx) const
    {
        return atomic_levels_[idx];
    }

    /// Add augmented-wave descriptor.
    inline void
    add_aw_descriptor(int n, int l, double enu, int dme, int auto_enu)
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
    inline void
    add_lo_descriptor(int ilo, int n, int l, double enu, int dme, int auto_enu)
    {
        if ((int)lo_descriptors_.size() == ilo) {
            angular_momentum am(l);
            lo_descriptors_.push_back(local_orbital_descriptor(am));
        } else {
            if (l != lo_descriptors_[ilo].am.l()) {
                std::stringstream s;
                s << "wrong angular quantum number" << std::endl
                  << "atom type id: " << id() << " (" << symbol_ << ")" << std::endl
                  << "idxlo: " << ilo << std::endl
                  << "n: " << l << std::endl
                  << "l: " << n << std::endl
                  << "expected l: " << lo_descriptors_[ilo].am.l() << std::endl;
                RTE_THROW(s);
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
    inline void
    add_lo_descriptor(local_orbital_descriptor const& lod__)
    {
        lo_descriptors_.push_back(lod__);
    }

    /// Add atomic radial function to the list.
    inline void
    add_ps_atomic_wf(int n__, angular_momentum am__, std::vector<double> f__, double occ__ = 0.0)
    {
        Spline<double> rwf(radial_grid_, f__);
        auto d = std::sqrt(inner(rwf, rwf, 0, radial_grid_.num_points()));
        if (d < 1e-4) {
            std::stringstream s;
            s << "small norm (" << d << ") of radial atomic pseudo wave-function for n=" << n__
              << " and j=" << am__.j();
            RTE_THROW(s);
        }

        ps_atomic_wfs_.emplace_back(n__, am__, occ__, std::move(rwf));
    }

    /// Return a tuple describing a given atomic radial function
    auto const&
    ps_atomic_wf(int idx__) const
    {
        return ps_atomic_wfs_[idx__];
    }

    /// Add a radial function of beta-projector to a list of functions.
    /** This is the only allowed way to add beta projectors. */
    inline void
    add_beta_radial_function(angular_momentum am__, std::vector<double> beta__)
    {
        if (augment_) {
            RTE_THROW("can't add more beta projectors");
        }
        Spline<double> s(radial_grid_, beta__);
        beta_radial_functions_.push_back(std::make_pair(am__, std::move(s)));
    }

    /// Number of beta-radial functions.
    inline int
    num_beta_radial_functions() const
    {
        return beta_radial_functions_.size();
    }

    /// Return a radial beta function.
    inline auto const&
    beta_radial_function(rf_index idxrf__) const
    {
        return beta_radial_functions_[idxrf__];
    }

    /// Add radial function of the augmentation charge.
    /** Radial functions of beta projectors must be added already. Their total number will be used to
        deterimine the storage size for the radial functions of the augmented charge. */
    inline void
    add_q_radial_function(int idxrf1__, int idxrf2__, int l__, std::vector<double> qrf__)
    {
        /* sanity check */
        if (l__ > 2 * lmax_beta()) {
            std::stringstream s;
            s << "wrong l for Q radial functions of atom type " << label_ << std::endl
              << "current l: " << l__ << std::endl
              << "lmax_beta: " << lmax_beta() << std::endl
              << "maximum allowed l: " << 2 * lmax_beta();

            RTE_THROW(s);
        }

        if (!augment_) {
            /* once we add a Q-radial function, we need to augment the charge */
            augment_ = true;
            /* number of radial beta-functions */
            int nbrf              = num_beta_radial_functions();
            q_radial_functions_l_ = mdarray<Spline<double>, 2>({nbrf * (nbrf + 1) / 2, 2 * lmax_beta() + 1});

            for (int l = 0; l <= 2 * lmax_beta(); l++) {
                for (int idx = 0; idx < nbrf * (nbrf + 1) / 2; idx++) {
                    q_radial_functions_l_(idx, l) = Spline<double>(radial_grid_);
                }
            }
        }

        q_radial_functions_l_(packed_index(idxrf1__, idxrf2__), l__) = Spline<double>(radial_grid_, qrf__);
    }

    /// Return true if this atom type has an augementation charge.
    inline bool
    augment() const
    {
        return augment_;
    }

    /// Set the radial function of the local potential.
    inline std::vector<double>&
    local_potential(std::vector<double> vloc__)
    {
        local_potential_ = vloc__;
        return local_potential_;
    }

    /// Get the radial function of the local potential.
    inline std::vector<double> const&
    local_potential() const
    {
        return local_potential_;
    }

    inline std::vector<double>&
    ps_core_charge_density(std::vector<double> ps_core__)
    {
        ps_core_charge_density_ = ps_core__;
        return ps_core_charge_density_;
    }

    inline std::vector<double> const&
    ps_core_charge_density() const
    {
        return ps_core_charge_density_;
    }

    inline std::vector<double>&
    ps_total_charge_density(std::vector<double> ps_dens__)
    {
        ps_total_charge_density_ = ps_dens__;
        return ps_total_charge_density_;
    }

    inline std::vector<double> const&
    ps_total_charge_density() const
    {
        return ps_total_charge_density_;
    }

    /// Add all-electron PAW wave-function.
    inline void
    add_ae_paw_wf(std::vector<double> f__)
    {
        ae_paw_wfs_.push_back(f__);
    }

    /// Get all-electron PAW wave-function.
    inline std::vector<double> const&
    ae_paw_wf(int i__) const
    {
        return ae_paw_wfs_[i__];
    }

    /// Get the number of all-electron PAW wave-functions.
    inline int
    num_ae_paw_wf() const
    {
        return static_cast<int>(ae_paw_wfs_.size());
    }

    inline void
    add_ps_paw_wf(std::vector<double> f__)
    {
        ps_paw_wfs_.push_back(f__);
    }

    inline std::vector<double> const&
    ps_paw_wf(int i__) const
    {
        return ps_paw_wfs_[i__];
    }

    inline int
    num_ps_paw_wf() const
    {
        return static_cast<int>(ps_paw_wfs_.size());
    }

    inline auto const&
    ae_paw_wfs_array() const
    {
        return ae_paw_wfs_array_;
    }

    inline auto const&
    ps_paw_wfs_array() const
    {
        return ps_paw_wfs_array_;
    }

    inline auto const&
    paw_ae_core_charge_density() const
    {
        return paw_ae_core_charge_density_;
    }

    inline auto&
    paw_ae_core_charge_density(std::vector<double> inp__)
    {
        paw_ae_core_charge_density_ = inp__;
        return paw_ae_core_charge_density_;
    }

    inline auto const&
    paw_wf_occ() const
    {
        return paw_wf_occ_;
    }

    inline auto&
    paw_wf_occ(std::vector<double> inp__)
    {
        paw_wf_occ_ = inp__;
        return paw_wf_occ_;
    }

    /// Return atom type id.
    inline int
    id() const
    {
        return id_;
    }

    /// Return ionic charge (as positive integer).
    inline int
    zn() const
    {
        RTE_ASSERT(zn_ > 0);
        return zn_;
    }

    /// Set ionic charge.
    inline int
    zn(int zn__)
    {
        zn_ = zn__;
        return zn_;
    }

    inline std::string const&
    symbol() const
    {
        return symbol_;
    }

    inline std::string const&
    name() const
    {
        return name_;
    }

    /// Return atomic mass.
    inline double
    mass() const
    {
        return mass_;
    }

    /// Return muffin-tin radius.
    /** This is the last point of the radial grid. */
    inline double
    mt_radius() const
    {
        return radial_grid_.last();
    }

    /// Return number of muffin-tin radial grid points.
    inline int
    num_mt_points() const
    {
        RTE_ASSERT(radial_grid_.num_points() > 0);
        return radial_grid_.num_points();
    }

    inline Radial_grid<double> const&
    radial_grid() const
    {
        RTE_ASSERT(radial_grid_.num_points() > 0);
        return radial_grid_;
    }

    inline Radial_grid<double> const&
    free_atom_radial_grid() const
    {
        return free_atom_radial_grid_;
    }

    inline double
    radial_grid(int ir) const
    {
        return radial_grid_[ir];
    }

    inline double
    free_atom_radial_grid(int ir) const
    {
        return free_atom_radial_grid_[ir];
    }

    inline int
    num_atomic_levels() const
    {
        return static_cast<int>(atomic_levels_.size());
    }

    inline double
    num_core_electrons() const
    {
        return num_core_electrons_;
    }

    inline double
    num_valence_electrons() const
    {
        return num_valence_electrons_;
    }

    /// Get free atom density at i-th point of radial grid.
    inline double
    free_atom_density(const int idx) const
    {
        return free_atom_density_spline_(idx);
    }

    /// Get free atom density at point x.
    inline double
    free_atom_density(double x) const
    {
        return free_atom_density_spline_.at_point(x);
    }

    /// Set the free atom all-electron density.
    inline void
    free_atom_density(std::vector<double> rho__)
    {
        free_atom_density_ = rho__;
    }

    inline void
    aw_default_l(radial_solution_descriptor_set aw_default_l__)
    {
        aw_default_l_ = aw_default_l__;
    }

    inline int
    num_aw_descriptors() const
    {
        return static_cast<int>(aw_descriptors_.size());
    }

    inline auto const&
    aw_descriptor(int l) const
    {
        RTE_ASSERT(l < (int)aw_descriptors_.size());
        return aw_descriptors_[l];
    }

    inline int
    num_lo_descriptors() const
    {
        return (int)lo_descriptors_.size();
    }

    inline auto const&
    lo_descriptor(int idx) const
    {
        return lo_descriptors_[idx];
    }

    inline int
    max_aw_order() const
    {
        return max_aw_order_;
    }

    /// Order of augmented wave radial functions for a given l.
    inline int
    aw_order(int l__) const
    {
        return static_cast<int>(aw_descriptor(l__).size());
    }

    /// Return const reference to the index of radial functions.
    /** The index can be used to determine the total number of radial functions */
    inline auto const&
    indexr() const
    {
        return indexr_;
    }

    inline auto const&
    indexr_wfs() const
    {
        return indexr_wfs_;
    }

    inline auto const&
    indexr_hub() const
    {
        return indexr_hub_;
    }

    inline auto const&
    indexr(int i) const
    {
        // RTE_ASSERT(i >= 0 && i < (int)indexr_.size());
        return indexr_[rf_index(i)];
    }

    inline int
    indexr_by_l_order(int l, int order) const
    {
        return indexr_.index_of(angular_momentum(l), order);
    }

    inline int
    indexr_by_idxlo(int idxlo) const
    {
        return indexr_.index_of(rf_lo_index(idxlo));
    }

    inline auto const&
    indexb() const
    {
        return indexb_;
    }

    inline auto const&
    indexb(int i) const
    {
        // RTE_ASSERT(i >= 0 && i < (int)indexb_.size());
        return indexb_[i];
    }

    inline int
    indexb_by_l_m_order(int l, int m, int order) const
    {
        return indexb_.index_by_l_m_order(l, m, order);
    }

    inline int
    indexb_by_lm_order(int lm, int order) const
    {
        return indexb_.index_by_lm_order(lm, order);
    }

    inline int
    mt_aw_basis_size() const
    {
        return indexb_.size_aw();
    }

    inline int
    mt_lo_basis_size() const
    {
        return indexb_.size_lo();
    }

    /// Total number of muffin-tin basis functions (APW + LO).
    inline int
    mt_basis_size() const
    {
        return indexb_.size();
    }

    /// Total number of radial basis functions.
    inline int
    mt_radial_basis_size() const
    {
        return indexr_.size();
    }

    inline auto const&
    indexb_wfs() const
    {
        return indexb_wfs_;
    }

    /// Return whole index of hubbard basis functions.
    inline auto const&
    indexb_hub() const
    {
        return indexb_hub_;
    }

    inline auto const&
    hubbard_radial_function(int i) const
    {
        return lo_descriptors_hub_[i].f();
    }

    inline void
    set_symbol(const std::string symbol__)
    {
        symbol_ = symbol__;
    }

    inline void
    set_zn(int zn__)
    {
        zn_ = zn__;
    }

    inline void
    set_mass(double mass__)
    {
        mass_ = mass__;
    }

    inline void
    set_configuration(int n, int l, int k, double occupancy, bool core)
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
    inline int
    num_atoms() const
    {
        return static_cast<int>(atom_id_.size());
    }

    /// Return atom ID (global index) by the index of atom within a given type.
    inline int
    atom_id(int idx) const
    {
        return atom_id_[idx];
    }

    /// Add global index of atom to this atom type.
    inline void
    add_atom_id(int atom_id__)
    {
        atom_id_.push_back(atom_id__);
    }

    inline bool
    initialized() const
    {
        return initialized_;
    }

    inline std::string const&
    label() const
    {
        return label_;
    }

    inline std::string const&
    file_name() const
    {
        return file_name_;
    }

    inline void
    d_mtrx_ion(matrix<double> const& d_mtrx_ion__)
    {
        d_mtrx_ion_ = matrix<double>({num_beta_radial_functions(), num_beta_radial_functions()},
                                     mdarray_label("Atom_type::d_mtrx_ion_"));
        copy(d_mtrx_ion__, d_mtrx_ion_);
    }

    inline auto const&
    d_mtrx_ion() const
    {
        return d_mtrx_ion_;
    }

    inline bool
    is_paw() const
    {
        return is_paw_;
    }

    inline bool
    is_paw(bool is_paw__)
    {
        is_paw_ = is_paw__;
        return is_paw_;
    }

    double
    paw_core_energy() const
    {
        return paw_core_energy_;
    }

    double
    paw_core_energy(double paw_core_energy__)
    {
        paw_core_energy_ = paw_core_energy__;
        return paw_core_energy_;
    }

    inline auto const&
    idx_radial_integrals() const
    {
        return idx_radial_integrals_;
    }

    inline auto&
    rf_coef() const
    {
        return rf_coef_;
    }

    inline auto&
    vrf_coef() const
    {
        return vrf_coef_;
    }

    inline auto const&
    parameters() const
    {
        return parameters_;
    }

    inline auto
    f_coefficients(int xi1, int xi2, int s1, int s2) const
    {
        return f_coefficients_(xi1, xi2, s1, s2);
    }

    inline auto const&
    q_radial_function(int idxrf1__, int idxrf2__, int l__) const
    {
        if (idxrf1__ > idxrf2__) {
            std::swap(idxrf1__, idxrf2__);
        }
        /* combined index */
        int ijv = idxrf2__ * (idxrf2__ + 1) / 2 + idxrf1__;

        return q_radial_functions_l_(ijv, l__);
    }

    inline bool
    spin_orbit_coupling() const
    {
        return spin_orbit_coupling_;
    }

    inline bool
    spin_orbit_coupling(bool so__)
    {
        this->spin_orbit_coupling_ = so__;
        return this->spin_orbit_coupling_;
    }

    /// Get the Hubbard correction switch.
    inline bool
    hubbard_correction() const
    {
        return hubbard_correction_;
    }

    /// Set the Hubbard correction switch.
    inline bool
    hubbard_correction(bool ldapu__)
    {
        this->hubbard_correction_ = ldapu__;
        return this->hubbard_correction_;
    }

    /// Compare indices of beta projectors.
    /** Compare the angular, total angular momentum and radial part of the beta projectors,
     *  leaving the m index free. Only useful when spin orbit coupling is included. */
    inline bool
    compare_index_beta_functions(const int xi, const int xj) const
    {
        return ((indexb(xi).am == indexb(xj).am) && (indexb(xi).idxrf == indexb(xj).idxrf));
    }

    /// Return a vector containing all information about the localized atomic
    /// orbitals used to generate the Hubbard subspace.
    inline const auto&
    lo_descriptor_hub() const
    {
        return lo_descriptors_hub_;
    }

    inline auto const&
    lo_descriptor_hub(int idx__) const
    {
        return lo_descriptors_hub_[idx__];
    }

    inline int
    lmax_apw() const
    {
        if (this->lmax_apw_ == -1) {
            return parameters_.cfg().parameters().lmax_apw();
        } else {
            return this->lmax_apw_;
        }
    }

    inline int
    lmmax_apw() const
    {
        return sf::lmmax(this->lmax_apw());
    }

    inline int
    lmax_lo() const
    {
        int lmax{-1};
        for (auto& e : lo_descriptors_) {
            lmax = std::max(lmax, e.am.l());
        }
        return lmax;
    }

    /// Return maximum orbital quantum number for the atomic wave-functions.
    inline int
    lmax_ps_atomic_wf() const
    {
        int lmax{-1};
        for (auto& e : ps_atomic_wfs_) {
            auto l = e.am.l();
            lmax   = std::max(lmax, l);
        }
        return lmax;
    }

    /// Maximum orbital quantum number between all beta-projector radial functions.
    inline int
    lmax_beta() const
    {
        int lmax{-1};

        for (auto& e : beta_radial_functions_) {
            lmax = std::max(lmax, e.first.l());
        }
        return lmax;
    }

    auto const&
    gaunt_coefs() const
    {
        return *gaunt_coefs_;
    }
};

} // namespace sirius

#endif // __ATOM_TYPE_HPP__
