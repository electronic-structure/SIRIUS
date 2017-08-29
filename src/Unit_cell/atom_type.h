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
namespace sirius {

/// A helper class to establish various index mappings for the atomic radial functions.
class radial_functions_index
{
  private:
    /// A list of radial function index descriptors.
    /** This list establishes a mapping \f$ f_{\mu}(r) \leftrightarrow  f_{\ell \nu}(r) \f$ between a
     *  composite index \f$ \mu \f$ of radial functions and
     *  corresponding \f$ \ell \nu \f$ indices, where \f$ \ell \f$ is the orbital quantum number and
     *  \f$ \nu \f$ is the order of radial function for a given \f$ \ell \f$. */
    std::vector<radial_function_index_descriptor> radial_function_index_descriptors_;

    mdarray<int, 2> index_by_l_order_;

    mdarray<int, 1> index_by_idxlo_;

    /// Number of radial functions for each angular momentum quantum number.
    std::vector<int> num_rf_;

    /// Number of local orbitals for each angular momentum quantum number.
    std::vector<int> num_lo_;

    // Maximum number of radial functions across all angular momentums.
    int max_num_rf_;

    int lmax_aw_;

    int lmax_lo_;

    int lmax_;

  public:
    void init(std::vector<local_orbital_descriptor> const& lo_descriptors__)
    {
        std::vector<radial_solution_descriptor_set> aw_descriptors;
        init(aw_descriptors, lo_descriptors__);
    }

    void init(std::vector<radial_solution_descriptor_set> const& aw_descriptors,
              std::vector<local_orbital_descriptor> const& lo_descriptors)
    {
        lmax_aw_ = static_cast<int>(aw_descriptors.size()) - 1;
        lmax_lo_ = -1;
        for (size_t idxlo = 0; idxlo < lo_descriptors.size(); idxlo++) {
            int l    = lo_descriptors[idxlo].l;
            lmax_lo_ = std::max(lmax_lo_, l);
        }

        lmax_ = std::max(lmax_aw_, lmax_lo_);

        num_rf_ = std::vector<int>(lmax_ + 1, 0);
        num_lo_ = std::vector<int>(lmax_ + 1, 0);

        max_num_rf_ = 0;

        radial_function_index_descriptors_.clear();

        for (int l = 0; l <= lmax_aw_; l++) {
            assert(aw_descriptors[l].size() <= 3);

            for (size_t order = 0; order < aw_descriptors[l].size(); order++) {
                radial_function_index_descriptors_.push_back(radial_function_index_descriptor(l, num_rf_[l]));
                num_rf_[l]++;
            }
        }

        for (int idxlo = 0; idxlo < static_cast<int>(lo_descriptors.size()); idxlo++) {
            int l = lo_descriptors[idxlo].l;
            radial_function_index_descriptors_.push_back(
                radial_function_index_descriptor(l, lo_descriptors[idxlo].total_angular_momentum, num_rf_[l], idxlo));
            num_rf_[l]++;
            num_lo_[l]++;
        }

        for (int l = 0; l <= lmax_; l++) {
            max_num_rf_ = std::max(max_num_rf_, num_rf_[l]);
        }

        index_by_l_order_ = mdarray<int, 2>(lmax_ + 1, max_num_rf_);

        if (lo_descriptors.size()) {
            index_by_idxlo_ = mdarray<int, 1>(lo_descriptors.size());
        }

        for (int i = 0; i < (int)radial_function_index_descriptors_.size(); i++) {
            int l     = radial_function_index_descriptors_[i].l;
            int order = radial_function_index_descriptors_[i].order;
            int idxlo = radial_function_index_descriptors_[i].idxlo;
            index_by_l_order_(l, order) = i;
            if (idxlo >= 0)
                index_by_idxlo_(idxlo) = i;
        }
    }

    inline int size() const
    {
        return (int)radial_function_index_descriptors_.size();
    }

    inline radial_function_index_descriptor const& operator[](int i) const
    {
        assert(i >= 0 && i < (int)radial_function_index_descriptors_.size());
        return radial_function_index_descriptors_[i];
    }

    inline int index_by_l_order(int l, int order) const
    {
        return index_by_l_order_(l, order);
    }

    inline int index_by_idxlo(int idxlo) const
    {
        return index_by_idxlo_(idxlo);
    }

    /// Number of radial functions for a given orbital quantum number.
    inline int num_rf(int l) const
    {
        assert(l >= 0 && l < (int)num_rf_.size());
        return num_rf_[l];
    }

    /// Number of local orbitals for a given orbital quantum number.
    inline int num_lo(int l) const
    {
        assert(l >= 0 && l < (int)num_lo_.size());
        return num_lo_[l];
    }

    /// Maximum possible number of radial functions for an orbital quantum number.
    inline int max_num_rf() const
    {
        return max_num_rf_;
    }

    inline int lmax() const
    {
        return lmax_;
    }

    inline int lmax_lo() const
    {
        return lmax_lo_;
    }
};

class basis_functions_index
{
  private:
    std::vector<basis_function_index_descriptor> basis_function_index_descriptors_;

    mdarray<int, 2> index_by_lm_order_;

    mdarray<int, 1> index_by_idxrf_;

    /// number of augmented wave basis functions
    int size_aw_;

    /// number of local orbital basis functions
    int size_lo_;

  public:
    basis_functions_index()
        : size_aw_(0)
        , size_lo_(0)
    {
    }

    void init(radial_functions_index& indexr)
    {
        basis_function_index_descriptors_.clear();

        index_by_idxrf_ = mdarray<int, 1>(indexr.size());

        for (int idxrf = 0; idxrf < indexr.size(); idxrf++) {
            int l     = indexr[idxrf].l;
            int order = indexr[idxrf].order;
            int idxlo = indexr[idxrf].idxlo;

            index_by_idxrf_(idxrf) = (int)basis_function_index_descriptors_.size();

            for (int m = -l; m <= l; m++)
                basis_function_index_descriptors_.push_back(
                    basis_function_index_descriptor(l, m, indexr[idxrf].j, order, idxlo, idxrf));
        }
        index_by_lm_order_ = mdarray<int, 2>(Utils::lmmax(indexr.lmax()), indexr.max_num_rf());

        for (int i = 0; i < (int)basis_function_index_descriptors_.size(); i++) {
            int lm    = basis_function_index_descriptors_[i].lm;
            int order = basis_function_index_descriptors_[i].order;
            index_by_lm_order_(lm, order) = i;

            // get number of aw basis functions
            if (basis_function_index_descriptors_[i].idxlo < 0)
                size_aw_ = i + 1;
        }

        size_lo_ = (int)basis_function_index_descriptors_.size() - size_aw_;

        assert(size_aw_ >= 0);
        assert(size_lo_ >= 0);
    }

    /// Return total number of MT basis functions.
    inline int size() const
    {
        return static_cast<int>(basis_function_index_descriptors_.size());
    }

    inline int size_aw() const
    {
        return size_aw_;
    }

    inline int size_lo() const
    {
        return size_lo_;
    }

    inline int index_by_l_m_order(int l, int m, int order) const
    {
        return index_by_lm_order_(Utils::lm_by_l_m(l, m), order);
    }

    inline int index_by_lm_order(int lm, int order) const
    {
        return index_by_lm_order_(lm, order);
    }

    inline int index_by_idxrf(int idxrf) const
    {
        return index_by_idxrf_(idxrf);
    }

    inline basis_function_index_descriptor const& operator[](int i) const
    {
        assert(i >= 0 && i < (int)basis_function_index_descriptors_.size());
        return basis_function_index_descriptors_[i];
    }
};

class Atom_type
{
  private:
    Simulation_parameters const& parameters_;

    /// Unique id of atom type in the range [0, \f$ N_{types} \f$).
    int id_{-1};

    /// Unique string label for the atom type.
    std::string label_;

    /// Chemical element symbol.
    std::string symbol_;

    /// Chemical element name.
    std::string name_;

    /// Nucleus charge, treated as positive(!) integer.
    int zn_{0};

    /// Atom mass.
    double mass_{0};

    /// Muffin-tin radius.
    double mt_radius_{0};

    /// Number of muffin-tin points.
    int num_mt_points_{0};

    /// Beginning of the radial grid.
    double radial_grid_origin_{0};

    /// List of atomic levels.
    std::vector<atomic_level_descriptor> atomic_levels_;

    /// Number of core electrons.
    double num_core_electrons_{0};

    /// Number of valence electrons.
    double num_valence_electrons_{0};

    /// Default augmented wave configuration.
    radial_solution_descriptor_set aw_default_l_;

    /// augmented wave configuration for specific l
    std::vector<radial_solution_descriptor_set> aw_specific_l_;

    /// list of radial descriptor sets used to construct augmented waves
    std::vector<radial_solution_descriptor_set> aw_descriptors_;

    /// list of radial descriptor sets used to construct local orbitals
    std::vector<local_orbital_descriptor> lo_descriptors_;

    /// Maximum number of AW radial functions across angular momentums.
    int max_aw_order_{0};

    int offset_lo_{-1};

    radial_functions_index indexr_;

    basis_functions_index indexb_;

    pseudopotential_descriptor pp_desc_;

    /// Inverse of (Q_{\xi \xi'j}^{-1} + beta_pw^{H}_{\xi} * beta_pw_{xi'})
    /** Used in Chebyshev iterative solver as a block-diagonal preconditioner */
    matrix<double_complex> p_mtrx_;

    /// f_coefficients defined in Ref. PRB 71 115106 Eq.9 only
    /// valid when SO interactions are on
    mdarray<double_complex, 4> f_coefficients_;

    std::vector<int> atom_id_;

    std::string file_name_;

    mdarray<int, 2> idx_radial_integrals_;

    mutable mdarray<double, 3> rf_coef_;
    mutable mdarray<double, 3> vrf_coef_;

    mdarray<Spline<double>, 1> beta_rf_;
    mdarray<Spline<double>, 2> q_rf_;

    bool initialized_{false};

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
    /// Radial grid.
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
        , mt_radius_(2.0)
        , num_mt_points_(2000 + zn__ * 50)
        , atomic_levels_(levels__)
    {
        radial_grid_ = Radial_grid_factory<double>(grid_type__, num_mt_points_, 1e-6 / zn_, 20.0 + 0.25 * zn_);
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

    inline void set_radial_grid(int num_points__ = -1, double const* points__ = nullptr)
    {
        if (num_mt_points_ == 0) {
            TERMINATE("number of muffin-tin points is zero");
        }
        if (num_points__ < 0 && points__ == nullptr) {
            /* create default exponential grid */
            radial_grid_ = Radial_grid_exp<double>(num_mt_points_, radial_grid_origin_, mt_radius_);
        } else {
            assert(num_points__ == num_mt_points_);
            radial_grid_ = Radial_grid_ext<double>(num_points__, points__);
        }
        if (parameters_.processing_unit() == GPU) {
#ifdef __GPU
            radial_grid_.copy_to_device();
#endif
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

    inline double mt_radius() const
    {
        return mt_radius_;
    }

    inline int num_mt_points() const
    {
        assert(num_mt_points_ > 0);
        return num_mt_points_;
    }

    inline Radial_grid<double> const& radial_grid() const
    {
        assert(num_mt_points_ > 0);
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
        return free_atom_density_spline_[idx];
    }

    inline double free_atom_density(double x) const
    {
        return free_atom_density_spline_(x);
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

    inline pseudopotential_descriptor const& pp_desc() const
    {
        return pp_desc_;
    }

    inline pseudopotential_descriptor& pp_desc()
    {
        return pp_desc_;
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

    inline void
    set_mt_radius(double mt_radius__) // TODO: this can cause inconsistency with radial_grid; remove this method
                                      // mt_radius should always be the last point of radial_grid
    {
        mt_radius_ = mt_radius__;
    }

    inline void set_num_mt_points(int num_mt_points__)
    {
        num_mt_points_ = num_mt_points__;
    }

    inline void set_radial_grid_origin(double radial_grid_origin__)
    {
        radial_grid_origin_ = radial_grid_origin__;
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

    inline void set_d_mtrx_ion(matrix<double>& d_mtrx_ion__)
    {
        pp_desc_.d_mtrx_ion = matrix<double>(d_mtrx_ion__.size(0), d_mtrx_ion__.size(1));
        d_mtrx_ion__ >> pp_desc_.d_mtrx_ion;
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
    inline Spline<double> const& beta_rf(int idxrf__) const
    {
        return beta_rf_[idxrf__];
    }

    inline Spline<double> const& q_rf(int idx__, int l__) const
    {
        return q_rf_(idx__, l__);
    }

    /// compare the angular, total angular momentum and radial part of
    /// the beta projectors, leaving the m index free. Only useful
    /// when spin orbit coupling is included.
    inline bool compare_index_beta_functions(const int xi, const int xj) const
    {
        return ((indexb(xi).l == indexb(xj).l) && (indexb(xi).idxrf == indexb(xj).idxrf) &&
                (std::abs(indexb(xi).j - indexb(xj).j) < 1e-8));
    }

  private:
    void generate_f_coefficients(void);
    inline double ClebschGordan(const int l, const double j, const double m, const int spin);
    inline double_complex calculate_U_sigma_m(const int l, const double j, const int mj, const int m, const int sigma);
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

    /* set default radial grid if it was not done by user */
    if (radial_grid_.num_points() == 0) {
        set_radial_grid();
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
        local_orbital_descriptor lod;
        for (int i = 0; i < pp_desc_.num_beta_radial_functions; i++) {
            /* think of |beta> functions as of local orbitals */
            lod.l = pp_desc_.beta_l[i];
            if (pp_desc_.spin_orbit_coupling)
                lod.total_angular_momentum = pp_desc_.beta_j[i];
            lo_descriptors_.push_back(lod);
        }
    }

    /* initialize index of radial functions */
    indexr_.init(aw_descriptors_, lo_descriptors_);

    /* initialize index of muffin-tin basis functions */
    indexb_.init(indexr_);

    if (!parameters_.full_potential()) {
        /* number of radial beta-functions */
        int nbrf = mt_radial_basis_size();
        /* maximum l of beta-projectors */
        int lmax_beta = indexr().lmax();
        /* interpolate beta radial functions */
        beta_rf_ = mdarray<Spline<double>, 1>(mt_radial_basis_size());
        for (int idxrf = 0; idxrf < nbrf; idxrf++) {
            beta_rf_[idxrf] = Spline<double>(radial_grid());
            int nr          = pp_desc().num_beta_radial_points[idxrf];
            for (int ir = 0; ir < nr; ir++) {
                beta_rf_[idxrf][ir] = pp_desc().beta_radial_functions(ir, idxrf);
            }
            beta_rf_[idxrf].interpolate();
        }
        /* interpolate Q-operator radial functions */
        if (pp_desc().augment) {
            q_rf_ = mdarray<Spline<double>, 2>(nbrf * (nbrf + 1) / 2, 2 * lmax_beta + 1);
            #pragma omp parallel for
            for (int idx = 0; idx < nbrf * (nbrf + 1) / 2; idx++) {
                for (int l = 0; l <= 2 * lmax_beta; l++) {
                    q_rf_(idx, l) = Spline<double>(radial_grid());
                    for (int ir = 0; ir < num_mt_points(); ir++) {
                        q_rf_(idx, l)[ir] = pp_desc().q_radial_functions_l(ir, idx, l);
                    }
                    q_rf_(idx, l).interpolate();
                }
            }
        }
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
#ifdef __GPU
        idx_radial_integrals_.allocate(memory_t::device);
        idx_radial_integrals_.copy_to_device();
        rf_coef_ = mdarray<double, 3>(num_mt_points_, 4, indexr().size(), memory_t::host_pinned | memory_t::device);
        vrf_coef_ =
            mdarray<double, 3>(num_mt_points_, 4, lmmax_pot * indexr().size() * (parameters_.num_mag_dims() + 1),
                               memory_t::host_pinned | memory_t::device);
#else
        TERMINATE_NO_GPU
#endif
    }

    if (this->pp_desc().spin_orbit_coupling)
        this->generate_f_coefficients();
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

        b(0) = free_atom_density_spline_[irmt];
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
            free_atom_density_spline_[i] =
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
    printf("mt_radius      : %f\n", mt_radius_);
    printf("num_mt_points  : %i\n", num_mt_points_);
    printf("grid_origin    : %f\n", radial_grid_[0]);
    printf("grid_name      : %s\n", radial_grid_.name().c_str());
    printf("\n");
    printf("number of core electrons    : %f\n", num_core_electrons_);
    printf("number of valence electrons : %f\n", num_valence_electrons_);

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

    num_mt_points_ = parser["pseudo_potential"]["header"]["mesh_size"];

    auto rgrid = parser["pseudo_potential"]["radial_grid"].get<std::vector<double>>();
    if (static_cast<int>(rgrid.size()) != num_mt_points_) {
        TERMINATE("wrong mesh size");
    }

    pp_desc_.vloc = parser["pseudo_potential"]["local_potential"].get<std::vector<double>>();

    pp_desc_.core_charge_density =
        parser["pseudo_potential"].value("core_charge_density", std::vector<double>(rgrid.size(), 0));

    pp_desc_.total_charge_density = parser["pseudo_potential"]["total_charge_density"].get<std::vector<double>>();

    if (pp_desc_.vloc.size() != rgrid.size() || pp_desc_.core_charge_density.size() != rgrid.size() ||
        pp_desc_.total_charge_density.size() != rgrid.size()) {
        std::cout << pp_desc_.vloc.size() << " " << pp_desc_.core_charge_density.size() << " "
                  << pp_desc_.total_charge_density.size() << std::endl;
        TERMINATE("wrong array size");
    }

    mt_radius_ = rgrid.back();

    set_radial_grid(num_mt_points_, rgrid.data());

    if (parser["pseudo_potential"]["header"].count("spin_orbit"))
        pp_desc_.spin_orbit_coupling = parser["pseudo_potential"]["header"]["spin_orbit"];

    pp_desc_.num_beta_radial_functions = parser["pseudo_potential"]["header"]["number_of_proj"];

    pp_desc_.beta_radial_functions = mdarray<double, 2>(num_mt_points_, pp_desc_.num_beta_radial_functions);
    pp_desc_.beta_radial_functions.zero();

    pp_desc_.num_beta_radial_points.resize(pp_desc_.num_beta_radial_functions);
    pp_desc_.beta_l.resize(pp_desc_.num_beta_radial_functions);

    if (pp_desc_.spin_orbit_coupling)
        pp_desc_.beta_j.resize(pp_desc_.num_beta_radial_functions);

    local_orbital_descriptor lod;
    int lmax_beta{0};
    for (int i = 0; i < pp_desc_.num_beta_radial_functions; i++) {
        auto beta = parser["pseudo_potential"]["beta_projectors"][i]["radial_function"].get<std::vector<double>>();
        if (static_cast<int>(beta.size()) > num_mt_points_) {
            std::stringstream s;
            s << "wrong size of beta functions for atom type " << symbol_ << " (label: " << label_ << ")" << std::endl
              << "size of beta radial functions in the file: " << beta.size() << std::endl
              << "radial grid size: " << num_mt_points_;
            TERMINATE(s);
        }
        pp_desc_.num_beta_radial_points[i] = static_cast<int>(beta.size());
        std::memcpy(&pp_desc_.beta_radial_functions(0, i), &beta[0],
                    pp_desc_.num_beta_radial_points[i] * sizeof(double));

        pp_desc_.beta_l[i] = parser["pseudo_potential"]["beta_projectors"][i]["angular_momentum"];
        lmax_beta          = std::max(lmax_beta, pp_desc_.beta_l[i]);
        if (pp_desc_.spin_orbit_coupling) {
            pp_desc_.beta_j[i] = parser["pseudo_potential"]["beta_projectors"][i]["total_angular_momentum"];
        }
    }

    assert(lmax_beta >= 0);

    pp_desc_.d_mtrx_ion = mdarray<double, 2>(pp_desc_.num_beta_radial_functions, pp_desc_.num_beta_radial_functions);
    pp_desc_.d_mtrx_ion.zero();
    auto dion = parser["pseudo_potential"]["D_ion"].get<std::vector<double>>();

    for (int i = 0; i < pp_desc_.num_beta_radial_functions; i++) {
        for (int j = 0; j < pp_desc_.num_beta_radial_functions; j++) {
            pp_desc_.d_mtrx_ion(i, j) = dion[j * pp_desc_.num_beta_radial_functions + i];
        }
    }

    if (parser["pseudo_potential"].count("augmentation")) {
        pp_desc_.augment              = true;
        pp_desc_.q_radial_functions_l = mdarray<double, 3>(
            num_mt_points_, pp_desc_.num_beta_radial_functions * (pp_desc_.num_beta_radial_functions + 1) / 2,
            2 * lmax_beta + 1);
        pp_desc_.q_radial_functions_l.zero();

        for (size_t k = 0; k < parser["pseudo_potential"]["augmentation"].size(); k++) {
            int i    = parser["pseudo_potential"]["augmentation"][k]["i"];
            int j    = parser["pseudo_potential"]["augmentation"][k]["j"];
            int idx  = j * (j + 1) / 2 + i;
            int l    = parser["pseudo_potential"]["augmentation"][k]["angular_momentum"];
            auto qij = parser["pseudo_potential"]["augmentation"][k]["radial_function"].get<std::vector<double>>();
            if ((int)qij.size() != num_mt_points_) {
                TERMINATE("wrong size of qij");
            }

            std::memcpy(&pp_desc_.q_radial_functions_l(0, idx, l), &qij[0], num_mt_points_ * sizeof(double));
        }
    }

    /* read starting wave functions ( UPF CHI ) */
    if (parser["pseudo_potential"].count("atomic_wave_functions")) {
        size_t nwf = parser["pseudo_potential"]["atomic_wave_functions"].size();
        for (size_t k = 0; k < nwf; k++) {
            std::pair<int, std::vector<double>> wf;
            wf.second =
                parser["pseudo_potential"]["atomic_wave_functions"][k]["radial_function"].get<std::vector<double>>();

            if ((int)wf.second.size() != num_mt_points_) {
                std::stringstream s;
                s << "wrong size of atomic functions for atom type " << symbol_ << " (label: " << label_ << ")"
                  << std::endl
                  << "size of atomic radial functions in the file: " << wf.second.size() << std::endl
                  << "radial grid size: " << num_mt_points_;
                TERMINATE(s);
            }
            wf.first = parser["pseudo_potential"]["atomic_wave_functions"][k]["angular_momentum"];
            pp_desc_.atomic_pseudo_wfs_.push_back(wf);

            ///* read occupation of the function */
            // double occ = parser["pseudo_potential"]["atomic_wave_functions"][k]["occupation"];
            // pp_desc_.atomic_pseudo_wfs_occ_.push_back(occ);
        }
    }
}

inline void Atom_type::read_pseudo_paw(json const& parser)
{
    pp_desc_.is_paw = true;

    /* read core energy */
    pp_desc_.core_energy = parser["pseudo_potential"]["header"]["paw_core_energy"];

    /* cutoff index */
    pp_desc_.cutoff_radius_index = parser["pseudo_potential"]["header"]["cutoff_radius_index"];

    /* read augmentation multipoles and integrals */
    // pp_desc_.aug_integrals = parser["pseudo_potential"]["paw_data"]["aug_integrals"].get<std::vector<double>>();

    // pp_desc_.aug_multopoles = parser["pseudo_potential"]["paw_data"]["aug_multipoles"].get<std::vector<double>>();

    /* read core density and potential */
    pp_desc_.all_elec_core_charge =
        parser["pseudo_potential"]["paw_data"]["ae_core_charge_density"].get<std::vector<double>>();

    pp_desc_.all_elec_loc_potential =
        parser["pseudo_potential"]["paw_data"]["ae_local_potential"].get<std::vector<double>>();

    /* read occupations */
    pp_desc_.occupations = parser["pseudo_potential"]["paw_data"]["occupations"].get<std::vector<double>>();

    /* setups for reading AE and PS basis wave functions */
    int num_wfc = pp_desc_.num_beta_radial_functions;

    pp_desc_.all_elec_wfc = mdarray<double, 2>(num_mt_points_, num_wfc);
    pp_desc_.pseudo_wfc   = mdarray<double, 2>(num_mt_points_, num_wfc);

    pp_desc_.all_elec_wfc.zero();
    pp_desc_.pseudo_wfc.zero();

    /* read ae and ps wave functions */
    for (int i = 0; i < num_wfc; i++) {
        /* read ae wave func */
        auto wfc = parser["pseudo_potential"]["paw_data"]["ae_wfc"][i]["radial_function"].get<std::vector<double>>();

        if ((int)wfc.size() > num_mt_points_) {
            std::stringstream s;
            s << "wrong size of beta functions for atom type " << symbol_ << " (label: " << label_ << ")" << std::endl
              << "size of beta radial functions in the file: " << wfc.size() << std::endl
              << "radial grid size: " << num_mt_points_;
            TERMINATE(s);
        }

        std::memcpy(&pp_desc_.all_elec_wfc(0, i), wfc.data(), (pp_desc_.cutoff_radius_index) * sizeof(double));

        /* read ps wave func */
        wfc.clear();

        wfc = parser["pseudo_potential"]["paw_data"]["ps_wfc"][i]["radial_function"].get<std::vector<double>>();

        if ((int)wfc.size() > num_mt_points_) {
            std::stringstream s;
            s << "wrong size of beta functions for atom type " << symbol_ << " (label: " << label_ << ")" << std::endl
              << "size of beta radial functions in the file: " << wfc.size() << std::endl
              << "radial grid size: " << num_mt_points_;
            TERMINATE(s);
        }
        std::memcpy(&pp_desc_.pseudo_wfc(0, i), wfc.data(), (pp_desc_.cutoff_radius_index) * sizeof(double));
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
        name_               = parser["name"];
        symbol_             = parser["symbol"];
        mass_               = parser["mass"];
        zn_                 = parser["number"];
        radial_grid_origin_ = parser["rmin"];
        mt_radius_          = parser["rmt"];
        num_mt_points_      = parser["nrmt"];

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
	    }
            else {
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

    int result = 0;
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
    if (!this->pp_desc().spin_orbit_coupling)
        return;

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

} // namespace

#endif // __ATOM_TYPE_H__
