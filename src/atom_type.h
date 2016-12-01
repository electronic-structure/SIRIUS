// Copyright (c) 2013-2016 Anton Kozhevnikov, Thomas Schulthess
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

#include "descriptors.h"
#include "vector3d.hpp"
#include "utils.h"
#include "radial_grid.h"
#include "radial_solver.h"
#include "xc_functional.h"
#include "simulation_parameters.h"

namespace sirius {

class radial_functions_index
{
    private: 

        std::vector<radial_function_index_descriptor> radial_function_index_descriptors_;

        mdarray<int, 2> index_by_l_order_;

        mdarray<int, 1> index_by_idxlo_;

        /// number of radial functions for each angular momentum quantum number
        std::vector<int> num_rf_;

        /// number of local orbitals for each angular momentum quantum number
        std::vector<int> num_lo_;

        // maximum number of radial functions across all angular momentums
        int max_num_rf_;

        int lmax_aw_;

        int lmax_lo_;

        int lmax_;
    
    public:

        void init(const std::vector<local_orbital_descriptor>& lo_descriptors__)
        {
            std::vector<radial_solution_descriptor_set> aw_descriptors;
            init(aw_descriptors, lo_descriptors__);
        }

        void init(const std::vector<radial_solution_descriptor_set>& aw_descriptors, 
                  const std::vector<local_orbital_descriptor>& lo_descriptors)
        {
            lmax_aw_ = static_cast<int>(aw_descriptors.size()) - 1;
            lmax_lo_ = -1;
            for (size_t idxlo = 0; idxlo < lo_descriptors.size(); idxlo++)
            {
                int l = lo_descriptors[idxlo].l;
                lmax_lo_ = std::max(lmax_lo_, l);
            }

            lmax_ = std::max(lmax_aw_, lmax_lo_);

            num_rf_ = std::vector<int>(lmax_ + 1, 0);
            num_lo_ = std::vector<int>(lmax_ + 1, 0);
            
            max_num_rf_ = 0;

            radial_function_index_descriptors_.clear();

            for (int l = 0; l <= lmax_aw_; l++)
            {
                assert(aw_descriptors[l].size() <= 3);

                for (size_t order = 0; order < aw_descriptors[l].size(); order++)
                {
                    radial_function_index_descriptors_.push_back(radial_function_index_descriptor(l, num_rf_[l]));
                    num_rf_[l]++;
                }
            }

            for (int idxlo = 0; idxlo < (int)lo_descriptors.size(); idxlo++)
            {
                int l = lo_descriptors[idxlo].l;
                radial_function_index_descriptors_.push_back(radial_function_index_descriptor(l, num_rf_[l], idxlo));
                num_rf_[l]++;
                num_lo_[l]++;
            }

            for (int l = 0; l <= lmax_; l++) max_num_rf_ = std::max(max_num_rf_, num_rf_[l]);

            index_by_l_order_ = mdarray<int, 2>(lmax_ + 1, max_num_rf_);

            if (lo_descriptors.size())
            {
                index_by_idxlo_ = mdarray<int, 1>(lo_descriptors.size());
            }

            for (int i = 0; i < (int)radial_function_index_descriptors_.size(); i++)
            {
                int l = radial_function_index_descriptors_[i].l;
                int order = radial_function_index_descriptors_[i].order;
                int idxlo = radial_function_index_descriptors_[i].idxlo;
                index_by_l_order_(l, order) = i;
                if (idxlo >= 0) index_by_idxlo_(idxlo) = i; 
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

        basis_functions_index() : size_aw_(0), size_lo_(0)
        {
        }
        
        void init(radial_functions_index& indexr)
        {
            basis_function_index_descriptors_.clear();

            index_by_idxrf_ = mdarray<int, 1>(indexr.size());

            for (int idxrf = 0; idxrf < indexr.size(); idxrf++)
            {
                int l = indexr[idxrf].l;
                int order = indexr[idxrf].order;
                int idxlo = indexr[idxrf].idxlo;

                index_by_idxrf_(idxrf) = (int)basis_function_index_descriptors_.size();

                for (int m = -l; m <= l; m++)
                    basis_function_index_descriptors_.push_back(basis_function_index_descriptor(l, m, order, idxlo, idxrf));
            }

            index_by_lm_order_ = mdarray<int, 2>(Utils::lmmax(indexr.lmax()), indexr.max_num_rf());

            for (int i = 0; i < (int)basis_function_index_descriptors_.size(); i++)
            {
                int lm = basis_function_index_descriptors_[i].lm;
                int order = basis_function_index_descriptors_[i].order;
                index_by_lm_order_(lm, order) = i;
                
                // get number of aw basis functions
                if (basis_function_index_descriptors_[i].idxlo < 0) size_aw_ = i + 1;
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
        
        /// maximum number of aw radial functions across angular momentums
        int max_aw_order_{0};

        int offset_lo_{-1};

        radial_functions_index indexr_;
        
        basis_functions_index indexb_;

        pseudopotential_descriptor pp_desc_;

        //uspp_descriptor uspp_;

        //PAW_descriptor paw_;

        /// Inverse of (Q_{\xi \xi'j}^{-1} + beta_pw^{H}_{\xi} * beta_pw_{xi'})
        /** Used in Chebyshev iterative solver as a block-diagonal preconditioner */
        matrix<double_complex> p_mtrx_;

        std::vector<int> atom_id_;

        std::string file_name_;

        mdarray<int, 2> idx_radial_integrals_;

        mutable mdarray<double, 3> rf_coef_;
        mutable mdarray<double, 3> vrf_coef_;

        bool initialized_{false};
       
        void read_input_core(json const& parser);

        void read_input_aw(json const& parser);

        void read_input_lo(json const& parser);

        void read_pseudo_uspp(json const& parser);

        void read_pseudo_paw(json const& parser);

        void read_input(const std::string& fname);
    
        void init_aw_descriptors(int lmax)
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
        Radial_grid radial_grid_;

        /// Density of a free atom.
        Spline<double> free_atom_density_spline_;

        std::vector<double> free_atom_density_;

        /// Radial grid of a free atom.
        Radial_grid free_atom_radial_grid_;

    public:
        
        Atom_type(Simulation_parameters const&          parameters__,
                  std::string                           symbol__, 
                  std::string                           name__, 
                  int                                   zn__, 
                  double                                mass__, 
                  std::vector<atomic_level_descriptor>& levels__,
                  radial_grid_t                         grid_type__)
            : parameters_(parameters__),
              symbol_(symbol__), 
              name_(name__), 
              zn_(zn__), 
              mass_(mass__), 
              mt_radius_(2.0), 
              num_mt_points_(2000 + zn__ * 50), 
              atomic_levels_(levels__)
        {
            radial_grid_ = Radial_grid(grid_type__, num_mt_points_, 1e-6 / zn_, 20.0 + 0.25 * zn_); 
        }
 
        Atom_type(Simulation_parameters const& parameters__,
                  int                          id__, 
                  std::string                  label__, 
                  std::string                  file_name__)
            : parameters_(parameters__),
              id_(id__), 
              label_(label__),
              file_name_(file_name__)
        {
        }

        Atom_type(Atom_type&& src) = default;

        void init(int offset_lo__);

        void set_radial_grid(int num_points__ = -1, double const* points__ = nullptr)
        {
            if (num_mt_points_ == 0) {
                TERMINATE("number of muffin-tin points is zero");
            }
            if (num_points__ < 0 && points__ == nullptr) {
                /* create default exponential grid */
                radial_grid_ = Radial_grid(lin_exp_grid, num_mt_points_, radial_grid_origin_, mt_radius_); 
            } else {
                assert(num_points__ == num_mt_points_);
                radial_grid_ = Radial_grid(num_points__, points__);
            }
            if (parameters_.processing_unit() == GPU) {
                #ifdef __GPU
                radial_grid_.copy_to_device();
                #endif
            }
        }

        /// Add augmented-wave descriptor.
        void add_aw_descriptor(int n, int l, double enu, int dme, int auto_enu)
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
            
            rsd.l = l;
            rsd.dme = dme;
            rsd.enu = enu;
            rsd.auto_enu = auto_enu;
            aw_descriptors_[l].push_back(rsd);
        }

        /// Add local orbital descriptor
        void add_lo_descriptor(int ilo, int n, int l, double enu, int dme, int auto_enu)
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
                      << "expected l: " <<  lo_descriptors_[ilo].l << std::endl;
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
            
            rsd.l = l;
            rsd.dme = dme;
            rsd.enu = enu;
            rsd.auto_enu = auto_enu;
            lo_descriptors_[ilo].rsd_set.push_back(rsd);
        }

        void add_lo_descriptor(local_orbital_descriptor const& lod__)
        {
            lo_descriptors_.push_back(lod__);
        }

        void init_free_atom(bool smooth);

        void print_info() const;

        inline int id() const
        {
            return id_;
        }

        inline int zn() const
        {
            assert(zn_ > 0);
            return zn_;
        }

        std::string const& symbol() const
        { 
            return symbol_;
        }

        std::string const& name() const
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

        inline Radial_grid const& radial_grid() const
        {
            assert(num_mt_points_ > 0);
            assert(radial_grid_.num_points() > 0);
            return radial_grid_;
        }

        inline Radial_grid const& free_atom_radial_grid() const
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

        //inline double free_atom_potential(const int idx) const
        //{
        //    return free_atom_potential_[idx];
        //}

        //inline double free_atom_potential(double x) const
        //{
        //    return free_atom_potential_(x);
        //}

        //Spline<double> const& free_atom_potential() const
        //{
        //    return free_atom_potential_;
        //}

        inline int num_aw_descriptors() const
        {
            return (int)aw_descriptors_.size();
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
            for (int i = 0; i < free_atom_radial_grid().num_points(); i++)
            {
                if (free_atom_radial_grid(i) > mt_radius()) return i - 1;
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
        
        inline void set_mt_radius(double mt_radius__) // TODO: this can cause inconsistency with radial_grid; remove this method
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
            level.n = n;
            level.l = l;
            level.k = k;
            level.occupancy = occupancy;
            level.core = core;
            atomic_levels_.push_back(level);
        }

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

        void set_free_atom_radial_grid(int num_points__, double const* points__)
        {
            if (num_points__ <= 0) TERMINATE("wrong number of radial points");
            free_atom_radial_grid_ = Radial_grid(num_points__, points__);
        }

        void set_free_atom_density(int num_points__, double const* dens__)
        {
            free_atom_density_.resize(num_points__);
            std::memcpy(free_atom_density_.data(), dens__, num_points__ * sizeof(double));
        }
};

};

#endif // __ATOM_TYPE_H__

