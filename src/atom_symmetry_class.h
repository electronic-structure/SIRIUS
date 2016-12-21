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

/** \file atom_symmetry_class.h
 *   
 *  \brief Contains declaration and partial implementation of sirius::Atom_symmetry_class class.
 */

#ifndef __ATOM_SYMMETRY_CLASS_H__
#define __ATOM_SYMMETRY_CLASS_H__

#include "sirius_io.h"
#include "atom_type.h"
#include "communicator.hpp"

namespace sirius {

/// Data and methods specific to the symmetry class of the atom.
/** Atoms transforming into each other under symmetry opeartions belong to the same symmetry class. They have the
 *  same spherical part of the on-site potential and, as a consequence, the same radial functions. 
 */
class Atom_symmetry_class
{
    private:
        
        /// Symmetry class id in the range [0, N_class).
        int id_;

        /// List of atoms of this class.
        std::vector<int> atom_id_;
        
        /// Pointer to atom type.
        Atom_type const& atom_type_;

        /// Spherical part of the effective potential.
        std::vector<double> spherical_potential_;

        /// List of radial functions for the LAPW basis.
        /** This array stores all the radial functions (AW and LO) and their derivatives. Radial derivatives of functions 
         *  are multiplied by \f$ x \f$.\n
         *  1-st dimension: index of radial point \n
         *  2-nd dimension: index of radial function \n
         *  3-nd dimension: 0 - function itself, 1 - radial derivative */
        mdarray<double, 3> radial_functions_;
        
        /// Surface derivatives of AW radial functions.
        mdarray<double, 3> aw_surface_derivatives_;

        /// Spherical part of radial integral.
        mdarray<double, 2> h_spherical_integrals_;

        /// Overlap integrals.
        mdarray<double, 3> o_radial_integrals_;

        /// Overlap integrals for IORA relativistic treatment.
        mdarray<double, 2> o1_radial_integrals_;

        /// Spin-orbit interaction integrals.
        mdarray<double, 3> so_radial_integrals_;

        /// Core charge density.
        std::vector<double> core_charge_density_;

        /// Core eigen-value sum.
        double core_eval_sum_{0};

        /// Core leakage.
        double core_leakage_{0};
        
        /// list of radial descriptor sets used to construct augmented waves 
        mutable std::vector<radial_solution_descriptor_set> aw_descriptors_;
        
        /// list of radial descriptor sets used to construct local orbitals
        mutable std::vector<local_orbital_descriptor> lo_descriptors_;
        
        /// Generate radial functions for augmented waves
        void generate_aw_radial_functions(relativity_t rel__);

        /// Generate local orbital raidal functions
        void generate_lo_radial_functions(relativity_t rel__);

    public:
    
        /// Constructor
        Atom_symmetry_class(int id_, Atom_type const& atom_type_) 
            : id_(id_)
            , atom_type_(atom_type_)
        {
            if (!atom_type_.initialized()) {
                TERMINATE("atom type is not initialized");
            }

            aw_surface_derivatives_ = mdarray<double, 3>(atom_type_.max_aw_order(), atom_type_.num_aw_descriptors(), 3);

            radial_functions_ = mdarray<double, 3>(atom_type_.num_mt_points(), atom_type_.mt_radial_basis_size(), 2);

            h_spherical_integrals_ = mdarray<double, 2>(atom_type_.mt_radial_basis_size(), atom_type_.mt_radial_basis_size());
            h_spherical_integrals_.zero();
            
            o_radial_integrals_ = mdarray<double, 3>(atom_type_.indexr().lmax() + 1, atom_type_.indexr().max_num_rf(), 
                                                     atom_type_.indexr().max_num_rf());
            o_radial_integrals_.zero();
            
            so_radial_integrals_ = mdarray<double, 3>(atom_type_.indexr().lmax() + 1, atom_type_.indexr().max_num_rf(), 
                                                      atom_type_.indexr().max_num_rf());
            so_radial_integrals_.zero();

            if (atom_type_.parameters().valence_relativity() == relativity_t::iora) {
                o1_radial_integrals_ = mdarray<double, 2>(atom_type_.mt_radial_basis_size(), atom_type_.mt_radial_basis_size());
                o1_radial_integrals_.zero();
            }

            /* copy descriptors because enu is defferent between atom classes */
            aw_descriptors_.resize(atom_type_.num_aw_descriptors());
            for (int i = 0; i < num_aw_descriptors(); i++) {
                aw_descriptors_[i] = atom_type_.aw_descriptor(i);
            }

            lo_descriptors_.resize(atom_type_.num_lo_descriptors());
            for (int i = 0; i < num_lo_descriptors(); i++) {
                lo_descriptors_[i] = atom_type_.lo_descriptor(i);
            }
            
            core_charge_density_.resize(atom_type_.num_mt_points());
            std::memset(&core_charge_density_[0], 0, atom_type_.num_mt_points() * sizeof(double));
        }

        /// Set the spherical component of the potential
        /** Atoms belonging to the same symmetry class have the same spherical potential. */
        void set_spherical_potential(std::vector<double> const& vs__);

        void generate_radial_functions(relativity_t rel__);

        void sync_radial_functions(Communicator const& comm__, int const rank__);
      
        void sync_radial_integrals(Communicator const& comm__, int const rank__);
        
        void sync_core_charge_density(Communicator const& comm__, int const rank__);

        /// Check if local orbitals are linearly independent
        std::vector<int> check_lo_linear_independence(double etol__);

        /// Dump local orbitals to the file for debug purposes
        void dump_lo();
       
        /// Compute m-th order radial derivative at the MT surface.
        inline double aw_surface_dm(int l, int order, int dm) const
        {
            assert(dm <= 2);
            return aw_surface_derivatives_(order, l, dm);
        }

        inline void set_aw_surface_deriv(int l, int order, int dm, double deriv)
        {
            assert(dm <= 2);
            aw_surface_derivatives_(order, l, dm) = deriv;
        }

        /// Find core states and generate core density.
        void generate_core_charge_density(relativity_t core_rel__);

        void find_enu(relativity_t rel__);

        void write_enu(runtime::pstdout& pout) const;

        /// Generate radial overlap and SO integrals
        /** In the case of spin-orbit interaction the following integrals are computed:
         *  \f[
         *      \int f_{p}(r) \Big( \frac{1}{(2 M c)^2} \frac{1}{r} \frac{d V}{d r} \Big) f_{p'}(r) r^2 dr
         *  \f]
         *  
         *  Relativistic mass M is defined as
         *  \f[
         *      M = 1 - \frac{1}{2 c^2} V
         *  \f]
         */
        void generate_radial_integrals(relativity_t rel__);

        /// Return symmetry class id.
        inline int id() const
        {
            return id_;
        }

        /// Add atom id to the current class.
        inline void add_atom_id(int atom_id__)
        {
            atom_id_.push_back(atom_id__);
        }

        /// Return number of atoms belonging to the current symmetry class.
        inline int num_atoms() const
        {
            return static_cast<int>(atom_id_.size());
        }

        inline int atom_id(int idx) const
        {
            return atom_id_[idx];
        }

        /// Get a value of the radial functions.
        inline double radial_function(int ir, int idx) const
        {
            return radial_functions_(ir, idx, 0);
        }

        /// Get a reference to the value of the radial function.
        inline double& radial_function(int ir, int idx)
        {
            return radial_functions_(ir, idx, 0);
        }

        /// Get a value of the radial function derivative.
        inline double radial_function_derivative(int ir, int idx) const
        {
            return radial_functions_(ir, idx, 1);
        }

        /// Get a reference to the value of the radial function derivative.
        inline double& radial_function_derivative(int ir, int idx)
        {
            return radial_functions_(ir, idx, 1);
        }

        inline double h_spherical_integral(int i1, int i2) const
        {
            return h_spherical_integrals_(i1, i2);
        }

        inline double const& o_radial_integral(int l, int order1, int order2) const
        {
            return o_radial_integrals_(l, order1, order2);
        }

        inline void set_o_radial_integral(int l, int order1, int order2, double oint__)
        {
            o_radial_integrals_(l, order1, order2) = oint__;
        }

        inline double const& o1_radial_integral(int xi1__, int xi2__) const
        {
            return o1_radial_integrals_(xi1__, xi2__);
        }

        inline double so_radial_integral(int l, int order1, int order2) const
        {
            return so_radial_integrals_(l, order1, order2);
        }

        inline double core_charge_density(int ir) const
        {
            assert(ir >= 0 && ir < (int)core_charge_density_.size());

            return core_charge_density_[ir];
        }

        inline Atom_type const& atom_type() const
        {
            return atom_type_;
        }

        inline double core_eval_sum() const
        {
            return core_eval_sum_;
        }

        inline double core_leakage() const
        {
            return core_leakage_;
        }

        inline int num_aw_descriptors() const
        {
            return static_cast<int>(aw_descriptors_.size());
        }

        inline radial_solution_descriptor_set& aw_descriptor(int idx__) const
        {
            return aw_descriptors_[idx__];
        }

        inline int num_lo_descriptors() const
        {
            return static_cast<int>(lo_descriptors_.size());
        }

        inline local_orbital_descriptor& lo_descriptor(int idx__) const
        {
            return lo_descriptors_[idx__];
        }

        inline void set_aw_enu(int l, int order, double enu)
        {
            aw_descriptors_[l][order].enu = enu;
        }

        inline double get_aw_enu(int l, int order) const
        {
            return aw_descriptors_[l][order].enu;
        }

        inline void set_lo_enu(int idxlo, int order, double enu)
        {
            lo_descriptors_[idxlo].rsd_set[order].enu = enu;
        }

        inline double get_lo_enu(int idxlo, int order) const
        {
            return lo_descriptors_[idxlo].rsd_set[order].enu;
        }
};

}

#endif // __ATOM_SYMMETRY_CLASS_H__
