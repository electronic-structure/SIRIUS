#ifndef __ATOM_SYMMETRY_CLASS_H__
#define __ATOM_SYMMETRY_CLASS_H__

/** \file atom_symmetry_class.h

    \brief Data and methods specific to each atom symmetry class. 

    Atoms transforming into each other under symmetry opeartions belong to the same symmetry class. They have the
    same spherical part of the on-site potential and, as a consequence, the same radial functions. 
*/

namespace sirius {

class Atom_symmetry_class
{
    private:
        
        /// symmetry class id in the range [0, N_class - 1]
        int id_;

        /// list of atoms of this class
        std::vector<int> atom_id_;
        
        /// atom type
        Atom_type* atom_type_;

        /// spherical part of effective potential 
        std::vector<double> spherical_potential_;

        /// list of radial functions
        mdarray<double, 3> radial_functions_;
        
        /// surface derivatives of aw radial functions
        mdarray<double, 2> aw_surface_derivatives_;

        /// spherical part of radial integral
        mdarray<double, 2> h_spherical_integrals_;

        /// overlap integrals
        mdarray<double, 3> o_radial_integrals_;

        /// spin-orbit interaction integrals
        mdarray<double, 3> so_radial_integrals_;

        /// core charge density
        std::vector<double> core_charge_density_;

        /// core eigen-value sum
        double core_eval_sum_;

        /// core leakage
        double core_leakage_;
        
        /// list of radial descriptor sets used to construct augmented waves 
        std::vector<radial_solution_descriptor_set> aw_descriptors_;
        
        /// list of radial descriptor sets used to construct local orbitals
        std::vector<local_orbital_descriptor> lo_descriptors_;
        
        /// Generate radial functions for augmented waves
        void generate_aw_radial_functions();

        /// Generate local orbital raidal functions
        void generate_lo_radial_functions();

        /// Check if local orbitals are linearly independent
        void check_lo_linear_independence();

        /// Dump local orbitals to the file for debug purposes
        void dump_lo();
        
        /// Transform radial functions
        /** Local orbitals are orthogonalized and all radial functions are divided by r. */
        void transform_radial_functions(bool ort_lo, bool ort_aw);

    public:
    
        /// Constructor
        Atom_symmetry_class(int id_, Atom_type* atom_type_) : 
            id_(id_), atom_type_(atom_type_), core_eval_sum_(0.0), core_leakage_(0.0)
        {
            initialize();
        }

        /// Initialize the symmetry class
        void initialize();

        /// Set the spherical component of the potential
        /** Atoms belonging to the same symmetry class have the same spherical potential.
        */
        void set_spherical_potential(std::vector<double>& veff);

        void generate_radial_functions();

        void sync_radial_functions(int rank);
      
        void sync_radial_integrals(int rank);
        
        void sync_core_charge_density(int rank);
       
        /// Compute m-th order radial derivative at the MT surface
        double aw_surface_dm(int l, int order, int dm);
        
        /// Find core states and generate core density
        void generate_core_charge_density();

        void write_enu(pstdout& pout);
        
        /// Generate radial overlap and SO integrals
        /** In the case of spin-orbit interaction the following integrals are computed:
            \f[
                \int f_{p}(r) \Big( \frac{1}{(2 M c)^2} \frac{1}{r} \frac{d V}{d r} \Big) f_{p'}(r) r^2 dr
            \f]
            
            Relativistic mass M is defined as
            \f[
                M = 1 - \frac{1}{2 c^2} V
            \f]
        */
        void generate_radial_integrals();
        
        /// Return symmetry class id
        inline int id()
        {
            return id_;
        }

        /// Add atom id to the current class
        inline void add_atom_id(int _atom_id)
        {
            atom_id_.push_back(_atom_id);
        }
        
        /// Return number of atoms belonging to the current symmetry class
        inline int num_atoms()
        {
            return (int)atom_id_.size();
        }

        inline int atom_id(int idx)
        {
            return atom_id_[idx];
        }

        inline double radial_function(int ir, int idx)
        {
            return radial_functions_(ir, idx, 0);
        }

        inline double h_radial_function(int ir, int idx)
        {
            return radial_functions_(ir, idx, 1);
        }
        
        inline double h_spherical_integral(int i1, int i2)
        {
            return h_spherical_integrals_(i1, i2);
        }

        inline double o_radial_integral(int l, int order1, int order2)
        {
            return o_radial_integrals_(l, order1, order2);
        }
        
        inline double so_radial_integral(int l, int order1, int order2)
        {
            return so_radial_integrals_(l, order1, order2);
        }
        
        double core_charge_density(int ir)
        {
            assert(ir >= 0 && ir < (int)core_charge_density_.size());

            return core_charge_density_[ir];
        }

        inline Atom_type* atom_type()
        {
            return atom_type_;
        }

        inline double core_eval_sum()
        {
            return core_eval_sum_;
        }

        inline double core_leakage()
        {
            return core_leakage_;
        }
        
        inline int num_aw_descriptors()
        {
            return (int)aw_descriptors_.size();
        }

        inline radial_solution_descriptor_set& aw_descriptor(int idx)
        {
            return aw_descriptors_[idx];
        }
        
        inline int num_lo_descriptors()
        {
            return (int)lo_descriptors_.size();
        }

        inline local_orbital_descriptor& lo_descriptor(int idx)
        {
            return lo_descriptors_[idx];
        }

        inline void set_aw_enu(int l, int order, double enu)
        {
            aw_descriptors_[l][order].enu = enu;
        }
        
        inline double get_aw_enu(int l, int order)
        {
            return aw_descriptors_[l][order].enu;
        }
        
        inline void set_lo_enu(int idxlo, int order, double enu)
        {
            lo_descriptors_[idxlo].rsd_set[order].enu = enu;
        }
        
        inline double get_lo_enu(int idxlo, int order)
        {
            return lo_descriptors_[idxlo].rsd_set[order].enu;
        }
            
};

#include "atom_symmetry_class.hpp"

};

#endif // __ATOM_SYMMETRY_CLASS_H__
