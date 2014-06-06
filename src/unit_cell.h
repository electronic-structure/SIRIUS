// Copyright (c) 2013-2014 Anton Kozhevnikov, Thomas Schulthess
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

/** \file unit_cell.h
 *   
 *  \brief Contains definition and partial implementation of sirius::Unit_cell class.
 */

#ifndef __UNIT_CELL_H__
#define __UNIT_CELL_H__

extern "C" {
#include <spglib.h>
}

#include <algorithm>
#include "descriptors.h"
#include "atom_type.h"
#include "atom_symmetry_class.h"
#include "atom.h"
#include "mpi_grid.h"
#include "symmetry.h"

namespace sirius {

class Unit_cell
{
    private:
        
        /// Mapping between atom type label and an ordered internal id in the range [0, \f$ N_{types} \f$).
        std::map<std::string, int> atom_type_id_map_;
         
        /// List of atom types.
        std::vector<Atom_type*> atom_types_;

        /// List of atom classes.
        std::vector<Atom_symmetry_class*> atom_symmetry_classes_;
        
        /// List of atoms.
        std::vector<Atom*> atoms_;
       
        /// Split index of atoms.
        splindex<block> spl_num_atoms_;
        
        /// Split index of atom symmetry classes.
        splindex<block> spl_num_atom_symmetry_classes_;

        /// Bravais lattice vectors in row order.
        double lattice_vectors_[3][3];
        
        /// Inverse Bravais lattice vectors in column order.
        /** This matrix is used to find fractional coordinates by Cartesian coordinates */
        double inverse_lattice_vectors_[3][3];
        
        /// vectors of the reciprocal lattice in row order (inverse Bravais lattice vectors scaled by 2*Pi)
        double reciprocal_lattice_vectors_[3][3];

        /// volume of the unit cell; volume of Brillouin zone is (2Pi)^3 / omega
        double omega_;
       
        /// total volume of the muffin tin spheres
        double volume_mt_;
        
        /// volume of the interstitial region
        double volume_it_;

        /// spglib structure with symmetry information
        SpglibDataset* spg_dataset_;
        
        /// total nuclear charge
        int total_nuclear_charge_;
        
        /// total number of core electrons
        double num_core_electrons_;
        
        /// total number of valence electrons
        double num_valence_electrons_;

        /// total number of electrons
        double num_electrons_;

        /// list of equivalent atoms, provided externally
        std::vector<int> equivalent_atoms_;
    
        /// maximum number of muffin-tin points across all atom types
        int max_num_mt_points_;
        
        /// total number of MT basis functions
        int mt_basis_size_;
        
        /// maximum number of MT basis functions across all atoms
        int max_mt_basis_size_;

        /// maximum number of MT radial basis functions across all atoms
        int max_mt_radial_basis_size_;

        /// Total number of augmented wave basis functions in the muffin-tins.
        /** This is equal to the total number of matching coefficients for each plane-wave. */
        int mt_aw_basis_size_;

        /// List of augmented wave basis descriptors.
        /** Establishes mapping between global index in the range [0, mt_aw_basis_size_) 
         *  and corresponding atom and local index \f$ \xi \f$ 
         */
        std::vector<mt_basis_descriptor> mt_aw_basis_descriptors_; 
        
        /// total number of local orbital basis functions
        int mt_lo_basis_size_;

        /// maximum AW basis size across all atoms
        int max_mt_aw_basis_size_;

        /// list of nearest neighbours for each atom
        std::vector< std::vector<nearest_neighbour_descriptor> > nearest_neighbours_;

        /// minimum muffin-tin radius
        double min_mt_radius_;
        
        /// maximum muffin-tin radius
        double max_mt_radius_;
        
        /// scale muffin-tin radii automatically
        int auto_rmt_;

        int lmax_beta_;

        electronic_structure_method_t esm_type_;

        MPI_comm_bundle mpi_atoms_;
        
        splindex<block> spl_atoms_;

        mdarray<int, 2> beta_t_idx_;
        
        /// total number of beta-projectors among atom types
        int num_beta_t_;

        mdarray<double, 2> atom_pos_;

        /// Automatically determine new muffin-tin radii as a half distance between neighbor atoms.
        /** In order to guarantee a unique solution muffin-tin radii are dermined as a half distance
         *  bethween nearest atoms. Initial values of the muffin-tin radii are ignored. 
         */
        std::vector<double> find_mt_radii();
        
        /// Check if MT spheres overlap
        bool check_mt_overlap(int& ia__, int& ja__);

        int next_atom_type_id(const std::string label);

    public:
    
        Unit_cell(electronic_structure_method_t esm_type__) 
            : omega_(0),
              volume_mt_(0),
              volume_it_(0),
              spg_dataset_(NULL), 
              total_nuclear_charge_(0),
              num_core_electrons_(0),
              num_valence_electrons_(0),
              num_electrons_(0),
              auto_rmt_(0), 
              lmax_beta_(-1),
              esm_type_(esm_type__)
        {
        }
        
        /// Initialize the unit cell data
        /** Several things must be done during this phase:
         *    1. Compute number of electrons
         *    2. Compute MT basis function indices
         *    3. [if needed] Scale MT radii
         *    4. Check MT overlap 
         *    5. Create radial grid for each atom type
         *    6. Find symmetry and assign symmetry class to each atom
         *    7. Create split indices for atoms and atom classes
         *
         *  Initialization must be broken into two parts: one is called once, and the second one is called
         *  each time the atoms change the position.
         */
        void initialize(int lmax_apw, int lmax_pot, int num_mag_dims);

        /// Update the unit cell after moving the atoms.
        /** When the unit cell is initialized for the first time, or when the atoms are moved, several things
         *  must be recomputed:
         *    1. New atom positions may lead to a new symmetry, which can give a different number of atom 
         *       symmetry classes. Symmetry information must be updated.
         *    2. New atom positions can lead to new MT radii if they are determined automatically. MT radii and 
         *       radial meshes must be updated. 
         *  
         *  Becasue of (1) the G and G+k phase factors must be updated. Because of (2) Bessel funcion moments
         *  and G+k APW basis must be also updated. Because of (1 and 2) step function must be updated.
         *
         *  \todo Think how to implement this dependency in a reliable way without any handwork.
         */
        void update();

        /// Clear the unit cell data.
        void clear();
       
        /// Add new atom type to the list of atom types and read necessary data from the .json file
        void add_atom_type(const std::string label, const std::string file_name, 
                           electronic_structure_method_t esm_type);
        
        /// Add new atom to the list of atom types.
        void add_atom(const std::string label, double* position, double* vector_field);

        /// Add new atom without vector field to the list of atom types.
        void add_atom(const std::string label, double* position);
        
        /// Print basic info.
        void print_info();

        unit_cell_parameters_descriptor unit_cell_parameters();
        
        /// Get crystal symmetries and equivalent atoms.
        /** Makes a call to spglib providing the basic unit cell information: lattice vectors and atomic types 
         *  and positions. Gets back symmetry operations and a table of equivalent atoms. The table of equivalent 
         *  atoms is then used to make a list of atom symmetry classes and related data.
         */
        void get_symmetry();

        /// Write structure to CIF file.
        void write_cif();

        void write_json();
        
        /// Set lattice vectors.
        /** Initializes lattice vectors, inverse lattice vector matrix, reciprocal lattice vectors and the
         *  unit cell volume. 
         */
        void set_lattice_vectors(double* a1, double* a2, double* a3);
       
        /// Find the cluster of nearest neighbours around each atom
        void find_nearest_neighbours(double cluster_radius);

        bool is_point_in_mt(vector3d<double> vc, int& ja, int& jr, double& dr, double tp[2]);
        
        void generate_radial_functions();

        void generate_radial_integrals();
        
        std::string chemical_formula();

        int atom_id_by_position(vector3d<double> position__)
        {
            const double eps = 1e-10;

            for (int ia = 0; ia < num_atoms(); ia++)
            {
                vector3d<double> pos = atom(ia)->position();
                if (fabs(pos[0] - position__[0]) < eps && 
                    fabs(pos[1] - position__[1]) < eps && 
                    fabs(pos[2] - position__[2]) < eps) return ia;
            }
            return -1;
        } 

        template <typename T>
        inline vector3d<double> get_cartesian_coordinates(vector3d<T> a)
        {
            vector3d<double> b;
            for (int x = 0; x < 3; x++)
            {
                for (int l = 0; l < 3; l++) b[x] += a[l] * lattice_vectors_[l][x];
            }
            return b;
        }

        inline vector3d<double> get_fractional_coordinates(vector3d<double> a)
        {
            vector3d<double> b;
            for (int l = 0; l < 3; l++)
            {
                for (int x = 0; x < 3; x++) b[l] += a[x] * inverse_lattice_vectors_[x][l];
            }
            return b;
        }
        
        /// Get x coordinate of lattice vector l
        inline double lattice_vectors(int l, int x)
        {
            return lattice_vectors_[l][x];
        }
        
        /// Get x coordinate of reciprocal lattice vector l
        inline double reciprocal_lattice_vectors(int l, int x)
        {
            return reciprocal_lattice_vectors_[l][x];
        }

        /// Unit cell volume.
        inline double omega()
        {
            return omega_;
        }
        
        /// Pointer to atom by atom id.
        inline Atom* atom(int id)
        {
            return atoms_[id];
        }
        
        /// Number of atom types.
        inline int num_atom_types()
        {
            assert(atom_types_.size() == atom_type_id_map_.size());

            return (int)atom_types_.size();
        }

        /// Pointer to atom type by label.
        inline Atom_type* atom_type(const std::string label)
        {
            return atom_types_[atom_type_id_map_[label]];
        }
 
        /// Pointer to atom type by internal id.
        inline Atom_type* atom_type(int id)
        {
            return atom_types_[id];
        }
       
        /// Number of atom symmetry classes.
        inline int num_atom_symmetry_classes()
        {
            return (int)atom_symmetry_classes_.size();
        }
       
        /// Pointer to symmetry class by class id.
        inline Atom_symmetry_class* atom_symmetry_class(int id)
        {
            return atom_symmetry_classes_[id];
        }
        
        /// Total number of electrons (core + valence)
        inline double num_electrons()
        {
            return num_electrons_;
        }

        /// Number of valence electrons
        inline double num_valence_electrons()
        {
            return num_valence_electrons_;
        }
        
        /// Number of core electrons
        inline double num_core_electrons()
        {
            return num_core_electrons_;
        }
        
        /// Number of atoms in the unit cell.
        inline int num_atoms()
        {
            return (int)atoms_.size();
        }
       
        /// Maximum number of muffin-tin points across all atom types
        inline int max_num_mt_points()
        {
            return max_num_mt_points_;
        }
        
        /// Total number of the augmented wave basis functions over all atoms
        inline int mt_aw_basis_size()
        {
            return mt_aw_basis_size_;
        }

        /// Total number of local orbital basis functions over all atoms
        inline int mt_lo_basis_size()
        {
            return mt_lo_basis_size_;
        }

        /// Total number of the muffin-tin basis functions.
        /** Total number of MT basis functions equals to the sum of the total number of augmented wave
         *  basis functions and the total number of local orbital basis functions across all atoms. It controls 
         *  the size of the muffin-tin part of the first-variational states and second-variational wave functions. 
         */
        inline int mt_basis_size()
        {
            return mt_basis_size_;
        }
        
        /// Maximum number of basis functions across all atom types
        inline int max_mt_basis_size()
        {
            return max_mt_basis_size_;
        }

        /// Maximum number of radial functions actoss all atom types
        inline int max_mt_radial_basis_size()
        {
            return max_mt_radial_basis_size_;
        }

        /// Minimum muffin-tin radius
        inline double min_mt_radius()
        {
            return min_mt_radius_;
        }

        /// Maximum muffin-tin radius
        inline double max_mt_radius()
        {
            return max_mt_radius_;
        }

        /// Maximum number of AW basis functions across all atom types
        inline int max_mt_aw_basis_size()
        {
            return max_mt_aw_basis_size_;
        }

        inline void set_auto_rmt(int auto_rmt__)
        {
            auto_rmt_ = auto_rmt__;
        }

        inline int auto_rmt()
        {
            return auto_rmt_;
        }
        
        void set_equivalent_atoms(int* equivalent_atoms__)
        {
            equivalent_atoms_.resize(num_atoms());
            memcpy(&equivalent_atoms_[0], equivalent_atoms__, num_atoms() * sizeof(int));
        }
        
        inline splindex<block>& spl_num_atoms()
        {
            return spl_num_atoms_;
        }

        inline int spl_num_atoms(int i)
        {
            return static_cast<int>(spl_num_atoms_[i]);
        }
        
        inline splindex<block>& spl_num_atom_symmetry_classes()
        {
            return spl_num_atom_symmetry_classes_;
        }

        inline int spl_num_atom_symmetry_classes(int i)
        {
            return static_cast<int>(spl_num_atom_symmetry_classes_[i]);
        }

        inline double volume_mt()
        {
            return volume_mt_;
        }

        inline double volume_it()
        {
            return volume_it_;
        }

        inline int lmax_beta()
        {
            return lmax_beta_;
        }

        inline bool full_potential()
        {
            return (esm_type_ == full_potential_lapwlo || esm_type_ == full_potential_pwlo);
        }

        inline int num_nearest_neighbours(int ia)
        {
            return (int)nearest_neighbours_[ia].size();
        }

        inline nearest_neighbour_descriptor& nearest_neighbour(int i, int ia)
        {
            return nearest_neighbours_[ia][i];
        }

        inline int num_beta_t()
        {
            return num_beta_t_;
        }

        inline mdarray<double, 2>& atom_pos()
        {
            return atom_pos_;
        }

        inline mdarray<int, 2>& beta_t_idx()
        {
            return beta_t_idx_;
        }

        inline mt_basis_descriptor& mt_aw_basis_descriptor(int idx)
        {
            return mt_aw_basis_descriptors_[idx];
        }
};
    
};

#endif // __UNIT_CELL_H__

