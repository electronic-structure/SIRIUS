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

/** \file unit_cell.h
 *   
 *  \brief Contains definition and partial implementation of sirius::Unit_cell class.
 */

#ifndef __UNIT_CELL_H__
#define __UNIT_CELL_H__

#include <algorithm>
#include "descriptors.h"
#include "atom_type.h"
#include "atom_symmetry_class.h"
#include "atom.h"
#include "mpi_grid.h"
#include "symmetry.h"
#include "input.h"
#include "simulation_parameters.h"

namespace sirius {

class Unit_cell
{
    private:
        
        /// Basic parameters of the simulation.
        Simulation_parameters const& parameters_;
        
        /// Mapping between atom type label and an ordered internal id in the range [0, \f$ N_{types} \f$).
        std::map<std::string, int> atom_type_id_map_;
         
        /// List of atom types.
        std::vector<Atom_type> atom_types_;

        /// List of atom classes.
        std::vector<Atom_symmetry_class> atom_symmetry_classes_;
        
        /// List of atoms.
        std::vector<Atom> atoms_;
       
        /// Split index of atoms.
        splindex<block> spl_num_atoms_;
        
        /// Split index of atom symmetry classes.
        splindex<block> spl_num_atom_symmetry_classes_;

        /// Bravais lattice vectors in column order.
        /** The following convention is used to transform fractional coordinates to Cartesian:  
         *  \f[
         *    \vec v_{C} = {\bf L} \vec v_{f}
         *  \f]
         */
        matrix3d<double> lattice_vectors_;
        
        /// Inverse Bravais lattice vectors in column order.
        /** This matrix is used to find fractional coordinates by Cartesian coordinates:
         *  \f[
         *    \vec v_{f} = {\bf L}^{-1} \vec v_{C}
         *  \f]
         */
        matrix3d<double> inverse_lattice_vectors_;

        /// Reciprocal lattice vectors in column order.
        /** The following convention is used:
         *  \f[
         *    \vec a_{i} \vec b_{j} = 2 \pi \delta_{ij}
         *  \f]
         *  or in matrix notation
         *  \f[
         *    {\bf A} {\bf B}^{T} = 2 \pi {\bf I}
         *  \f]
         */
        matrix3d<double> reciprocal_lattice_vectors_;
        
        /// Volume \f$ \Omega \f$ of the unit cell. Volume of Brillouin zone is then \f$ (2\Pi)^3 / \Omega \f$.
        double omega_{0};
       
        /// Total volume of the muffin-tin spheres.
        double volume_mt_{0};
        
        /// Volume of the interstitial region.
        double volume_it_{0};

        /// Total nuclear charge.
        int total_nuclear_charge_{0};
        
        /// Total number of core electrons.
        double num_core_electrons_{0};
        
        /// Total number of valence electrons.
        double num_valence_electrons_{0};

        /// Total number of electrons.
        double num_electrons_{0};

        /// List of equivalent atoms, provided externally.
        std::vector<int> equivalent_atoms_;
    
        /// Maximum number of muffin-tin points across all atom types.
        int max_num_mt_points_{0};
        
        /// Total number of MT basis functions.
        int mt_basis_size_{0};
        
        /// Maximum number of MT basis functions across all atoms.
        int max_mt_basis_size_{0};

        /// Maximum number of MT radial basis functions across all atoms.
        int max_mt_radial_basis_size_{0};

        /// Total number of augmented wave basis functions in the muffin-tins.
        /** This is equal to the total number of matching coefficients for each plane-wave. */
        int mt_aw_basis_size_{0};

        /// List of augmented wave basis descriptors.
        /** Establishes mapping between global index in the range [0, mt_aw_basis_size_) 
         *  and corresponding atom and local index \f$ \xi \f$ */
        std::vector<mt_basis_descriptor> mt_aw_basis_descriptors_; 

        std::vector<mt_basis_descriptor> mt_lo_basis_descriptors_; 
        
        /// Total number of local orbital basis functions.
        int mt_lo_basis_size_;

        /// Maximum AW basis size across all atoms.
        int max_mt_aw_basis_size_;

        /// List of nearest neighbours for each atom.
        std::vector< std::vector<nearest_neighbour_descriptor> > nearest_neighbours_;

        /// Minimum muffin-tin radius.
        double min_mt_radius_;
        
        /// Maximum muffin-tin radius.
        double max_mt_radius_;
        
        /// Maximum orbital quantum number of radial functions between all atom types.
        int lmax_{-1};

        Communicator_bundle comm_bundle_atoms_;
        
        mdarray<double, 2> atom_pos_;

        std::unique_ptr<Symmetry> symmetry_;

        Communicator const& comm_;

        /// Automatically determine new muffin-tin radii as a half distance between neighbor atoms.
        /** In order to guarantee a unique solution muffin-tin radii are dermined as a half distance
         *  bethween nearest atoms. Initial values of the muffin-tin radii are ignored. */
        std::vector<double> find_mt_radii();
        
        /// Check if MT spheres overlap
        bool check_mt_overlap(int& ia__, int& ja__);

        int next_atom_type_id(std::string label__)
        {
            /* check if the label was already added */
            if (atom_type_id_map_.count(label__) != 0) {   
                std::stringstream s;
                s << "atom type with label " << label__ << " is already in list";
                TERMINATE(s);
            }
            /* take text id */
            atom_type_id_map_[label__] = static_cast<int>(atom_types_.size());
            return atom_type_id_map_[label__];
        }

    public:
    
        Unit_cell(Simulation_parameters const& parameters__, Communicator const& comm__)
            : parameters_(parameters__),
              comm_(comm__)
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
         *    7. Create split indices for atoms and atom classes */
        void initialize();

        /// Add new atom type to the list of atom types and read necessary data from the .json file
        void add_atom_type(const std::string label, const std::string file_name)
        {
            PROFILE();

            if (atoms_.size()) {
                TERMINATE("Can't add new atom type if atoms are already added");
            }

            int id = next_atom_type_id(label);
            atom_types_.push_back(std::move(Atom_type(parameters_, id, label, file_name)));
        }
        
        /// Add new atom to the list of atom types.
        void add_atom(const std::string label, vector3d<double> position, vector3d<double> vector_field)
        {
            PROFILE();

            if (atom_type_id_map_.count(label) == 0) {
                std::stringstream s;
                s << "atom type with label " << label << " is not found";
                TERMINATE(s);
            }
            if (atom_id_by_position(position) >= 0) {
                std::stringstream s;
                s << "atom with the same position is already in list" << std::endl
                  << "  position : " << position[0] << " " << position[1] << " " << position[2];
                TERMINATE(s);
            }
            
            atoms_.push_back(std::move(Atom(atom_type(label), position, vector_field)));
            atom_type(label).add_atom_id(static_cast<int>(atoms_.size()) - 1);
        }

        /// Add new atom without vector field to the list of atom types.
        void add_atom(const std::string label, vector3d<double> position)
        {
            PROFILE();
            add_atom(label, position, {0, 0, 0});
        }
        
        /// Print basic info.
        void print_info();

        unit_cell_parameters_descriptor unit_cell_parameters();
        
        /// Get crystal symmetries and equivalent atoms.
        /** Makes a call to spglib providing the basic unit cell information: lattice vectors and atomic types 
         *  and positions. Gets back symmetry operations and a table of equivalent atoms. The table of equivalent 
         *  atoms is then used to make a list of atom symmetry classes and related data. */
        void get_symmetry();

        /// Write structure to CIF file.
        void write_cif();

        void write_json();
        
        /// Set lattice vectors.
        /** Initializes lattice vectors, inverse lattice vector matrix, reciprocal lattice vectors and the
         *  unit cell volume. */
        void set_lattice_vectors(vector3d<double> a0__, vector3d<double> a1__, vector3d<double> a2__)
        {
            for (int x: {0, 1, 2}) {
                lattice_vectors_(x, 0) = a0__[x];
                lattice_vectors_(x, 1) = a1__[x];
                lattice_vectors_(x, 2) = a2__[x];
            }
            inverse_lattice_vectors_ = inverse(lattice_vectors_);
            omega_ = std::abs(lattice_vectors_.det());
            reciprocal_lattice_vectors_ = transpose(inverse(lattice_vectors_)) * twopi;
        }

        /// Find the cluster of nearest neighbours around each atom
        void find_nearest_neighbours(double cluster_radius);

        bool is_point_in_mt(vector3d<double> vc, int& ja, int& jr, double& dr, double tp[2]) const;
        
        void generate_radial_functions();

        void generate_radial_integrals();
        
        std::string chemical_formula();

        /// Make periodic function out of form factors.
        /** Return vector of plane-wave coefficients */
        std::vector<double_complex> make_periodic_function(mdarray<double, 2>& form_factors__, Gvec const& gvec__) const;

        int atom_id_by_position(vector3d<double> position__)
        {
            for (int ia = 0; ia < num_atoms(); ia++)
            {
                auto vd = atom(ia).position() - position__;
                if (vd.length() < 1e-10) return ia;
            }
            return -1;
        } 

        template <typename T>
        inline vector3d<double> get_cartesian_coordinates(vector3d<T> a__) const
        {
            return lattice_vectors_ * a__;
        }

        inline vector3d<double> get_fractional_coordinates(vector3d<double> a) const
        {
            return inverse_lattice_vectors_ * a;
        }
        
        /// Unit cell volume.
        inline double omega() const
        {
            return omega_;
        }

        /// Number of atom types.
        inline int num_atom_types() const
        {
            assert(atom_types_.size() == atom_type_id_map_.size());

            return static_cast<int>(atom_types_.size());
        }

        /// Return atom type instance by id.
        inline Atom_type& atom_type(int id__)
        {
            assert(id__ >= 0 && id__ < (int)atom_types_.size());
            return atom_types_[id__];
        }

        /// Return const atom type instance by id.
        inline Atom_type const& atom_type(int id__) const
        {
            assert(id__ >= 0 && id__ < (int)atom_types_.size());
            return atom_types_[id__];
        }

        /// Return atom type instance by label.
        inline Atom_type& atom_type(std::string const label__)
        {
            int id = atom_type_id_map_.at(label__);
            return atom_type(id);
        }

        /// Return const atom type instance by label.
        inline Atom_type const& atom_type(std::string const label__) const
        {
            int id = atom_type_id_map_.at(label__);
            return atom_type(id);
        }

        /// Number of atom symmetry classes.
        inline int num_atom_symmetry_classes() const
        {
            return static_cast<int>(atom_symmetry_classes_.size());
        }
       
        /// Return const symmetry class instance by class id.
        inline Atom_symmetry_class const& atom_symmetry_class(int id__) const
        {
            return atom_symmetry_classes_[id__];
        }

        /// Return symmetry class instance by class id.
        inline Atom_symmetry_class& atom_symmetry_class(int id__)
        {
            return atom_symmetry_classes_[id__];
        }

        /// Number of atoms in the unit cell.
        inline int num_atoms() const
        {
            return static_cast<int>(atoms_.size());
        }
        
        /// Return const atom instance by id.
        inline Atom const& atom(int id__) const
        {
            assert(id__ >= 0 && id__ < (int)atoms_.size());
            return atoms_[id__];
        }

        /// Return atom instance by id.
        inline Atom& atom(int id__)
        {
            assert(id__ >= 0 && id__ < (int)atoms_.size());
            return atoms_[id__];
        }
       
        /// Total number of electrons (core + valence)
        inline double num_electrons() const
        {
            return num_electrons_;
        }

        /// Number of valence electrons
        inline double num_valence_electrons() const
        {
            return num_valence_electrons_;
        }
        
        /// Number of core electrons
        inline double num_core_electrons() const
        {
            return num_core_electrons_;
        }
        
        /// Maximum number of muffin-tin points across all atom types
        inline int max_num_mt_points() const
        {
            return max_num_mt_points_;
        }
        
        /// Total number of the augmented wave basis functions over all atoms
        inline int mt_aw_basis_size() const
        {
            return mt_aw_basis_size_;
        }

        /// Total number of local orbital basis functions over all atoms
        inline int mt_lo_basis_size() const
        {
            return mt_lo_basis_size_;
        }

        /// Total number of the muffin-tin basis functions.
        /** Total number of MT basis functions equals to the sum of the total number of augmented wave
         *  basis functions and the total number of local orbital basis functions across all atoms. It controls 
         *  the size of the muffin-tin part of the first-variational states and second-variational wave functions. */
        inline int mt_basis_size() const
        {
            return mt_basis_size_;
        }
        
        /// Maximum number of basis functions across all atom types
        inline int max_mt_basis_size() const
        {
            return max_mt_basis_size_;
        }

        /// Maximum number of radial functions actoss all atom types
        inline int max_mt_radial_basis_size() const
        {
            return max_mt_radial_basis_size_;
        }

        /// Minimum muffin-tin radius
        inline double min_mt_radius() const
        {
            return min_mt_radius_;
        }

        /// Maximum muffin-tin radius
        inline double max_mt_radius() const
        {
            return max_mt_radius_;
        }

        /// Maximum number of AW basis functions across all atom types
        inline int max_mt_aw_basis_size() const
        {
            return max_mt_aw_basis_size_;
        }

        void set_equivalent_atoms(int const* equivalent_atoms__)
        {
            equivalent_atoms_.resize(num_atoms());
            memcpy(&equivalent_atoms_[0], equivalent_atoms__, num_atoms() * sizeof(int));
        }
        
        inline splindex<block> const& spl_num_atoms() const
        {
            return spl_num_atoms_;
        }

        inline int spl_num_atoms(int i) const
        {
            return static_cast<int>(spl_num_atoms_[i]);
        }
        
        inline splindex<block> const& spl_num_atom_symmetry_classes() const
        {
            return spl_num_atom_symmetry_classes_;
        }

        inline int spl_num_atom_symmetry_classes(int i) const
        {
            return static_cast<int>(spl_num_atom_symmetry_classes_[i]);
        }

        inline double volume_mt() const
        {
            return volume_mt_;
        }

        inline double volume_it() const
        {
            return volume_it_;
        }

        inline int lmax() const
        {
            return lmax_;
        }

        inline int num_nearest_neighbours(int ia) const
        {
            return static_cast<int>(nearest_neighbours_[ia].size());
        }

        inline nearest_neighbour_descriptor const& nearest_neighbour(int i, int ia) const
        {
            return nearest_neighbours_[ia][i];
        }

        inline mdarray<double, 2>& atom_pos()
        {
            return atom_pos_;
        }

        inline mt_basis_descriptor const& mt_aw_basis_descriptor(int idx) const
        {
            return mt_aw_basis_descriptors_[idx];
        }

        inline mt_basis_descriptor const& mt_lo_basis_descriptor(int idx) const
        {
            return mt_lo_basis_descriptors_[idx];
        }

        inline Symmetry const& symmetry() const
        {
            return (*symmetry_);
        }

        inline matrix3d<double> const& lattice_vectors() const
        {
            return lattice_vectors_;
        }

        inline matrix3d<double> const& reciprocal_lattice_vectors() const
        {
            return reciprocal_lattice_vectors_;
        }
        
        /// Return a single lattice vector.
        inline vector3d<double> lattice_vector(int idx__) const
        {
            return vector3d<double>(lattice_vectors_(0, idx__), lattice_vectors_(1, idx__), lattice_vectors_(2, idx__));
        }

        void import(Unit_cell_input_section const& inp__)
        {
            PROFILE();
 
            if (inp__.exist_) {
                /* first, load all types */
                for (int iat = 0; iat < (int)inp__.labels_.size(); iat++) {
                    auto label = inp__.labels_[iat];
                    auto fname = inp__.atom_files_.at(label);
                    add_atom_type(label, fname);
                }
                /* then load atoms */
                for (int iat = 0; iat < (int)inp__.labels_.size(); iat++) {
                    auto label = inp__.labels_[iat];
                    auto fname = inp__.atom_files_.at(label);
                    for (size_t ia = 0; ia < inp__.coordinates_[iat].size(); ia++) {
                        auto v = inp__.coordinates_[iat][ia];
                        vector3d<double> p(v[0], v[1], v[2]);
                        vector3d<double> f(v[3], v[4], v[5]);
                        add_atom(label, p, f);
                    }
                }

                set_lattice_vectors(inp__.a0_, inp__.a1_, inp__.a2_);
            }
        }
};
    
};

#endif // __UNIT_CELL_H__

