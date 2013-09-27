namespace sirius {

struct nearest_neighbour_descriptor
{
    /// id of neighbour atom
    int atom_id;

    /// translation along each lattice vector
    int translation[3];

    /// distance from the central atom
    double distance;
};

class Unit_cell_test;

class Unit_cell
{
    friend class Unit_cell_test;

    private:
        
        /// mapping between atom type id and an ordered index in the range [0, N_{types} - 1]
        std::map<int, int> atom_type_index_by_id_;
         
        /// list of atom types
        std::vector<Atom_type*> atom_types_;

        /// list of atom classes
        std::vector<Atom_symmetry_class*> atom_symmetry_classes_;
        
        /// list of atoms
        std::vector<Atom*> atoms_;
       
        /// split index of atoms
        splindex<block> spl_num_atoms_;
        
        /// split index of atom symmetry classes 
        splindex<block> spl_num_atom_symmetry_classes_;

        /// Bravais lattice vectors in row order
        double lattice_vectors_[3][3];
        
        /// inverse Bravais lattice vectors in column order 
        /** This matrix is used to find fractional coordinates by Cartesian coordinates */
        double inverse_lattice_vectors_[3][3];
        
        /// vectors of the reciprocal lattice in row order (inverse Bravais lattice vectors scaled by 2*Pi)
        double reciprocal_lattice_vectors_[3][3];

        /// volume of the unit cell; volume of Brillouin zone is (2Pi)^3 / omega
        double omega_;
       
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

        /// total number of augmented wave basis functions in the MT (= number of matching coefficients for each plane-wave)
        int mt_aw_basis_size_;
        
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

        /// Get crystal symmetries and equivalent atoms.
        /** Makes a call to spglib providing the basic unit cell information: lattice vectors and atomic types 
            and positions. Gets back symmetry operations and a table of equivalent atoms. The table of equivalent 
            atoms is then used to make a list of atom symmetry classes and related data.
        */
        void get_symmetry();
        
        /// Automatically determine new muffin-tin radii as a half distance between neighbor atoms.
        /** In order to guarantee a unique solution muffin-tin radii are dermined as a half distance
            bethween nearest atoms. Initial values of the muffin-tin radii (provided in the input file) 
            are ignored. */
        std::vector<double> find_mt_radii();
        
        /// Check if MT spheres overlap
        bool check_mt_overlap(int& ia__, int& ja__);

    protected:

        /// Initialize the unit cell data
        /** Several things must be done during this phase:
              1. Compute number of electrons
              2. Compute MT basis function indices
              3. [if needed] Scale MT radii
              4. Check MT overlap 
              5. Create radial grid for each atom type
              6. Find symmetry and assign symmetry class to each atom
              7. Create split indices for atoms and atom classes

            Initialization must be broken into two parts: one is called once, and the second one is called
            each time the atoms change the position.

            \todo This must be redesigned because currently initialization of the Unit_cell (which is a part of Global 
                  class) depends on the "to be determined" parameters such as num_mag_dims. Probably Unit_cell must 
                  become a separate object.
        */
        void init(int lmax_apw, int lmax_pot, int num_mag_dims);

        /// Update the unit cell after moving the atoms.
        /** When the unit cell is initialized for the first time, or when the atoms are moved, several things
            must be recomputed:
              1. New atom positions may lead to a new symmetry, which can give a different number of atom 
                 symmetry classes. Symmetry information must be updated.
              2. New atom positions can lead to new MT radii if they are determined automatically. MT radii and 
                 radial meshes must be updated. 
            
            Becasue of (1) the G and G+k phase factors must be updated. Because of (2) Bessel funcion moments
            and G+k APW basis must be also updated. Because of (1 and 2) step function must be updated.

            \todo Think how to implement this dependency in a reliable way without any handwork.
        */
        void update();

        /// Clear the unit cell data
        void clear();
        
    public:
    
        Unit_cell() : spg_dataset_(NULL), auto_rmt_(0)
        {
            assert(sizeof(int) == 4);
            assert(sizeof(double) == 8);
        }
       
        /// Add new atom type to the list of atom types.
        void add_atom_type(int atom_type_id, const std::string label);
        
        /// Add new atom to the list of atom types.
        void add_atom(int atom_type_id, double* position, double* vector_field);

        /// Add new atom without vector field to the list of atom types
        void add_atom(int atom_type_id, double* position);
        
        /// Print basic info
        void print_info();

        /// Write structure to CIF file
        void write_cif();
        
        /// Set lattice vectors.
        /** Initializes lattice vectors, inverse lattice vector matrix, reciprocal lattice vectors and the
            unit cell volume. */
        void set_lattice_vectors(double* a1, double* a2, double* a3);
       
        /// Find the cluster of nearest neighbours around each atom
        void find_nearest_neighbours(double cluster_radius);

        bool is_point_in_mt(double vc[3], int& ja, int& jr, double& dr, double tp[2]);
        
        template <lattice_t Tl>
        void find_translation_limits(double radius, int* limits);
        
        template <lattice_t Tl>
        void reduce_coordinates(double vc[3], int ntr[3], double vf[3]);

        /// Convert coordinates (fractional <-> Cartesian) of direct or reciprocal lattices
        template<coordinates_t cT, lattice_t lT, typename T>
        void get_coordinates(T* a, double* b);
        
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
            assert(atom_types_.size() == atom_type_index_by_id_.size());

            return (int)atom_types_.size();
        }

        /// Atom type index by atom type id.
        inline int atom_type_index_by_id(int id)
        {
            return atom_type_index_by_id_[id];
        }
        
        /// Pointer to atom type by type id
        inline Atom_type* atom_type_by_id(int id)
        {
            return atom_types_[atom_type_index_by_id(id)];
        }
 
        /// Pointer to atom type by type index (not(!) by atom type id)
        inline Atom_type* atom_type(int idx)
        {
            return atom_types_[idx];
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
            basis functions and the total number of local orbital basis functions across all atoms. It controls 
            the size of the muffin-tin part of the first-variational states and second-variational wave functions.
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
            return spl_num_atoms_[i];
        }
        
        inline splindex<block>& spl_num_atom_symmetry_classes()
        {
            return spl_num_atom_symmetry_classes_;
        }

        inline int spl_num_atom_symmetry_classes(int i)
        {
            return spl_num_atom_symmetry_classes_[i];
        }
};

#include "unit_cell.hpp"

class Unit_cell_test
{
    public:

        Unit_cell_test()
        {
            Unit_cell unit_cell;
            
            double a0[] = {0.5, 0.5, 0.0};
            double a1[] = {0.5, 0.0, 0.5};
            double a2[] = {0.0, 0.5, 0.5};
            unit_cell.set_lattice_vectors(&a0[0], &a1[0], &a2[0]);
            
            unit_cell.set_auto_rmt(1);

            unit_cell.add_atom_type(1, "C");

            {
            double pos0[] = {0, 0, 0};
            double pos1[] = {0.25, 0.25, 0.25};
            unit_cell.add_atom(1, pos0);
            unit_cell.add_atom(1, pos1);
            }

            unit_cell.init(10, 10, 0);
            unit_cell.print_info();

            {
            double pos1[] = {0.251, 0.251, 0.251};
            unit_cell.atom(1)->set_position(pos1);
            }
            unit_cell.update();
            unit_cell.print_info();
            
            {
            double pos1[] = {0.251, 0.252, 0.253};
            unit_cell.atom(1)->set_position(pos1);
            }
            unit_cell.update();
            unit_cell.print_info();
        }
};
    
};

