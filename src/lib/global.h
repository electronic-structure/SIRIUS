#ifndef __GLOBAL_H__
#define __GLOBAL_H__

namespace sirius {

class Global 
{
    private:

        /// list of atom classes
        std::vector<AtomSymmetryClass*> atom_symmetry_class_by_id_;

        /// unique mapping between atom type id and atom type
        std::map<int, AtomType*> atom_type_by_id_;
   
        /// list of atoms
        std::vector<Atom*> atoms_;
        
        /// Bravais lattice vectors in row order
        double lattice_vectors_[3][3];
        
        /// inverse Bravais lattice vectors in column order (used to find fractional coordinates by Cartesian coordinates)
        double inverse_lattice_vectors_[3][3];
        
        /// vectors of the reciprocal lattice in row order (inverse Bravais lattice vectors scaled by 2*Pi)
        double reciprocal_lattice_vectors_[3][3];
       
        /// spglib structure with symmetry information
        SpglibDataset* spg_dataset_;

        /// plane wave cutoff radius (in inverse a.u. of length)
        double pw_cutoff_;
        
        /// fft wrapper
        FFT3D fft_;

        void get_symmetry()
        {
            Timer t("get_symmetry");
            
            if (spg_dataset_) 
                error(__FILE__, __LINE__, "spg_dataset is already allocated");
                
            if (atom_symmetry_class_by_id_.size() != 0)
                error(__FILE__, __LINE__, "atom_symmetry_class_by_id_ list is not empty");
            
            double lattice[3][3];

            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++) 
                    lattice[i][j] = lattice_vectors_[j][i];

            mdarray<double,2> positions(NULL, 3, atoms_.size());
            positions.allocate();
            
            std::vector<int> types(atoms_.size());

            for (int i = 0; i < (int)atoms_.size(); i++)
            {
                atoms_[i]->get_position(&positions(0, i));
                types[i] = atoms_[i]->type_id();
            }
            spg_dataset_ = spg_get_dataset(lattice, (double(*)[3])&positions(0, 0), &types[0], atoms_.size(), 1e-5);

            if (spg_dataset_->spacegroup_number == 0)
                error(__FILE__, __LINE__, "spg_get_dataset() returned 0 for the space group");

            if (spg_dataset_->n_atoms != (int)atoms_.size())
                error(__FILE__, __LINE__, "wrong number of atoms");

            AtomSymmetryClass* atom_symmetry_class;
            
            int atom_class_id = -1;

            for (int i = 0; i < (int)atoms_.size(); i++)
            {
                if (atoms_[i]->symmetry_class_id() == -1) // if class id is not assigned to this atom
                {
                    atom_class_id++; // take next id 
                    atom_symmetry_class = new AtomSymmetryClass(atom_class_id, atoms_[i]->type());
                    atom_symmetry_class_by_id_.push_back(atom_symmetry_class);

                    for (int j = 0; j < (int)atoms_.size(); j++) // scan all atoms
                        if (spg_dataset_->equivalent_atoms[j] == spg_dataset_->equivalent_atoms[i]) // assign new class id for all equivalent atoms
                        {
                            atom_symmetry_class->add_atom_id(j);
                            atoms_[j]->set_symmetry_class(atom_symmetry_class);
                        }
                }
            }
        }
        
        
    public:
    
        Global() : spg_dataset_(NULL),
                   pw_cutoff_(pw_cutoff_default)
        {
            assert(sizeof(int4) == 4);
            assert(sizeof(real8) == 8);
        }

        void print_info()
        {
            printf("\n");
            printf("SIRIUS v0.1\n");
            printf("\n");

            printf("lattice vectors\n");
            for (int i = 0; i < 3; i++)
                printf("  a%1i : %18.10f %18.10f %18.10f \n", i + 1, lattice_vectors_[i][0], 
                                                                     lattice_vectors_[i][1], 
                                                                     lattice_vectors_[i][2]); 
            printf("reciprocal lattice vectors\n");
            for (int i = 0; i < 3; i++)
                printf("  b%1i : %18.10f %18.10f %18.10f \n", i + 1, reciprocal_lattice_vectors_[i][0], 
                                                                     reciprocal_lattice_vectors_[i][1], 
                                                                     reciprocal_lattice_vectors_[i][2]);
            std::map<int, AtomType*>::iterator it;    
            printf("\n"); 
            printf("number of atom types : %i\n", (int)atom_type_by_id_.size());
            for (it = atom_type_by_id_.begin(); it != atom_type_by_id_.end(); it++)
                printf("type id : %i   symbol : %s   label : %s\n", (*it).first, (*it).second->symbol().c_str(), (*it).second->label().c_str()); 
                
            printf("number of atoms : %i\n", (int)atoms_.size());
            printf("number of symmetry classes : %i\n", (int)atom_symmetry_class_by_id_.size());

            printf("\n"); 
            printf("atom id    type id    class id\n");
            printf("------------------------------\n");
            for (int i = 0; i < (int)atoms_.size(); i++)
            {
                printf("%6i     %6i      %6i\n", i, atoms_[i]->type_id(), atoms_[i]->symmetry_class_id()); 
           
            }

            printf("\n");
            for (int ic = 0; ic < (int)atom_symmetry_class_by_id_.size(); ic++)
            {
                printf("class id : %i   atom id : ", ic);
                for (int i = 0; i < atom_symmetry_class_by_id_[ic]->num_atoms(); i++)
                    printf("%i ", atom_symmetry_class_by_id_[ic]->atom_id(i));  
                printf("\n");
            }

            printf("\n");
            printf("space group number   : %i\n", spg_dataset_->spacegroup_number);
            printf("international symbol : %s\n", spg_dataset_->international_symbol);
            printf("Hall symbol          : %s\n", spg_dataset_->hall_symbol);
            printf("number of operations : %i\n", spg_dataset_->n_operations);
            
            printf("\n");
            Timer::print();
        }
        
        void set_lattice_vectors(double* a1, 
                                 double* a2, 
                                 double* a3)
        {
            for (int x = 0; x < 3; x++)
            {
                lattice_vectors_[0][x] = a1[x];
                lattice_vectors_[1][x] = a2[x];
                lattice_vectors_[2][x] = a3[x];
            }
            double a[3][3];
            memcpy(&a[0][0], &lattice_vectors_[0][0], 9 * sizeof(double));
            
            double t1;
            t1 = a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]) + 
                 a[0][1] * (a[1][2] * a[2][0] - a[1][0] * a[2][2]) + 
                 a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]);
            
            if (fabs(t1) < 1e-20)
                error(__FILE__, __LINE__, "lattice vectors are linearly dependent");
            
            t1 = 1.0 / t1;

            double b[3][3];
            b[0][0] = t1 * (a[1][1] * a[2][2] - a[1][2] * a[2][1]);
            b[0][1] = t1 * (a[0][2] * a[2][1] - a[0][1] * a[2][2]);
            b[0][2] = t1 * (a[0][1] * a[1][2] - a[0][2] * a[1][1]);
            b[1][0] = t1 * (a[1][2] * a[2][0] - a[1][0] * a[2][2]);
            b[1][1] = t1 * (a[0][0] * a[2][2] - a[0][2] * a[2][0]);
            b[1][2] = t1 * (a[0][2] * a[1][0] - a[0][0] * a[1][2]);
            b[2][0] = t1 * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
            b[2][1] = t1 * (a[0][1] * a[2][0] - a[0][0] * a[2][1]);
            b[2][2] = t1 * (a[0][0] * a[1][1] - a[0][1] * a[1][0]);

            memcpy(&inverse_lattice_vectors_[0][0], &b[0][0], 9 * sizeof(double));

            for (int l = 0; l < 3; l++)
                for (int x = 0; x < 3; x++)
                    reciprocal_lattice_vectors_[l][x] = twopi * inverse_lattice_vectors_[x][l];
        }

        /*! 
            \brief Get fractional coordinates by Cartesian coordinates. 
        */
        void get_fractional_coordinates(double* cart_coord, double* frac_coord)
        {
            for (int l = 0; l < 3; l++)
            {
                frac_coord[l] = 0.0;
                for (int x = 0; x < 3; x++)
                    frac_coord[l] += cart_coord[x] * inverse_lattice_vectors_[x][l];
            }
        }

        /*! 
            \brief Get reciprocal fractional coordinates by reciprocal Cartesian coordinates. 
        */
        void get_reciprocal_fractional_coordinates(double* cart_coord, double* frac_coord)
        {
            for (int l = 0; l < 3; l++)
            {
                frac_coord[l] = 0.0;
                for (int x = 0; x < 3; x++)
                    frac_coord[l] += lattice_vectors_[l][x] * cart_coord[x] / twopi;
            }
        }
        
        void get_reciprocal_cartesian_coordinates(int* frac_coord, double* cart_coord)
        {
            for (int x = 0; x < 3; x++)
            {
                cart_coord[x] = 0.0;
                for (int l = 0; l < 3; l++)
                    cart_coord[x] += reciprocal_lattice_vectors_[l][x] * frac_coord[l];
            }
        }

        /*! 
            \brief Add new atom type to the collection of atom types.. 
        */
        void add_atom_type(int atom_type_id, 
                           const std::string& label)
        {
            if (atom_type_by_id_.count(atom_type_id) != 0) 
            {   
                std::stringstream s;
                s << "atom type with id " << atom_type_id << " is already in list";
                error(__FILE__, __LINE__, s);
            }
                
            atom_type_by_id_[atom_type_id] = new AtomType(atom_type_id, label);
        }
        
        void add_atom(int atom_type_id, 
                      double* position, 
                      double* vector_field)
        {
            double eps = 1e-10;
            double pos[3];
            
            if (atom_type_by_id_.count(atom_type_id) == 0)
            {
                std::stringstream s;
                s << "atom type with id " << atom_type_id << " is not found";
                error(__FILE__, __LINE__, s);
            }
 
            for (int i = 0; i < (int)atoms_.size(); i++)
            {
                atoms_[i]->get_position(pos);
                if (fabs(pos[0] - position[0]) < eps &&
                    fabs(pos[1] - position[1]) < eps &&
                    fabs(pos[2] - position[2]) < eps)
                {
                    std::stringstream s;
                    s << "atom with the same position is already in list" << std::endl
                      << "  position : " << position[0] << " " << position[1] << " " << position[2];
                    
                    error(__FILE__, __LINE__, s);
                }
            }

            atoms_.push_back(new Atom(atom_type_by_id_[atom_type_id], position, vector_field));
        }

        void init_fft_grid()
        {
            int max_frac_coord[] = {0, 0, 0};
            double frac_coord[3];
            // try three directions
            for (int i = 0; i < 3; i++)
            {
                double cart_coord[] = {0.0, 0.0, 0.0};
                cart_coord[i] = pw_cutoff_;
                get_reciprocal_fractional_coordinates(cart_coord, frac_coord);
                for (int i = 0; i < 3; i++)
                    max_frac_coord[i] = std::max(max_frac_coord[i], 2 * abs(int(frac_coord[i])) + 1);
            }
            
            fft_.set_grid_size(max_frac_coord);
            fft_.init();
            
            mdarray<int,3> index_by_reciprocal_coordinates_;
            
            index_by_reciprocal_coordinates_.set_dimensions(dimension(fft_.grid_limits(0, 0), fft_.grid_limits(0, 1)),
                                                            dimension(fft_.grid_limits(1, 0), fft_.grid_limits(1, 1)),
                                                            dimension(fft_.grid_limits(2, 0), fft_.grid_limits(2, 1)));
            index_by_reciprocal_coordinates_.allocate();

            std::vector<double> length;
            int ig = 0;
            for (int i = fft_.grid_limits(0, 0); i <= fft_.grid_limits(0, 1); i++)
                for (int j = fft_.grid_limits(1, 0); j <= fft_.grid_limits(1, 1); j++)
                    for (int k = fft_.grid_limits(2, 0); k <= fft_.grid_limits(2, 1); k++)
                    {
                        index_by_reciprocal_coordinates_(i, j, k) = ig++;
                        int fracc[] = {i, j, k};
                        double cartc[3];
                        get_reciprocal_cartesian_coordinates(fracc, cartc);
                        length.push_back(vector_length(cartc));
                    }

            std::vector<size_t> reorder(fft_.size());
            gsl_heapsort_index(&reorder[0], &length[0], fft_.size(), sizeof(double), compare_doubles);
            
            /*for (int i = 0; i < fft_.size(); i++)
                std::cout << length[reorder[i]] << std::endl;*/

        }
        
        inline FFT3D& fft()
        {
            return fft_;
        }
        
        void initialize()
        {
            get_symmetry();
            init_fft_grid();
        }

        void clear()
        {
            if (spg_dataset_)
            {
                spg_free_dataset(spg_dataset_);
                spg_dataset_ = NULL;
            }
            
            // delete atom types
            std::map<int, AtomType*>::iterator it;    
            for (it = atom_type_by_id_.begin(); it != atom_type_by_id_.end(); it++)
                delete (*it).second;
            atom_type_by_id_.clear();

            // delete atom classes
            for (int i = 0; i < (int)atom_symmetry_class_by_id_.size(); i++)
                delete atom_symmetry_class_by_id_[i];
            atom_symmetry_class_by_id_.clear();

            // delete atoms
            for (int i = 0; i < (int)atoms_.size(); i++)
                delete atoms_[i];
            atoms_.clear();
        }
};

};

sirius::Global sirius_global;

#endif // __GLOBAL_H__
