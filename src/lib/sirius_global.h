namespace sirius {

class SiriusGlobal : public sirius_gvec
{
    public:
    
        void initialize()
        {
            get_symmetry();
            init_fft_grid();
            find_nearest_neighbours();
            find_mt_radii();
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
            std::map<int,AtomType*>::iterator it;    
            printf("\n"); 
            printf("number of atom types : %i\n", (int)atom_type_by_id_.size());
            for (it = atom_type_by_id_.begin(); it != atom_type_by_id_.end(); it++)
                printf("type id : %i   symbol : %s   label : %s   mt_radius : %f\n", (*it).first, 
                                                                                     (*it).second->symbol().c_str(), 
                                                                                     (*it).second->label().c_str(),
                                                                                     (*it).second->mt_radius()); 
                
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
            printf("plane wave cutoff : %f\n", pw_cutoff_);
            printf("FFT grid size : %i %i %i   total : %i\n", fft_.size(0), fft_.size(1), fft_.size(2), fft_.size());
            
            printf("\n");
            Timer::print();
        }
};

};


