namespace sirius {

class SiriusGlobal : public sirius_gvec
{
    public:
    
        void initialize()
        {
            sirius_unit_cell::init();
            sirius_geometry::init();
            init_fft_grid();
        }
        
        void clear()
        {
            sirius_unit_cell::clear();
        }

        void print_info()
        {
            printf("\n");
            printf("SIRIUS v0.2\n");
            printf("\n");

            printf("lattice vectors\n");
            for (int i = 0; i < 3; i++)
                printf("  a%1i : %18.10f %18.10f %18.10f \n", i + 1, lattice_vectors(i, 0), 
                                                                     lattice_vectors(i, 1), 
                                                                     lattice_vectors(i, 2)); 
            printf("reciprocal lattice vectors\n");
            for (int i = 0; i < 3; i++)
                printf("  b%1i : %18.10f %18.10f %18.10f \n", i + 1, reciprocal_lattice_vectors(i, 0), 
                                                                     reciprocal_lattice_vectors(i, 1), 
                                                                     reciprocal_lattice_vectors(i, 2));
            
            printf("\n"); 
            printf("number of atom types : %i\n", num_atom_types());
            for (int i = 0; i < num_atom_types(); i++)
            {
                int id = atom_type_id(i);
                printf("type id : %i   symbol : %2s   label : %2s   mt_radius : %10.6f\n", id,
                                                                                           atom_type_by_id(id)->symbol().c_str(), 
                                                                                           atom_type_by_id(id)->label().c_str(),
                                                                                           atom_type_by_id(id)->mt_radius()); 
            }

            printf("number of atoms : %i\n", num_atoms());
            printf("number of symmetry classes : %i\n", num_symmetry_classes());

            printf("\n"); 
            printf("atom id    type id    class id\n");
            printf("------------------------------\n");
            for (int i = 0; i < num_atoms(); i++)
            {
                printf("%6i     %6i      %6i\n", i, atom(i)->type_id(), atom(i)->symmetry_class_id()); 
           
            }

            /*printf("\n");
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
            Timer::print();*/
        }
};

};


