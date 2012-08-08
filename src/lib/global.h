namespace sirius {

class Global : public step_func
{
    private:
    
        /// maximum l for APW functions
        int lmaxapw_;
        
        /// maximum l for potential and dinsity
        int lmax;
        
    public:
    
        void initialize()
        {
            unit_cell::init();
            geometry::init();
            reciprocal_lattice::init();
            step_func::init();
            
            for (int i = 0; i < num_atom_types(); i++)
                 atom_type(i)->init(8);
        }
        
        void clear()
        {
            unit_cell::clear();
        }

        void print_info()
        {
            printf("\n");
            printf("SIRIUS v0.2\n");
            printf("\n");

            unit_cell::print_info();
            reciprocal_lattice::print_info();

            printf("\n");
            for (int i = 0; i < num_atom_types(); i++)
                atom_type(i)->print_info();

            printf("\n");
            Timer::print();
        }
};

Global sirius_global;

};


