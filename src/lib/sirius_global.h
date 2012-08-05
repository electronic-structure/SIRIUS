namespace sirius {

class SiriusGlobal : public sirius_gvec
{
    public:
    
        void initialize()
        {
            sirius_unit_cell::init();
            sirius_geometry::init();
            sirius_gvec::init();
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

            sirius_unit_cell::print_info();

            /*printf("\n");
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


