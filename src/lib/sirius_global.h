namespace sirius {

class SiriusGlobal : public sirius_step_func
{
    public:
    
        void initialize()
        {
            sirius_unit_cell::init();
            sirius_geometry::init();
            sirius_gvec::init();
            sirius_step_func::init();
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
            sirius_gvec::print_info();

            printf("\n");
            Timer::print();
        }
};

};


