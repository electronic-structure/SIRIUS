namespace sirius {

class Global : public step_function
{
    private:
    
        /// maximum l for APW functions
        int lmax_apw_;
        
        /// maximum l for density
        int lmax_rho_;
        
        /// maximum l for potential
        int lmax_pot_;
        
        /// maximum number of muffin-tin points across all atom types
        int max_num_mt_points_;
        
        /// number of magnetic field components
        //int nmag;
        
    public:
    
        Global() : lmax_apw_(lmax_apw_default),
                   lmax_rho_(lmax_rho_default),
                   lmax_pot_(lmax_pot_default)
        {
        }

        void set_lmax_apw(int _lmax_apw)
        {
            lmax_apw_ = _lmax_apw;
        }

        void set_lmax_rho(int _lmax_rho)
        {
            lmax_rho_ = _lmax_rho;
        }

        void set_lmax_pot(int _lmax_pot)
        {
            lmax_pot_ = _lmax_pot;
        }

        inline int lmax_apw()
        {
            return lmax_apw_;
        }

        inline int lmmax_apw()
        {
            return (lmax_apw_ + 1) * (lmax_apw_ + 1);
        }
        
        inline int lmax_rho()
        {
            return lmax_rho_;
        }

        inline int lmmax_rho()
        {
            return (lmax_rho_ + 1) * (lmax_rho_ + 1);
        }
        
        inline int lmax_pot()
        {
            return lmax_pot_;
        }

        inline int lmmax_pot()
        {
            return (lmax_pot_ + 1) * (lmax_pot_ + 1);
        }

        inline int max_num_mt_points()
        {
            return max_num_mt_points_;
        }

        void initialize()
        {
            unit_cell::init();
            geometry::init();
            reciprocal_lattice::init();
            step_function::init();
           
            max_num_mt_points_ = 0;
            for (int i = 0; i < num_atom_types(); i++)
            {
                 atom_type(i)->init(lmax_apw());
                 max_num_mt_points_ = std::max(max_num_mt_points_, atom_type(i)->num_mt_points());
            }
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

Global global;

};


