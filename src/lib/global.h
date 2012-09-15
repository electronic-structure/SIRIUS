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
        
        /// cutoff for augmented-wave functions
        double aw_cutoff_;
        
        /// maximum number of muffin-tin points across all atom types
        int max_num_mt_points_;
        
        /// minimum muffin-tin radius
        double min_mt_radius_;
        
        /// number of magnetic field components
        //int nmag;
        
    public:
    
        Global() : lmax_apw_(lmax_apw_default),
                   lmax_rho_(lmax_rho_default),
                   lmax_pot_(lmax_pot_default),
                   aw_cutoff_(aw_cutoff_default)
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

        inline double aw_cutoff()
        {
            return aw_cutoff_;
        }

        inline double min_mt_radius()
        {
            return min_mt_radius_;
        }

        void initialize()
        {
            unit_cell::init();
            geometry::init();
            reciprocal_lattice::init();
            step_function::init();
           
            max_num_mt_points_ = 0;
            min_mt_radius_ = 1e100;
            
            for (int i = 0; i < num_atom_types(); i++)
            {
                 atom_type(i)->init(lmax_apw());
                 max_num_mt_points_ = std::max(max_num_mt_points_, atom_type(i)->num_mt_points());
                 min_mt_radius_ = std::min(min_mt_radius_, atom_type(i)->mt_radius());
            }

            for (int ic = 0; ic < num_atom_symmetry_classes(); ic++)
                atom_symmetry_class(ic)->init();

            for (int ia = 0; ia < num_atoms(); ia++)
                atom(ia)->init(lmax_pot());

            assert(num_atoms() != 0);
            assert(num_atom_types() != 0);
            assert(num_atom_symmetry_classes() != 0);
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


