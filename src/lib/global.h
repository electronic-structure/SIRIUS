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

        /// total number of augmented wave basis functions
        int num_aw_;

        //mdarray<complex16,3> complex_gaunt_;
        mdarray<std::vector< std::pair<int,complex16> >,2> complex_gaunt_packed_;
        
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

        inline int num_aw()
        {
            return num_aw_;
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

            num_aw_ = 0;
            for (int ia = 0; ia < num_atoms(); ia++)
            {
                atom(ia)->init(lmax_pot(), num_aw_);
                num_aw_ += atom(ia)->type()->indexb().num_aw();
            }

            //complex_gaunt_.set_dimensions(lmmax_pot(), lmmax_apw(), lmmax_apw());
            //complex_gaunt_.allocate();

            complex_gaunt_packed_.set_dimensions(lmmax_apw(), lmmax_apw());
            complex_gaunt_packed_.allocate();

            for (int l1 = 0; l1 <= lmax_apw(); l1++) 
            for (int m1 = -l1; m1 <= l1; m1++)
            {
                int lm1 = lm_by_l_m(l1, m1);
                for (int l2 = 0; l2 <= lmax_apw(); l2++)
                for (int m2 = -l2; m2 <= l2; m2++)
                {
                    int lm2 = lm_by_l_m(l2, m2);
                    for (int l3 = 0; l3 <= lmax_pot(); l3++)
                    for (int m3 = -l3; m3 <= l3; m3++)
                    {
                        int lm3 = lm_by_l_m(l3, m3);
                        complex16 z = SHT::complex_gaunt(l1, l3, l2, m1, m3, m2);
                        //complex_gaunt_(lm_by_l_m(l3, m3), lm_by_l_m(l1, m1), lm_by_l_m(l2, m2)) = SHT::complex_gaunt(l1, l3, l2, m1, m3, m2);
                        if (abs(z) > 1e-12) complex_gaunt_packed_(lm1, lm2).push_back(std::pair<int,complex16>(lm3, z));
                    }
                }
            }
                         
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
            printf("SIRIUS v0.3\n");
            printf("\n");

            unit_cell::print_info();
            reciprocal_lattice::print_info();

            printf("\n");
            for (int i = 0; i < num_atom_types(); i++)
                atom_type(i)->print_info();

            printf("\n");
            printf("total number of aw basis functions : %i\n", num_aw_);

            printf("\n");
            Timer::print();
        }
        
        template <typename T>
        inline void sum_L3_complex_gaunt(int lm1, int lm2, T* v, complex16& zsum)
        {
            for (int k = 0; k < (int)complex_gaunt_packed_(lm1, lm2).size(); k++)
            {
                int lm3 = complex_gaunt_packed_(lm1, lm2)[k].first;
                zsum += complex_gaunt_packed_(lm1, lm2)[k].second * v[complex_gaunt_packed_(lm1, lm2)[k].first];
            }
        }
};

Global global;

};


