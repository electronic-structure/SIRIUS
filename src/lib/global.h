namespace sirius {

class Global : public StepFunction
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
        
        // number of magnetic field components
        //int nmag;
        
        /// total number of MT basis functions
        int mt_basis_size_;

        /// total number of augmented wave basis functions in the MT (= number of matching coefficients for each plane-wave)
        int mt_aw_basis_size_;

        /// total number of local orbital basis functions
        int mt_lo_basis_size_;

        /// number of first-variational states
        int num_fv_states_;

        //mdarray<complex16,3> complex_gaunt_;
        mdarray<std::vector< std::pair<int,complex16> >,2> complex_gaunt_packed_;
        
        PeriodicFunction<double> charge_density_;
        
        PeriodicFunction<double> effective_potential_;

    public:
    
        Global() : lmax_apw_(lmax_apw_default),
                   lmax_rho_(lmax_rho_default),
                   lmax_pot_(lmax_pot_default),
                   aw_cutoff_(aw_cutoff_default)
        {
        }

        void set_lmax_apw(int lmax_apw__)
        {
            lmax_apw_ = lmax_apw__;
        }

        void set_lmax_rho(int lmax_rho__)
        {
            lmax_rho_ = lmax_rho__;
        }

        void set_lmax_pot(int lmax_pot__)
        {
            lmax_pot_ = lmax_pot__;
        }

        inline int lmax_apw()
        {
            return lmax_apw_;
        }

        inline int lmmax_apw()
        {
            return lmmax_by_lmax(lmax_apw_);
        }
        
        inline int lmax_rho()
        {
            return lmax_rho_;
        }

        inline int lmmax_rho()
        {
            return lmmax_by_lmax(lmax_rho_);
        }
        
        inline int lmax_pot()
        {
            return lmax_pot_;
        }

        inline int lmmax_pot()
        {
            return lmmax_by_lmax(lmax_pot_);
        }

        inline int max_num_mt_points()
        {
            return max_num_mt_points_;
        }

        inline double aw_cutoff()
        {
            return aw_cutoff_;
        }

        inline void set_aw_cutoff(double aw_cutoff__)
        {
            aw_cutoff_ = aw_cutoff__;
        }

        inline double min_mt_radius()
        {
            return min_mt_radius_;
        }

        /*!
            \brief Total number of the augmented radial basis functions
        */
        inline int mt_aw_basis_size()
        {
            return mt_aw_basis_size_;
        }

        /*!
            \brief Total number of local orbital basis functions
        */
        inline int mt_lo_basis_size()
        {
            return mt_lo_basis_size_;
        }

        /*!
            \brief Total number of the muffin-tin basis functions.

            Total number of MT basis functions equals to the sum of the total number of augmented radial 
            basis functions and the total number of local orbital basis functions across all atoms. It controls 
            the size of the first- and second-variational wave functions.
        */
        inline int mt_basis_size()
        {
            return mt_basis_size_;
        }

        inline int num_fv_states()
        {
            return num_fv_states_;
        }
        
        inline PeriodicFunction<double>& charge_density()
        {
            return charge_density_;
        }
        
        inline PeriodicFunction<double>& effective_potential()
        {
            return effective_potential_;
        }

        void initialize()
        {
            unit_cell::init();
            geometry::init();
            reciprocal_lattice::init();
            StepFunction::init();
           
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

            mt_basis_size_ = 0;
            mt_aw_basis_size_ = 0;
            mt_lo_basis_size_ = 0;
            for (int ia = 0; ia < num_atoms(); ia++)
            {
                atom(ia)->init(lmax_pot(), mt_aw_basis_size_, mt_lo_basis_size_, mt_basis_size_);
                mt_aw_basis_size_ += atom(ia)->type()->mt_aw_basis_size();
                mt_lo_basis_size_ += atom(ia)->type()->mt_lo_basis_size();
                mt_basis_size_ += atom(ia)->type()->mt_basis_size();
            }

            assert(mt_basis_size_ == mt_aw_basis_size_ + mt_lo_basis_size_);

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
                        if (abs(z) > 1e-12) complex_gaunt_packed_(lm1, lm2).push_back(std::pair<int,complex16>(lm3, z));
                    }
                }
            }
                         
            assert(num_atoms() != 0);
            assert(num_atom_types() != 0);
            assert(num_atom_symmetry_classes() != 0);

            num_fv_states_ = int(num_electrons() / 2.0) + 10;
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
            printf("total number of aw muffin-tin basis functions : %i\n", mt_aw_basis_size());
            printf("total number of lo basis functions : %i\n", mt_lo_basis_size());

            printf("\n");
            Timer::print();
        }
        
        template <typename T>
        inline void sum_L3_complex_gaunt(int lm1, int lm2, T* v, complex16& zsum)
        {
            for (int k = 0; k < (int)complex_gaunt_packed_(lm1, lm2).size(); k++)
                zsum += complex_gaunt_packed_(lm1, lm2)[k].second * v[complex_gaunt_packed_(lm1, lm2)[k].first];
        }
};

Global global;

};


