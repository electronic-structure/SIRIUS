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
        
        /// total number of MT basis functions
        int mt_basis_size_;
        
        /// maximum number of MT basis functions across all atoms
        int max_mt_basis_size_;

        /// maximum number of MT radial basis functions across all atoms
        int max_mt_radial_basis_size_;

        /// total number of augmented wave basis functions in the MT (= number of matching coefficients for each plane-wave)
        int mt_aw_basis_size_;

        /// total number of local orbital basis functions
        int mt_lo_basis_size_;

        /// number of first-variational states
        int num_fv_states_;

        /// number of bands (= number of spinor states)
        int num_bands_;

        PeriodicFunction<double> charge_density_;
        
        PeriodicFunction<double> magnetization_[3];
        
        PeriodicFunction<double> effective_potential_;
        
        PeriodicFunction<double> effective_magnetic_field_[3];
        
        /// number of spin componensts (1 or 2)
        int num_spins_;

        /// number of components of density matrix (1, 2 or 4)
        int num_dmat_;

    public:
    
        Global() : lmax_apw_(lmax_apw_default),
                   lmax_rho_(lmax_rho_default),
                   lmax_pot_(lmax_pot_default),
                   aw_cutoff_(aw_cutoff_default),
                   num_spins_(1),
                   num_dmat_(1)
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

        void set_num_spins(int num_spins__)
        {
            num_spins_ = num_spins__;
        }

        void set_num_dmat(int num_dmat__)
        {
            num_dmat_ = num_dmat__;
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
        
        inline int max_mt_basis_size()
        {
            return max_mt_basis_size_;
        }

        inline int max_mt_radial_basis_size()
        {
            return max_mt_radial_basis_size_;
        }

        inline int num_fv_states()
        {
            return num_fv_states_;
        }

        inline int num_bands()
        {
            return num_bands_;
        }
        
        inline PeriodicFunction<double>& charge_density()
        {
            return charge_density_;
        }
        
        inline PeriodicFunction<double>& magnetization(int i)
        {
            assert(i >= 0 && i < 3);

            return magnetization_[i];
        }
        
        inline PeriodicFunction<double>& effective_potential()
        {
            return effective_potential_;
        }
        
        inline PeriodicFunction<double>& effective_magnetic_field(int i)
        {
            assert(i >= 0 && i < 3);
            return effective_magnetic_field_[i];
        }

        inline int num_spins()
        {
            assert(num_spins_ == 1 || num_spins_ == 2);
            
            return num_spins_;
        }

        inline int num_dmat()
        {
            assert(num_dmat_ == 1 || num_dmat_ == 2 || num_dmat_ == 4);
            
            return num_dmat_;
        }

        inline int max_occupancy()
        {
            return (2 / num_spins());
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
            max_mt_basis_size_ = 0;
            max_mt_radial_basis_size_ = 0;
            for (int ia = 0; ia < num_atoms(); ia++)
            {
                atom(ia)->init(lmax_pot(), mt_aw_basis_size_, mt_lo_basis_size_, mt_basis_size_);
                mt_aw_basis_size_ += atom(ia)->type()->mt_aw_basis_size();
                mt_lo_basis_size_ += atom(ia)->type()->mt_lo_basis_size();
                mt_basis_size_ += atom(ia)->type()->mt_basis_size();
                max_mt_basis_size_ = std::max(max_mt_basis_size_, atom(ia)->type()->mt_basis_size());
                max_mt_radial_basis_size_ = std::max(max_mt_radial_basis_size_, atom(ia)->type()->mt_radial_basis_size());
            }

            assert(mt_basis_size_ == mt_aw_basis_size_ + mt_lo_basis_size_);
            assert(num_atoms() != 0);
            assert(num_atom_types() != 0);
            assert(num_atom_symmetry_classes() != 0);

            num_fv_states_ = int(num_electrons() / 2.0) + 10;
            num_bands_ = num_fv_states_ * num_spins_;
        }

        void clear()
        {
            unit_cell::clear();
        }

        void print_info()
        {
            printf("\n");
            printf("SIRIUS v0.3\n");
            printf("git hash : %s\n", git_hash);
            printf("build date : %s\n", build_date);
            int num_threads = 1;
            #pragma omp parallel default(shared)
            {
                if (omp_get_thread_num() == 0)
                    num_threads = omp_get_num_threads();
            }
            printf("\n");
            printf("number of OMP threads : %i\n", num_threads); 

            unit_cell::print_info();
            reciprocal_lattice::print_info();

            printf("\n");
            for (int i = 0; i < num_atom_types(); i++)
                atom_type(i)->print_info();

            printf("\n");
            printf("total number of aw muffin-tin basis functions : %i\n", mt_aw_basis_size());
            printf("total number of lo basis functions : %i\n", mt_lo_basis_size());
        }
        
        void generate_radial_functions()
        {
            Timer t("sirius::global::generate_radial_functions");
            
            for (int ic = 0; ic < num_atom_symmetry_classes(); ic++)
                atom_symmetry_class(ic)->generate_radial_functions();
        }

        void generate_radial_integrals()
        {
            Timer t("sirius::global::generate_radial_integrals");
            
            for (int ic = 0; ic < num_atom_symmetry_classes(); ic++)
                atom_symmetry_class(ic)->generate_radial_integrals();

            for (int ia = 0; ia < num_atoms(); ia++)
                atom(ia)->generate_radial_integrals(lmax_pot(), &effective_potential().f_rlm(0, 0, ia));
        }

        void zero_density()
        {
            charge_density().zero();
            if (num_spins() == 2)
                for (int i = 0; i < num_dmat() - 1; i++)
                    magnetization(i).zero();
        }
};

Global global;

};


