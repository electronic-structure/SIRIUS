namespace sirius {

/** \file global.h
    \brief Global variables 
*/

/// Global variables
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
       
        /// number of spin componensts (1 or 2)
        int num_spins_;

        /// number of dimensions of the magnetization and effective magnetic field (0, 1 or 3)
        int num_mag_dims_;

        /// true if spin-orbit correction is applied
        bool so_correction_;
       
        /// true if UJ correction is applied
        bool uj_correction_;

        /// run-time information (energies, charges, etc.)
        run_time_info rti_;

    public:
    
        Global() : lmax_apw_(lmax_apw_default),
                   lmax_rho_(lmax_rho_default),
                   lmax_pot_(lmax_pot_default),
                   aw_cutoff_(aw_cutoff_default),
                   num_spins_(1),
                   num_mag_dims_(0),
                   so_correction_(false),
                   uj_correction_(false)
        {
        }
            
        ~Global()
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

        void set_num_mag_dims(int num_mag_dims__)
        {
            num_mag_dims_ = num_mag_dims__;
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
        
        inline int num_spins()
        {
            assert(num_spins_ == 1 || num_spins_ == 2);
            
            return num_spins_;
        }

        inline int num_mag_dims()
        {
            assert(num_mag_dims_ == 0 || num_mag_dims_ == 1 || num_mag_dims_ == 3);
            
            return num_mag_dims_;
        }

        inline int max_occupancy()
        {
            return (2 / num_spins());
        }
        
        inline run_time_info& rti()
        {
            return rti_;
        }
        
        inline void set_so_correction(bool so_correction__)
        {
            so_correction_ = so_correction__; 
        }

        inline void set_uj_correction(bool uj_correction__)
        {
            uj_correction_ = uj_correction__; 
        }

        inline bool so_correction()
        {
            return so_correction_;
        }
        
        inline bool uj_correction()
        {
            return uj_correction_;
        }

        /// Initialize the global variables
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
                atom(ia)->init(lmax_pot(), num_mag_dims_, mt_aw_basis_size_, mt_lo_basis_size_, mt_basis_size_);
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

            num_fv_states_ = int(num_valence_electrons() / 2.0) + 10;
            num_bands_ = num_fv_states_ * num_spins_;
        }

        /// Clear global variables
        void clear()
        {
            unit_cell::clear();
            reciprocal_lattice::clear();
        }

        void print_info()
        {
            if (Platform::verbose())
            {
                printf("\n");
                printf("SIRIUS v0.4\n");
                printf("git hash : %s\n", git_hash);
                printf("build date : %s\n", build_date);
                printf("\n");
                printf("number of MPI ranks   : %i\n", Platform::num_mpi_ranks());
                printf("number of OMP threads : %i\n", Platform::num_threads()); 

                unit_cell::print_info();
                reciprocal_lattice::print_info();

                printf("\n");
                for (int i = 0; i < num_atom_types(); i++)
                    atom_type(i)->print_info();

                printf("\n");
                printf("total number of aw muffin-tin basis functions : %i\n", mt_aw_basis_size());
                printf("total number of lo basis functions : %i\n", mt_lo_basis_size());
            }
        }
        
        void generate_radial_functions()
        {
            Timer t("sirius::Global::generate_radial_functions");
            
            for (int ic = 0; ic < num_atom_symmetry_classes(); ic++)
                atom_symmetry_class(ic)->generate_radial_functions();
        }
        
        void generate_radial_integrals()
        {
            Timer t("sirius::Global::generate_radial_integrals");
            
            for (int ic = 0; ic < num_atom_symmetry_classes(); ic++)
                atom_symmetry_class(ic)->generate_radial_integrals();

            for (int ia = 0; ia < num_atoms(); ia++)
                atom(ia)->generate_radial_integrals();
        }

        /// Get the total energy of the electronic subsystem.

        /** From the definition of the density functional we have:
            
            \f[
                E[\rho] = T[\rho] + E^{H}[\rho] + E^{XC}[\rho] + E^{ext}[\rho]
            \f]
            where \f$ T[\rho] \f$ is the kinetic energy, \f$ E^{H}[\rho] \f$ - electrostatic energy of
            electron-electron density interaction, \f$ E^{XC}[\rho] \f$ - exchange-correlation energy
            and \f$ E^{ext}[\rho] \f$ - energy in the external field of nuclei.
            
            Electrostatic and external field energies are grouped in the following way:
            \f[
                \frac{1}{2} \int \int \frac{\rho({\bf r})\rho({\bf r'}) d{\bf r} d{\bf r'}}{|{\bf r} - {\bf r'}|} + 
                    \int \rho({\bf r}) V^{nuc}({\bf r}) d{\bf r} = \frac{1}{2} \int V^{H}({\bf r})\rho({\bf r})d{\bf r} + 
                    \frac{1}{2} \int \rho({\bf r}) V^{nuc}({\bf r}) d{\bf r}
            \f]
            Here \f$ V^{H}({\bf r}) \f$ is the total (electron + nuclei) electrostatic potential returned by the 
            poisson solver. Next we transform the remaining term:
            \f[
                \frac{1}{2} \int \rho({\bf r}) V^{nuc}({\bf r}) d{\bf r} = 
                \frac{1}{2} \int \int \frac{\rho({\bf r})\rho^{nuc}({\bf r'}) d{\bf r} d{\bf r'}}{|{\bf r} - {\bf r'}|} = 
                \frac{1}{2} \int V^{H,el}({\bf r}) \rho^{nuc}({\bf r}) d{\bf r}
            \f]
        */
        double total_energy()
        {
            return kinetic_energy() + rti().energy_exc + 0.5 * rti().energy_vha + rti().energy_enuc;
        }
        
        inline double kinetic_energy()
        {
            return rti().core_eval_sum +rti().valence_eval_sum - rti().energy_veff - rti().energy_bxc; 
        }

        /// Print run-time information.
        void print_rti()
        {
            if (Platform::verbose())
            {
                double total_core_leakage = 0.0;

                printf("\n");
                printf("Charges and magnetic moments\n");
                for (int i = 0; i < 80; i++) printf("-");
                printf("\n"); 
                printf("atom      charge    core leakage");
                if (num_mag_dims())
                    printf("              moment              |moment|");
                printf("\n");
                for (int i = 0; i < 80; i++) printf("-");
                printf("\n"); 

                for (int ia = 0; ia < num_atoms(); ia++)
                {
                    double core_leakage = atom(ia)->symmetry_class()->core_leakage();
                    total_core_leakage += core_leakage;
                    printf("%4i  %10.6f  %10.8e", ia, rti().mt_charge[ia], core_leakage);
                    if (num_mag_dims())
                    {
                        double v[] = {0, 0, 0};
                        v[2] = rti().mt_magnetization[0][ia];
                        if (num_mag_dims() == 3)
                        {
                            v[0] = rti().mt_magnetization[1][ia];
                            v[1] = rti().mt_magnetization[2][ia];
                        }
                        printf("  (%8.4f %8.4f %8.4f)  %10.6f", v[0], v[1], v[2], vector_length(v));
                    }
                    printf("\n");
                }
                
                printf("\n");
                printf("interstitial charge   : %10.6f\n", rti().it_charge);
                if (num_mag_dims())
                {
                    double v[] = {0, 0, 0};
                    v[2] = rti().it_magnetization[0];
                    if (num_mag_dims() == 3)
                    {
                        v[0] = rti().it_magnetization[1];
                        v[1] = rti().it_magnetization[2];
                    }
                    printf("interstitial moment   : (%8.4f %8.4f %8.4f)\n", v[0], v[1], v[2]);
                    printf("interstitial |moment| : %10.6f\n", vector_length(v));
                }
                
                printf("\n");
                printf("total charge          : %10.6f\n", rti().total_charge);
                printf("total core leakage    : %10.8e\n", total_core_leakage);
                if (num_mag_dims())
                {
                    double v[] = {0, 0, 0};
                    v[2] = rti().total_magnetization[0];
                    if (num_mag_dims() == 3)
                    {
                        v[0] = rti().total_magnetization[1];
                        v[1] = rti().total_magnetization[2];
                    }
                    printf("total moment          : (%8.4f %8.4f %8.4f)\n", v[0], v[1], v[2]);
                    printf("total |moment|        : %10.6f\n", vector_length(v));
                }
                printf("pseudo charge error : %18.12f\n", rti().pseudo_charge_error);
                
                printf("\n");
                printf("Energy\n");
                for (int i = 0; i < 80; i++) printf("-");
                printf("\n"); 

                printf("kinetic energy   : %18.8f\n", kinetic_energy());
                printf("<rho|V^{XC}>     : %18.8f\n", rti().energy_vxc);
                printf("<rho|E^{XC}>     : %18.8f\n", rti().energy_exc);
                printf("<mag|B^{XC}>     : %18.8f\n", rti().energy_bxc);
                printf("<rho|V^{H}>      : %18.8f\n", rti().energy_vha);
                printf("Total energy     : %18.8f\n", total_energy());

                printf("\n");
                printf("band gap (eV) : %18.8f\n", rti().band_gap * ha2ev);
                printf("Efermi        : %18.8f\n", rti().energy_fermi);
            }
        }
};

};


