namespace sirius {



/** \file global.h

    \brief Global variables 
*/

/// Global variables

/** This class should be created first.
*/
class Global : public StepFunction
{
    private:

        /// true if class was initialized
        bool initialized_;
    
        /// maximum l for APW functions
        int lmax_apw_;
        
        /// maximum l for density
        int lmax_rho_;
        
        /// maximum l for potential
        int lmax_pot_;
        
        /// cutoff for augmented-wave functions
        double aw_cutoff_;
        
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

        /// general purpose synchronization flag
        int sync_flag_;

        /// MPI grid dimensions
        std::vector<int> mpi_grid_dims_;
        
        /// MPI grid
        MPIGrid mpi_grid_;

        int cyclic_block_size_;
        
        splindex<block> spl_num_atoms_;
        
        splindex<block> spl_fft_size_;

        splindex<block> spl_num_atom_symmetry_classes_;

        timeval start_time_;

        linalg_t eigen_value_solver_; 
        
        processing_unit_t processing_unit_;

        /// read from the input file if it exists
        void read_input()
        {
            std::string fname("sirius.json");
            
            int num_fft_threads = Platform::num_threads();

            if (Utils::file_exists(fname))
            {
                JsonTree parser(fname);
                parser["mpi_grid_dims"] >> mpi_grid_dims_; 
                cyclic_block_size_ = parser["cyclic_block_size"].get<int>(cyclic_block_size_);
                num_fft_threads = parser["num_fft_threads"].get<int>(num_fft_threads);
                num_fv_states_ = parser["num_fv_states"].get<int>(num_fv_states_);
                
                if (parser.exist("eigen_value_solver"))
                {
                    std::string ev_solver_name = parser["eigen_value_solver"].get<std::string>();
                    if (ev_solver_name == "lapack") 
                    {
                        eigen_value_solver_ = lapack;
                    }
                    else if (ev_solver_name == "scalapack") 
                    {
                        eigen_value_solver_ = scalapack;
                    }
                    else if (ev_solver_name == "elpa") 
                    {
                        eigen_value_solver_ = elpa;
                    }
                    else if (ev_solver_name == "magma") 
                    {
                        eigen_value_solver_ = magma;
                    }
                    else
                    {
                        error(__FILE__, __LINE__, "wrong eigen value solver", fatal_err);
                    }
                }

                if (parser.exist("processing_unit"))
                {
                    std::string pu = parser["processing_unit"].get<std::string>();
                    if (pu == "cpu")
                    {
                        processing_unit_ = cpu;
                    }
                    else if (pu == "gpu")
                    {
                        processing_unit_ = gpu;
                    }
                    else
                    {
                        error(__FILE__, __LINE__, "wrong processing unit", fatal_err);
                    }
                }
            }

            Platform::set_num_fft_threads(std::min(num_fft_threads, Platform::num_threads()));
        }

        std::string start_time(const char* fmt)
        {
            char buf[100]; 
            
            tm* ptm = localtime(&start_time_.tv_sec); 
            strftime(buf, sizeof(buf), fmt, ptm); 
            return std::string(buf);
        }

        std::string chemical_formula()
        {
            std::string name;
            for (int i = 0; i < num_atom_types(); i++)
            {
                name += atom_type(i)->symbol();
                int n = 0;
                for (int j = 0; j < num_atoms(); j++)
                {
                    if (atom(j)->type_id() == atom_type(i)->id()) n++;
                }
                if (n != 1) 
                {
                    std::stringstream s;
                    s << n;
                    name = (name + s.str());
                }
            }

            return name;
        }

    public:
    
        Global() : initialized_(false),
                   lmax_apw_(lmax_apw_default),
                   lmax_rho_(lmax_rho_default),
                   lmax_pot_(lmax_pot_default),
                   aw_cutoff_(aw_cutoff_default),
                   num_fv_states_(-1),
                   num_spins_(1),
                   num_mag_dims_(0),
                   so_correction_(false),
                   uj_correction_(false),
                   cyclic_block_size_(16),
                   eigen_value_solver_(lapack),
                   #ifdef _GPU_
                   processing_unit_(gpu)
                   #else
                   processing_unit_(cpu)
                   #endif
        {
            gettimeofday(&start_time_, NULL);
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
            return Utils::lmmax_by_lmax(lmax_apw_);
        }
        
        inline int lmax_rho()
        {
            return lmax_rho_;
        }

        inline int lmmax_rho()
        {
            return Utils::lmmax_by_lmax(lmax_rho_);
        }
        
        inline int lmax_pot()
        {
            return lmax_pot_;
        }

        inline int lmmax_pot()
        {
            return Utils::lmmax_by_lmax(lmax_pot_);
        }

        inline double aw_cutoff()
        {
            return aw_cutoff_;
        }

        inline void set_aw_cutoff(double aw_cutoff__)
        {
            aw_cutoff_ = aw_cutoff__;
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

        inline void set_sync_flag(int flg)
        {
            sync_flag_ = flg;
            Platform::allreduce<op_max>(&sync_flag_, 1);
        }

        inline int sync_flag()
        {
            return sync_flag_;
        }
        
        inline MPIGrid& mpi_grid()
        {
            return mpi_grid_;
        }

        inline int cyclic_block_size()
        {
            return cyclic_block_size_;
        }

        inline bool initialized()
        {
            return initialized_;
        }

        inline splindex<block>& spl_num_atoms()
        {
            return spl_num_atoms_;
        }

        inline int spl_num_atoms(int i)
        {
            return spl_num_atoms_[i];
        }
        
        inline splindex<block>& spl_fft_size()
        {
            return spl_fft_size_;
        }

        inline int spl_fft_size(int i)
        {
            return spl_fft_size_[i];
        }

        inline splindex<block>& spl_num_atom_symmetry_classes()
        {
            return spl_num_atom_symmetry_classes_;
        }

        inline int spl_num_atom_symmetry_classes(int i)
        {
            return spl_num_atom_symmetry_classes_[i];
        }

        inline linalg_t eigen_value_solver()
        {
            return eigen_value_solver_;
        }

        inline processing_unit_t processing_unit()
        {
            return processing_unit_;
        }

        /// Initialize the global variables
        void initialize()
        {
            if (initialized_) error(__FILE__, __LINE__, "Can't initialize global variables more than once.");

            read_input();
            
            // initialize variables, related to the unit cell
            UnitCell::init(lmax_apw(), lmax_pot(), num_mag_dims());
            
            ReciprocalLattice::init();
            StepFunction::init();

            // check MPI grid dimensions and set a default grid if needed
            if (!mpi_grid_dims_.size()) mpi_grid_dims_ = Utils::intvec(Platform::num_mpi_ranks());

            // setup MPI grid
            mpi_grid_.initialize(mpi_grid_dims_);
            
            if (num_fv_states_ < 0) num_fv_states_ = int(num_valence_electrons() / 2.0) + 20;

            if (eigen_value_solver() == scalapack || eigen_value_solver() == elpa)
            {
                int ncol = mpi_grid_.dimension_size(2);

                int n = num_fv_states_ / (ncol * cyclic_block_size_);
                
                num_fv_states_ = (n + 1) * ncol * cyclic_block_size_;
            }

            num_bands_ = num_fv_states_ * num_spins_;

            spl_num_atoms_.split(num_atoms(), Platform::num_mpi_ranks(), Platform::mpi_rank());
            
            spl_fft_size_.split(fft().size(), Platform::num_mpi_ranks(), Platform::mpi_rank());
            
            spl_num_atom_symmetry_classes_.split(num_atom_symmetry_classes(), Platform::num_mpi_ranks(), 
                                                 Platform::mpi_rank());
            
            initialized_ = true;
        }

        /// Clear global variables
        void clear()
        {
            UnitCell::clear();
            ReciprocalLattice::clear();

            mpi_grid_.finalize();

            initialized_ = false;
        }

        void print_info()
        {
            if (Platform::verbose())
            {
                printf("\n");
                printf("SIRIUS 0.7\n");
                printf("git hash : %s\n", git_hash);
                printf("build date : %s\n", build_date);
                printf("start time : %s\n", start_time("%c").c_str());
                printf("\n");
                printf("number of MPI ranks           : %i\n", Platform::num_mpi_ranks());
                printf("MPI grid                      :");
                for (int i = 0; i < mpi_grid_.num_dimensions(); i++) printf(" %i", mpi_grid_.size(1 << i));
                printf("\n");
                printf("number of OMP threads         : %i\n", Platform::num_threads()); 
                printf("number of OMP threads for FFT : %i\n", Platform::num_fft_threads()); 

                UnitCell::print_info();
                ReciprocalLattice::print_info();
                StepFunction::print_info();

                printf("\n");
                for (int i = 0; i < num_atom_types(); i++) atom_type(i)->print_info();

                printf("\n");
                printf("total number of aw muffin-tin basis functions : %i\n", mt_aw_basis_size());
                printf("total number of lo basis functions : %i\n", mt_lo_basis_size());
                printf("number of first-variational states : %i\n", num_fv_states());
                printf("\n");
                printf("eigen-value solver: ");
                switch (eigen_value_solver())
                {
                    case lapack:
                    {
                        printf("LAPACK\n");
                        break;
                    }
                    case scalapack:
                    {
                        printf("ScaLAPACK, block size %i\n", cyclic_block_size());
                        break;
                    }
                    case elpa:
                    {
                        printf("ELPA, block size %i\n", cyclic_block_size());
                        break;
                    }
                    case magma:
                    {
                        printf("MAGMA\n");
                        break;
                    }
                }
                printf("processing unit : ");
                switch (processing_unit())
                {
                    case cpu:
                    {
                        printf("CPU\n");
                        break;
                    }
                    case gpu:
                    {
                        printf("GPU\n");
                        break;
                    }
                }
            }
        }
        
        void generate_radial_functions()
        {
            Timer t("sirius::Global::generate_radial_functions");
           
            for (int icloc = 0; icloc < spl_num_atom_symmetry_classes().local_size(); icloc++)
                atom_symmetry_class(spl_num_atom_symmetry_classes(icloc))->generate_radial_functions();

            for (int ic = 0; ic < num_atom_symmetry_classes(); ic++)
            {
                int rank = spl_num_atom_symmetry_classes().location(1, ic);
                atom_symmetry_class(ic)->sync_radial_functions(rank);
            }

            //if (Platform::mpi_rank() == 0)
            //{
            //    FILE* fout = fopen("enu.txt", "w");
            //    for (int ic = 0; ic < num_atom_symmetry_classes(); ic++)
            //        atom_symmetry_class(ic)->write_enu(fout);
            //    fclose(fout);
            //}
        }
        
        void generate_radial_integrals()
        {
            Timer t("sirius::Global::generate_radial_integrals");
            
            for (int icloc = 0; icloc < spl_num_atom_symmetry_classes().local_size(); icloc++)
                atom_symmetry_class(spl_num_atom_symmetry_classes(icloc))->generate_radial_integrals();

            for (int ic = 0; ic < num_atom_symmetry_classes(); ic++)
            {
                int rank = spl_num_atom_symmetry_classes().location(1, ic);
                atom_symmetry_class(ic)->sync_radial_integrals(rank);
            }

            for (int ialoc = 0; ialoc < spl_num_atoms().local_size(); ialoc++)
                atom(spl_num_atoms(ialoc))->generate_radial_integrals();

            for (int ia = 0; ia < num_atoms(); ia++)
            {
                int rank = spl_num_atoms().location(1, ia);
                atom(ia)->sync_radial_integrals(rank);
            }
        }

        void solve_free_atoms()
        {
            splindex<block> spl_num_atom_types(num_atom_types(), Platform::num_mpi_ranks(), Platform::mpi_rank());

            std::vector<double> enu;
            for (int i = 0; i < spl_num_atom_types.local_size(); i++)
                atom_type(spl_num_atom_types[i])->solve_free_atom(1e-8, 1e-5, 1e-4, enu);

            for (int i = 0; i < num_atom_types(); i++)
            {
                int rank = spl_num_atom_types.location(1, i);
                atom_type(i)->sync_free_atom(rank);
            }
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
            assert(rti().energy_exc == rti().energy_exc);
            assert(rti().energy_vha == rti().energy_vha);
            assert(rti().energy_enuc == rti().energy_enuc);

            return (kinetic_energy() + rti().energy_exc + 0.5 * rti().energy_vha + rti().energy_enuc);
        }
        
        inline double kinetic_energy()
        {
            assert(rti().core_eval_sum == rti().core_eval_sum);
            assert(rti().valence_eval_sum == rti().valence_eval_sum);
            assert(rti().energy_veff == rti().energy_veff);
            assert(rti().energy_bxc == rti().energy_bxc);

            return (rti().core_eval_sum + rti().valence_eval_sum - rti().energy_veff - rti().energy_bxc); 
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
                        printf("  (%8.4f %8.4f %8.4f)  %10.6f", v[0], v[1], v[2], Utils::vector_length(v));
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
                    printf("interstitial |moment| : %10.6f\n", Utils::vector_length(v));
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
                    printf("total |moment|        : %10.6f\n", Utils::vector_length(v));
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

        void write_json_output()
        {
            if (Platform::verbose())
            {
                std::string fname = std::string("output_") + start_time("%Y%m%d%H%M%S") + std::string(".json");
                json_write jw(fname);
                
                jw.single("git_hash", git_hash);
                jw.single("build_date", build_date);
                jw.single("num_ranks", Platform::num_mpi_ranks());
                jw.single("num_threads", Platform::num_threads());
                jw.single("num_fft_threads", Platform::num_fft_threads());
                jw.single("cyclic_block_size", cyclic_block_size());
                std::vector<int> mpigrid;
                for (int i = 0; i < mpi_grid_.num_dimensions(); i++) mpigrid.push_back(mpi_grid_.size(1 << i));
                jw.single("mpi_grid", mpigrid);
                std::vector<int> fftgrid(3);
                for (int i = 0; i < 3; i++) fftgrid[i] = fft().size(i);
                jw.single("fft_grid", fftgrid);
                jw.single("chemical_formula", chemical_formula());
                jw.single("num_atoms", num_atoms());
                jw.single("num_fv_states", num_fv_states());
                jw.single("num_bands", num_bands());
                jw.single("aw_cutoff", aw_cutoff());
                
                if (num_mag_dims())
                {
                    std::vector<double> v(3, 0);
                    v[2] = rti().total_magnetization[0];
                    if (num_mag_dims() == 3)
                    {
                        v[0] = rti().total_magnetization[1];
                        v[1] = rti().total_magnetization[2];
                    }
                    jw.single("total_moment", v);
                    jw.single("total_moment_len", Utils::vector_length(&v[0]));
                }
                
                jw.single("total_energy", total_energy());

                jw.single("timers", Timer::timer_descriptors());
            }
        }
};

};


