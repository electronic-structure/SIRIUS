/** \file global.h

    \brief Global variables 
*/

/** \page stdvarname Standard variable names
    
    Below it the list of standard names for some of the loop variables:
    
    l - index of orbital quantum number \n
    m - index of azimutal quantum nuber \n
    lm - combined index of (l,m) quantum numbers \n
    ia - index of atom \n
    ic - index of atom class \n
    iat - index of atom type \n
    ir - index of r-point \n
    ig - index of G-vector \n
    idxlo - index of local orbital \n
    idxrf - index of radial function \n

    The loc suffix is added to the variable to indicate that it runs over local fraction of elements for the given
    MPI rank. Typical code looks like this:
    
    \code{.cpp}
        // zero array
        memset(&mt_val[0], 0, parameters_.num_atoms() * sizeof(T));
        
        // loop over local fraction of atoms
        for (int ialoc = 0; ialoc < parameters_.spl_num_atoms().local_size(); ialoc++)
        {
            // get global index of atom
            int ia = parameters_.spl_num_atoms(ialoc);

            int nmtp = parameters_.atom(ia)->num_mt_points();
           
            // integrate spgerical part of the function
            Spline<T> s(nmtp, parameters_.atom(ia)->type()->radial_grid());
            for (int ir = 0; ir < nmtp; ir++) s[ir] = f_mt<local>(0, ir, ialoc);
            mt_val[ia] = s.interpolate().integrate(2) * fourpi * y00;
        }

        // simple array synchronization
        Platform::allreduce(&mt_val[0], parameters_.num_atoms());
    \endcode
*/

namespace sirius {

/// Global variables and methods
class Global
{
    private:

        /// true if class was initialized
        bool initialized_;
    
        /// maximum l for APW functions
        int lmax_apw_;
        
        /// maximum l for plane waves
        int lmax_pw_;
        
        /// maximum l for density
        int lmax_rho_;
        
        /// maximum l for potential
        int lmax_pot_;

        /// maxim overall l
        int lmax_;
        
        /// cutoff for augmented-wave functions
        double aw_cutoff_;

        /// cutoff for plane-waves
        double pw_cutoff_;
        
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

        /// general purpose synchronization flag
        int sync_flag_;

        /// MPI grid dimensions
        std::vector<int> mpi_grid_dims_;
        
        /// MPI grid
        MPI_grid mpi_grid_;

        /// block size for block-cyclic data distribution  
        int cyclic_block_size_;
        
        /// starting time of the program
        timeval start_time_;

        /// type of eigen-value solver
        linalg_t eigen_value_solver_; 
        
        /// type of the processing unit
        processing_unit_t processing_unit_;

        GauntCoefficients gaunt_;
        
        /// Block-cyclic distribution of the first-variational states along columns of the MPI grid.
        /** Very important! The number of first-variational states is aligned in such a way that each row or 
            column MPI rank gets equal row or column fraction of the fv states. This is done in order to have a 
            simple acces to the fv states when, for example, spin blocks of the second-variational Hamiltonian are
            constructed. */
        splindex<block_cyclic> spl_fv_states_col_;

        /// Block-cyclic distribution of the first-variational states along rows of the MPI grid.
        splindex<block_cyclic> spl_fv_states_row_;
        
        splindex<block_cyclic> spl_spinor_wf_col_;
        
        splindex<block> sub_spl_spinor_wf_;

        splindex<block> sub_spl_fv_states_col_;

        #if defined(_SCALAPACK_) || defined(_ELPA_)
        /// BLACS communication context for an eigen-value solver and related operations
        int blacs_context_; // TODO: if more contexts are necessary, they should be moved to mpi_grid.h 
        #endif
        
        /// smearing function width
        double smearing_width_;
        
        /// type of potential
        potential_t potential_type_;

        /// type of basis
        basis_t basis_type_;
        
        /// step function is used in full-potential methods
        Step_function* step_function_;

        Reciprocal_lattice* reciprocal_lattice_;

        Unit_cell* unit_cell_;

        /// read from the input file if it exists
        void read_input()
        {
            std::string fname("sirius.json");
            
            int num_fft_threads = Platform::num_threads();

            if (Utils::file_exists(fname))
            {
                JSON_tree parser(fname);
                mpi_grid_dims_ = parser["mpi_grid_dims"].get(mpi_grid_dims_); 
                cyclic_block_size_ = parser["cyclic_block_size"].get(cyclic_block_size_);
                num_fft_threads = parser["num_fft_threads"].get(num_fft_threads);
                num_fv_states_ = parser["num_fv_states"].get(num_fv_states_);
                smearing_width_ = parser["smearing_width"].get(smearing_width_);
                
                if (parser.exist("eigen_value_solver"))
                {
                    std::string ev_solver_name;
                    parser["eigen_value_solver"] >> ev_solver_name;
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
                        error_local(__FILE__, __LINE__, "wrong eigen value solver");
                    }
                }

                if (parser.exist("processing_unit"))
                {
                    std::string pu;
                    parser["processing_unit"] >> pu;
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
                        error_local(__FILE__, __LINE__, "wrong processing unit");
                    }
                }

                if (parser.exist("potential_type"))
                {
                    std::string str;
                    parser["potential_type"] >> str;
                    if (str == "full_potential")
                    {
                        potential_type_ = full_potential;
                    }
                    else if (str == "ultrasoft_pseudopotential")
                    {
                        potential_type_ = ultrasoft_pseudopotential;
                    }
                    else
                    {
                        error_local(__FILE__, __LINE__, "wrong type of potential");
                    }
                }

                if (parser.exist("basis_type"))
                {
                    std::string str;
                    parser["basis_type"] >> str;
                    if (str == "apwlo")
                    {
                        basis_type_ = apwlo;
                    }
                    else if (str == "pw")
                    {
                        basis_type_ = pw;
                    }
                    else
                    {
                        error_local(__FILE__, __LINE__, "wrong type of basis");
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

    public:
    
        Global() : initialized_(false), lmax_apw_(lmax_apw_default), lmax_pw_(-1), lmax_rho_(lmax_rho_default), 
                   lmax_pot_(lmax_pot_default), aw_cutoff_(aw_cutoff_default), pw_cutoff_(pw_cutoff_default), 
                   num_fv_states_(-1), num_spins_(1), num_mag_dims_(0), so_correction_(false), uj_correction_(false), 
                   cyclic_block_size_(64), eigen_value_solver_(lapack),
                   #ifdef _GPU_
                   processing_unit_(gpu),
                   #else
                   processing_unit_(cpu),
                   #endif
                   #if defined(_SCALAPACK_) || defined(_ELPA_)
                   blacs_context_(-1),
                   #endif
                   smearing_width_(0.001), potential_type_(full_potential), basis_type_(apwlo), step_function_(NULL),
                   reciprocal_lattice_(NULL)
        {
            gettimeofday(&start_time_, NULL); // measure the start time
            read_input(); // read initial data from sirius.json
            unit_cell_ = new Unit_cell(potential_type_); // create new empty unit cell
        }
            
        ~Global()
        {
            clear();
            delete unit_cell_;
            #if defined(_SCALAPACK_) || defined(_ELPA_)
            if (eigen_value_solver() == scalapack || eigen_value_solver() == elpa) 
                linalg<scalapack>::free_blacs_context(blacs_context_);
            #endif
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
            return Utils::lmmax(lmax_apw_);
        }
        
        inline int lmax_pw()
        {
            return lmax_pw_;
        }

        inline int lmmax_pw()
        {
            return Utils::lmmax(lmax_pw_);
        }
        
        inline int lmax_rho()
        {
            return lmax_rho_;
        }

        inline int lmmax_rho()
        {
            return Utils::lmmax(lmax_rho_);
        }
        
        inline int lmax_pot()
        {
            return lmax_pot_;
        }

        inline int lmmax_pot()
        {
            return Utils::lmmax(lmax_pot_);
        }

        inline int lmax()
        {
            return lmax_;
        }

        inline int lmmax()
        {
            return Utils::lmmax(lmax_);
        }

        inline double aw_cutoff()
        {
            return aw_cutoff_;
        }

        inline void set_aw_cutoff(double aw_cutoff__)
        {
            aw_cutoff_ = aw_cutoff__;
        }

        /// Return plane-wave cutoff for G-vectors.
        inline double pw_cutoff()
        {
            return pw_cutoff_;
        }

        /// Set plane-wave cutoff.
        inline void set_pw_cutoff(double pw_cutoff__)
        {
            pw_cutoff_ = pw_cutoff__;
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
        
        inline MPI_grid& mpi_grid()
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

        inline linalg_t eigen_value_solver()
        {
            return eigen_value_solver_;
        }

        inline processing_unit_t processing_unit()
        {
            return processing_unit_;
        }

        inline double smearing_width()
        {
            return smearing_width_;
        }

        bool need_sv()
        {
            if (num_spins() == 2 || uj_correction() || so_correction()) return true;
            return false;
        }
        
        inline splindex<block_cyclic>& spl_fv_states_col()
        {
            return spl_fv_states_col_;
        }
        
        inline int spl_fv_states_col(int icol_loc)
        {
            return spl_fv_states_col_[icol_loc];
        }

        inline splindex<block_cyclic>& spl_fv_states_row()
        {
            return spl_fv_states_row_;
        }
        
        inline int spl_fv_states_row(int irow_loc)
        {
            return spl_fv_states_row_[irow_loc];
        }
        
        inline splindex<block_cyclic>& spl_spinor_wf_col()
        {
            return spl_spinor_wf_col_;
        }
        
        inline int spl_spinor_wf_col(int jloc)
        {
            return spl_spinor_wf_col_[jloc];
        }
        
        inline int num_sub_bands()
        {
            return sub_spl_spinor_wf_.local_size();
        }

        inline splindex<block>& sub_spl_fv_states_col()
        {
            return sub_spl_fv_states_col_;
        }
        
        inline int sub_spl_fv_states_col(int idx)
        {
            return sub_spl_fv_states_col_[idx];
        }

        inline int idxbandglob(int sub_index)
        {
            return spl_spinor_wf_col_[sub_spl_spinor_wf_[sub_index]];
        }
        
        inline int idxbandloc(int sub_index)
        {
            return sub_spl_spinor_wf_[sub_index];
        }

        /// Initialize the global variables
        void initialize()
        {
            if (initialized_) error_local(__FILE__, __LINE__, "Can't initialize global variables more than once.");

            if (basis_type() == pwlo)
            {
                lmax_pw_ = lmax_apw_;
                lmax_apw_ = -1;
            }

            if (basis_type() == pw)
            {
                lmax_apw_ = -1;
                lmax_rho_ = lmax_pot_ = -1;
            }

            lmax_ = std::max(std::max(std::max(lmax_pot_, lmax_rho_), lmax_apw_), lmax_pw_); 

            // initialize variables, related to the unit cell
            unit_cell_->initialize(lmax_apw(), lmax_pot(), num_mag_dims());

            // create a reciprocal lattice
            reciprocal_lattice_ = new Reciprocal_lattice(unit_cell_, pw_cutoff(), lmax_);

            if (basis_type() == apwlo || basis_type() == pwlo) step_function_ = new Step_function(unit_cell_, reciprocal_lattice_);

            gaunt_.set_lmax(std::max(lmax_apw(), lmax_pw()), std::max(lmax_apw(), lmax_pw()), lmax_pot());

            // check MPI grid dimensions and set a default grid if needed
            if (!mpi_grid_dims_.size()) 
            {
                mpi_grid_dims_ = std::vector<int>(1);
                mpi_grid_dims_[0] = Platform::num_mpi_ranks();
            }

            // setup MPI grid
            mpi_grid_.initialize(mpi_grid_dims_);
            
            if (num_fv_states_ < 0) num_fv_states_ = int(unit_cell_->num_valence_electrons() / 2.0) + 20;

            int nrow = mpi_grid().dimension_size(_dim_row_);
            int ncol = mpi_grid().dimension_size(_dim_col_);
            
            int irow = mpi_grid().coordinate(_dim_row_);
            int icol = mpi_grid().coordinate(_dim_col_);

            if (eigen_value_solver() == scalapack || eigen_value_solver() == elpa)
            {
                int n = num_fv_states_ / (ncol * cyclic_block_size()) + 
                        std::min(1, num_fv_states_ % (ncol * cyclic_block_size()));

                while ((n * ncol) % nrow) n++;
                
                num_fv_states_ = n * ncol * cyclic_block_size();

                #if defined(_SCALAPACK_) || defined(_ELPA_)
                int rc = (1 << _dim_row_) | (1 << _dim_col_);
                MPI_Comm comm = mpi_grid().communicator(rc);
                blacs_context_ = linalg<scalapack>::create_blacs_context(comm);

                mdarray<int, 2> map_ranks(nrow, ncol);
                for (int i = 0; i < nrow; i++)
                {
                    for (int j = 0; j < ncol; j++)
                    {
                        std::vector<int> xy(2);
                        xy[0] = j;
                        xy[1] = i;
                        map_ranks(i, j) = mpi_grid().cart_rank(comm, xy);
                    }
                }
                linalg<scalapack>::gridmap(&blacs_context_, map_ranks.get_ptr(), map_ranks.ld(), nrow, ncol);

                // check the grid
                int nrow1, ncol1, irow1, icol1;
                linalg<scalapack>::gridinfo(blacs_context_, &nrow1, &ncol1, &irow1, &icol1);

                if (irow != irow1 || icol != icol1 || nrow != nrow1 || ncol != ncol1) 
                {
                    std::stringstream s;
                    s << "wrong grid" << std::endl
                      << "            row | col | nrow | ncol " << std::endl
                      << " mpi_grid " << irow << " " << icol << " " << nrow << " " << ncol << std::endl  
                      << " blacs    " << irow1 << " " << icol1 << " " << nrow1 << " " << ncol1;
                    error_local(__FILE__, __LINE__, s);
                }
                #endif
            }

            num_bands_ = num_fv_states_ * num_spins_;
    
            // distribue first-variational states along columns
            spl_fv_states_col_.split(num_fv_states(), ncol, icol, cyclic_block_size());
   
            // distribue first-variational states along rows
            spl_fv_states_row_.split(num_fv_states(), nrow, irow, cyclic_block_size());

            // distribue spinor wave-functions along columns
            spl_spinor_wf_col_.split(num_bands(), ncol, icol, cyclic_block_size());
            
            // additionally split along rows 
            sub_spl_spinor_wf_.split(spl_spinor_wf_col_.local_size(), nrow, irow);
            
            sub_spl_fv_states_col_.split(spl_fv_states_col().local_size(), nrow, irow);

            // check if the distribution of fv states is consistent with the distribtion of spinor wave functions
            for (int ispn = 0; ispn < num_spins(); ispn++)
            {
                for (int i = 0; i < spl_fv_states_col_.local_size(); i++)
                {
                    if (spl_spinor_wf_col_[i + ispn * spl_fv_states_col_.local_size()] != 
                        spl_fv_states_col_[i] + ispn * num_fv_states())
                    {
                        error_local(__FILE__, __LINE__, "Wrong distribution of wave-functions");
                    }
                }
            }

            if (verbosity_level >= 3 && Platform::mpi_rank() == 0 && nrow * ncol > 1)
            {
                printf("\n");
                printf("table of column distribution of first-variational states\n");
                printf("(columns of the table correspond to column MPI ranks)\n");
                for (int i0 = 0; i0 < spl_fv_states_col_.local_size(0); i0++)
                {
                    for (int i1 = 0; i1 < ncol; i1++) printf("%6i", spl_fv_states_col_.global_index(i0, i1));
                    printf("\n");
                }
                
                printf("\n");
                printf("table of row distribution of first-variational states\n");
                printf("(columns of the table correspond to row MPI ranks)\n");
                for (int i0 = 0; i0 < spl_fv_states_row_.local_size(0); i0++)
                {
                    for (int i1 = 0; i1 < nrow; i1++) printf("%6i", spl_fv_states_row_.global_index(i0, i1));
                    printf("\n");
                }

                printf("\n");
                printf("First-variational states index -> (local index, rank) for column distribution\n");
                for (int i = 0; i < num_fv_states(); i++)
                {
                    printf("%6i -> (%6i %6i)\n", i, spl_fv_states_col_.location(_splindex_offs_, i), 
                                                    spl_fv_states_col_.location(_splindex_rank_, i));
                }
                
                printf("\n");
                printf("First-variational states index -> (local index, rank) for row distribution\n");
                for (int i = 0; i < num_fv_states(); i++)
                {
                    printf("%6i -> (%6i %6i)\n", i, spl_fv_states_row_.location(_splindex_offs_, i), 
                                                    spl_fv_states_row_.location(_splindex_rank_, i));
                }
                
                printf("\n");
                printf("table of column distribution of spinor wave functions\n");
                printf("(columns of the table correspond to MPI ranks)\n");
                for (int i0 = 0; i0 < spl_spinor_wf_col_.local_size(0); i0++)
                {
                    for (int i1 = 0; i1 < ncol; i1++) printf("%6i", spl_spinor_wf_col_.global_index(i0, i1));
                    printf("\n");
                }
            }
            
            if (Platform::mpi_rank() == 0 && verbosity_level >= 1) print_info();
            initialized_ = true;
        }

        /// Clear global variables
        void clear()
        {
            if (initialized_)
            {
                unit_cell_->clear();
                delete reciprocal_lattice_;
                delete step_function_;
                mpi_grid_.finalize();
                initialized_ = false;
            }
        }

        void print_info()
        {
            printf("\n");
            printf("SIRIUS version : %2i.%02i\n", major_version, minor_version);
            printf("git hash       : %s\n", git_hash);
            printf("build date     : %s\n", build_date);
            printf("start time     : %s\n", start_time("%c").c_str());
            printf("\n");
            printf("number of MPI ranks           : %i\n", Platform::num_mpi_ranks());
            printf("MPI grid                      :");
            for (int i = 0; i < mpi_grid_.num_dimensions(); i++) printf(" %i", mpi_grid_.size(1 << i));
            printf("\n");
            printf("number of OMP threads         : %i\n", Platform::num_threads()); 
            printf("number of OMP threads for FFT : %i\n", Platform::num_fft_threads()); 

            unit_cell_->print_info();
            reciprocal_lattice_->print_info();

            for (int i = 0; i < unit_cell_->num_atom_types(); i++) unit_cell_->atom_type(i)->print_info();

            printf("\n");
            printf("total number of aw muffin-tin basis functions : %i\n", unit_cell_->mt_aw_basis_size());
            printf("total number of lo basis functions : %i\n", unit_cell_->mt_lo_basis_size());
            printf("number of first-variational states : %i\n", num_fv_states());
            printf("number of bands                    : %i\n", num_bands());
            printf("number of spins                    : %i\n", num_spins());
            printf("number of magnetic dimensions      : %i\n", num_mag_dims());
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
            unit_cell_->write_cif();
        }
        
        //** /// Print run-time information.
        //** void print_rti()
        //** {
        //**     if (Platform::mpi_rank() == 0)
        //**     {
        //**         double total_core_leakage = 0.0;

        //**         printf("\n");
        //**         printf("Charges and magnetic moments\n");
        //**         for (int i = 0; i < 80; i++) printf("-");
        //**         printf("\n"); 
        //**         printf("atom      charge    core leakage");
        //**         if (num_mag_dims()) printf("              moment              |moment|");
        //**         printf("\n");
        //**         for (int i = 0; i < 80; i++) printf("-");
        //**         printf("\n"); 

        //**         for (int ia = 0; ia < num_atoms(); ia++)
        //**         {
        //**             double core_leakage = atom(ia)->symmetry_class()->core_leakage();
        //**             total_core_leakage += core_leakage;
        //**             printf("%4i  %10.6f  %10.8e", ia, rti().mt_charge[ia], core_leakage);
        //**             if (num_mag_dims())
        //**             {
        //**                 double v[] = {0, 0, 0};
        //**                 v[2] = rti().mt_magnetization[0][ia];
        //**                 if (num_mag_dims() == 3)
        //**                 {
        //**                     v[0] = rti().mt_magnetization[1][ia];
        //**                     v[1] = rti().mt_magnetization[2][ia];
        //**                 }
        //**                 printf("  (%8.4f %8.4f %8.4f)  %10.6f", v[0], v[1], v[2], Utils::vector_length(v));
        //**             }
        //**             printf("\n");
        //**         }
        //**         
        //**         printf("\n");
        //**         printf("interstitial charge   : %10.6f\n", rti().it_charge);
        //**         if (num_mag_dims())
        //**         {
        //**             double v[] = {0, 0, 0};
        //**             v[2] = rti().it_magnetization[0];
        //**             if (num_mag_dims() == 3)
        //**             {
        //**                 v[0] = rti().it_magnetization[1];
        //**                 v[1] = rti().it_magnetization[2];
        //**             }
        //**             printf("interstitial moment   : (%8.4f %8.4f %8.4f)\n", v[0], v[1], v[2]);
        //**             printf("interstitial |moment| : %10.6f\n", Utils::vector_length(v));
        //**         }
        //**         
        //**         printf("\n");
        //**         printf("total charge          : %10.6f\n", rti().total_charge);
        //**         printf("total core leakage    : %10.8e\n", total_core_leakage);
        //**         if (num_mag_dims())
        //**         {
        //**             double v[] = {0, 0, 0};
        //**             v[2] = rti().total_magnetization[0];
        //**             if (num_mag_dims() == 3)
        //**             {
        //**                 v[0] = rti().total_magnetization[1];
        //**                 v[1] = rti().total_magnetization[2];
        //**             }
        //**             printf("total moment          : (%8.4f %8.4f %8.4f)\n", v[0], v[1], v[2]);
        //**             printf("total |moment|        : %10.6f\n", Utils::vector_length(v));
        //**         }
        //**         printf("pseudo charge error : %18.12f\n", rti().pseudo_charge_error);
        //**         
        //**         printf("\n");
        //**         printf("Energy\n");
        //**         for (int i = 0; i < 80; i++) printf("-");
        //**         printf("\n"); 

        //**         printf("valence_eval_sum : %18.8f\n", rti().valence_eval_sum);
        //**         printf("core_eval_sum    : %18.8f\n", rti().core_eval_sum);

        //**         printf("kinetic energy   : %18.8f\n", kinetic_energy());
        //**         printf("<rho|V^{XC}>     : %18.8f\n", rti().energy_vxc);
        //**         printf("<rho|E^{XC}>     : %18.8f\n", rti().energy_exc);
        //**         printf("<mag|B^{XC}>     : %18.8f\n", rti().energy_bxc);
        //**         printf("<rho|V^{H}>      : %18.8f\n", rti().energy_vha);
        //**         printf("Total energy     : %18.8f\n", total_energy());

        //**         printf("\n");
        //**         printf("band gap (eV) : %18.8f\n", rti().band_gap * ha2ev);
        //**         printf("Efermi        : %18.8f\n", rti().energy_fermi);
        //**     }
        //** }

        void write_json_output()
        {
            if (Platform::mpi_rank() == 0)
            {
                std::string fname = std::string("output_") + start_time("%Y%m%d%H%M%S") + std::string(".json");
                JSON_write jw(fname);
                
                jw.single("git_hash", git_hash);
                jw.single("build_date", build_date);
                jw.single("num_ranks", Platform::num_mpi_ranks());
                jw.single("num_threads", Platform::num_threads());
                jw.single("num_fft_threads", Platform::num_fft_threads());
                jw.single("cyclic_block_size", cyclic_block_size());
                jw.single("mpi_grid", mpi_grid_dims_);
                std::vector<int> fftgrid(3);
                for (int i = 0; i < 3; i++) fftgrid[i] = reciprocal_lattice_->fft()->size(i);
                jw.single("fft_grid", fftgrid);
                jw.single("chemical_formula", unit_cell()->chemical_formula());
                jw.single("num_atoms", unit_cell()->num_atoms());
                jw.single("num_fv_states", num_fv_states());
                jw.single("num_bands", num_bands());
                jw.single("aw_cutoff", aw_cutoff());
                jw.single("pw_cutoff", pw_cutoff());
                jw.single("omega", unit_cell()->omega());
                
                //** if (num_mag_dims())
                //** {
                //**     std::vector<double> v(3, 0);
                //**     v[2] = rti().total_magnetization[0];
                //**     if (num_mag_dims() == 3)
                //**     {
                //**         v[0] = rti().total_magnetization[1];
                //**         v[1] = rti().total_magnetization[2];
                //**     }
                //**     jw.single("total_moment", v);
                //**     jw.single("total_moment_len", Utils::vector_length(&v[0]));
                //** }
                
                //** jw.single("total_energy", total_energy());
                //** jw.single("kinetic_energy", kinetic_energy());
                //** jw.single("energy_veff", rti_.energy_veff);
                //** jw.single("energy_vha", rti_.energy_vha);
                //** jw.single("energy_vxc", rti_.energy_vxc);
                //** jw.single("energy_bxc", rti_.energy_bxc);
                //** jw.single("energy_exc", rti_.energy_exc);
                //** jw.single("energy_enuc", rti_.energy_enuc);
                //** jw.single("core_eval_sum", rti_.core_eval_sum);
                //** jw.single("valence_eval_sum", rti_.valence_eval_sum);
                //** jw.single("band_gap", rti_.band_gap);
                //** jw.single("energy_fermi", rti_.energy_fermi);
                
                jw.single("timers", Timer::timers());
            }
        }

        inline GauntCoefficients& gaunt()
        {
            return gaunt_;
        }

        void create_storage_file()
        {
            if (Platform::mpi_rank() == 0) 
            {
                // create new hdf5 file
                HDF5_tree fout(storage_file_name, true);
                fout.create_node("parameters");
                fout.create_node("effective_potential");
                fout.create_node("effective_magnetic_field");
                fout.create_node("density");
                fout.create_node("magnetization");
                
                fout["parameters"].write("num_spins", num_spins());
                fout["parameters"].write("num_mag_dims", num_mag_dims());
                fout["parameters"].write("num_bands", num_bands());
            }
            Platform::barrier();
        }

        void update() // TODO: better way to update unit cell after relaxation
        {
            unit_cell_->update();
            reciprocal_lattice_->update();
            delete step_function_;
            step_function_ = new Step_function(unit_cell_, reciprocal_lattice_);
        }
       
        inline int blacs_context()
        {
            #ifdef _SCALAPACK_
            return blacs_context_;
            #else
            return -1;
            #endif
        }

        inline potential_t potential_type()
        {
            return potential_type_;
        }

        inline basis_t basis_type()
        {
            return basis_type_;
        }

        inline Step_function* step_function()
        {
            return step_function_;
        }

        inline double step_function(int ir)
        {
            return step_function_->theta_it(ir);
        }

        inline Reciprocal_lattice* reciprocal_lattice()
        {
            return reciprocal_lattice_;
        }

        inline Unit_cell* unit_cell()
        {
            return unit_cell_;
        }
};

};


