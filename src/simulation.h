#ifndef __SIMULATION_H__
#define __SIMULATION_H__

#include "input.h"
#include "mpi_grid.h"
#include "step_function.h"

namespace sirius {

class Simulation_parameters
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
    
        /// Cutoff for augmented-wave functions.
        double aw_cutoff_;
    
        /// Cutoff for plane-waves (for density and potential expansion).
        double pw_cutoff_;
    
        /// Cutoff for |G+k| plane-waves.
        double gk_cutoff_;
        
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
    
        /// MPI grid dimensions
        std::vector<int> mpi_grid_dims_;
        
        /// Starting time of the program.
        timeval start_time_;
    
        ev_solver_t std_evp_solver_type_;
    
        ev_solver_t gen_evp_solver_type_;
    
        /// Type of the processing unit.
        processing_unit_t processing_unit_;
    
        /// Smearing function width.
        double smearing_width_;

        int num_fft_threads_;

        int num_fft_workers_;

        electronic_structure_method_t esm_type_;
        
        int cyclic_block_size_;

        Iterative_solver_input_section iterative_solver_input_section_;
        
        XC_functionals_input_section xc_functionals_input_section_;
        
        Mixer_input_section mixer_input_section_;
        
        /// Import data from initial input parameters.
        void import(Input_parameters const& iip__)
        {
            mpi_grid_dims_  = iip__.common_input_section_.mpi_grid_dims_;
            num_fv_states_  = iip__.common_input_section_.num_fv_states_;
            smearing_width_ = iip__.common_input_section_.smearing_width_;
            
            std::string evsn[] = {iip__.common_input_section_.std_evp_solver_type_, iip__.common_input_section_.gen_evp_solver_type_};
            ev_solver_t* evst[] = {&std_evp_solver_type_, &gen_evp_solver_type_};

            for (int i = 0; i < 2; i++)
            {
                std::string name = evsn[i];
                if (name == "lapack") 
                {
                    *evst[i] = ev_lapack;
                }
                else if (name == "scalapack") 
                {
                    *evst[i] = ev_scalapack;
                }
                else if (name == "elpa1") 
                {
                    *evst[i] = ev_elpa1;
                }
                else if (name == "elpa2") 
                {
                    *evst[i] = ev_elpa2;
                }
                else if (name == "magma") 
                {
                    *evst[i] = ev_magma;
                }
                else if (name == "plasma")
                {
                    *evst[i] = ev_plasma;
                }
                else if (name == "rs_gpu")
                {
                    *evst[i] = ev_rs_gpu;
                }
                else if (name == "rs_cpu")
                {
                    *evst[i] = ev_rs_cpu;
                }
                else
                {
                    TERMINATE("wrong eigen value solver");
                }
            }

            std::string pu = iip__.common_input_section_.processing_unit_;
            if (pu == "cpu" || pu == "CPU")
            {
                processing_unit_ = CPU;
            }
            else if (pu == "gpu" || pu == "GPU")
            {
                processing_unit_ = GPU;
            }
            else
            {
                TERMINATE("wrong processing unit");
            }

            std::string esm = iip__.common_input_section_.electronic_structure_method_;
            if (esm == "full_potential_lapwlo")
            {
                esm_type_ = full_potential_lapwlo;
            }
            else if (esm == "full_potential_pwlo")
            {
                esm_type_ = full_potential_pwlo;
            }
            else if (esm == "ultrasoft_pseudopotential")
            {
                esm_type_ = ultrasoft_pseudopotential;
            } 
            else if (esm == "norm_conserving_pseudopotential")
            {
                esm_type_ = norm_conserving_pseudopotential;
            }
            else
            {
                TERMINATE("wrong type of electronic structure method");
            }
        }
    
    public:
    
        //Real_space_prj* real_space_prj_;
    
        /// Initiail input parameters from the input file and command line.
        //input_parameters iip_;
    
        int work_load_;
    
        Simulation_parameters(Input_parameters const& iip__)
            : initialized_(false), 
              lmax_apw_(8), 
              lmax_pw_(-1), 
              lmax_rho_(8), 
              lmax_pot_(8), 
              aw_cutoff_(7.0), 
              pw_cutoff_(20.0), 
              gk_cutoff_(5.0), 
              num_fv_states_(-1), 
              num_spins_(1), 
              num_mag_dims_(0), 
              so_correction_(false), 
              uj_correction_(false),
              std_evp_solver_type_(ev_lapack),
              gen_evp_solver_type_(ev_lapack),
              processing_unit_(CPU),
              smearing_width_(0.001), 
              esm_type_(full_potential_lapwlo),
              cyclic_block_size_(32)
        {
            /* get the starting time */
            gettimeofday(&start_time_, NULL);
    
            iterative_solver_input_section_ = iip__.iterative_solver_input_section();
            xc_functionals_input_section_   = iip__.xc_functionals_input_section();
            mixer_input_section_            = iip__.mixer_input_section();

            cyclic_block_size_ = iip__.common_input_section_.cyclic_block_size_;
        }
            
        ~Simulation_parameters()
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

        inline void set_aw_cutoff(double aw_cutoff__)
        {
            aw_cutoff_ = aw_cutoff__;
        }

        /// Set plane-wave cutoff.
        inline void set_pw_cutoff(double pw_cutoff__)
        {
            pw_cutoff_ = pw_cutoff__;
        }
    
        inline void set_gk_cutoff(double gk_cutoff__)
        {
            gk_cutoff_ = gk_cutoff__;
        }
    
        inline void set_num_fv_states(int num_fv_states__)
        {
            num_fv_states_ = num_fv_states__;
        }

        inline void set_so_correction(bool so_correction__)
        {
            so_correction_ = so_correction__; 
        }
    
        inline void set_uj_correction(bool uj_correction__)
        {
            uj_correction_ = uj_correction__; 
        }
    
        inline int lmax_apw() const
        {
            return lmax_apw_;
        }
    
        inline int lmmax_apw() const
        {
            return Utils::lmmax(lmax_apw_);
        }
        
        inline int lmax_pw() const
        {
            return lmax_pw_;
        }
    
        inline int lmmax_pw() const
        {
            return Utils::lmmax(lmax_pw_);
        }
        
        inline int lmax_rho() const
        {
            return lmax_rho_;
        }
    
        inline int lmmax_rho() const
        {
            return Utils::lmmax(lmax_rho_);
        }
        
        inline int lmax_pot() const
        {
            return lmax_pot_;
        }
    
        inline int lmmax_pot() const
        {
            return Utils::lmmax(lmax_pot_);
        }

        inline int lmax_beta() const
        {
            STOP();
            return -1;
            //return unit_cell_->lmax_beta();
        }
    
        inline double aw_cutoff() const
        {
            return aw_cutoff_;
        }
    
        /// Return plane-wave cutoff for G-vectors.
        inline double pw_cutoff() const
        {
            return pw_cutoff_;
        }
    
        inline double gk_cutoff() const
        {
            return gk_cutoff_;
        }
    
        inline int num_fv_states() const
        {
            return num_fv_states_;
        }
    
        inline int num_bands() const
        {
            return num_bands_;
        }
        
        inline int num_spins() const
        {
            assert(num_spins_ == 1 || num_spins_ == 2);
            
            return num_spins_;
        }
    
        inline int num_mag_dims() const
        {
            assert(num_mag_dims_ == 0 || num_mag_dims_ == 1 || num_mag_dims_ == 3);
            
            return num_mag_dims_;
        }
    
        inline int max_occupancy() const
        {
            return (2 / num_spins());
        }
        
        inline bool so_correction() const
        {
            return so_correction_;
        }
        
        inline bool uj_correction() const
        {
            return uj_correction_;
        }
    
        inline bool initialized() const
        {
            return initialized_;
        }
    
        inline processing_unit_t processing_unit() const
        {
            return processing_unit_;
        }
    
        inline double smearing_width() const
        {
            return smearing_width_;
        }
    
        bool need_sv() const
        {
            if (num_spins() == 2 || uj_correction() || so_correction()) return true;
            return false;
        }
        
        inline std::vector<int> const& mpi_grid_dims() const
        {
            return mpi_grid_dims_;
        }

        inline int num_fft_threads() const
        {
            return num_fft_threads_;
        }
    
        inline int num_fft_workers() const
        {
            return num_fft_workers_;
        }

        inline int cyclic_block_size() const
        {
            return cyclic_block_size_;
        }
    
        /// Initialize the global variables
        void initialize();
    
        /// Clear global variables
        void clear();
    
        void print_info();
    
        void write_json_output();
    
        void create_storage_file();
    
        std::string start_time(const char* fmt);
    
        inline electronic_structure_method_t esm_type() const
        {
            return esm_type_;
        }
    
        inline wave_function_distribution_t wave_function_distribution()
        {
            switch (esm_type_)
            {
                case full_potential_lapwlo:
                case full_potential_pwlo:
                {
                    return block_cyclic_2d;
                    break;
                }
                case ultrasoft_pseudopotential:
                case norm_conserving_pseudopotential:
                {
                    return slab;
                    break;
                }
                default:
                {
                    TERMINATE("wrong method type");
                }
            }
            return block_cyclic_2d;
        }
    
        inline ev_solver_t std_evp_solver_type()
        {
            return std_evp_solver_type_;
        }
    
        inline ev_solver_t gen_evp_solver_type()
        {
            return gen_evp_solver_type_;
        }

        inline Mixer_input_section const& mixer_input_section() const
        {
            return mixer_input_section_;
        }

        inline XC_functionals_input_section const& xc_functionals_input_section() const
        {
            return xc_functionals_input_section_;
        }

        inline Iterative_solver_input_section const& iterative_solver_input_section() const
        {
            return iterative_solver_input_section_;
        }
};

class Simulation_context
{
    private:

        /// Parameters of simulcaiton.
        Simulation_parameters const& parameters_; 
    
        /// Communicator for this simulation.
        Communicator const& comm_;

        /// MPI grid for this simulation.
        MPI_grid mpi_grid_;

        
        /// Unit cell of the simulation.
        Unit_cell unit_cell_;

        /// Reciprocal lattice of the unit cell.
        Reciprocal_lattice* reciprocal_lattice_;

        /// Step function is used in full-potential methods.
        Step_function* step_function_;

        /// FFT wrapper for dense grid.
        FFT3D<CPU>* fft_;

        /// FFT wrapper for coarse grid.
        FFT3D<CPU>* fft_coarse_;

        #ifdef _GPU_
        FFT3D<GPU>* fft_gpu_;

        FFT3D<GPU>* fft_gpu_coarse_;
        #endif

    public:
        
        Simulation_context(Simulation_parameters const& parameters__,
                           Communicator const& comm__)
            : parameters_(parameters__),
              comm_(comm__),
              unit_cell_(parameters_.esm_type(), comm_, parameters_.processing_unit())
        {
        }

        inline bool full_potential()
        {
            return (parameters_.esm_type() == full_potential_lapwlo || parameters_.esm_type() == full_potential_pwlo);
        }

        /// Initialize the similation (can only be called once).
        void initialize()
        {
            /* check MPI grid dimensions and set a default grid if needed */
            auto mpi_grid_dims = parameters_.mpi_grid_dims();
            if (!mpi_grid_dims.size()) 
            {
                mpi_grid_dims = std::vector<int>(1);
                mpi_grid_dims[0] = comm_.size();
            }

            /* setup MPI grid */
            mpi_grid_ = MPI_grid(mpi_grid_dims, comm_);

            /* initialize variables, related to the unit cell */
            unit_cell_.initialize(parameters_.lmax_apw(), parameters_.lmax_pot(), parameters_.num_mag_dims());

            /* create FFT interface */
            fft_ = new FFT3D<CPU>(Utils::find_translation_limits(parameters_.pw_cutoff(), unit_cell_.reciprocal_lattice_vectors()),
                                  parameters_.num_fft_threads(), parameters_.num_fft_workers());
            
            fft_->init_gvec(parameters_.pw_cutoff(), unit_cell_.reciprocal_lattice_vectors());

            #ifdef _GPU_
            fft_gpu_ = new FFT3D<GPU>(fft_->grid_size(), 1);
            #endif
            
            if (parameters_.esm_type() == ultrasoft_pseudopotential ||
                parameters_.esm_type() == norm_conserving_pseudopotential)
            {
                /* create FFT interface for coarse grid */
                fft_coarse_ = new FFT3D<CPU>(Utils::find_translation_limits(parameters_.gk_cutoff() * 2, unit_cell_.reciprocal_lattice_vectors()),
                                             parameters_.num_fft_threads(), parameters_.num_fft_workers());
                
                fft_coarse_->init_gvec(parameters_.gk_cutoff() * 2, unit_cell_.reciprocal_lattice_vectors());

                #ifdef _GPU_
                fft_gpu_coarse_ = new FFT3D<GPU>(fft_coarse_->grid_size(), 2);
                #endif
            }
    
            if (unit_cell_.num_atoms() != 0) unit_cell_.symmetry()->check_gvec_symmetry(fft_);

            /* create a reciprocal lattice */
            int lmax = -1;
            switch (parameters_.esm_type())
            {
                case full_potential_lapwlo:
                {
                    lmax = parameters_.lmax_pot();
                    break;
                }
                case full_potential_pwlo:
                {
                    STOP();
                }
                case ultrasoft_pseudopotential:
                case norm_conserving_pseudopotential:
                {
                    lmax = 2 * unit_cell_.lmax_beta();
                    break;
                }
            }
            
            reciprocal_lattice_ = new Reciprocal_lattice(unit_cell_, parameters_.esm_type(), fft_, lmax, comm_);

            if (full_potential()) step_function_ = new Step_function(unit_cell_, reciprocal_lattice_, fft_, comm_);
        }

        Simulation_parameters const& parameters() const
        {
            return parameters_;
        }

        Unit_cell& unit_cell()
        {
            return unit_cell_;
        }

        Step_function* step_function()
        {
            return step_function_;
        }

        Reciprocal_lattice* reciprocal_lattice()
        {
            return reciprocal_lattice_;
        }

        inline FFT3D<CPU>* fft() const
        {
            return fft_;
        }

        inline FFT3D<CPU>* fft_coarse()
        {
            return fft_coarse_;
        }

        #ifdef _GPU_
        inline FFT3D<GPU>* fft_gpu()
        {
            return fft_gpu_;
        }

        inline FFT3D<GPU>* fft_gpu_coarse()
        {
            return fft_gpu_coarse_;
        }
        #endif

        Communicator const& comm() const
        {
            return comm_;
        }

};

};

#endif // __SIMULATION_H__
