// Copyright (c) 2013-2015 Anton Kozhevnikov, Thomas Schulthess
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that 
// the following conditions are met:
// 
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the 
//    following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions 
//    and the following disclaimer in the documentation and/or other materials provided with the distribution.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED 
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR 
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/** \file simulation.h
 *   
 *  \brief Contains definition and implementation of Simulation_parameters and Simulation_context classes.
 */

#ifndef __SIMULATION_CONTEXT_H__
#define __SIMULATION_CONTEXT_H__

#include "simulation_parameters.h"
#include "mpi_grid.h"
#include "step_function.h"
#include "real_space_prj.h"
#include "version.h"
#include "debug.hpp"

namespace sirius {

class Simulation_context
{
    private:

        /// Parameters of simulcaiton.
        Simulation_parameters parameters_; 
    
        /// Communicator for this simulation.
        Communicator comm_;

        /// MPI grid for this simulation.
        MPI_grid* mpi_grid_;
        
        /// Unit cell of the simulation.
        Unit_cell unit_cell_;

        /// Reciprocal lattice of the unit cell.
        Reciprocal_lattice* reciprocal_lattice_;

        /// Step function is used in full-potential methods.
        Step_function* step_function_;

        /// FFT wrapper for dense grid.
        std::vector<FFT3D*> fft_;

        Gvec gvec_;

        /// FFT wrapper for coarse grid.
        std::vector<FFT3D*> fft_coarse_;

        Gvec gvec_coarse_;

        int gpu_thread_id_;

        Real_space_prj* real_space_prj_;

        /// Creation time of the context.
        timeval start_time_;

        std::string start_time_tag_;

        double iterative_solver_tolerance_;

        ev_solver_t std_evp_solver_type_;

        ev_solver_t gen_evp_solver_type_;
        
        bool initialized_;

    public:
        
        Simulation_context(Simulation_parameters const& parameters__,
                           Communicator const& comm__)
            : parameters_(parameters__),
              comm_(comm__),
              unit_cell_(parameters_, comm_),
              reciprocal_lattice_(nullptr),
              step_function_(nullptr),
              real_space_prj_(nullptr),
              iterative_solver_tolerance_(parameters_.iterative_solver_input_section().tolerance_),
              std_evp_solver_type_(ev_lapack),
              gen_evp_solver_type_(ev_lapack),
              initialized_(false)
        {
            PROFILE();

            gettimeofday(&start_time_, NULL);
            
            tm const* ptm = localtime(&start_time_.tv_sec); 
            char buf[100];
            strftime(buf, sizeof(buf), "%Y%m%d%H%M%S", ptm);
            start_time_tag_ = std::string(buf);

            unit_cell_.import(parameters_.unit_cell_input_section());

        }

        ~Simulation_context()
        {
            PROFILE();

            for (auto obj: fft_) delete obj;
            for (auto obj: fft_coarse_) delete obj;
            if (reciprocal_lattice_ != nullptr) delete reciprocal_lattice_;
            if (step_function_ != nullptr) delete step_function_;
            if (real_space_prj_ != nullptr) delete real_space_prj_;
            delete mpi_grid_;
        }

        /// Initialize the similation (can only be called once).
        void initialize()
        {
            PROFILE();

            if (initialized_) TERMINATE("Simulation context is already initialized.");
            
            /* check if we can use a GPU device */
            if (parameters_.processing_unit() == GPU)
            {
                #ifndef __GPU
                TERMINATE_NO_GPU
                #endif
            }

            switch (parameters_.esm_type())
            {
                case full_potential_lapwlo:
                {
                    break;
                }
                case full_potential_pwlo:
                {
                    parameters_.set_lmax_pw(parameters_.lmax_apw());
                    parameters_.set_lmax_apw(-1);
                    break;
                }
                case ultrasoft_pseudopotential:
                case norm_conserving_pseudopotential:
                {
                    parameters_.set_lmax_apw(-1);
                    parameters_.set_lmax_rho(-1);
                    parameters_.set_lmax_pot(-1);
                    break;
                }
            }

            /* check MPI grid dimensions and set a default grid if needed */
            auto mpi_grid_dims = parameters_.mpi_grid_dims();
            if (!mpi_grid_dims.size()) 
            {
                mpi_grid_dims = std::vector<int>(1);
                mpi_grid_dims[0] = comm_.size();
            }
            parameters_.set_mpi_grid_dims(mpi_grid_dims);

            /* setup MPI grid */
            mpi_grid_ = new MPI_grid(mpi_grid_dims, comm_);

            /* initialize variables, related to the unit cell */
            unit_cell_.initialize();

            #ifdef __PRINT_MEMORY_USAGE
            MEMORY_USAGE_INFO();
            #endif

            if (comm_.rank() == 0)
            {
                unit_cell_.write_cif();
                unit_cell_.write_json();
            }

            parameters_.set_lmax_beta(unit_cell_.lmax_beta());

            auto rlv = unit_cell_.reciprocal_lattice_vectors();

            int nfft_threads = parameters_.num_fft_threads();
            int nfft_workers = parameters_.num_fft_workers();

            bool do_parallel_fft = false; //parameters_.full_potential() || mpi_grid_->dimension_size(_dim_row_) == 1)

            if (do_parallel_fft)
            {
                nfft_workers *= nfft_threads;
                nfft_threads = 1;
            }

            gpu_thread_id_ = -1;
            if (nfft_threads > 1) gpu_thread_id_ = nfft_threads - 1;

            for (int tid = 0; tid < nfft_threads; tid++)
            {
                /* in case of parallel FFT */
                if (do_parallel_fft)
                {
                    fft_.push_back(new FFT3D(Utils::find_translation_limits(parameters_.pw_cutoff(), rlv),
                                             nfft_workers, mpi_grid_->communicator(1 << _dim_row_), CPU));
                    if (!parameters_.full_potential())
                    {
                        fft_coarse_.push_back(new FFT3D(Utils::find_translation_limits(2 * parameters_.gk_cutoff(), rlv),
                                                        nfft_workers, mpi_grid_->communicator(1 << _dim_row_), CPU));
                    }
                }
                else /* serial FFT driver */
                {
                    if (tid == gpu_thread_id_)
                    {
                        fft_.push_back(new FFT3D(Utils::find_translation_limits(parameters_.pw_cutoff(), rlv),
                                                 nfft_workers, MPI_COMM_SELF, parameters_.processing_unit()));
                    }
                    else
                    {
                        fft_.push_back(new FFT3D(Utils::find_translation_limits(parameters_.pw_cutoff(), rlv),
                                                 nfft_workers, MPI_COMM_SELF, CPU));
                    }
                    if (!parameters_.full_potential())
                    {
                        if (tid == gpu_thread_id_)
                        {
                            fft_coarse_.push_back(new FFT3D(Utils::find_translation_limits(2 * parameters_.gk_cutoff(), rlv),
                                                            nfft_workers, MPI_COMM_SELF, parameters_.processing_unit()));
                        }
                        else
                        {
                            fft_coarse_.push_back(new FFT3D(Utils::find_translation_limits(2 * parameters_.gk_cutoff(), rlv),
                                                            nfft_workers, MPI_COMM_SELF, CPU));
                        }
                    }
                }
            }

            /* create a list of G-vectors for dense FFT grid */
            gvec_ = Gvec(vector3d<double>(0, 0, 0), parameters_.pw_cutoff(), rlv, fft_[0], true);

            if (!parameters_.full_potential())
            {
                /* create a list of G-vectors for corase FFT grid */
                gvec_coarse_ = Gvec(vector3d<double>(0, 0, 0), parameters_.gk_cutoff() * 2, rlv, fft_coarse_[0], false);
            }

            #ifdef __PRINT_MEMORY_USAGE
            MEMORY_USAGE_INFO();
            #endif
    
            if (unit_cell_.num_atoms() != 0) unit_cell_.symmetry()->check_gvec_symmetry(gvec_);

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
                    lmax = 2 * parameters_.lmax_beta();
                    break;
                }
            }
            
            reciprocal_lattice_ = new Reciprocal_lattice(unit_cell_, parameters_.esm_type(), gvec_, lmax, comm_);

            #ifdef __PRINT_MEMORY_USAGE
            MEMORY_USAGE_INFO();
            #endif

            if (parameters_.full_potential()) step_function_ = new Step_function(unit_cell_, reciprocal_lattice_, fft_[0], gvec_, comm_);

            if (parameters_.iterative_solver_input_section().real_space_prj_) 
            {
                real_space_prj_ = new Real_space_prj(unit_cell_, comm_, parameters_.iterative_solver_input_section().R_mask_scale_,
                                                     parameters_.iterative_solver_input_section().mask_alpha_,
                                                     parameters_.gk_cutoff(), parameters_.num_fft_threads(),
                                                     parameters_.num_fft_workers());
            }

            /* take 20% of empty non-magnetic states */
            if (parameters_.num_fv_states() < 0) 
            {
                int nfv = int(1e-8 + unit_cell_.num_valence_electrons() / 2.0) +
                              std::max(10, int(0.1 * unit_cell_.num_valence_electrons()));
                parameters_.set_num_fv_states(nfv);
            }
            
            if (parameters_.num_fv_states() < int(unit_cell_.num_valence_electrons() / 2.0))
                TERMINATE("not enough first-variational states");
            
            /* total number of bands */
            parameters_.set_num_bands(parameters_.num_fv_states() * parameters_.num_spins());

            std::map<std::string, ev_solver_t> str_to_ev_solver_t;

            str_to_ev_solver_t["lapack"]    = ev_lapack;
            str_to_ev_solver_t["scalapack"] = ev_scalapack;
            str_to_ev_solver_t["elpa1"]     = ev_elpa1;
            str_to_ev_solver_t["elpa2"]     = ev_elpa2;
            str_to_ev_solver_t["magma"]     = ev_magma;
            str_to_ev_solver_t["plasma"]    = ev_plasma;
            str_to_ev_solver_t["rs_cpu"]    = ev_rs_cpu;
            str_to_ev_solver_t["rs_gpu"]    = ev_rs_gpu;

            std::string evsn[] = {parameters_.std_evp_solver_name(), parameters_.gen_evp_solver_name()};

            if (evsn[0] == "")
            {
                if (mpi_grid_->size(1 << _dim_row_ | 1 << _dim_col_) == 1)
                {
                    evsn[0] = "lapack";
                }
                else
                {
                    evsn[0] = "scalapack";
                }
            }

            if (evsn[1] == "")
            {
                if (mpi_grid_->size(1 << _dim_row_ | 1 << _dim_col_) == 1)
                {
                    evsn[1] = "lapack";
                }
                else
                {
                    evsn[1] = "elpa1";
                }
            }

            ev_solver_t* evst[] = {&std_evp_solver_type_, &gen_evp_solver_type_};

            for (int i = 0; i < 2; i++)
            {
                auto name = evsn[i];

                if (str_to_ev_solver_t.count(name) == 0)
                {
                    std::stringstream s;
                    s << "wrong eigen value solver " << name;
                    TERMINATE(s);
                }
                *evst[i] = str_to_ev_solver_t[name];
            }

            #if (__VERBOSITY > 0)
            if (comm_.rank() == 0) print_info();
            #endif

            initialized_ = true;
        }

        Simulation_parameters const& parameters() const
        {
            return parameters_;
        }

        Unit_cell& unit_cell()
        {
            return unit_cell_;
        }

        Step_function const* step_function() const
        {
            return step_function_;
        }

        Reciprocal_lattice const* reciprocal_lattice() const
        {
            return reciprocal_lattice_;
        }

        inline FFT3D* fft(int thread_id__) const
        {
            return fft_[thread_id__];
        }

        inline FFT3D* fft_coarse(int thread_id__) const
        {
            return fft_coarse_[thread_id__];
        }

        Gvec const& gvec() const
        {
            return gvec_;
        }

        Gvec const& gvec_coarse() const
        {
            return gvec_coarse_;
        }

        Communicator const& comm() const
        {
            return comm_;
        }

        MPI_grid const& mpi_grid() const
        {
            return *mpi_grid_;
        }
        
        void create_storage_file() const
        {
            if (comm_.rank() == 0)
            {
                /* create new hdf5 file */
                HDF5_tree fout(storage_file_name, true);
                fout.create_node("parameters");
                fout.create_node("effective_potential");
                fout.create_node("effective_magnetic_field");
                fout.create_node("density");
                fout.create_node("magnetization");
                
                fout["parameters"].write("num_spins", parameters_.num_spins());
                fout["parameters"].write("num_mag_dims", parameters_.num_mag_dims());
                fout["parameters"].write("num_bands", parameters_.num_bands());
            }
            comm_.barrier();
        }

        Real_space_prj const* real_space_prj() const
        {
            return real_space_prj_;
        }

        void print_info()
        {
            printf("\n");
            printf("SIRIUS version : %2i.%02i\n", major_version, minor_version);
            printf("git hash       : %s\n", git_hash);
            printf("build date     : %s\n", build_date);
            //= printf("start time     : %s\n", start_time("%c").c_str());
            printf("\n");
            printf("number of MPI ranks           : %i\n", comm_.size());
            printf("MPI grid                      :");
            for (int i = 0; i < mpi_grid_->num_dimensions(); i++) printf(" %i", mpi_grid_->size(1 << i));
            printf("\n");
            printf("maximum number of OMP threads   : %i\n", Platform::max_num_threads()); 
            printf("number of OMP threads for FFT   : %i\n", parameters_.num_fft_threads()); 
            printf("number of pthreads for each FFT : %i\n", parameters_.num_fft_workers()); 
            printf("cyclic block size               : %i\n", parameters_.cyclic_block_size());
        
            unit_cell_.print_info();
        
            printf("\n");
            printf("plane wave cutoff                     : %f\n", parameters_.pw_cutoff());
            printf("number of G-vectors within the cutoff : %i\n", gvec_.num_gvec());
            printf("number of G-shells                    : %i\n", gvec_.num_shells());
            printf("FFT grid size   : %i %i %i   total : %i\n", fft_[0]->size(0), fft_[0]->size(1), fft_[0]->size(2), fft_[0]->size());
            printf("FFT grid limits : %i %i   %i %i   %i %i\n", fft_[0]->grid_limits(0).first, fft_[0]->grid_limits(0).second,
                                                                fft_[0]->grid_limits(1).first, fft_[0]->grid_limits(1).second,
                                                                fft_[0]->grid_limits(2).first, fft_[0]->grid_limits(2).second);
            
            if (!parameters_.full_potential())
            {
                printf("number of G-vectors on the coarse grid within the cutoff : %i\n", gvec_coarse_.num_gvec());
                printf("FFT coarse grid size   : %i %i %i   total : %i\n", fft_coarse_[0]->size(0), fft_coarse_[0]->size(1), fft_coarse_[0]->size(2), fft_coarse_[0]->size());
                printf("FFT coarse grid limits : %i %i   %i %i   %i %i\n", fft_coarse_[0]->grid_limits(0).first, fft_coarse_[0]->grid_limits(0).second,
                                                                           fft_coarse_[0]->grid_limits(1).first, fft_coarse_[0]->grid_limits(1).second,
                                                                           fft_coarse_[0]->grid_limits(2).first, fft_coarse_[0]->grid_limits(2).second);
            }
        
            for (int i = 0; i < unit_cell_.num_atom_types(); i++) unit_cell_.atom_type(i)->print_info();
        
            printf("\n");
            printf("total number of aw basis functions : %i\n", unit_cell_.mt_aw_basis_size());
            printf("total number of lo basis functions : %i\n", unit_cell_.mt_lo_basis_size());
            printf("number of first-variational states : %i\n", parameters_.num_fv_states());
            printf("number of bands                    : %i\n", parameters_.num_bands());
            printf("number of spins                    : %i\n", parameters_.num_spins());
            printf("number of magnetic dimensions      : %i\n", parameters_.num_mag_dims());
            printf("lmax_apw                           : %i\n", parameters_.lmax_apw());
            printf("lmax_pw                            : %i\n", parameters_.lmax_pw());
            printf("lmax_rho                           : %i\n", parameters_.lmax_rho());
            printf("lmax_pot                           : %i\n", parameters_.lmax_pot());
            printf("lmax_beta                          : %i\n", parameters_.lmax_beta());
        
            //== std::string evsn[] = {"standard eigen-value solver: ", "generalized eigen-value solver: "};
            //== ev_solver_t evst[] = {std_evp_solver_->type(), gen_evp_solver_->type()};
            //== for (int i = 0; i < 2; i++)
            //== {
            //==     printf("\n");
            //==     printf("%s", evsn[i].c_str());
            //==     switch (evst[i])
            //==     {
            //==         case ev_lapack:
            //==         {
            //==             printf("LAPACK\n");
            //==             break;
            //==         }
            //==         #ifdef __SCALAPACK
            //==         case ev_scalapack:
            //==         {
            //==             printf("ScaLAPACK, block size %i\n", linalg<scalapack>::cyclic_block_size());
            //==             break;
            //==         }
            //==         case ev_elpa1:
            //==         {
            //==             printf("ELPA1, block size %i\n", linalg<scalapack>::cyclic_block_size());
            //==             break;
            //==         }
            //==         case ev_elpa2:
            //==         {
            //==             printf("ELPA2, block size %i\n", linalg<scalapack>::cyclic_block_size());
            //==             break;
            //==         }
            //==         case ev_rs_gpu:
            //==         {
            //==             printf("RS_gpu\n");
            //==             break;
            //==         }
            //==         case ev_rs_cpu:
            //==         {
            //==             printf("RS_cpu\n");
            //==             break;
            //==         }
            //==         #endif
            //==         case ev_magma:
            //==         {
            //==             printf("MAGMA\n");
            //==             break;
            //==         }
            //==         case ev_plasma:
            //==         {
            //==             printf("PLASMA\n");
            //==             break;
            //==         }
            //==         default:
            //==         {
            //==             error_local(__FILE__, __LINE__, "wrong eigen-value solver");
            //==         }
            //==     }
            //== }
        
            printf("\n");
            printf("processing unit : ");
            switch (parameters_.processing_unit())
            {
                case CPU:
                {
                    printf("CPU\n");
                    break;
                }
                case GPU:
                {
                    printf("GPU\n");
                    break;
                }
            }
            
            printf("\n");
            printf("XC functionals : \n");
            for (auto& xc_label: parameters_.xc_functionals())
            {
                XC_functional xc(xc_label, parameters_.num_spins());
                printf("\n");
                printf("%s\n", xc_label.c_str());
                printf("%s\n", xc.name().c_str());
                printf("%s\n", xc.refs().c_str());
            }
        }

        inline std::string const& start_time_tag() const
        {
            return start_time_tag_;
        }

        inline void set_iterative_solver_tolerance(double tolerance__)
        {
            iterative_solver_tolerance_ = tolerance__;
        }

        inline double iterative_solver_tolerance() const
        {
            return iterative_solver_tolerance_;
        }

        inline ev_solver_t std_evp_solver_type() const
        {
            return std_evp_solver_type_;
        }
    
        inline ev_solver_t gen_evp_solver_type() const
        {
            return gen_evp_solver_type_;
        }

        inline int num_fft_threads() const
        {
            return (int)fft_.size();
        }

        inline int gpu_thread_id() const
        {
            return gpu_thread_id_;
        }


        //== void write_json_output()
        //== {
        //==     auto ts = Timer::collect_timer_stats();
        //==     if (comm_.rank() == 0)
        //==     {
        //==         char buf[100];
        //==         tm* ptm = localtime(&start_time_.tv_sec); 
        //==         strftime(buf, sizeof(buf), "%Y%m%d%H%M%S", ptm);
        //==         std::string fname = std::string("output_") + std::string(buf) + std::string(".json");
        //==         JSON_write jw(fname);
        //==         
        //==         jw.single("git_hash", git_hash);
        //==         jw.single("build_date", build_date);
        //==         jw.single("num_ranks", comm_.size());
        //==         jw.single("max_num_threads", Platform::max_num_threads());
        //==         jw.single("cyclic_block_size", parameters_.cyclic_block_size());
        //==         jw.single("mpi_grid", parameters_.mpi_grid_dims());
        //==         std::vector<int> fftgrid(3);
        //==         for (int i = 0; i < 3; i++) fftgrid[i] = fft_->size(i);
        //==         jw.single("fft_grid", fftgrid);
        //==         jw.single("chemical_formula", unit_cell_.chemical_formula());
        //==         jw.single("num_atoms", unit_cell_.num_atoms());
        //==         jw.single("num_fv_states", parameters_.num_fv_states());
        //==         jw.single("num_bands", parameters_.num_bands());
        //==         jw.single("aw_cutoff", parameters_.aw_cutoff());
        //==         jw.single("pw_cutoff", parameters_.pw_cutoff());
        //==         jw.single("omega", unit_cell_.omega());
        //==         
        //==         jw.single("timers", ts);
        //==     }
        //== }
};

};

#endif
