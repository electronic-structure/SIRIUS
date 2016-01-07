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
#include "fft3d_context.h"
#include "augmentation_operator.h"

namespace sirius {

class Simulation_context
{
    private:

        /// Copy of simulation parameters.
        Simulation_parameters parameters_; 
    
        /// Communicator for this simulation.
        Communicator const& comm_;

        /// MPI grid for this simulation.
        MPI_grid* mpi_grid_;

        MPI_grid* mpi_grid_fft_;

        FFT3D_context* fft_ctx_;

        FFT3D_context* fft_coarse_ctx_;
        
        /// Unit cell of the simulation.
        Unit_cell unit_cell_;

        /// Step function is used in full-potential methods.
        Step_function* step_function_;

        Gvec gvec_;

        Gvec gvec_coarse_;

        std::vector<Augmentation_operator*> augmentation_op_;

        Real_space_prj* real_space_prj_;

        /// Creation time of the context.
        timeval start_time_;

        std::string start_time_tag_;

        double iterative_solver_tolerance_;

        ev_solver_t std_evp_solver_type_;

        ev_solver_t gen_evp_solver_type_;

        double time_active_;
        
        bool initialized_;

        void init_fft();

    public:
        
        Simulation_context(Simulation_parameters const& parameters__,
                           Communicator const& comm__)
            : parameters_(parameters__),
              comm_(comm__),
              mpi_grid_(nullptr),
              mpi_grid_fft_(nullptr),
              fft_ctx_(nullptr),
              fft_coarse_ctx_(nullptr),
              unit_cell_(parameters_, comm_),
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

            time_active_ += Utils::current_time();

            if (Platform::rank() == 0)
            {
                printf("Simulation_context active time: %.4f sec.\n", time_active_);
            }

            for (auto e: augmentation_op_) delete e;
            if (step_function_ != nullptr) delete step_function_;
            if (real_space_prj_ != nullptr) delete real_space_prj_;
            if (fft_ctx_ != nullptr) delete fft_ctx_;
            if (fft_coarse_ctx_ != nullptr) delete fft_coarse_ctx_;
            if (mpi_grid_ != nullptr) delete mpi_grid_;
            if (mpi_grid_fft_ != nullptr) delete mpi_grid_fft_;
        }

        /// Initialize the similation (can only be called once).
        void initialize();

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

        inline FFT3D* fft(int thread_id__) const
        {
            return fft_ctx_->fft(thread_id__);
        }

        inline FFT3D* fft_coarse(int thread_id__) const
        {
            return fft_coarse_ctx_->fft(thread_id__);
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

        MPI_grid const& mpi_grid_fft() const
        {
            return *mpi_grid_fft_;
        }

        FFT3D_context& fft_ctx()
        {
            return *fft_ctx_;
        }

        FFT3D_context& fft_coarse_ctx()
        {
            return *fft_coarse_ctx_;
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
            printf("number of independent FFTs : %i\n", mpi_grid_fft_->dimension_size(0));
            printf("FFT comm size              : %i\n", mpi_grid_fft_->dimension_size(1));
            auto fft_grid = fft_ctx_->fft_grid();
            printf("FFT grid size   : %i %i %i   total : %i\n", fft_grid.size(0), fft_grid.size(1), fft_grid.size(2), fft_grid.size());
            printf("FFT grid limits : %i %i   %i %i   %i %i\n", fft_grid.limits(0).first, fft_grid.limits(0).second,
                                                                fft_grid.limits(1).first, fft_grid.limits(1).second,
                                                                fft_grid.limits(2).first, fft_grid.limits(2).second);
            
            if (!parameters_.full_potential())
            {
                fft_grid = fft_coarse_ctx_->fft_grid();
                printf("number of G-vectors on the coarse grid within the cutoff : %i\n", gvec_coarse_.num_gvec());
                printf("FFT coarse grid size   : %i %i %i   total : %i\n", fft_grid.size(0), fft_grid.size(1), fft_grid.size(2), fft_grid.size());
                printf("FFT coarse grid limits : %i %i   %i %i   %i %i\n", fft_grid.limits(0).first, fft_grid.limits(0).second,
                                                                           fft_grid.limits(1).first, fft_grid.limits(1).second,
                                                                           fft_grid.limits(2).first, fft_grid.limits(2).second);
            }
        
            for (int i = 0; i < unit_cell_.num_atom_types(); i++) unit_cell_.atom_type(i).print_info();
        
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
            printf("smearing width:                    : %f\n", parameters_.smearing_width());
        
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

        Augmentation_operator const& augmentation_op(int iat__) const
        {
            return (*augmentation_op_[iat__]);
        }

        /// Phase factors \f$ e^{i {\bf G} {\bf r}_{\alpha}} \f$
        inline double_complex gvec_phase_factor(int ig__, int ia__) const
        {
            auto G = gvec_[ig__];
            return std::exp(twopi * double_complex(0.0, G * unit_cell_.atom(ia__).position()));
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
