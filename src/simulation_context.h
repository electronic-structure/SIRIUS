// Copyright (c) 2013-2016 Anton Kozhevnikov, Thomas Schulthess
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

/** \file simulation_context.h
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
#include "augmentation_operator.h"

namespace sirius {

/// Simulation context is a set of parameters and objects describing a single simulation. 
/** The order of initialization of the simulation context is the following: first, the default parameter 
 *  values are set in set_defaults() method, then (optionally) import() method is called and the parameters are 
 *  overwritten with the those from the input file, and finally, the user sets the values with set_...() metods.
 *  Then the atom types and atoms can be added to the simulation and the context can be initialized with the 
 *  corresponding method. */
class Simulation_context: public Simulation_parameters
{
    private:

        /// Communicator for this simulation.
        Communicator const& comm_;

        /// MPI grid for this simulation.
        MPI_grid* mpi_grid_{nullptr};
        
        /// 2D MPI grid for the FFT driver.
        MPI_grid* mpi_grid_fft_{nullptr};

        MPI_grid* mpi_grid_fft_vloc_{nullptr};

        /// 2D BLACS grid for distributed linear algebra operations.
        BLACS_grid* blacs_grid_{nullptr};

        /// 1D BLACS grid for a "slice" data distribution of full-potential wave-functions.
        /** This grid is used to distribute band index and keep a whole wave-function */
        BLACS_grid* blacs_grid_slice_{nullptr};

        /// Unit cell of the simulation.
        Unit_cell unit_cell_;

        FFT3D* fft_{nullptr};

        FFT3D* fft_coarse_{nullptr};

        /// Step function is used in full-potential methods.
        Step_function* step_function_{nullptr};

        Gvec gvec_;

        Gvec gvec_coarse_;

        Gvec_FFT_distribution* gvec_fft_distr_{nullptr};

        std::vector<Augmentation_operator*> augmentation_op_;

        Real_space_prj* real_space_prj_{nullptr};

        /// Creation time of the context.
        timeval start_time_;

        std::string start_time_tag_;

        ev_solver_t std_evp_solver_type_{ev_lapack};

        ev_solver_t gen_evp_solver_type_{ev_lapack};

        mdarray<double_complex, 3> phase_factors_;

        double time_active_;
        
        bool initialized_{false};

        void init_fft();

        Simulation_context(Simulation_context const&) = delete;

        void init()
        {
            gettimeofday(&start_time_, NULL);
            
            tm const* ptm = localtime(&start_time_.tv_sec); 
            char buf[100];
            strftime(buf, sizeof(buf), "%Y%m%d%H%M%S", ptm);
            start_time_tag_ = std::string(buf);
        }

    public:
        
        Simulation_context(std::string const& fname__,
                           Communicator const& comm__)
            : comm_(comm__),
              unit_cell_(*this, comm_)
        {
            PROFILE();
            init();
            import(fname__);
            unit_cell_.import(unit_cell_input_section_);
        }

        Simulation_context(Communicator const& comm__)
            : comm_(comm__),
              unit_cell_(*this, comm_)
        {
            PROFILE();
            init();
        }

        ~Simulation_context()
        {
            PROFILE();

            time_active_ += runtime::wtime();

            if (mpi_comm_world().rank() == 0 && initialized_)
            {
                printf("Simulation_context active time: %.4f sec.\n", time_active_);
            }

            for (auto e: augmentation_op_) delete e;
            if (step_function_ != nullptr) delete step_function_;
            if (real_space_prj_ != nullptr) delete real_space_prj_;
            if (gvec_fft_distr_ != nullptr) delete gvec_fft_distr_;
            if (fft_ != nullptr) delete fft_;
            if (fft_coarse_ != nullptr) delete fft_coarse_;
            if (blacs_grid_slice_ != nullptr) delete blacs_grid_slice_;
            if (blacs_grid_ != nullptr) delete blacs_grid_;
            if (mpi_grid_ != nullptr) delete mpi_grid_;
            if (mpi_grid_fft_ != nullptr) delete mpi_grid_fft_;
            if (mpi_grid_fft_vloc_ != nullptr) delete mpi_grid_fft_vloc_;
        }

        /// Initialize the similation (can only be called once).
        void initialize();

        void print_info();

        Unit_cell& unit_cell()
        {
            return unit_cell_;
        }

        Step_function const& step_function() const
        {
            return *step_function_;
        }

        inline FFT3D& fft() const
        {
            return *fft_;
        }

        inline FFT3D& fft_coarse() const
        {
            return *fft_coarse_;
        }

        Gvec const& gvec() const
        {
            return gvec_;
        }

        Gvec_FFT_distribution const& gvec_fft_distr() const
        {
            return *gvec_fft_distr_;
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

        MPI_grid const& mpi_grid_fft_vloc() const
        {
            return *mpi_grid_fft_vloc_;
        }

        BLACS_grid const& blacs_grid() const
        {
            return *blacs_grid_;
        }

        BLACS_grid const& blacs_grid_slice() const
        {
            return *blacs_grid_slice_;
        }

        inline int num_fv_states() const
        {
            return num_fv_states_;
        }

        inline int num_bands() const
        {
            return num_spins() * num_fv_states_;
        }
        
        void create_storage_file() const
        {
            if (comm_.rank() == 0) {
                /* create new hdf5 file */
                HDF5_tree fout(storage_file_name, true);
                fout.create_node("parameters");
                fout.create_node("effective_potential");
                fout.create_node("effective_magnetic_field");
                fout.create_node("density");
                fout.create_node("magnetization");
                
                fout["parameters"].write("num_spins", num_spins());
                fout["parameters"].write("num_mag_dims", num_mag_dims());
                fout["parameters"].write("num_bands", num_bands());

                mdarray<int, 2> gv(3, gvec_.num_gvec());
                for (int ig = 0; ig < gvec_.num_gvec(); ig++) {
                    auto G = gvec_[ig];
                    for (int x: {0, 1, 2}) gv(x, ig) = G[x];
                }
                fout["parameters"].write("num_gvec", gvec_.num_gvec());
                fout["parameters"].write("gvec", gv);
            }
            comm_.barrier();
        }

        Real_space_prj const* real_space_prj() const
        {
            return real_space_prj_;
        }

        inline std::string const& start_time_tag() const
        {
            return start_time_tag_;
        }

        inline void set_iterative_solver_tolerance(double tolerance__)
        {
            iterative_solver_input_section_.energy_tolerance_ = tolerance__;
        }

        inline double iterative_solver_tolerance() const
        {
            return iterative_solver_input_section_.energy_tolerance_;
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
            //return std::exp(double_complex(0.0, twopi * (G * unit_cell_.atom(ia__).position())));
            return phase_factors_(0, G[0], ia__) *
                   phase_factors_(1, G[1], ia__) *
                   phase_factors_(2, G[2], ia__);
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
