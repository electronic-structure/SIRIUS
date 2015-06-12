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

#ifndef __SIMULATION_PARAMETERS_H__
#define __SIMULATION_PARAMETERS_H__

#include <algorithm>
#include "input.h"

/// SIRIUS namespace.
namespace sirius {

/// Parameters of the simulation. 
/** Parameters are first initialized from the initial input parameters and then by set..() methods.
 *  Any parameter used in the simulation must be first initialized here. Then the instance of the 
 *  Simulation_context class can be created where proper values of some parameters are set.
 */
class Simulation_parameters
{
    private:
    
        /// Maximum l for APW functions.
        int lmax_apw_;
        
        /// Maximum l for plane waves.
        int lmax_pw_;
        
        /// Maximum l for density.
        int lmax_rho_;
        
        /// Maximum l for potential
        int lmax_pot_;

        /// Maximum l for beta-projectors of the pseudopotential method.
        int lmax_beta_;
    
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

        int cyclic_block_size_;

        electronic_structure_method_t esm_type_;

        Iterative_solver_input_section iterative_solver_input_section_;
        
        XC_functionals_input_section xc_functionals_input_section_;
        
        Mixer_input_section mixer_input_section_;

        Unit_cell_input_section unit_cell_input_section_;

        std::map<std::string, ev_solver_t> str_to_ev_solver_t_;
        
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
                auto name = evsn[i];

                if (str_to_ev_solver_t_.count(name) == 0) TERMINATE("wrong eigen value solver");
                *evst[i] = str_to_ev_solver_t_[name];
            }

            std::string pu = iip__.common_input_section_.processing_unit_;
            std::transform(pu.begin(), pu.end(), pu.begin(), ::tolower);

            if (pu == "cpu")
            {
                processing_unit_ = CPU;
            }
            else if (pu == "gpu")
            {
                processing_unit_ = GPU;
            }
            else
            {
                TERMINATE("wrong processing unit");
            }

            std::string esm = iip__.common_input_section_.electronic_structure_method_;
            std::transform(esm.begin(), esm.end(), esm.begin(), ::tolower);
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

            iterative_solver_input_section_ = iip__.iterative_solver_input_section();
            xc_functionals_input_section_   = iip__.xc_functionals_input_section();
            mixer_input_section_            = iip__.mixer_input_section();
            unit_cell_input_section_        = iip__.unit_cell_input_section();

            cyclic_block_size_              = iip__.common_input_section_.cyclic_block_size_;

            num_fft_threads_                = iip__.common_input_section_.num_fft_threads_;
            num_fft_workers_                = iip__.common_input_section_.num_fft_workers_;
        }
    
    public:

        /// Create and initialize simulation parameters.
        /** The order of initialization is the following:
         *    - first, the default parameter values are set in the constructor
         *    - second, import() method is called and the parameters are overwritten with the input parameters
         *    - third, the user sets the values with set...() metods
         *    - fourh, the Simulation_context creates the copy of parameters and chekcs/sets the correct values
         */
        Simulation_parameters(Input_parameters const& iip__)
            : lmax_apw_(8), 
              lmax_pw_(-1), 
              lmax_rho_(8), 
              lmax_pot_(8),
              lmax_beta_(-1),
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
              cyclic_block_size_(32),
              esm_type_(full_potential_lapwlo)
        {
            LOG_FUNC_BEGIN();

            /* get the starting time */
            //gettimeofday(&start_time_, NULL);

            str_to_ev_solver_t_["lapack"]    = ev_lapack;
            str_to_ev_solver_t_["scalapack"] = ev_scalapack;
            str_to_ev_solver_t_["elpa1"]     = ev_elpa1;
            str_to_ev_solver_t_["elpa2"]     = ev_elpa2;
            str_to_ev_solver_t_["magma"]     = ev_magma;
            str_to_ev_solver_t_["plasma"]    = ev_plasma;
            str_to_ev_solver_t_["rs_cpu"]    = ev_rs_cpu;
            str_to_ev_solver_t_["rs_gpu"]    = ev_rs_gpu;

            import(iip__);

            LOG_FUNC_END();
        }

        Simulation_parameters()
        {
        }
            
        ~Simulation_parameters()
        {
        }
    
        inline void set_lmax_apw(int lmax_apw__)
        {
            lmax_apw_ = lmax_apw__;
        }
    
        inline void set_lmax_rho(int lmax_rho__)
        {
            lmax_rho_ = lmax_rho__;
        }
    
        inline void set_lmax_pot(int lmax_pot__)
        {
            lmax_pot_ = lmax_pot__;
        }

        inline void set_lmax_pw(int lmax_pw__)
        {
            lmax_pw_ = lmax_pw__;
        }

        inline void set_lmax_beta(int lmax_beta__)
        {
            lmax_beta_ = lmax_beta__;
        }
    
        //void set_num_spins(int num_spins__)
        //{
        //    num_spins_ = num_spins__;
        //}
    
        void set_num_mag_dims(int num_mag_dims__)
        {
            assert(num_mag_dims__ == 0 || num_mag_dims__ == 1 || num_mag_dims__ == 3);

            num_mag_dims_ = num_mag_dims__;
            num_spins_ = (num_mag_dims__ == 0) ? 1 : 2;
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

        inline void set_num_bands(int num_bands__)
        {
            num_bands_ = num_bands__;
        }

        inline void set_mpi_grid_dims(std::vector<int> const& mpi_grid_dims__)
        {
            mpi_grid_dims_ = mpi_grid_dims__;
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
            return lmax_beta_;
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
    
        inline electronic_structure_method_t esm_type() const
        {
            return esm_type_;
        }
    
        inline wave_function_distribution_t wave_function_distribution() const
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
    
        inline ev_solver_t std_evp_solver_type() const
        {
            return std_evp_solver_type_;
        }
    
        inline ev_solver_t gen_evp_solver_type() const
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

        Unit_cell_input_section const& unit_cell_input_section() const
        {
            return unit_cell_input_section_;
        }

        inline bool full_potential() const
        {
            return (esm_type_ == full_potential_lapwlo || esm_type_ == full_potential_pwlo);
        }
};

};

/** \page stdvarname Standard variable names
    
    Below is the list of standard names for some of the loop variables:
    
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
           
            // integrate spherical part of the function
            Spline<T> s(nmtp, parameters_.atom(ia)->type()->radial_grid());
            for (int ir = 0; ir < nmtp; ir++) s[ir] = f_mt<local>(0, ir, ialoc);
            mt_val[ia] = s.interpolate().integrate(2) * fourpi * y00;
        }

        // simple array synchronization
        Platform::allreduce(&mt_val[0], parameters_.num_atoms());
    \endcode
*/

#endif
