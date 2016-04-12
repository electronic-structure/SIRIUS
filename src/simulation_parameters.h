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

/** \file simulation_parameters.h
 *   
 *  \brief Contains definition and implementation of sirius::Simulation_parameters class.
 */

#ifndef __SIMULATION_PARAMETERS_H__
#define __SIMULATION_PARAMETERS_H__

#include "sirius_internal.h"
#include "input.h"

namespace sirius {

/// Basic parameters of the simulation. 
/** This is the parent class for sirius::Simulation_context. */
class Simulation_parameters
{
    protected:
    
        /// Maximum l for APW functions.
        int lmax_apw_;
        
        /// Maximum l for plane waves.
        int lmax_pw_;
        
        /// Maximum l for density.
        int lmax_rho_;
        
        /// Maximum l for potential
        int lmax_pot_;

        /// Cutoff for augmented-wave functions.
        double aw_cutoff_;
    
        /// Cutoff for plane-waves (for density and potential expansion).
        double pw_cutoff_;
    
        /// Cutoff for |G+k| plane-waves.
        double gk_cutoff_;
        
        /// number of first-variational states
        int num_fv_states_;
    
        /// number of spin componensts (1 or 2)
        int num_spins_;
    
        /// number of dimensions of the magnetization and effective magnetic field (0, 1 or 3)
        int num_mag_dims_;
    
        /// true if spin-orbit correction is applied
        bool so_correction_;
       
        /// true if UJ correction is applied
        bool uj_correction_;

        bool gamma_point_;

        bool reduce_gvec_;
    
        /// MPI grid dimensions
        std::vector<int> mpi_grid_dims_;

        int cyclic_block_size_;

        std::string std_evp_solver_name_;

        std::string gen_evp_solver_name_;
    
        /// Type of the processing unit.
        processing_unit_t processing_unit_;
    
        /// Smearing function width.
        double smearing_width_;

        electronic_structure_method_t esm_type_;

        Iterative_solver_input_section iterative_solver_input_section_;
        
        Mixer_input_section mixer_input_section_;

        Unit_cell_input_section unit_cell_input_section_;
        
        std::vector<std::string> xc_functionals_;

        std::string fft_mode_;

        void set_defaults()
        {
            lmax_apw_            = -1;
            lmax_pw_             = -1;
            lmax_rho_            = -1;
            lmax_pot_            = -1;
            aw_cutoff_           = 7.0;
            pw_cutoff_           = 20.0;
            gk_cutoff_           = 6.0;
            num_fv_states_       = -1;
            num_spins_           = 1;
            num_mag_dims_        = 0;
            so_correction_       = false;
            uj_correction_       = false;
            gamma_point_         = false;
            reduce_gvec_         = false;
            mpi_grid_dims_       = {1};
            cyclic_block_size_   = 32;
            processing_unit_     = CPU;
            smearing_width_      = 0.001;
            esm_type_            = full_potential_lapwlo;
            std_evp_solver_name_ = "";
            gen_evp_solver_name_ = "";
            fft_mode_            = "serial";
        }
        
        /// Import data from initial input parameters.
        void import(std::string const& fname__)
        {
            JSON_tree parser(fname__);
            /* read unit cell */
            unit_cell_input_section_.read(parser);
            /* read parameters of mixer */
            mixer_input_section_.read(parser);
            /* read parameters of iterative solver */
            iterative_solver_input_section_.read(parser);

            /* read list of XC functionals */
            /* The following part of the input file is parsed:
             * \code{.json}
             *     "xc_functionals" : ["name1", "name2", ...]
             * \endcode
             */
            if (parser.exist("xc_functionals"))
            {
                xc_functionals_.clear();
                for (int i = 0; i < parser["xc_functionals"].size(); i++)
                {
                    std::string s;
                    parser["xc_functionals"][i] >> s;
                    xc_functionals_.push_back(s);
                }
            }

            mpi_grid_dims_       = parser["mpi_grid_dims"].get(mpi_grid_dims_); 
            cyclic_block_size_   = parser["cyclic_block_size"].get(cyclic_block_size_);
            num_fv_states_       = parser["num_fv_states"].get(num_fv_states_);
            smearing_width_      = parser["smearing_width"].get(smearing_width_);
            std_evp_solver_name_ = parser["std_evp_solver_type"].get(std_evp_solver_name_);
            gen_evp_solver_name_ = parser["gen_evp_solver_type"].get(gen_evp_solver_name_);

            std::string pu = "cpu";
            pu = parser["processing_unit"].get(pu);
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

            std::string esm = "full_potential_lapwlo";
            esm = parser["electronic_structure_method"].get(esm);
            std::transform(esm.begin(), esm.end(), esm.begin(), ::tolower);
            set_esm_type(esm);

            fft_mode_ = parser["fft_mode"].get(fft_mode_);
            reduce_gvec_ = parser["reduce_gvec"].get<int>(reduce_gvec_);
        }

    public:

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

        inline void set_gamma_point(bool gamma_point__)
        {
            gamma_point_ = gamma_point__;
        }

        inline void set_mpi_grid_dims(std::vector<int> mpi_grid_dims__)
        {
            mpi_grid_dims_ = mpi_grid_dims__;
        }

        inline void add_xc_functional(std::string name__)
        {
            xc_functionals_.push_back(name__);
        }

        inline void set_esm_type(electronic_structure_method_t esm_type)
        {
        	esm_type_ = esm_type;
        }

        inline void set_esm_type(std::string name__)
        {
            if (name__ == "full_potential_lapwlo")
            {
                esm_type_ = full_potential_lapwlo;
            }
            else if (name__ == "full_potential_pwlo")
            {
                esm_type_ = full_potential_pwlo;
            }
            else if (name__ == "ultrasoft_pseudopotential")
            {
                esm_type_ = ultrasoft_pseudopotential;
            } 
            else if (name__ == "norm_conserving_pseudopotential")
            {
                esm_type_ = norm_conserving_pseudopotential;
            }
            else
            {
                TERMINATE("wrong type of electronic structure method");
            }

//            switch(name__)
//            {
//				case "full_potential_lapwlo":
//				{
//
//				}break;
//
//				case "full_potential_pwlo":
//				{
//
//				}break;
//
//				case "ultrasoft_pseudopotential":
//				{
//
//				}break;
//
//				case "norm_conserving_pseudopotential":
//				{
//
//				}break;
//
//				case "paw_pseudopotential":
//				{
//
//				}break;
//
//				default:
//				{
//					TERMINATE("wrong type of electronic structure method");
//				}break;
//            }
        }

        inline void set_processing_unit(processing_unit_t pu__)
        {
            processing_unit_ = pu__;
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

        inline bool gamma_point() const
        {
            return gamma_point_;
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

        inline int cyclic_block_size() const
        {
            return cyclic_block_size_;
        }
    
        inline electronic_structure_method_t esm_type() const
        {
            return esm_type_;
        }
    
        inline Mixer_input_section const& mixer_input_section() const
        {
            return mixer_input_section_;
        }

        inline Iterative_solver_input_section const& iterative_solver_input_section() const
        {
            return iterative_solver_input_section_;
        }

        inline bool full_potential() const
        {
            return (esm_type_ == full_potential_lapwlo || esm_type_ == full_potential_pwlo);
        }

        inline std::vector<std::string> const& xc_functionals() const
        {
            return xc_functionals_;
        }

        inline std::string const& std_evp_solver_name() const
        {
            return std_evp_solver_name_;
        }

        inline std::string const& gen_evp_solver_name() const
        {
            return gen_evp_solver_name_;
        }
};

};

#endif
