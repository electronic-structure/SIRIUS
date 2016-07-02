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
        int lmax_apw_{-1};
        
        /// Maximum l for plane waves.
        int lmax_pw_{-1};
        
        /// Maximum l for density.
        int lmax_rho_{-1};
        
        /// Maximum l for potential
        int lmax_pot_{-1};

        /// Cutoff for augmented-wave functions.
        double aw_cutoff_{7};
    
        /// Cutoff for plane-waves (for density and potential expansion).
        double pw_cutoff_{20};
    
        /// Cutoff for |G+k| plane-waves.
        double gk_cutoff_{6};
        
        /// Number of first-variational states.
        int num_fv_states_{-1};
    
        /// Number of spin componensts (1 or 2).
        int num_spins_{1};
    
        /// Number of dimensions of the magnetization and effective magnetic field (0, 1 or 3).
        int num_mag_dims_{0};

        /// Scale muffin-tin radii automatically.
        int auto_rmt_{1};
    
        /// True if spin-orbit correction is applied.
        bool so_correction_{false};
       
        /// True if UJ correction is applied.
        bool uj_correction_{false};
        
        /// True if gamma-point (real) version of the PW code is used.
        bool gamma_point_{false};

        /// Type of the processing unit.
        processing_unit_t processing_unit_{CPU};
    
        /// Smearing function width.
        double smearing_width_{0.001};
        
        /// List of XC functionals.
        std::vector<std::string> xc_functionals_;
        
        /// Type of relativity for valence states.
        relativity_t valence_relativity_{relativity_t::zora};
        
        /// Type of relativity for core states.
        relativity_t core_relativity_{relativity_t::dirac};
        
        /// Type of electronic structure method.
        electronic_structure_method_t esm_type_{full_potential_lapwlo};

        Iterative_solver_input_section iterative_solver_input_section_;
        
        Mixer_input_section mixer_input_section_;

        Unit_cell_input_section unit_cell_input_section_;

        Control_input_section control_input_section_;
        
        /// Import data from initial input parameters.
        void import(std::string const& fname__)
        {
            PROFILE();

            JSON_tree parser(fname__);
            /* read unit cell */
            unit_cell_input_section_.read(parser);
            /* read parameters of mixer */
            mixer_input_section_.read(parser);
            /* read parameters of iterative solver */
            iterative_solver_input_section_.read(parser);
            /* read controls */
            control_input_section_.read(parser);
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
            control_input_section_.mpi_grid_dims_ = mpi_grid_dims__;
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
            std::map<std::string, electronic_structure_method_t> m;

            m["full_potential_lapwlo"]           = full_potential_lapwlo;
            m["full_potential_pwlo"]             = full_potential_pwlo;
            m["ultrasoft_pseudopotential"]       = ultrasoft_pseudopotential;
            m["norm_conserving_pseudopotential"] = norm_conserving_pseudopotential;
            m["paw_pseudopotential"] 			 = paw_pseudopotential;

            if (m.count(name__) == 0)
            {
                std::stringstream s;
                s << "wrong type of electronic structure method: " << name__;
                TERMINATE(s);
            }
            esm_type_ = m[name__];
        }

        inline void set_core_relativity(std::string name__)
        {
            std::map<std::string, relativity_t> m;

            m["none"]  = relativity_t::none;
            m["dirac"] = relativity_t::dirac;

            if (m.count(name__) == 0) {
                std::stringstream s;
                s << "wrong type of core relativity: " << name__;
                TERMINATE(s);
            }
            core_relativity_ = m[name__];
        }

        inline void set_valence_relativity(std::string name__)
        {
            std::map<std::string, relativity_t> m;

            m["none"]            = relativity_t::none;
            m["zora"]            = relativity_t::zora;
            m["koelling_harmon"] = relativity_t::koelling_harmon;

            if (m.count(name__) == 0) {
                std::stringstream s;
                s << "wrong type of valence relativity: " << name__;
                TERMINATE(s);
            }
            valence_relativity_ = m[name__];
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

        inline double set_smearing_width(double smearing_width__)
        {
            return smearing_width_ = smearing_width__;
        }

        inline void set_auto_rmt(int auto_rmt__)
        {
            auto_rmt_ = auto_rmt__;
        }

        inline int auto_rmt() const
        {
            return auto_rmt_;
        }
    
        bool need_sv() const
        {
            if (num_spins() == 2 || uj_correction() || so_correction()) return true;
            return false;
        }
        
        inline std::vector<int> const& mpi_grid_dims() const
        {
            return control_input_section_.mpi_grid_dims_;
        }

        inline int cyclic_block_size() const
        {
            return control_input_section_.cyclic_block_size_;
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
            return control_input_section_.std_evp_solver_name_;
        }

        inline std::string const& gen_evp_solver_name() const
        {
            return control_input_section_.gen_evp_solver_name_;
        }

        inline relativity_t valence_relativity() const
        {
            return valence_relativity_;
        }

        inline relativity_t core_relativity() const
        {
            return core_relativity_;
        }

        inline double rmt_max() const
        {
            return control_input_section_.rmt_max_;
        }

        inline double spglib_tolerance() const
        {
            return control_input_section_.spglib_tolerance_;
        }
};

};

#endif
