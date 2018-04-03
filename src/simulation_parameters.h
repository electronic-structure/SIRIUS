// Copyright (c) 2013-2017 Anton Kozhevnikov, Thomas Schulthess
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
 *  \brief Contains definition and implementation of sirius::Simulation_parameters_base class.
 */

#ifndef __SIMULATION_PARAMETERS_H__
#define __SIMULATION_PARAMETERS_H__

#include "typedefs.h"
#include "utils.h"
#include "input.h"

namespace sirius {

/// Set of basic parameters of a simulation.
class Simulation_parameters
{
  protected:
    /// Type of the processing unit.
    device_t processing_unit_{CPU};

    /// Type of relativity for valence states.
    relativity_t valence_relativity_{relativity_t::zora};

    /// Type of relativity for core states.
    relativity_t core_relativity_{relativity_t::dirac};

    /// Type of electronic structure method.
    electronic_structure_method_t electronic_structure_method_{electronic_structure_method_t::full_potential_lapwlo};

    Iterative_solver_input iterative_solver_input_;

    Mixer_input mixer_input_;

    Unit_cell_input unit_cell_input_;

    Control_input control_input_;

    Parameters_input parameters_input_;

    Settings_input settings_input_;

    Hubbard_input hubbard_input_;

  public:

    /// Import parameters from a file or a serialized json string.
    void import(std::string const& str__)
    {
        PROFILE("sirius::Simulation_parameters::import");

        if (str__.size() == 0) {
            return;
        }

        json dict;
        if (str__.find("{") == std::string::npos) { /* this is a file */
            if (Utils::file_exists(str__)) {
                try {
                    std::ifstream(str__) >> dict;
                } catch(std::exception& e) {
                    std::stringstream s;
                    s << "wrong input json file" << std::endl
                      << e.what();
                    TERMINATE(s);
                }
            }
        } else { /* this is a json string */
            try {
                std::istringstream(str__) >> dict;
            } catch (std::exception& e) {
                std::stringstream s;
                s << "wrong input json string" << std::endl
                  << e.what();
                TERMINATE(s);
            }
        }

        /* read unit cell */
        unit_cell_input_.read(dict);
        /* read parameters of mixer */
        mixer_input_.read(dict);
        /* read parameters of iterative solver */
        iterative_solver_input_.read(dict);
        /* read controls */
        control_input_.read(dict);
        /* read parameters */
        parameters_input_.read(dict);
        /* read settings */
        settings_input_.read(dict);
        /* read hubbard parameters */
        hubbard_input_.read(dict);
    }

    inline void set_lmax_apw(int lmax_apw__)
    {
        parameters_input_.lmax_apw_ = lmax_apw__;
    }

    inline void set_lmax_rho(int lmax_rho__)
    {
        parameters_input_.lmax_rho_ = lmax_rho__;
    }

    inline void set_lmax_pot(int lmax_pot__)
    {
        parameters_input_.lmax_pot_ = lmax_pot__;
    }

    void set_num_mag_dims(int num_mag_dims__)
    {
        assert(num_mag_dims__ == 0 || num_mag_dims__ == 1 || num_mag_dims__ == 3);

        parameters_input_.num_mag_dims_ = num_mag_dims__;
    }

    inline void set_aw_cutoff(double aw_cutoff__)
    {
        parameters_input_.aw_cutoff_ = aw_cutoff__;
    }

    /// Set plane-wave cutoff.
    inline void set_pw_cutoff(double pw_cutoff__)
    {
        parameters_input_.pw_cutoff_ = pw_cutoff__;
    }

    inline void set_gk_cutoff(double gk_cutoff__)
    {
        parameters_input_.gk_cutoff_ = gk_cutoff__;
    }

    inline void set_so_correction(bool so_correction__)
    {
        parameters_input_.so_correction_ = so_correction__;
    }

    inline void set_hubbard_correction(bool hubbard_correction__)
    {
        parameters_input_.hubbard_correction_ = hubbard_correction__;
        hubbard_input_.simplified_hubbard_correction_ = false;
    }

    inline void set_hubbard_simplified_version()
    {
        hubbard_input_.simplified_hubbard_correction_ = true;
    }

    inline void set_orthogonalize_hubbard_orbitals(const bool test)
    {
        hubbard_input_.orthogonalize_hubbard_orbitals_ = true;
    }

    inline void set_normalize_hubbard_orbitals(const bool test)
    {
        hubbard_input_.normalize_hubbard_orbitals_ = true;
    }

    inline void set_gamma_point(bool gamma_point__)
    {
        parameters_input_.gamma_point_ = gamma_point__;
    }

    inline void set_mpi_grid_dims(std::vector<int> mpi_grid_dims__)
    {
        control_input_.mpi_grid_dims_ = mpi_grid_dims__;
    }

    inline void add_xc_functional(std::string name__)
    {
        parameters_input_.xc_functionals_.push_back(name__);
    }

    inline void electronic_structure_method(std::string name__)
    {
        parameters_input_.electronic_structure_method_ = name__;

        std::map<std::string, electronic_structure_method_t> m = {
            {"full_potential_lapwlo", electronic_structure_method_t::full_potential_lapwlo},
            {"pseudopotential", electronic_structure_method_t::pseudopotential}
        };

        if (m.count(name__) == 0) {
            std::stringstream s;
            s << "wrong type of electronic structure method: " << name__;
            TERMINATE(s);
        }
        electronic_structure_method_ = m[name__];
    }

    inline electronic_structure_method_t electronic_structure_method() const
    {
        return electronic_structure_method_;
    }

    inline void set_core_relativity(std::string name__)
    {
        parameters_input_.core_relativity_ = name__;

        std::map<std::string, relativity_t> m = {
            {"none", relativity_t::none},
            {"dirac", relativity_t::dirac}
        };

        if (m.count(name__) == 0) {
            std::stringstream s;
            s << "wrong type of core relativity: " << name__;
            TERMINATE(s);
        }
        core_relativity_ = m[name__];
    }

    inline void set_valence_relativity(std::string name__)
    {
        parameters_input_.valence_relativity_ = name__;

        std::map<std::string, relativity_t> m = {
            {"none", relativity_t::none},
            {"zora", relativity_t::zora},
            {"iora", relativity_t::iora},
            {"koelling_harmon", relativity_t::koelling_harmon}
        };

        if (m.count(name__) == 0) {
            std::stringstream s;
            s << "wrong type of valence relativity: " << name__;
            TERMINATE(s);
        }
        valence_relativity_ = m[name__];
    }

    inline void set_processing_unit(std::string name__)
    {
        std::transform(name__.begin(), name__.end(), name__.begin(), ::tolower);
        control_input_.processing_unit_ = name__;
        if (name__ == "cpu") {
            processing_unit_ = CPU;
        } else if (name__ == "gpu") {
            processing_unit_ = GPU;
        } else {
            TERMINATE("wrong processing unit");
        }
    }

    inline void set_processing_unit(device_t pu__)
    {
        processing_unit_ = pu__;
    }

    inline void set_molecule(bool molecule__)
    {
        parameters_input_.molecule_ = molecule__;
    }

    inline void set_verbosity(int level__)
    {
        control_input_.verbosity_ = level__;
    }

    inline int lmax_apw() const
    {
        return parameters_input_.lmax_apw_;
    }

    inline int lmmax_apw() const
    {
        return Utils::lmmax(parameters_input_.lmax_apw_);
    }

    inline int lmax_rho() const
    {
        return parameters_input_.lmax_rho_;
    }

    inline int lmmax_rho() const
    {
        return Utils::lmmax(parameters_input_.lmax_rho_);
    }

    inline int lmax_pot() const
    {
        return parameters_input_.lmax_pot_;
    }

    inline int lmmax_pot() const
    {
        return Utils::lmmax(parameters_input_.lmax_pot_);
    }

    inline double aw_cutoff() const
    {
        return parameters_input_.aw_cutoff_;
    }

    /// Plane-wave cutoff for G-vectors (in 1/[a.u.]).
    inline double pw_cutoff() const
    {
        return parameters_input_.pw_cutoff_;
    }

    /// Cutoff for G+k vectors (in 1/[a.u.]).
    inline double gk_cutoff() const
    {
        return parameters_input_.gk_cutoff_;
    }

    /// Number of dimensions in the magnetization vector.
    inline int num_mag_dims() const
    {
        assert(parameters_input_.num_mag_dims_ == 0 || parameters_input_.num_mag_dims_ == 1 ||
               parameters_input_.num_mag_dims_ == 3);

        return parameters_input_.num_mag_dims_;
    }

    /// Number of spin components.
    /** This parameter can take only two values: 1 -- non-magnetic calcaulation and wave-functions,
     *  2 -- spin-polarized calculation and wave-functions. */
    inline int num_spins() const
    {
        return (num_mag_dims() == 0) ? 1 : 2;
    }

    /// Number of components in the complex density matrix.
    /** In case of non-collinear magnetism only one out of two non-diagonal components is stored. */
    inline int num_mag_comp() const // TODO: rename; current name does not reflect the meaning
    {
        return (num_mag_dims() == 3) ? 3 : num_spins();
    }
    
    /// Number of spin dimensions of some arrays in case of magnetic calculation.
    /** Returns 1 for non magnetic calculation, 2 for spin-collinear case and 1 for non colllinear case. */
    inline int num_spin_dims()
    {
        return (num_mag_dims() == 3) ? 1 : num_spins();
    }
    
    /// Set the number of first-variational states.
    inline int num_fv_states(int num_fv_states__)
    {
        parameters_input_.num_fv_states_ = num_fv_states__;
        return parameters_input_.num_fv_states_;
    }

    /// Number of first-variational states.
    inline int num_fv_states() const
    {
        return parameters_input_.num_fv_states_;
    }

    /// Set the number of bands.
    inline int num_bands(int num_bands__)
    {
        parameters_input_.num_bands_ = num_bands__;
        return parameters_input_.num_bands_;
    }

    /// Total number of bands.
    inline int num_bands() const
    {
        if (num_fv_states() != -1) {
            if (num_mag_dims() != 3) {
                return num_fv_states();
            } else {
                return num_spins() * num_fv_states();
            }
        } else {
            return parameters_input_.num_bands_;
        }
    }

    inline int max_occupancy() const
    {
        return (num_mag_dims() == 0) ? 2 : 1;
    }

    inline bool so_correction() const
    {
        return parameters_input_.so_correction_;
    }

    inline bool hubbard_correction() const
    {
        return parameters_input_.hubbard_correction_;
    }

    inline bool gamma_point() const
    {
        return parameters_input_.gamma_point_;
    }

    inline device_t processing_unit() const
    {
        return processing_unit_;
    }

    inline double smearing_width() const
    {
        return parameters_input_.smearing_width_;
    }

    inline void set_smearing_width(double smearing_width__)
    {
        parameters_input_.smearing_width_ = smearing_width__;
    }

    inline void set_auto_rmt(int auto_rmt__)
    {
        parameters_input_.auto_rmt_ = auto_rmt__;
    }

    inline int auto_rmt() const
    {
        return parameters_input_.auto_rmt_;
    }

    bool need_sv() const
    {
        return (num_spins() == 2 || hubbard_correction() || so_correction());
    }

    inline std::vector<int> const& mpi_grid_dims() const
    {
        return control_input_.mpi_grid_dims_;
    }

    inline int cyclic_block_size() const
    {
        return control_input_.cyclic_block_size_;
    }

    inline bool full_potential() const
    {
        return (electronic_structure_method_ == electronic_structure_method_t::full_potential_lapwlo);
    }

    inline std::vector<std::string> const& xc_functionals() const
    {
        return parameters_input_.xc_functionals_;
    }

    inline std::string const& std_evp_solver_name() const
    {
        return control_input_.std_evp_solver_name_;
    }

    inline void set_std_evp_solver_name(std::string name__)
    {
        control_input_.std_evp_solver_name_ = name__;
    }

    inline std::string const& gen_evp_solver_name() const
    {
        return control_input_.gen_evp_solver_name_;
    }

    inline void set_gen_evp_solver_name(std::string name__)
    {
        control_input_.gen_evp_solver_name_ = name__;
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
        return control_input_.rmt_max_;
    }

    inline double spglib_tolerance() const
    {
        return control_input_.spglib_tolerance_;
    }

    inline bool molecule() const
    {
        return parameters_input_.molecule_;
    }

    inline memory_t main_memory_t() const
    {
        if (processing_unit_ == GPU) {
            return memory_t::device;
        }
        return memory_t::host;
    }

    inline memory_t dual_memory_t() const
    {
        if (processing_unit_ == GPU) {
            return (memory_t::host | memory_t::device);
        }
        return memory_t::host;
    }

    inline bool use_symmetry() const
    {
        return parameters_input_.use_symmetry_;
    }

    inline void set_use_symmetry(bool use_symmetry__)
    {
        parameters_input_.use_symmetry_ = use_symmetry__;
    }

    inline double iterative_solver_tolerance() const
    {
        return iterative_solver_input_.energy_tolerance_;
    }

    inline void set_iterative_solver_tolerance(double tolerance__)
    {
        iterative_solver_input_.energy_tolerance_ = tolerance__;
    }

    inline void set_iterative_solver_type(std::string type__)
    {
        iterative_solver_input_.type_ = type__;
    }

    inline Control_input const& control() const
    {
        return control_input_;
    }

    inline Mixer_input const& mixer_input() const
    {
        return mixer_input_;
    }

    inline Iterative_solver_input const& iterative_solver_input() const
    {
        return iterative_solver_input_;
    }

    inline Parameters_input const& parameters_input() const
    {
        return parameters_input_;
    }

    inline Parameters_input& parameters_input()
    {
        return parameters_input_;
    }

    inline Settings_input const& settings() const
    {
        return settings_input_;
    }

    inline Hubbard_input const& Hubbard() const
    {
        return hubbard_input_;
    }
};

}; // namespace sirius

#endif
