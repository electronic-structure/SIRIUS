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

/** \file simulation_parameters.hpp
 *
 *  \brief Contains definition and implementation of sirius::Simulation_parameters_base class.
 */

#ifndef __SIMULATION_PARAMETERS_HPP__
#define __SIMULATION_PARAMETERS_HPP__

#include "typedefs.hpp"
#include "input.hpp"

namespace sirius {

/// Json dictionary containing the options given by the interface.
#include "runtime_options_json.hpp"

/// Get all possible options for initializing sirius. It is a json dictionary.
inline const json& get_options_dictionary()
{
    if (all_options_dictionary_.size() == 0) {
        TERMINATE("Dictionary not initialized\n");
    }
    return all_options_dictionary_;
}

/// Set of basic parameters of a simulation.
class Simulation_parameters
{
  protected:
    /// Type of the processing unit.
    device_t processing_unit_{device_t::CPU};

    /// Type of relativity for valence states.
    relativity_t valence_relativity_{relativity_t::zora};

    /// Type of relativity for core states.
    relativity_t core_relativity_{relativity_t::dirac};

    /// Type of electronic structure method.
    electronic_structure_method_t electronic_structure_method_{electronic_structure_method_t::full_potential_lapwlo};

    /// Parameters of the iterative solver.
    Iterative_solver_input iterative_solver_input_;

    /// Parameters of the mixer.
    Mixer_input mixer_input_;

    /// Description of the unit cell.
    Unit_cell_input unit_cell_input_;

    /// Parameters controlling the execution.
    Control_input control_input_;

    /// Basic input parameters of PP-PW and FP-LAPW methods.
    Parameters_input parameters_input_;

    /// Internal parameters that control the numerical implementation.
    Settings_input settings_input_;

    /// LDA+U input parameters.
    Hubbard_input hubbard_input_;

    /// json dictionary containing all runtime options set up through the interface
    json runtime_options_dictionary_;

  public:
    /// Import parameters from a file or a serialized json string.
    void import(std::string const& str__)
    {
        PROFILE("sirius::Simulation_parameters::import");

        if (str__.size() == 0) {
            return;
        }

        json dict = utils::read_json_from_file_or_string(str__);

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

    /// Import parameters from a json dictionary.
    void import(json const& dict)
    {
        PROFILE("sirius::Simulation_parameters::import");

        if (dict.size() == 0) {
            return;
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

    /// Import from command line arguments.
    void import(cmd_args const& args__)
    {
        control_input_.processing_unit_     = args__.value("control.processing_unit",
                                                           control_input_.processing_unit_);
        control_input_.mpi_grid_dims_       = args__.value("control.mpi_grid_dims",
                                                           control_input_.mpi_grid_dims_);
        control_input_.std_evp_solver_name_ = args__.value("control.std_evp_solver_name",
                                                           control_input_.std_evp_solver_name_);
        control_input_.gen_evp_solver_name_ = args__.value("control.gen_evp_solver_name",
                                                           control_input_.gen_evp_solver_name_);
        control_input_.fft_mode_            = args__.value("control.fft_mode",
                                                           control_input_.fft_mode_);
        control_input_.memory_usage_        = args__.value("control.memory_usage",
                                                           control_input_.memory_usage_);

        parameters_input_.ngridk_           = args__.value("parameters.ngridk",
                                                           parameters_input_.ngridk_);
        parameters_input_.gamma_point_      = args__.value("parameters.gamma_point",
                                                           parameters_input_.gamma_point_);
        parameters_input_.pw_cutoff_        = args__.value("parameters.pw_cutoff",
                                                           parameters_input_.pw_cutoff_);

        iterative_solver_input_.orthogonalize_ = args__.value("iterative_solver.orthogonalize",
                                                              iterative_solver_input_.orthogonalize_);
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

    inline void set_hubbard_correction(bool hubbard_correction__)
    {
        parameters_input_.hubbard_correction_         = hubbard_correction__;
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
            {"pseudopotential", electronic_structure_method_t::pseudopotential}};

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

        std::map<std::string, relativity_t> m = {{"none", relativity_t::none}, {"dirac", relativity_t::dirac}};

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

        std::map<std::string, relativity_t> m = {{"none", relativity_t::none},
                                                 {"zora", relativity_t::zora},
                                                 {"iora", relativity_t::iora},
                                                 {"koelling_harmon", relativity_t::koelling_harmon}};

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

        /* set the default value */
        if (name__ == "") {
            if (acc::num_devices() > 0) {
                name__ = "gpu";
            } else {
                name__ = "cpu";
            }
        }
        control_input_.processing_unit_ = name__;
        if (name__ == "cpu") {
            this->set_processing_unit(device_t::CPU);
        } else if (name__ == "gpu") {
            this->set_processing_unit(device_t::GPU);
        } else {
            std::stringstream s;
            s << "wrong processing unit name: " << name__;
            TERMINATE(s);
        }
    }

    inline void set_processing_unit(device_t pu__)
    {
        if (acc::num_devices() == 0) {
            processing_unit_                = device_t::CPU;
            control_input_.processing_unit_ = "cpu";
        } else {
            processing_unit_ = pu__;
            if (pu__ == device_t::CPU) {
                control_input_.processing_unit_ = "cpu";
            } else if (pu__ == device_t::GPU) {
                control_input_.processing_unit_ = "gpu";
            } else {
                std::stringstream s;
                s << "wrong processing unit type";
                TERMINATE(s);
            }
        }
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
        return utils::lmmax(parameters_input_.lmax_apw_);
    }

    inline int lmax_rho() const
    {
        return parameters_input_.lmax_rho_;
    }

    inline int lmmax_rho() const
    {
        return utils::lmmax(parameters_input_.lmax_rho_);
    }

    inline int lmax_pot() const
    {
        return parameters_input_.lmax_pot_;
    }

    inline int lmmax_pot() const
    {
        return utils::lmmax(parameters_input_.lmax_pot_);
    }

    inline double aw_cutoff() const
    {
        return parameters_input_.aw_cutoff_;
    }

    inline double aw_cutoff(double aw_cutoff__)
    {
        parameters_input_.aw_cutoff_ = aw_cutoff__;
        return parameters_input_.aw_cutoff_;
    }

    /// Plane-wave cutoff for G-vectors (in 1/[a.u.]).
    inline double pw_cutoff() const
    {
        return parameters_input_.pw_cutoff_;
    }

    /// Set plane-wave cutoff.
    inline double pw_cutoff(double pw_cutoff__)
    {
        parameters_input_.pw_cutoff_ = pw_cutoff__;
        return parameters_input_.pw_cutoff_;
    }

    /// Cutoff for G+k vectors (in 1/[a.u.]).
    inline double gk_cutoff() const
    {
        return parameters_input_.gk_cutoff_;
    }

    /// Set the cutoff for G+k vectors.
    inline double gk_cutoff(double gk_cutoff__)
    {
        parameters_input_.gk_cutoff_ = gk_cutoff__;
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

    /// Number of independent spin dimensions of some arrays in case of magnetic calculation.
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

    inline bool so_correction(bool so_correction__)
    {
        parameters_input_.so_correction_ = so_correction__;
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

    /// Get the name of the standard eigen-value solver to use.
    inline std::string const& std_evp_solver_name() const
    {
        return control_input_.std_evp_solver_name_;
    }

    /// Set the name of the standard eigen-value solver to use.
    inline std::string& std_evp_solver_name(std::string name__)
    {
        control_input_.std_evp_solver_name_ = name__;
        return control_input_.std_evp_solver_name_;
    }

    /// Get the name of the generalized eigen-value solver to use.
    inline std::string const& gen_evp_solver_name() const
    {
        return control_input_.gen_evp_solver_name_;
    }

    /// Set the name of the generalized eigen-value solver to use.
    inline std::string& gen_evp_solver_name(std::string name__)
    {
        control_input_.gen_evp_solver_name_ = name__;
        return control_input_.gen_evp_solver_name_;
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

    inline bool use_symmetry() const
    {
        return parameters_input_.use_symmetry_;
    }

    inline bool use_symmetry(bool use_symmetry__)
    {
        parameters_input_.use_symmetry_ = use_symmetry__;
        return use_symmetry__;
    }

    inline double iterative_solver_tolerance() const
    {
        return iterative_solver_input_.energy_tolerance_;
    }

    inline double iterative_solver_tolerance(double tolerance__)
    {
        iterative_solver_input_.energy_tolerance_ = tolerance__;
        return iterative_solver_input_.energy_tolerance_;
    }

    inline void set_iterative_solver_type(std::string type__)
    {
        iterative_solver_input_.type_ = type__;
    }

    /// Set the tolerance for empty states.
    inline double empty_states_tolerance(double tolerance__)
    {
        iterative_solver_input_.empty_states_tolerance_ = tolerance__;
        return iterative_solver_input_.empty_states_tolerance_;
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

    /// get the options set at runtime
    json& get_runtime_options_dictionary()
    {
        return runtime_options_dictionary_;
    }

    /// print all options in the terminal
    void print_options()
    {
        const json& dict = get_options_dictionary();
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank != 0)
            MPI_Barrier(MPI_COMM_WORLD);

        std::cout << "the sirius library or the mini apps can be initialized through the interface" << std::endl;
        std::cout << "using the api directly or through a json dictionary. The following contains " << std::endl;
        std::cout << "a description of all the runtime options, that can be used directly to      " << std::endl;
        std::cout << "initialize sirius.                                                          " << std::endl;

        for (auto& el : dict.items()) {
            std::cout << "============================================================================\n";
            std::cout << "                                                                              ";
            std::cout << "                      section : " << el.key() << "                             \n";
            std::cout << "                                                                            \n";
            std::cout << "============================================================================\n";

            for (size_t s = 0; s < dict[el.key()].size(); s++) {
                std::cout << "name of the option : " << dict[el.key()][s]["name"].get<std::string>() << std::endl;
                std::cout << "description : " << dict[el.key()][s]["description"].get<std::string>() << std::endl;
                if (dict[el.key()][s].count("possible_values")) {
                    const auto& v = dict[el.key()][s]["description"].get<std::vector<std::string>>();
                    std::cout << "possible values : " << v[0];
                    for (size_t st = 1; st < v.size(); st++)
                        std::cout << " " << v[st];
                }
                std::cout << "default value : " << dict[el.key()]["default_values"].get<std::string>() << std::endl;
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
};

}; // namespace sirius

#endif
