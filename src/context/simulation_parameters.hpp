// Copyright (c) 2013-2021 Anton Kozhevnikov, Thomas Schulthess
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
 *  \brief Contains definition and implementation of sirius::Simulation_parameters class.
 */

#ifndef __SIMULATION_PARAMETERS_HPP__
#define __SIMULATION_PARAMETERS_HPP__

#include <mpi.h>
#include "typedefs.hpp"
#include "input.hpp"
#include "utils/cmd_args.hpp"
#include "utils/utils.hpp"
#include "memory.hpp"
#include "dft/smearing.hpp"
#include "context/config.hpp"

using namespace sddk;

namespace sirius {

/// Get all possible options for initializing sirius. It is a json dictionary.
json const& get_options_dictionary();

class Config : public config_t
{
  public:
    Config();
    void import(nlohmann::json const& in__);
    void lock()
    {
        dict_["locked"] = true;
    }
};

/// Set of basic parameters of a simulation.
class Simulation_parameters
{
  private:
    /// All user-provided paramters are stored here.
    Config cfg_;
  public:
    Config& cfg()
    {
        return cfg_;
    }

    Config const& cfg() const
    {
        return cfg_;
    }

  protected:
    /// Type of the processing unit.
    device_t processing_unit_{device_t::CPU};

    /// Type of relativity for valence states.
    relativity_t valence_relativity_{relativity_t::zora};

    /// Type of relativity for core states.
    relativity_t core_relativity_{relativity_t::dirac};

    /// Type of electronic structure method.
    electronic_structure_method_t electronic_structure_method_{electronic_structure_method_t::full_potential_lapwlo};

    /// Type of occupation numbers smearing.
    smearing::smearing_t smearing_{smearing::smearing_t::gaussian};

    /// Parameters of the iterative solver.
    //Iterative_solver_input iterative_solver_input_;

    /// Parameters controlling the execution.
    //Control_input control_input_;

    /// Basic input parameters of PP-PW and FP-LAPW methods.
    Parameters_input parameters_input_;

    /// LDA+U input parameters.
    Hubbard_input hubbard_input_;

    /// NLCG input parameters
    NLCG_input nlcg_input_;

    /// json dictionary containing all runtime options set up through the interface
    json runtime_options_dictionary_;

    /// Storage for various memory pools.
    mutable std::map<memory_t, memory_pool> memory_pool_;

    /* copy constructor is forbidden */
    Simulation_parameters(Simulation_parameters const&) = delete;
  public:

    Simulation_parameters()
    {
    }

    /// Import parameters from a file or a serialized json string.
    void import(std::string const &str);

    /// Import parameters from a json dictionary.
    void import(json const& dict);

    /// Import from command line arguments.
    void import(cmd_args const& args__);

    void set_lmax_apw(int lmax_apw__)
    {
        parameters_input_.lmax_apw_ = lmax_apw__;
    }

    void set_lmax_rho(int lmax_rho__)
    {
        parameters_input_.lmax_rho_ = lmax_rho__;
    }

    void set_lmax_pot(int lmax_pot__)
    {
        parameters_input_.lmax_pot_ = lmax_pot__;
    }

    void set_num_mag_dims(int num_mag_dims__)
    {
        assert(num_mag_dims__ == 0 || num_mag_dims__ == 1 || num_mag_dims__ == 3);

        parameters_input_.num_mag_dims_ = num_mag_dims__;
    }

    void set_hubbard_correction(bool hubbard_correction__)
    {
        parameters_input_.hubbard_correction_         = hubbard_correction__;
        hubbard_input_.simplified_hubbard_correction_ = false;
    }

    void set_hubbard_simplified_version()
    {
        hubbard_input_.simplified_hubbard_correction_ = true;
    }

    void set_orthogonalize_hubbard_orbitals(const bool test)
    {
        hubbard_input_.orthogonalize_hubbard_orbitals_ = true;
    }

    void set_normalize_hubbard_orbitals(const bool test)
    {
        hubbard_input_.normalize_hubbard_orbitals_ = true;
    }

    /// Set flag for Gamma-point calculation.
    bool gamma_point(bool gamma_point__)
    {
        parameters_input_.gamma_point_ = gamma_point__;
        return parameters_input_.gamma_point_;
    }

    /// Set dimensions of MPI grid for band diagonalization problem.
    std::vector<int> mpi_grid_dims(std::vector<int> mpi_grid_dims__)
    {
        cfg().control().mpi_grid_dims(mpi_grid_dims__);
        return mpi_grid_dims__;
    }

    void add_xc_functional(std::string name__)
    {
        parameters_input_.xc_functionals_.push_back(name__);
    }

    void electronic_structure_method(std::string name__);

    electronic_structure_method_t electronic_structure_method() const
    {
        return electronic_structure_method_;
    }

    /// Set core relativity for the LAPW method.
    void core_relativity(std::string name__);

    /// Set valence relativity for the LAPW method.
    void valence_relativity(std::string name__);

    void processing_unit(std::string name__);

    void smearing(std::string name__);

    smearing::smearing_t smearing() const
    {
        return smearing_;
    }

    void molecule(bool molecule__)
    {
        parameters_input_.molecule_ = molecule__;
    }

    auto verbosity() const
    {
        return cfg().control().verbosity();
    }

    /// Set verbosity level.
    int verbosity(int level__)
    {
        cfg().control().verbosity(level__);
        return level__;
    }

    auto print_checksum() const
    {
        return cfg().control().print_checksum();
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

    /// Number of non-zero spinor components.
    /** In non magnetic case this is equal to 1, in collinear magnetic case it is also equal to 1 (pure spinors),
     *  in non-collinear case the number of components is 2 (general spinor case). */
    inline int num_spinor_comp() const
    {
        if (num_mag_dims() != 3) {
            return 1;
        } else {
            return 2;
        }
    }

    /// Number of spinor wave-functions labeled by a sinlge band index.
    /** In magnetic collinear case the wave-functions have two spin components, but they describe different
     *  states (pure spin-up, pure spin-dn), thus the number of spinors packed in a single band index is 2.
     *  In non-collinear case we have full two-component spinors for each band index. */
    inline int num_spinors() const
    {
        if (num_mag_dims() == 1) {
            return 2;
        } else {
            return 1;
        }
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
    int num_bands() const
    {
        if (num_fv_states() != -1) {
            return num_fv_states() * num_spinor_comp();
        } else {
            return parameters_input_.num_bands_;
        }
    }

    /// Maximum band occupancy.
    inline int max_occupancy() const
    {
        return (num_mag_dims() == 0) ? 2 : 1;
    }

    /// Minimum occupancy to consider band to be occupied.
    inline double min_occupancy() const
    {
        return cfg_.settings().min_occupancy();
    }

    /// Set minimum occupancy.
    inline double min_occupancy(double val__)
    {
        cfg_.settings().min_occupancy(val__);
        return cfg_.settings().min_occupancy();
    }

    bool so_correction() const
    {
        return parameters_input_.so_correction_;
    }

    bool so_correction(bool so_correction__)
    {
        parameters_input_.so_correction_ = so_correction__;
        return parameters_input_.so_correction_;
    }

    bool hubbard_correction() const
    {
        return parameters_input_.hubbard_correction_;
    }

    bool gamma_point() const
    {
        return parameters_input_.gamma_point_;
    }

    device_t processing_unit() const
    {
        return processing_unit_;
    }

    double smearing_width() const
    {
        return parameters_input_.smearing_width_;
    }

    double smearing_width(double smearing_width__)
    {
        parameters_input_.smearing_width_ = smearing_width__;
        return parameters_input_.smearing_width_;
    }

    void set_auto_rmt(int auto_rmt__)
    {
        parameters_input_.auto_rmt_ = auto_rmt__;
    }

    int auto_rmt() const
    {
        return parameters_input_.auto_rmt_;
    }

    bool need_sv() const
    {
        return (num_spins() == 2 || hubbard_correction() || so_correction());
    }

    std::vector<int> mpi_grid_dims() const
    {
        return cfg().control().mpi_grid_dims();
    }

    int cyclic_block_size() const
    {
        return cfg().control().cyclic_block_size();
    }

    bool full_potential() const
    {
        return (electronic_structure_method_ == electronic_structure_method_t::full_potential_lapwlo);
    }

    std::vector<std::string> const& xc_functionals() const
    {
        return parameters_input_.xc_functionals_;
    }

    /// Get the name of the standard eigen-value solver to use.
    std::string std_evp_solver_name() const
    {
        return cfg().control().std_evp_solver_name();
    }

    /// Set the name of the standard eigen-value solver to use.
    std::string std_evp_solver_name(std::string name__)
    {
        cfg().control().std_evp_solver_name(name__);
        return name__;
    }

    /// Get the name of the generalized eigen-value solver to use.
    std::string gen_evp_solver_name() const
    {
        return cfg().control().gen_evp_solver_name();
    }

    /// Set the name of the generalized eigen-value solver to use.
    std::string gen_evp_solver_name(std::string name__)
    {
        cfg().control().gen_evp_solver_name(name__);
        return name__;
    }

    relativity_t valence_relativity() const
    {
        return valence_relativity_;
    }

    relativity_t core_relativity() const
    {
        return core_relativity_;
    }

    double rmt_max() const
    {
        return cfg().control().rmt_max();
    }

    double spglib_tolerance() const
    {
        return cfg().control().spglib_tolerance();
    }

    bool molecule() const
    {
        return parameters_input_.molecule_;
    }

    /// Get a `using symmetry` flag.
    bool use_symmetry() const
    {
        return parameters_input_.use_symmetry_;
    }

    bool use_symmetry(bool use_symmetry__)
    {
        parameters_input_.use_symmetry_ = use_symmetry__;
        return use_symmetry__;
    }

    /// Get tolerance of the iterative solver.
    double iterative_solver_tolerance() const
    {
        return cfg().iterative_solver().energy_tolerance();
    }

    /// Set the tolerance of the iterative solver.
    double iterative_solver_tolerance(double tolerance__)
    {
        cfg().iterative_solver().energy_tolerance(tolerance__);
        return tolerance__;
    }

    std::string iterative_solver_type(std::string type__)
    {
        cfg().iterative_solver().type(type__);
        return type__;
    }

    /// Set the tolerance for empty states.
    double empty_states_tolerance(double tolerance__)
    {
        cfg().iterative_solver().empty_states_tolerance(tolerance__);
        return tolerance__;
    }

    //Control_input const& control() const
    //{
    //    return control_input_;
    //}

    Parameters_input const& parameters_input() const
    {
        return parameters_input_;
    }

    Hubbard_input const& hubbard_input() const
    {
        return hubbard_input_;
    }

    NLCG_input const& nlcg_input() const
    {
        return nlcg_input_;
    }

    /// Get the options set at runtime.
    json& get_runtime_options_dictionary()
    {
        return runtime_options_dictionary_;
    }

    /// Set the variable which controls the type of sperical coverage.
    inline int sht_coverage(int sht_coverage__)
    {
        cfg_.settings().sht_coverage(sht_coverage__);
        return cfg_.settings().sht_coverage();
    }

    inline std::string esm_bc(std::string const& esm_bc__)
    {
        parameters_input_.esm_bc_ = esm_bc__;
        parameters_input_.enable_esm_ = true;
        return parameters_input_.esm_bc_;
    }

    /// Print all options in the terminal.
    void print_options() const;

    /// Return a reference to a memory pool.
    /** A memory pool is created when this function called for the first time. */
    memory_pool& mem_pool(memory_t M__) const
    {
        if (memory_pool_.count(M__) == 0) {
            memory_pool_.emplace(M__, memory_pool(M__));
        }
        return memory_pool_.at(M__);
    }

    /// Get a default memory pool for a given device.
    memory_pool& mem_pool(device_t dev__)
    {
        switch (dev__) {
            case device_t::CPU: {
                return mem_pool(memory_t::host);
                break;
            }
            case device_t::GPU: {
                return mem_pool(memory_t::device);
                break;
            }
        }
        return mem_pool(memory_t::host); // make compiler happy
    }
};

}; // namespace sirius

#endif
