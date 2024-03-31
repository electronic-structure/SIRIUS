/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file simulation_parameters.hpp
 *
 *  \brief Contains definition and implementation of sirius::Simulation_parameters class.
 */

#ifndef __SIMULATION_PARAMETERS_HPP__
#define __SIMULATION_PARAMETERS_HPP__

#include "core/typedefs.hpp"
#include "core/sf/specfunc.hpp"
#include "core/cmd_args.hpp"
#include "core/json.hpp"
#include "core/memory.hpp"
#include "dft/smearing.hpp"
#include "context/config.hpp"

namespace sirius {

/// Get all possible options for initializing sirius. It is a json dictionary.
nlohmann::json const&
get_options_dictionary();

nlohmann::json const&
get_section_options(std::string const& section__);

class Config : public config_t
{
  public:
    Config();
    void
    import(nlohmann::json const& in__);
    void
    lock()
    {
        dict_["locked"] = true;
    }
    void
    unlock()
    {
        dict_.erase("locked");
    }
};

/// Set of basic parameters of a simulation.
/** This class provides shortcuts to the mostly used input parameters, for example `verbosity`. */
class Simulation_parameters
{
  private:
    /// All user-provided paramters are stored here.
    Config cfg_;

  public:
    Config&
    cfg()
    {
        return cfg_;
    }

    Config const&
    cfg() const
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

    /* copy constructor is forbidden */
    Simulation_parameters(Simulation_parameters const&) = delete;

  public:
    Simulation_parameters()
    {
    }

    /// Import parameters from a file or a serialized json string.
    void
    import(std::string const& str__);

    /// Import parameters from a json dictionary.
    void
    import(nlohmann::json const& dict__);

    /// Import from command line arguments.
    void
    import(cmd_args const& args__);

    void
    lmax_apw(int lmax_apw__)
    {
        cfg().parameters().lmax_apw(lmax_apw__);
    }

    void
    lmax_rho(int lmax_rho__)
    {
        cfg().parameters().lmax_rho(lmax_rho__);
    }

    void
    lmax_pot(int lmax_pot__)
    {
        cfg().parameters().lmax_pot(lmax_pot__);
    }

    void
    set_num_mag_dims(int num_mag_dims__)
    {
        RTE_ASSERT(num_mag_dims__ == 0 || num_mag_dims__ == 1 || num_mag_dims__ == 3);

        cfg().parameters().num_mag_dims(num_mag_dims__);
    }

    void
    set_hubbard_correction(bool hubbard_correction__)
    {
        cfg().parameters().hubbard_correction(hubbard_correction__);
        cfg().hubbard().simplified(false);
    }

    /// Set flag for Gamma-point calculation.
    bool
    gamma_point(bool gamma_point__)
    {
        cfg().parameters().gamma_point(gamma_point__);
        return gamma_point__;
    }

    /// Set dimensions of MPI grid for band diagonalization problem.
    std::vector<int>
    mpi_grid_dims(std::vector<int> mpi_grid_dims__)
    {
        cfg().control().mpi_grid_dims(mpi_grid_dims__);
        return mpi_grid_dims__;
    }

    void
    add_xc_functional(std::string name__)
    {
        auto xcfunc = cfg().parameters().xc_functionals();
        xcfunc.push_back(name__);
        cfg().parameters().xc_functionals(xcfunc);
    }

    void
    electronic_structure_method(std::string name__);

    auto
    electronic_structure_method() const
    {
        return electronic_structure_method_;
    }

    /// Set core relativity for the LAPW method.
    void
    core_relativity(std::string name__);

    /// Set valence relativity for the LAPW method.
    void
    valence_relativity(std::string name__);

    void
    processing_unit(std::string name__);

    void
    smearing(std::string name__);

    auto
    smearing() const -> smearing::smearing_t
    {
        return smearing_;
    }

    void
    molecule(bool molecule__)
    {
        cfg().parameters().molecule(molecule__);
    }

    auto
    verbosity() const
    {
        return cfg().control().verbosity();
    }

    /// Set verbosity level.
    int
    verbosity(int level__)
    {
        cfg().control().verbosity(level__);
        return level__;
    }

    inline int
    lmax_rho() const
    {
        return cfg().parameters().lmax_rho();
    }

    inline int
    lmmax_rho() const
    {
        return sf::lmmax(lmax_rho());
    }

    inline int
    lmax_pot() const
    {
        return cfg().parameters().lmax_pot();
    }

    inline int
    lmmax_pot() const
    {
        return sf::lmmax(this->lmax_pot());
    }

    inline double
    aw_cutoff() const
    {
        return cfg().parameters().aw_cutoff();
    }

    inline double
    aw_cutoff(double aw_cutoff__)
    {
        cfg().parameters().aw_cutoff(aw_cutoff__);
        return aw_cutoff__;
    }

    /// Plane-wave cutoff for G-vectors (in 1/[a.u.]).
    inline double
    pw_cutoff() const
    {
        return cfg().parameters().pw_cutoff();
    }

    /// Set plane-wave cutoff.
    inline double
    pw_cutoff(double pw_cutoff__)
    {
        cfg().parameters().pw_cutoff(pw_cutoff__);
        return pw_cutoff__;
    }

    /// Cutoff for G+k vectors (in 1/[a.u.]).
    inline double
    gk_cutoff() const
    {
        return cfg().parameters().gk_cutoff();
    }

    /// Set the cutoff for G+k vectors.
    inline double
    gk_cutoff(double gk_cutoff__)
    {
        cfg().parameters().gk_cutoff(gk_cutoff__);
        return gk_cutoff__;
    }

    /// Number of dimensions in the magnetization vector.
    inline int
    num_mag_dims() const
    {
        auto nmd = cfg().parameters().num_mag_dims();
        RTE_ASSERT(nmd == 0 || nmd == 1 || nmd == 3);
        return nmd;
    }

    /// Number of spin components.
    /** This parameter can take only two values: 1 -- non-magnetic calcaulation and wave-functions,
     *  2 -- spin-polarized calculation and wave-functions. */
    inline int
    num_spins() const
    {
        return (num_mag_dims() == 0) ? 1 : 2;
    }

    /// Number of components in the complex density matrix.
    /** In case of non-collinear magnetism only one out of two non-diagonal components is stored. */
    inline int
    num_mag_comp() const // TODO: rename; current name does not reflect the meaning
    {
        return (num_mag_dims() == 3) ? 3 : num_spins(); // std::max(mag_dims, spins)
    }

    /// Number of non-zero spinor components.
    /** In non magnetic case this is equal to 1, in collinear magnetic case it is also equal to 1 (pure spinors),
     *  in non-collinear case the number of components is 2 (general spinor case). */
    inline int
    num_spinor_comp() const
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
    inline int
    num_spinors() const
    {
        if (num_mag_dims() == 1) {
            return 2;
        } else {
            return 1;
        }
    }

    /// Set the number of first-variational states.
    inline int
    num_fv_states(int num_fv_states__)
    {
        cfg().parameters().num_fv_states(num_fv_states__);
        return num_fv_states__;
    }

    /// Number of first-variational states.
    inline int
    num_fv_states() const
    {
        return cfg().parameters().num_fv_states();
    }

    /// Set the number of bands.
    inline int
    num_bands(int num_bands__)
    {
        cfg().parameters().num_bands(num_bands__);
        return num_bands__;
    }

    /// Total number of bands.
    int
    num_bands() const
    {
        if (this->num_fv_states() != -1) {
            return this->num_fv_states() * this->num_spinor_comp();
        } else {
            return cfg().parameters().num_bands();
        }
    }

    /// Maximum band occupancy.
    inline int
    max_occupancy() const
    {
        return (this->num_mag_dims() == 0) ? 2 : 1;
    }

    /// Minimum occupancy to consider band to be occupied.
    inline double
    min_occupancy() const
    {
        return cfg_.iterative_solver().min_occupancy();
    }

    /// Set minimum occupancy.
    inline double
    min_occupancy(double val__)
    {
        cfg().iterative_solver().min_occupancy(val__);
        return cfg().iterative_solver().min_occupancy();
    }

    bool
    so_correction() const
    {
        return cfg().parameters().so_correction();
    }

    bool
    so_correction(bool so_correction__)
    {
        cfg().parameters().so_correction(so_correction__);
        return so_correction__;
    }

    bool
    hubbard_correction() const
    {
        return cfg().parameters().hubbard_correction();
    }

    bool
    hubbard_constrained_calculation() const
    {
        return cfg().hubbard().constrained_calculation();
    }

    bool
    gamma_point() const
    {
        return cfg().parameters().gamma_point();
    }

    auto
    processing_unit() const
    {
        return processing_unit_;
    }

    double
    smearing_width() const
    {
        return cfg().parameters().smearing_width();
    }

    double
    smearing_width(double smearing_width__)
    {
        cfg().parameters().smearing_width(smearing_width__);
        return smearing_width__;
    }

    void
    set_auto_rmt(int auto_rmt__)
    {
        cfg().parameters().auto_rmt(auto_rmt__);
    }

    int
    auto_rmt() const
    {
        return cfg().parameters().auto_rmt();
    }

    bool
    need_sv() const
    {
        return (num_spins() == 2 || hubbard_correction() || so_correction());
    }

    std::vector<int>
    mpi_grid_dims() const
    {
        return cfg().control().mpi_grid_dims();
    }

    int
    cyclic_block_size() const
    {
        return cfg().control().cyclic_block_size();
    }

    bool
    full_potential() const
    {
        return (electronic_structure_method_ == electronic_structure_method_t::full_potential_lapwlo);
    }

    std::vector<std::string>
    xc_functionals() const
    {
        return cfg().parameters().xc_functionals();
    }

    /// Get the name of the standard eigen-value solver to use.
    std::string
    std_evp_solver_name() const
    {
        return cfg().control().std_evp_solver_name();
    }

    /// Set the name of the standard eigen-value solver to use.
    std::string
    std_evp_solver_name(std::string name__)
    {
        cfg().control().std_evp_solver_name(name__);
        return name__;
    }

    /// Get the name of the generalized eigen-value solver to use.
    std::string
    gen_evp_solver_name() const
    {
        return cfg().control().gen_evp_solver_name();
    }

    /// Set the name of the generalized eigen-value solver to use.
    std::string
    gen_evp_solver_name(std::string name__)
    {
        cfg().control().gen_evp_solver_name(name__);
        return name__;
    }

    relativity_t
    valence_relativity() const
    {
        return valence_relativity_;
    }

    relativity_t
    core_relativity() const
    {
        return core_relativity_;
    }

    double
    rmt_max() const
    {
        return cfg().control().rmt_max();
    }

    double
    spglib_tolerance() const
    {
        return cfg().control().spglib_tolerance();
    }

    bool
    molecule() const
    {
        return cfg().parameters().molecule();
    }

    /// Get a use_symmetry flag.
    /** If crystal symmetry is used, density and potential are symmetrized. */
    bool
    use_symmetry() const
    {
        return cfg().parameters().use_symmetry();
    }

    /// Set a use_symmetry flag.
    bool
    use_symmetry(bool use_symmetry__)
    {
        cfg().parameters().use_symmetry(use_symmetry__);
        return use_symmetry__;
    }

    std::string
    iterative_solver_type(std::string type__)
    {
        cfg().iterative_solver().type(type__);
        return type__;
    }

    /// Set the tolerance for empty states.
    double
    empty_states_tolerance(double tolerance__)
    {
        cfg().iterative_solver().empty_states_tolerance(tolerance__);
        return tolerance__;
    }

    /// Set the variable which controls the type of sperical coverage.
    inline int
    sht_coverage(int sht_coverage__)
    {
        cfg_.settings().sht_coverage(sht_coverage__);
        return cfg_.settings().sht_coverage();
    }
};

}; // namespace sirius

#endif
