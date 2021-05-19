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

/** \file sirius_api.cpp
 *
 *  \brief Fortran API.
 */

#include <ctype.h>
#include <iostream>
#include "sirius.hpp"
#include "utils/any_ptr.hpp"
#include "utils/profiler.hpp"
#include "error_codes.hpp"
#ifdef SIRIUS_NLCGLIB
#include "nlcglib/adaptor.hpp"
#include "nlcglib/nlcglib.hpp"
#endif

static inline void sirius_exit(int error_code__, std::string msg__ = "")
{
    switch (error_code__) {
        case SIRIUS_ERROR_UNKNOWN: {
            printf("SIRIUS: unknown error\n");
            break;
        }
        case SIRIUS_ERROR_RUNTIME: {
            printf("SIRIUS: run-time error\n");
            break;
        }
        case SIRIUS_ERROR_EXCEPTION: {
            printf("SIRIUS: exception\n");
            break;
        }
        default: {
            printf("SIRIUS: unknown error code: %i\n", error_code__);
            break;
        }
    }

    if (msg__.size()) {
        printf("%s\n", msg__.c_str());
    }
    if (!Communicator::is_finalized()) {
        Communicator::world().abort(error_code__);
    }
    fflush(stdout);
    std::cout << std::flush;
    std::exit(error_code__);
}

template <typename F>
static void call_sirius(F&& f__, int* error_code__)
{
    try {
        f__();
        if (error_code__) {
            *error_code__ = SIRIUS_SUCCESS;
            return;
        }
    }
    catch (std::runtime_error const& e) {
        if (error_code__) {
            *error_code__ = SIRIUS_ERROR_RUNTIME;
            return;
       } else {
           sirius_exit(SIRIUS_ERROR_RUNTIME, e.what());
       }
    }
    catch (std::exception const&  e) {
        if (error_code__) {
            *error_code__ = SIRIUS_ERROR_EXCEPTION;
            return;
       } else {
           sirius_exit(SIRIUS_ERROR_EXCEPTION, e.what());
       }
    }
    catch (...) {
        if (error_code__) {
            *error_code__ = SIRIUS_ERROR_UNKNOWN;
            return;
        } else {
            sirius_exit(SIRIUS_ERROR_UNKNOWN);
        }
    }
}

sirius::Simulation_context& get_sim_ctx(void* const* h)
{
    if (h == nullptr || *h == nullptr) {
        throw std::runtime_error("Non-existing simulation context handler");
    }
    return static_cast<utils::any_ptr*>(*h)->get<sirius::Simulation_context>();
}

sirius::DFT_ground_state& get_gs(void* const* h)
{
    if (h == nullptr || *h == nullptr) {
        throw std::runtime_error("Non-existing DFT ground state handler");
    }
    return static_cast<utils::any_ptr*>(*h)->get<sirius::DFT_ground_state>();
}

sirius::K_point_set& get_ks(void* const* h)
{
    if (h == nullptr || *h == nullptr) {
        throw std::runtime_error("Non-existing K-point set handler");
    }
    return static_cast<utils::any_ptr*>(*h)->get<sirius::K_point_set>();
}

/// Index of Rlm in QE in the block of lm coefficients for a given l.
static inline int idx_m_qe(int m__)
{
    return (m__ > 0) ? 2 * m__ - 1 : -2 * m__;
}

/// Mapping of atomic indices from SIRIUS to QE order.
static inline std::vector<int> atomic_orbital_index_map_QE(sirius::Atom_type const& type__)
{
    int nbf = type__.mt_basis_size();

    std::vector<int> idx_map(nbf);
    for (int xi = 0; xi < nbf; xi++) {
        int m       = type__.indexb(xi).m;
        int idxrf   = type__.indexb(xi).idxrf;
        idx_map[xi] = type__.indexb().index_by_idxrf(idxrf) + idx_m_qe(m); /* beginning of lm-block + new offset in lm block */
    }
    return idx_map;
}

static inline int phase_Rlm_QE(::sirius::Atom_type const& type__, int xi__)
{
    return (type__.indexb(xi__).m < 0 && (-type__.indexb(xi__).m) % 2 == 0) ? -1 : 1;
}

extern "C" {

/*
@api begin
sirius_initialize:
  doc: Initialize the SIRIUS library.
  arguments:
    call_mpi_init:
      type: bool
      attr: in, required
      doc: If .true. then MPI_Init must be called prior to initialization.
@api end
*/
void sirius_initialize(bool const* call_mpi_init__)
{
    sirius::initialize(*call_mpi_init__);
}

/*
@api begin
sirius_finalize:
  doc: Shut down the SIRIUS library
  arguments:
    call_mpi_fin:
      type: bool
      attr: in, optional
      doc: If .true. then MPI_Finalize must be called after the shutdown.
    call_device_reset:
      type: bool
      attr: in, optional
      doc: If .true. then cuda device is reset after shutdown.
    call_fftw_fin:
      type: bool
      attr: in, optional
      doc: If .true. then fft_cleanup must be called after the shutdown.
@api end
*/
void sirius_finalize(bool const* call_mpi_fin__, bool const *call_device_reset__, bool const* call_fftw_fin__)
{
    bool mpi_fin{true};
    bool device_reset{true};
    bool fftw_fin{true};

    if (call_mpi_fin__!= nullptr) {
        mpi_fin = *call_mpi_fin__;
    }

    if (call_device_reset__ != nullptr) {
        device_reset = *call_device_reset__;
    }

    if (call_fftw_fin__ != nullptr) {
        fftw_fin = *call_fftw_fin__;
    }

    sirius::finalize(mpi_fin, device_reset, fftw_fin);
}

/*
@api begin
sirius_start_timer:
  doc: Start the timer.
  arguments:
    name:
      type: string
      attr: in, required
      doc: Timer label.
@api end
*/
void sirius_start_timer(char const* name__)
{
    ::utils::global_rtgraph_timer.start(name__);
}

/*
@api begin
sirius_stop_timer:
  doc: Stop the running timer.
  arguments:
    name:
      type: string
      attr: in, required
      doc: Timer label.
@api end
*/
void sirius_stop_timer(char const* name__)
{
    ::utils::global_rtgraph_timer.stop(name__);
}

/*
@api begin
sirius_print_timers:
  doc: Print all timers.
  arguments:
    flatten:
      type: bool
      attr: in, required
      doc: If true, flat list of timers is printed.
@api end
*/
void sirius_print_timers(bool* flatten__)
{
    auto timing_result = ::utils::global_rtgraph_timer.process();
    if (*flatten__) {
        timing_result = timing_result.flatten(1).sort_nodes();
    }
    std::cout << timing_result.print({rt_graph::Stat::Count, rt_graph::Stat::Total, rt_graph::Stat::Percentage,
                                      rt_graph::Stat::SelfPercentage, rt_graph::Stat::Median, rt_graph::Stat::Min,
                                      rt_graph::Stat::Max});
}

/*
@api begin
sirius_serialize_timers:
  doc: Save all timers to JSON file.
  arguments:
    fname:
      type: string
      attr: in, required
      doc: Name of the output JSON file.
@api end
*/
void sirius_serialize_timers(char const* fname__)
{
    auto timing_result = ::utils::global_rtgraph_timer.process();
    std::ofstream ofs(fname__, std::ofstream::out | std::ofstream::trunc);
    ofs << timing_result.json();
}

/*
@api begin
sirius_integrate:
  doc: Spline integration of f(x)*x^m.
  arguments:
    m:
      type: int
      attr: in, required
      doc: Defines the x^{m} factor.
    np:
      type: int
      attr: in, required
      doc: Number of x-points.
    x:
      type: double
      attr: in, required
      doc: List of x-points.
    f:
      type: double
      attr: in, required
      doc: List of function values.
    result:
      type: double
      attr: out, required
      doc: Resulting value.
@api end
*/
void sirius_integrate(int const* m__, int const* np__, double const* x__, double const* f__, double* result__)
{
    sirius::Radial_grid_ext<double> rgrid(*np__, x__);
    sirius::Spline<double> s(rgrid, std::vector<double>(f__, f__ + *np__));
    *result__ = s.integrate(*m__);
}

/*
@api begin
sirius_context_initialized:
  doc: Check if the simulation context is initialized.
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Simulation context handler.
    status:
      type: bool
      attr: out, required
      doc: Status of the library (true if initialized)
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void sirius_context_initialized(void* const* handler__, bool* status__, int* error_code__)
{
    call_sirius([&]()
    {
        auto& sim_ctx = get_sim_ctx(handler__);
        *status__ = sim_ctx.initialized();
    }, error_code__);
}

/*
@api begin
sirius_create_context:
  doc: Create context of the simulation.
  full_doc: Simulation context is the complex data structure that holds all the parameters of the
    individual simulation.

    The context must be created, populated with the correct parameters and
    initialized before using all subsequent SIRIUS functions.
  arguments:
    fcomm:
      type: int
      attr: in, required, value
      doc: Entire communicator of the simulation.
    handler:
      type: void*
      attr: out, required
      doc: New empty simulation context.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void sirius_create_context(int fcomm__, void** handler__, int* error_code__)
{
    call_sirius([&]()
    {
        auto& comm = Communicator::map_fcomm(fcomm__);
        *handler__ = new utils::any_ptr(new sirius::Simulation_context(comm));
    }, error_code__);
}

/*
@api begin
sirius_import_parameters:
  doc: Import parameters of simulation from a JSON string
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Simulation context handler.
    str:
      type: string
      attr: in, optional
      doc: JSON string with parameters or a JSON file.
    error_code:
      type: int
      attr: out, optional
      doc: Error code
@api end
*/
void sirius_import_parameters(void* const* handler__, char const* str__, int* error_code__)
{
    call_sirius([&]()
    {
        auto& sim_ctx = get_sim_ctx(handler__);
        if (str__) {
            sim_ctx.import(std::string(str__));
        } else {
            sim_ctx.import(sim_ctx.get_runtime_options_dictionary());
        }
    }, error_code__);
}

/*
@api begin
sirius_set_parameters:
  doc: Set parameters of the simulation.
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Simulation context handler
    lmax_apw:
      type: int
      attr: in, optional
      doc: Maximum orbital quantum number for APW functions.
    lmax_rho:
      type: int
      attr: in, optional
      doc: Maximum orbital quantum number for density.
    lmax_pot:
      type: int
      attr: in, optional
      doc: Maximum orbital quantum number for potential.
    num_fv_states:
      type: int
      attr: in, optional
      doc: Number of first-variational states.
    num_bands:
      type: int
      attr: in, optional
      doc: Number of bands.
    num_mag_dims:
      type: int
      attr: in, optional
      doc: Number of magnetic dimensions.
    pw_cutoff:
      type: double
      attr: in, optional
      doc: Cutoff for G-vectors.
    gk_cutoff:
      type: double
      attr: in, optional
      doc: Cutoff for G+k-vectors.
    fft_grid_size:
      type: int
      attr: in, optional, dimension(3)
      doc: Size of the fine-grain FFT grid.
    auto_rmt:
      type: int
      attr: in, optional
      doc: Set the automatic search of muffin-tin radii.
    gamma_point:
      type: bool
      attr: in, optional
      doc: True if this is a Gamma-point calculation.
    use_symmetry:
      type: bool
      attr: in, optional
      doc: True if crystal symmetry is taken into account.
    so_correction:
      type: bool
      attr: in, optional
      doc: True if spin-orbit correnctio is enabled.
    valence_rel:
      type: string
      attr: in, optional
      doc: Valence relativity treatment.
    core_rel:
      type: string
      attr: in, optional
      doc: Core relativity treatment.
    iter_solver_tol:
      type: double
      attr: in, optional
      doc: Tolerance of the iterative solver.
    iter_solver_tol_empty:
      type: double
      attr: in, optional
      doc: Tolerance for the empty states.
    iter_solver_type:
      type: string
      attr: in, optional
      doc: Type of iterative solver.
    verbosity:
      type: int
      attr: in, optional
      doc: Verbosity level.
    hubbard_correction:
      type: bool
      attr: in, optional
      doc: True if LDA+U correction is enabled.
    hubbard_correction_kind:
      type: int
      attr: in, optional
      doc: Type of LDA+U implementation (simplified or full).
    hubbard_orbitals:
      type: string
      attr: in, optional
      doc: Type of localized orbitals.
    sht_coverage:
      type: int
      attr: in, optional
      doc: Type of spherical coverage (0 for Lebedev-Laikov, 1 for uniform).
    min_occupancy:
      type: double
      attr: in, optional
      doc: Minimum band occupancy to trat is as "occupied".
    smearing:
      type: string
      attr: in, optional
      doc: Type of occupancy smearing.
    smearing_width:
      type: double
      attr: in, optional
      doc: Smearing width
    spglib_tol:
      type: double
      attr: in, optional
      doc: Tolerance for the spglib symmetry search.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void sirius_set_parameters(void*  const* handler__,
                           int    const* lmax_apw__,
                           int    const* lmax_rho__,
                           int    const* lmax_pot__,
                           int    const* num_fv_states__,
                           int    const* num_bands__,
                           int    const* num_mag_dims__,
                           double const* pw_cutoff__,
                           double const* gk_cutoff__,
                           int    const* fft_grid_size__,
                           int    const* auto_rmt__,
                           bool   const* gamma_point__,
                           bool   const* use_symmetry__,
                           bool   const* so_correction__,
                           char   const* valence_rel__,
                           char   const* core_rel__,
                           double const* iter_solver_tol__,
                           double const* iter_solver_tol_empty__,
                           char   const* iter_solver_type__,
                           int    const* verbosity__,
                           bool   const* hubbard_correction__,
                           int    const* hubbard_correction_kind__,
                           char   const* hubbard_orbitals__,
                           int    const* sht_coverage__,
                           double const* min_occupancy__,
                           char   const* smearing__,
                           double const* smearing_width__,
                           double const* spglib_tol__,
                           int*          error_code__)
{
    call_sirius([&]()
    {
        auto& sim_ctx = get_sim_ctx(handler__);
        if (lmax_apw__ != nullptr) {
            sim_ctx.set_lmax_apw(*lmax_apw__);
        }
        if (lmax_rho__ != nullptr) {
            sim_ctx.set_lmax_rho(*lmax_rho__);
        }
        if (lmax_pot__ != nullptr) {
            sim_ctx.set_lmax_pot(*lmax_pot__);
        }
        if (num_fv_states__ != nullptr) {
            sim_ctx.num_fv_states(*num_fv_states__);
        }
        if (num_bands__ != nullptr) {
            sim_ctx.num_bands(*num_bands__);
        }
        if (num_mag_dims__ != nullptr) {
            sim_ctx.set_num_mag_dims(*num_mag_dims__);
        }
        if (pw_cutoff__ != nullptr) {
            sim_ctx.pw_cutoff(*pw_cutoff__);
        }
        if (gk_cutoff__ != nullptr) {
            sim_ctx.gk_cutoff(*gk_cutoff__);
        }
        if (auto_rmt__ != nullptr) {
            sim_ctx.set_auto_rmt(*auto_rmt__);
        }
        if (gamma_point__ != nullptr) {
            sim_ctx.gamma_point(*gamma_point__);
        }
        if (use_symmetry__ != nullptr) {
            sim_ctx.use_symmetry(*use_symmetry__);
        }
        if (so_correction__ != nullptr) {
            sim_ctx.so_correction(*so_correction__);
        }
        if (valence_rel__ != nullptr) {
            sim_ctx.valence_relativity(valence_rel__);
        }
        if (core_rel__ != nullptr) {
            sim_ctx.core_relativity(core_rel__);
        }
        if (iter_solver_tol__ != nullptr) {
            sim_ctx.iterative_solver_tolerance(*iter_solver_tol__);
        }
        if (iter_solver_tol_empty__ != nullptr) {
            sim_ctx.empty_states_tolerance(*iter_solver_tol_empty__);
        }
        if (iter_solver_type__ != nullptr) {
            sim_ctx.iterative_solver_type(std::string(iter_solver_type__));
        }
        if (verbosity__ != nullptr) {
            sim_ctx.verbosity(*verbosity__);
        }
        if (hubbard_correction__ != nullptr) {
            sim_ctx.set_hubbard_correction(*hubbard_correction__);
        }
        if (hubbard_correction_kind__ != nullptr) {
            if (*hubbard_correction_kind__ == 0) {
                sim_ctx.cfg().hubbard().simplified(true);
            }
        }
        if (hubbard_orbitals__ != nullptr) {
            std::string s(hubbard_orbitals__);
            if (s == "ortho-atomic") {
                sim_ctx.cfg().hubbard().orthogonalize(true);
            }
            if (s == "norm-atomic") {
                 sim_ctx.cfg().hubbard().normalize(true);
            }
        }
        if (fft_grid_size__ != nullptr) {
            sim_ctx.fft_grid_size({fft_grid_size__[0], fft_grid_size__[1], fft_grid_size__[2]});
        }
        if (sht_coverage__ != nullptr) {
            sim_ctx.sht_coverage(*sht_coverage__);
        }
        if (min_occupancy__ != nullptr) {
            sim_ctx.min_occupancy(*min_occupancy__);
        }
        if (smearing__ != nullptr) {
            sim_ctx.smearing(smearing__);
        }
        if (smearing_width__ != nullptr) {
            sim_ctx.smearing_width(*smearing_width__);
        }
        if (spglib_tol__ != nullptr) {
            sim_ctx.cfg().control().spglib_tolerance(*spglib_tol__);
        }
    }, error_code__);
}

/*
@api begin
sirius_get_parameters:
  doc: Get parameters of the simulation.
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Simulation context handler
    lmax_apw:
      type: int
      attr: out, optional
      doc: Maximum orbital quantum number for APW functions.
    lmax_rho:
      type: int
      attr: out, optional
      doc: Maximum orbital quantum number for density.
    lmax_pot:
      type: int
      attr: out, optional
      doc: Maximum orbital quantum number for potential.
    num_fv_states:
      type: int
      attr: out, optional
      doc: Number of first-variational states.
    num_bands:
      type: int
      attr: out, optional
      doc: Number of bands.
    num_spins:
      type: int
      attr: out, optional
      doc: Number of spins.
    num_mag_dims:
      type: int
      attr: out, optional
      doc: Number of magnetic dimensions.
    pw_cutoff:
      type: double
      attr: out, optional
      doc: Cutoff for G-vectors.
    gk_cutoff:
      type: double
      attr: out, optional
      doc: Cutoff for G+k-vectors.
    fft_grid_size:
      type: int
      attr: out, optional, dimension(3)
      doc: Size of the fine-grain FFT grid.
    auto_rmt:
      type: int
      attr: out, optional
      doc: Set the automatic search of muffin-tin radii.
    gamma_point:
      type: bool
      attr: out, optional
      doc: True if this is a Gamma-point calculation.
    use_symmetry:
      type: bool
      attr: out, optional
      doc: True if crystal symmetry is taken into account.
    so_correction:
      type: bool
      attr: out, optional
      doc: True if spin-orbit correnctio is enabled.
    iter_solver_tol:
      type: double
      attr: out, optional
      doc: Tolerance of the iterative solver.
    iter_solver_tol_empty:
      type: double
      attr: out, optional
      doc: Tolerance for the empty states.
    verbosity:
      type: int
      attr: out, optional
      doc: Verbosity level.
    hubbard_correction:
      type: bool
      attr: out, optional
      doc: True if LDA+U correction is enabled.
    evp_work_count:
      type: double
      attr: out, optional
      doc: Internal counter of total eigen-value problem work.
    num_loc_op_applied:
      type: int
      attr: out, optional
      doc: Internal counter of the number of wave-functions to which Hamiltonian was applied.
    num_sym_op:
      type: int
      attr: out, optional
      doc: Number of symmetry operations discovered by spglib
    electronic_structure_method:
      type: string
      attr: out, optional
      doc: Type of electronic structure method.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void sirius_get_parameters(void* const* handler__,
                           int*         lmax_apw__,
                           int*         lmax_rho__,
                           int*         lmax_pot__,
                           int*         num_fv_states__,
                           int*         num_bands__,
                           int*         num_spins__,
                           int*         num_mag_dims__,
                           double*      pw_cutoff__,
                           double*      gk_cutoff__,
                           int*         fft_grid_size__,
                           int*         auto_rmt__,
                           bool*        gamma_point__,
                           bool*        use_symmetry__,
                           bool*        so_correction__,
                           double*      iter_solver_tol__,
                           double*      iter_solver_tol_empty__,
                           int*         verbosity__,
                           bool*        hubbard_correction__,
                           double*      evp_work_count__,
                           int*         num_loc_op_applied__,
                           int*         num_sym_op__,
                           char*        electronic_structure_method__,
                           int*         error_code__)
{
    call_sirius([&]()
    {
        auto& sim_ctx = get_sim_ctx(handler__);
        if (lmax_apw__) {
            *lmax_apw__ = sim_ctx.lmax_apw();
        }
        if (lmax_rho__) {
            *lmax_rho__ = sim_ctx.lmax_rho();
        }
        if (lmax_pot__) {
            *lmax_pot__ = sim_ctx.lmax_pot();
        }
        if (num_fv_states__) {
            *num_fv_states__ = sim_ctx.num_fv_states();
        }
        if (num_bands__) {
            *num_bands__ = sim_ctx.num_bands();
        }
        if (num_spins__) {
            *num_spins__ = sim_ctx.num_spins();
        }
        if (num_mag_dims__) {
            *num_mag_dims__ = sim_ctx.num_mag_dims();
        }
        if (pw_cutoff__) {
            *pw_cutoff__ = sim_ctx.pw_cutoff();
        }
        if (gk_cutoff__) {
            *gk_cutoff__ = sim_ctx.gk_cutoff();
        }
        if (auto_rmt__) {
            *auto_rmt__ = sim_ctx.auto_rmt();
        }
        if (gamma_point__) {
            *gamma_point__ = sim_ctx.gamma_point();
        }
        if (use_symmetry__) {
            *use_symmetry__ = sim_ctx.use_symmetry();
        }
        if (so_correction__) {
            *so_correction__ = sim_ctx.so_correction();
        }
        if (iter_solver_tol__) {
            *iter_solver_tol__ = sim_ctx.iterative_solver_tolerance();
        }
        if (iter_solver_tol_empty__) {
            *iter_solver_tol_empty__ = sim_ctx.cfg().iterative_solver().empty_states_tolerance();
        }
        if (verbosity__) {
            *verbosity__ = sim_ctx.verbosity();
        }
        if (hubbard_correction__) {
            *hubbard_correction__ = sim_ctx.hubbard_correction();
        }
        if (fft_grid_size__) {
            for (int x: {0, 1, 2}) {
                fft_grid_size__[x] = sim_ctx.fft_grid()[x];
            }
        }
        if (evp_work_count__) {
            *evp_work_count__ = sim_ctx.evp_work_count();
        }
        if (num_loc_op_applied__) {
            *num_loc_op_applied__ = sim_ctx.num_loc_op_applied();
        }
        if (num_sym_op__) {
            if (sim_ctx.use_symmetry()) {
                *num_sym_op__ = sim_ctx.unit_cell().symmetry().size();
            } else {
                *num_sym_op__ = 0;
            }
        }
        if (electronic_structure_method__) {
            auto str = sim_ctx.cfg().parameters().electronic_structure_method();
            std::copy(str.c_str(), str.c_str() + str.length() + 1, electronic_structure_method__);
        }
    }, error_code__);
}


/*
@api begin
sirius_add_xc_functional:
  doc: Add one of the XC functionals.
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Simulation context handler
    name:
      type: string
      attr: in, required
      doc: LibXC label of the functional.
@api end
*/
void sirius_add_xc_functional(void* const* handler__, char const* name__)
{
    auto& sim_ctx = get_sim_ctx(handler__);
    sim_ctx.add_xc_functional(std::string(name__));
}

/*
@api begin
sirius_insert_xc_functional:
  doc: Add one of the XC functionals.
  arguments:
    gs_handler:
      type: void*
      attr: in, required
      doc: Handler of the ground state
    name:
      type: string
      attr: in, required
      doc: LibXC label of the functional.
@api end
*/
void sirius_insert_xc_functional(void* const* gs_handler__,
                                 char const* name__)
{ // TODO: deprecate and remove
    auto& gs = get_gs(gs_handler__);
    auto& potential = gs.potential();
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank==0)
        std::cout << "insert functional: " << name__ << "\n";
    potential.insert_xc_functionals({name__});
}

/*
@api begin
sirius_set_mpi_grid_dims:
  doc: Set dimensions of the MPI grid.
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Simulation context handler
    ndims:
      type: int
      attr: in, required
      doc: Number of dimensions.
    dims:
      type: int
      attr: in, required, dimension(ndims)
      doc: Size of each dimension.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void sirius_set_mpi_grid_dims(void* const* handler__, int const* ndims__, int const* dims__, int* error_code__)
{
    call_sirius([&]()
    {
        assert(*ndims__ > 0);
        auto& sim_ctx = get_sim_ctx(handler__);
        std::vector<int> dims(dims__, dims__ + *ndims__);
        sim_ctx.mpi_grid_dims(dims);
    }, error_code__);
}

/*
@api begin
sirius_set_lattice_vectors:
  doc: Set vectors of the unit cell.
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Simulation context handler
    a1:
      type: double
      attr: in, required, dimension(3)
      doc: 1st vector
    a2:
      type: double
      attr: in, required, dimension(3)
      doc: 2nd vector
    a3:
      type: double
      attr: in, required, dimension(3)
      doc: 3rd vector
@api end
*/
void sirius_set_lattice_vectors(void*  const* handler__,
                                double const* a1__,
                                double const* a2__,
                                double const* a3__)
{
    auto& sim_ctx = get_sim_ctx(handler__);
    sim_ctx.unit_cell().set_lattice_vectors(vector3d<double>(a1__), vector3d<double>(a2__), vector3d<double>(a3__));
}

/*
@api begin
sirius_initialize_context:
  doc: Initialize simulation context.
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Simulation context handler.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void sirius_initialize_context(void* const* handler__, int* error_code__)
{
    call_sirius([&]()
    {
        auto& sim_ctx = get_sim_ctx(handler__);
        sim_ctx.initialize();
        return 0;
    }, error_code__);
}

/*
@api begin
sirius_update_context:
  doc: Update simulation context after changing lattice or atomic positions.
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Simulation context handler.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void sirius_update_context(void* const* handler__, int* error_code__)
{
    call_sirius([&]()
    {
        auto& sim_ctx = get_sim_ctx(handler__);
        sim_ctx.update();
        return 0;
    }, error_code__);
}

/*
@api begin
sirius_print_info:
  doc: Print basic info
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Simulation context handler.
@api end
*/
void sirius_print_info(void* const* handler__)
{
    auto& sim_ctx = get_sim_ctx(handler__);
    sim_ctx.print_info();
}

/*
@api begin
sirius_free_handler:
  doc: Free any handler of object created by SIRIUS.
  arguments:
    handler:
      type: void*
      attr: inout, required
      doc: Handler of the object.
    error_code:
      type: int
      attr: out, optional
      doc: Error code
@api end
*/
void sirius_free_handler(void** handler__, int* error_code__)
{
    call_sirius([&]()
    {
        if (*handler__ != nullptr) {
            delete static_cast<utils::any_ptr*>(*handler__);
        }
        *handler__ = nullptr;
    }, error_code__);
}

/*
@api begin
sirius_set_periodic_function_ptr:
  doc: Set pointer to density or megnetization.
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Handler of the DFT ground state object.
    label:
      type: string
      attr: in, required
      doc: Label of the function.
    f_mt:
      type: double
      attr: in, optional
      doc: Pointer to the muffin-tin part of the function.
    f_rg:
      type: double
      attr: in, optional
      doc: Pointer to the regualr-grid part of the function.
@api end
*/
void sirius_set_periodic_function_ptr(void*  const* handler__,
                                      char   const* label__,
                                      double*       f_mt__,
                                      double*       f_rg__)
{
    auto& gs = get_gs(handler__);
    std::string label(label__);

    std::map<std::string, sirius::Periodic_function<double>*> func_map = {
        {"rho",  &gs.density().component(0)},
        {"magz", &gs.density().component(1)},
        {"magx", &gs.density().component(2)},
        {"magy", &gs.density().component(3)},
        {"veff", &gs.potential().component(0)},
        {"bz",   &gs.potential().component(1)},
        {"bx",   &gs.potential().component(2)},
        {"by",   &gs.potential().component(3)},
        {"vha",  &gs.potential().hartree_potential()}
    };

    sirius::Periodic_function<double>* f;
    try {
        f = func_map.at(label);
    } catch(...) {
        std::stringstream s;
        s << "wrong label: " << label;
        TERMINATE(s);
    }

    if (f_mt__) {
        f->set_mt_ptr(f_mt__);
    }
    if (f_rg__) {
        f->set_rg_ptr(f_rg__);
    }
}

/*
@api begin
sirius_set_periodic_function:
  doc: Get values of the periodic function.
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Handler of the DFT ground state object.
    label:
      type: string
      attr: in, required
      doc: Label of the function.
    f_rg:
      type: double
      attr: in, optional, dimension(*)
      doc: Real space values on the regular grid.
    f_rg_global:
      type: bool
      attr: in, optional
      doc: If true, real-space array is global.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.

@api end
*/
void sirius_set_periodic_function(void* const* handler__, char const* label__, double const* f_rg__,
        bool const* f_rg_global__, int* error_code__)
{
    call_sirius([&]()
    {
        auto& gs = get_gs(handler__);
        std::string label(label__);
        std::map<std::string, sirius::Periodic_function<double>*> func_map = {
            {"rho",  &gs.density().component(0)},
            {"magz", &gs.density().component(1)},
            {"magx", &gs.density().component(2)},
            {"magy", &gs.density().component(3)},
            {"veff", &gs.potential().component(0)},
            {"bz",   &gs.potential().component(1)},
            {"bx",   &gs.potential().component(2)},
            {"by",   &gs.potential().component(3)},
            {"vha",  &gs.potential().hartree_potential()}
        };
        if (func_map.count(label) == 0) {
            throw std::runtime_error("wrong label (" + label + ") for the periodic function");
        }
        if (f_rg__) {
            if (f_rg_global__ == nullptr) {
                throw std::runtime_error("missing bool argument `f_rg_global`");
            }
            bool is_local = !(*f_rg_global__);
            func_map[label]->copy_from(nullptr, f_rg__, is_local);
        }
    }, error_code__);
}

/*
@api begin
sirius_get_periodic_function:
  doc: Get values of the periodic function.
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Handler of the DFT ground state object.
    label:
      type: string
      attr: in, required
      doc: Label of the function.
    f_rg:
      type: double
      attr: out, optional, dimension(*)
      doc: Real space values on the regular grid.
    f_rg_global:
      type: bool
      attr: in, optional
      doc: If true, real-space array is global.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.

@api end
*/
void sirius_get_periodic_function(void* const* handler__, char const* label__, double* f_rg__,
        bool const* f_rg_global__, int* error_code__)
{
    call_sirius([&]()
    {
        auto& gs = get_gs(handler__);
        std::string label(label__);
        std::map<std::string, sirius::Periodic_function<double>*> func_map = {
            {"rho",  &gs.density().component(0)},
            {"magz", &gs.density().component(1)},
            {"magx", &gs.density().component(2)},
            {"magy", &gs.density().component(3)},
            {"veff", &gs.potential().component(0)},
            {"bz",   &gs.potential().component(1)},
            {"bx",   &gs.potential().component(2)},
            {"by",   &gs.potential().component(3)},
            {"vha",  &gs.potential().hartree_potential()}
        };
        if (func_map.count(label) == 0) {
            throw std::runtime_error("wrong label (" + label + ") for the periodic function");
        }
        if (f_rg__) {
            if (f_rg_global__ == nullptr) {
                throw std::runtime_error("missing bool argument `f_rg_global`");
            }
            bool is_local = !(*f_rg_global__);
            func_map[label]->copy_to(nullptr, f_rg__, is_local);
        }
    }, error_code__);
}

/*
@api begin
sirius_create_kset:
  doc: Create k-point set from the list of k-points.
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Simulation context handler.
    num_kpoints:
      type: int
      attr: in, required
      doc: Total number of k-points in the set.
    kpoints:
      type: double
      attr: in, required, dimension(3,num_kpoints)
      doc: List of k-points in lattice coordinates.
    kpoint_weights:
      type: double
      attr: in, required, dimension(num_kpoints)
      doc: Weights of k-points.
    init_kset:
      type: bool
      attr: in, required
      doc: If .true. k-set will be initialized.
    kset_handler:
      type: void*
      attr: out, required
      doc: Handler of the newly created k-point set.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void sirius_create_kset(void* const* handler__, int const* num_kpoints__, double* kpoints__,
                        double const* kpoint_weights__, bool const* init_kset__, void** kset_handler__,
                        int* error_code__)
{
    call_sirius([&]()
    {
        auto& sim_ctx = get_sim_ctx(handler__);

        mdarray<double, 2> kpoints(kpoints__, 3, *num_kpoints__);

        sirius::K_point_set* new_kset = new sirius::K_point_set(sim_ctx);
        new_kset->add_kpoints(kpoints, kpoint_weights__);
        if (*init_kset__) {
            new_kset->initialize();
        }
        *kset_handler__ = new utils::any_ptr(new_kset);
    }, error_code__);
}

/*
@api begin
sirius_create_kset_from_grid:
  doc: Create k-point set from a grid.
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Simulation context handler.
    k_grid:
      type: int
      attr: in, required, dimension(3)
      doc: dimensions of the k points grid.
    k_shift:
      type: int
      attr: in, required, dimension(3)
      doc: k point shifts.
    use_symmetry:
      type: bool
      attr: in, required
      doc: If .true. k-set will be generated using symmetries.
    kset_handler:
      type: void*
      attr: out, required
      doc: Handler of the newly created k-point set.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void sirius_create_kset_from_grid(void* const* handler__, int const* k_grid__, int const* k_shift__,
                                  bool const* use_symmetry, void** kset_handler__, int* error_code__)
{
    call_sirius([&]()
    {
        auto& sim_ctx = get_sim_ctx(handler__);
        std::vector<int> k_grid(3);
        std::vector<int> k_shift(3);

        k_grid[0] = k_grid__[0];
        k_grid[1] = k_grid__[1];
        k_grid[2] = k_grid__[2];

        k_shift[0] = k_shift__[0];
        k_shift[1] = k_shift__[1];
        k_shift[2] = k_shift__[2];

        sirius::K_point_set* new_kset = new sirius::K_point_set(sim_ctx, k_grid, k_shift, *use_symmetry);

        *kset_handler__ = new utils::any_ptr(new_kset);
    }, error_code__);
}

/*
@api begin
sirius_create_ground_state:
  doc: Create a ground state object.
  arguments:
    ks_handler:
      type: void*
      attr: in, required
      doc: Handler of the k-point set.
    gs_handler:
      type: void*
      attr: out, required
      doc: Handler of the newly created ground state object.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void sirius_create_ground_state(void* const* ks_handler__, void** gs_handler__, int* error_code__)
{
    call_sirius([&]()
    {
        auto& ks = get_ks(ks_handler__);

        *gs_handler__ = new utils::any_ptr(new sirius::DFT_ground_state(ks));
    }, error_code__);
}

/*
@api begin
sirius_initialize_kset:
  doc: Initialize k-point set.
  arguments:
    ks_handler:
      type: void*
      attr: in, required
      doc: K-point set handler.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void sirius_initialize_kset(void* const* ks_handler__, int* error_code__)
{
    call_sirius([&]()
    {
        auto& ks = get_ks(ks_handler__);
        ks.initialize();
    }, error_code__);
}

/*
@api begin
sirius_find_ground_state:
  doc: Find the ground state.
  arguments:
    gs_handler:
      type: void*
      attr: in, required
      doc: Handler of the ground state.
    density_tol:
      type: double
      attr: in, optional
      doc: Tolerance on RMS in density.
    energy_tol:
      type: double
      attr: in, optional
      doc: Tolerance in total energy difference.
    niter:
      type: int
      attr: in, optional
      doc: Maximum number of SCF iterations.
    save_state:
      type: bool
      attr: in, optional
      doc: boolean variable indicating if we want to save the ground state.
@api end
*/
void sirius_find_ground_state(void*  const* gs_handler__,
                              double const* density_tol__,
                              double const* energy_tol__,
                              int    const* niter__,
                              bool   const* save_state__)
{
    auto& gs = get_gs(gs_handler__);
    auto& ctx = gs.ctx();
    auto& inp = ctx.cfg().parameters();
    gs.initial_state();

    double rho_tol = inp.density_tol();
    if (density_tol__) {
        rho_tol = *density_tol__;
    }

    double etol = inp.energy_tol();
    if (energy_tol__) {
        etol = *energy_tol__;
    }

    int niter = inp.num_dft_iter();
    if (niter__) {
        niter = *niter__;
    }

    bool save{false};
    if (save_state__ != nullptr) {
        save = *save_state__;
    }

    auto result = gs.find(rho_tol, etol, ctx.iterative_solver_tolerance(), niter, save);
}

/*
@api begin
sirius_check_scf_density:
  doc: Check the self-consistent density
  arguments:
    gs_handler:
      type: void*
      attr: in, required
      doc: Handler of the ground state.
    error_code:
      type: int
      attr: out, optional
      doc: Error code
@api end
*/
void sirius_check_scf_density(void* const* gs_handler__, int* error_code__)
{
    call_sirius([&]()
    {
        auto& gs = get_gs(gs_handler__);
        gs.check_scf_density();
    }, error_code__);
}

/*
@api begin
sirius_find_ground_state_robust:
  doc: Find the ground state using the robust
  arguments:
    gs_handler:
      type: void*
      attr: in, required
      doc: Handler of the ground state.
    ks_handler:
      type: void*
      attr: in, required
      doc: Handler of the k-point set.
    scf_density_tol:
      type: double
      attr: in, optional
      doc: Tolerance on RMS in density.
    scf_energy_tol:
      type: double
      attr: in, optional
      doc: Tolerance in total energy difference.
    scf_ninit__:
      type: int
      attr: in, optional
      doc: Number of SCF iterations.
    temp__:
      type: double
      attr: in, optional
      doc: Temperature.
    tol__:
      type: double
      attr: in, optional
      doc: Tolerance.
    cg_restart__:
      type: int
      attr: in, optional
      doc: CG restart.
    kappa__:
      type: double
      attr: in, optional
      doc: Scalar preconditioner for pseudo Hamiltonian
@api end
*/
void sirius_find_ground_state_robust(void*  const* gs_handler__,
                                     void*  const* ks_handler__,
                                     double const* scf_density_tol__,
                                     double const* scf_energy_tol__,
                                     int    const* scf_ninit__,
                                     double const* temp__,
                                     double const* tol__
                                    )
{
#ifdef SIRIUS_NLCGLIB
    auto& gs = get_gs(gs_handler__);
    auto& ctx = gs.ctx();
    auto& inp = ctx.parameters_input();
    gs.initial_state();

    double rho_tol = inp.density_tol_;
    if (scf_density_tol__) {
        rho_tol = *scf_density_tol__;
    }

    double etol = inp.energy_tol_;
    if (scf_energy_tol__) {
        etol = *scf_energy_tol__;
    }

    int niter = inp.num_dft_iter_;
    if (scf_ninit__) {
        niter = *scf_ninit__;
    }

    // do a couple of SCF iterations to obtain a good initial guess
    bool save_state = false;
    auto result = gs.find(rho_tol, etol, ctx.iterative_solver_tolerance(), niter, save_state);

    // now call the direct solver
    // call nlcg solver
    auto& potential = gs.potential();
    auto& density = gs.density();

    auto& kset = get_ks(ks_handler__);

    auto nlcg_params  = ctx.nlcg_input();
    double temp       = nlcg_params.T_;
    double tol        = nlcg_params.tol_;
    double kappa      = nlcg_params.kappa_;
    double tau        = nlcg_params.tau_;
    int maxiter       = nlcg_params.maxiter_;
    int restart       = nlcg_params.restart_;
    std::string smear = nlcg_params.smearing_;
    std::string pu = nlcg_params.processing_unit_;

    nlcglib::smearing_type smearing;
    if (smear.compare("FD") == 0) {
        smearing = nlcglib::smearing_type::FERMI_DIRAC;
    } else if (smear.compare("GS") == 0) {
        smearing = nlcglib::smearing_type::GAUSSIAN_SPLINE;
    } else {
        throw std::runtime_error("invalid smearing type given");
    }

    sirius::Energy energy(kset, density, potential);
    if (is_device_memory(ctx.preferred_memory_t())) {
        if (pu.empty() || pu.compare("gpu") == 0) {
            nlcglib::nlcg_mvp2_device(energy, smearing, temp, tol, kappa, tau, maxiter, restart);
        } else if (pu.compare("cpu") == 0) {
            nlcglib::nlcg_mvp2_device_cpu(energy, smearing, temp, tol, kappa, tau, maxiter, restart);
        } else {
            throw std::runtime_error("invalid processing unit for nlcg given: " + pu);
        }
    } else {
        if (pu.empty() || pu.compare("gpu") == 0) {
            nlcglib::nlcg_mvp2_cpu(energy, smearing, temp, tol, kappa, tau, maxiter, restart);
        } else if (pu.compare("cpu") == 0) {
            nlcglib::nlcg_mvp2_cpu_device(energy, smearing, temp, tol, kappa, tau, maxiter, restart);
        } else {
            throw std::runtime_error("invalid processing unit for nlcg given: " + pu);
        }
    }
#else
    throw std::runtime_error("SIRIUS was not compiled with NLCG option.");
#endif

}


/*
@api begin
sirius_update_ground_state:
  doc: Update a ground state object after change of atomic coordinates or lattice vectors.
  arguments:
    gs_handler:
      type: void*
      attr: in, required
      doc: Ground-state handler.
@api end
*/
void sirius_update_ground_state(void** handler__)
{
    auto& gs = get_gs(handler__);
    gs.update();
}

/*
@api begin
sirius_add_atom_type:
  doc: Add new atom type to the unit cell.
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Simulation context handler.
    label:
      type: string
      attr: in, required
      doc: Atom type unique label.
    fname:
      type: string
      attr: in, optional
      doc: Species file name (in JSON format).
    zn:
      type: int
      attr: in, optional
      doc: Nucleus charge.
    symbol:
      type: string
      attr: in, optional
      doc: Atomic symbol.
    mass:
      type: double
      attr: in, optional
      doc: Atomic mass.
    spin_orbit:
      type: bool
      attr: in, optional
      doc: True if spin-orbit correction is enabled for this atom type.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void sirius_add_atom_type(void* const* handler__, char const* label__, char const* fname__, int const* zn__,
                          char const* symbol__, double const* mass__, bool const* spin_orbit__, int* error_code__)
{
    call_sirius([&]()
    {
        auto& sim_ctx = get_sim_ctx(handler__);

        std::string label = std::string(label__);
        std::string fname = (fname__ == nullptr) ? std::string("") : std::string(fname__);
        sim_ctx.unit_cell().add_atom_type(label, fname);

        auto& type = sim_ctx.unit_cell().atom_type(label);
        if (zn__ != nullptr) {
            type.set_zn(*zn__);
        }
        if (symbol__ != nullptr) {
            type.set_symbol(std::string(symbol__));
        }
        if (mass__ != nullptr) {
            type.set_mass(*mass__);
        }
        if (spin_orbit__ != nullptr) {
            type.spin_orbit_coupling(*spin_orbit__);
        }
        return 0;
    }, error_code__);
}

/*
@api begin
sirius_set_atom_type_radial_grid:
  doc: Set radial grid of the atom type.
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Simulation context handler.
    label:
      type: string
      attr: in, required
      doc: Atom type label.
    num_radial_points:
      type: int
      attr: in, required
      doc: Number of radial grid points.
    radial_points:
      type: double
      attr: in, required, dimension(num_radial_points)
      doc: List of radial grid points.
@api end
*/
void sirius_set_atom_type_radial_grid(void*  const* handler__,
                                      char   const* label__,
                                      int    const* num_radial_points__,
                                      double const* radial_points__)
{
    auto& sim_ctx = get_sim_ctx(handler__);

    auto& type = sim_ctx.unit_cell().atom_type(std::string(label__));
    type.set_radial_grid(*num_radial_points__, radial_points__);
}

/*
@api begin
sirius_set_atom_type_radial_grid_inf:
  doc: Set radial grid of the free atom (up to effectice infinity).
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Simulation context handler.
    label:
      type: string
      attr: in, required
      doc: Atom type label.
    num_radial_points:
      type: int
      attr: in, required
      doc: Number of radial grid points.
    radial_points:
      type: double
      attr: in, required, dimension(num_radial_points)
      doc: List of radial grid points.
@api end
*/
void sirius_set_atom_type_radial_grid_inf(void*  const* handler__,
                                          char   const* label__,
                                          int    const* num_radial_points__,
                                          double const* radial_points__)
{
    auto& sim_ctx = get_sim_ctx(handler__);

    auto& type = sim_ctx.unit_cell().atom_type(std::string(label__));
    type.set_free_atom_radial_grid(*num_radial_points__, radial_points__);
}

/*
@api begin
sirius_add_atom_type_radial_function:
  doc: Add one of the radial functions.
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Simulation context handler.
    atom_type:
      type: string
      attr: in, required
      doc: Label of the atom type.
    label:
      type: string
      attr: in, required
      doc: Label of the radial function.
    rf:
      type: double
      attr: in, required, dimension(num_points)
      doc: Array with radial function values.
    num_points:
      type: int
      attr: in, required
      doc: Length of radial function array.
    n:
      type: int
      attr: in, optional
      doc: Orbital quantum number.
    l:
      type: int
      attr: in, optional
      doc: angular momentum.
    idxrf1:
      type: int
      attr: in, optional
      doc: First index of radial function (for Q-operator).
    idxrf2:
      type: int
      attr: in, optional
      doc: Second index of radial function (for Q-operator).
    occ:
      type: double
      attr: in, optional
      doc: Occupancy of the wave-function.
@api end
*/
void sirius_add_atom_type_radial_function(void*  const* handler__,
                                          char   const* atom_type__,
                                          char   const* label__,
                                          double const* rf__,
                                          int    const* num_points__,
                                          int    const* n__,
                                          int    const* l__,
                                          int    const* idxrf1__,
                                          int    const* idxrf2__,
                                          double const* occ__)
{
    auto& sim_ctx = get_sim_ctx(handler__);

    auto& type = sim_ctx.unit_cell().atom_type(std::string(atom_type__));
    std::string label(label__);

    if (label == "beta") { /* beta-projectors */
        if (l__ == nullptr) {
            TERMINATE("orbital quantum number must be provided for beta-projector");
        }
        type.add_beta_radial_function(*l__, std::vector<double>(rf__, rf__ + *num_points__));
    } else if (label == "ps_atomic_wf") { /* pseudo-atomic wave functions */
        if (l__ == nullptr) {
            TERMINATE("orbital quantum number must be provided for pseudo-atomic radial function");
        }

        int n = (n__) ? *n__ : -1;
        double occ = (occ__) ? *occ__ : 0.0;
        type.add_ps_atomic_wf(n, sirius::experimental::angular_momentum(*l__),
                std::vector<double>(rf__, rf__ + *num_points__), occ);
    } else if (label == "ps_rho_core") {
        type.ps_core_charge_density(std::vector<double>(rf__, rf__ + *num_points__));
    } else if (label == "ps_rho_total") {
        type.ps_total_charge_density(std::vector<double>(rf__, rf__ + *num_points__));
    } else if (label == "vloc") {
        type.local_potential(std::vector<double>(rf__, rf__ + *num_points__));
    } else if (label == "q_aug") { /* augmentation charge */
        if (l__ == nullptr) {
            TERMINATE("orbital quantum number must be provided for augmentation charge radial function");
        }
        if (idxrf1__ == nullptr || idxrf2__ == nullptr) {
            TERMINATE("both radial-function indices must be provided for augmentation charge radial function");
        }
        type.add_q_radial_function(*idxrf1__, *idxrf2__, *l__, std::vector<double>(rf__, rf__ + *num_points__));
    } else if (label == "ae_paw_wf") {
        type.add_ae_paw_wf(std::vector<double>(rf__, rf__ + *num_points__));
    } else if (label == "ps_paw_wf") {
        type.add_ps_paw_wf(std::vector<double>(rf__, rf__ + *num_points__));
    } else if (label == "ae_paw_core") {
        type.paw_ae_core_charge_density(std::vector<double>(rf__, rf__ + *num_points__));
    } else if (label == "ae_rho") {
        type.free_atom_density(std::vector<double>(rf__, rf__ + *num_points__));
    } else {
        std::stringstream s;
        s << "wrong label of radial function: " << label__;
        TERMINATE(s);
    }
}

/*
@api begin
sirius_set_atom_type_hubbard:
  doc: Set the hubbard correction for the atomic type.
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Simulation context handler.
    label:
      type: string
      attr: in, required
      doc: Atom type label.
    l:
      type: int
      attr: in, required
      doc: Orbital quantum number.
    n:
      type: int
      attr: in, required
      doc: principal quantum number (s, p, d, f)
    occ:
      type: double
      attr: in, required
      doc: Atomic shell occupancy.
    U:
      type: double
      attr: in, required
      doc: Hubbard U parameter.
    J:
      type: double
      attr: in, required
      doc: Exchange J parameter for the full interaction treatment.
    alpha:
      type: double
      attr: in, required
      doc: J_alpha for the simple interaction treatment.
    beta:
      type: double
      attr: in, required
      doc: J_beta for the simple interaction treatment.
    J0:
      type: double
      attr: in, required
      doc: J0 for the simple interaction treatment.
@api end
*/
void sirius_set_atom_type_hubbard(void*  const* handler__,
                                  char   const* label__,
                                  int    const* l__,
                                  int    const* n__,
                                  double const* occ__,
                                  double const* U__,
                                  double const* J__,
                                  double const* alpha__,
                                  double const* beta__,
                                  double const* J0__)
{
    auto& sim_ctx = get_sim_ctx(handler__);
    auto& type = sim_ctx.unit_cell().atom_type(std::string(label__));
    type.hubbard_correction(true);
    type.add_hubbard_orbital(*n__, *l__, *occ__, *U__, J__[1], J__, *alpha__, *beta__, *J0__, std::vector<double>());
}

/*
@api begin
sirius_set_atom_type_dion:
  doc: Set ionic part of D-operator matrix.
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Simulation context handler.
    label:
      type: string
      attr: in, required
      doc: Atom type label.
    num_beta:
      type: int
      attr: in, required
      doc: Number of beta-projectors.
    dion:
      type: double
      attr: in, required, dimension(num_beta, num_beta)
      doc: Ionic part of D-operator matrix.
@api end
*/
void sirius_set_atom_type_dion(void*  const* handler__,
                               char   const* label__,
                               int    const* num_beta__,
                               double*       dion__)
{
    auto& sim_ctx = get_sim_ctx(handler__);
    auto& type = sim_ctx.unit_cell().atom_type(std::string(label__));
    matrix<double> dion(dion__, *num_beta__, *num_beta__);
    type.d_mtrx_ion(dion);
}

/*
@api begin
sirius_set_atom_type_paw:
  doc: Set PAW related data.
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Simulation context handler.
    label:
      type: string
      attr: in, required
      doc: Atom type label.
    core_energy:
      type: double
      attr: in, required
      doc: Core-electrons energy contribution.
    occupations:
      type: double
      attr: in, required, dimension(num_occ)
      doc: array of orbital occupancies
    num_occ:
      type: int
      attr: in, required
      doc: size of the occupations array
@api end
*/
void sirius_set_atom_type_paw(void*  const* handler__,
                              char   const* label__,
                              double const* core_energy__,
                              double const* occupations__,
                              int    const* num_occ__)
{
    auto& sim_ctx = get_sim_ctx(handler__);

    auto& type = sim_ctx.unit_cell().atom_type(std::string(label__));

    if (*num_occ__ != type.num_beta_radial_functions()) {
        TERMINATE("PAW error: different number of occupations and wave functions!");
    }

    /* we load PAW, so we set is_paw to true */
    type.is_paw(true);

    type.paw_core_energy(*core_energy__);

    type.paw_wf_occ(std::vector<double>(occupations__, occupations__ + type.num_beta_radial_functions()));
}

/*
@api begin
sirius_add_atom:
  doc: Add atom to the unit cell.
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Simulation context handler.
    label:
      type: string
      attr: in, required
      doc: Atom type label.
    position:
      type: double
      attr: in, required, dimension(3)
      doc: Atom position in lattice coordinates.
    vector_field:
      type: double
      attr: in, optional, dimension(3)
      doc: Starting magnetization.
@api end
*/
void sirius_add_atom(void*  const* handler__,
                     char   const* label__,
                     double const* position__,
                     double const* vector_field__)
{
    auto& sim_ctx = get_sim_ctx(handler__);
    if (vector_field__ != nullptr) {
        sim_ctx.unit_cell().add_atom(std::string(label__), std::vector<double>(position__, position__ + 3), vector_field__);
    } else {
        sim_ctx.unit_cell().add_atom(std::string(label__), std::vector<double>(position__, position__ + 3));
    }
}

/*
@api begin
sirius_set_atom_position:
  doc: Set new atomic position.
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Simulation context handler.
    ia:
      type: int
      attr: in, required
      doc: Index of atom; index starts form 1
    position:
      type: double
      attr: in, required, dimension(3)
      doc: Atom position in lattice coordinates.
@api end
*/
void sirius_set_atom_position(void*  const* handler__,
                              int    const* ia__,
                              double const* position__)
{
    auto& sim_ctx = get_sim_ctx(handler__);
    sim_ctx.unit_cell().atom(*ia__ - 1).set_position(std::vector<double>(position__, position__ + 3));
}

/*
@api begin
sirius_set_pw_coeffs:
  doc: Set plane-wave coefficients of a periodic function.
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Ground state handler.
    label:
      type: string
      attr: in, required
      doc: Label of the function.
    pw_coeffs:
      type: complex
      attr: in, required, dimension(*)
      doc: Local array of plane-wave coefficients.
    transform_to_rg:
      type: bool
      attr: in, optional
      doc: True if function has to be transformed to real-space grid.
    ngv:
      type: int
      attr: in, optional
      doc: Local number of G-vectors.
    gvl:
      type: int
      attr: in, optional, dimension(3, *)
      doc: List of G-vectors in lattice coordinates (Miller indices).
    comm:
      type: int
      attr: in, optional
      doc: MPI communicator used in distribution of G-vectors
@api end
*/
void sirius_set_pw_coeffs(void*                const* handler__,
                          char                 const* label__,
                          std::complex<double> const* pw_coeffs__,
                          bool                 const* transform_to_rg__,
                          int                  const* ngv__,
                          int*                        gvl__,
                          int                  const* comm__)
{
    PROFILE("sirius_api::sirius_set_pw_coeffs");

    auto& gs = get_gs(handler__);

    std::string label(label__);

    if (gs.ctx().full_potential()) {
        if (label == "veff") {
            gs.potential().set_veff_pw(pw_coeffs__);
        } else if (label == "rm_inv") {
            gs.potential().set_rm_inv_pw(pw_coeffs__);
        } else if (label == "rm2_inv") {
            gs.potential().set_rm2_inv_pw(pw_coeffs__);
        } else {
            TERMINATE("wrong label");
        }
    } else {
        assert(ngv__ != nullptr);
        assert(gvl__ != nullptr);
        assert(comm__ != nullptr);

        Communicator comm(MPI_Comm_f2c(*comm__));
        mdarray<int, 2> gvec(gvl__, 3, *ngv__);

        std::vector<double_complex> v(gs.ctx().gvec().num_gvec(), 0);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < *ngv__; i++) {
            vector3d<int> G(gvec(0, i), gvec(1, i), gvec(2, i));
            //auto gvc = gs.ctx().unit_cell().reciprocal_lattice_vectors() * vector3d<double>(G[0], G[1], G[2]);
            //if (gvc.length() > gs.ctx().pw_cutoff()) {
            //    continue;
            //}
            int ig = gs.ctx().gvec().index_by_gvec(G);
            if (ig >= 0) {
                v[ig] = pw_coeffs__[i];
            } else {
                if (gs.ctx().gamma_point()) {
                    ig = gs.ctx().gvec().index_by_gvec(G * (-1));
                    if (ig == -1) {
                        std::stringstream s;
                        auto gvc = dot(gs.ctx().unit_cell().reciprocal_lattice_vectors(),
                                       vector3d<double>(G[0], G[1], G[2]));
                        s << "wrong index of G-vector" << std::endl
                          << "input G-vector: " << G << " (length: " << gvc.length() << " [a.u.^-1])" << std::endl;
                        TERMINATE(s);
                    } else {
                        v[ig] = std::conj(pw_coeffs__[i]);
                    }
                }
            }
        }
        comm.allreduce(v.data(), gs.ctx().gvec().num_gvec());

        std::map<std::string, sirius::Smooth_periodic_function<double>*> func = {
            {"rho",   &gs.density().rho()},
            {"rhoc",  &gs.density().rho_pseudo_core()},
            {"magz",  &gs.density().magnetization(0)},
            {"magx",  &gs.density().magnetization(1)},
            {"magy",  &gs.density().magnetization(2)},
            {"veff",  &gs.potential().effective_potential()},
            {"bz",    &gs.potential().effective_magnetic_field(0)},
            {"bx",    &gs.potential().effective_magnetic_field(1)},
            {"by",    &gs.potential().effective_magnetic_field(2)},
            {"vloc",  &gs.potential().local_potential()},
            {"vxc",   &gs.potential().xc_potential()},
            {"dveff", &gs.potential().dveff()},
        };

        try {
            func.at(label)->scatter_f_pw(v);
            if (transform_to_rg__ && *transform_to_rg__) {
                func.at(label)->fft_transform(1);
            }
        } catch(...) {
            TERMINATE("wrong label");
        }
    }
}

/*
@api begin
sirius_get_pw_coeffs:
  doc: Get plane-wave coefficients of a periodic function.
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Ground state handler.
    label:
      type: string
      attr: in, required
      doc: Label of the function.
    pw_coeffs:
      type: complex
      attr: in, required, dimension(*)
      doc: Local array of plane-wave coefficients.
    ngv:
      type: int
      attr: in, optional
      doc: Local number of G-vectors.
    gvl:
      type: int
      attr: in, optional, dimension(3, *)
      doc: List of G-vectors in lattice coordinates (Miller indices).
    comm:
      type: int
      attr: in, optional
      doc: MPI communicator used in distribution of G-vectors
@api end
*/
void sirius_get_pw_coeffs(void*                const* handler__,
                          char                 const* label__,
                          std::complex<double>*       pw_coeffs__,
                          int                  const* ngv__,
                          int*                        gvl__,
                          int                  const* comm__)
{
    PROFILE("sirius_api::sirius_get_pw_coeffs");

    auto& gs = get_gs(handler__);

    std::string label(label__);
    if (gs.ctx().full_potential()) {
        STOP();
    } else {
        assert(ngv__ != NULL);
        assert(gvl__ != NULL);
        assert(comm__ != NULL);

        Communicator comm(MPI_Comm_f2c(*comm__));
        mdarray<int, 2> gvec(gvl__, 3, *ngv__);

        std::map<std::string, sirius::Smooth_periodic_function<double>*> func = {
            {"rho",  &gs.density().rho()},
            {"magz", &gs.density().magnetization(0)},
            {"magx", &gs.density().magnetization(1)},
            {"magy", &gs.density().magnetization(2)},
            {"veff", &gs.potential().effective_potential()},
            {"vloc", &gs.potential().local_potential()},
            {"rhoc", &gs.density().rho_pseudo_core()}
        };

        std::vector<double_complex> v;
        try {
            v = func.at(label)->gather_f_pw();
        } catch(...) {
            TERMINATE("wrong label");
        }

        for (int i = 0; i < *ngv__; i++) {
            vector3d<int> G(gvec(0, i), gvec(1, i), gvec(2, i));

            //auto gvc = gs.ctx().unit_cell().reciprocal_lattice_vectors() * vector3d<double>(G[0], G[1], G[2]);
            //if (gvc.length() > gs.ctx().pw_cutoff()) {
            //    pw_coeffs__[i] = 0;
            //    continue;
            //}

            bool is_inverse{false};
            int ig = gs.ctx().gvec().index_by_gvec(G);
            if (ig == -1 && gs.ctx().gvec().reduced()) {
                ig = gs.ctx().gvec().index_by_gvec(G * (-1));
                is_inverse = true;
            }
            if (ig == -1) {
                std::stringstream s;
                auto gvc = dot(gs.ctx().unit_cell().reciprocal_lattice_vectors(), vector3d<double>(G[0], G[1], G[2]));
                s << "wrong index of G-vector" << std::endl
                  << "input G-vector: " << G << " (length: " << gvc.length() << " [a.u.^-1])" << std::endl;
                TERMINATE(s);
            }
            if (is_inverse) {
                pw_coeffs__[i] = std::conj(v[ig]);
            } else {
                pw_coeffs__[i] = v[ig];
            }
        }
    }
}

/*
@api begin
sirius_get_pw_coeffs_real:
  doc: Get atom type contribution to plane-wave coefficients of a periodic function.
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Simulation context handler.
    atom_type:
      type: string
      attr: in, required
      doc: Label of the atom type.
    label:
      type: string
      attr: in, required
      doc: Label of the function.
    pw_coeffs:
      type: double
      attr: out, required, dimension(*)
      doc: Local array of plane-wave coefficients.
    ngv:
      type: int
      attr: in, optional
      doc: Local number of G-vectors.
    gvl:
      type: int
      attr: in, optional, dimension(3, *)
      doc: List of G-vectors in lattice coordinates (Miller indices).
    comm:
      type: int
      attr: in, optional
      doc: MPI communicator used in distribution of G-vectors
@api end
*/
void sirius_get_pw_coeffs_real(void* const* handler__,
                               char  const* atom_type__,
                               char  const* label__,
                               double*      pw_coeffs__,
                               int   const* ngv__,
                               int*         gvl__,
                               int   const* comm__)
{
    PROFILE("sirius_api::sirius_get_pw_coeffs_real");

    std::string label(label__);
    std::string atom_label(atom_type__);
    auto& sim_ctx = get_sim_ctx(handler__);

    int iat = sim_ctx.unit_cell().atom_type(atom_label).id();

    auto make_pw_coeffs = [&](std::function<double(double)> f)
    {
        mdarray<int, 2> gvec(gvl__, 3, *ngv__);

        double fourpi_omega = fourpi / sim_ctx.unit_cell().omega();
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < *ngv__; i++) {
            auto gc = dot(sim_ctx.unit_cell().reciprocal_lattice_vectors(),
                          vector3d<int>(gvec(0, i), gvec(1, i), gvec(2, i)));
            pw_coeffs__[i] = fourpi_omega * f(gc.length());
        }
    };

    if (label == "rhoc") {
        make_pw_coeffs([&](double g)
                       {
                           return sim_ctx.ps_core_ri().value<int>(iat, g);
                       });
    } else if (label == "rhoc_dg") {
        make_pw_coeffs([&](double g)
                       {
                           return sim_ctx.ps_core_ri_djl().value<int>(iat, g);
                       });
    } else if (label == "vloc") {
        make_pw_coeffs([&](double g)
                       {
                           return sim_ctx.vloc_ri().value(iat, g);
                       });
    } else if (label == "rho") {
        make_pw_coeffs([&](double g)
                       {
                           return sim_ctx.ps_rho_ri().value<int>(iat, g);
                       });
    } else {
        std::stringstream s;
        s << "[sirius_get_pw_coeffs_real] wrong label" << std::endl
          << "  label : " << label;
        TERMINATE(s);
    }
}

/*
@api begin
sirius_initialize_subspace:
  doc: Initialize the subspace of wave-functions.
  arguments:
    gs_handler:
      type: void*
      attr: in, required
      doc: Ground state handler.
    ks_handler:
      type: void*
      attr: in, required
      doc: K-point set handler.
@api end
*/
void sirius_initialize_subspace(void* const* gs_handler__,
                                void* const* ks_handler__)
{
    auto& gs = get_gs(gs_handler__);
    auto& ks = get_ks(ks_handler__);
    sirius::Hamiltonian0 H0(gs.potential());
    sirius::Band(ks.ctx()).initialize_subspace(ks, H0);
}

/*
@api begin
sirius_find_eigen_states:
  doc: Find eigen-states of the Hamiltonian
  arguments:
    gs_handler:
      type: void*
      attr: in, required
      doc: Ground state handler.
    ks_handler:
      type: void*
      attr: in, required
      doc: K-point set handler.
    precompute_pw:
      type: bool
      attr: in, optional
      doc: Generate plane-wave coefficients of the potential
    precompute_rf:
      type: bool
      attr: in, optional
      doc: Generate radial functions
    precompute_ri:
      type: bool
      attr: in, optional
      doc: Generate radial integrals
    iter_solver_tol:
      type: double
      attr: in, optional
      doc: Iterative solver tolerance.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void sirius_find_eigen_states(void* const* gs_handler__, void* const* ks_handler__, bool const* precompute_pw__,
        bool const* precompute_rf__, bool const* precompute_ri__, double const* iter_solver_tol__, int* error_code__)
{
    call_sirius([&]()
    {
        auto& gs = get_gs(gs_handler__);
        auto& ks = get_ks(ks_handler__);
        if (iter_solver_tol__ != nullptr) {
            ks.ctx().iterative_solver_tolerance(*iter_solver_tol__);
        }
        sirius::Hamiltonian0 H0(gs.potential());
        if (precompute_pw__ && *precompute_pw__) {
            H0.potential().generate_pw_coefs();
        }
        if ((precompute_rf__ && *precompute_rf__) || (precompute_ri__ && *precompute_ri__)) {
            H0.potential().update_atomic_potential();
        }
        if (precompute_rf__ && *precompute_rf__) {
            const_cast<sirius::Unit_cell&>(gs.ctx().unit_cell()).generate_radial_functions();
        }
        if (precompute_ri__ && *precompute_ri__) {
            const_cast<sirius::Unit_cell&>(gs.ctx().unit_cell()).generate_radial_integrals();
        }
        sirius::Band(ks.ctx()).solve(ks, H0, false);
    }, error_code__);
}

/*
@api begin
sirius_generate_d_operator_matrix:
  doc: Generate D-operator matrix.
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Ground state handler.
@api end
*/
void sirius_generate_d_operator_matrix(void* const* handler__)
{
    auto& gs = get_gs(handler__);
    gs.potential().generate_D_operator_matrix();
}

/*
@api begin
sirius_generate_initial_density:
  doc: Generate initial density.
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Ground state handler.
@api end
*/
void sirius_generate_initial_density(void* const* handler__)
{
    auto& gs = get_gs(handler__);
    gs.density().initial_density();
}

/*
@api begin
sirius_generate_effective_potential:
  doc: Generate effective potential and magnetic field.
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Ground state handler.
@api end
*/
void sirius_generate_effective_potential(void* const* handler__)
{
    auto& gs = get_gs(handler__);
    gs.potential().generate(gs.density(), gs.ctx().use_symmetry(), false);
}

/*
@api begin
sirius_generate_density:
  doc: Generate charge density and magnetization.
  arguments:
    gs_handler:
      type: void*
      attr: in, required
      doc: Ground state handler.
    add_core:
      type: bool
      attr: in, optional
      doc: Add core charge density in the muffin-tins.
    transform_to_rg:
      type: bool
      attr: in, optional
      doc: If true, density and magnetization are transformed to real-space grid.
@api end
*/
void sirius_generate_density(void* const* gs_handler__,
                             bool const*  add_core__,
                             bool const*  transform_to_rg__)
{
    auto& gs = get_gs(gs_handler__);
    bool add_core{false};
    if (add_core__ != nullptr) {
        add_core = *add_core__;
    }
    bool transform_to_rg{false};
    if (transform_to_rg__ != nullptr) {
        transform_to_rg = *transform_to_rg__;
    }

    gs.density().generate(gs.k_point_set(), add_core, gs.ctx().use_symmetry(), transform_to_rg);
}

/*
@api begin
sirius_set_band_occupancies:
  doc: Set band occupancies.
  arguments:
    ks_handler:
      type: void*
      attr: in, required
      doc: K-point set handler.
    ik:
      type: int
      attr: in, required
      doc: Global index of k-point.
    ispn:
      type: int
      attr: in, required
      doc: Spin component.
    band_occupancies:
      type: double
      attr: in, required
      doc: Array of band occupancies.
@api end
*/
void sirius_set_band_occupancies(void*  const* ks_handler__,
                                 int    const* ik__,
                                 int    const* ispn__,
                                 double const* band_occupancies__)
{
    auto& ks = get_ks(ks_handler__);
    int ik = *ik__ - 1;
    for (int i = 0; i < ks.ctx().num_bands(); i++) {
        ks[ik]->band_occupancy(i, *ispn__, band_occupancies__[i]);
    }
}

/*
@api begin
sirius_get_band_occupancies:
  doc: Set band occupancies.
  arguments:
    ks_handler:
      type: void*
      attr: in, required
      doc: K-point set handler.
    ik:
      type: int
      attr: in, required
      doc: Global index of k-point.
    ispn:
      type: int
      attr: in, required
      doc: Spin component.
    band_occupancies:
      type: double
      attr: out, required
      doc: Array of band occupancies.
@api end
*/
void
sirius_get_band_occupancies(void* const* ks_handler__, int const* ik__, int const* ispn__,
                            double* band_occupancies__)
{
    auto& ks = get_ks(ks_handler__);
    int ik   = *ik__ - 1;
    for (int i = 0; i < ks.ctx().num_bands(); i++) {
        band_occupancies__[i] = ks[ik]->band_occupancy(i, *ispn__);
    }
}

/*
@api begin
sirius_get_band_energies:
  doc: Get band energies.
  arguments:
    ks_handler:
      type: void*
      attr: in, required
      doc: K-point set handler.
    ik:
      type: int
      attr: in, required
      doc: Global index of k-point.
    ispn:
      type: int
      attr: in, required
      doc: Spin component.
    band_energies:
      type: double
      attr: out, required
      doc: Array of band energies.
@api end
*/
void sirius_get_band_energies(void*  const* ks_handler__,
                              int    const* ik__,
                              int    const* ispn__,
                              double*       band_energies__)
{
    auto& ks = get_ks(ks_handler__);
    int ik = *ik__ - 1;
    for (int i = 0; i < ks.ctx().num_bands(); i++) {
        band_energies__[i] = ks[ik]->band_energy(i, *ispn__);
    }
}

/*
@api begin
sirius_get_d_operator_matrix:
  doc: Get D-operator matrix
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Simulation context handler.
    ia:
      type: int
      attr: in, required
      doc: Global index of atom.
    ispn:
      type: int
      attr: in, required
      doc: Spin component.
    d_mtrx:
      type: double
      attr: out, required, dimension(ld, ld)
      doc: D-matrix.
    ld:
      type: int
      attr: in, required
      doc: Leading dimension of D-matrix.
@api end
*/
void sirius_get_d_operator_matrix(void* const* handler__,
                                  int   const* ia__,
                                  int   const* ispn__,
                                  double*      d_mtrx__,
                                  int   const* ld__)
{
    auto& sim_ctx = get_sim_ctx(handler__);

    mdarray<double, 2> d_mtrx(d_mtrx__, *ld__, *ld__);

    auto& atom = sim_ctx.unit_cell().atom(*ia__ - 1);
    auto idx_map = atomic_orbital_index_map_QE(atom.type());
    int nbf = atom.mt_basis_size();

    d_mtrx.zero();

    for (int xi1 = 0; xi1 < nbf; xi1++) {
        int p1 = phase_Rlm_QE(atom.type(), xi1);
        for (int xi2 = 0; xi2 < nbf; xi2++) {
            int p2 = phase_Rlm_QE(atom.type(), xi2);
            d_mtrx(idx_map[xi1], idx_map[xi2]) = atom.d_mtrx(xi1, xi2, *ispn__ - 1) * p1 * p2;
        }
    }
}

/*
@api begin
sirius_set_d_operator_matrix:
  doc: Set D-operator matrix
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Simulation context handler.
    ia:
      type: int
      attr: in, required
      doc: Global index of atom.
    ispn:
      type: int
      attr: in, required
      doc: Spin component.
    d_mtrx:
      type: double
      attr: out, required
      doc: D-matrix.
    ld:
      type: int
      attr: in, required
      doc: Leading dimension of D-matrix.
@api end
*/
void sirius_set_d_operator_matrix(void* const* handler__,
                                  int   const* ia__,
                                  int   const* ispn__,
                                  double*      d_mtrx__,
                                  int   const* ld__)
{
    auto& sim_ctx = get_sim_ctx(handler__);

    mdarray<double, 2> d_mtrx(d_mtrx__, *ld__, *ld__);

    auto& atom = sim_ctx.unit_cell().atom(*ia__ - 1);
    auto idx_map = atomic_orbital_index_map_QE(atom.type());
    int nbf = atom.mt_basis_size();

    for (int xi1 = 0; xi1 < nbf; xi1++) {
        int p1 = phase_Rlm_QE(atom.type(), xi1);
        for (int xi2 = 0; xi2 < nbf; xi2++) {
            int p2 = phase_Rlm_QE(atom.type(), xi2);
            atom.d_mtrx(xi1, xi2, *ispn__ - 1) = d_mtrx(idx_map[xi1], idx_map[xi2]) * p1 * p2;
        }
    }
}

/*
@api begin
sirius_set_q_operator_matrix:
  doc: Set Q-operator matrix
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Simulation context handler.
    label:
      type: string
      attr: in, required
      doc: Atom type label.
    q_mtrx:
      type: double
      attr: out, required, dimension(ld,ld)
      doc: Q-matrix.
    ld:
      type: int
      attr: in, required
      doc: Leading dimension of Q-matrix.
@api end
*/
void sirius_set_q_operator_matrix(void* const* handler__,
                                  char  const* label__,
                                  double*      q_mtrx__,
                                  int   const* ld__)
{
    auto& sim_ctx = get_sim_ctx(handler__);

    auto& type = sim_ctx.unit_cell().atom_type(std::string(label__));
    mdarray<double, 2> q_mtrx(q_mtrx__, *ld__, *ld__);
    mdarray<double, 2> qm(*ld__, *ld__);

    auto idx_map = atomic_orbital_index_map_QE(type);
    int nbf = type.mt_basis_size();

    for (int xi1 = 0; xi1 < nbf; xi1++) {
        int p1 = phase_Rlm_QE(type, xi1);
        for (int xi2 = 0; xi2 < nbf; xi2++) {
            int p2 = phase_Rlm_QE(type, xi2);
            qm(xi1, xi2) = q_mtrx(idx_map[xi1], idx_map[xi2]) * p1 * p2;
        }
    }
    sim_ctx.augmentation_op(type.id())->q_mtrx(qm);
}

/*
@api begin
sirius_get_q_operator_matrix:
  doc: Get Q-operator matrix
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Simulation context handler.
    label:
      type: string
      attr: in, required
      doc: Atom type label.
    q_mtrx:
      type: double
      attr: out, required, dimension(ld, ld)
      doc: Q-matrix.
    ld:
      type: int
      attr: in, required
      doc: Leading dimension of Q-matrix.
@api end
*/
void sirius_get_q_operator_matrix(void* const* handler__,
                                  char  const* label__,
                                  double*      q_mtrx__,
                                  int   const* ld__)
{
    auto& sim_ctx = get_sim_ctx(handler__);

    auto& type = sim_ctx.unit_cell().atom_type(std::string(label__));
    mdarray<double, 2> q_mtrx(q_mtrx__, *ld__, *ld__);

    auto idx_map = atomic_orbital_index_map_QE(type);
    int nbf = type.mt_basis_size();

    for (int xi1 = 0; xi1 < nbf; xi1++) {
        int p1 = phase_Rlm_QE(type, xi1);
        for (int xi2 = 0; xi2 < nbf; xi2++) {
            int p2 = phase_Rlm_QE(type, xi2);
            q_mtrx(idx_map[xi1], idx_map[xi2]) = sim_ctx.augmentation_op(type.id())->q_mtrx(xi1, xi2) * p1 * p2;
        }
    }
}

/*
@api begin
sirius_get_density_matrix:
  doc: Get all components of complex density matrix.
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: DFT ground state handler.
    ia:
      type: int
      attr: in, required
      doc: Global index of atom.
    dm:
      type: complex
      attr: out, required
      doc: Complex density matrix.
    ld:
      type: int
      attr: in, required
      doc: Leading dimension of the density matrix.
@api end
*/
void sirius_get_density_matrix(void*          const* handler__,
                               int            const* ia__,
                               std::complex<double>* dm__,
                               int            const* ld__) // TODO: this should be generalized for any phase factor convention
{
    auto& gs = get_gs(handler__);

    mdarray<double_complex, 3> dm(dm__, *ld__, *ld__, 3);

    auto& atom = gs.ctx().unit_cell().atom(*ia__ - 1);
    if (gs.ctx().full_potential()) {
        int nbf = std::min(atom.mt_basis_size(), *ld__);
        for (int icomp = 0; icomp < gs.ctx().num_mag_comp(); icomp++) {
            for (int i = 0; i < nbf; i++) {
                for (int j = 0; j < nbf; j++) {
                    dm(i, j, icomp) = gs.density().density_matrix()(i, j, icomp, *ia__ - 1);
                }
            }
        }
    } else {
        auto idx_map = atomic_orbital_index_map_QE(atom.type());
        int nbf = atom.mt_basis_size();
        assert(nbf <= *ld__);

        for (int icomp = 0; icomp < gs.ctx().num_mag_comp(); icomp++) {
            for (int i = 0; i < nbf; i++) {
                int p1 = phase_Rlm_QE(atom.type(), i);
                for (int j = 0; j < nbf; j++) {
                    int p2 = phase_Rlm_QE(atom.type(), j);
                    dm(idx_map[i], idx_map[j], icomp) = gs.density().density_matrix()(i, j, icomp, *ia__ - 1) * static_cast<double>(p1 * p2);
                }
            }
        }
    }
}

/*
@api begin
sirius_set_density_matrix:
  doc: Set all components of complex density matrix.
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: DFT ground state handler.
    ia:
      type: int
      attr: in, required
      doc: Global index of atom.
    dm:
      type: complex
      attr: out, required
      doc: Complex density matrix.
    ld:
      type: int
      attr: in, required
      doc: Leading dimension of the density matrix.
@api end
*/
void sirius_set_density_matrix(void*          const* handler__,
                               int            const* ia__,
                               std::complex<double>* dm__,
                               int            const* ld__)
{
    auto& gs = get_gs(handler__);

    mdarray<double_complex, 3> dm(dm__, *ld__, *ld__, 3);

    auto& atom = gs.ctx().unit_cell().atom(*ia__ - 1);
    auto idx_map = atomic_orbital_index_map_QE(atom.type());
    int nbf = atom.mt_basis_size();
    assert(nbf <= *ld__);

    for (int icomp = 0; icomp < gs.ctx().num_mag_comp(); icomp++) {
        for (int i = 0; i < nbf; i++) {
            int p1 = phase_Rlm_QE(atom.type(), i);
            for (int j = 0; j < nbf; j++) {
                int p2 = phase_Rlm_QE(atom.type(), j);
                gs.density().density_matrix()(i, j, icomp, *ia__ - 1) = dm(idx_map[i], idx_map[j], icomp) * static_cast<double>(p1 * p2);
            }
        }
    }
}

/*
@api begin
sirius_get_energy:
  doc: Get one of the total energy components.
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: DFT ground state handler.
    label:
      type: string
      attr: in, required
      doc: Label of the energy component to get.
    energy:
      type: double
      attr: out, required
      doc: Total energy component.
@api end
*/
void sirius_get_energy(void* const* handler__, char const* label__, double* energy__)
{
    auto& gs = get_gs(handler__);

    auto& kset = gs.k_point_set();
    auto& ctx = kset.ctx();
    auto& unit_cell = kset.unit_cell();
    auto& potential = gs.potential();
    auto& density = gs.density();

    std::string label(label__);

    std::map<std::string, std::function<double()>> func = {
        {"total",      [&](){ return sirius::total_energy(ctx, kset, density, potential, gs.ewald_energy()); }},
        {"evalsum",    [&](){ return sirius::eval_sum(unit_cell, kset); }},
        {"exc",        [&](){ return sirius::energy_exc(density, potential); }},
        {"vxc",        [&](){ return sirius::energy_vxc(density, potential); }},
        {"bxc",        [&](){ return sirius::energy_bxc(density, potential); }},
        {"veff",       [&](){ return sirius::energy_veff(density, potential); }},
        {"vloc",       [&](){ return sirius::energy_vloc(density, potential); }},
        {"vha",        [&](){ return sirius::energy_vha(potential); }},
        {"enuc",       [&](){ return sirius::energy_enuc(ctx, potential); }},
        {"kin",        [&](){ return sirius::energy_kin(ctx, kset, density, potential); }},
        {"one-el",     [&](){ return sirius::one_electron_energy(density, potential); }},
        {"descf",      [&](){ return gs.scf_energy(); }},
        {"demet",      [&](){ return kset.entropy_sum(); }},
        {"paw-one-el", [&](){ return potential.PAW_one_elec_energy(density); }},
        {"paw",        [&](){ return potential.PAW_total_energy(); }},
        {"fermi",      [&](){ return kset.energy_fermi(); }}
    };

    try {
        *energy__ = func.at(label)();
    } catch(...) {
        std::stringstream s;
        s << "[sirius_get_energy] wrong label: " << label;
        TERMINATE(s);
    }
}

/*
@api begin
sirius_get_forces:
  doc: Get one of the total force components.
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: DFT ground state handler.
    label:
      type: string
      attr: in, required
      doc: Label of the force component to get.
    forces:
      type: double
      attr: out, required, dimension(3, *)
      doc: Total force component for each atom.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void sirius_get_forces(void* const* handler__, char const* label__, double* forces__, int* error_code__)
{
    call_sirius([&]()
    {
        std::string label(label__);

        auto& gs = get_gs(handler__);

        auto get_forces = [&](mdarray<double, 2> const& sirius_forces__)
        {
            for (size_t i = 0; i < sirius_forces__.size(); i++){
                forces__[i] = sirius_forces__[i];
            }
        };

        auto& forces = gs.forces();

        std::map<std::string, mdarray<double, 2> const& (sirius::Force::*)(void)> func = {
            {"total",    &sirius::Force::calc_forces_total},
            {"vloc",     &sirius::Force::calc_forces_vloc},
            {"core",     &sirius::Force::calc_forces_core},
            {"ewald",    &sirius::Force::calc_forces_ewald},
            {"nonloc",   &sirius::Force::calc_forces_nonloc},
            {"us",       &sirius::Force::calc_forces_us},
            {"usnl",     &sirius::Force::calc_forces_usnl},
            {"scf_corr", &sirius::Force::calc_forces_scf_corr},
            {"hubbard",  &sirius::Force::calc_forces_hubbard},
            {"ibs",      &sirius::Force::calc_forces_ibs},
            {"hf",       &sirius::Force::calc_forces_hf},
            {"rho",      &sirius::Force::calc_forces_rho}
        };

        if (func.count(label) == 0) {
            throw std::runtime_error("wrong label (" + label + ") for the component of forces");
        }

        get_forces((forces.*func.at(label))());
    }, error_code__);
}

/*
@api begin
sirius_get_stress_tensor:
  doc: Get one of the stress tensor components.
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: DFT ground state handler.
    label:
      type: string
      attr: in, required
      doc: Label of the stress tensor component to get.
    stress_tensor:
      type: double
      attr: out, required, dimension(3, 3)
      doc: Component of the total stress tensor.
    error_code:
      type: int
      attr: out, optional
      doc: Error code..
@api end
*/
void sirius_get_stress_tensor(void* const* handler__, char const* label__, double* stress_tensor__, int* error_code__)
{
    call_sirius([&]()
    {
        std::string label(label__);

        auto& gs = get_gs(handler__);

        auto& stress_tensor = gs.stress();

        std::map<std::string, matrix3d<double> (sirius::Stress::*)(void)> func = {
            {"total",   &sirius::Stress::calc_stress_total},
            {"vloc",    &sirius::Stress::calc_stress_vloc},
            {"har",     &sirius::Stress::calc_stress_har},
            {"ewald",   &sirius::Stress::calc_stress_ewald},
            {"kin",     &sirius::Stress::calc_stress_kin},
            {"nonloc",  &sirius::Stress::calc_stress_nonloc},
            {"us",      &sirius::Stress::calc_stress_us},
            {"xc",      &sirius::Stress::calc_stress_xc},
            {"core",    &sirius::Stress::calc_stress_core},
            {"hubbard", &sirius::Stress::calc_stress_hubbard},
        };

        if (func.count(label) == 0) {
            throw std::runtime_error("wrong label (" + label + ") for the component of stress tensor");
        }

        matrix3d<double> s;

        s = ((stress_tensor.*func.at(label))());

        for (int mu = 0; mu < 3; mu++) {
            for (int nu = 0; nu < 3; nu++) {
                stress_tensor__[nu + mu * 3] = s(mu, nu);
            }
        }
    }, error_code__);
}

/*
@api begin
sirius_get_num_beta_projectors:
  doc: Get the number of beta-projectors for an atom type.
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Simulation context handler.
    label:
      type: string
      attr: in, required
      doc: Atom type label.
    num_bp:
      type: int
      attr: out, required
      doc: Number of beta projectors for each atom type.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void sirius_get_num_beta_projectors(void* const* handler__, char  const* label__, int* num_bp__, int* error_code__)
{
    call_sirius([&]()
    {
        auto& sim_ctx = get_sim_ctx(handler__);

        auto& type = sim_ctx.unit_cell().atom_type(std::string(label__));

        *num_bp__ = type.mt_basis_size();
    }, error_code__);
}

/*
@api begin
sirius_get_q_operator:
  doc: Get plane-wave coefficients of Q-operator
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Simulation context handler.
    label:
      type: string
      attr: in, required
      doc: Label of the atom type.
    xi1:
      type: int
      attr: in, required
      doc: First index of beta-projector atomic function.
    xi2:
      type: int
      attr: in, required
      doc: Second index of beta-projector atomic function.
    ngv:
      type: int
      attr: in, required
      doc: Number of G-vectors.
    gvl:
      type: int
      attr: in, required, dimension(3, ngv)
      doc: G-vectors in lattice coordinats.
    q_pw:
      type: complex
      attr: out, required, dimension(ngv)
      doc: Plane-wave coefficients of Q augmentation operator.
@api end
*/
void sirius_get_q_operator(void*          const* handler__,
                           char           const* label__,
                           int            const* xi1__,
                           int            const* xi2__,
                           int            const* ngv__,
                           int*                  gvl__,
                           std::complex<double>* q_pw__)
{
    PROFILE("sirius_api::sirius_get_q_operator");

    auto& sim_ctx = get_sim_ctx(handler__);

    auto& type = sim_ctx.unit_cell().atom_type(std::string(label__));

    mdarray<int, 2> gvl(gvl__, 3, *ngv__);

    auto qe_order = atomic_orbital_index_map_QE(type);

    int xi1{-1};
    int xi2{-1};

    /* find sirius indices, corresponding to QE indices */
    for (int xi = 0; xi < type.mt_basis_size(); xi++) {
        if (qe_order[xi] == (*xi1__ - 1)) {
            xi1 = xi;
        }
        if (qe_order[xi] == (*xi2__ - 1)) {
            xi2 = xi;
        }
    }

    auto p1 = phase_Rlm_QE(type, xi1);
    auto p2 = phase_Rlm_QE(type, xi2);

    int idx = utils::packed_index(xi1, xi2);

    std::vector<double_complex> q_pw(sim_ctx.gvec().num_gvec());
    for (int ig = 0; ig < sim_ctx.gvec().count(); ig++) {
        double x = sim_ctx.augmentation_op(type.id())->q_pw(idx, 2 * ig);
        double y = sim_ctx.augmentation_op(type.id())->q_pw(idx, 2 * ig + 1);
        q_pw[sim_ctx.gvec().offset() + ig] = double_complex(x, y) * static_cast<double>(p1 * p2);
    }
    sim_ctx.comm().allgather(q_pw.data(), sim_ctx.gvec().count(), sim_ctx.gvec().offset());

    for (int i = 0; i < *ngv__; i++) {
        vector3d<int> G(gvl(0, i), gvl(1, i), gvl(2, i));

        auto gvc = dot(sim_ctx.unit_cell().reciprocal_lattice_vectors(), vector3d<double>(G[0], G[1], G[2]));
        if (gvc.length() > sim_ctx.pw_cutoff()) {
            q_pw__[i] = 0;
            continue;
        }

        bool is_inverse{false};
        int ig = sim_ctx.gvec().index_by_gvec(G);
        if (ig == -1 && sim_ctx.gvec().reduced()) {
            ig = sim_ctx.gvec().index_by_gvec(G * (-1));
            is_inverse = true;
        }
        if (ig == -1) {
            std::stringstream s;
            auto gvc = dot(sim_ctx.unit_cell().reciprocal_lattice_vectors(), vector3d<double>(G[0], G[1], G[2]));
            s << "wrong index of G-vector" << std::endl
              << "input G-vector: " << G << " (length: " << gvc.length() << " [a.u.^-1])" << std::endl;
            TERMINATE(s);
        } else {
            if (is_inverse) {
                q_pw__[i] = std::conj(q_pw[ig]);
            } else {
                q_pw__[i] = q_pw[ig];
            }
        }
    }
}

/*
@api begin
sirius_get_wave_functions:
  doc: Get wave-functions.
  arguments:
    ks_handler:
      type: void*
      attr: in, required
      doc: K-point set handler.
    ik:
      type: int
      attr: in, required
      doc: Global index of k-point
    ispn:
      type: int
      attr: in, required
      doc: Spin index.
    npw:
      type: int
      attr: in, required
      doc: Local number of G+k vectors.
    gvec_k:
      type: int
      attr: in, required
      doc: List of G-vectors.
    evc:
      type: complex
      attr: out, required
      doc: Wave-functions.
    ld1:
      type: int
      attr: in, required
      doc: Leading dimension of evc array.
    ld2:
      type: int
      attr: in, required
      doc: Second dimension of evc array.
@api end
*/
void sirius_get_wave_functions(void*          const* ks_handler__,
                               int            const* ik__,
                               int            const* ispn__,
                               int            const* npw__,
                               int*                  gvec_k__,
                               std::complex<double>* evc__,
                               int            const* ld1__,
                               int            const* ld2__)
{
    PROFILE("sirius_api::sirius_get_wave_functions");

    auto& ks = get_ks(ks_handler__);

    auto& kset = ks;
    auto& sim_ctx = kset.ctx();

    int jk = *ik__ - 1;
    int jspn = *ispn__ - 1;

    int jrank{-1};
    if (jk >= 0) {
         /* find the rank where this k-point is stored */
        jrank = kset.spl_num_kpoints().local_rank(jk);
    }

    std::vector<int> rank_with_jk(kset.comm().size());
    kset.comm().allgather(&jrank, rank_with_jk.data(), 1, kset.comm().rank());

    std::vector<int> jk_of_rank(kset.comm().size());
    kset.comm().allgather(&jk, jk_of_rank.data(), 1, kset.comm().rank());

    std::vector<int> jspn_of_rank(kset.comm().size());
    kset.comm().allgather(&jspn, jspn_of_rank.data(), 1, kset.comm().rank());

    int my_rank = kset.comm().rank();

    std::vector<int> igmap;

    auto gvec_mapping = [&](Gvec const& gkvec)
    {
        std::vector<int> igm(*npw__, std::numeric_limits<int>::max());

        mdarray<int, 2> gvec_k(gvec_k__, 3, *npw__);

        for (int ig = 0; ig < *npw__; ig++) {
            /* G vector of host code */
            auto gvc = dot(kset.ctx().unit_cell().reciprocal_lattice_vectors(), 
                       (vector3d<double>(gvec_k(0, ig), gvec_k(1, ig), gvec_k(2, ig)) + gkvec.vk()));
            if (gvc.length() > kset.ctx().gk_cutoff()) {
                continue;
            }
            int ig1 = gkvec.index_by_gvec({gvec_k(0, ig), gvec_k(1, ig), gvec_k(2, ig)});
            /* vector is out of bounds */
            if (ig1 >= gkvec.num_gvec()) {
                continue;
            }
            /* index of G was not found */
            if (ig1 < 0) {
                /* try -G */
                ig1 = gkvec.index_by_gvec({-gvec_k(0, ig), -gvec_k(1, ig), -gvec_k(2, ig)});
                /* index of -G was not found */
                if (ig1 < 0) {
                    continue;
                } else {
                    /* this will tell co conjugate PW coefficients as we take them from -G index */
                    igm[ig] = -ig1;
                }
            } else {
                igm[ig] = ig1;
            }
        }
        return igm;
    };

    auto store_wf = [&](std::vector<double_complex>& wf_tmp, int i, int s, mdarray<double_complex, 3>& evc)
    {
        int ispn = s;
        if (sim_ctx.num_mag_dims() == 1) {
            ispn = 0;
        }
        for (int ig = 0; ig < *npw__; ig++) {
            int ig1 = igmap[ig];
            /* if this is a valid index */
            if (ig1 != std::numeric_limits<int>::max()) {
                double_complex z;
                if (ig1 < 0) {
                    z = std::conj(wf_tmp[-ig1]);
                } else {
                    z = wf_tmp[ig1];
                }
                evc(ig, ispn, i) = z;
            }
        }
    };

    for (int r = 0; r < kset.comm().size(); r++) {
        /* index of k-point we need to pass */
        int this_jk = jk_of_rank[r];

        if (this_jk >= 0) {
            auto gkvec = kset.send_recv_gkvec(this_jk, r);

            /* if this is a rank witch needs jk or a rank which stores jk */
            if (my_rank == r || my_rank == rank_with_jk[r]) {

                /* build G-vector mapping */
                if (my_rank == r) {
                    igmap = gvec_mapping(gkvec);
                }

                /* target array of wave-functions */
                mdarray<double_complex, 3> evc;
                if (my_rank == r) {
                    /* [npwx, npol, nbnd] array dimensions */
                    evc = mdarray<double_complex, 3>(evc__, *ld1__, *ld2__, sim_ctx.num_bands());
                    evc.zero();
                }

                std::unique_ptr<Gvec_partition> gvp;
                std::unique_ptr<Wave_functions> wf;

                if (my_rank == r) {
                    gvp = std::unique_ptr<Gvec_partition>(
                        new Gvec_partition(gkvec, sim_ctx.comm_fft_coarse(), sim_ctx.comm_band_ortho_fft_coarse()));
                    wf = std::unique_ptr<Wave_functions>(
                        new Wave_functions(*gvp, sim_ctx.num_bands(), sim_ctx.preferred_memory_t()));
                }

                int ispn0{0};
                int ispn1{1};
                /* fetch two components in non-collinear case, otherwise fetch only one component */
                if (sim_ctx.num_mag_dims() != 3) {
                    ispn0 = ispn1 = jspn_of_rank[r];
                }
                /* send wave-functions for each spin channel */
                for (int s = ispn0; s <= ispn1; s++) {
                    int tag = Communicator::get_tag(r, rank_with_jk[r]) + s;
                    Request req;

                    /* make a check of send-receive sizes */
                    if (true) {
                        int send_size;
                        int send_size1;
                        if (my_rank == rank_with_jk[r]) {
                            auto kp = kset[this_jk];
                            int gkvec_count = kp->gkvec().count();
                            send_size = gkvec_count * sim_ctx.num_bands();
                            req = kset.comm().isend(&send_size, 1, r, tag);
                        }
                        if (my_rank == r) {
                            int gkvec_count = gkvec.count();
                            kset.comm().recv(&send_size1, 1, rank_with_jk[r], tag);
                            if (send_size1 != gkvec_count * sim_ctx.num_bands()) {
                                std::stringstream s;
                                s << "wrong send-receive buffer sizes\n"
                                  << "     send size   : " << send_size1 << "\n"
                                  << "  receive size   : " << gkvec_count * sim_ctx.num_bands() << "\n"
                                  << " number of bands : " << sim_ctx.num_bands();
                                TERMINATE(s);
                            }
                        }
                        if (my_rank == rank_with_jk[r]) {
                            req.wait();
                        }
                    }

                    if (my_rank == rank_with_jk[r]) {
                        auto kp = kset[this_jk];
                        int gkvec_count = kp->gkvec().count();
                        /* send wave-functions */
                        req = kset.comm().isend(&kp->spinor_wave_functions().pw_coeffs(s).prime(0, 0),
                                gkvec_count * sim_ctx.num_bands(), r, tag);
                    }
                    if (my_rank == r) {
                        int gkvec_count = gkvec.count();
                        int gkvec_offset = gkvec.offset();
                        /* receive the array with wave-functions */
                        kset.comm().recv(&wf->pw_coeffs(0).prime(0, 0), gkvec_count * sim_ctx.num_bands(),
                                rank_with_jk[r], tag);
                        std::vector<double_complex> wf_tmp(gkvec.num_gvec());
                        /* store wave-functions */
                        for (int i = 0; i < sim_ctx.num_bands(); i++) {
                            /* gather full column of PW coefficients */
                            sim_ctx.comm_band().allgather(&wf->pw_coeffs(0).prime(0, i), wf_tmp.data(),
                                    gkvec_count, gkvec_offset);
                            store_wf(wf_tmp, i, s, evc);
                        }
                    }
                    if (my_rank == rank_with_jk[r]) {
                        req.wait();
                    }
                }
            }
        }
    }
}

/*
@api begin
sirius_set_hubbard_occupancies:
  doc: Set occupation matrix for LDA+U.
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Ground state handler.
    occ:
      type: complex
      attr: inout, required
      doc: Occupation matrix.
    ld:
      type: int
      attr: in, required
      doc: Leading dimensions of the occupation matrix.
@api end
*/
void sirius_set_hubbard_occupancies(void* const* handler__, std::complex<double>* occ__, int const *ld__)
{
    auto& gs = get_gs(handler__);
    gs.density().occupation_matrix().access("set", occ__, *ld__);
}

/*
@api begin
sirius_get_hubbard_occupancies:
  doc: Get occupation matrix for LDA+U.
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Ground state handler.
    occ:
      type: complex
      attr: inout, required
      doc: Occupation matrix.
    ld:
      type: int
      attr: in, required
      doc: Leading dimensions of the occupation matrix.
@api end
*/
void sirius_get_hubbard_occupancies(void* const* handler__, std::complex<double>* occ__, int const *ld__)
{
    auto& gs = get_gs(handler__);
    gs.density().occupation_matrix().access("get", occ__, *ld__);
}

/*
@api begin
sirius_set_hubbard_potential:
  doc: Set LDA+U potential matrix.
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Ground state handler.
    pot:
      type: complex
      attr: inout, required
      doc: Potential correction matrix.
    ld:
      type: int
      attr: in, required
      doc: Leading dimensions of the matrix.
@api end
*/
void sirius_set_hubbard_potential(void* const* handler__, std::complex<double>* pot__, int const *ld__)
{
    auto& gs = get_gs(handler__);
    gs.potential().U().access_hubbard_potential("set", pot__, *ld__);
}


/*
@api begin
sirius_get_hubbard_potential:
  doc: Set LDA+U potential matrix.
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Ground state handler.
    pot:
      type: complex
      attr: inout, required
      doc: Potential correction matrix.
    ld:
      type: int
      attr: in, required
      doc: Leading dimensions of the matrix.
@api end
*/
void sirius_get_hubbard_potential(void* const* handler__, std::complex<double>* pot__, int const *ld__)
{
    auto& gs = get_gs(handler__);
    gs.potential().U().access_hubbard_potential("get", pot__, *ld__);
}

/*
@api begin
sirius_add_atom_type_aw_descriptor:
  doc: Add descriptor of the augmented wave radial function.
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Simulation context handler.
    label:
      type: string
      attr: in, required
      doc: Atom type label.
    n:
      type: int
      attr: in, required
      doc: Principal quantum number.
    l:
      type: int
      attr: in, required
      doc: Orbital quantum number.
    enu:
      type: double
      attr: in, required
      doc: Linearization energy.
    dme:
      type: int
      attr: in, required
      doc: Order of energy derivative.
    auto_enu:
      type: bool
      attr: in, required
      doc: True if automatic search of linearization energy is allowed for this radial solution.
@api end
*/
void sirius_add_atom_type_aw_descriptor(void*  const* handler__,
                                        char   const* label__,
                                        int    const* n__,
                                        int    const* l__,
                                        double const* enu__,
                                        int    const* dme__,
                                        bool   const* auto_enu__)
{
    auto& sim_ctx = get_sim_ctx(handler__);
    auto& type = sim_ctx.unit_cell().atom_type(std::string(label__));
    type.add_aw_descriptor(*n__, *l__, *enu__, *dme__, *auto_enu__);
}

/*
@api begin
sirius_add_atom_type_lo_descriptor:
  doc: Add descriptor of the local orbital radial function.
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Simulation context handler.
    label:
      type: string
      attr: in, required
      doc: Atom type label.
    ilo:
      type: int
      attr: in, required
      doc: Index of the local orbital to which the descriptro is added.
    n:
      type: int
      attr: in, required
      doc: Principal quantum number.
    l:
      type: int
      attr: in, required
      doc: Orbital quantum number.
    enu:
      type: double
      attr: in, required
      doc: Linearization energy.
    dme:
      type: int
      attr: in, required
      doc: Order of energy derivative.
    auto_enu:
      type: bool
      attr: in, required
      doc: True if automatic search of linearization energy is allowed for this radial solution.
@api end
*/
void sirius_add_atom_type_lo_descriptor(void*  const* handler__,
                                        char   const* label__,
                                        int    const* ilo__,
                                        int    const* n__,
                                        int    const* l__,
                                        double const* enu__,
                                        int    const* dme__,
                                        bool   const* auto_enu__)
{
    auto& sim_ctx = get_sim_ctx(handler__);
    auto& type = sim_ctx.unit_cell().atom_type(std::string(label__));
    type.add_lo_descriptor(*ilo__ - 1, *n__, *l__, *enu__, *dme__, *auto_enu__);
}

/*
@api begin
sirius_set_atom_type_configuration:
  doc: Set configuration of atomic levels.
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Simulation context handler.
    label:
      type: string
      attr: in, required
      doc: Atom type label.
    n:
      type: int
      attr: in, required
      doc: Principal quantum number.
    l:
      type: int
      attr: in, required
      doc: Orbital quantum number.
    k:
      type: int
      attr: in, required
      doc: kappa (used in relativistic solver).
    occupancy:
      type: double
      attr: in, required
      doc: Level occupancy.
    core:
      type: bool
      attr: in, required
      doc: Tru if this is a core state.
@api end
*/
void sirius_set_atom_type_configuration(void*  const* handler__,
                                        char   const* label__,
                                        int    const* n__,
                                        int    const* l__,
                                        int    const* k__,
                                        double const* occupancy__,
                                        bool   const* core__)
{
    auto& sim_ctx = get_sim_ctx(handler__);
    auto& type = sim_ctx.unit_cell().atom_type(std::string(label__));
    type.set_configuration(*n__, *l__, *k__, *occupancy__, *core__);
}

/*
@api begin
sirius_generate_coulomb_potential:
  doc: Generate Coulomb potential by solving Poisson equation
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: DFT ground state handler
    vclmt:
      type: double
      attr: out, optional
      doc: Muffin-tin part of Coulomb potential
    lmmax:
      type: int
      attr: in, optional
      doc: Number of spherical harmonics 
    max_num_mt_points:
      type: int
      attr: in, optional
      doc: Maximum number of muffin-tin points
    num_atoms:
      type: int
      attr: in, optional
      doc: Number of atoms
    vha_el:
      type: double
      attr: out, optional
      doc: Electronic part of Hartree potential at each atom's origin.
    vclrg:
      type: double
      attr: out, optional
      doc: Interstitital part of the Coulomb potential
    num_rg_points:
      type: int
      attr: out, optional
      doc: Error code
@api end
*/
void sirius_generate_coulomb_potential(void* const* handler__, double* vclmt__, int const* lmmax__,
        int const* max_num_mt_points__, int const* num_atoms__, double* vha_el__, double* vclrg__,
        int const* num_rg_points__, int* error_code__)
{
    call_sirius([&]()
    {
        auto& gs = get_gs(handler__);

        gs.density().rho().fft_transform(-1);
        gs.potential().poisson(gs.density().rho());

        if (vclmt__) {
            if (!lmmax__) {
                throw std::runtime_error("missing 'lmmax' argument");
            }
            if (*lmmax__ != gs.potential().hartree_potential().angular_domain_size()) {
                throw std::runtime_error("wrong number of spherical harmonics");
            }
            if (!max_num_mt_points__) {
                throw std::runtime_error("missing 'max_num_mt_points' argument");
            }
            if (*max_num_mt_points__ != gs.ctx().unit_cell().max_num_mt_points()) {
                throw std::runtime_error("wrong maximum number of muffin-tin radial points");
            }
            if (!num_atoms__) {
                throw std::runtime_error("missing `num_atoms' argument");
            }
            if (*num_atoms__ != gs.ctx().unit_cell().num_atoms()) {
                throw std::runtime_error("wrong number of atoms");
            }
            gs.potential().hartree_potential().copy_to(vclmt__, nullptr, false);
            if (vha_el__) {
                for (int ia = 0; ia < gs.ctx().unit_cell().num_atoms(); ia++) {
                    vha_el__[ia] = gs.potential().vha_el(ia);
                }
            }
        }

        if (vclrg__) {
            if (!num_rg_points__) {
                throw std::runtime_error("missing 'num_rg_points' argument");
            }
            bool is_local_rg;
            if (gs.ctx().fft_grid().num_points() == *num_rg_points__) {
                is_local_rg = false;
            } else if (static_cast<int>(spfft_grid_size(gs.ctx().spfft())) == *num_rg_points__) {
                is_local_rg = true;
            } else {
                throw std::runtime_error("wrong number of regular grid points");
            }
            gs.potential().hartree_potential().copy_to(nullptr, vclrg__, is_local_rg);
        }
    }, error_code__);
}

/*
@api begin
sirius_generate_xc_potential:
  doc: Generate XC potential using LibXC
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Ground state handler
    is_local_rg:
      type: bool
      attr: in, required
      doc: true if regular grid pointer is local
    vxcmt:
      type: double
      attr: out, required
      doc: Muffin-tin part of potential
    vxcrg:
      type: double
      attr: out, required
      doc: Regular-grid part of potential
    bxcmt_x:
      type: double
      attr: out, optional
      doc: Muffin-tin part of effective magentic field (x-component)
    bxcmt_y:
      type: double
      attr: out, optional
      doc: Muffin-tin part of effective magentic field (y-component)
    bxcmt_z:
      type: double
      attr: out, optional
      doc: Muffin-tin part of effective magentic field (z-component)
    bxcrg_x:
      type: double
      attr: out, optional
      doc: Regular-grid part of effective magnetic field (x-component)
    bxcrg_y:
      type: double
      attr: out, optional
      doc: Regular-grid part of effective magnetic field (y-component)
    bxcrg_z:
      type: double
      attr: out, optional
      doc: Regular-grid part of effective magnetic field (z-component)
@api end
*/
void sirius_generate_xc_potential(void* const* handler__,
                                  bool  const* is_local_rg__,
                                  double*      vxcmt__,
                                  double*      vxcrg__,
                                  double*      bxcmt_x__,
                                  double*      bxcmt_y__,
                                  double*      bxcmt_z__,
                                  double*      bxcrg_x__,
                                  double*      bxcrg_y__,
                                  double*      bxcrg_z__)
{
    auto& gs = get_gs(handler__);
    gs.potential().xc(gs.density());

    gs.potential().xc_potential().copy_to(vxcmt__, vxcrg__, *is_local_rg__);

    if (gs.ctx().num_mag_dims() == 1) {
        /* z component */
        gs.potential().effective_magnetic_field(0).copy_to(bxcmt_z__, bxcrg_z__, *is_local_rg__);
    }
    if (gs.ctx().num_mag_dims() == 3) {
        /* z component */
        gs.potential().effective_magnetic_field(0).copy_to(bxcmt_z__, bxcrg_z__, *is_local_rg__);
        /* x component */
        gs.potential().effective_magnetic_field(1).copy_to(bxcmt_x__, bxcrg_x__, *is_local_rg__);
        /* y component */
        gs.potential().effective_magnetic_field(2).copy_to(bxcmt_y__, bxcrg_y__, *is_local_rg__);
    }
}

/*
@api begin
sirius_get_kpoint_inter_comm:
  doc: Get communicator which is used to split k-points
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Simulation context handler
    fcomm:
      type: int
      attr: out, required
      doc: Fortran communicator
@api end
*/
void sirius_get_kpoint_inter_comm(void * const* handler__,
                                  int*          fcomm__)
{
    auto& sim_ctx = get_sim_ctx(handler__);
    *fcomm__ = MPI_Comm_c2f(sim_ctx.comm_k().mpi_comm());
}

/*
@api begin
sirius_get_kpoint_inner_comm:
  doc: Get communicator which is used to parallise band problem
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Simulation context handler
    fcomm:
      type: int
      attr: out, required
      doc: Fortran communicator
@api end
*/
void sirius_get_kpoint_inner_comm(void * const* handler__,
                                  int*          fcomm__)
{
    auto& sim_ctx = get_sim_ctx(handler__);
    *fcomm__ = MPI_Comm_c2f(sim_ctx.comm_band().mpi_comm());
}

/*
@api begin
sirius_get_fft_comm:
  doc: Get communicator which is used to parallise FFT
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Simulation context handler
    fcomm:
      type: int
      attr: out, required
      doc: Fortran communicator
@api end
*/
void sirius_get_fft_comm(void * const* handler__,
                         int*          fcomm__)
{
    auto& sim_ctx = get_sim_ctx(handler__);
    *fcomm__ = MPI_Comm_c2f(sim_ctx.comm_fft().mpi_comm());
}

/*
@api begin
sirius_get_num_gvec:
  doc: Get total number of G-vectors
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Simulation context handler
    num_gvec:
      type: int
      attr: out, required
      doc: Total number of G-vectors
    error_code:
      type: int
      attr: out, optional
      doc: Error code
@api end
*/
void sirius_get_num_gvec(void* const* handler__, int* num_gvec__, int* error_code__)
{
    call_sirius([&]()
    {
        auto& sim_ctx = get_sim_ctx(handler__);
        *num_gvec__ = sim_ctx.gvec().num_gvec();
    }, error_code__);
}

// TODO: add dimensions keyword to the argument properties
/*
@api begin
sirius_get_gvec_arrays:
  doc: Get G-vector arrays.
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Simulation context handler
    gvec:
      type: int
      attr: in, optional, dimension(3, *)
      doc: G-vectors in lattice coordinates.
    gvec_cart:
      type: double
      attr: in, optional, dimension(3, *)
      doc: G-vectors in Cartesian coordinates.
    gvec_len:
      type: double
      attr: in, optional, dimension(*)
      doc: Length of G-vectors.
    index_by_gvec:
      type: int
      attr: in, optional
      doc: G-vector index by lattice coordinates.
@api end
*/
void sirius_get_gvec_arrays(void* const* handler__,
                            int*         gvec__,
                            double*      gvec_cart__,
                            double*      gvec_len__,
                            int*         index_by_gvec__)
{
    auto& sim_ctx = get_sim_ctx(handler__);

    if (gvec__ != nullptr) {
        mdarray<int, 2> gvec(gvec__, 3, sim_ctx.gvec().num_gvec());
        for (int ig = 0; ig < sim_ctx.gvec().num_gvec(); ig++) {
            auto gv = sim_ctx.gvec().gvec(ig);
            for (int x: {0, 1, 2}) {
                gvec(x, ig) = gv[x];
            }
        }
    }
    if (gvec_cart__ != nullptr) {
        mdarray<double, 2> gvec_cart(gvec_cart__, 3, sim_ctx.gvec().num_gvec());
        for (int ig = 0; ig < sim_ctx.gvec().num_gvec(); ig++) {
            auto gvc = sim_ctx.gvec().gvec_cart<index_domain_t::global>(ig);
            for (int x: {0, 1, 2}) {
                gvec_cart(x, ig) = gvc[x];
            }
        }
    }
    if (gvec_len__ != nullptr) {
        for (int ig = 0; ig < sim_ctx.gvec().num_gvec(); ig++) {
            gvec_len__[ig] = sim_ctx.gvec().gvec_len(ig);
        }
    }
    if (index_by_gvec__ != nullptr) {
        auto d0 = sim_ctx.fft_grid().limits(0);
        auto d1 = sim_ctx.fft_grid().limits(1);
        auto d2 = sim_ctx.fft_grid().limits(2);

        sddk::mdarray<int, 3> index_by_gvec(index_by_gvec__, d0, d1, d2);
        std::fill(index_by_gvec.at(memory_t::host), index_by_gvec.at(memory_t::host) + index_by_gvec.size(), -1);

        for (int ig = 0; ig < sim_ctx.gvec().num_gvec(); ig++) {
            auto G = sim_ctx.gvec().gvec(ig);
            index_by_gvec(G[0], G[1], G[2]) = ig + 1;
        }
    }
}

/*
@api begin
sirius_get_num_fft_grid_points:
  doc: Get local number of FFT grid points.
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Simulation context handler
    num_fft_grid_points:
      type: int
      attr: out, required
      doc: Local number of FFT grid points in the real-space mesh.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void sirius_get_num_fft_grid_points(void* const* handler__, int* num_fft_grid_points__, int* error_code__)
{
    call_sirius([&]()
    {
        auto& sim_ctx = get_sim_ctx(handler__);
        *num_fft_grid_points__ = sim_ctx.spfft().local_slice_size();
    }, error_code__);
}

/*
@api begin
sirius_get_fft_index:
  doc: Get mapping between G-vector index and FFT index
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Simulation context handler
    fft_index:
      type: int
      attr: out, required
      doc: Index inside FFT buffer
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void sirius_get_fft_index(void* const* handler__, int* fft_index__, int* error_code__)
{
    call_sirius([&]()
    {
        auto& sim_ctx = get_sim_ctx(handler__);
        for (int ig = 0; ig < sim_ctx.gvec().num_gvec(); ig++) {
            auto G = sim_ctx.gvec().gvec(ig);
            fft_index__[ig] = sim_ctx.fft_grid().index_by_freq(G[0], G[1], G[2]) + 1;
        }
    }, error_code__);
}

/*
@api begin
sirius_get_max_num_gkvec:
  doc: Get maximum number of G+k vectors across all k-points in the set
  arguments:
    ks_handler:
      type: void*
      attr: in, required
      doc: K-point set handler.
    max_num_gkvec:
      type: int
      attr: out, required
      doc: Maximum number of G+k vectors
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void sirius_get_max_num_gkvec(void* const* ks_handler__, int* max_num_gkvec__, int* error_code__)
{
    call_sirius([&]()
    {
        auto& ks = get_ks(ks_handler__);
        *max_num_gkvec__ = ks.max_num_gkvec();
    }, error_code__);
}

/*
@api begin
sirius_get_gkvec_arrays:
  doc: Get all G+k vector related arrays
  arguments:
    ks_handler:
      type: void*
      attr: in, required
      doc: K-point set handler.
    ik:
      type: int
      attr: in, required
      doc: Global index of k-point
    num_gkvec:
      type: int
      attr: out, required
      doc: Number of G+k vectors.
    gvec_index:
      type: int
      attr: out, required
      doc: Index of the G-vector part of G+k vector.
    gkvec:
      type: double
      attr: out, required
      doc: G+k vectors in fractional coordinates.
    gkvec_cart:
      type: double
      attr: out, required
      doc: G+k vectors in Cartesian coordinates.
    gkvec_len:
      type: double
      attr: out, required
      doc: Length of G+k vectors.
    gkvec_tp:
      type: double
      attr: out, required
      doc: Theta and Phi angles of G+k vectors.
@api end
*/
void sirius_get_gkvec_arrays(void* const* ks_handler__,
                             int*         ik__,
                             int*         num_gkvec__,
                             int*         gvec_index__,
                             double*      gkvec__,
                             double*      gkvec_cart__,
                             double*      gkvec_len,
                             double*      gkvec_tp__)
{

    auto& ks = get_ks(ks_handler__);

    auto kp = ks[*ik__ - 1];

    /* get rank that stores a given k-point */
    int rank = ks.spl_num_kpoints().local_rank(*ik__ - 1);

    auto& comm_k = ks.ctx().comm_k();

    if (rank == comm_k.rank()) {
        *num_gkvec__ = kp->num_gkvec();
        mdarray<double, 2> gkvec(gkvec__, 3, kp->num_gkvec());
        mdarray<double, 2> gkvec_cart(gkvec_cart__, 3, kp->num_gkvec());
        mdarray<double, 2> gkvec_tp(gkvec_tp__, 2, kp->num_gkvec());

        for (int igk = 0; igk < kp->num_gkvec(); igk++) {
            auto gkc = kp->gkvec().gkvec_cart<index_domain_t::global>(igk);
            auto G = kp->gkvec().gvec(igk);

            gvec_index__[igk] = ks.ctx().gvec().index_by_gvec(G) + 1; // Fortran counts from 1
            for (int x: {0, 1, 2}) {
                gkvec(x, igk) = kp->gkvec().gkvec(igk)[x];
                gkvec_cart(x, igk) = gkc[x];
            }
            auto rtp = sirius::SHT::spherical_coordinates(gkc);
            gkvec_len[igk] = rtp[0];
            gkvec_tp(0, igk) = rtp[1];
            gkvec_tp(1, igk) = rtp[2];
        }
    }
    comm_k.bcast(num_gkvec__,  1,                rank);
    comm_k.bcast(gvec_index__, *num_gkvec__,     rank);
    comm_k.bcast(gkvec__,      *num_gkvec__ * 3, rank);
    comm_k.bcast(gkvec_cart__, *num_gkvec__ * 3, rank);
    comm_k.bcast(gkvec_len,    *num_gkvec__,     rank);
    comm_k.bcast(gkvec_tp__,   *num_gkvec__ * 2, rank);
}

/*
@api begin
sirius_get_step_function:
  doc: Get the unit-step function.
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Simulation context handler
    cfunig:
      type: complex
      attr: out, required
      doc: Plane-wave coefficients of step function.
    cfunrg:
      type: double
      attr: out, required
      doc: Values of the step function on the regular grid.
@api end
*/
void sirius_get_step_function(void* const*          handler__,
                              std::complex<double>* cfunig__,
                              double*               cfunrg__)
{
    auto& sim_ctx = get_sim_ctx(handler__);
    for (int i = 0; i < sim_ctx.spfft().local_slice_size(); i++) {
        cfunrg__[i] = sim_ctx.theta(i);
    }
    for (int ig = 0; ig < sim_ctx.gvec().num_gvec(); ig++) {
        cfunig__[ig] = sim_ctx.theta_pw(ig);
    }
}

/*
@api begin
sirius_set_h_radial_integrals:
  doc: Set LAPW Hamiltonian radial integrals.
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Simulation context handler.
    ia:
      type: int
      attr: in, required
      doc: Index of atom.
    lmmax:
      type: int
      attr: in, required
      doc: Number of lm-component of the potential.
    val:
      type: double
      attr: in, required
      doc: Values of the radial integrals.
    l1:
      type: int
      attr: in, optional
      doc: 1st index of orbital quantum number.
    o1:
      type: int
      attr: in, optional
      doc: 1st index of radial function order for l1.
    ilo1:
      type: int
      attr: in, optional
      doc: 1st index or local orbital.
    l2:
      type: int
      attr: in, optional
      doc: 2nd index of orbital quantum number.
    o2:
      type: int
      attr: in, optional
      doc: 2nd index of radial function order for l2.
    ilo2:
      type: int
      attr: in, optional
      doc: 2nd index or local orbital.
@api end
*/
void sirius_set_h_radial_integrals(void* const* handler__,
                                   int*         ia__,
                                   int*         lmmax__,
                                   double*      val__,
                                   int*         l1__,
                                   int*         o1__,
                                   int*         ilo1__,
                                   int*         l2__,
                                   int*         o2__,
                                   int*         ilo2__)
{
    auto& sim_ctx = get_sim_ctx(handler__);
    int ia = *ia__ - 1;
    int idxrf1{-1};
    int idxrf2{-1};
    if ((l1__ != nullptr && o1__ != nullptr && ilo1__ != nullptr) ||
        (l2__ != nullptr && o2__ != nullptr && ilo2__ != nullptr)) {
        TERMINATE("wrong combination of radial function indices");
    }
    if (l1__ != nullptr && o1__ != nullptr) {
        idxrf1 = sim_ctx.unit_cell().atom(ia).type().indexr_by_l_order(*l1__, *o1__ - 1);
    } else if (ilo1__ != nullptr) {
        idxrf1 = sim_ctx.unit_cell().atom(ia).type().indexr_by_idxlo(*ilo1__ - 1);
    } else {
        TERMINATE("1st radial function index is not valid");
    }

    if (l2__ != nullptr && o2__ != nullptr) {
        idxrf2 = sim_ctx.unit_cell().atom(ia).type().indexr_by_l_order(*l2__, *o2__ - 1);
    } else if (ilo2__ != nullptr) {
        idxrf2 = sim_ctx.unit_cell().atom(ia).type().indexr_by_idxlo(*ilo2__ - 1);
    } else {
        TERMINATE("2nd radial function index is not valid");
    }

    for (int lm = 0; lm < *lmmax__; lm++) {
        sim_ctx.unit_cell().atom(ia).h_radial_integrals(idxrf1, idxrf2)[lm] = val__[lm];
    }
}

/*
@api begin
sirius_set_o_radial_integral:
  doc: Set LAPW overlap radial integral.
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Simulation context handler.
    ia:
      type: int
      attr: in, required
      doc: Index of atom.
    val:
      type: double
      attr: in, required
      doc: Value of the radial integral.
    l:
      type: int
      attr: in, required
      doc: Orbital quantum number.
    o1:
      type: int
      attr: in, optional
      doc: 1st index of radial function order.
    ilo1:
      type: int
      attr: in, optional
      doc: 1st index or local orbital.
    o2:
      type: int
      attr: in, optional
      doc: 2nd index of radial function order.
    ilo2:
      type: int
      attr: in, optional
      doc: 2nd index or local orbital.
@api end
*/
void sirius_set_o_radial_integral(void* const* handler__,
                                  int*         ia__,
                                  double*      val__,
                                  int*         l__,
                                  int*         o1__,
                                  int*         ilo1__,
                                  int*         o2__,
                                  int*         ilo2__)
{

    auto& sim_ctx = get_sim_ctx(handler__);
    int ia = *ia__ - 1;
    if ((o1__ != nullptr && ilo1__ != nullptr) ||
        (o2__ != nullptr && ilo2__ != nullptr)) {
        TERMINATE("wrong combination of radial function indices");
    }

    if (o1__ != nullptr && ilo2__ != nullptr) {
        int idxrf2 = sim_ctx.unit_cell().atom(ia).type().indexr_by_idxlo(*ilo2__ - 1);
        int order2 = sim_ctx.unit_cell().atom(ia).type().indexr(idxrf2).order;
        sim_ctx.unit_cell().atom(ia).symmetry_class().set_o_radial_integral(*l__, *o1__ - 1, order2, *val__);
    }

    if (o2__ != nullptr && ilo1__ != nullptr) {
        int idxrf1 = sim_ctx.unit_cell().atom(ia).type().indexr_by_idxlo(*ilo1__ - 1);
        int order1 = sim_ctx.unit_cell().atom(ia).type().indexr(idxrf1).order;
        sim_ctx.unit_cell().atom(ia).symmetry_class().set_o_radial_integral(*l__, order1, *o2__ - 1, *val__);
    }

    if (ilo1__ != nullptr && ilo2__ != nullptr) {
        int idxrf1 = sim_ctx.unit_cell().atom(ia).type().indexr_by_idxlo(*ilo1__ - 1);
        int order1 = sim_ctx.unit_cell().atom(ia).type().indexr(idxrf1).order;
        int idxrf2 = sim_ctx.unit_cell().atom(ia).type().indexr_by_idxlo(*ilo2__ - 1);
        int order2 = sim_ctx.unit_cell().atom(ia).type().indexr(idxrf2).order;
        sim_ctx.unit_cell().atom(ia).symmetry_class().set_o_radial_integral(*l__, order1, order2, *val__);
    }
}

/*
@api begin
sirius_set_o1_radial_integral:
  doc: Set a correction to LAPW overlap radial integral.
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Simulation context handler.
    ia:
      type: int
      attr: in, required
      doc: Index of atom.
    val:
      type: double
      attr: in, required
      doc: Value of the radial integral.
    l1:
      type: int
      attr: in, optional
      doc: 1st index of orbital quantum number.
    o1:
      type: int
      attr: in, optional
      doc: 1st index of radial function order for l1.
    ilo1:
      type: int
      attr: in, optional
      doc: 1st index or local orbital.
    l2:
      type: int
      attr: in, optional
      doc: 2nd index of orbital quantum number.
    o2:
      type: int
      attr: in, optional
      doc: 2nd index of radial function order for l2.
    ilo2:
      type: int
      attr: in, optional
      doc: 2nd index or local orbital.
@api end
*/
void sirius_set_o1_radial_integral(void* const* handler__,
                                   int*         ia__,
                                   double*      val__,
                                   int*         l1__,
                                   int*         o1__,
                                   int*         ilo1__,
                                   int*         l2__,
                                   int*         o2__,
                                   int*         ilo2__)
{
    auto& sim_ctx = get_sim_ctx(handler__);
    int ia = *ia__ - 1;
    int idxrf1{-1};
    int idxrf2{-1};
    if ((l1__ != nullptr && o1__ != nullptr && ilo1__ != nullptr) ||
        (l2__ != nullptr && o2__ != nullptr && ilo2__ != nullptr)) {
        TERMINATE("wrong combination of radial function indices");
    }
    if (l1__ != nullptr && o1__ != nullptr) {
        idxrf1 = sim_ctx.unit_cell().atom(ia).type().indexr_by_l_order(*l1__, *o1__ - 1);
    } else if (ilo1__ != nullptr) {
        idxrf1 = sim_ctx.unit_cell().atom(ia).type().indexr_by_idxlo(*ilo1__ - 1);
    } else {
        TERMINATE("1st radial function index is not valid");
    }

    if (l2__ != nullptr && o2__ != nullptr) {
        idxrf2 = sim_ctx.unit_cell().atom(ia).type().indexr_by_l_order(*l2__, *o2__ - 1);
    } else if (ilo2__ != nullptr) {
        idxrf2 = sim_ctx.unit_cell().atom(ia).type().indexr_by_idxlo(*ilo2__ - 1);
    } else {
        TERMINATE("2nd radial function index is not valid");
    }
    sim_ctx.unit_cell().atom(ia).symmetry_class().set_o1_radial_integral(idxrf1, idxrf2, *val__);
}

/*
@api begin
sirius_set_radial_function:
  doc: Set LAPW radial functions
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Simulation context handler.
    ia:
      type: int
      attr: in, required
      doc: Index of atom.
    deriv_order:
      type: int
      attr: in, required
      doc: Radial derivative order.
    f:
      type: double
      attr: in, required
      doc: Values of the radial function.
    l:
      type: int
      attr: in, optional
      doc: Orbital quantum number.
    o:
      type: int
      attr: in, optional
      doc: Order of radial function for l.
    ilo:
      type: int
      attr: in, optional
      doc: Local orbital index.
@api end
*/
void sirius_set_radial_function(void*  const* handler__,
                                int    const* ia__,
                                int    const* deriv_order__,
                                double const* f__,
                                int    const* l__,
                                int    const* o__,
                                int    const* ilo__)
{
    auto& sim_ctx = get_sim_ctx(handler__);

    int ia = *ia__ - 1;

    auto& atom = sim_ctx.unit_cell().atom(ia);

    if (l__ != nullptr && o__ != nullptr && ilo__ != nullptr) {
        TERMINATE("wrong combination of radial function indices");
    }
    if (!(*deriv_order__ == 0 || *deriv_order__ == 1)) {
        TERMINATE("wrond radial derivative order");
    }

    int idxrf{-1};
    if (l__ != nullptr && o__ != nullptr) {
        idxrf = atom.type().indexr_by_l_order(*l__, *o__ - 1);
    } else if (ilo__ != nullptr) {
        idxrf = atom.type().indexr_by_idxlo(*ilo__ - 1);
    } else {
        TERMINATE("radial function index is not valid");
    }

    if (*deriv_order__ == 0) {
        for (int ir = 0; ir < atom.num_mt_points(); ir++) {
            atom.symmetry_class().radial_function(ir, idxrf) = f__[ir];
        }
    } else {
        for (int ir = 0; ir < atom.num_mt_points(); ir++) {
            atom.symmetry_class().radial_function_derivative(ir, idxrf) = f__[ir] * atom.type().radial_grid()[ir];
        }
    }
    if (l__ != nullptr && o__ != nullptr) {
        int n = atom.num_mt_points();
        atom.symmetry_class().aw_surface_deriv(*l__, *o__ - 1, *deriv_order__, f__[n - 1]);
    }
}

/*
@api begin
sirius_get_radial_function:
  doc: Get LAPW radial functions
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Simulation context handler.
    ia:
      type: int
      attr: in, required
      doc: Index of atom.
    deriv_order:
      type: int
      attr: in, required
      doc: Radial derivative order.
    f:
      type: double
      attr: out, required
      doc: Values of the radial function.
    l:
      type: int
      attr: in, optional
      doc: Orbital quantum number.
    o:
      type: int
      attr: in, optional
      doc: Order of radial function for l.
    ilo:
      type: int
      attr: in, optional
      doc: Local orbital index.
@api end
*/
void sirius_get_radial_function(void* const* handler__,
                                int   const* ia__,
                                int   const* deriv_order__,
                                double*      f__,
                                int   const* l__,
                                int   const* o__,
                                int   const* ilo__)
{
    auto& sim_ctx = get_sim_ctx(handler__);

    int ia = *ia__ - 1;

    auto& atom = sim_ctx.unit_cell().atom(ia);

    if (l__ != nullptr && o__ != nullptr && ilo__ != nullptr) {
        TERMINATE("wrong combination of radial function indices");
    }
    if (!(*deriv_order__ == 0 || *deriv_order__ == 1)) {
        TERMINATE("wrond radial derivative order");
    }

    int idxrf{-1};
    if (l__ != nullptr && o__ != nullptr) {
        idxrf = atom.type().indexr_by_l_order(*l__, *o__ - 1);
    } else if (ilo__ != nullptr) {
        idxrf = atom.type().indexr_by_idxlo(*ilo__ - 1);
    } else {
        TERMINATE("radial function index is not valid");
    }

    if (*deriv_order__ == 0) {
        for (int ir = 0; ir < atom.num_mt_points(); ir++) {
            f__[ir] = atom.symmetry_class().radial_function(ir, idxrf);
        }
    } else {
        for (int ir = 0; ir < atom.num_mt_points(); ir++) {
            f__[ir] = atom.symmetry_class().radial_function_derivative(ir, idxrf) / atom.type().radial_grid()[ir];
        }
    }
}

/*
@api begin
sirius_set_equivalent_atoms:
  doc: Set equivalent atoms.
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Simulation context handler.
    equivalent_atoms:
      type: int
      attr: in, required, dimension(*)
      doc: Array with equivalent atom IDs.
@api end
*/
void sirius_set_equivalent_atoms(void* const* handler__,
                                 int*         equivalent_atoms__)
{
    auto& sim_ctx = get_sim_ctx(handler__);
    sim_ctx.unit_cell().set_equivalent_atoms(equivalent_atoms__);
}

/*
@api begin
sirius_update_atomic_potential:
  doc: Set the new spherical potential.
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Ground state handler.
@api end
*/
void sirius_update_atomic_potential(void* const* handler__)
{
    auto& gs = get_gs(handler__);
    gs.potential().update_atomic_potential();
}


/*
@api begin
sirius_option_get_length:
  doc: return the number of options in a given section
  arguments:
    section:
      type: string
      attr: in, required
      doc: name of the seciton
    length:
      type: int
      attr: out, required
      doc: number of options contained in  the section
@api end
*/
void sirius_option_get_length(char const* section__, int *length__)
{
    auto const& parser = sirius::get_options_dictionary();

    auto section = std::string(section__);
    std::transform(section.begin(), section.end(), section.begin(), ::tolower);

    *length__ = parser[section].size();
}

/*
@api begin
sirius_option_get_name_and_type:
  doc: Return the name and a type of an option from its index.
  arguments:
    section:
      type: string
      attr: in, required
      doc: Name of the section.
    elem:
      type: int
      attr: in, required
      doc: Index of the option.
    key_name:
      type: string
      attr: out, required
      doc: Name of the option.
    type:
      type: int
      attr: out, required
      doc: Type of the option (real, integer, boolean, string).
@api end
*/

void sirius_option_get_name_and_type(char const* section__, int const* elem__, char* key_name__, int* type__)
{
    const json &dict = sirius::get_options_dictionary();

    auto section = std::string(section__);
    std::transform(section.begin(), section.end(), section.begin(), ::tolower);

    int elem = 0;
    *type__ = -1;
    for (auto& el : dict[section].items()) {
        if (elem == *elem__) {
            if (!dict[section][el.key()].count("default_value")) {
                std::cout << "key : " << el.key() << "\n the default_value key is missing" << std::endl;
                exit(0);
            }
            if (dict[section][el.key()]["default_value"].is_array()) {
                *type__ = 10;
                if (dict[section][el.key()]["default_value"][0].is_number_integer()) {
                    *type__ += 1;
                }
                if (dict[section][el.key()]["default_value"][0].is_number_float()) {
                    *type__ += 2;
                }
                if (dict[section][el.key()]["default_value"][0].is_boolean()) {
                    *type__ += 3;
                }
                if (dict[section][el.key()]["default_value"][0].is_string()) {
                    *type__ += 4;
                }
            } else {
                if (dict[section][el.key()]["default_value"].is_number_integer()) {
                    *type__ = 1;
                }
                if (dict[section][el.key()]["default_value"].is_number_float()) {
                    *type__ = 2;
                }
                if (dict[section][el.key()]["default_value"].is_boolean()) {
                    *type__ = 3;
                }
                if (dict[section][el.key()]["default_value"].is_string()) {
                    *type__ = 4;
                }
            }
            std::memcpy(key_name__, el.key().c_str(), el.key().size() + 1);
        }
        elem++;
    }
}

/*
@api begin
sirius_option_get_description_usage:
  doc: return the description and usage of a given option
  arguments:
    section:
      type: string
      attr: in, required
      doc: name of the section
    name:
      type: string
      attr: in, required
      doc: name of the option
    desc:
      type: string
      attr: out, required
      doc: description of the option
    usage:
      type: string
      attr: out, required
      doc: how to use the option
@api end
*/
void sirius_option_get_description_usage(char const* section__, char const* name__, char* desc__, char* usage__)
{
    const json &parser =  sirius::get_options_dictionary();

    auto section = std::string(section__);
    std::transform(section.begin(), section.end(), section.begin(), ::tolower);

    auto name = std::string(name__);

    if (parser[section][name].count("description")) {
        auto description = parser[section][name].value("description", "");
        std::copy(description.begin(), description.end(), desc__);
    }
    if (parser[section][name].count("usage")) {
        auto usage = parser[section][name].value("usage", "");
        std::copy(usage.begin(), usage.end(), usage__);
    }
}

/*
@api begin
sirius_option_get_int:
  doc: return the default value of the option
  arguments:
    section:
      type: string
      attr: in, required
      doc: name of the section of interest
    name:
      type: string
      attr: in, required
      doc: name of the element
    default_value:
      type: int
      attr: out, required
      doc: table containing the default values (if vector)
    length:
      type: int
      attr: out, required
      doc: length of the table containing the default values
@api end
*/

void sirius_option_get_int(char const* section__, char const* name__, int *default_value__, int *length__)
{
    auto const &parser = sirius::get_options_dictionary();

    auto section = std::string(section__);
    std::transform(section.begin(), section.end(), section.begin(), ::tolower);

    auto name = std::string(name__);

    if (!parser[section][name].count("default_value")) {
        std::cout << "default value is missing" << std::endl;
    }
    if (parser[section][name]["default_value"].is_array()) {
        std::vector<int> v = parser[section][name]["default_value"].get<std::vector<int>>();
        *length__ = v.size();
        std::memcpy(default_value__, &v[0], v.size() * sizeof(int));
    }  else {
        *default_value__ = parser[section][name].value("default_value", -1);
    }
}

/*
@api begin
sirius_option_get_double:
  doc: return the default value of the option
  arguments:
    section:
      type: string
      attr: in, required
      doc: name of the section of interest
    name:
      type: string
      attr: in, required
      doc: name of the element
    default_value:
      type: double
      attr: out, required
      doc: table containing the default values (if vector)
    length:
      type: int
      attr: out, required
      doc: length of the table containing the default values
@api end
*/
void sirius_option_get_double(char * section, char * name, double *default_value, int *length)
{
    const json &parser =  sirius::get_options_dictionary();

    // ugly as hell but fortran is a piece of ....
    for ( char *p = section; *p; p++) *p = tolower(*p);
    // ugly as hell but fortran is a piece of ....
    ///for ( char *p = name; *p; p++) *p = tolower(*p);

    if (!parser[section][name].count("default_value"))
        std::cout << "default value is mossing" << std::endl;
    if (parser[section][name]["default_value"].is_array()) {
        std::vector<double> v = parser[section][name]["default_value"].get<std::vector<double>>();
        *length = v.size();
        memcpy(default_value, &v[0], v.size() * sizeof(double));
    }  else {
        *default_value = parser[section][name].value("default_value", 0.0);
    }
}

/*
@api begin
sirius_option_get_logical:
  doc: return the default value of the option
  arguments:
    section:
      type: string
      attr: in, required
      doc: name of the section
    name:
      type: string
      attr: in, required
      doc: name of the element
    default_value:
      type: bool
      attr: out, required
      doc: table containing the default values
    length:
      type: int
      attr: out, required
      doc: length of the table containing the default values
@api end
*/
void sirius_option_get_logical(char * section, char * name, bool *default_value, int *length)
{
    const json &parser =  sirius::get_options_dictionary();

    // ugly as hell but fortran is a piece of ....
    for ( char *p = section; *p; p++) *p = tolower(*p);
    // ugly as hell but fortran is a piece of ....
    ///for ( char *p = name; *p; p++) *p = tolower(*p);
    if (!parser[section][name].count("default_value"))
        std::cout << "default value is mossing" << std::endl;
    if (parser[section][name]["default_value"].is_array()) {
        std::vector<bool> v = parser[section][name]["default_value"].get<std::vector<bool>>();
        *length = v.size();
        std::copy(v.begin(), v.end(), default_value);
    }  else {
        *default_value = parser[section][name].value("default_value", false);
    }
}

/*
@api begin
sirius_option_get_string:
  doc: return the default value of the option
  arguments:
    section:
      type: string
      attr: in, required
      doc: name of the section
    name:
      type: string
      attr: in, required
      doc: name of the option
    default_value:
      type: string
      attr: out, required
      doc: table containing the string
@api end
*/
void sirius_option_get_string(char* section, char * name, char *default_value)
{
    const json &parser =  sirius::get_options_dictionary();

    // ugly as hell but fortran is a piece of ....
    for ( char *p = section; *p; p++) *p = tolower(*p);
    for ( char *p = name; *p; p++) *p = tolower(*p);

    if (!parser[section][name].count("default_value"))
        std::cout << "default value is mossing" << std::endl;
    std::string value = parser[section][name].value("default_value", "");
    std::copy(value.begin(), value.end() - 1, default_value);
}

/*
@api begin
sirius_option_get_number_of_possible_values:
  doc: return the number of possible values for a string option
  arguments:
    section:
      type: string
      attr: in, required
      doc: name of the section
    name:
      type: string
      attr: in, required
      doc: name of the option
    num_:
      type: int
      attr: out, required
      doc: number of elements
@api end
*/
void sirius_option_get_number_of_possible_values(char* section, char * name, int *num_)
{
    const json &parser =  sirius::get_options_dictionary();

    // ugly as hell but fortran is a piece of ....
    for ( char *p = section; *p; p++) *p = tolower(*p);
    for ( char *p = name; *p; p++) *p = tolower(*p);

    if (parser[section][name].count("possible_values")) {
        auto tmp =  parser[section][name]["possible_values"].get<std::vector<std::string>>();
        *num_ = tmp.size();
        return;
    }
    *num_ = -1;
}

/*
@api begin
sirius_option_string_get_value:
  doc: return the possible values for a string parameter
  arguments:
    section:
      type: string
      attr: in, required
      doc: name of the section
    name:
      type: string
      attr: in, required
      doc: name of the option
    elem_:
      type: int
      attr: in, required
      doc: index of the value
    value_n:
      type: string
      attr: out, required
      doc: string containing the value
@api end
*/
void sirius_option_string_get_value(char* section, char * name, int *elem_, char *value_n)
{
    const json &parser = sirius::get_options_dictionary();

    // ugly as hell but fortran is a piece of ....
    for ( char *p = section; *p; p++) *p = tolower(*p);
    for ( char *p = name; *p; p++) *p = tolower(*p);

    // for string I do not consider a table of several strings to be returned. I
    // need to specialize however the possible values that the string can have
    if (parser[section][name].count("possible_values")) {
        auto tmp = parser[section][name]["possible_values"].get<std::vector<std::string>>();
        std::memcpy(value_n, tmp[*elem_].c_str(), tmp[*elem_].size() + 1);
    }
}

/*
@api begin
sirius_option_get_section_name:
  doc: return the name of a given section
  arguments:
    elem:
      type: int
      attr: in, required
      doc: index of the section
    section_name:
      type: string
      attr: out, required
      doc: name of the section
@api end
*/
void sirius_option_get_section_name(int *elem, char *section_name)
{
    const json &dict = sirius::get_options_dictionary();
    int elem_ = 0;

    for (auto& el : dict.items())
    {
        if (elem_ == *elem) {
            std::memcpy(section_name, el.key().c_str(), el.key().size() + 1);
            break;
        }
        elem_++;
    }
}

/*
@api begin
sirius_option_get_number_of_sections:
  doc: return the number of sections
  arguments:
    length:
      type: int
      attr: out, required
      doc: number of sections
@api end
*/
void sirius_option_get_number_of_sections(int *length)
{
    const json &parser =  sirius::get_options_dictionary();
    *length = parser.size();
}


/*
@api begin
sirius_option_set_int:
  doc: set the value of the option name in a  (internal) json dictionary
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Simulation context handler.
    section:
      type: string
      attr: in, required
      doc: string containing the options in json format
    name:
      type: string
      attr: in, required
      doc: name of the element to pick
    default_values:
      type: int
      attr: in, required
      doc: table containing the values
    length:
      type: int
      attr: in, required
      doc: length of the table containing the values
@api end
*/
void sirius_option_set_int(void* const* handler__, char*section, char *name, int *default_values, int *length)
{
    auto& sim_ctx = get_sim_ctx(handler__);
    /* dictionary describing all the possible options */
    json const& parser = sirius::get_options_dictionary();

    /* dictionary containing the values of the options for the simulations */
    json& conf_dict = sim_ctx.get_runtime_options_dictionary();

    /* lower case for section and options */
    for ( char *p = section; *p; p++) *p = tolower(*p);
    for ( char *p = name; *p; p++) *p = tolower(*p);

    if (parser[section].count(name)) {
        // check that the option exists
        if (*length > 1) {
            // we are dealing with a vector
            std::vector<int> v(*length);
            for (int s = 0; s < *length; s++)
                v[s] = default_values[s];
            conf_dict[section][name] = v;
        } else {
            conf_dict[section][name] = *default_values;
        }
    }
}

/*
@api begin
sirius_option_set_double:
  doc: set the value of the option name in a (internal) json dictionary
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Simulation context handler.
    section:
      type: string
      attr: in, required
      doc: name of the section
    name:
      type: string
      attr: in, required
      doc: name of the element to pick
    default_values:
      type: double
      attr: in, required
      doc: table containing the values
    length:
      type: int
      attr: in, required
      doc: length of the table containing the values
@api end
*/
void sirius_option_set_double(void* const* handler__, char*section, char *name, double *default_values, int *length)
{
    auto& sim_ctx = get_sim_ctx(handler__);
    const json &parser =  sirius::get_options_dictionary();
    json &conf_dict = sim_ctx.get_runtime_options_dictionary();
    // ugly as hell but fortran is a piece of ....
    for ( char *p = section; *p; p++) *p = tolower(*p);
    // ugly as hell but fortran is a piece of ....
    for ( char *p = name; *p; p++) *p = tolower(*p);

    if (parser[section].count(name)) {
        // check that the option exists
        if (*length > 1) {
            // we are dealing with a vector
            std::vector<double> v(*length);
            for (int s = 0; s < *length; s++)
                v[s] = default_values[s];
            conf_dict[section][name] = v;
        } else {
            conf_dict[section][name] = *default_values;
        }
    }
}

/*
@api begin
sirius_option_set_logical:
  doc: set the value of the option name in a  (internal) json dictionary
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Simulation context handler.
    section:
      type: string
      attr: in, required
      doc: name of the section
    name:
      type: string
      attr: in, required
      doc: name of the element to pick
    default_values:
      type: int
      attr: in, required
      doc: table containing the values
    length:
      type: int
      attr: in, required
      doc: length of the table containing the values
@api end
*/
void sirius_option_set_logical(void* const* handler__, char*section, char *name, int *default_values, int *length)
{
    auto& sim_ctx = get_sim_ctx(handler__);
    // the first one is static
    const json &parser =  sirius::get_options_dictionary();
    json &conf_dict = sim_ctx.get_runtime_options_dictionary();
    // ugly as hell but fortran is a piece of ....
    for ( char *p = section; *p; p++) *p = tolower(*p);
    // ugly as hell but fortran is a piece of ....
    for ( char *p = name; *p; p++) *p = tolower(*p);
    std::cout << section << " " << name << std::endl;

    if (parser[section].count(name)) {
        // check that the option exists
        if (*length > 1) {
            // we are dealing with a vector
            std::vector<bool> v(*length);
            for (int s = 0; s < *length; s++) {
                v[s] = (default_values[s] == 1);
            }
            conf_dict[section][name] = v;
        } else {
            conf_dict[section][name] = (*default_values == 1);
        }
    }
}

/*
@api begin
sirius_option_set_string:
  doc: set the value of the option name in a  (internal) json dictionary
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Simulation context handler.
    section:
      type: string
      attr: in, required
      doc: name of the section
    name:
      type: string
      attr: in, required
      doc: name of the element to pick
    default_values:
      type: string
      attr: in, required
      doc: table containing the values
@api end
*/
void sirius_option_set_string(void* const* handler__, char * section, char * name, char *default_values)
{
    auto& sim_ctx = get_sim_ctx(handler__);
    // the first one is static
    const json &parser =  sirius::get_options_dictionary();
    json &conf_dict = sim_ctx.get_runtime_options_dictionary();
    // ugly as hell but fortran is a piece of ....
    for ( char *p = section; *p; p++) *p = tolower(*p);
    // ugly as hell but fortran is a piece of ....
    for ( char *p = name; *p; p++) *p = tolower(*p);
    if (parser[section].count(name)) {
        if (!default_values) {
            std::cout << "option not set up because the string null" << std::endl;
            return;
        }
        // ugly as hell but fortran is a piece of ....
        for ( char *p = default_values; *p; p++) {
            *p = tolower(*p);
        }
        std::string st = default_values;
        conf_dict[section][name] = st;
    }
}

/*
@api begin
sirius_option_add_string_to:
  doc: add a string value to the option in the json dictionary
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Simulation context handler.
    section:
      type: string
      attr: in, required
      doc: name of the section
    name:
      type: string
      attr: in, required
      doc: name of the element to pick
    default_values:
      type: string
      attr: in, required
      doc: string to be added
@api end
*/
void sirius_option_add_string_to(void* const* handler__, char * section, char * name, char *default_values)
{
    auto& sim_ctx = get_sim_ctx(handler__);
    // the first one is static
    const json &parser =  sirius::get_options_dictionary();
    json &conf_dict = sim_ctx.get_runtime_options_dictionary();
    // ugly as hell but fortran is a piece of ....
    for ( char *p = section; *p; p++) *p = tolower(*p);
    // ugly as hell but fortran is a piece of ....
    for ( char *p = name; *p; p++) *p = tolower(*p);
    if (parser[section].count(name)) {
        if (!default_values) {
            std::cout << "option not set up because the string null" << std::endl;
            return;
        }
        if (conf_dict[section].count(name)) {
            auto v = conf_dict[section][name].get<std::vector<std::string>>();
            v.push_back(default_values);
            conf_dict[section][name] = v;
        } else {
            std::vector<std::string> st;
            st.clear();
            st.push_back(default_values);
            conf_dict[section][name] = st;
        }
    }
}

/*
@api begin
sirius_dump_runtime_setup:
  doc: Dump the runtime setup in a file.
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Simulation context handler.
    filename:
      type: string
      attr: in, required
      doc: String containing the name of the file.
    error_code:
      type: int
      attr: out, optional
      doc: Error code
@api end
*/
void sirius_dump_runtime_setup(void* const* handler__, char* filename__, int* error_code__)
{
    call_sirius([&]()
    {
        auto& sim_ctx = get_sim_ctx(handler__);
        std::ofstream fi(filename__, std::ofstream::out | std::ofstream::trunc);
        auto conf_dict = sim_ctx.serialize(); //get_runtime_options_dictionary();
        fi << conf_dict.dump(4);
    }, error_code__);
}

/*
@api begin
sirius_get_fv_eigen_vectors:
  doc: Get the first-variational eigen vectors
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: K-point set handler
    ik:
      type: int
      attr: in, required
      doc: Global index of the k-point
    fv_evec:
      type: complex
      attr: out, required
      doc: Output first-variational eigenvector array
    ld:
      type: int
      attr: in, required
      doc: Leading dimension of fv_evec
    num_fv_states:
      type: int
      attr: in, required
      doc: Number of first-variational states
@api end
*/
void sirius_get_fv_eigen_vectors(void*          const* handler__,
                                 int            const* ik__,
                                 std::complex<double>* fv_evec__,
                                 int            const* ld__,
                                 int            const* num_fv_states__)
{
    auto& ks = get_ks(handler__);
    mdarray<std::complex<double>, 2> fv_evec(fv_evec__, *ld__, *num_fv_states__);
    int ik = *ik__ - 1;
    ks[ik]->get_fv_eigen_vectors(fv_evec);
}

/*
@api begin
sirius_get_fv_eigen_values:
  doc: Get the first-variational eigen values
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: K-point set handler
    ik:
      type: int
      attr: in, required
      doc: Global index of the k-point
    fv_eval:
      type: double
      attr: out, required
      doc: Output first-variational eigenvector array
    num_fv_states:
      type: int
      attr: in, required
      doc: Number of first-variational states
@api end
*/
void sirius_get_fv_eigen_values(void*          const* handler__,
                                int            const* ik__,
                                double              * fv_eval__,
                                int            const* num_fv_states__)
{
    auto& ks = get_ks(handler__);
    if (*num_fv_states__ != ks.ctx().num_fv_states()) {
        TERMINATE("wrong number of first-variational states");
    }
    int ik = *ik__ - 1;
    for (int i = 0; i < *num_fv_states__; i++) {
        fv_eval__[i] = ks[ik]->fv_eigen_value(i);
    }
}

/*
@api begin
sirius_get_sv_eigen_vectors:
  doc: Get the second-variational eigen vectors
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: K-point set handler
    ik:
      type: int
      attr: in, required
      doc: Global index of the k-point
    sv_evec:
      type: complex
      attr: out, required
      doc: Output second-variational eigenvector array
    num_bands:
      type: int
      attr: in, required
      doc: Number of second-variational bands.
@api end
*/
void sirius_get_sv_eigen_vectors(void*          const* handler__,
                                 int            const* ik__,
                                 std::complex<double>* sv_evec__,
                                 int            const* num_bands__)
{
    auto& ks = get_ks(handler__);
    mdarray<std::complex<double>, 2> sv_evec(sv_evec__, *num_bands__, *num_bands__);
    int ik = *ik__ - 1;
    ks[ik]->get_sv_eigen_vectors(sv_evec);
}

/*
@api begin
sirius_set_rg_values:
  doc: Set the values of the function on the regular grid.
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: DFT ground state handler.
    label:
      type: string
      attr: in, required
      doc: Label of the function.
    grid_dims:
      type: int
      attr: in, required
      doc: Dimensions of the FFT grid.
    local_box_origin:
      type: int
      attr: in, required
      doc: Coordinates of the local box origin for each MPI rank
    local_box_size:
      type: int
      attr: in, required
      doc: Dimensions of the local box for each MPI rank.
    fcomm:
      type: int
      attr: in, required
      doc: Fortran communicator used to partition FFT grid into local boxes.
    values:
      type: double
      attr: in, required
      doc: Values of the function (local buffer for each MPI rank).
    transform_to_pw:
      type: bool
      attr: in, optional
      doc: If true, transform function to PW domain.
@api end
*/
void sirius_set_rg_values(void*  const* handler__,
                          char   const* label__,
                          int    const* grid_dims__,
                          int    const* local_box_origin__,
                          int    const* local_box_size__,
                          int    const* fcomm__,
                          double const* values__,
                          bool   const* transform_to_pw__)
{
    PROFILE("sirius_api::sirius_set_rg_values");

    auto& gs = get_gs(handler__);

    std::string label(label__);

    for (int x: {0, 1, 2}) {
        if (grid_dims__[x] != gs.ctx().fft_grid()[x]) {
            std::stringstream s;
            s << "wrong FFT grid size\n"
                 "  SIRIUS internal: " << gs.ctx().fft_grid()[0] << " " <<  gs.ctx().fft_grid()[1] << " "
                                       << gs.ctx().fft_grid()[2] << "\n"
                 "  host code:       " << grid_dims__[0] << " " << grid_dims__[1] << " " << grid_dims__[2];
            TERMINATE(s.str());
        }
    }

    std::map<std::string, sirius::Smooth_periodic_function<double>*> func = {
        {"rho",  &gs.density().rho()},
        {"magz", &gs.density().magnetization(0)},
        {"magx", &gs.density().magnetization(1)},
        {"magy", &gs.density().magnetization(2)},
        {"veff", &gs.potential().effective_potential()},
        {"bz",   &gs.potential().effective_magnetic_field(0)},
        {"bx",   &gs.potential().effective_magnetic_field(1)},
        {"by",   &gs.potential().effective_magnetic_field(2)},
        {"vxc",  &gs.potential().xc_potential()},
    };

    try {
        auto& f = func.at(label);

        auto& comm = Communicator::map_fcomm(*fcomm__);

        mdarray<int, 2> local_box_size(const_cast<int*>(local_box_size__), 3, comm.size());
        mdarray<int, 2> local_box_origin(const_cast<int*>(local_box_origin__), 3, comm.size());

        for (int rank = 0; rank < comm.size(); rank++) {
            /* dimensions of this rank's local box */
            int nx = local_box_size(0, rank);
            int ny = local_box_size(1, rank);
            int nz = local_box_size(2, rank);

            mdarray<double, 3> buf(nx, ny, nz);
            /* if this is that rank's turn to broadcast */
            if (comm.rank() == rank) {
                /* copy values to buf */
                std::copy(values__, values__ + nx * ny * nz, &buf[0]);
            }
            /* send a copy of local box to all ranks */
            comm.bcast(&buf[0], nx * ny * nz, rank);

            for (int iz = 0; iz < nz; iz++) {
                /* global z coordinate inside FFT box */
                int z = local_box_origin(2, rank) + iz - 1; /* Fortran counts from 1 */
                /* each rank on SIRIUS side, for which this condition is fulfilled copies data from the local box */
                if (z >= gs.ctx().spfft().local_z_offset() && z < gs.ctx().spfft().local_z_offset() + gs.ctx().spfft().local_z_length()) {
                    /* make z local for SIRIUS FFT partitioning */
                    z -= gs.ctx().spfft().local_z_offset();
                    for (int iy = 0; iy < ny; iy++) {
                        /* global y coordinate inside FFT box */
                        int y = local_box_origin(1, rank) + iy - 1; /* Fortran counts from 1 */
                        for (int ix = 0; ix < nx; ix++) {
                            /* global x coordinate inside FFT box */
                            int x = local_box_origin(0, rank) + ix - 1; /* Fortran counts from 1 */
                            f->f_rg(gs.ctx().fft_grid().index_by_coord(x, y, z)) = buf(ix, iy, iz);
                        }
                    }
                }
            }
        } /* loop over ranks */
        if (transform_to_pw__ && *transform_to_pw__) {
            f->fft_transform(-1);
        }
    } catch(...) {
        TERMINATE("wrong label");
    }
}

/*
@api begin
sirius_get_rg_values:
  doc: Get the values of the function on the regular grid.
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: DFT ground state handler.
    label:
      type: string
      attr: in, required
      doc: Label of the function.
    grid_dims:
      type: int
      attr: in, required
      doc: Dimensions of the FFT grid.
    local_box_origin:
      type: int
      attr: in, required
      doc: Coordinates of the local box origin for each MPI rank
    local_box_size:
      type: int
      attr: in, required
      doc: Dimensions of the local box for each MPI rank.
    fcomm:
      type: int
      attr: in, required
      doc: Fortran communicator used to partition FFT grid into local boxes.
    values:
      type: double
      attr: out, required
      doc: Values of the function (local buffer for each MPI rank).
    transform_to_rg:
      type: bool
      attr: in, optional
      doc: If true, transform function to regular grid before fetching the values.
@api end
*/
void sirius_get_rg_values(void*  const* handler__,
                          char   const* label__,
                          int    const* grid_dims__,
                          int    const* local_box_origin__,
                          int    const* local_box_size__,
                          int    const* fcomm__,
                          double*       values__,
                          bool   const* transform_to_rg__)
{
    PROFILE("sirius_api::sirius_get_rg_values");

    auto& gs = get_gs(handler__);

    std::string label(label__);

    for (int x: {0, 1, 2}) {
        if (grid_dims__[x] != gs.ctx().fft_grid()[x]) {
            TERMINATE("wrong FFT grid size");
        }
    }

    std::map<std::string, sirius::Smooth_periodic_function<double>*> func = {
        {"rho",  &gs.density().rho()},
        {"magz", &gs.density().magnetization(0)},
        {"magx", &gs.density().magnetization(1)},
        {"magy", &gs.density().magnetization(2)},
        {"veff", &gs.potential().effective_potential()},
        {"bz",   &gs.potential().effective_magnetic_field(0)},
        {"bx",   &gs.potential().effective_magnetic_field(1)},
        {"by",   &gs.potential().effective_magnetic_field(2)},
        {"vxc",  &gs.potential().xc_potential()},
    };

    try {
        auto& f = func.at(label);

        auto& comm = Communicator::map_fcomm(*fcomm__);

        if (transform_to_rg__ && *transform_to_rg__) {
            f->fft_transform(1);
        }

        auto& fft_comm = gs.ctx().comm_fft();
        auto spl_z = split_fft_z(gs.ctx().fft_grid()[2], fft_comm);

        mdarray<int, 2> local_box_size(const_cast<int*>(local_box_size__), 3, comm.size());
        mdarray<int, 2> local_box_origin(const_cast<int*>(local_box_origin__), 3, comm.size());

        for (int rank = 0; rank < fft_comm.size(); rank++) {
            /* slab of FFT grid for a given rank */
            mdarray<double, 3> buf(f->spfft().dim_x(), f->spfft().dim_y(), spl_z.local_size(rank));
            if (rank == fft_comm.rank()) {
                std::copy(&f->f_rg(0), &f->f_rg(0) + f->spfft().local_slice_size(), &buf[0]);
            }
            fft_comm.bcast(&buf[0], static_cast<int>(buf.size()), rank);

            /* ranks on the F90 side */
            int r = comm.rank();

            /* dimensions of this rank's local box */
            int nx = local_box_size(0, r);
            int ny = local_box_size(1, r);
            int nz = local_box_size(2, r);
            mdarray<double, 3> values(values__, nx, ny, nz);

            for (int iz = 0; iz < nz; iz++) {
                /* global z coordinate inside FFT box */
                int z = local_box_origin(2, r) + iz - 1; /* Fortran counts from 1 */
                if (z >= spl_z.global_offset(rank) && z < spl_z.global_offset(rank) + spl_z.local_size(rank)) {
                    /* make z local for SIRIUS FFT partitioning */
                    z -= spl_z.global_offset(rank);
                    for (int iy = 0; iy < ny; iy++) {
                        /* global y coordinate inside FFT box */
                        int y = local_box_origin(1, r) + iy - 1; /* Fortran counts from 1 */
                        for (int ix = 0; ix < nx; ix++) {
                            /* global x coordinate inside FFT box */
                            int x = local_box_origin(0, r) + ix - 1; /* Fortran counts from 1 */
                            values(ix, iy, iz) = buf(x, y, z);
                        }
                    }
                }
            }
        } /* loop over ranks */
    } catch(...) {
        TERMINATE("wrong label");
    }
}


/*
@api begin
sirius_get_total_magnetization:
  doc: Get the total magnetization of the system.
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: DFT ground state handler.
    mag:
      type: double
      attr: out, required
      doc: 3D magnetization vector (x,y,z components).
@api end
*/
void sirius_get_total_magnetization(void* const* handler__,
                                    double*      mag__)
{
    auto& gs = get_gs(handler__);

    mdarray<double, 1> total_mag(mag__, 3);
    total_mag.zero();
    for (int j = 0; j < gs.ctx().num_mag_dims(); j++) {
        auto result = gs.density().magnetization(j).integrate();
        total_mag[j] = std::get<0>(result);
    }
    if (gs.ctx().num_mag_dims() == 3) {
        /* swap z and x and change order from z,x,y to x,z,y */
        std::swap(total_mag[0], total_mag[1]);
        /* swap z and y and change order x,z,y to x,y,z */
        std::swap(total_mag[1], total_mag[2]);
    }
}

/*
@api begin
sirius_get_num_kpoints:
  doc: Get the total number of kpoints
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Kpoint set handler
    num_kpoints:
      type: int
      attr: out, required
      doc: number of kpoints in the set
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/

void sirius_get_num_kpoints(void* const* handler__,
                            int *num_kpoints__,
                            int *error_code__)
{
    call_sirius([&]()
    {
        auto& ks = get_ks(handler__);
        *num_kpoints__ = ks.num_kpoints();
    }, error_code__);
}

/*
@api begin
sirius_get_kpoint_properties:
  doc: Get the kpoint properties
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Kpoint set handler
    ik:
      type: int
      attr: in, required
      doc: Index of the kpoint
    weight:
      type: double
      attr: out, required
      doc: Weight of the kpoint
    coordinates:
      type: double
      attr: out, optional
      doc: Coordinates of the kpoint
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void sirius_get_kpoint_properties(void* const* handler__,
                                  int const* ik__,  // TODO assume Fortran index (starting from 1)
                                  double *weight__,
                                  double *coordinates__,
                                  int *error_code__)
{
    call_sirius([&]()
    {
        auto& ks = get_ks(handler__);
        int ik = *ik__;
        *weight__ = ks[ik]->weight();

        if (coordinates__) {
            coordinates__[0] = ks[ik]->vk()[0];
            coordinates__[1] = ks[ik]->vk()[1];
            coordinates__[2] = ks[ik]->vk()[2];
        }
    }, error_code__);
}

/*
@api begin
sirius_get_matching_coefficients:
  doc: Get matching coefficients for all atoms.
  full_doc: Warning! Generation of matching coefficients for all atoms has a large memory footprint. Use it with caution.
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: K-point set handler.
    ik:
      type: int
      attr: in, required
      doc: Index of k-point.
    alm:
      type: complex
      attr: out, required
      doc: Matching coefficients.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void sirius_get_matching_coefficients(void* const* handler__, int const* ik__, std::complex<double>* alm__,
                                      int* error_code__)
{
    call_sirius([&]()
    {
        auto& ks = get_ks(handler__);
        auto& sctx = ks.ctx();

        auto& uc = sctx.unit_cell();
        auto& kp = *ks[*ik__ - 1];
        auto& gk = kp.gkvec();

        std::vector<int> igk(gk.num_gvec());
        std::iota(igk.begin(), igk.end(), 0);

        sddk::mdarray<std::complex<double>, 3> alm(alm__, gk.num_gvec(), uc.max_mt_aw_basis_size(), uc.num_atoms());
        sirius::Matching_coefficients Alm(uc, sctx.lmax_apw(), gk.num_gvec(), igk, gk);
        for (int ia = 0; ia < uc.num_atoms(); ia++) {
            sddk::mdarray<std::complex<double>, 2> alm_tmp(&alm(0, 0, ia), gk.num_gvec(), uc.max_mt_aw_basis_size());
            Alm.generate<false>(uc.atom(ia), alm_tmp);
        }
    }, error_code__);
}

/*
@api begin
sirius_set_callback_function:
  doc: Set callback function to compute various radial integrals.
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Simulation context handler.
    label:
      type: string
      attr: in, required
      doc: Lable of the callback function.
    fptr:
      type: func
      attr: in, required, value
      doc: Pointer to callback function.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void sirius_set_callback_function(void* const* handler__, char const* label__, void(*fptr__)(), int* error_code__)
{
    call_sirius([&]()
    {
        auto label = std::string(label__);
        std::transform(label.begin(), label.end(), label.begin(), ::tolower);
        auto& sim_ctx = get_sim_ctx(handler__);
        if (label == "beta_ri") {
            sim_ctx.beta_ri_callback(reinterpret_cast<void(*)(int, double, double*, int)>(fptr__));
        } else if (label == "beta_ri_djl") {
            sim_ctx.beta_ri_djl_callback(reinterpret_cast<void(*)(int, double, double*, int)>(fptr__));
        } else if (label == "aug_ri") {
            sim_ctx.aug_ri_callback(reinterpret_cast<void(*)(int, double, double*, int, int)>(fptr__));
        } else if (label == "aug_ri_djl") {
            sim_ctx.aug_ri_djl_callback(reinterpret_cast<void(*)(int, double, double*, int, int)>(fptr__));
        } else if (label == "vloc_ri") {
            sim_ctx.vloc_ri_callback(reinterpret_cast<void(*)(int, int, double*, double*)>(fptr__));
        } else if (label == "vloc_ri_djl") {
            sim_ctx.vloc_ri_djl_callback(reinterpret_cast<void(*)(int, int, double*, double*)>(fptr__));
        } else if (label == "rhoc_ri") {
            sim_ctx.rhoc_ri_callback(reinterpret_cast<void(*)(int, int, double*, double*)>(fptr__));
        } else if (label == "rhoc_ri_djl") {
            sim_ctx.rhoc_ri_djl_callback(reinterpret_cast<void(*)(int, int, double*, double*)>(fptr__));
        } else if (label == "band_occ") {
            sim_ctx.band_occ_callback(reinterpret_cast<void(*)(void)>(fptr__));
        } else if (label == "veff") {
            sim_ctx.veff_callback(reinterpret_cast<void(*)(void)>(fptr__));
        } else if (label == "ps_rho_ri") {
            sim_ctx.ps_rho_ri_callback(reinterpret_cast<void(*)(int, int, double*, double*)>(fptr__));
        } else {
            std::stringstream s;
            s << "Wrong label of the callback function: " << label;
            throw std::runtime_error(s.str());
        }
    }, error_code__);
}

/*
@api begin
sirius_nlcg:
  doc: Robust wave function optimizer
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Ground state handler
    ks_handler:
      type: void*
      attr: in, required
      doc: point set handler
@api end
*/

void sirius_nlcg(void* const* handler__,
                 void* const* ks_handler__)
{
#ifdef SIRIUS_NLCGLIB
    // call nlcg solver
    auto& gs = get_gs(handler__);
    auto& potential = gs.potential();
    auto& density = gs.density();

    auto& kset = get_ks(ks_handler__);
    auto& ctx = kset.ctx();

    auto nlcg_params  = ctx.nlcg_input();
    double temp       = nlcg_params.T_;
    double tol        = nlcg_params.tol_;
    double kappa      = nlcg_params.kappa_;
    double tau        = nlcg_params.tau_;
    int maxiter       = nlcg_params.maxiter_;
    int restart       = nlcg_params.restart_;
    std::string smear = nlcg_params.smearing_;
    std::string pu = nlcg_params.processing_unit_;

    nlcglib::smearing_type smearing;
    if (smear.compare("FD") == 0) {
        smearing = nlcglib::smearing_type::FERMI_DIRAC;
    } else if (smear.compare("GS") == 0) {
        smearing = nlcglib::smearing_type::GAUSSIAN_SPLINE;
    } else {
        throw std::runtime_error("invalid smearing type given");
    }

    sirius::Energy energy(kset, density, potential);
    if (is_device_memory(ctx.preferred_memory_t())) {
        if (pu.empty() || pu.compare("gpu") == 0) {
            nlcglib::nlcg_mvp2_device(energy, smearing, temp, tol, kappa, tau, maxiter, restart);
        } else if (pu.compare("cpu") == 0) {
            nlcglib::nlcg_mvp2_device_cpu(energy, smearing, temp, tol, kappa, tau, maxiter, restart);
        } else {
            throw std::runtime_error("invalid processing unit for nlcg given: " + pu);
        }
    } else {
        if (pu.empty() || pu.compare("gpu") == 0) {
            nlcglib::nlcg_mvp2_cpu(energy, smearing, temp, tol, kappa, tau, maxiter, restart);
        } else if (pu.compare("cpu") == 0) {
            nlcglib::nlcg_mvp2_cpu_device(energy, smearing, temp, tol, kappa, tau, maxiter, restart);
        } else {
            throw std::runtime_error("invalid processing unit for nlcg given: " + pu);
        }
    }

#else
    throw std::runtime_error("SIRIUS was not compiled with NLCG option.");
#endif
}

/*
@api begin
sirius_nlcg_params:
  doc: Robust wave function optimizer
  arguments:
    handler:
      type: void*
      attr: in, required
      doc: Ground state handler
    ks_handler:
      type: void*
      attr: in, required
      doc: point set handler
    temp:
      type: double
      attr: in, required
      doc: Temperature in Kelvin
    smearing:
      type: string
      attr: in, required
      doc: smearing label
    kappa:
      type: double
      attr: in, required
      doc: pseudo-Hamiltonian scalar preconditioner
    tau:
      type: double
      attr: in, required
      doc: backtracking search reduction parameter
    tol:
      type: double
      attr: in, required
      doc: CG tolerance
    maxiter:
      type: int
      attr: in, required
      doc: CG maxiter
    restart:
      type: int
      attr: in, required
      doc: CG restart
    processing_unit:
      type: string
      attr: in, required
      doc: processing_unit = ["cpu"|"gpu"|"none"]
@api end
*/

void sirius_nlcg_params(void* const* handler__,
                        void* const* ks_handler__,
                        double const* temp__,
                        char const* smearing__,
                        double const* kappa__,
                        double const* tau__,
                        double const* tol__,
                        int const* maxiter__,
                        int const* restart__,
                        char const* processing_unit__)
{
#ifdef SIRIUS_NLCGLIB
    // call nlcg solver
    auto& gs = get_gs(handler__);
    auto& potential = gs.potential();
    auto& density = gs.density();

    auto& kset = get_ks(ks_handler__);
    auto& ctx = kset.ctx();

    double temp = *temp__;
    double kappa = *kappa__;
    double tau = *tau__;
    double tol = *tol__;
    int maxiter = *maxiter__;
    int restart = *restart__;

    std::string smear(smearing__);
    std::string pu(processing_unit__);

    nlcglib::smearing_type smearing_t;
    if (smear.compare("FD") == 0) {
        smearing_t = nlcglib::smearing_type::FERMI_DIRAC;
    } else if (smear.compare("GS") == 0) {
        smearing_t = nlcglib::smearing_type::GAUSSIAN_SPLINE;
    } else {
        throw std::runtime_error("invalid smearing type given");
    }

    if(pu.compare("none") == 0) {
      // use same processing unit as SIRIUS
      pu = ctx.control().processing_unit_;
    }

    nlcglib::nlcg_info info;

    sirius::Energy energy(kset, density, potential);
    if (is_device_memory(ctx.preferred_memory_t())) {
        if (pu.empty() || pu.compare("gpu") == 0) {
            info = nlcglib::nlcg_mvp2_device(energy, smearing_t, temp, tol, kappa, tau, maxiter, restart);
        } else if (pu.compare("cpu") == 0) {
            info = nlcglib::nlcg_mvp2_device_cpu(energy, smearing_t, temp, tol, kappa, tau, maxiter, restart);
        } else {
            throw std::runtime_error("invalid processing unit for nlcg given: " + pu);
        }
    } else {
        if (pu.empty() || pu.compare("cpu") == 0) {
            info = nlcglib::nlcg_mvp2_cpu(energy, smearing_t, temp, tol, kappa, tau, maxiter, restart);
        } else if (pu.compare("gpu") == 0) {
            info = nlcglib::nlcg_mvp2_cpu_device(energy, smearing_t, temp, tol, kappa, tau, maxiter, restart);
        } else {
            throw std::runtime_error("invalid processing unit for nlcg given: " + pu);
        }
    }

#else
    throw std::runtime_error("SIRIUS was not compiled with NLCG option.");
#endif
}

} // extern "C"
