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
#include "sirius.h"
#include "utils/any_ptr.hpp"
#include "utils/profiler.hpp"
#include "error_codes.hpp"

static inline void sirius_exit(int error_code__, std::string msg__ = "")
{
    switch (error_code__) {
        case SIRIUS_ERROR_UNKNOWN: {
            printf("Unknown error\n");
            break;
        }
        case SIRIUS_ERROR_RUNTIME: {
            printf("Run-time error\n");
            break;
        }
        default: {
            printf("Unknown error code: %i\n", error_code__);
            break;
        }
    }

    if (msg__.size()) {
        printf("%s\n", msg__.c_str());
    }
    if (!Communicator::is_finalized()) {
        Communicator::world().abort(error_code__);
    }
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
    catch (...) {
        if (error_code__) {
            *error_code__ = SIRIUS_ERROR_UNKNOWN;
            return;
        } else {
            sirius_exit(SIRIUS_ERROR_UNKNOWN);
        }
    }
}

// TODO: try..catch in all calls to SIRIUS, return error codes to fortran as last optional argument

sirius::Simulation_context& get_sim_ctx(void* const* h)
{
    assert(h != nullptr);
    return static_cast<utils::any_ptr*>(*h)->get<sirius::Simulation_context>();
}

sirius::DFT_ground_state& get_gs(void* const* h)
{
    assert(h != nullptr);
    return static_cast<utils::any_ptr*>(*h)->get<sirius::DFT_ground_state>();
}

sirius::K_point_set& get_ks(void* const* h)
{
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

/* @fortran begin function void sirius_initialize       Initialize the SIRIUS library.
   @fortran argument in required bool call_mpi_init     If .true. then MPI_Init must be called prior to initialization.
   @fortran end */
void sirius_initialize(bool const* call_mpi_init__)
{
    sirius::initialize(*call_mpi_init__);
}

/* @fortran begin function void sirius_finalize          Shut down the SIRIUS library
   @fortran argument in optional bool call_mpi_fin       If .true. then MPI_Finalize must be called after the shutdown.
   @fortran argument in optional bool call_device_reset  If .true. then cuda device is reset after shutdown.
   @fortran argument in optional bool call_fftw_fin      If .true. then fft_cleanup must be called after the shutdown.
   @fortran end */

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

/* @fortran begin function void sirius_start_timer      Start the timer.
   @fortran argument in required string name            Timer label.
   @fortran end */
void sirius_start_timer(char const* name__)
{
    ::utils::global_rtgraph_timer.start(name__);
}

/* @fortran begin function void sirius_stop_timer       Stop the running timer.
   @fortran argument in required string name            Timer label.
   @fortran end */
void sirius_stop_timer(char const* name__)
{
    ::utils::global_rtgraph_timer.stop(name__);
}

/* @fortran begin function void sirius_print_timers      Print all timers.
   @fortran end */
void sirius_print_timers(void)
{
    std::cout << ::utils::global_rtgraph_timer.process().print();
}

/* @fortran begin function void sirius_serialize_timers    Save all timers to JSON file.
   @fortran argument in required string fname              Name of the output JSON file.
   @fortran end */
void sirius_serialize_timers(char const* fname__)
{
    std::ofstream ofs(fname__, std::ofstream::out | std::ofstream::trunc);
    ofs << ::utils::global_rtgraph_timer.process().json();
}

/* @fortran begin function void sirius_integrate        Spline integration of f(x)*x^m.
   @fortran argument in  required int    m              Defines the x^{m} factor.
   @fortran argument in  required int    np             Number of x-points.
   @fortran argument in  required double x              List of x-points.
   @fortran argument in  required double f              List of function values.
   @fortran argument out required double result         Resulting value.
   @fortran end */
void sirius_integrate(int    const* m__,
                      int    const* np__,
                      double const* x__,
                      double const* f__,
                      double*       result__)
{
    sirius::Radial_grid_ext<double> rgrid(*np__, x__);
    sirius::Spline<double> s(rgrid, std::vector<double>(f__, f__ + *np__));
    *result__ = s.integrate(*m__);
}

/* @fortran begin function bool sirius_context_initialized      Check if the simulation context is initialized.
   @fortran argument in required void* handler                  Simulation context handler.
   @fortran end */
bool sirius_context_initialized(void* const* handler__)
{
    if (*handler__ == nullptr) {
        return false;
    }
    auto& sim_ctx = get_sim_ctx(handler__);
    return sim_ctx.initialized();
}

/* @fortran begin function void* sirius_create_context        Create context of the simulation.
   @fortran argument in  required int   fcomm                 Entire communicator of the simulation.
   @fortran details
   Simulation context is the complex data structure that holds all the parameters of the individual simulation.
   The context must be created, populated with the correct parameters and initialized before using all subsequent
   SIRIUS functions.
   @fortran end */
void* sirius_create_context(int const* fcomm__)
{
    auto& comm = Communicator::map_fcomm(*fcomm__);
    return new utils::any_ptr(new sirius::Simulation_context(comm));
}

/* @fortran begin function void sirius_import_parameters        Import parameters of simulation from a JSON string
   @fortran argument in required void* handler                  Simulation context handler.
   @fortran argument in optional string str                     JSON string with parameters or a JSON file.
   @fortran end */
void sirius_import_parameters(void* const* handler__,
                              char  const* str__)
{
    auto& sim_ctx = get_sim_ctx(handler__);
    if (str__) {
        sim_ctx.import(std::string(str__));
    } else {
        sim_ctx.import(sim_ctx.get_runtime_options_dictionary());
    }
}

/* @fortran begin function void sirius_set_parameters            Set parameters of the simulation.
   @fortran argument in required void* handler                   Simulation context handler
   @fortran argument in optional int lmax_apw                    Maximum orbital quantum number for APW functions.
   @fortran argument in optional int lmax_rho                    Maximum orbital quantum number for density.
   @fortran argument in optional int lmax_pot                    Maximum orbital quantum number for potential.
   @fortran argument in optional int num_fv_states               Number of first-variational states.
   @fortran argument in optional int num_bands                   Number of bands.
   @fortran argument in optional int num_mag_dims                Number of magnetic dimensions.
   @fortran argument in optional double pw_cutoff                Cutoff for G-vectors.
   @fortran argument in optional double gk_cutoff                Cutoff for G+k-vectors.
   @fortran argument in optional int fft_grid_size               Size of the fine-grain FFT grid.
   @fortran argument in optional int auto_rmt                    Set the automatic search of muffin-tin radii.
   @fortran argument in optional bool gamma_point                True if this is a Gamma-point calculation.
   @fortran argument in optional bool use_symmetry               True if crystal symmetry is taken into account.
   @fortran argument in optional bool so_correction              True if spin-orbit correnctio is enabled.
   @fortran argument in optional string valence_rel              Valence relativity treatment.
   @fortran argument in optional string core_rel                 Core relativity treatment.
   @fortran argument in optional string esm_bc                   Type of boundary condition for effective screened medium.
   @fortran argument in optional double iter_solver_tol          Tolerance of the iterative solver.
   @fortran argument in optional double iter_solver_tol_empty    Tolerance for the empty states.
   @fortran argument in optional string iter_solver_type         Type of iterative solver.
   @fortran argument in optional int    verbosity                Verbosity level.
   @fortran argument in optional bool   hubbard_correction       True if LDA+U correction is enabled.
   @fortran argument in optional int    hubbard_correction_kind  Type of LDA+U implementation (simplified or full).
   @fortran argument in optional string hubbard_orbitals         Type of localized orbitals.
   @fortran argument in optional int    sht_coverage             Type of spherical coverage (0: Lebedev-Laikov, 1: uniform).
   @fortran argument in optional double min_occupancy            Minimum band occupancy to trat is as "occupied".
   @fortran end */
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
                           char   const* esm_bc__,
                           double const* iter_solver_tol__,
                           double const* iter_solver_tol_empty__,
                           char   const* iter_solver_type__,
                           int    const* verbosity__,
                           bool   const* hubbard_correction__,
                           int    const* hubbard_correction_kind__,
                           char   const* hubbard_orbitals__,
                           int    const* sht_coverage__,
                           double const* min_occupancy__)
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
        sim_ctx.set_valence_relativity(valence_rel__);
    }
    if (core_rel__ != nullptr) {
        sim_ctx.set_core_relativity(core_rel__);
    }
    if (esm_bc__ != nullptr) {
        sim_ctx.esm_bc(std::string(esm_bc__));
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
            sim_ctx.set_hubbard_simplified_version();
        }
    }
    if (hubbard_orbitals__ != nullptr) {
        std::string s(hubbard_orbitals__);
        if (s == "ortho-atomic") {
            sim_ctx.set_orthogonalize_hubbard_orbitals(true);
        }
        if (s == "norm-atomic") {
            sim_ctx.set_normalize_hubbard_orbitals(true);
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
}

/* @fortran begin function void sirius_get_parameters             Get parameters of the simulation.
   @fortran argument in  required void* handler                   Simulation context handler
   @fortran argument out optional int lmax_apw                    Maximum orbital quantum number for APW functions.
   @fortran argument out optional int lmax_rho                    Maximum orbital quantum number for density.
   @fortran argument out optional int lmax_pot                    Maximum orbital quantum number for potential.
   @fortran argument out optional int num_fv_states               Number of first-variational states.
   @fortran argument out optional int num_bands                   Number of bands.
   @fortran argument out optional int num_mag_dims                Number of magnetic dimensions.
   @fortran argument out optional double pw_cutoff                Cutoff for G-vectors.
   @fortran argument out optional double gk_cutoff                Cutoff for G+k-vectors.
   @fortran argument out optional int fft_grid_size               Size of the fine-grain FFT grid.
   @fortran argument out optional int auto_rmt                    Set the automatic search of muffin-tin radii.
   @fortran argument out optional bool gamma_point                True if this is a Gamma-point calculation.
   @fortran argument out optional bool use_symmetry               True if crystal symmetry is taken into account.
   @fortran argument out optional bool so_correction              True if spin-orbit correnctio is enabled.
   @fortran argument out optional double iter_solver_tol          Tolerance of the iterative solver.
   @fortran argument out optional double iter_solver_tol_empty    Tolerance for the empty states.
   @fortran argument out optional int    verbosity                Verbosity level.
   @fortran argument out optional bool   hubbard_correction       True if LDA+U correction is enabled.
   @fortran end */
void sirius_get_parameters(void* const* handler__,
                           int*         lmax_apw__,
                           int*         lmax_rho__,
                           int*         lmax_pot__,
                           int*         num_fv_states__,
                           int*         num_bands__,
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
                           bool*        hubbard_correction__)
{
    auto& sim_ctx = get_sim_ctx(handler__);
    if (lmax_apw__ != nullptr) {
        *lmax_apw__ = sim_ctx.lmax_apw();
    }
    if (lmax_rho__ != nullptr) {
        *lmax_rho__ = sim_ctx.lmax_rho();
    }
    if (lmax_pot__ != nullptr) {
        *lmax_pot__ = sim_ctx.lmax_pot();
    }
    if (num_fv_states__ != nullptr) {
        *num_fv_states__ = sim_ctx.num_fv_states();
    }
    if (num_bands__ != nullptr) {
        *num_bands__ = sim_ctx.num_bands();
    }
    if (num_mag_dims__ != nullptr) {
        *num_mag_dims__ = sim_ctx.num_mag_dims();
    }
    if (pw_cutoff__ != nullptr) {
        *pw_cutoff__ = sim_ctx.pw_cutoff();
    }
    if (gk_cutoff__ != nullptr) {
        *gk_cutoff__ = sim_ctx.gk_cutoff();
    }
    if (auto_rmt__ != nullptr) {
        *auto_rmt__ = sim_ctx.auto_rmt();
    }
    if (gamma_point__ != nullptr) {
        *gamma_point__ = sim_ctx.gamma_point();
    }
    if (use_symmetry__ != nullptr) {
        *use_symmetry__ = sim_ctx.use_symmetry();
    }
    if (so_correction__ != nullptr) {
        *so_correction__ = sim_ctx.so_correction();
    }
    if (iter_solver_tol__ != nullptr) {
        *iter_solver_tol__ = sim_ctx.iterative_solver_tolerance();
    }
    if (iter_solver_tol_empty__ != nullptr) {
        *iter_solver_tol_empty__ = sim_ctx.iterative_solver_input().empty_states_tolerance_;
    }
    if (verbosity__ != nullptr) {
        *verbosity__ = sim_ctx.control().verbosity_;
    }
    if (hubbard_correction__ != nullptr) {
        *hubbard_correction__ = sim_ctx.hubbard_correction();
    }
    if (fft_grid_size__ != nullptr) {
        for (int x: {0, 1, 2}) {
            fft_grid_size__[x] = sim_ctx.fft_grid()[x];
        }
    }
}


/* @fortran begin function void sirius_add_xc_functional         Add one of the XC functionals.
   @fortran argument in required void* handler                   Simulation context handler
   @fortran argument in required string name                     LibXC label of the functional.
   @fortran end */
void sirius_add_xc_functional(void* const* handler__,
                              char  const* name__)
{
    auto& sim_ctx = get_sim_ctx(handler__);
    sim_ctx.add_xc_functional(std::string(name__));
}

/* @fortran begin function void sirius_set_mpi_grid_dims      Set dimensions of the MPI grid.
   @fortran argument in required void*  handler               Simulation context handler
   @fortran argument in required int    ndims                 Number of dimensions.
   @fortran argument in required int    dims                  Size of each dimension.
   @fortran end */
void sirius_set_mpi_grid_dims(void* const* handler__,
                              int   const* ndims__,
                              int   const* dims__)
{
    assert(*ndims__ > 0);
    auto& sim_ctx = get_sim_ctx(handler__);
    std::vector<int> dims(dims__, dims__ + *ndims__);
    sim_ctx.mpi_grid_dims(dims);
}

/* @fortran begin function void sirius_set_lattice_vectors   Set vectors of the unit cell.
   @fortran argument in required void* handler               Simulation context handler
   @fortran argument in required double a1                   1st vector
   @fortran argument in required double a2                   2nd vector
   @fortran argument in required double a3                   3rd vector
   @fortran end */
void sirius_set_lattice_vectors(void*  const* handler__,
                                double const* a1__,
                                double const* a2__,
                                double const* a3__)
{
    auto& sim_ctx = get_sim_ctx(handler__);
    sim_ctx.unit_cell().set_lattice_vectors(vector3d<double>(a1__), vector3d<double>(a2__), vector3d<double>(a3__));
}

/* @fortran begin function void sirius_initialize_context     Initialize simulation context.
   @fortran argument in required void* handler                Simulation context handler.
   @fortran end */
void sirius_initialize_context(void* const* handler__)
{
    auto& sim_ctx = get_sim_ctx(handler__);
    sim_ctx.initialize();
}

/* @fortran begin function void sirius_update_context     Update simulation context after changing lattice or atomic positions.
   @fortran argument in required void* handler            Simulation context handler.
   @fortran end */
void sirius_update_context(void* const* handler__)
{
    auto& sim_ctx = get_sim_ctx(handler__);
    sim_ctx.update();
}

/* @fortran begin function void sirius_print_info      Print basic info
   @fortran argument in required void* handler         Simulation context handler.
   @fortran end */
void sirius_print_info(void* const* handler__)
{
    auto& sim_ctx = get_sim_ctx(handler__);
    sim_ctx.print_info();
}

/* @fortran begin function void sirius_free_handler     Free any handler of object created by SIRIUS.
   @fortran argument inout required void* handler       Handler of the object.
   @fortran end */
void sirius_free_handler(void** handler__)
{
    if (*handler__ != nullptr) {
        delete static_cast<utils::any_ptr*>(*handler__);
    }
    *handler__ = nullptr;
}

/* @fortran begin function void sirius_set_periodic_function_ptr   Set pointer to density or megnetization.
   @fortran argument in required void* handler                     Handler of the DFT ground state object.
   @fortran argument in required string label                      Label of the function.
   @fortran argument in optional double f_mt                       Pointer to the muffin-tin part of the function.
   @fortran argument in optional double f_rg                       Pointer to the regualr-grid part of the function.
   @fortran end */
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

/* @fortran begin function void* sirius_create_kset        Create k-point set from the list of k-points.
   @fortran argument in  required void*  handler           Simulation context handler.
   @fortran argument in  required int    num_kpoints       Total number of k-points in the set.
   @fortran argument in  required double kpoints           List of k-points in lattice coordinates.
   @fortran argument in  required double kpoint_weights    Weights of k-points.
   @fortran argument in  required bool   init_kset         If .true. k-set will be initialized.
   @fortran end */
void* sirius_create_kset(void*  const* handler__,
                         int    const* num_kpoints__,
                         double*       kpoints__,
                         double const* kpoint_weights__,
                         bool   const* init_kset__)
{
    auto& sim_ctx = get_sim_ctx(handler__);

    mdarray<double, 2> kpoints(kpoints__, 3, *num_kpoints__);

    sirius::K_point_set* new_kset = new sirius::K_point_set(sim_ctx);
    new_kset->add_kpoints(kpoints, kpoint_weights__);
    if (*init_kset__) {
        new_kset->initialize();
    }

    return new utils::any_ptr(new_kset);
}

/* @fortran begin function void* sirius_create_kset_from_grid        Create k-point set from a grid.
   @fortran argument in  required void*  handler                     Simulation context handler.
   @fortran argument in  required int    k_grid                      dimensions of the k points grid.
   @fortran argument in  required int k_shift                        k point shifts.
   @fortran argument in  required bool   use_symmetry                If .true. k-set will be generated using symmetries.
   @fortran end */

void *sirius_create_kset_from_grid(void* const* handler__,
                                   int   const* k_grid__,
                                   int   const* k_shift__,
                                   bool  const* use_symmetry)
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

    return new utils::any_ptr(new_kset);
}

/* @fortran begin function void* sirius_create_ground_state    Create a ground state object.
   @fortran argument in  required void*  ks_handler            Handler of the k-point set.
   @fortran end */
void* sirius_create_ground_state(void* const* ks_handler__)
{
    auto& ks = get_ks(ks_handler__);

    return new utils::any_ptr(new sirius::DFT_ground_state(ks));
}

/* @fortran begin function void sirius_find_ground_state        Find the ground state.
   @fortran argument in required void*  gs_handler              Handler of the ground state.
   @fortran argument in optional double density_tol             Tolerance on RMS in density.
   @fortran argument in optional double energy_tol              Tolerance in total energy difference.
   @fortran argument in optional int    niter                   Maximum number of SCF iterations.
   @fortran argument in optional bool   save_state              boolean variable indicating if we want to save the ground state.
   @fortran end */
void sirius_find_ground_state(void*  const* gs_handler__,
                              double const* density_tol__,
                              double const* energy_tol__,
                              int    const* niter__,
                              bool   const* save_state__)
{
    auto& gs = get_gs(gs_handler__);
    auto& ctx = gs.ctx();
    auto& inp = ctx.parameters_input();
    gs.initial_state();

    double rho_tol = inp.density_tol_;
    if (density_tol__) {
        rho_tol = *density_tol__;
    }

    double etol = inp.energy_tol_;
    if (energy_tol__) {
        etol = *energy_tol__;
    }

    int niter = inp.num_dft_iter_;
    if (niter__) {
        niter = *niter__;
    }

    bool save{false};
    if (save_state__ != nullptr) {
        save = *save_state__;
    }

    auto result = gs.find(rho_tol, etol, ctx.iterative_solver_tolerance(), niter, save);
}

/* @fortran begin function void sirius_update_ground_state   Update a ground state object after change of atomic coordinates or lattice vectors.
   @fortran argument in  required void*  gs_handler          Ground-state handler.
   @fortran end */
void sirius_update_ground_state(void** handler__)
{
    auto& gs = get_gs(handler__);
    gs.update();
}

/* @fortran begin function void sirius_add_atom_type     Add new atom type to the unit cell.
   @fortran argument in  required void*  handler         Simulation context handler.
   @fortran argument in  required string label           Atom type unique label.
   @fortran argument in  optional string fname           Species file name (in JSON format).
   @fortran argument in  optional int    zn              Nucleus charge.
   @fortran argument in  optional string symbol          Atomic symbol.
   @fortran argument in  optional double mass            Atomic mass.
   @fortran argument in  optional bool   spin_orbit      True if spin-orbit correction is enabled for this atom type.
   @fortran end */
void sirius_add_atom_type(void*  const* handler__,
                          char   const* label__,
                          char   const* fname__,
                          int    const* zn__,
                          char   const* symbol__,
                          double const* mass__,
                          bool   const* spin_orbit__)
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
}

/* @fortran begin function void sirius_set_atom_type_radial_grid        Set radial grid of the atom type.
   @fortran argument in  required void*  handler                        Simulation context handler.
   @fortran argument in  required string label                          Atom type label.
   @fortran argument in  required int    num_radial_points              Number of radial grid points.
   @fortran argument in  required double radial_points                  List of radial grid points.
   @fortran end */
void sirius_set_atom_type_radial_grid(void*  const* handler__,
                                      char   const* label__,
                                      int    const* num_radial_points__,
                                      double const* radial_points__)
{
    auto& sim_ctx = get_sim_ctx(handler__);

    auto& type = sim_ctx.unit_cell().atom_type(std::string(label__));
    type.set_radial_grid(*num_radial_points__, radial_points__);
}

/* @fortran begin function void sirius_set_atom_type_radial_grid_inf    Set radial grid of the free atom (up to effectice infinity).
   @fortran argument in  required void*  handler                        Simulation context handler.
   @fortran argument in  required string label                          Atom type label.
   @fortran argument in  required int    num_radial_points              Number of radial grid points.
   @fortran argument in  required double radial_points                  List of radial grid points.
   @fortran end */
void sirius_set_atom_type_radial_grid_inf(void*  const* handler__,
                                          char   const* label__,
                                          int    const* num_radial_points__,
                                          double const* radial_points__)
{
    auto& sim_ctx = get_sim_ctx(handler__);

    auto& type = sim_ctx.unit_cell().atom_type(std::string(label__));
    type.set_free_atom_radial_grid(*num_radial_points__, radial_points__);
}

/* @fortran begin function void sirius_add_atom_type_radial_function    Add one of the radial functions.
   @fortran argument in  required void*  handler                        Simulation context handler.
   @fortran argument in  required string atom_type                      Label of the atom type.
   @fortran argument in  required string label                          Label of the radial function.
   @fortran argument in  required double rf                             Array with radial function values.
   @fortran argument in  required int    num_points                     Length of radial function array.
   @fortran argument in  optional int    n                              Orbital quantum number.
   @fortran argument in  optional int    l                              angular momentum.
   @fortran argument in  optional int    idxrf1                         First index of radial function (for Q-operator).
   @fortran argument in  optional int    idxrf2                         Second index of radial function (for Q-operator).
   @fortran argument in  optional double occ                            Occupancy of the wave-function.
   @fortran end */
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
        type.add_ps_atomic_wf(n, *l__, std::vector<double>(rf__, rf__ + *num_points__), occ);
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

/* @fortran begin function void sirius_set_atom_type_hubbard    Set the hubbard correction for the atomic type.
   @fortran argument in  required void*   handler               Simulation context handler.
   @fortran argument in  required string  label                 Atom type label.
   @fortran argument in  required int     l                     Orbital quantum number.
   @fortran argument in  required int     n                     principal quantum number (s, p, d, f)
   @fortran argument in  required double  occ                   Atomic shell occupancy.
   @fortran argument in  required double  U                     Hubbard U parameter.
   @fortran argument in  required double  J                     Exchange J parameter for the full interaction treatment.
   @fortran argument in  required double  alpha                 J_alpha for the simple interaction treatment.
   @fortran argument in  required double  beta                  J_beta for the simple interaction treatment.
   @fortran argument in  required double  J0                    J0 for the simple interaction treatment.
   @fortran end */
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
    type.add_hubbard_orbital(*n__, *l__, *occ__, *U__, J__[1], J__, *alpha__, *beta__, *J0__);
}

/* @fortran begin function void sirius_set_atom_type_dion     Set ionic part of D-operator matrix.
   @fortran argument in  required void*   handler             Simulation context handler.
   @fortran argument in  required string  label               Atom type label.
   @fortran argument in  required int     num_beta            Number of beta-projectors.
   @fortran argument in  required double  dion                Ionic part of D-operator matrix.
   @fortran end */
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

/* @fortran begin function void sirius_set_atom_type_paw   Set PAW related data.
   @fortran argument in  required void*   handler          Simulation context handler.
   @fortran argument in  required string  label            Atom type label.
   @fortran argument in  required double  core_energy      Core-electrons energy contribution.
   @fortran argument in  required double  occupations      ?
   @fortran argument in  required int     num_occ          ?
   @fortran end */
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

/* @fortran begin function void sirius_add_atom         Add atom to the unit cell.
   @fortran argument in  required void*   handler       Simulation context handler.
   @fortran argument in  required string  label         Atom type label.
   @fortran argument in  required double  position      Atom position in lattice coordinates.
   @fortran argument in  optional double  vector_field  Starting magnetization.
   @fortran end */
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

/* @fortran begin function void sirius_set_atom_position  Set new atomic position.
   @fortran argument in  required void*   handler       Simulation context handler.
   @fortran argument in  required int     ia            Index of atom.
   @fortran argument in  required double  position      Atom position in lattice coordinates.
   @fortran end */
void sirius_set_atom_position(void*  const* handler__,
                              int    const* ia__,
                              double const* position__)
{
    auto& sim_ctx = get_sim_ctx(handler__);
    sim_ctx.unit_cell().atom(*ia__ - 1).set_position(std::vector<double>(position__, position__ + 3));
}

/* @fortran begin function void sirius_set_pw_coeffs         Set plane-wave coefficients of a periodic function.
   @fortran argument in  required void*   handler            Ground state handler.
   @fortran argument in  required string  label              Label of the function.
   @fortran argument in  required complex pw_coeffs          Local array of plane-wave coefficients.
   @fortran argument in  optional bool    transform_to_rg    True if function has to be transformed to real-space grid.
   @fortran argument in  optional int     ngv                Local number of G-vectors.
   @fortran argument in  optional int     gvl                List of G-vectors in lattice coordinates (Miller indices).
   @fortran argument in  optional int     comm               MPI communicator used in distribution of G-vectors
   @fortran end */
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
                        auto gvc = gs.ctx().unit_cell().reciprocal_lattice_vectors() * vector3d<double>(G[0], G[1], G[2]);
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

/* @fortran begin function void sirius_get_pw_coeffs      Get plane-wave coefficients of a periodic function.
   @fortran argument in  required void*   handler         Ground state handler.
   @fortran argument in  required string  label           Label of the function.
   @fortran argument in  required complex pw_coeffs       Local array of plane-wave coefficients.
   @fortran argument in  optional int     ngv             Local number of G-vectors.
   @fortran argument in  optional int     gvl             List of G-vectors in lattice coordinates (Miller indices).
   @fortran argument in  optional int     comm            MPI communicator used in distribution of G-vectors
   @fortran end */
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
                auto gvc = gs.ctx().unit_cell().reciprocal_lattice_vectors() * vector3d<double>(G[0], G[1], G[2]);
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

/* @fortran begin function void sirius_get_pw_coeffs_real   Get atom type contribution to plane-wave coefficients of a periodic function.
   @fortran argument in  required void*   handler           Simulation context handler.
   @fortran argument in  required string  atom_type         Label of the atom type.
   @fortran argument in  required string  label             Label of the function.
   @fortran argument in  required double  pw_coeffs         Local array of plane-wave coefficients.
   @fortran argument in  optional int     ngv               Local number of G-vectors.
   @fortran argument in  optional int     gvl               List of G-vectors in lattice coordinates (Miller indices).
   @fortran argument in  optional int     comm              MPI communicator used in distribution of G-vectors
   @fortran end */
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
            auto gc = sim_ctx.unit_cell().reciprocal_lattice_vectors() * vector3d<int>(gvec(0, i), gvec(1, i), gvec(2, i));
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

/* @fortran begin function void sirius_initialize_subspace    Initialize the subspace of wave-functions.
   @fortran argument in  required void*   gs_handler          Ground state handler.
   @fortran argument in  required void*   ks_handler          K-point set handler.
   @fortran end */
void sirius_initialize_subspace(void* const* gs_handler__,
                                void* const* ks_handler__)
{
    auto& gs = get_gs(gs_handler__);
    auto& ks = get_ks(ks_handler__);
    sirius::Hamiltonian0 H0(gs.potential());
    sirius::Band(ks.ctx()).initialize_subspace(ks, H0);
}

/* @fortran begin function void sirius_find_eigen_states     Find eigen-states of the Hamiltonian/
   @fortran argument in  required void*   gs_handler         Ground state handler.
   @fortran argument in  required void*   ks_handler         K-point set handler.
   @fortran argument in  required bool    precompute         True if neccessary data to setup eigen-value problem must be automatically precomputed.
   @fortran argument in  optional double  iter_solver_tol    Iterative solver tolerance.
   @fortran end */
void sirius_find_eigen_states(void* const* gs_handler__,
                              void* const* ks_handler__,
                              bool  const* precompute__,
                              double const* iter_solver_tol__)
{
    auto& gs = get_gs(gs_handler__);
    auto& ks = get_ks(ks_handler__);
    if (iter_solver_tol__ != nullptr) {
        ks.ctx().iterative_solver_tolerance(*iter_solver_tol__);
    }
    sirius::Hamiltonian0 H0(gs.potential());
    sirius::Band(ks.ctx()).solve(ks, H0, *precompute__);
}

/* @fortran begin function void sirius_generate_d_operator_matrix     Generate D-operator matrix.
   @fortran argument in  required void*   handler                     Ground state handler.
   @fortran end */
void sirius_generate_d_operator_matrix(void* const* handler__)
{
    auto& gs = get_gs(handler__);
    gs.potential().generate_D_operator_matrix();
}

/* @fortran begin function void sirius_generate_initial_density     Generate initial density.
   @fortran argument in  required void*   handler                   Ground state handler.
   @fortran end */
void sirius_generate_initial_density(void* const* handler__)
{
    auto& gs = get_gs(handler__);
    gs.density().initial_density();
}

/* @fortran begin function void sirius_generate_effective_potential     Generate effective potential and magnetic field.
   @fortran argument in  required void*   handler                       Ground state handler.
   @fortran end */
void sirius_generate_effective_potential(void* const* handler__)
{
    auto& gs = get_gs(handler__);
    gs.potential().generate(gs.density());
}

/* @fortran begin function void sirius_generate_density    Generate charge density and magnetization.
   @fortran argument in  required void*   gs_handler       Ground state handler.
   @fortran argument in  optional bool    add_core         Add core charge density in the muffin-tins.
   @fortran argument in  optional bool    transform_to_rg  If true, density and magnetization are transformed to real-space grid.
   @fortran end */
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

    gs.density().generate(gs.k_point_set(), add_core, transform_to_rg);
}

/* @fortran begin function void sirius_set_band_occupancies   Set band occupancies.
   @fortran argument in  required void*   ks_handler          K-point set handler.
   @fortran argument in  required int     ik                  Global index of k-point.
   @fortran argument in  required int     ispn                Spin component.
   @fortran argument in  required double  band_occupancies    Array of band occupancies.
   @fortran end */
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

/* @fortran begin function void sirius_get_band_energies         Get band energies.
   @fortran argument in  required void*   ks_handler             K-point set handler.
   @fortran argument in  required int     ik                     Global index of k-point.
   @fortran argument in  required int     ispn                   Spin component.
   @fortran argument out required double  band_energies          Array of band energies.
   @fortran end */
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

/* @fortran begin function void sirius_get_band_occupancies      Get band occupancies.
   @fortran argument in  required void*   ks_handler             K-point set handler.
   @fortran argument in  required int     ik                     Global index of k-point.
   @fortran argument in  required int     ispn                   Spin component.
   @fortran argument out required double  band_occupancies       Array of band occupancies.
   @fortran end */
void sirius_get_band_occupancies(void*  const* ks_handler__,
                                 int    const* ik__,
                                 int    const* ispn__,
                                 double*       band_occupancies__)
{
    auto& ks = get_ks(ks_handler__);
    int ik = *ik__ - 1;
    for (int i = 0; i < ks.ctx().num_bands(); i++) {
        band_occupancies__[i] = ks[ik]->band_occupancy(i, *ispn__);
    }
}

/* @fortran begin function void sirius_get_d_operator_matrix       Get D-operator matrix
   @fortran argument in  required void*   handler                  Simulation context handler.
   @fortran argument in  required int     ia                       Global index of atom.
   @fortran argument in  required int     ispn                     Spin component.
   @fortran argument out required double  d_mtrx                   D-matrix.
   @fortran argument in  required int     ld                       Leading dimention of D-matrix.
   @fortran end */
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

/* @fortran begin function void sirius_set_d_operator_matrix       Set D-operator matrix
   @fortran argument in  required void*   handler                  Simulation context handler.
   @fortran argument in  required int     ia                       Global index of atom.
   @fortran argument in  required int     ispn                     Spin component.
   @fortran argument out required double  d_mtrx                   D-matrix.
   @fortran argument in  required int     ld                       Leading dimention of D-matrix.
   @fortran end */
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

/* @fortran begin function void sirius_set_q_operator_matrix    Set Q-operator matrix
   @fortran argument in  required void*   handler               Simulation context handler.
   @fortran argument in  required string  label                 Atom type label.
   @fortran argument out required double  q_mtrx                Q-matrix.
   @fortran argument in  required int     ld                    Leading dimention of Q-matrix.
   @fortran end */
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

/* @fortran begin function void sirius_get_q_operator_matrix    Get Q-operator matrix
   @fortran argument in  required void*   handler               Simulation context handler.
   @fortran argument in  required string  label                 Atom type label.
   @fortran argument out required double  q_mtrx                Q-matrix.
   @fortran argument in  required int     ld                    Leading dimention of Q-matrix.
   @fortran end */
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

/* @fortran begin function void sirius_get_density_matrix       Get all components of complex density matrix.
   @fortran argument in  required void*   handler               DFT ground state handler.
   @fortran argument in  required int     ia                    Global index of atom.
   @fortran argument out required complex dm                    Complex density matrix.
   @fortran argument in  required int     ld                    Leading dimention of the density matrix.
   @fortran end */
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

/* @fortran begin function void sirius_set_density_matrix       Set all components of complex density matrix.
   @fortran argument in  required void*   handler               DFT ground state handler.
   @fortran argument in  required int     ia                    Global index of atom.
   @fortran argument out required complex dm                    Complex density matrix.
   @fortran argument in  required int     ld                    Leading dimention of the density matrix.
   @fortran end */
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

/* @fortran begin function void sirius_get_energy    Get one of the total energy components.
   @fortran argument in  required void*   handler    DFT ground state handler.
   @fortran argument in  required string  label      Label of the energy component to get.
   @fortran argument out required double  energy     Total energy component.
   @fortran end */
void sirius_get_energy(void* const* handler__,
                       char  const* label__,
                       double*      energy__)
{
    auto& gs = get_gs(handler__);

    auto& kset = gs.k_point_set();
    auto& ctx = kset.ctx();
    auto& unit_cell = kset.unit_cell();
    auto& potential = gs.potential();
    auto& density = gs.density();

    std::string label(label__);

    std::map<std::string, std::function<double()>> func = {
        {"total",   [&](){ return sirius::total_energy(ctx, kset, density, potential, gs.ewald_energy()); }},
        {"evalsum", [&](){ return sirius::eval_sum(unit_cell, kset); }},
        {"exc",     [&](){ return sirius::energy_exc(density, potential); }},
        {"vxc",     [&](){ return sirius::energy_vxc(density, potential); }},
        {"bxc",     [&](){ return sirius::energy_bxc(density, potential, ctx.num_mag_dims()); }},
        {"veff",    [&](){ return sirius::energy_veff(density, potential); }},
        {"vloc",    [&](){ return sirius::energy_vloc(density, potential); }},
        {"vha",     [&](){ return sirius::energy_vha(potential); }},
        {"enuc",    [&](){ return sirius::energy_enuc(ctx, potential); }},
        {"kin",     [&](){ return sirius::energy_kin(ctx, kset, density, potential); }}};

    try {
        *energy__ = func.at(label)();
    } catch(...) {
        std::stringstream s;
        s << "[sirius_get_energy] wrong label: " << label;
        TERMINATE(s);
    }
}

/* @fortran begin function void sirius_get_forces     Get one of the total force components.
   @fortran argument in  required void*   handler     DFT ground state handler.
   @fortran argument in  required string  label       Label of the force component to get.
   @fortran argument out required double  forces      Total force component for each atom.
   @fortran end */
void sirius_get_forces(void* const* handler__,
                       char  const* label__,
                       double*      forces__)
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

    try {
        get_forces((forces.*func.at(label))());
    } catch(...) {
        std::stringstream s;
        s << "wrong label (" << label <<") for the component of forces";
        TERMINATE(s);
    }
}

/* @fortran begin function void sirius_get_stress_tensor     Get one of the stress tensor components.
   @fortran argument in  required void*   handler            DFT ground state handler.
   @fortran argument in  required string  label              Label of the stress tensor component to get.
   @fortran argument out required double  stress_tensor      Component of the total stress tensor.
   @fortran end */
void sirius_get_stress_tensor(void* const* handler__,
                              char  const* label__,
                              double*      stress_tensor__)
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

    matrix3d<double> s;

    try {
        s = ((stress_tensor.*func.at(label))());
    } catch(...) {
        std::stringstream s;
        s << "wrong label (" << label <<") for the component of stress tensor";
        TERMINATE(s);
    }

    for (int mu = 0; mu < 3; mu++) {
        for (int nu = 0; nu < 3; nu++) {
            stress_tensor__[nu + mu * 3] = s(mu, nu);
        }
    }
}

/* @fortran begin function int sirius_get_num_beta_projectors     Get the number of beta-projectors for an atom type.
   @fortran argument in  required void*   handler                  Simulation context handler.
   @fortran argument in  required string  label                    Atom type label.
   @fortran end */
int sirius_get_num_beta_projectors(void* const* handler__,
                                   char  const* label__)
{
    auto& sim_ctx = get_sim_ctx(handler__);

    auto& type = sim_ctx.unit_cell().atom_type(std::string(label__));
    return type.mt_basis_size();
}

/* @fortran begin function void sirius_get_q_operator Get plane-wave coefficients of Q-operator
   @fortran argument in   required void*   handler    Simulation context handler.
   @fortran argument in   required string  label      Label of the atom type.
   @fortran argument in   required int     xi1        First index of beta-projector atomic function.
   @fortran argument in   required int     xi2        Second index of beta-projector atomic function.
   @fortran argument in   required int     ngv        Number of G-vectors.
   @fortran argument in   required int     gvl        G-vectors in lattice coordinats.
   @fortran argument out  required complex q_pw       Plane-wave coefficients of Q augmentation operator.
   @fortran end */
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
    sim_ctx.comm().allgather(q_pw.data(), sim_ctx.gvec().offset(), sim_ctx.gvec().count());

    for (int i = 0; i < *ngv__; i++) {
        vector3d<int> G(gvl(0, i), gvl(1, i), gvl(2, i));

        auto gvc = sim_ctx.unit_cell().reciprocal_lattice_vectors() * vector3d<double>(G[0], G[1], G[2]);
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
            auto gvc = sim_ctx.unit_cell().reciprocal_lattice_vectors() * vector3d<double>(G[0], G[1], G[2]);
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

/* @fortran begin function void sirius_get_wave_functions     Get wave-functions.
   @fortran argument in   required void*   ks_handler         K-point set handler.
   @fortran argument in   required int     ik                 Global index of k-point
   @fortran argument in   required int     ispn               Spin index.
   @fortran argument in   required int     npw                Local number of G+k vectors.
   @fortran argument in   required int     gvec_k             List of G-vectors.
   @fortran argument out  required complex evc                Wave-functions.
   @fortran argument in   required int     ld1                Leading dimention of evc array.
   @fortran argument in   required int     ld2                Second dimention of evc array.
   @fortran end */
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
    kset.comm().allgather(&jrank, rank_with_jk.data(), kset.comm().rank(), 1);

    std::vector<int> jk_of_rank(kset.comm().size());
    kset.comm().allgather(&jk, jk_of_rank.data(), kset.comm().rank(), 1);

    std::vector<int> jspn_of_rank(kset.comm().size());
    kset.comm().allgather(&jspn, jspn_of_rank.data(), kset.comm().rank(), 1);

    int my_rank = kset.comm().rank();

    std::vector<int> igmap;

    auto gvec_mapping = [&](Gvec const& gkvec)
    {
        std::vector<int> igm(*npw__, std::numeric_limits<int>::max());

        mdarray<int, 2> gvec_k(gvec_k__, 3, *npw__);

        for (int ig = 0; ig < *npw__; ig++) {
            /* G vector of host code */
            auto gvc = kset.ctx().unit_cell().reciprocal_lattice_vectors() *
                       (vector3d<double>(gvec_k(0, ig), gvec_k(1, ig), gvec_k(2, ig)) + gkvec.vk());
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

            /* if this is a rank wich need jk or a rank which stores jk */
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

                    /* make a check of send-recieve sizes */
                    if (true) {
                        int send_size;
                        if (my_rank == rank_with_jk[r]) {
                            auto kp = kset[this_jk];
                            int gkvec_count = kp->gkvec().count();
                            send_size = gkvec_count * sim_ctx.num_bands();
                            req = kset.comm().isend(&send_size, 1, r, tag);
                        }
                        if (my_rank == r) {
                            int gkvec_count = gkvec.count();
                            kset.comm().recv(&send_size, 1, rank_with_jk[r], tag);
                            if (send_size != gkvec_count * sim_ctx.num_bands()) {
                                std::stringstream s;
                                s << "wrong send-recieve buffer sizes\n"
                                  << "     send size   : " << send_size << "\n"
                                  << "  recieve size   : " << gkvec_count * sim_ctx.num_bands() << "\n"
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
                        req = kset.comm().isend(&kp->spinor_wave_functions().pw_coeffs(s).prime(0, 0), gkvec_count * sim_ctx.num_bands(), r, tag);
                    }
                    if (my_rank == r) {
                        int gkvec_count = gkvec.count();
                        int gkvec_offset = gkvec.offset();
                        /* recieve the array with wave-functions */
                        kset.comm().recv(&wf->pw_coeffs(0).prime(0, 0), gkvec_count * sim_ctx.num_bands(), rank_with_jk[r], tag);
                        std::vector<double_complex> wf_tmp(gkvec.num_gvec());
                        /* store wave-functions */
                        for (int i = 0; i < sim_ctx.num_bands(); i++) {
                            /* gather full column of PW coefficients */
                            sim_ctx.comm_band().allgather(&wf->pw_coeffs(0).prime(0, i), wf_tmp.data(), gkvec_offset, gkvec_count);
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

/* @fortran begin function double sirius_get_radial_integral     Get value of the radial integral.
   @fortran argument in   required void*   handler               Simulation context handler.
   @fortran argument in   required string  atom_type             Label of the atom type.
   @fortran argument in   required string  label                 Label of the radial integral.
   @fortran argument in   required double  q                     Length of the reciprocal wave-vector.
   @fortran argument in   required int     idx                   Index of the radial integral.
   @fortran argument in   optional int     l                     Orbital quantum number (for Q-radial integrals).
   @fortran end */
double sirius_get_radial_integral(void*  const* handler__,
                                  char   const* atom_type__,
                                  char   const* label__,
                                  double const* q__,
                                  int    const* idx__,
                                  int    const* l__)
{
    auto& sim_ctx = get_sim_ctx(handler__);

    auto& type = sim_ctx.unit_cell().atom_type(std::string(atom_type__));

    std::string label(label__);

    if (label == "aug") {
        if (l__ == nullptr) {
            TERMINATE("orbital quantum number must be provided for augmentation operator radial integrals");
        }
        return sim_ctx.aug_ri().value<int, int, int>(*idx__ - 1, *l__, type.id(), *q__);
    } else if (label == "aug_dj") {
        if (l__ == nullptr) {
            TERMINATE("orbital quantum number must be provided for augmentation operator radial integrals");
        }
        return sim_ctx.aug_ri_djl().value<int, int, int>(*idx__ - 1, *l__, type.id(), *q__);
    } else if (label == "beta") {
        return sim_ctx.beta_ri().value<int, int>(*idx__ - 1, type.id(), *q__);
    } else if (label == "beta_dj") {
        return sim_ctx.beta_ri_djl().value<int, int>(*idx__ - 1, type.id(), *q__);
    } else {
        TERMINATE("wrong label of radial integral");
        return 0.0; // make compiler happy
    }
}

/* @fortran begin function void sirius_calculate_hubbard_occupancies  Compute occupation matrix.
   @fortran argument in required void* handler                        Ground state handler.
   @fortran end */
void sirius_calculate_hubbard_occupancies(void* const* handler__)
{
    auto& gs = get_gs(handler__);
    gs.potential().U().hubbard_compute_occupation_numbers(gs.k_point_set());
}


/* @fortran begin function void sirius_set_hubbard_occupancies          Set occupation matrix for LDA+U.
   @fortran argument in    required void* handler                       Ground state handler.
   @fortran argument inout required complex occ                         Occupation matrix.
   @fortran argument in    required int     ld                          Leading dimensions of the occupation matrix.
   @fortran end */
void sirius_set_hubbard_occupancies(void* const* handler__,
                                    std::complex<double>*      occ__,
                                    int   const *ld__)
{
    auto& gs = get_gs(handler__);
    gs.potential().U().access_hubbard_occupancies("set", occ__, ld__);
}

/* @fortran begin function void sirius_get_hubbard_occupancies          Get occupation matrix for LDA+U.
   @fortran argument in    required void* handler                       Ground state handler.
   @fortran argument inout required complex occ                         Occupation matrix.
   @fortran argument in    required int     ld                          Leading dimensions of the occupation matrix.
   @fortran end */
void sirius_get_hubbard_occupancies(void* const* handler__,
                                    std::complex<double>*      occ__,
                                    int   const *ld__)
{
    auto& gs = get_gs(handler__);
    gs.potential().U().access_hubbard_occupancies("get", occ__, ld__);
}

/* @fortran begin function void sirius_set_hubbard_potential              Set LDA+U potential matrix.
   @fortran argument in    required void* handler                         Ground state handler.
   @fortran argument inout required complex pot                           Potential correction matrix.
   @fortran argument in    required int    ld                             Leading dimensions of the matrix.
   @fortran end */
void sirius_set_hubbard_potential(void* const* handler__,
                                  std::complex<double>*      pot__,
                                  int   const *ld__)
{
    auto& gs = get_gs(handler__);
    gs.potential().U().access_hubbard_potential("set", pot__, ld__);
}


/* @fortran begin function void sirius_get_hubbard_potential              Set LDA+U potential matrix.
   @fortran argument in    required void* handler                         Ground state handler.
   @fortran argument inout required complex pot                           Potential correction matrix.
   @fortran argument in    required int    ld                             Leading dimensions of the matrix.
   @fortran end */
void sirius_get_hubbard_potential(void* const* handler__,
                                  std::complex<double>*      pot__,
                                  int   const *ld__)
{
    auto& gs = get_gs(handler__);
    gs.potential().U().access_hubbard_potential("get", pot__, ld__);
}

/* @fortran begin function void sirius_add_atom_type_aw_descriptor    Add descriptor of the augmented wave radial function.
   @fortran argument in    required void*  handler                    Simulation context handler.
   @fortran argument in    required string label                      Atom type label.
   @fortran argument in    required int    n                          Principal quantum number.
   @fortran argument in    required int    l                          Orbital quantum number.
   @fortran argument in    required double enu                        Linearization energy.
   @fortran argument in    required int    dme                        Order of energy derivative.
   @fortran argument in    required bool   auto_enu                   True if automatic search of linearization energy is allowed for this radial solution.
   @fortran end */
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

/* @fortran begin function void sirius_add_atom_type_lo_descriptor    Add descriptor of the local orbital radial function.
   @fortran argument in    required void*  handler                    Simulation context handler.
   @fortran argument in    required string label                      Atom type label.
   @fortran argument in    required int    ilo                        Index of the local orbital to which the descriptro is added.
   @fortran argument in    required int    n                          Principal quantum number.
   @fortran argument in    required int    l                          Orbital quantum number.
   @fortran argument in    required double enu                        Linearization energy.
   @fortran argument in    required int    dme                        Order of energy derivative.
   @fortran argument in    required bool   auto_enu                   True if automatic search of linearization energy is allowed for this radial solution.
   @fortran end */
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

/* @fortran begin function void sirius_set_atom_type_configuration   Set configuration of atomic levels.
   @fortran argument in required void*  handler    Simulation context handler.
   @fortran argument in required string label      Atom type label.
   @fortran argument in required int    n          Principal quantum number.
   @fortran argument in required int    l          Orbital quantum number.
   @fortran argument in required int    k          kappa (used in relativistic solver).
   @fortran argument in required double occupancy  Level occupancy.
   @fortran argument in required bool   core       Tru if this is a core state.
   @fortran end */
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

/* @fortran begin function void sirius_generate_coulomb_potential    Generate Coulomb potential by solving Poisson equation
   @fortran argument in required void*   handler      Ground state handler
   @fortran argument in required bool    is_local_rg  true if regular grid pointer is local
   @fortran argument out required double vclmt        Muffin-tin part of potential
   @fortran argument out required double vclrg        Regular-grid part of potential
   @fortran end */
void sirius_generate_coulomb_potential(void* const* handler__,
                                       bool  const* is_local_rg__,
                                       double*      vclmt__,
                                       double*      vclrg__)
{
    auto& gs = get_gs(handler__);

    gs.density().rho().fft_transform(-1);
    gs.potential().poisson(gs.density().rho());
    gs.potential().hartree_potential().copy_to(vclmt__, vclrg__, *is_local_rg__);
}

/* @fortran begin function void sirius_generate_xc_potential    Generate XC potential using LibXC
   @fortran argument in required void*   handler     Ground state handler
   @fortran argument in required bool    is_local_rg true if regular grid pointer is local
   @fortran argument out required double vxcmt       Muffin-tin part of potential
   @fortran argument out required double vxcrg       Regular-grid part of potential
   @fortran argument out optional double bxcmt_x     Muffin-tin part of effective magentic field (x-component)
   @fortran argument out optional double bxcmt_y     Muffin-tin part of effective magentic field (y-component)
   @fortran argument out optional double bxcmt_z     Muffin-tin part of effective magentic field (z-component)
   @fortran argument out optional double bxcrg_x     Regular-grid part of effective magnetic field (x-component)
   @fortran argument out optional double bxcrg_y     Regular-grid part of effective magnetic field (y-component)
   @fortran argument out optional double bxcrg_z     Regular-grid part of effective magnetic field (z-component)
   @fortran end */
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

/* @fortran begin function void sirius_get_kpoint_inter_comm  Get communicator which is used to split k-points
   @fortran argument in required void* handler   Simulation context handler
   @fortran argument out required int fcomm      Fortran communicator
   @fortran end */
void sirius_get_kpoint_inter_comm(void * const* handler__,
                                  int*          fcomm__)
{
    auto& sim_ctx = get_sim_ctx(handler__);
    *fcomm__ = MPI_Comm_c2f(sim_ctx.comm_k().mpi_comm());
}

/* @fortran begin function void sirius_get_kpoint_inner_comm  Get communicator which is used to parallise band problem
   @fortran argument in required void* handler   Simulation context handler
   @fortran argument out required int fcomm      Fortran communicator
   @fortran end */
void sirius_get_kpoint_inner_comm(void * const* handler__,
                                  int*          fcomm__)
{
    auto& sim_ctx = get_sim_ctx(handler__);
    *fcomm__ = MPI_Comm_c2f(sim_ctx.comm_band().mpi_comm());
}

/* @fortran begin function void sirius_get_fft_comm  Get communicator which is used to parallise FFT
   @fortran argument in required void* handler   Simulation context handler
   @fortran argument out required int fcomm      Fortran communicator
   @fortran end */
void sirius_get_fft_comm(void * const* handler__,
                         int*          fcomm__)
{
    auto& sim_ctx = get_sim_ctx(handler__);
    *fcomm__ = MPI_Comm_c2f(sim_ctx.comm_fft().mpi_comm());
}

/* @fortran begin function int sirius_get_num_gvec  Get total number of G-vectors
   @fortran argument in required void* handler      Simulation context handler
   @fortran end */
int sirius_get_num_gvec(void* const* handler__)
{
    auto& sim_ctx = get_sim_ctx(handler__);
    return sim_ctx.gvec().num_gvec();
}

/* @fortran begin function void sirius_get_gvec_arrays   Get G-vector arrays.
   @fortran argument in required void*  handler          Simulation context handler
   @fortran argument in optional int    gvec             G-vectors in lattice coordinates.
   @fortran argument in optional double gvec_cart        G-vectors in Cartesian coordinates.
   @fortran argument in optional double gvec_len         Length of G-vectors.
   @fortran argument in optional int    index_by_gvec    G-vector index by lattice coordinates.
   @fortran end */
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

        mdarray<int, 3> index_by_gvec(index_by_gvec__, d0, d1, d2);
        std::fill(index_by_gvec.at(memory_t::host), index_by_gvec.at(memory_t::host) + index_by_gvec.size(), -1);

        for (int ig = 0; ig < sim_ctx.gvec().num_gvec(); ig++) {
            auto G = sim_ctx.gvec().gvec(ig);
            index_by_gvec(G[0], G[1], G[2]) = ig + 1;
        }
    }
}

/* @fortran begin function int sirius_get_num_fft_grid_points Get local number of FFT grid points.
   @fortran argument in required void* handler                Simulation context handler
   @fortran end */
int sirius_get_num_fft_grid_points(void* const* handler__)
{
    auto& sim_ctx = get_sim_ctx(handler__);
    return sim_ctx.spfft().local_slice_size();
}

/* @fortran begin function void sirius_get_fft_index   Get mapping between G-vector index and FFT index
   @fortran argument in  required void* handler        Simulation context handler
   @fortran argument out required int   fft_index      Index inside FFT buffer
   @fortran end */
void sirius_get_fft_index(void* const* handler__,
                          int*         fft_index__)
{
    auto& sim_ctx = get_sim_ctx(handler__);
    for (int ig = 0; ig < sim_ctx.gvec().num_gvec(); ig++) {
        auto G = sim_ctx.gvec().gvec(ig);
        fft_index__[ig] = sim_ctx.fft_grid().index_by_freq(G[0], G[1], G[2]) + 1;
    }
}

/* @fortran begin function int sirius_get_max_num_gkvec    Get maximum number of G+k vectors across all k-points in the set
   @fortran argument in required void*       ks_handler    K-point set handler.
   @fortran end */
int sirius_get_max_num_gkvec(void* const* ks_handler__)
{
    auto& ks = get_ks(ks_handler__);
    return ks.max_num_gkvec();
}

/* @fortran begin function void sirius_get_gkvec_arrays  Get all G+k vector related arrays
   @fortran argument in required void*   ks_handler    K-point set handler.
   @fortran argument in required int     ik            Global index of k-point
   @fortran argument out required int    num_gkvec     Number of G+k vectors.
   @fortran argument out required int    gvec_index    Index of the G-vector part of G+k vector.
   @fortran argument out required double gkvec         G+k vectors in fractional coordinates.
   @fortran argument out required double gkvec_cart    G+k vectors in Cartesian coordinates.
   @fortran argument out required double gkvec_len     Length of G+k vectors.
   @fortran argument out required double gkvec_tp      Theta and Phi angles of G+k vectors.
   @fortran end */
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

/* @fortran begin function void sirius_get_step_function  Get the unit-step function.
   @fortran argument in  required void* handler        Simulation context handler
   @fortran argument out required complex cfunig       Plane-wave coefficients of step function.
   @fortran argument out required double  cfunrg       Values of the step function on the regular grid.
   @fortran end */
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

/* @fortran begin function void sirius_get_vha_el   Get electronic part of Hartree potential at atom origins.
   @fortran argument in required void* handler      DFT ground state handler.
   @fortran argument out required double vha_el     Electronic part of Hartree potential at each atom's origin.
   @fortran end */
void sirius_get_vha_el(void* const* handler__,
                       double*      vha_el__)
{
    auto& gs = get_gs(handler__);
    for (int ia = 0; ia < gs.ctx().unit_cell().num_atoms(); ia++) {
        vha_el__[ia] = gs.potential().vha_el(ia);
    }
}

/* @fortran begin function void sirius_set_h_radial_integrals   Set LAPW Hamiltonian radial integrals.
   @fortran argument in required void*  handler    Simulation context handler.
   @fortran argument in required int    ia         Index of atom.
   @fortran argument in required int    lmmax      Number of lm-component of the potential.
   @fortran argument in required double val        Values of the radial integrals.
   @fortran argument in optional int    l1         1st index of orbital quantum number.
   @fortran argument in optional int    o1         1st index of radial function order for l1.
   @fortran argument in optional int    ilo1       1st index or local orbital.
   @fortran argument in optional int    l2         2nd index of orbital quantum number.
   @fortran argument in optional int    o2         2nd index of radial function order for l2.
   @fortran argument in optional int    ilo2       2nd index or local orbital.
   @fortran end */
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

/* @fortran begin function void sirius_set_o_radial_integral   Set LAPW overlap radial integral.
   @fortran argument in required void*  handler    Simulation context handler.
   @fortran argument in required int    ia         Index of atom.
   @fortran argument in required double val        Value of the radial integral.
   @fortran argument in required int    l          Orbital quantum number.
   @fortran argument in optional int    o1         1st index of radial function order.
   @fortran argument in optional int    ilo1       1st index or local orbital.
   @fortran argument in optional int    o2         2nd index of radial function order.
   @fortran argument in optional int    ilo2       2nd index or local orbital.
   @fortran end */
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

/* @fortran begin function void sirius_set_o1_radial_integral   Set a correction to LAPW overlap radial integral.
   @fortran argument in required void*  handler    Simulation context handler.
   @fortran argument in required int    ia         Index of atom.
   @fortran argument in required double val        Value of the radial integral.
   @fortran argument in optional int    l1         1st index of orbital quantum number.
   @fortran argument in optional int    o1         1st index of radial function order for l1.
   @fortran argument in optional int    ilo1       1st index or local orbital.
   @fortran argument in optional int    l2         2nd index of orbital quantum number.
   @fortran argument in optional int    o2         2nd index of radial function order for l2.
   @fortran argument in optional int    ilo2       2nd index or local orbital.
   @fortran end */
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

/* @fortran begin function void sirius_set_radial_function   Set LAPW radial functions
   @fortran argument in required void*  handler              Simulation context handler.
   @fortran argument in required int    ia                   Index of atom.
   @fortran argument in required int    deriv_order          Radial derivative order.
   @fortran argument in required double f                    Values of the radial function.
   @fortran argument in optional int    l                    Orbital quantum number.
   @fortran argument in optional int    o                    Order of radial function for l.
   @fortran argument in optional int    ilo                  Local orbital index.
   @fortran end */
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

/* @fortran begin function void sirius_get_radial_function   Get LAPW radial functions
   @fortran argument in required void*  handler              Simulation context handler.
   @fortran argument in required int    ia                   Index of atom.
   @fortran argument in required int    deriv_order          Radial derivative order.
   @fortran argument out required double f                   Values of the radial function.
   @fortran argument in optional int    l                    Orbital quantum number.
   @fortran argument in optional int    o                    Order of radial function for l.
   @fortran argument in optional int    ilo                  Local orbital index.
   @fortran end */
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

/* @fortran begin function void sirius_set_equivalent_atoms   Set equivalent atoms.
   @fortran argument in required void*  handler               Simulation context handler.
   @fortran argument in required int    equivalent_atoms      Array with equivalent atom IDs.
   @fortran end */
void sirius_set_equivalent_atoms(void* const* handler__,
                                 int*         equivalent_atoms__)
{
    auto& sim_ctx = get_sim_ctx(handler__);
    sim_ctx.unit_cell().set_equivalent_atoms(equivalent_atoms__);
}

/* @fortran begin function void sirius_update_atomic_potential   Set the new spherical potential.
   @fortran argument in required void*  handler                  Ground state handler.
   @fortran end */
void sirius_update_atomic_potential(void* const* handler__)
{
    auto& gs = get_gs(handler__);
    gs.potential().update_atomic_potential();
}


/* @fortran begin function void sirius_option_get_length     return the number of options in a given section
   @fortran argument in  required string  section            name of the seciton
   @fortran argument out required int     length             number of options contained in  the section
   @fortran end */

void sirius_option_get_length(char const* section__, int *length__)
{
    auto const& parser = sirius::get_options_dictionary();

    auto section = std::string(section__);
    std::transform(section.begin(), section.end(), section.begin(), ::tolower);

    *length__ = parser[section].size();
}

/* @fortran begin function void sirius_option_get_name_and_type    Return the name and a type of an option from its index.
   @fortran argument in  required string  section                  Name of the section.
   @fortran argument in  required int     elem                     Index of the option.
   @fortran argument out required string  key_name                 Name of the option.
   @fortran argument out required int     type                     Type of the option (real, integer, boolean, string).
   @fortran end */

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
            std::memcpy(key_name__, el.key().c_str(), el.key().size());
        }
        elem++;
    }
}

/* @fortran begin function void sirius_option_get_description_usage  return the description and usage of a given option
   @fortran argument in  required string  section                    name of the section
   @fortran argument in  required string  name                       name of the option
   @fortran argument out required string  desc                       description of the option
   @fortran argument out required string  usage                      how to use the option
   @fortran end */
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

/* @fortran begin function void sirius_option_get_int                return the default value of the option
   @fortran argument in  required string  section                    name of the section of interest
   @fortran argument in  required string  name                       name of the element
   @fortran argument out required int     default_value              table containing the default values (if vector)
   @fortran argument out required int     length                     length of the table containing the default values
   @fortran end */

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

/* @fortran begin function void sirius_option_get_double                     return the default value of the option
   @fortran argument in  required string  section                            name of the section of interest
   @fortran argument in  required string  name                               name of the element
   @fortran argument out required double default_value                       table containing the default values (if vector)
   @fortran argument out required int    length                              length of the table containing the default values
   @fortran end */
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

/* @fortran begin function void sirius_option_get_logical                    return the default value of the option
   @fortran argument in  required string  section                            name of the section
   @fortran argument in  required string  name                               name of the element
   @fortran argument out required bool   default_value                       table containing the default values
   @fortran argument out required int    length                              length of the table containing the default values
   @fortran end */
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

/* @fortran begin function void sirius_option_get_string                     return the default value of the option
   @fortran argument in  required string  section                            name of the section
   @fortran argument in  required string  name                               name of the option
   @fortran argument out required string  default_value                      table containing the string
   @fortran end */
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

/* @fortran begin function void sirius_option_get_number_of_possible_values  return the number of possible values for a string option
   @fortran argument in  required string  section                            name of the section
   @fortran argument in  required string  name                               name of the option
   @fortran argument out required int   num_                                 number of elements
   @fortran end */
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

/* @fortran begin function void sirius_option_string_get_value               return the possible values for a string parameter
   @fortran argument in  required string  section                            name of the section
   @fortran argument in  required string  name                               name of the option
   @fortran argument in  required int    elem_                               index of the value
   @fortran argument out required string  value_n                            string containing the value
   @fortran end */
void sirius_option_string_get_value(char* section, char * name, int *elem_, char *value_n)
{
    const json &parser =  sirius::get_options_dictionary();

    // ugly as hell but fortran is a piece of ....
    for ( char *p = section; *p; p++) *p = tolower(*p);
    for ( char *p = name; *p; p++) *p = tolower(*p);

    // for string I do not consider a table of several strings to be returned. I
    // need to specialize however the possible values that the string can have
    if (parser[section][name].count("possible_values")) {
        auto tmp = parser[section][name]["possible_values"].get<std::vector<std::string>>();
        // BIG BIG BIG WARNINNG. THE STRING IS NOT null terminated because
        // fortran does not understand the concept of null terminated string

        std::memcpy(value_n, tmp[*elem_].c_str(), tmp[*elem_].size());
    }
}

/* @fortran begin function void sirius_option_get_section_name               return the name of a given section
   @fortran argument in  required int     elem_                              index of the section
   @fortran argument out  required string  section_name                      name of the section
   @fortran end */
void sirius_option_get_section_name(int *elem, char *section_name)
{
    const json &dict = sirius::get_options_dictionary();
    int elem_ = 0;

    for (auto& el : dict.items())
    {
        if (elem_ == *elem) {
            std::memcpy(section_name, el.key().c_str(), el.key().size());
            break;
        }
        elem_++;
    }
}

/* @fortran begin function void sirius_option_get_number_of_sections         return the number of sections
   @fortran argument out  required int     length                            number of sections
   @fortran end */
void sirius_option_get_number_of_sections(int *length)
{
    const json &parser =  sirius::get_options_dictionary();
    *length = parser.size();
}


/* @fortran begin function void sirius_option_set_int                        set the value of the option name in a  (internal) json dictionary
   @fortran argument in  required void*  handler                             Simulation context handler.
   @fortran argument in  required string  section                            string containing the options in json format
   @fortran argument in  required string  name                               name of the element to pick
   @fortran argument in required int    default_values                       table containing the values
   @fortran argument in required int    length                               length of the table containing the values
   @fortran end */
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

/* @fortran begin function void sirius_option_set_double                     set the value of the option name in a (internal) json dictionary
   @fortran argument in  required void*  handler                             Simulation context handler.
   @fortran argument in  required string  section                            name of the section
   @fortran argument in  required string  name                               name of the element to pick
   @fortran argument in required double default_values                       table containing the values
   @fortran argument in required int    length                               length of the table containing the values
   @fortran end */
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

/* @fortran begin function void sirius_option_set_logical                    set the value of the option name in a  (internal) json dictionary
   @fortran argument in  required void*  handler                             Simulation context handler.
   @fortran argument in  required string  section                            name of the section
   @fortran argument in  required string  name                               name of the element to pick
   @fortran argument in required int   default_values                        table containing the values
   @fortran argument in required int    length                               length of the table containing the values
   @fortran end */
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
            for (int s = 0; s < *length; s++)
                v[s] = (default_values[s] == 1);
            conf_dict[section][name] = v;
        } else {
            conf_dict[section][name] = (*default_values == 1);
        }
    }
}

/* @fortran begin function void sirius_option_set_string                     set the value of the option name in a  (internal) json dictionary
   @fortran argument in  required void*  handler                             Simulation context handler.
   @fortran argument in  required string  section                            name of the section
   @fortran argument in  required string  name                               name of the element to pick
   @fortran argument in required string   default_values                     table containing the values
   @fortran end */
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
        for ( char *p = default_values; *p; p++) *p = tolower(*p);
        std::string st = default_values;
        conf_dict[section][name] = st;
    }
}

/* @fortran begin function void sirius_option_add_string_to                  add a string value to the option in the json dictionary
   @fortran argument in  required void*  handler                             Simulation context handler.
   @fortran argument in  required string  section                            name of the section
   @fortran argument in  required string  name                               name of the element to pick
   @fortran argument in required string   default_values                     string to be added
   @fortran end */
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

/* @fortran begin function void sirius_dump_runtime_setup                    Dump the runtime setup in a file.
   @fortran argument in  required void*  handler                             Simulation context handler.
   @fortran argument in  required string filename                            String containing the name of the file.
   @fortran end */
void sirius_dump_runtime_setup(void* const* handler__, char *filename)
{
    auto& sim_ctx = get_sim_ctx(handler__);
    std::ofstream fi(filename);
    json &conf_dict = sim_ctx.get_runtime_options_dictionary();
    fi << conf_dict;
    fi.close();
}

/* @fortran begin function void sirius_get_fv_eigen_vectors         Get the first-variational eigen vectors
   @fortran argument in  required void* handler                     K-point set handler
   @fortran argument in  required int   ik                          Global index of the k-point
   @fortran argument out required complex fv_evec                   Output first-variational eigenvector array
   @fortran argument in  required int    ld                         Leading dimension of fv_evec
   @fortran argument in  required int    num_fv_states              Number of first-variational states
   @fortran end */
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

/* @fortran begin function void sirius_get_fv_eigen_values          Get the first-variational eigen values
   @fortran argument in  required void*  handler                    K-point set handler
   @fortran argument in  required int    ik                         Global index of the k-point
   @fortran argument out required double fv_eval                    Output first-variational eigenvector array
   @fortran argument in  required int    num_fv_states              Number of first-variational states
   @fortran end */
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

/* @fortran begin function void sirius_get_sv_eigen_vectors         Get the second-variational eigen vectors
   @fortran argument in  required void*   handler                   K-point set handler
   @fortran argument in  required int     ik                        Global index of the k-point
   @fortran argument out required complex sv_evec                   Output second-variational eigenvector array
   @fortran argument in  required int     num_bands                 Number of second-variational bands.
   @fortran end */
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

/* @fortran begin function void sirius_set_rg_values          Set the values of the function on the regular grid.
   @fortran argument in  required void*  handler              DFT ground state handler.
   @fortran argument in  required string label                Label of the function.
   @fortran argument in  required int    grid_dims            Dimensions of the FFT grid.
   @fortran argument in  required int    local_box_origin     Coordinates of the local box origin for each MPI rank
   @fortran argument in  required int    local_box_size       Dimensions of the local box for each MPI rank.
   @fortran argument in  required int    fcomm                Fortran communicator used to partition FFT grid into local boxes.
   @fortran argument in  required double values               Values of the function (local buffer for each MPI rank).
   @fortran argument in  optional bool   transform_to_pw      If true, transform function to PW domain.
   @fortran end */
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

/* @fortran begin function void sirius_get_rg_values          Get the values of the function on the regular grid.
   @fortran argument in  required void*  handler              DFT ground state handler.
   @fortran argument in  required string label                Label of the function.
   @fortran argument in  required int    grid_dims            Dimensions of the FFT grid.
   @fortran argument in  required int    local_box_origin     Coordinates of the local box origin for each MPI rank
   @fortran argument in  required int    local_box_size       Dimensions of the local box for each MPI rank.
   @fortran argument in  required int    fcomm                Fortran communicator used to partition FFT grid into local boxes.
   @fortran argument out required double values               Values of the function (local buffer for each MPI rank).
   @fortran argument in  optional bool   transform_to_rg      If true, transform function to regular grid before fetching the values.
   @fortran end */
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


/* @fortran begin function void sirius_get_total_magnetization  Get the total magnetization of the system.
   @fortran argument in  required void*  handler                DFT ground state handler.
   @fortran argument out required double mag                    3D magnetization vector (x,y,z components).
   @fortran end */
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

/* @fortran begin function void sirius_get_num_kpoints         Get the total number of kpoints
   @fortran argument in   required void* handler               Kpoint set handler
   @fortran argument out  required int   num_kpoints           number of kpoints in the set
   @fortran argument out  optional int   error_code            Error code.
   @fortran end */

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

/* @fortran begin function void sirius_get_num_bands         Get the number of computed bands
   @fortran argument in   required void* handler             Simulation context handler.
   @fortran argument out  required int   num_kpoints         Number of kpoints in the set
   @fortran argument out  optional int   error_code          Error code.
   @fortran end */
void sirius_get_num_bands(void* const* handler__,
                          int *num_bands__,
                          int *error_code__)
{ // TODO: already exists in sirius_get_parameters
    call_sirius([&]()
    {
        auto& sim_ctx = get_sim_ctx(handler__);
        *num_bands__ = sim_ctx.num_bands();
    }, error_code__);
}

/* @fortran begin function void sirius_get_num_spin_components        Get the number of spin components
   @fortran argument in   required void* handler                      Simulation context handler
   @fortran argument out  required int   num_spin_components          Number of spin components.
   @fortran argument out  optional int   error_code                   Error code.
   @fortran end */
void sirius_get_num_spin_components(void* const* handler__,
                                    int *num_spin_components__,
                                    int *error_code__)
{ // TODO: merge into sirius_get_parameters
    call_sirius([&]()
    {
        auto& sim_ctx = get_sim_ctx(handler__);
        if (sim_ctx.num_mag_dims() == 0) {
            *num_spin_components__ = 1;
        } else {
            *num_spin_components__ = 2;
        }
    }, error_code__);
}

/* @fortran begin function void sirius_get_kpoint_properties      Get the kpoint properties
   @fortran argument in  required void*    handler                Kpoint set handler
   @fortran argument in  required int      ik                     Index of the kpoint
   @fortran argument out required double   weight                 Weight of the kpoint
   @fortran argument out optional double   coordinates            Coordinates of the kpoint
   @fortran argument out optional int      error_code             Error code.
   @fortran end */
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

/* @fortran begin function void sirius_get_max_mt_aw_basis_size   Get maximum APW basis size across all atoms.
   @fortran argument in  required void*    handler                Simulation context handler.
   @fortran argument out required int      max_mt_aw_basis_size   Maximum APW basis size.
   @fortran argument out optional int      error_code             Error code.
   @fortran end */
void sirius_get_max_mt_aw_basis_size(void* const* handler__, int* max_mt_aw_basis_size__, int* error_code__)
{ // TODO: merge into sirius_get_parameters
    call_sirius([&]()
    {
        auto& sim_ctx = get_sim_ctx(handler__);
        *max_mt_aw_basis_size__ = sim_ctx.unit_cell().max_mt_aw_basis_size();
    }, error_code__);
}

/* @fortran begin function void sirius_get_matching_coefficients  Get matching coefficients for all atoms.
   @fortran argument in  required void*    handler                K-point set handler.
   @fortran argument in  required int      ik                     Index of k-point.
   @fortran argument out required complex  alm                    Matching coefficients.
   @fortran argument out optional int      error_code             Error code.
   @fortran details
   Warning! Generation of matching coefficients for all atoms has a large memory footprint. Use it with caution.
   @fortran end */
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


} // extern "C"
