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
#include "core/any_ptr.hpp"
#include "core/profiler.hpp"
#include "error_codes.hpp"
#ifdef SIRIUS_NLCGLIB
#include "nlcglib/adaptor.hpp"
#include "nlcglib/nlcglib.hpp"
#include "nlcglib/ultrasoft_precond.hpp"
#include "nlcglib/overlap.hpp"
#include "hamiltonian/hamiltonian.hpp"
#endif
#include "symmetry/crystal_symmetry.hpp"
#include "multi_cg/multi_cg.hpp"
#include "hamiltonian/check_wave_functions.hpp"
#include "hamiltonian/initialize_subspace.hpp"
#include "hamiltonian/diagonalize.hpp"
#include "dft/dft_ground_state.hpp"
#include "sirius.hpp"

using namespace sirius;

struct sirius_context_handler_t
{
    void* handler_ptr_{nullptr};
};

struct sirius_ground_state_handler_t
{
    void* handler_ptr_{nullptr};
};

struct sirius_kpoint_set_handler_t
{
    void* handler_ptr_{nullptr};
};

Simulation_context&
get_sim_ctx(void* const* h);

enum class option_type_t : int
{
    INTEGER_TYPE       = 1,
    LOGICAL_TYPE       = 2,
    STRING_TYPE        = 3,
    NUMBER_TYPE        = 4,
    OBJECT_TYPE        = 5,
    ARRAY_TYPE         = 6,
    INTEGER_ARRAY_TYPE = 7,
    LOGICAL_ARRAY_TYPE = 8,
    NUMBER_ARRAY_TYPE  = 9,
    STRING_ARRAY_TYPE  = 10,
    OBJECT_ARRAY_TYPE  = 11,
    ARRAY_ARRAY_TYPE   = 12
};

template <typename T>
void
sirius_option_set_value(Simulation_context& sim_ctx__, std::string section__, std::string name__, T const* values__,
                        int const* max_length__)
{
    std::transform(section__.begin(), section__.end(), section__.begin(), ::tolower);

    auto& conf_dict = const_cast<nlohmann::json&>(sim_ctx__.cfg().dict());

    const auto& section_schema = get_section_options(section__);

    if (!section_schema.count(name__)) {
        std::transform(name__.begin(), name__.end(), name__.begin(), ::tolower);
    }
    if (!section_schema.count(name__)) {
        RTE_THROW("section : " + section__ + ", option : " + name__ + " is invalid");
    }

    if (section_schema[name__]["type"] == "array") {
        if (max_length__ == nullptr) {
            RTE_THROW("maximum length of the input buffer is not provided");
        }
        std::vector<T> v(values__, values__ + *max_length__);
        conf_dict[section__][name__] = v;
    } else {
        conf_dict[section__][name__] = *values__;
    }
}

void
sirius_option_set_value(Simulation_context& sim_ctx__, std::string section__, std::string name__, char const* values__,
                        int const* max_length__, bool append__)
{
    std::transform(section__.begin(), section__.end(), section__.begin(), ::tolower);

    auto& conf_dict = const_cast<nlohmann::json&>(sim_ctx__.cfg().dict());

    const auto& section_schema = get_section_options(section__);

    if (!section_schema.count(name__)) {
        std::transform(name__.begin(), name__.end(), name__.begin(), ::tolower);
    }
    if (!section_schema.count(name__)) {
        RTE_THROW("section : " + section__ + ", option : " + name__ + " is invalid");
    }

    if (max_length__ == nullptr) {
        RTE_THROW("maximum length of the input string is not provided");
    }
    auto st = std::string(values__, *max_length__);
    if (!append__) {
        conf_dict[section__][name__] = st;
    } else {
        conf_dict[section__][name__].push_back(st);
    }
}

template <typename T>
void
sirius_option_get_value(std::string section__, std::string name__, T* default_value__, int const* max_length__)
{
    const auto& section_schema = get_section_options(section__);

    if (!section_schema.count(name__)) {
        std::transform(name__.begin(), name__.end(), name__.begin(), ::tolower);
    }
    if (!section_schema.count(name__)) {
        RTE_THROW("section : " + section__ + ", option : " + name__ + " is invalid");
    }

    if (!section_schema[name__].count("default")) {
        RTE_THROW("default value for '" + name__ + "' is missing");
    }

    if (section_schema[name__]["type"] == "array") {
        if (max_length__ == nullptr) {
            RTE_THROW("maximum length of the output buffer is not provided");
        }
        if (section_schema[name__]["items"] != "array") {
            std::vector<T> v = section_schema[name__]["default"].get<std::vector<T>>();
            int l            = static_cast<int>(v.size());
            if (l > *max_length__) {
                RTE_THROW("not enough space to store '" + name__ + "' values");
            }
            std::copy(v.begin(), v.end(), default_value__);
        }
    } else {
        *default_value__ = section_schema[name__]["default"].get<T>();
    }
}

void
sirius_option_get_value(std::string section__, std::string name__, char* default_value__, int const* max_length__,
                        int const* enum_idx__)
{
    const auto& section_schema = get_section_options(section__);

    if (!section_schema.count(name__)) {
        std::transform(name__.begin(), name__.end(), name__.begin(), ::tolower);
    }
    if (!section_schema.count(name__)) {
        RTE_THROW("section : " + section__ + ", option : " + name__ + " is invalid");
    }

    if (!section_schema[name__].count("default")) {
        RTE_THROW("default value for '" + name__ + "' is missing");
    }

    if (section_schema[name__]["type"] == "array") {
        RTE_THROW("array of strings is not supported");
    } else {
        if (section_schema[name__]["type"] != "string") {
            RTE_THROW("not a string type");
        }
        std::string v;
        if (enum_idx__ == nullptr) {
            v = section_schema[name__]["default"].get<std::string>();
        } else {
            if (section_schema[name__].count("enum")) {
                v = section_schema[name__]["enum"][*enum_idx__ - 1].get<std::string>();
            } else {
                RTE_THROW("not an enum type");
            }
        }
        if (max_length__ == nullptr) {
            RTE_THROW("length of the string is not provided");
        }
        if (static_cast<int>(v.size()) > *max_length__) {
            std::stringstream s;
            s << "option '" << name__ << "' is too large to fit into output string of size " << *max_length__;
            RTE_THROW(s);
        }
        std::fill(default_value__, default_value__ + *max_length__, ' ');
        std::copy(v.begin(), v.end(), default_value__);
    }
}

static inline void
sirius_print_error(int error_code__, std::string msg__ = "")
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
    fflush(stdout);
    std::cout << std::flush;
}

static inline void
sirius_exit(int error_code__, std::string msg__ = "")
{
    sirius_print_error(error_code__, msg__);
    if (!mpi::Communicator::is_finalized()) {
        mpi::Communicator::world().abort(error_code__);
    } else {
        std::exit(error_code__);
    }
}

template <typename F>
static void
call_sirius(F&& f__, int* error_code__)
{
    try {
        f__();
        if (error_code__) {
            *error_code__ = SIRIUS_SUCCESS;
            return;
        }
    } catch (std::runtime_error const& e) {
        if (error_code__) {
            *error_code__ = SIRIUS_ERROR_RUNTIME;
            sirius_print_error(*error_code__, e.what());
            return;
        } else {
            sirius_exit(SIRIUS_ERROR_RUNTIME, e.what());
        }
    } catch (std::exception const& e) {
        if (error_code__) {
            *error_code__ = SIRIUS_ERROR_EXCEPTION;
            sirius_print_error(*error_code__, e.what());
            return;
        } else {
            sirius_exit(SIRIUS_ERROR_EXCEPTION, e.what());
        }
    } catch (...) {
        if (error_code__) {
            *error_code__ = SIRIUS_ERROR_UNKNOWN;
            sirius_print_error(*error_code__);
            return;
        } else {
            sirius_exit(SIRIUS_ERROR_UNKNOWN);
        }
    }
}

template <typename T>
auto
get_value(T const* ptr__, T v0__ = T())
{
    return (ptr__) ? *ptr__ : v0__;
}

Simulation_context&
get_sim_ctx(void* const* h)
{
    if (h == nullptr || *h == nullptr) {
        RTE_THROW("Non-existing simulation context handler");
    }
    return static_cast<any_ptr*>(*h)->get<Simulation_context>();
}

DFT_ground_state&
get_gs(void* const* h)
{
    if (h == nullptr || *h == nullptr) {
        RTE_THROW("Non-existing DFT ground state handler");
    }
    return static_cast<any_ptr*>(*h)->get<DFT_ground_state>();
}

K_point_set&
get_ks(void* const* h)
{
    if (h == nullptr || *h == nullptr) {
        RTE_THROW("Non-existing K-point set handler");
    }
    return static_cast<any_ptr*>(*h)->get<K_point_set>();
}

/// Index of Rlm in QE in the block of lm coefficients for a given l.
static inline int
idx_m_qe(int m__)
{
    return (m__ > 0) ? 2 * m__ - 1 : -2 * m__;
}

/// Mapping of atomic indices from SIRIUS to QE order.
static inline std::vector<int>
atomic_orbital_index_map_QE(Atom_type const& type__)
{
    int nbf = type__.mt_basis_size();

    std::vector<int> idx_map(nbf);
    for (int xi = 0; xi < nbf; xi++) {
        int m       = type__.indexb(xi).m;
        auto idxrf  = type__.indexb(xi).idxrf;
        idx_map[xi] = type__.indexb().index_of(rf_index(idxrf)) +
                      idx_m_qe(m); /* beginning of lm-block + new offset in lm block */
    }
    return idx_map;
}

static inline int
phase_Rlm_QE(Atom_type const& type__, int xi__)
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
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void
sirius_initialize(bool const* call_mpi_init__, int* error_code__)
{
    call_sirius(
            [&]() {
                sirius::initialize(*call_mpi_init__);
#ifdef SIRIUS_NLCGLIB
                nlcglib::initialize();
#endif
            },
            error_code__);
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
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void
sirius_finalize(bool const* call_mpi_fin__, bool const* call_device_reset__, bool const* call_fftw_fin__,
                int* error_code__)
{
    call_sirius(
            [&]() {
#ifdef SIRIUS_NLCGLIB
                nlcglib::finalize();
#endif
                bool mpi_fin{true};
                bool device_reset{true};
                bool fftw_fin{true};

                if (call_mpi_fin__ != nullptr) {
                    mpi_fin = *call_mpi_fin__;
                }

                if (call_device_reset__ != nullptr) {
                    device_reset = *call_device_reset__;
                }

                if (call_fftw_fin__ != nullptr) {
                    fftw_fin = *call_fftw_fin__;
                }

                sirius::finalize(mpi_fin, device_reset, fftw_fin);
            },
            error_code__);
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
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void
sirius_start_timer(char const* name__, int* error_code__)
{
    call_sirius([&]() { global_rtgraph_timer.start(name__); }, error_code__);
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
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void
sirius_stop_timer(char const* name__, int* error_code__)
{
    call_sirius([&]() { global_rtgraph_timer.stop(name__); }, error_code__);
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
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void
sirius_print_timers(bool* flatten__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto timing_result = global_rtgraph_timer.process();
                if (*flatten__) {
                    timing_result = timing_result.flatten(1).sort_nodes();
                }
                std::cout << timing_result.print({rt_graph::Stat::Count, rt_graph::Stat::Total,
                                                  rt_graph::Stat::Percentage, rt_graph::Stat::SelfPercentage,
                                                  rt_graph::Stat::Median, rt_graph::Stat::Min, rt_graph::Stat::Max});
            },
            error_code__);
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
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void
sirius_serialize_timers(char const* fname__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto timing_result = global_rtgraph_timer.process();
                std::ofstream ofs(fname__, std::ofstream::out | std::ofstream::trunc);
                ofs << timing_result.json();
            },
            error_code__);
}

/*
@api begin
sirius_context_initialized:
  doc: Check if the simulation context is initialized.
  arguments:
    handler:
      type: ctx_handler
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
void
sirius_context_initialized(void* const* handler__, bool* status__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& sim_ctx = get_sim_ctx(handler__);
                *status__     = sim_ctx.initialized();
            },
            error_code__);
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
      type: ctx_handler
      attr: out, required
      doc: New empty simulation context.
    fcomm_k:
      type: int
      attr: in, optional
      doc: Communicator for k-point parallelization.
    fcomm_band:
      type: int
      attr: in, optional
      doc: Communicator for band parallelization.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void
sirius_create_context(int fcomm__, void** handler__, int* fcomm_k__, int* fcomm_band__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& comm   = mpi::Communicator::map_fcomm(fcomm__);
                auto& comm_k = (fcomm_k__) ? mpi::Communicator::map_fcomm(*fcomm_k__) : mpi::Communicator();
                auto const& comm_band =
                        (fcomm_band__) ? mpi::Communicator::map_fcomm(*fcomm_band__) : mpi::Communicator();
                *handler__ = new any_ptr(new Simulation_context(comm, comm_k, comm_band));
            },
            error_code__);
}

/*
@api begin
sirius_import_parameters:
  doc: Import parameters of simulation from a JSON string
  arguments:
    handler:
      type: ctx_handler
      attr: in, required
      doc: Simulation context handler.
    str:
      type: string
      attr: in, required
      doc: JSON string with parameters or a JSON file.
    error_code:
      type: int
      attr: out, optional
      doc: Error code
@api end
*/
void
sirius_import_parameters(void* const* handler__, char const* str__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& sim_ctx = get_sim_ctx(handler__);
                sim_ctx.import(std::string(str__));
            },
            error_code__);
}

/*
@api begin
sirius_set_parameters:
  doc: Set parameters of the simulation.
  arguments:
    handler:
      type: ctx_handler
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
    hubbard_full_orthogonalization:
      type: bool
      attr: in, optional
      doc: Use all atomic orbitals found in all ps potentials to compute the orthogonalization operator.
    hubbard_constrained_calculation:
      type: bool
      attr: in, optional
      doc: Use the constrained hubbard method to intiate the scf loop
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
    electronic_structure_method:
      type: string
      attr: in, optional
      doc: Type of electronic structure method.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void
sirius_set_parameters(void* const* handler__, int const* lmax_apw__, int const* lmax_rho__, int const* lmax_pot__,
                      int const* num_fv_states__, int const* num_bands__, int const* num_mag_dims__,
                      double const* pw_cutoff__, double const* gk_cutoff__, int const* fft_grid_size__,
                      int const* auto_rmt__, bool const* gamma_point__, bool const* use_symmetry__,
                      bool const* so_correction__, char const* valence_rel__, char const* core_rel__,
                      double const* iter_solver_tol_empty__, char const* iter_solver_type__, int const* verbosity__,
                      bool const* hubbard_correction__, int const* hubbard_correction_kind__,
                      bool const* hubbard_full_orthogonalization__, bool const* hubbard_constrained_calculation__,
                      char const* hubbard_orbitals__, int const* sht_coverage__, double const* min_occupancy__,
                      char const* smearing__, double const* smearing_width__, double const* spglib_tol__,
                      char const* electronic_structure_method__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& sim_ctx = get_sim_ctx(handler__);
                if (lmax_apw__ != nullptr) {
                    sim_ctx.lmax_apw(*lmax_apw__);
                }
                if (lmax_rho__ != nullptr) {
                    sim_ctx.lmax_rho(*lmax_rho__);
                }
                if (lmax_pot__ != nullptr) {
                    sim_ctx.lmax_pot(*lmax_pot__);
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
                if (hubbard_full_orthogonalization__ != nullptr) {
                    if (*hubbard_full_orthogonalization__) {
                        sim_ctx.cfg().hubbard().hubbard_subspace_method("full_orthogonalization");
                    }
                }

                if (hubbard_constrained_calculation__ != nullptr) {
                    sim_ctx.cfg().hubbard().constrained_calculation(*hubbard_constrained_calculation__);
                }

                if (hubbard_orbitals__ != nullptr) {
                    std::string s(hubbard_orbitals__);
                    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
                    bool jump = false;
                    if (s == "ortho-atomic") {
                        sim_ctx.cfg().hubbard().hubbard_subspace_method("full_orthogonalization");
                        jump = true;
                    }
                    if (s == "norm-atomic") {
                        sim_ctx.cfg().hubbard().hubbard_subspace_method("normalize");
                        jump = true;
                    }
                    if (!jump) {
                        sim_ctx.cfg().hubbard().hubbard_subspace_method(s);
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
                if (electronic_structure_method__ != nullptr) {
                    sim_ctx.cfg().parameters().electronic_structure_method(electronic_structure_method__);
                }
            },
            error_code__);
}

/*
@api begin
sirius_get_parameters:
  doc: Get parameters of the simulation.
  arguments:
    handler:
      type: ctx_handler
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
      doc: Tolerance of the iterative solver (deprecated).
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
void
sirius_get_parameters(void* const* handler__, int* lmax_apw__, int* lmax_rho__, int* lmax_pot__, int* num_fv_states__,
                      int* num_bands__, int* num_spins__, int* num_mag_dims__, double* pw_cutoff__, double* gk_cutoff__,
                      int* fft_grid_size__, int* auto_rmt__, bool* gamma_point__, bool* use_symmetry__,
                      bool* so_correction__, double* iter_solver_tol__, double* iter_solver_tol_empty__,
                      int* verbosity__, bool* hubbard_correction__, double* evp_work_count__, int* num_loc_op_applied__,
                      int* num_sym_op__, char* electronic_structure_method__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& sim_ctx = get_sim_ctx(handler__);
                if (lmax_apw__) {
                    *lmax_apw__ = sim_ctx.unit_cell().lmax_apw();
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
                    for (int x : {0, 1, 2}) {
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
            },
            error_code__);
}

/*
@api begin
sirius_add_xc_functional:
  doc: Add one of the XC functionals.
  arguments:
    handler:
      type: ctx_handler
      attr: in, required
      doc: Simulation context handler
    name:
      type: string
      attr: in, required
      doc: LibXC label of the functional.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void
sirius_add_xc_functional(void* const* handler__, char const* name__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& sim_ctx = get_sim_ctx(handler__);
                sim_ctx.add_xc_functional(std::string(name__));
            },
            error_code__);
}

/*
@api begin
sirius_set_mpi_grid_dims:
  doc: Set dimensions of the MPI grid.
  arguments:
    handler:
      type: ctx_handler
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
void
sirius_set_mpi_grid_dims(void* const* handler__, int const* ndims__, int const* dims__, int* error_code__)
{
    call_sirius(
            [&]() {
                RTE_ASSERT(*ndims__ > 0);
                auto& sim_ctx = get_sim_ctx(handler__);
                std::vector<int> dims(dims__, dims__ + *ndims__);
                sim_ctx.mpi_grid_dims(dims);
            },
            error_code__);
}

/*
@api begin
sirius_set_lattice_vectors:
  doc: Set vectors of the unit cell.
  arguments:
    handler:
      type: ctx_handler
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
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void
sirius_set_lattice_vectors(void* const* handler__, double const* a1__, double const* a2__, double const* a3__,
                           int* error_code__)
{
    call_sirius(
            [&]() {
                auto& sim_ctx = get_sim_ctx(handler__);
                sim_ctx.unit_cell().set_lattice_vectors(r3::vector<double>(a1__), r3::vector<double>(a2__),
                                                        r3::vector<double>(a3__));
            },
            error_code__);
}

/*
@api begin
sirius_initialize_context:
  doc: Initialize simulation context.
  arguments:
    handler:
      type: ctx_handler
      attr: in, required
      doc: Simulation context handler.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void
sirius_initialize_context(void* const* handler__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& sim_ctx = get_sim_ctx(handler__);
                sim_ctx.initialize();
                return 0;
            },
            error_code__);
}

/*
@api begin
sirius_update_context:
  doc: Update simulation context after changing lattice or atomic positions.
  arguments:
    handler:
      type: ctx_handler
      attr: in, required
      doc: Simulation context handler.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void
sirius_update_context(void* const* handler__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& sim_ctx = get_sim_ctx(handler__);
                sim_ctx.update();
                return 0;
            },
            error_code__);
}

/*
@api begin
sirius_print_info:
  doc: Print basic info
  arguments:
    handler:
      type: ctx_handler
      attr: in, required
      doc: Simulation context handler.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void
sirius_print_info(void* const* handler__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& sim_ctx = get_sim_ctx(handler__);
                sim_ctx.print_info(sim_ctx.out());
            },
            error_code__);
}

/*
@api begin
sirius_free_object_handler:
  doc: Free any object handler created by SIRIUS.
  full_doc: This is an internal function. Use sirius_free_handler() in your code.
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
void
sirius_free_object_handler(void** handler__, int* error_code__)
{
    call_sirius(
            [&]() {
                if (*handler__ != nullptr) {
                    delete static_cast<any_ptr*>(*handler__);
                }
                *handler__ = nullptr;
            },
            error_code__);
}

/*
@api begin
sirius_set_periodic_function_ptr:
  doc: Set pointer to density or magnetization.
  arguments:
    handler:
      type: ctx_handler
      attr: in, required
      doc: Simulation context handler.
    label:
      type: string
      attr: in, required
      doc: Label of the function.
    f_mt:
      type: double
      attr: in, optional, dimension(:,:,:)
      doc: Pointer to the muffin-tin part of the function.
    lmmax:
      type: int
      attr: in, optional
      doc: Number of lm components.
    nrmtmax:
      type: int
      attr: in, optional
      doc: Maximum number of muffin-tin points.
    num_atoms:
      type: int
      attr: in, optional
      doc: Total number of atoms.
    f_rg:
      type: double
      attr: in, optional, dimension(:)
      doc: Pointer to the regular-grid part of the function.
    size_x:
      type: int
      attr: in, optional
      doc: Size of X-dimension of FFT grid.
    size_y:
      type: int
      attr: in, optional
      doc: Size of Y-dimension of FFT grid.
    size_z:
      type: int
      attr: in, optional
      doc: Local or global size of Z-dimension of FFT grid depending on offset_z
    offset_z:
      type: int
      attr: in, optional
      doc: Offset in the Z-dimension of FFT grid for this MPI rank.
    error_code:
      type: int
      attr: out, optional
      doc: Error code
@api end
*/
void
sirius_set_periodic_function_ptr(void* const* handler__, char const* label__, double* f_mt__, int const* lmmax__,
                                 int const* nrmtmax__, int const* num_atoms__, double* f_rg__, int const* size_x__,
                                 int const* size_y__, int const* size_z__, int const* offset_z__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& sim_ctx = get_sim_ctx(handler__);
                std::string label(label__);

                int lmmax     = get_value(lmmax__);
                int nrmtmax   = get_value(nrmtmax__);
                int num_atoms = get_value(num_atoms__);
                int size_x    = get_value(size_x__);
                int size_y    = get_value(size_y__);
                int size_z    = get_value(size_z__);
                int offset_z  = get_value(offset_z__, -1);

                spheric_function_set_ptr_t<double> mt_ptr(f_mt__, lmmax, nrmtmax, num_atoms);
                smooth_periodic_function_ptr_t<double> rg_ptr(f_rg__, size_x, size_y, size_z, offset_z);

                sim_ctx.set_periodic_function_ptr(label, periodic_function_ptr_t<double>(mt_ptr, rg_ptr));
            },
            error_code__);
}

/*
@api begin
sirius_set_periodic_function:
  doc: Set values of the periodic function.
  arguments:
    handler:
      type: gs_handler
      attr: in, required
      doc: Handler of the DFT ground state object.
    label:
      type: string
      attr: in, required
      doc: Label of the function.
    f_mt:
      type: double
      attr: in, optional, dimension(:,:,:)
      doc: Pointer to the muffin-tin part of the function.
    lmmax:
      type: int
      attr: in, optional
      doc: Number of lm components.
    nrmtmax:
      type: int
      attr: in, optional
      doc: Maximum number of muffin-tin points.
    num_atoms:
      type: int
      attr: in, optional
      doc: Total number of atoms.
    f_rg:
      type: double
      attr: in, optional, dimension(:)
      doc: Pointer to the regular-grid part of the function.
    size_x:
      type: int
      attr: in, optional
      doc: Size of X-dimension of FFT grid.
    size_y:
      type: int
      attr: in, optional
      doc: Size of Y-dimension of FFT grid.
    size_z:
      type: int
      attr: in, optional
      doc: Local or global size of Z-dimension of FFT grid depending on offset_z
    offset_z:
      type: int
      attr: in, optional
      doc: Offset in the Z-dimension of FFT grid for this MPI rank.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void
sirius_set_periodic_function(void* const* handler__, char const* label__, double* f_mt__, int const* lmmax__,
                             int const* nrmtmax__, int const* num_atoms__, double* f_rg__, int const* size_x__,
                             int const* size_y__, int const* size_z__, int const* offset_z__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& gs = get_gs(handler__);
                std::string label(label__);
                std::map<std::string, Periodic_function<double>*> func_map = {
                        {"rho", &gs.density().component(0)},         {"magz", &gs.density().component(1)},
                        {"magx", &gs.density().component(2)},        {"magy", &gs.density().component(3)},
                        {"veff", &gs.potential().component(0)},      {"bz", &gs.potential().component(1)},
                        {"bx", &gs.potential().component(2)},        {"by", &gs.potential().component(3)},
                        {"vha", &gs.potential().hartree_potential()}};

                if (!func_map.count(label)) {
                    RTE_THROW("wrong label (" + label + ") for the periodic function");
                }

                int lmmax     = get_value(lmmax__);
                int nrmtmax   = get_value(nrmtmax__);
                int num_atoms = get_value(num_atoms__);
                int size_x    = get_value(size_x__);
                int size_y    = get_value(size_y__);
                int size_z    = get_value(size_z__);
                int offset_z  = get_value(offset_z__, -1);

                if (f_mt__) {
                    spheric_function_set_ptr_t<double> mt_ptr(f_mt__, lmmax, nrmtmax, num_atoms);
                    copy(mt_ptr, func_map[label]->mt());
                }
                if (f_rg__) {
                    smooth_periodic_function_ptr_t<double> rg_ptr(f_rg__, size_x, size_y, size_z, offset_z);
                    copy(rg_ptr, func_map[label]->rg());
                }
            },
            error_code__);
}

/*
@api begin
sirius_get_periodic_function:
  doc: Get values of the periodic function.
  arguments:
    handler:
      type: gs_handler
      attr: in, required
      doc: Handler of the DFT ground state object.
    label:
      type: string
      attr: in, required
      doc: Label of the function.
    f_mt:
      type: double
      attr: in, optional, dimension(:,:,:)
      doc: Pointer to the muffin-tin part of the function.
    lmmax:
      type: int
      attr: in, optional
      doc: Number of lm components.
    nrmtmax:
      type: int
      attr: in, optional
      doc: Maximum number of muffin-tin points.
    num_atoms:
      type: int
      attr: in, optional
      doc: Total number of atoms.
    f_rg:
      type: double
      attr: in, optional, dimension(:)
      doc: Pointer to the regular-grid part of the function.
    size_x:
      type: int
      attr: in, optional
      doc: Size of X-dimension of FFT grid.
    size_y:
      type: int
      attr: in, optional
      doc: Size of Y-dimension of FFT grid.
    size_z:
      type: int
      attr: in, optional
      doc: Local or global size of Z-dimension of FFT grid depending on offset_z
    offset_z:
      type: int
      attr: in, optional
      doc: Offset in the Z-dimension of FFT grid for this MPI rank.
    error_code:
      type: int
      attr: out, optional
      doc: Error code
@api end
*/
void
sirius_get_periodic_function(void* const* handler__, char const* label__, double* f_mt__, int const* lmmax__,
                             int const* nrmtmax__, int const* num_atoms__, double* f_rg__, int const* size_x__,
                             int const* size_y__, int const* size_z__, int const* offset_z__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& gs = get_gs(handler__);
                std::string label(label__);
                std::map<std::string, Periodic_function<double>*> func_map = {
                        {"rho", &gs.density().component(0)},          {"magz", &gs.density().component(1)},
                        {"magx", &gs.density().component(2)},         {"magy", &gs.density().component(3)},
                        {"veff", &gs.potential().component(0)},       {"bz", &gs.potential().component(1)},
                        {"bx", &gs.potential().component(2)},         {"by", &gs.potential().component(3)},
                        {"vha", &gs.potential().hartree_potential()}, {"exc", &gs.potential().xc_energy_density()},
                        {"vxc", &gs.potential().xc_potential()}};

                if (!func_map.count(label)) {
                    RTE_THROW("wrong label (" + label + ") for the periodic function");
                }

                int lmmax     = get_value(lmmax__);
                int nrmtmax   = get_value(nrmtmax__);
                int num_atoms = get_value(num_atoms__);
                int size_x    = get_value(size_x__);
                int size_y    = get_value(size_y__);
                int size_z    = get_value(size_z__);
                int offset_z  = get_value(offset_z__, -1);

                if (f_mt__) {
                    spheric_function_set_ptr_t<double> mt_ptr(f_mt__, lmmax, nrmtmax, num_atoms);
                    copy(func_map[label]->mt(), mt_ptr);
                }
                if (f_rg__) {
                    smooth_periodic_function_ptr_t<double> rg_ptr(f_rg__, size_x, size_y, size_z, offset_z);
                    copy(func_map[label]->rg(), rg_ptr);
                }
            },
            error_code__);
}

/*
@api begin
sirius_create_kset:
  doc: Create k-point set from the list of k-points.
  arguments:
    handler:
      type: ctx_handler
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
      type: ks_handler
      attr: out, required
      doc: Handler of the newly created k-point set.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void
sirius_create_kset(void* const* handler__, int const* num_kpoints__, double* kpoints__, double const* kpoint_weights__,
                   bool const* init_kset__, void** kset_handler__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& sim_ctx = get_sim_ctx(handler__);

                mdarray<double, 2> kpoints({3, *num_kpoints__}, kpoints__);

                K_point_set* new_kset = new K_point_set(sim_ctx);
                new_kset->add_kpoints(kpoints, kpoint_weights__);
                if (*init_kset__) {
                    new_kset->initialize();
                }
                *kset_handler__ = new any_ptr(new_kset);
            },
            error_code__);
}

/*
@api begin
sirius_create_kset_from_grid:
  doc: Create k-point set from a grid.
  arguments:
    handler:
      type: ctx_handler
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
      type: ks_handler
      attr: out, required
      doc: Handler of the newly created k-point set.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void
sirius_create_kset_from_grid(void* const* handler__, int const* k_grid__, int const* k_shift__,
                             bool const* use_symmetry, void** kset_handler__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& sim_ctx = get_sim_ctx(handler__);
                std::vector<int> k_grid(3);
                std::vector<int> k_shift(3);

                k_grid[0] = k_grid__[0];
                k_grid[1] = k_grid__[1];
                k_grid[2] = k_grid__[2];

                k_shift[0] = k_shift__[0];
                k_shift[1] = k_shift__[1];
                k_shift[2] = k_shift__[2];

                K_point_set* new_kset = new K_point_set(sim_ctx, k_grid, k_shift, *use_symmetry);

                *kset_handler__ = new any_ptr(new_kset);
            },
            error_code__);
}

/*
@api begin
sirius_create_ground_state:
  doc: Create a ground state object.
  arguments:
    ks_handler:
      type: ks_handler
      attr: in, required
      doc: Handler of the k-point set.
    gs_handler:
      type: gs_handler
      attr: out, required
      doc: Handler of the newly created ground state object.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void
sirius_create_ground_state(void* const* ks_handler__, void** gs_handler__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& ks = get_ks(ks_handler__);

                *gs_handler__ = new any_ptr(new DFT_ground_state(ks));
            },
            error_code__);
}

/*
@api begin
sirius_initialize_kset:
  doc: Initialize k-point set.
  arguments:
    ks_handler:
      type: ks_handler
      attr: in, required
      doc: K-point set handler.
    count:
      type: int
      attr: in, optional, dimension(:)
      doc: Local number of k-points for each MPI rank.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void
sirius_initialize_kset(void* const* ks_handler__, int* count__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& ks = get_ks(ks_handler__);
                if (count__) {
                    std::vector<int> count(count__, count__ + ks.comm().size());
                    ks.initialize(count);
                } else {
                    ks.initialize();
                }
            },
            error_code__);
}

/*
@api begin
sirius_find_ground_state:
  doc: Find the ground state.
  arguments:
    gs_handler:
      type: gs_handler
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
    iter_solver_tol:
      type: double
      attr: in, optional
      doc: Initial tolerance of the iterative solver.
    initial_guess:
      type: bool
      attr: in, optional
      doc: Boolean variable indicating if we want to start from the initial guess or from previous state.
    max_niter:
      type: int
      attr: in, optional
      doc: Maximum number of SCF iterations.
    save_state:
      type: bool
      attr: in, optional
      doc: Boolean variable indicating if we want to save the ground state.
    converged:
      type: bool
      attr: out, optional
      doc: Boolean variable indicating if the calculation has converged
    niter:
      type: int
      attr: out, optional
      doc: Actual number of SCF iterations.
    rho_min:
      type: double
      attr: out, optional
      doc: Minimum value of density on the real-space grid. If negative, total energy can't be trusted. Valid only if
           SCF calculation is converged.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void
sirius_find_ground_state(void* const* gs_handler__, double const* density_tol__, double const* energy_tol__,
                         double const* iter_solver_tol__, bool const* initial_guess__, int const* max_niter__,
                         bool const* save_state__, bool* converged__, int* niter__, double* rho_min__,
                         int* error_code__)
{
    call_sirius(
            [&]() {
                auto& gs  = get_gs(gs_handler__);
                auto& ctx = gs.ctx();
                auto& inp = ctx.cfg().parameters();

                bool initial_guess = (initial_guess__) ? *initial_guess__ : true;
                if (initial_guess) {
                    gs.initial_state();
                }

                double rho_tol = (density_tol__) ? *density_tol__ : inp.density_tol();

                double etol = (energy_tol__) ? *energy_tol__ : inp.energy_tol();

                double iter_solver_tol =
                        (iter_solver_tol__) ? *iter_solver_tol__ : ctx.cfg().iterative_solver().energy_tolerance();

                int max_niter = (max_niter__) ? *max_niter__ : inp.num_dft_iter();

                bool save = (save_state__) ? *save_state__ : false;

                auto result = gs.find(rho_tol, etol, iter_solver_tol, max_niter, save);

                if (result["converged"].get<bool>()) {
                    if (converged__) {
                        *converged__ = true;
                    }
                    if (niter__) {
                        *niter__ = result["num_scf_iterations"].get<int>();
                    }
                    if (rho_min__) {
                        *rho_min__ = result["rho_min"].get<double>();
                    }
                } else {
                    if (converged__) {
                        *converged__ = false;
                    }
                    if (niter__) {
                        *niter__ = max_niter;
                    }
                    if (rho_min__) {
                        *rho_min__ = 0;
                    }
                }
            },
            error_code__);
}

/*
@api begin
sirius_check_scf_density:
  doc: Check the self-consistent density
  arguments:
    gs_handler:
      type: gs_handler
      attr: in, required
      doc: Handler of the ground state.
    error_code:
      type: int
      attr: out, optional
      doc: Error code
@api end
*/
void
sirius_check_scf_density(void* const* gs_handler__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& gs = get_gs(gs_handler__);
                gs.check_scf_density();
            },
            error_code__);
}

/*
@api begin
sirius_update_ground_state:
  doc: Update a ground state object after change of atomic coordinates or lattice vectors.
  arguments:
    gs_handler:
      type: gs_handler
      attr: in, required
      doc: Ground-state handler.
    error_code:
      type: int
      attr: out, optional
      doc: Error code
@api end
*/
void
sirius_update_ground_state(void** handler__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& gs = get_gs(handler__);
                gs.update();
            },
            error_code__);
}

/*
@api begin
sirius_add_atom_type:
  doc: Add new atom type to the unit cell.
  arguments:
    handler:
      type: ctx_handler
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
void
sirius_add_atom_type(void* const* handler__, char const* label__, char const* fname__, int const* zn__,
                     char const* symbol__, double const* mass__, bool const* spin_orbit__, int* error_code__)
{
    call_sirius(
            [&]() {
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
            },
            error_code__);
}

/*
@api begin
sirius_set_atom_type_radial_grid:
  doc: Set radial grid of the atom type.
  arguments:
    handler:
      type: ctx_handler
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
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void
sirius_set_atom_type_radial_grid(void* const* handler__, char const* label__, int const* num_radial_points__,
                                 double const* radial_points__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& sim_ctx = get_sim_ctx(handler__);

                auto& type = sim_ctx.unit_cell().atom_type(std::string(label__));
                type.set_radial_grid(*num_radial_points__, radial_points__);
            },
            error_code__);
}

/*
@api begin
sirius_set_atom_type_radial_grid_inf:
  doc: Set radial grid of the free atom (up to effectice infinity).
  arguments:
    handler:
      type: ctx_handler
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
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void
sirius_set_atom_type_radial_grid_inf(void* const* handler__, char const* label__, int const* num_radial_points__,
                                     double const* radial_points__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& sim_ctx = get_sim_ctx(handler__);

                auto& type = sim_ctx.unit_cell().atom_type(std::string(label__));
                type.set_free_atom_radial_grid(*num_radial_points__, radial_points__);
            },
            error_code__);
}

/*
@api begin
sirius_add_atom_type_radial_function:
  doc: Add one of the radial functions.
  arguments:
    handler:
      type: ctx_handler
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
      doc: First index of radial function (for Q-operator). Indices start from 1.
    idxrf2:
      type: int
      attr: in, optional
      doc: Second index of radial function (for Q-operator). Indices start form 1.
    occ:
      type: double
      attr: in, optional
      doc: Occupancy of the wave-function.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void
sirius_add_atom_type_radial_function(void* const* handler__, char const* atom_type__, char const* label__,
                                     double const* rf__, int const* num_points__, int const* n__, int const* l__,
                                     int const* idxrf1__, int const* idxrf2__, double const* occ__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& sim_ctx = get_sim_ctx(handler__);

                auto& type = sim_ctx.unit_cell().atom_type(std::string(atom_type__));
                std::string label(label__);

                if (label == "beta") { /* beta-projectors */
                    if (l__ == nullptr) {
                        RTE_THROW("orbital quantum number must be provided for beta-projector");
                    }
                    int l = *l__;
                    if (type.spin_orbit_coupling()) {
                        if (l >= 0) {
                            type.add_beta_radial_function(angular_momentum(l, 1),
                                                          std::vector<double>(rf__, rf__ + *num_points__));
                        } else {
                            type.add_beta_radial_function(angular_momentum(-l, -1),
                                                          std::vector<double>(rf__, rf__ + *num_points__));
                        }
                    } else {
                        type.add_beta_radial_function(angular_momentum(l),
                                                      std::vector<double>(rf__, rf__ + *num_points__));
                    }
                } else if (label == "ps_atomic_wf") { /* pseudo-atomic wave functions */
                    if (l__ == nullptr) {
                        RTE_THROW("orbital quantum number must be provided for pseudo-atomic radial function");
                    }
                    int n      = (n__) ? *n__ : -1;
                    double occ = (occ__) ? *occ__ : 0.0;
                    type.add_ps_atomic_wf(n, angular_momentum(*l__), std::vector<double>(rf__, rf__ + *num_points__),
                                          occ);
                } else if (label == "ps_rho_core") {
                    type.ps_core_charge_density(std::vector<double>(rf__, rf__ + *num_points__));
                } else if (label == "ps_rho_total") {
                    type.ps_total_charge_density(std::vector<double>(rf__, rf__ + *num_points__));
                } else if (label == "vloc") {
                    type.local_potential(std::vector<double>(rf__, rf__ + *num_points__));
                } else if (label == "q_aug") { /* augmentation charge */
                    if (l__ == nullptr) {
                        RTE_THROW("orbital quantum number must be provided for augmentation charge radial function");
                    }
                    if (idxrf1__ == nullptr || idxrf2__ == nullptr) {
                        RTE_THROW("both radial-function indices must be provided for augmentation charge radial "
                                  "function");
                    }
                    type.add_q_radial_function(*idxrf1__ - 1, *idxrf2__ - 1, *l__,
                                               std::vector<double>(rf__, rf__ + *num_points__));
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
                    RTE_THROW(s);
                }
            },
            error_code__);
}

/*
@api begin
sirius_set_atom_type_hubbard:
  doc: Set the hubbard correction for the atomic type.
  arguments:
    handler:
      type: ctx_handler
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
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void
sirius_set_atom_type_hubbard(void* const* handler__, char const* label__, int const* l__, int const* n__,
                             double const* occ__, double const* U__, double const* J__, double const* alpha__,
                             double const* beta__, double const* J0__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& sim_ctx = get_sim_ctx(handler__);
                auto& type    = sim_ctx.unit_cell().atom_type(std::string(label__));
                type.hubbard_correction(true);
                if (type.file_name().empty()) {
                    type.add_hubbard_orbital(*n__, *l__, *occ__, *U__, J__[1], J__, *alpha__, *beta__, *J0__,
                                             std::vector<double>(), true);
                } else {
                    // we use a an external file containing the potential
                    // information which means that we do not have all information
                    // yet.
                    //
                    // let's build the local hubbard section instead
                    //

                    json elem;

                    elem["atom_type"]               = label__;
                    elem["n"]                       = *n__;
                    elem["l"]                       = *l__;
                    elem["total_initial_occupancy"] = *occ__;
                    elem["U"]                       = *U__;
                    elem["J"]                       = *J__;
                    elem["alpha"]                   = *alpha__;
                    elem["beta"]                    = *beta__;
                    sim_ctx.cfg().hubbard().local().append(elem);
                }
            },
            error_code__);
}

/*
@api begin
sirius_set_atom_type_dion:
  doc: Set ionic part of D-operator matrix.
  arguments:
    handler:
      type: ctx_handler
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
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void
sirius_set_atom_type_dion(void* const* handler__, char const* label__, int const* num_beta__, double* dion__,
                          int* error_code__)
{
    call_sirius(
            [&]() {
                auto& sim_ctx = get_sim_ctx(handler__);
                auto& type    = sim_ctx.unit_cell().atom_type(std::string(label__));
                matrix<double> dion({*num_beta__, *num_beta__}, dion__);
                type.d_mtrx_ion(dion);
            },
            error_code__);
}

/*
@api begin
sirius_set_atom_type_paw:
  doc: Set PAW related data.
  arguments:
    handler:
      type: ctx_handler
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
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void
sirius_set_atom_type_paw(void* const* handler__, char const* label__, double const* core_energy__,
                         double const* occupations__, int const* num_occ__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& sim_ctx = get_sim_ctx(handler__);

                auto& type = sim_ctx.unit_cell().atom_type(std::string(label__));

                if (*num_occ__ != type.num_beta_radial_functions()) {
                    RTE_THROW("PAW error: different number of occupations and wave functions!");
                }

                /* we load PAW, so we set is_paw to true */
                type.is_paw(true);

                type.paw_core_energy(*core_energy__);

                type.paw_wf_occ(std::vector<double>(occupations__, occupations__ + type.num_beta_radial_functions()));
            },
            error_code__);
}

/*
@api begin
sirius_add_atom:
  doc: Add atom to the unit cell.
  arguments:
    handler:
      type: ctx_handler
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
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void
sirius_add_atom(void* const* handler__, char const* label__, double const* position__, double const* vector_field__,
                int* error_code__)
{
    call_sirius(
            [&]() {
                auto& sim_ctx = get_sim_ctx(handler__);
                if (vector_field__ != nullptr) {
                    sim_ctx.unit_cell().add_atom(std::string(label__), std::vector<double>(position__, position__ + 3),
                                                 vector_field__);
                } else {
                    sim_ctx.unit_cell().add_atom(std::string(label__), std::vector<double>(position__, position__ + 3));
                }
            },
            error_code__);
}

/*
@api begin
sirius_set_atom_position:
  doc: Set new atomic position.
  arguments:
    handler:
      type: ctx_handler
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
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void
sirius_set_atom_position(void* const* handler__, int const* ia__, double const* position__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& sim_ctx = get_sim_ctx(handler__);
                sim_ctx.unit_cell().atom(*ia__ - 1).set_position(std::vector<double>(position__, position__ + 3));
            },
            error_code__);
}

/*
@api begin
sirius_set_pw_coeffs:
  doc: Set plane-wave coefficients of a periodic function.
  arguments:
    handler:
      type: gs_handler
      attr: in, required
      doc: Ground state handler.
    label:
      type: string
      attr: in, required
      doc: Label of the function.
    pw_coeffs:
      type: complex
      attr: in, required, dimension(:)
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
      attr: in, optional, dimension(:,:)
      doc: List of G-vectors in lattice coordinates (Miller indices).
    comm:
      type: int
      attr: in, optional
      doc: MPI communicator used in distribution of G-vectors
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void
sirius_set_pw_coeffs(void* const* handler__, char const* label__, std::complex<double> const* pw_coeffs__,
                     bool const* transform_to_rg__, int const* ngv__, int* gvl__, int const* comm__, int* error_code__)
{
    PROFILE("sirius_api::sirius_set_pw_coeffs");
    call_sirius(
            [&]() {
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
                        RTE_THROW("wrong label: " + label);
                    }
                } else {
                    RTE_ASSERT(ngv__ != nullptr);
                    RTE_ASSERT(gvl__ != nullptr);
                    RTE_ASSERT(comm__ != nullptr);

                    mpi::Communicator comm(MPI_Comm_f2c(*comm__));
                    mdarray<int, 2> gvec({3, *ngv__}, gvl__);

                    std::vector<std::complex<double>> v(gs.ctx().gvec().num_gvec(), 0);
                    #pragma omp parallel for schedule(static)
                    for (int i = 0; i < *ngv__; i++) {
                        r3::vector<int> G(gvec(0, i), gvec(1, i), gvec(2, i));
                        // auto gvc = gs.ctx().unit_cell().reciprocal_lattice_vectors() * r3::vector<double>(G[0], G[1],
                        // G[2]); if (gvc.length() > gs.ctx().pw_cutoff()) {
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
                                                   r3::vector<double>(G[0], G[1], G[2]));
                                    s << "wrong index of G-vector" << std::endl
                                      << "input G-vector: " << G << " (length: " << gvc.length() << " [a.u.^-1])"
                                      << std::endl;
                                    RTE_THROW(s);
                                } else {
                                    v[ig] = std::conj(pw_coeffs__[i]);
                                }
                            }
                        }
                    }
                    comm.allreduce(v.data(), gs.ctx().gvec().num_gvec());

                    std::map<std::string, Smooth_periodic_function<double>*> func = {
                            {"rho", &gs.density().rho().rg()},
                            {"rhoc", &gs.density().rho_pseudo_core()},
                            {"magz", &gs.density().mag(0).rg()},
                            {"magx", &gs.density().mag(1).rg()},
                            {"magy", &gs.density().mag(2).rg()},
                            {"veff", &gs.potential().effective_potential().rg()},
                            {"bz", &gs.potential().effective_magnetic_field(0).rg()},
                            {"bx", &gs.potential().effective_magnetic_field(1).rg()},
                            {"by", &gs.potential().effective_magnetic_field(2).rg()},
                            {"vloc", &gs.potential().local_potential()},
                            {"vxc", &gs.potential().xc_potential().rg()},
                            {"dveff", &gs.potential().dveff()},
                    };

                    if (!func.count(label)) {
                        RTE_THROW("wrong label: " + label);
                    }

                    func.at(label)->scatter_f_pw(v);

                    if (transform_to_rg__ && *transform_to_rg__) {
                        func.at(label)->fft_transform(1);
                    }
                }
            },
            error_code__);
}

/*
@api begin
sirius_get_pw_coeffs:
  doc: Get plane-wave coefficients of a periodic function.
  arguments:
    handler:
      type: gs_handler
      attr: in, required
      doc: Ground state handler.
    label:
      type: string
      attr: in, required
      doc: Label of the function.
    pw_coeffs:
      type: complex
      attr: in, required, dimension(:)
      doc: Local array of plane-wave coefficients.
    ngv:
      type: int
      attr: in, optional
      doc: Local number of G-vectors.
    gvl:
      type: int
      attr: in, optional, dimension(:,:)
      doc: List of G-vectors in lattice coordinates (Miller indices).
    comm:
      type: int
      attr: in, optional
      doc: MPI communicator used in distribution of G-vectors
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void
sirius_get_pw_coeffs(void* const* handler__, char const* label__, std::complex<double>* pw_coeffs__, int const* ngv__,
                     int* gvl__, int const* comm__, int* error_code__)
{
    PROFILE("sirius_api::sirius_get_pw_coeffs");

    call_sirius(
            [&]() {
                auto& gs = get_gs(handler__);

                std::string label(label__);
                if (gs.ctx().full_potential()) {
                    RTE_THROW("not implemented");
                } else {
                    RTE_ASSERT(ngv__ != NULL);
                    RTE_ASSERT(gvl__ != NULL);
                    RTE_ASSERT(comm__ != NULL);

                    mpi::Communicator comm(MPI_Comm_f2c(*comm__));
                    mdarray<int, 2> gvec({3, *ngv__}, gvl__);

                    std::map<std::string, Smooth_periodic_function<double>*> func = {
                            {"rho", &gs.density().rho().rg()},
                            {"magz", &gs.density().mag(0).rg()},
                            {"magx", &gs.density().mag(1).rg()},
                            {"magy", &gs.density().mag(2).rg()},
                            {"veff", &gs.potential().effective_potential().rg()},
                            {"vloc", &gs.potential().local_potential()},
                            {"rhoc", &gs.density().rho_pseudo_core()}};

                    if (!func.count(label)) {
                        RTE_THROW("wrong label: " + label);
                    }
                    auto v = func.at(label)->gather_f_pw();

                    for (int i = 0; i < *ngv__; i++) {
                        r3::vector<int> G(gvec(0, i), gvec(1, i), gvec(2, i));

                        // auto gvc = gs.ctx().unit_cell().reciprocal_lattice_vectors() * r3::vector<double>(G[0], G[1],
                        // G[2]); if (gvc.length() > gs.ctx().pw_cutoff()) {
                        //    pw_coeffs__[i] = 0;
                        //    continue;
                        //}

                        bool is_inverse{false};
                        int ig = gs.ctx().gvec().index_by_gvec(G);
                        if (ig == -1 && gs.ctx().gvec().reduced()) {
                            ig         = gs.ctx().gvec().index_by_gvec(G * (-1));
                            is_inverse = true;
                        }
                        if (ig == -1) {
                            std::stringstream s;
                            auto gvc = dot(gs.ctx().unit_cell().reciprocal_lattice_vectors(),
                                           r3::vector<double>(G[0], G[1], G[2]));
                            s << "wrong index of G-vector" << std::endl
                              << "input G-vector: " << G << " (length: " << gvc.length() << " [a.u.^-1])" << std::endl;
                            RTE_WARNING(s);
                            pw_coeffs__[i] = 0;
                            // RTE_THROW(s);
                        } else {
                            if (is_inverse) {
                                pw_coeffs__[i] = std::conj(v[ig]);
                            } else {
                                pw_coeffs__[i] = v[ig];
                            }
                        }
                    }
                }
            },
            error_code__);
}

/*
@api begin
sirius_initialize_subspace:
  doc: Initialize the subspace of wave-functions.
  arguments:
    gs_handler:
      type: gs_handler
      attr: in, required
      doc: Ground state handler.
    ks_handler:
      type: ks_handler
      attr: in, required
      doc: K-point set handler.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void
sirius_initialize_subspace(void* const* gs_handler__, void* const* ks_handler__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& gs = get_gs(gs_handler__);
                auto& ks = get_ks(ks_handler__);
                Hamiltonian0<double> H0(gs.potential(), true);
                initialize_subspace(ks, H0);
            },
            error_code__);
}

/*
@api begin
sirius_find_eigen_states:
  doc: Find eigen-states of the Hamiltonian
  arguments:
    gs_handler:
      type: gs_handler
      attr: in, required
      doc: Ground state handler.
    ks_handler:
      type: ks_handler
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
    iter_solver_tol:
      type: int
      attr: in, optional
      doc: Iterative solver number of steps.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void
sirius_find_eigen_states(void* const* gs_handler__, void* const* ks_handler__, bool const* precompute_pw__,
                         bool const* precompute_rf__, bool const* precompute_ri__, double const* iter_solver_tol__,
                         int const* iter_solver_steps__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& gs = get_gs(gs_handler__);
                auto& ks = get_ks(ks_handler__);
                double tol =
                        (iter_solver_tol__) ? *iter_solver_tol__ : ks.ctx().cfg().iterative_solver().energy_tolerance();
                int steps =
                        (iter_solver_steps__) ? *iter_solver_steps__ : ks.ctx().cfg().iterative_solver().num_steps();
                if (precompute_pw__ && *precompute_pw__) {
                    gs.potential().generate_pw_coefs();
                }
                if ((precompute_rf__ && *precompute_rf__) || (precompute_ri__ && *precompute_ri__)) {
                    gs.potential().update_atomic_potential();
                }
                if (precompute_rf__ && *precompute_rf__) {
                    const_cast<Unit_cell&>(gs.ctx().unit_cell()).generate_radial_functions(gs.ctx().out());
                }
                if (precompute_ri__ && *precompute_ri__) {
                    const_cast<Unit_cell&>(gs.ctx().unit_cell()).generate_radial_integrals();
                }
                Hamiltonian0<double> H0(gs.potential(), false);
                diagonalize<double, double>(H0, ks, tol, steps);
            },
            error_code__);
}

/*
@api begin
sirius_generate_initial_density:
  doc: Generate initial density.
  arguments:
    handler:
      type: gs_handler
      attr: in, required
      doc: Ground state handler.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void
sirius_generate_initial_density(void* const* handler__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& gs = get_gs(handler__);
                gs.density().initial_density();
            },
            error_code__);
}

/*
@api begin
sirius_generate_effective_potential:
  doc: Generate effective potential and magnetic field.
  arguments:
    handler:
      type: gs_handler
      attr: in, required
      doc: Ground state handler.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void
sirius_generate_effective_potential(void* const* handler__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& gs = get_gs(handler__);
                gs.potential().generate(gs.density(), gs.ctx().use_symmetry(), false);
            },
            error_code__);
}

/*
@api begin
sirius_generate_density:
  doc: Generate charge density and magnetization.
  arguments:
    gs_handler:
      type: gs_handler
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
    paw_only:
      type: bool
      attr: in, optional
      doc: it true, only local PAW density is generated
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void
sirius_generate_density(void* const* gs_handler__, bool const* add_core__, bool const* transform_to_rg__,
                        bool const* paw_only__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& gs             = get_gs(gs_handler__);
                auto add_core        = get_value<bool>(add_core__, false);
                auto transform_to_rg = get_value<bool>(transform_to_rg__, false);
                auto paw_only        = get_value<bool>(paw_only__, false);

                if (paw_only) {
                    gs.density().generate_paw_density();
                } else {
                    gs.density().generate<double>(gs.k_point_set(), gs.ctx().use_symmetry(), add_core, transform_to_rg);
                }
            },
            error_code__);
}

/*
@api begin
sirius_set_band_occupancies:
  doc: Set band occupancies.
  arguments:
    ks_handler:
      type: ks_handler
      attr: in, required
      doc: K-point set handler.
    ik:
      type: int
      attr: in, required
      doc: Global index of k-point.
    ispn:
      type: int
      attr: in, required
      doc: Spin component index.
    band_occupancies:
      type: double
      attr: in, required, dimension(:)
      doc: Array of band occupancies.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void
sirius_set_band_occupancies(void* const* ks_handler__, int const* ik__, int const* ispn__,
                            double const* band_occupancies__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& ks = get_ks(ks_handler__);
                int ik   = *ik__ - 1;
                for (int i = 0; i < ks.ctx().num_bands(); i++) {
                    ks.get<double>(ik)->band_occupancy(i, *ispn__ - 1, band_occupancies__[i]);
                }
            },
            error_code__);
}

/*
@api begin
sirius_get_band_occupancies:
  doc: Set band occupancies.
  arguments:
    ks_handler:
      type: ks_handler
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
      attr: out, required, dimension(:)
      doc: Array of band occupancies.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void
sirius_get_band_occupancies(void* const* ks_handler__, int const* ik__, int const* ispn__, double* band_occupancies__,
                            int* error_code__)
{
    call_sirius(
            [&]() {
                auto& ks = get_ks(ks_handler__);
                int ik   = *ik__ - 1;
                for (int i = 0; i < ks.ctx().num_bands(); i++) {
                    band_occupancies__[i] = ks.get<double>(ik)->band_occupancy(i, *ispn__ - 1);
                }
            },
            error_code__);
}

/*
@api begin
sirius_get_band_energies:
  doc: Get band energies.
  arguments:
    ks_handler:
      type: ks_handler
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
      attr: out, required, dimension(:)
      doc: Array of band energies.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void
sirius_get_band_energies(void* const* ks_handler__, int const* ik__, int const* ispn__, double* band_energies__,
                         int* error_code__)
{
    call_sirius(
            [&]() {
                auto& ks = get_ks(ks_handler__);
                int ik   = *ik__ - 1;
                for (int i = 0; i < ks.ctx().num_bands(); i++) {
                    band_energies__[i] = ks.get<double>(ik)->band_energy(i, *ispn__ - 1);
                }
            },
            error_code__);
}

/*
@api begin
sirius_get_energy:
  doc: Get one of the total energy components.
  arguments:
    handler:
      type: gs_handler
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
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void
sirius_get_energy(void* const* handler__, char const* label__, double* energy__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& gs = get_gs(handler__);

                auto& kset      = gs.k_point_set();
                auto& ctx       = kset.ctx();
                auto& unit_cell = kset.unit_cell();
                auto& potential = gs.potential();
                auto& density   = gs.density();

                std::string label(label__);

                std::map<std::string, std::function<double()>> func = {
                        {"total",
                         [&]() { return sirius::total_energy(ctx, kset, density, potential, gs.ewald_energy()); }},
                        {"evalsum", [&]() { return sirius::eval_sum(unit_cell, kset); }},
                        {"exc", [&]() { return sirius::energy_exc(density, potential); }},
                        {"vxc", [&]() { return sirius::energy_vxc(density, potential); }},
                        {"bxc", [&]() { return sirius::energy_bxc(density, potential); }},
                        {"veff", [&]() { return sirius::energy_veff(density, potential); }},
                        {"vloc", [&]() { return sirius::energy_vloc(density, potential); }},
                        {"vha", [&]() { return sirius::energy_vha(potential); }},
                        {"enuc", [&]() { return sirius::energy_enuc(ctx, potential); }},
                        {"kin", [&]() { return sirius::energy_kin(ctx, kset, density, potential); }},
                        {"one-el", [&]() { return sirius::one_electron_energy(density, potential); }},
                        {"descf", [&]() { return gs.scf_correction_energy(); }},
                        {"demet", [&]() { return kset.entropy_sum(); }},
                        {"paw-one-el", [&]() { return potential.PAW_one_elec_energy(density); }},
                        {"paw", [&]() { return potential.PAW_total_energy(density); }},
                        {"fermi", [&]() { return kset.energy_fermi(); }},
                        {"hubbard", [&]() { return sirius::hubbard_energy(density); }},
                        {"band-gap", [&]() { return kset.band_gap(); }}};

                if (!func.count(label)) {
                    RTE_THROW("wrong label: " + label);
                }

                *energy__ = func.at(label)();
            },
            error_code__);
}

/*
@api begin
sirius_get_forces:
  doc: Get one of the total force components.
  arguments:
    handler:
      type: gs_handler
      attr: in, required
      doc: DFT ground state handler.
    label:
      type: string
      attr: in, required
      doc: Label of the force component to get.
    forces:
      type: double
      attr: out, required, dimension(:,:)
      doc: Total force component for each atom.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void
sirius_get_forces(void* const* handler__, char const* label__, double* forces__, int* error_code__)
{
    call_sirius(
            [&]() {
                std::string label(label__);

                auto& gs = get_gs(handler__);

                auto get_forces = [&](mdarray<double, 2> const& sirius_forces__) {
                    for (size_t i = 0; i < sirius_forces__.size(); i++) {
                        forces__[i] = sirius_forces__[i];
                    }
                };

                auto& forces = gs.forces();

                std::map<std::string, mdarray<double, 2> const& (sirius::Force::*)(void)> func = {
                        {"total", &sirius::Force::calc_forces_total},
                        {"vloc", &sirius::Force::calc_forces_vloc},
                        {"core", &sirius::Force::calc_forces_core},
                        {"ewald", &sirius::Force::calc_forces_ewald},
                        {"nonloc", &sirius::Force::calc_forces_nonloc},
                        {"us", &sirius::Force::calc_forces_us},
                        {"usnl", &sirius::Force::calc_forces_usnl},
                        {"scf_corr", &sirius::Force::calc_forces_scf_corr},
                        {"hubbard", &sirius::Force::calc_forces_hubbard},
                        {"ibs", &sirius::Force::calc_forces_ibs},
                        {"hf", &sirius::Force::calc_forces_hf},
                        {"rho", &sirius::Force::calc_forces_rho}};

                if (!func.count(label)) {
                    RTE_THROW("wrong label (" + label + ") for the component of forces");
                }

                get_forces((forces.*func.at(label))());
            },
            error_code__);
}

/*
@api begin
sirius_get_stress_tensor:
  doc: Get one of the stress tensor components.
  arguments:
    handler:
      type: gs_handler
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
      doc: Error code.
@api end
*/
void
sirius_get_stress_tensor(void* const* handler__, char const* label__, double* stress_tensor__, int* error_code__)
{
    call_sirius(
            [&]() {
                std::string label(label__);

                auto& gs = get_gs(handler__);

                auto& stress_tensor = gs.stress();

                std::map<std::string, r3::matrix<double> (sirius::Stress::*)(void)> func = {
                        {"total", &sirius::Stress::calc_stress_total},
                        {"vloc", &sirius::Stress::calc_stress_vloc},
                        {"har", &sirius::Stress::calc_stress_har},
                        {"ewald", &sirius::Stress::calc_stress_ewald},
                        {"kin", &sirius::Stress::calc_stress_kin},
                        {"nonloc", &sirius::Stress::calc_stress_nonloc},
                        {"us", &sirius::Stress::calc_stress_us},
                        {"xc", &sirius::Stress::calc_stress_xc},
                        {"core", &sirius::Stress::calc_stress_core},
                        {"hubbard", &sirius::Stress::calc_stress_hubbard},
                };

                if (!func.count(label)) {
                    RTE_THROW("wrong label (" + label + ") for the component of stress tensor");
                }

                r3::matrix<double> s;

                s = (stress_tensor.*func.at(label))();

                for (int mu = 0; mu < 3; mu++) {
                    for (int nu = 0; nu < 3; nu++) {
                        stress_tensor__[nu + mu * 3] = s(mu, nu);
                    }
                }
            },
            error_code__);
}

/*
@api begin
sirius_get_num_beta_projectors:
  doc: Get the number of beta-projectors for an atom type.
  arguments:
    handler:
      type: ctx_handler
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
void
sirius_get_num_beta_projectors(void* const* handler__, char const* label__, int* num_bp__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& sim_ctx = get_sim_ctx(handler__);

                auto& type = sim_ctx.unit_cell().atom_type(std::string(label__));

                *num_bp__ = type.mt_basis_size();
            },
            error_code__);
}

/*
@api begin
sirius_get_wave_functions:
  doc: Get wave-functions.
  arguments:
    ks_handler:
      type: ks_handler
      attr: in, required
      doc: K-point set handler.
    vkl:
      type: double
      attr: in, optional, dimension(3)
      doc: Latttice coordinates of the k-point.
    spin:
      type: int
      attr: in, optional
      doc: Spin index in case of collinear magnetism.
    num_gvec_loc:
      type: int
      attr: in, optional
      doc: Local number of G-vectors for a k-point.
    gvec_loc:
      type: int
      attr: in, optional, dimension(:,:)
      doc: List of G-vectors.
    evec:
      type: complex
      attr: out, optional, dimension(:,:)
      doc: Wave-functions.
    ld:
      type: int
      attr: in, optional
      doc: Leading dimension of evec array.
    num_spin_comp:
      type: int
      attr: in, optional
      doc: Number of spin components.
    error_code:
      type: int
      attr: out, optional
      doc: Error code
@api end
*/
void
sirius_get_wave_functions(void* const* ks_handler__, double const* vkl__, int const* spin__, int const* num_gvec_loc__,
                          int const* gvec_loc__, std::complex<double>* evec__, int const* ld__,
                          int const* num_spin_comp__, int* error_code__)
{
    PROFILE("sirius_api::sirius_get_wave_functions");

    // TODO: refactor this part; use QE order of G-vectors

    auto gvec_mapping = [&](sirius::fft::Gvec const& gkvec) {
        std::vector<int> igm(*num_gvec_loc__);

        mdarray<int, 2> gv({3, *num_gvec_loc__}, const_cast<int*>(gvec_loc__));

        /* go in the order of host code */
        for (int ig = 0; ig < *num_gvec_loc__; ig++) {
            ///* G vector of host code */
            // auto gvc = dot(kset.ctx().unit_cell().reciprocal_lattice_vectors(),
            //               (r3::vector<double>(gvec_k(0, ig), gvec_k(1, ig), gvec_k(2, ig)) + gkvec.vk()));
            // if (gvc.length() > kset.ctx().gk_cutoff()) {
            //    continue;
            //}
            int ig1 = gkvec.index_by_gvec({gv(0, ig), gv(1, ig), gv(2, ig)});
            /* index of G was not found */
            if (ig1 < 0) {
                /* try -G */
                ig1 = gkvec.index_by_gvec({-gv(0, ig), -gv(1, ig), -gv(2, ig)});
                /* index of -G was not found */
                if (ig1 < 0) {
                    RTE_THROW("index of G-vector is not found");
                } else {
                    /* this will tell to conjugate PW coefficients as we take them from -G index */
                    igm[ig] = -ig1;
                }
            } else {
                igm[ig] = ig1;
            }
        }
        return igm;
    };

    call_sirius(
            [&]() {
                auto& ks = get_ks(ks_handler__);

                auto& sim_ctx = ks.ctx();

                std::vector<int> buf(ks.comm().size());

                int jk{-1};
                if (vkl__) {
                    jk = ks.find_kpoint(vkl__);
                    if (jk == -1) {
                        std::stringstream s;
                        s << "k-point is not found";
                        RTE_THROW(s);
                    }
                }
                ks.comm().allgather(&jk, buf.data(), 1, ks.comm().rank());
                int dest_rank{-1};
                for (int i = 0; i < ks.comm().size(); i++) {
                    if (buf[i] >= 0) {
                        dest_rank = i;
                        jk        = buf[i];
                        break;
                    }
                }
                int num_spin_comp{-1};
                if (num_spin_comp__) {
                    num_spin_comp = *num_spin_comp__;
                    if (!(num_spin_comp == 1 || num_spin_comp == 2)) {
                        RTE_THROW("wrong number of spin components");
                    }
                }
                ks.comm().bcast(&num_spin_comp, 1, dest_rank);
                if ((sim_ctx.num_mag_dims() == 3 && num_spin_comp != 2) ||
                    (sim_ctx.num_mag_dims() != 3 && num_spin_comp == 2)) {
                    RTE_THROW("inconsistent number of spin components");
                }

                int spin{-1};
                if (spin__) {
                    spin = *spin__ - 1;
                    if (!(spin == 0 || spin == 1)) {
                        RTE_THROW("wrong spin index");
                    }
                }
                ks.comm().bcast(&spin, 1, dest_rank);

                /* rank where k-point vkl resides on the SIRIUS side */
                int src_rank = ks.spl_num_kpoints().location(typename sirius::kp_index_t::global(jk)).ib;

                if (ks.comm().rank() == src_rank || ks.comm().rank() == dest_rank) {
                    /* send G+k copy to destination rank (where host code receives the data) */
                    auto gkvec = ks.get_gkvec(typename sirius::kp_index_t::global(jk), dest_rank);

                    mdarray<std::complex<double>, 2> wf;
                    if (ks.comm().rank() == dest_rank) {
                        /* check number of G+k vectors */
                        int ngk = *num_gvec_loc__;
                        gkvec.comm().allreduce(&ngk, 1);
                        if (ngk != gkvec.num_gvec()) {
                            r3::vector<double> vkl(vkl__);
                            std::stringstream s;
                            s << "wrong number of G+k vectors for k-point " << vkl << ", jk = " << jk << std::endl
                              << "expected number : " << gkvec.num_gvec() << std::endl
                              << "actual number   : " << ngk << std::endl
                              << "local number of G+k vectors passed by rank " << gkvec.comm().rank() << " is "
                              << *num_gvec_loc__;
                            RTE_THROW(s);
                        }
                        wf = mdarray<std::complex<double>, 2>({gkvec.count(), sim_ctx.num_bands()});
                    }

                    int ispn0{0};
                    int ispn1{1};
                    /* fetch two components in non-collinear case, otherwise fetch only one component */
                    if (sim_ctx.num_mag_dims() != 3) {
                        ispn0 = ispn1 = spin;
                    }
                    /* send wave-functions for each spin channel */
                    for (int s = ispn0; s <= ispn1; s++) {
                        int tag = mpi::Communicator::get_tag(src_rank, dest_rank) + s;
                        mpi::Request req;

                        /* send wave-functions */
                        if (ks.comm().rank() == src_rank) {
                            auto kp   = ks.get<double>(jk);
                            int count = kp->spinor_wave_functions().ld();
                            req       = ks.comm().isend(kp->spinor_wave_functions().at(memory_t::host, 0,
                                                                                       sirius::wf::spin_index(0),
                                                                                       sirius::wf::band_index(0)),
                                                        count * sim_ctx.num_bands(), dest_rank, tag);
                        }
                        /* receive wave-functions */
                        if (ks.comm().rank() == dest_rank) {
                            int count = gkvec.count();
                            /* receive the array with wave-functions */
                            ks.comm().recv(&wf(0, 0), count * sim_ctx.num_bands(), src_rank, tag);

                            std::vector<std::complex<double>> wf_tmp(gkvec.num_gvec());
                            int offset = gkvec.offset();
                            mdarray<std::complex<double>, 3> evec({*ld__, num_spin_comp, sim_ctx.num_bands()}, evec__);

                            auto igmap = gvec_mapping(gkvec);

                            auto store_wf = [&](std::vector<std::complex<double>>& wf_tmp, int i, int s) {
                                int ispn = s;
                                if (sim_ctx.num_mag_dims() == 1) {
                                    ispn = 0;
                                }
                                for (int ig = 0; ig < *num_gvec_loc__; ig++) {
                                    int ig1 = igmap[ig];
                                    std::complex<double> z;
                                    if (ig1 < 0) {
                                        z = std::conj(wf_tmp[-ig1]);
                                    } else {
                                        z = wf_tmp[ig1];
                                    }
                                    evec(ig, ispn, i) = z;
                                }
                            };

                            /* store wave-functions */
                            for (int i = 0; i < sim_ctx.num_bands(); i++) {
                                /* gather full column of PW coefficients */
                                sim_ctx.comm_band().allgather(&wf(0, i), wf_tmp.data(), count, offset);
                                store_wf(wf_tmp, i, s);
                            }
                        }
                        if (ks.comm().rank() == src_rank) {
                            req.wait();
                        }
                    }
                }
            },
            error_code__);
}

/*
@api begin
sirius_add_atom_type_aw_descriptor:
  doc: Add descriptor of the augmented wave radial function.
  arguments:
    handler:
      type: ctx_handler
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
    error_code:
      type: int
      attr: out, optional
      doc: Error code
@api end
*/
void
sirius_add_atom_type_aw_descriptor(void* const* handler__, char const* label__, int const* n__, int const* l__,
                                   double const* enu__, int const* dme__, bool const* auto_enu__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& sim_ctx = get_sim_ctx(handler__);
                auto& type    = sim_ctx.unit_cell().atom_type(std::string(label__));
                type.add_aw_descriptor(*n__, *l__, *enu__, *dme__, *auto_enu__);
            },
            error_code__);
}

/*
@api begin
sirius_add_atom_type_lo_descriptor:
  doc: Add descriptor of the local orbital radial function.
  arguments:
    handler:
      type: ctx_handler
      attr: in, required
      doc: Simulation context handler.
    label:
      type: string
      attr: in, required
      doc: Atom type label.
    ilo:
      type: int
      attr: in, required
      doc: Index of the local orbital to which the descriptor is added.
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
    error_code:
      type: int
      attr: out, optional
      doc: Error code
@api end
*/
void
sirius_add_atom_type_lo_descriptor(void* const* handler__, char const* label__, int const* ilo__, int const* n__,
                                   int const* l__, double const* enu__, int const* dme__, bool const* auto_enu__,
                                   int* error_code__)
{
    call_sirius(
            [&]() {
                auto& sim_ctx = get_sim_ctx(handler__);
                auto& type    = sim_ctx.unit_cell().atom_type(std::string(label__));
                type.add_lo_descriptor(*ilo__ - 1, *n__, *l__, *enu__, *dme__, *auto_enu__);
            },
            error_code__);
}

/*
@api begin
sirius_set_atom_type_configuration:
  doc: Set configuration of atomic levels.
  arguments:
    handler:
      type: ctx_handler
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
    error_code:
      type: int
      attr: out, optional
      doc: Error code
@api end
*/
void
sirius_set_atom_type_configuration(void* const* handler__, char const* label__, int const* n__, int const* l__,
                                   int const* k__, double const* occupancy__, bool const* core__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& sim_ctx = get_sim_ctx(handler__);
                auto& type    = sim_ctx.unit_cell().atom_type(std::string(label__));
                type.set_configuration(*n__, *l__, *k__, *occupancy__, *core__);
            },
            error_code__);
}

/*
@api begin
sirius_generate_coulomb_potential:
  doc: Generate Coulomb potential by solving Poisson equation
  arguments:
    handler:
      type: gs_handler
      attr: in, required
      doc: DFT ground state handler
    vh_el:
      type: double
      attr: out, optional, dimension(:)
      doc: Electronic part of Hartree potential at each atom's origin.
    error_code:
      type: int
      attr: out, optional
      doc: Error code
@api end
*/
void
sirius_generate_coulomb_potential(void* const* handler__, double* vh_el__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& gs = get_gs(handler__);

                gs.density().rho().rg().fft_transform(-1);
                gs.potential().poisson(gs.density().rho());

                if (vh_el__) {
                    for (int ia = 0; ia < gs.ctx().unit_cell().num_atoms(); ia++) {
                        vh_el__[ia] = gs.potential().vh_el(ia);
                    }
                }
            },
            error_code__);
}

/*
@api begin
sirius_generate_xc_potential:
  doc: Generate XC potential using LibXC
  arguments:
    handler:
      type: gs_handler
      attr: in, required
      doc: Ground state handler
    error_code:
      type: int
      attr: out, optional
      doc: Error code
@api end
*/
void
sirius_generate_xc_potential(void* const* handler__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& gs = get_gs(handler__);
                gs.potential().xc(gs.density());
            },
            error_code__);
}

/*
@api begin
sirius_get_kpoint_inter_comm:
  doc: Get communicator which is used to split k-points
  arguments:
    handler:
      type: ctx_handler
      attr: in, required
      doc: Simulation context handler
    fcomm:
      type: int
      attr: out, required
      doc: Fortran communicator
    error_code:
      type: int
      attr: out, optional
      doc: Error code
@api end
*/
void
sirius_get_kpoint_inter_comm(void* const* handler__, int* fcomm__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& sim_ctx = get_sim_ctx(handler__);
                *fcomm__      = MPI_Comm_c2f(sim_ctx.comm_k().native());
            },
            error_code__);
}

/*
@api begin
sirius_get_kpoint_inner_comm:
  doc: Get communicator which is used to parallise band problem
  arguments:
    handler:
      type: ctx_handler
      attr: in, required
      doc: Simulation context handler
    fcomm:
      type: int
      attr: out, required
      doc: Fortran communicator
    error_code:
      type: int
      attr: out, optional
      doc: Error code
@api end
*/
void
sirius_get_kpoint_inner_comm(void* const* handler__, int* fcomm__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& sim_ctx = get_sim_ctx(handler__);
                *fcomm__      = MPI_Comm_c2f(sim_ctx.comm_band().native());
            },
            error_code__);
}

/*
@api begin
sirius_get_fft_comm:
  doc: Get communicator which is used to parallise FFT
  arguments:
    handler:
      type: ctx_handler
      attr: in, required
      doc: Simulation context handler
    fcomm:
      type: int
      attr: out, required
      doc: Fortran communicator
    error_code:
      type: int
      attr: out, optional
      doc: Error code
@api end
*/
void
sirius_get_fft_comm(void* const* handler__, int* fcomm__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& sim_ctx = get_sim_ctx(handler__);
                *fcomm__      = MPI_Comm_c2f(sim_ctx.comm_fft().native());
            },
            error_code__);
}

/*
@api begin
sirius_get_num_gvec:
  doc: Get total number of G-vectors on the fine grid.
  arguments:
    handler:
      type: ctx_handler
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
void
sirius_get_num_gvec(void* const* handler__, int* num_gvec__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& sim_ctx = get_sim_ctx(handler__);
                *num_gvec__   = sim_ctx.gvec().num_gvec();
            },
            error_code__);
}

/*
@api begin
sirius_get_gvec_arrays:
  doc: Get G-vector arrays.
  arguments:
    handler:
      type: ctx_handler
      attr: in, required
      doc: Simulation context handler
    gvec:
      type: int
      attr: in, optional, dimension(:,:)
      doc: G-vectors in lattice coordinates.
    gvec_cart:
      type: double
      attr: in, optional, dimension(:,:)
      doc: G-vectors in Cartesian coordinates.
    gvec_len:
      type: double
      attr: in, optional, dimension(:)
      doc: Length of G-vectors.
    index_by_gvec:
      type: int
      attr: in, optional, dimension(:,:,:)
      doc: G-vector index by lattice coordinates.
    error_code:
      type: int
      attr: out, optional
      doc: Error code
@api end
*/
void
sirius_get_gvec_arrays(void* const* handler__, int* gvec__, double* gvec_cart__, double* gvec_len__,
                       int* index_by_gvec__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& sim_ctx = get_sim_ctx(handler__);

                if (gvec__ != nullptr) {
                    mdarray<int, 2> gvec({3, sim_ctx.gvec().num_gvec()}, gvec__);
                    for (int ig = 0; ig < sim_ctx.gvec().num_gvec(); ig++) {
                        auto gv = sim_ctx.gvec().gvec<index_domain_t::global>(ig);
                        for (int x : {0, 1, 2}) {
                            gvec(x, ig) = gv[x];
                        }
                    }
                }
                if (gvec_cart__ != nullptr) {
                    mdarray<double, 2> gvec_cart({3, sim_ctx.gvec().num_gvec()}, gvec_cart__);
                    for (int ig = 0; ig < sim_ctx.gvec().num_gvec(); ig++) {
                        auto gvc = sim_ctx.gvec().gvec_cart<index_domain_t::global>(ig);
                        for (int x : {0, 1, 2}) {
                            gvec_cart(x, ig) = gvc[x];
                        }
                    }
                }
                if (gvec_len__ != nullptr) {
                    for (int ig = 0; ig < sim_ctx.gvec().num_gvec(); ig++) {
                        gvec_len__[ig] = sim_ctx.gvec().gvec_len<index_domain_t::global>(ig);
                    }
                }
                if (index_by_gvec__ != nullptr) {
                    auto d0 = sim_ctx.fft_grid().limits(0);
                    auto d1 = sim_ctx.fft_grid().limits(1);
                    auto d2 = sim_ctx.fft_grid().limits(2);

                    mdarray<int, 3> index_by_gvec({index_range(d0.first, d0.second + 1),
                                                   index_range(d1.first, d1.second + 1),
                                                   index_range(d2.first, d2.second + 1)},
                                                  index_by_gvec__);
                    std::fill(index_by_gvec.at(memory_t::host), index_by_gvec.at(memory_t::host) + index_by_gvec.size(),
                              -1);

                    for (int ig = 0; ig < sim_ctx.gvec().num_gvec(); ig++) {
                        auto G = sim_ctx.gvec().gvec<index_domain_t::global>(ig);

                        index_by_gvec(G[0], G[1], G[2]) = ig + 1;
                    }
                }
            },
            error_code__);
}

/*
@api begin
sirius_get_num_fft_grid_points:
  doc: Get local number of FFT grid points.
  arguments:
    handler:
      type: ctx_handler
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
void
sirius_get_num_fft_grid_points(void* const* handler__, int* num_fft_grid_points__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& sim_ctx          = get_sim_ctx(handler__);
                *num_fft_grid_points__ = sim_ctx.spfft<double>().local_slice_size();
            },
            error_code__);
}

/*
@api begin
sirius_get_fft_index:
  doc: Get mapping between G-vector index and FFT index
  arguments:
    handler:
      type: ctx_handler
      attr: in, required
      doc: Simulation context handler
    fft_index:
      type: int
      attr: out, required, dimension(:)
      doc: Index inside FFT buffer
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void
sirius_get_fft_index(void* const* handler__, int* fft_index__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& sim_ctx = get_sim_ctx(handler__);
                for (int ig = 0; ig < sim_ctx.gvec().num_gvec(); ig++) {
                    auto G          = sim_ctx.gvec().gvec<index_domain_t::global>(ig);
                    fft_index__[ig] = sim_ctx.fft_grid().index_by_freq(G[0], G[1], G[2]) + 1;
                }
            },
            error_code__);
}

/*
@api begin
sirius_get_max_num_gkvec:
  doc: Get maximum number of G+k vectors across all k-points in the set
  arguments:
    ks_handler:
      type: ks_handler
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
void
sirius_get_max_num_gkvec(void* const* ks_handler__, int* max_num_gkvec__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& ks         = get_ks(ks_handler__);
                *max_num_gkvec__ = ks.max_num_gkvec();
            },
            error_code__);
}

/*
@api begin
sirius_get_gkvec_arrays:
  doc: Get all G+k vector related arrays
  arguments:
    ks_handler:
      type: ks_handler
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
      attr: out, required, dimension(:)
      doc: Index of the G-vector part of G+k vector.
    gkvec:
      type: double
      attr: out, required, dimension(:,:)
      doc: G+k vectors in fractional coordinates.
    gkvec_cart:
      type: double
      attr: out, required, dimension(:,:)
      doc: G+k vectors in Cartesian coordinates.
    gkvec_len:
      type: double
      attr: out, required, dimension(:)
      doc: Length of G+k vectors.
    gkvec_tp:
      type: double
      attr: out, required, dimension(:,:)
      doc: Theta and Phi angles of G+k vectors.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void
sirius_get_gkvec_arrays(void* const* ks_handler__, int* ik__, int* num_gkvec__, int* gvec_index__, double* gkvec__,
                        double* gkvec_cart__, double* gkvec_len, double* gkvec_tp__, int* error_code__)
{

    call_sirius(
            [&]() {
                auto& ks = get_ks(ks_handler__);

                auto kp = ks.get<double>(*ik__ - 1);

                /* get rank that stores a given k-point */
                int rank = ks.spl_num_kpoints().location(kp_index_t::global(*ik__ - 1)).ib;

                auto& comm_k = ks.ctx().comm_k();

                if (rank == comm_k.rank()) {
                    *num_gkvec__ = kp->num_gkvec();
                    mdarray<double, 2> gkvec({3, kp->num_gkvec()}, gkvec__);
                    mdarray<double, 2> gkvec_cart({3, kp->num_gkvec()}, gkvec_cart__);
                    mdarray<double, 2> gkvec_tp({2, kp->num_gkvec()}, gkvec_tp__);

                    for (int igk = 0; igk < kp->num_gkvec(); igk++) {
                        auto gkc = kp->gkvec().gkvec_cart<index_domain_t::global>(igk);
                        auto G   = kp->gkvec().gvec<index_domain_t::global>(igk);

                        gvec_index__[igk] = ks.ctx().gvec().index_by_gvec(G) + 1; // Fortran counts from 1
                        for (int x : {0, 1, 2}) {
                            gkvec(x, igk)      = kp->gkvec().template gkvec<index_domain_t::global>(igk)[x];
                            gkvec_cart(x, igk) = gkc[x];
                        }
                        auto rtp         = r3::spherical_coordinates(gkc);
                        gkvec_len[igk]   = rtp[0];
                        gkvec_tp(0, igk) = rtp[1];
                        gkvec_tp(1, igk) = rtp[2];
                    }
                }
                comm_k.bcast(num_gkvec__, 1, rank);
                comm_k.bcast(gvec_index__, *num_gkvec__, rank);
                comm_k.bcast(gkvec__, *num_gkvec__ * 3, rank);
                comm_k.bcast(gkvec_cart__, *num_gkvec__ * 3, rank);
                comm_k.bcast(gkvec_len, *num_gkvec__, rank);
                comm_k.bcast(gkvec_tp__, *num_gkvec__ * 2, rank);
            },
            error_code__);
}

/*
@api begin
sirius_get_step_function:
  doc: Get the unit-step function.
  arguments:
    handler:
      type: ctx_handler
      attr: in, required
      doc: Simulation context handler
    cfunig:
      type: complex
      attr: out, required, dimension(:)
      doc: Plane-wave coefficients of step function.
    cfunrg:
      type: double
      attr: out, required, dimension(:)
      doc: Values of the step function on the regular grid.
    num_rg_points:
      type: int
      attr: in, required
      doc: Number of real-space points.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void
sirius_get_step_function(void* const* handler__, std::complex<double>* cfunig__, double* cfunrg__, int* num_rg_points__,
                         int* error_code__) // TODO: generalise with get_periodic_function
{
    call_sirius(
            [&]() {
                auto& sim_ctx = get_sim_ctx(handler__);
                for (int ig = 0; ig < sim_ctx.gvec().num_gvec(); ig++) {
                    cfunig__[ig] = sim_ctx.theta_pw(ig);
                }
                auto& fft = sim_ctx.spfft<double>();

                // silence a gcc warning.
                bool is_local_rg{false};
                if (*num_rg_points__ == static_cast<int>(fft::spfft_grid_size(fft))) {
                    is_local_rg = false;
                } else if (*num_rg_points__ == static_cast<int>(fft::spfft_grid_size_local(fft))) {
                    is_local_rg = true;
                } else {
                    RTE_THROW("wrong number of real space points");
                }

                int offs = (is_local_rg) ? 0 : fft.dim_x() * fft.dim_y() * fft.local_z_offset();
                if (fft.local_slice_size()) {
                    for (int i = 0; i < fft.local_slice_size(); i++) {
                        cfunrg__[offs + i] = sim_ctx.theta(i);
                    }
                }
                if (!is_local_rg) {
                    mpi::Communicator(fft.communicator()).allgather(cfunrg__, fft.local_slice_size(), offs);
                }
            },
            error_code__);
}

/*
@api begin
sirius_set_h_radial_integrals:
  doc: Set LAPW Hamiltonian radial integrals.
  arguments:
    handler:
      type: ctx_handler
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
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void
sirius_set_h_radial_integrals(void* const* handler__, int* ia__, int* lmmax__, double* val__, int* l1__, int* o1__,
                              int* ilo1__, int* l2__, int* o2__, int* ilo2__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& sim_ctx = get_sim_ctx(handler__);
                int ia        = *ia__ - 1;
                int idxrf1{-1};
                int idxrf2{-1};
                if ((l1__ != nullptr && o1__ != nullptr && ilo1__ != nullptr) ||
                    (l2__ != nullptr && o2__ != nullptr && ilo2__ != nullptr)) {
                    RTE_THROW("wrong combination of radial function indices");
                }
                if (l1__ != nullptr && o1__ != nullptr) {
                    idxrf1 = sim_ctx.unit_cell().atom(ia).type().indexr_by_l_order(*l1__, *o1__ - 1);
                } else if (ilo1__ != nullptr) {
                    idxrf1 = sim_ctx.unit_cell().atom(ia).type().indexr_by_idxlo(*ilo1__ - 1);
                } else {
                    RTE_THROW("1st radial function index is not valid");
                }

                if (l2__ != nullptr && o2__ != nullptr) {
                    idxrf2 = sim_ctx.unit_cell().atom(ia).type().indexr_by_l_order(*l2__, *o2__ - 1);
                } else if (ilo2__ != nullptr) {
                    idxrf2 = sim_ctx.unit_cell().atom(ia).type().indexr_by_idxlo(*ilo2__ - 1);
                } else {
                    RTE_THROW("2nd radial function index is not valid");
                }

                for (int lm = 0; lm < *lmmax__; lm++) {
                    sim_ctx.unit_cell().atom(ia).h_radial_integrals(idxrf1, idxrf2)[lm] = val__[lm];
                }
            },
            error_code__);
}

/*
@api begin
sirius_set_o_radial_integral:
  doc: Set LAPW overlap radial integral.
  arguments:
    handler:
      type: ctx_handler
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
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void
sirius_set_o_radial_integral(void* const* handler__, int* ia__, double* val__, int* l__, int* o1__, int* ilo1__,
                             int* o2__, int* ilo2__, int* error_code__)
{

    call_sirius(
            [&]() {
                auto& sim_ctx = get_sim_ctx(handler__);
                int ia        = *ia__ - 1;
                if ((o1__ != nullptr && ilo1__ != nullptr) || (o2__ != nullptr && ilo2__ != nullptr)) {
                    RTE_THROW("wrong combination of radial function indices");
                }

                if (o1__ != nullptr && ilo2__ != nullptr) {
                    int idxrf2 = sim_ctx.unit_cell().atom(ia).type().indexr_by_idxlo(*ilo2__ - 1);
                    int order2 = sim_ctx.unit_cell().atom(ia).type().indexr(idxrf2).order;
                    sim_ctx.unit_cell().atom(ia).symmetry_class().set_o_radial_integral(*l__, *o1__ - 1, order2,
                                                                                        *val__);
                }

                if (o2__ != nullptr && ilo1__ != nullptr) {
                    int idxrf1 = sim_ctx.unit_cell().atom(ia).type().indexr_by_idxlo(*ilo1__ - 1);
                    int order1 = sim_ctx.unit_cell().atom(ia).type().indexr(idxrf1).order;
                    sim_ctx.unit_cell().atom(ia).symmetry_class().set_o_radial_integral(*l__, order1, *o2__ - 1,
                                                                                        *val__);
                }

                if (ilo1__ != nullptr && ilo2__ != nullptr) {
                    int idxrf1 = sim_ctx.unit_cell().atom(ia).type().indexr_by_idxlo(*ilo1__ - 1);
                    int order1 = sim_ctx.unit_cell().atom(ia).type().indexr(idxrf1).order;
                    int idxrf2 = sim_ctx.unit_cell().atom(ia).type().indexr_by_idxlo(*ilo2__ - 1);
                    int order2 = sim_ctx.unit_cell().atom(ia).type().indexr(idxrf2).order;
                    sim_ctx.unit_cell().atom(ia).symmetry_class().set_o_radial_integral(*l__, order1, order2, *val__);
                }
            },
            error_code__);
}

/*
@api begin
sirius_set_o1_radial_integral:
  doc: Set a correction to LAPW overlap radial integral.
  arguments:
    handler:
      type: ctx_handler
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
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void
sirius_set_o1_radial_integral(void* const* handler__, int* ia__, double* val__, int* l1__, int* o1__, int* ilo1__,
                              int* l2__, int* o2__, int* ilo2__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& sim_ctx = get_sim_ctx(handler__);
                int ia        = *ia__ - 1;
                int idxrf1{-1};
                int idxrf2{-1};
                if ((l1__ != nullptr && o1__ != nullptr && ilo1__ != nullptr) ||
                    (l2__ != nullptr && o2__ != nullptr && ilo2__ != nullptr)) {
                    RTE_THROW("wrong combination of radial function indices");
                }
                if (l1__ != nullptr && o1__ != nullptr) {
                    idxrf1 = sim_ctx.unit_cell().atom(ia).type().indexr_by_l_order(*l1__, *o1__ - 1);
                } else if (ilo1__ != nullptr) {
                    idxrf1 = sim_ctx.unit_cell().atom(ia).type().indexr_by_idxlo(*ilo1__ - 1);
                } else {
                    RTE_THROW("1st radial function index is not valid");
                }

                if (l2__ != nullptr && o2__ != nullptr) {
                    idxrf2 = sim_ctx.unit_cell().atom(ia).type().indexr_by_l_order(*l2__, *o2__ - 1);
                } else if (ilo2__ != nullptr) {
                    idxrf2 = sim_ctx.unit_cell().atom(ia).type().indexr_by_idxlo(*ilo2__ - 1);
                } else {
                    RTE_THROW("2nd radial function index is not valid");
                }
                sim_ctx.unit_cell().atom(ia).symmetry_class().set_o1_radial_integral(idxrf1, idxrf2, *val__);
            },
            error_code__);
}

/*
@api begin
sirius_set_radial_function:
  doc: Set LAPW radial functions
  arguments:
    handler:
      type: ctx_handler
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
      attr: in, required, dimension(:)
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
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void
sirius_set_radial_function(void* const* handler__, int const* ia__, int const* deriv_order__, double const* f__,
                           int const* l__, int const* o__, int const* ilo__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& sim_ctx = get_sim_ctx(handler__);

                int ia = *ia__ - 1;

                auto& atom = sim_ctx.unit_cell().atom(ia);
                int n      = atom.num_mt_points();

                if (l__ != nullptr && o__ != nullptr && ilo__ != nullptr) {
                    RTE_THROW("wrong combination of radial function indices");
                }
                if (!(*deriv_order__ == 0 || *deriv_order__ == 1)) {
                    RTE_THROW("wrond radial derivative order");
                }

                int idxrf{-1};
                if (l__ != nullptr && o__ != nullptr) {
                    idxrf = atom.type().indexr_by_l_order(*l__, *o__ - 1);
                } else if (ilo__ != nullptr) {
                    idxrf = atom.type().indexr_by_idxlo(*ilo__ - 1);
                } else {
                    RTE_THROW("radial function index is not valid");
                }

                if (*deriv_order__ == 0) {
                    atom.symmetry_class().radial_function(idxrf, std::vector<double>(f__, f__ + n));
                } else {
                    std::vector<double> f(n);
                    for (int ir = 0; ir < n; ir++) {
                        f[ir] = f__[ir] * atom.type().radial_grid()[ir];
                    }
                    atom.symmetry_class().radial_function_derivative(idxrf, f);
                }
                if (l__ != nullptr && o__ != nullptr) {
                    atom.symmetry_class().aw_surface_deriv(*l__, *o__ - 1, *deriv_order__, f__[n - 1]);
                }
            },
            error_code__);
}

/*
@api begin
sirius_set_equivalent_atoms:
  doc: Set equivalent atoms.
  arguments:
    handler:
      type: ctx_handler
      attr: in, required
      doc: Simulation context handler.
    equivalent_atoms:
      type: int
      attr: in, required, dimension(:)
      doc: Array with equivalent atom IDs.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void
sirius_set_equivalent_atoms(void* const* handler__, int* equivalent_atoms__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& sim_ctx = get_sim_ctx(handler__);
                sim_ctx.unit_cell().set_equivalent_atoms(equivalent_atoms__);
            },
            error_code__);
}

/*
@api begin
sirius_update_atomic_potential:
  doc: Set the new spherical potential.
  arguments:
    handler:
      type: gs_handler
      attr: in, required
      doc: Ground state handler.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void
sirius_update_atomic_potential(void* const* handler__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& gs = get_gs(handler__);
                gs.potential().update_atomic_potential();
            },
            error_code__);
}

/*
@api begin
sirius_option_get_number_of_sections:
  doc: Return the total number of sections defined in the input JSON schema.
  arguments:
    length:
      type: int
      attr: out, required
      doc: Number of sections.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void
sirius_option_get_number_of_sections(int* length__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto const& dict = get_options_dictionary();
                *length__        = static_cast<int>(dict["properties"].size());
            },
            error_code__);
}

/*
@api begin
sirius_option_get_section_name:
  doc: Return the name of a given section.
  arguments:
    elem:
      type: int
      attr: in, required, value
      doc: Index of the section (starting from 1).
    section_name:
      type: string
      attr: out, required
      doc: Name of the section
    section_name_length:
      type: int
      attr: in, required, value
      doc: Maximum length of the output string. Enough capacity should be provided.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void
sirius_option_get_section_name(int elem__, char* section_name__, int section_name_length__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto const& dict = sirius::get_options_dictionary();

                /* initialize the string to zero; the fortran interface takes care of
                   finishing the string with the proper character for Fortran */

                std::fill(section_name__, section_name__ + section_name_length__, 0);
                auto it = dict["properties"].begin();
                /* we can't do pointer arighmetics on the iterator */
                for (int i = 0; i < elem__ - 1; i++, it++)
                    ;
                auto key = it.key();
                if (static_cast<int>(key.size()) > section_name_length__ - 1) {
                    std::stringstream s;
                    s << "section name '" << key << "' is too large to fit into output string of size "
                      << section_name_length__;
                    RTE_THROW(s);
                }
                std::copy(key.begin(), key.end(), section_name__);
            },
            error_code__);
}

/*
@api begin
sirius_option_get_section_length:
  doc: Return the number of options in a given section.
  arguments:
    section:
      type: string
      attr: in, required
      doc: Name of the seciton.
    length:
      type: int
      attr: out, required
      doc: Number of options contained in the section.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void
sirius_option_get_section_length(char const* section__, int* length__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto section = std::string(section__);
                std::transform(section.begin(), section.end(), section.begin(), ::tolower);
                auto const& parser = sirius::get_section_options(section);
                *length__          = static_cast<int>(parser.size());
            },
            error_code__);
}

/*
@api begin
sirius_option_get_info:
  doc: Return information about the option.
  arguments:
    section:
      type: string
      attr: in, required
      doc: Name of the section.
    elem:
      type: int
      attr: in, required, value
      doc: Index of the option (starting from 1)
    key_name:
      type: string
      attr: out, required
      doc: Name of the option.
    key_name_len:
      type: int
      attr: in, required, value
      doc: Maximum length for the string (on the caller side). No allocation is done.
    type:
      type: int
      attr: out, required
      doc: Type of the option (real, integer, boolean, string, or array of the same types).
    length:
      type: int
      attr: out, required
      doc: Length of the default value (1 for the scalar types, otherwise the lenght of the array).
    enum_size:
      type: int
      attr: out, required
      doc: Number of elements in the enum type, zero otherwise.
    title:
      type: string
      attr: out, required
      doc: Short description of the option (can be empty).
    title_len:
      type: int
      attr: in, required, value
      doc: Maximum length for the short description.
    description:
      type: string
      attr: out, required
      doc: Detailed description of the option (can be empty).
    description_len:
      type: int
      attr: in, required, value
      doc: Maximum length for the detailed description.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/

void
sirius_option_get_info(char const* section__, int elem__, char* key_name__, int key_name_len__, int* type__,
                       int* length__, int* enum_size__, char* title__, int title_len__, char* description__,
                       int description_len__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto section = std::string(section__);
                std::transform(section.begin(), section.end(), section.begin(), ::tolower);
                auto const& dict = sirius::get_section_options(section);

                std::map<std::string, option_type_t> type_list = {
                        {"string", option_type_t::STRING_TYPE},
                        {"number", option_type_t::NUMBER_TYPE},
                        {"integer", option_type_t::INTEGER_TYPE},
                        {"boolean", option_type_t::LOGICAL_TYPE},
                        {"array", option_type_t::ARRAY_TYPE},
                        {"object", option_type_t::OBJECT_TYPE},
                        {"number_array_type", option_type_t::NUMBER_ARRAY_TYPE},
                        {"boolean_array_type", option_type_t::LOGICAL_ARRAY_TYPE},
                        {"integer_array_type", option_type_t::INTEGER_ARRAY_TYPE},
                        {"string_array_type", option_type_t::STRING_ARRAY_TYPE},
                        {"object_array_type", option_type_t::OBJECT_ARRAY_TYPE},
                        {"array_array_type", option_type_t::ARRAY_ARRAY_TYPE}};

                std::fill(key_name__, key_name__ + key_name_len__, 0);
                std::fill(title__, title__ + title_len__, 0);
                std::fill(description__, description__ + description_len__, 0);

                auto it = dict.begin();
                /* we can't do pointer arighmetics on the iterator */
                for (int i = 0; i < elem__ - 1; i++, it++)
                    ;
                auto key = it.key();
                if (!dict[key].count("default")) {
                    RTE_THROW("the default value is missing for key '" + key + "' in section '" + section + "'");
                }
                if (dict[key]["type"] == "array") {
                    std::string tmp = dict[key]["items"]["type"].get<std::string>() + "_array_type";
                    *type__         = static_cast<int>(type_list[tmp]);
                    *length__       = static_cast<int>(dict[key]["default"].size());
                    *enum_size__    = 0;
                } else {
                    *type__   = static_cast<int>(type_list[dict[key]["type"].get<std::string>()]);
                    *length__ = 1;
                    if (dict[key].count("enum")) {
                        *enum_size__ = static_cast<int>(dict[key]["enum"].size());
                    } else {
                        *enum_size__ = 0;
                    }
                }
                if (static_cast<int>(key.size()) < key_name_len__ - 1) {
                    std::copy(key.begin(), key.end(), key_name__);
                } else {
                    RTE_THROW("the key_name string variable needs to be large enough to contain the full option name");
                }

                if (dict[key].count("title")) {
                    auto title = dict[key].value<std::string>("title", "");
                    if (title.size()) {
                        if (static_cast<int>(title.size()) < title_len__ - 1) {
                            std::copy(title.begin(), title.end(), title__);
                        } else {
                            std::copy(title.begin(), title.begin() + title_len__ - 1, title__);
                        }
                    }
                }

                if (dict[key].count("description")) {
                    auto description = dict[key].value<std::string>("description", "");
                    if (description.size()) {
                        if (static_cast<int>(description.size()) < description_len__ - 1) {
                            std::copy(description.begin(), description.end(), description__);
                        } else {
                            std::copy(description.begin(), description.begin() + description_len__ - 1, description__);
                        }
                    }
                }
            },
            error_code__);
}

/*
@api begin
sirius_option_get:
  doc: Return the default value of the option as defined in the JSON schema.
  arguments:
    section:
      type: string
      attr: in, required
      doc: Name of the section of interest.
    name:
      type: string
      attr: in, required
      doc: Name of the element
    type:
      type: int
      attr: in, required
      doc: Type of the option (real, integer, boolean)
    data_ptr:
      type: void*
      attr: in, required, value
      doc: Output buffer for the default value or list of values.
    max_length:
      type: int
      attr: in, optional
      doc: Maximum length of the buffer containing the default values.
    enum_idx:
      type: int
      attr: in, optional
      doc: Index of the element in case of the enum type.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void
sirius_option_get(char const* section__, char const* name__, int const* type__, void* data_ptr__,
                  int const* max_length__, int const* enum_idx__, int* error_code__)
{
    /* data_ptr is desctibed as `in, required, value`; this is small hack to allow Fortran to pass C_LOC(var) directly;
     * this is justified because pointer itself is really unchanged and thus can be 'in' */
    call_sirius(
            [&]() {
                auto t = static_cast<option_type_t>(*type__);
                std::string section(section__);
                std::string name(name__);
                switch (t) {
                    case option_type_t::INTEGER_TYPE:
                    case option_type_t::INTEGER_ARRAY_TYPE: {
                        sirius_option_get_value<int>(section, name, static_cast<int*>(data_ptr__), max_length__);
                        break;
                    }
                    case option_type_t::NUMBER_TYPE:
                    case option_type_t::NUMBER_ARRAY_TYPE: {
                        sirius_option_get_value<double>(section, name, static_cast<double*>(data_ptr__), max_length__);
                        break;
                    }
                    case option_type_t::LOGICAL_TYPE:
                    case option_type_t::LOGICAL_ARRAY_TYPE: {
                        sirius_option_get_value<bool>(section, name, static_cast<bool*>(data_ptr__), max_length__);
                        break;
                    }
                    case option_type_t::STRING_TYPE: {
                        sirius_option_get_value(section, name, static_cast<char*>(data_ptr__), max_length__,
                                                enum_idx__);
                        break;
                    }
                    default: {
                        RTE_THROW("wrong option type");
                        break;
                    }
                }
            },
            error_code__);
}

/*
@api begin
sirius_option_set:
  doc: Set the value of the option name in a (internal) json dictionary
  arguments:
    handler:
      type: ctx_handler
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
    type:
      type: int
      attr: in, required
      doc: Type of the option (real, integer, boolean)
    data_ptr:
      type: void*
      attr: in, required, value
      doc: Buffer for the value or list of values.
    max_length:
      type: int
      attr: in, optional
      doc: Maximum length of the buffer containing the default values.
    append:
      type: bool
      attr: in, optional
      doc: If true then value is appended to the list of values.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void
sirius_option_set(void* const* handler__, char const* section__, char const* name__, int const* type__,
                  void const* data_ptr__, int const* max_length__, bool const* append__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& sim_ctx = get_sim_ctx(handler__);
                auto t        = static_cast<option_type_t>(*type__);
                std::string section(section__);
                std::string name(name__);
                switch (t) {
                    case option_type_t::INTEGER_TYPE:
                    case option_type_t::INTEGER_ARRAY_TYPE: {
                        sirius_option_set_value<int>(sim_ctx, section, name, static_cast<int const*>(data_ptr__),
                                                     max_length__);
                        break;
                    }
                    case option_type_t::NUMBER_TYPE:
                    case option_type_t::NUMBER_ARRAY_TYPE: {
                        sirius_option_set_value<double>(sim_ctx, section, name, static_cast<double const*>(data_ptr__),
                                                        max_length__);
                        break;
                    }
                    case option_type_t::LOGICAL_TYPE:
                    case option_type_t::LOGICAL_ARRAY_TYPE: {
                        sirius_option_set_value<bool>(sim_ctx, section, name, static_cast<bool const*>(data_ptr__),
                                                      max_length__);
                        break;
                    }
                    case option_type_t::STRING_TYPE: {
                        bool append = (append__ != nullptr) ? *append__ : false;
                        sirius_option_set_value(sim_ctx, section, name, static_cast<char const*>(data_ptr__),
                                                max_length__, append);
                        break;
                    }
                    default: {
                        RTE_THROW("wrong option type");
                        break;
                    }
                }
            },
            error_code__);
}

/*
@api begin
sirius_dump_runtime_setup:
  doc: Dump the runtime setup in a file.
  arguments:
    handler:
      type: ctx_handler
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
void
sirius_dump_runtime_setup(void* const* handler__, char* filename__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& sim_ctx = get_sim_ctx(handler__);
                std::ofstream fi(filename__, std::ofstream::out | std::ofstream::trunc);
                auto conf_dict = sim_ctx.serialize();
                fi << conf_dict.dump(4);
            },
            error_code__);
}

/*
@api begin
sirius_get_fv_eigen_vectors:
  doc: Get the first-variational eigen vectors
  arguments:
    handler:
      type: ks_handler
      attr: in, required
      doc: K-point set handler
    ik:
      type: int
      attr: in, required
      doc: Global index of the k-point
    fv_evec:
      type: complex
      attr: out, required, dimension(:,:)
      doc: Output first-variational eigenvector array
    ld:
      type: int
      attr: in, required
      doc: Leading dimension of fv_evec
    num_fv_states:
      type: int
      attr: in, required
      doc: Number of first-variational states
    error_code:
      type: int
      attr: out, optional
      doc: Error code
@api end
*/
void
sirius_get_fv_eigen_vectors(void* const* handler__, int const* ik__, std::complex<double>* fv_evec__, int const* ld__,
                            int const* num_fv_states__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& ks = get_ks(handler__);
                mdarray<std::complex<double>, 2> fv_evec({*ld__, *num_fv_states__}, fv_evec__);
                int ik = *ik__ - 1;
                ks.get<double>(ik)->get_fv_eigen_vectors(fv_evec);
            },
            error_code__);
}

/*
@api begin
sirius_get_fv_eigen_values:
  doc: Get the first-variational eigen values
  arguments:
    handler:
      type: ks_handler
      attr: in, required
      doc: K-point set handler
    ik:
      type: int
      attr: in, required
      doc: Global index of the k-point
    fv_eval:
      type: double
      attr: out, required, dimension(:)
      doc: Output first-variational eigenvector array
    num_fv_states:
      type: int
      attr: in, required
      doc: Number of first-variational states
    error_code:
      type: int
      attr: out, optional
      doc: Error code
@api end
*/
void
sirius_get_fv_eigen_values(void* const* handler__, int const* ik__, double* fv_eval__, int const* num_fv_states__,
                           int* error_code__)
{
    call_sirius(
            [&]() {
                auto& ks = get_ks(handler__);
                if (*num_fv_states__ != ks.ctx().num_fv_states()) {
                    RTE_THROW("wrong number of first-variational states");
                }
                int ik = *ik__ - 1;
                for (int i = 0; i < *num_fv_states__; i++) {
                    fv_eval__[i] = ks.get<double>(ik)->fv_eigen_value(i);
                }
            },
            error_code__);
}

/*
@api begin
sirius_get_sv_eigen_vectors:
  doc: Get the second-variational eigen vectors
  arguments:
    handler:
      type: ks_handler
      attr: in, required
      doc: K-point set handler
    ik:
      type: int
      attr: in, required
      doc: Global index of the k-point
    sv_evec:
      type: complex
      attr: out, required, dimension(:,:)
      doc: Output second-variational eigenvector array
    num_bands:
      type: int
      attr: in, required
      doc: Number of second-variational bands.
    error_code:
      type: int
      attr: out, optional
      doc: Error code
@api end
*/
void
sirius_get_sv_eigen_vectors(void* const* handler__, int const* ik__, std::complex<double>* sv_evec__,
                            int const* num_bands__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& ks = get_ks(handler__);
                mdarray<std::complex<double>, 2> sv_evec({*num_bands__, *num_bands__}, sv_evec__);
                int ik = *ik__ - 1;
                ks.get<double>(ik)->get_sv_eigen_vectors(sv_evec);
            },
            error_code__);
}

/*
@api begin
sirius_set_rg_values:
  doc: Set the values of the function on the regular grid.
  arguments:
    handler:
      type: gs_handler
      attr: in, required
      doc: DFT ground state handler.
    label:
      type: string
      attr: in, required
      doc: Label of the function.
    grid_dims:
      type: int
      attr: in, required, dimension(3)
      doc: Dimensions of the FFT grid.
    local_box_origin:
      type: int
      attr: in, required, dimension(:,:)
      doc: Coordinates of the local box origin for each MPI rank
    local_box_size:
      type: int
      attr: in, required, dimension(:,:)
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
    error_code:
      type: int
      attr: out, optional
      doc: Error code
@api end
*/
void
sirius_set_rg_values(void* const* handler__, char const* label__, int const* grid_dims__, int const* local_box_origin__,
                     int const* local_box_size__, int const* fcomm__, double const* values__,
                     bool const* transform_to_pw__, int* error_code__)
{
    PROFILE("sirius_api::sirius_set_rg_values");

    call_sirius(
            [&]() {
                auto& gs = get_gs(handler__);

                std::string label(label__);

                for (int x : {0, 1, 2}) {
                    if (grid_dims__[x] != gs.ctx().fft_grid()[x]) {
                        std::stringstream s;
                        s << "wrong FFT grid size\n"
                             "  SIRIUS internal: "
                          << gs.ctx().fft_grid()[0] << " " << gs.ctx().fft_grid()[1] << " " << gs.ctx().fft_grid()[2]
                          << "\n"
                             "  host code:       "
                          << grid_dims__[0] << " " << grid_dims__[1] << " " << grid_dims__[2];
                        RTE_THROW(s);
                    }
                }

                std::map<std::string, sirius::Smooth_periodic_function<double>*> func = {
                        {"rho", &gs.density().rho().rg()},
                        {"magz", &gs.density().mag(0).rg()},
                        {"magx", &gs.density().mag(1).rg()},
                        {"magy", &gs.density().mag(2).rg()},
                        {"veff", &gs.potential().effective_potential().rg()},
                        {"bz", &gs.potential().effective_magnetic_field(0).rg()},
                        {"bx", &gs.potential().effective_magnetic_field(1).rg()},
                        {"by", &gs.potential().effective_magnetic_field(2).rg()},
                        {"vxc", &gs.potential().xc_potential().rg()},
                };
                if (!func.count(label)) {
                    RTE_THROW("wrong label: " + label);
                }

                auto& f = func.at(label);

                auto& comm = mpi::Communicator::map_fcomm(*fcomm__);

                mdarray<int, 2> local_box_size({3, comm.size()}, const_cast<int*>(local_box_size__));
                mdarray<int, 2> local_box_origin({3, comm.size()}, const_cast<int*>(local_box_origin__));

                for (int rank = 0; rank < comm.size(); rank++) {
                    /* dimensions of this rank's local box */
                    int nx = local_box_size(0, rank);
                    int ny = local_box_size(1, rank);
                    int nz = local_box_size(2, rank);

                    mdarray<double, 3> buf({nx, ny, nz});
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
                        /* each rank on SIRIUS side, for which this condition is fulfilled copies data from the local
                         * box */
                        if (z >= gs.ctx().spfft<double>().local_z_offset() &&
                            z < gs.ctx().spfft<double>().local_z_offset() + gs.ctx().spfft<double>().local_z_length()) {
                            /* make z local for SIRIUS FFT partitioning */
                            z -= gs.ctx().spfft<double>().local_z_offset();
                            for (int iy = 0; iy < ny; iy++) {
                                /* global y coordinate inside FFT box */
                                int y = local_box_origin(1, rank) + iy - 1; /* Fortran counts from 1 */
                                for (int ix = 0; ix < nx; ix++) {
                                    /* global x coordinate inside FFT box */
                                    int x = local_box_origin(0, rank) + ix - 1; /* Fortran counts from 1 */
                                    f->value(gs.ctx().fft_grid().index_by_coord(x, y, z)) = buf(ix, iy, iz);
                                }
                            }
                        }
                    }
                } /* loop over ranks */
                if (transform_to_pw__ && *transform_to_pw__) {
                    f->fft_transform(-1);
                }
            },
            error_code__);
}

/*
@api begin
sirius_get_rg_values:
  doc: Get the values of the function on the regular grid.
  arguments:
    handler:
      type: gs_handler
      attr: in, required
      doc: DFT ground state handler.
    label:
      type: string
      attr: in, required
      doc: Label of the function.
    grid_dims:
      type: int
      attr: in, required, dimensions(3)
      doc: Dimensions of the FFT grid.
    local_box_origin:
      type: int
      attr: in, required, dimensions(:,:)
      doc: Coordinates of the local box origin for each MPI rank
    local_box_size:
      type: int
      attr: in, required, dimensions(:,:)
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
    error_code:
      type: int
      attr: out, optional
      doc: Error code
@api end
*/
void
sirius_get_rg_values(void* const* handler__, char const* label__, int const* grid_dims__, int const* local_box_origin__,
                     int const* local_box_size__, int const* fcomm__, double* values__, bool const* transform_to_rg__,
                     int* error_code__)
{
    PROFILE("sirius_api::sirius_get_rg_values");

    call_sirius(
            [&]() {
                auto& gs = get_gs(handler__);

                std::string label(label__);

                for (int x : {0, 1, 2}) {
                    if (grid_dims__[x] != gs.ctx().fft_grid()[x]) {
                        RTE_THROW("wrong FFT grid size");
                    }
                }

                std::map<std::string, sirius::Smooth_periodic_function<double>*> func = {
                        {"rho", &gs.density().rho().rg()},
                        {"magz", &gs.density().mag(0).rg()},
                        {"magx", &gs.density().mag(1).rg()},
                        {"magy", &gs.density().mag(2).rg()},
                        {"veff", &gs.potential().effective_potential().rg()},
                        {"bz", &gs.potential().effective_magnetic_field(0).rg()},
                        {"bx", &gs.potential().effective_magnetic_field(1).rg()},
                        {"by", &gs.potential().effective_magnetic_field(2).rg()},
                        {"vxc", &gs.potential().xc_potential().rg()},
                };

                if (!func.count(label)) {
                    RTE_THROW("wrong label: " + label);
                }

                auto& f = func.at(label);

                auto& comm = mpi::Communicator::map_fcomm(*fcomm__);

                if (transform_to_rg__ && *transform_to_rg__) {
                    f->fft_transform(1);
                }

                auto& fft_comm = gs.ctx().comm_fft();
                auto spl_z     = fft::split_z_dimension(gs.ctx().fft_grid()[2], fft_comm);

                mdarray<int, 2> local_box_size({3, comm.size()}, const_cast<int*>(local_box_size__));
                mdarray<int, 2> local_box_origin({3, comm.size()}, const_cast<int*>(local_box_origin__));

                for (int rank = 0; rank < fft_comm.size(); rank++) {
                    /* slab of FFT grid for a given rank */
                    mdarray<double, 3> buf({f->spfft().dim_x(), f->spfft().dim_y(), spl_z.local_size(block_id(rank))});
                    if (rank == fft_comm.rank()) {
                        std::copy(&f->value(0), &f->value(0) + f->spfft().local_slice_size(), &buf[0]);
                    }
                    fft_comm.bcast(&buf[0], static_cast<int>(buf.size()), rank);

                    /* ranks on the F90 side */
                    int r = comm.rank();

                    /* dimensions of this rank's local box */
                    int nx = local_box_size(0, r);
                    int ny = local_box_size(1, r);
                    int nz = local_box_size(2, r);
                    mdarray<double, 3> values({nx, ny, nz}, values__);

                    for (int iz = 0; iz < nz; iz++) {
                        /* global z coordinate inside FFT box */
                        int z = local_box_origin(2, r) + iz - 1; /* Fortran counts from 1 */
                        if (z >= spl_z.global_offset(block_id(rank)) &&
                            z < spl_z.global_offset(block_id(rank)) + spl_z.local_size(block_id(rank))) {
                            /* make z local for SIRIUS FFT partitioning */
                            z -= spl_z.global_offset(block_id(rank));
                            for (int iy = 0; iy < ny; iy++) {
                                /* global y coordinate inside FFT box */
                                int y = local_box_origin(1, r) + iy - 1; /* Fortran counts from 1 */
                                for (int ix = 0; ix < nx; ix++) {
                                    /* global x coordinate inside FFT box */
                                    int x              = local_box_origin(0, r) + ix - 1; /* Fortran counts from 1 */
                                    values(ix, iy, iz) = buf(x, y, z);
                                }
                            }
                        }
                    }
                } /* loop over ranks */
            },
            error_code__);
}

/*
@api begin
sirius_get_total_magnetization:
  doc: Get the total magnetization of the system.
  arguments:
    handler:
      type: gs_handler
      attr: in, required
      doc: DFT ground state handler.
    mag:
      type: double
      attr: out, required
      doc: 3D magnetization vector (x,y,z components).
    error_code:
      type: int
      attr: out, optional
      doc: Error code
@api end
*/
void
sirius_get_total_magnetization(void* const* handler__, double* mag__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& gs = get_gs(handler__);

                mdarray<double, 1> total_mag({3}, mag__);
                total_mag.zero();
                for (int j = 0; j < gs.ctx().num_mag_dims(); j++) {
                    auto result  = gs.density().mag(j).integrate();
                    total_mag[j] = std::get<0>(result);
                }
                if (gs.ctx().num_mag_dims() == 3) {
                    /* swap z and x and change order from z,x,y to x,z,y */
                    std::swap(total_mag[0], total_mag[1]);
                    /* swap z and y and change order x,z,y to x,y,z */
                    std::swap(total_mag[1], total_mag[2]);
                }
            },
            error_code__);
}

/*
@api begin
sirius_get_num_kpoints:
  doc: Get the total number of kpoints
  arguments:
    handler:
      type: ks_handler
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

void
sirius_get_num_kpoints(void* const* handler__, int* num_kpoints__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& ks       = get_ks(handler__);
                *num_kpoints__ = ks.num_kpoints();
            },
            error_code__);
}

/*
@api begin
sirius_get_kpoint_properties:
  doc: Get the kpoint properties
  arguments:
    handler:
      type: ks_handler
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
void
sirius_get_kpoint_properties(void* const* handler__, int const* ik__, double* weight__, double* coordinates__,
                             int* error_code__)
{
    call_sirius(
            [&]() {
                auto& ks  = get_ks(handler__);
                int ik    = *ik__ - 1;
                *weight__ = ks.get<double>(ik)->weight();

                if (coordinates__) {
                    coordinates__[0] = ks.get<double>(ik)->vk()[0];
                    coordinates__[1] = ks.get<double>(ik)->vk()[1];
                    coordinates__[2] = ks.get<double>(ik)->vk()[2];
                }
            },
            error_code__);
}

/*
@api begin
sirius_set_callback_function:
  doc: Set callback function to compute various radial integrals.
  arguments:
    handler:
      type: ctx_handler
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
void
sirius_set_callback_function(void* const* handler__, char const* label__, void (*fptr__)(), int* error_code__)
{
    call_sirius(
            [&]() {
                auto label = std::string(label__);
                std::transform(label.begin(), label.end(), label.begin(), ::tolower);
                auto& sim_ctx = get_sim_ctx(handler__);
                if (label == "beta_ri") {
                    sim_ctx.cb().beta_ri_ = reinterpret_cast<void (*)(int, double, double*, int)>(fptr__);
                } else if (label == "beta_ri_djl") {
                    sim_ctx.cb().beta_ri_djl_ = reinterpret_cast<void (*)(int, double, double*, int)>(fptr__);
                } else if (label == "aug_ri") {
                    sim_ctx.cb().aug_ri_ = reinterpret_cast<void (*)(int, double, double*, int, int)>(fptr__);
                } else if (label == "aug_ri_djl") {
                    sim_ctx.cb().aug_ri_djl_ = reinterpret_cast<void (*)(int, double, double*, int, int)>(fptr__);
                } else if (label == "vloc_ri") {
                    sim_ctx.cb().vloc_ri_ = reinterpret_cast<void (*)(int, int, double*, double*)>(fptr__);
                } else if (label == "vloc_ri_djl") {
                    sim_ctx.cb().vloc_ri_djl_ = reinterpret_cast<void (*)(int, int, double*, double*)>(fptr__);
                } else if (label == "rhoc_ri") {
                    sim_ctx.cb().rhoc_ri_ = reinterpret_cast<void (*)(int, int, double*, double*)>(fptr__);
                } else if (label == "rhoc_ri_djl") {
                    sim_ctx.cb().rhoc_ri_djl_ = reinterpret_cast<void (*)(int, int, double*, double*)>(fptr__);
                } else if (label == "band_occ") {
                    sim_ctx.cb().band_occ_ = reinterpret_cast<void (*)(void)>(fptr__);
                } else if (label == "veff") {
                    sim_ctx.cb().veff_ = reinterpret_cast<void (*)(void)>(fptr__);
                } else if (label == "ps_rho_ri") {
                    sim_ctx.cb().ps_rho_ri_ = reinterpret_cast<void (*)(int, int, double*, double*)>(fptr__);
                } else if (label == "ps_atomic_wf_ri") {
                    sim_ctx.cb().ps_atomic_wf_ri_ = reinterpret_cast<void (*)(int, double, double*, int)>(fptr__);
                } else if (label == "ps_atomic_wf_ri_djl") {
                    sim_ctx.cb().ps_atomic_wf_ri_djl_ = reinterpret_cast<void (*)(int, double, double*, int)>(fptr__);
                } else {
                    RTE_THROW("Wrong label of the callback function: " + label);
                }
            },
            error_code__);
}

/*
@api begin
sirius_nlcg:
  doc: Robust wave function optimizer.
  arguments:
    handler:
      type: gs_handler
      attr: in, required
      doc: Ground state handler.
    ks_handler:
      type: ks_handler
      attr: in, required
      doc: K-point set handler.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/

void
sirius_nlcg(void* const* handler__, void* const* ks_handler__, int* error_code__)
{
    call_sirius(
            [&]() {
#if defined(SIRIUS_NLCGLIB)
                // call nlcg solver
                auto& gs        = get_gs(handler__);
                auto& potential = gs.potential();
                auto& density   = gs.density();

                auto& kset = get_ks(ks_handler__);
                auto& ctx  = kset.ctx();

                auto nlcg_params  = ctx.cfg().nlcg();
                double temp       = nlcg_params.T();
                double tol        = nlcg_params.tol();
                double kappa      = nlcg_params.kappa();
                double tau        = nlcg_params.tau();
                int maxiter       = nlcg_params.maxiter();
                int restart       = nlcg_params.restart();
                std::string smear = ctx.cfg().parameters().smearing();
                std::string pu    = ctx.cfg().control().processing_unit();

                nlcglib::smearing_type smearing;
                if (smear.compare("FD") == 0) {
                    smearing = nlcglib::smearing_type::FERMI_DIRAC;
                } else if (smear.compare("GS") == 0) {
                    smearing = nlcglib::smearing_type::GAUSSIAN_SPLINE;
                } else {
                    RTE_THROW("invalid smearing type given: " + smear);
                }

                sirius::Energy energy(kset, density, potential);
                if (is_device_memory(ctx.processing_unit_memory_t())) {
                    if (pu.empty() || pu.compare("gpu") == 0) {
                        nlcglib::nlcg_mvp2_device(energy, smearing, temp, tol, kappa, tau, maxiter, restart);
                    } else if (pu.compare("cpu") == 0) {
                        nlcglib::nlcg_mvp2_device_cpu(energy, smearing, temp, tol, kappa, tau, maxiter, restart);
                    } else {
                        RTE_THROW("invalid processing unit for nlcg given: " + pu);
                    }
                } else {
                    if (pu.empty() || pu.compare("gpu") == 0) {
                        nlcglib::nlcg_mvp2_cpu(energy, smearing, temp, tol, kappa, tau, maxiter, restart);
                    } else if (pu.compare("cpu") == 0) {
                        nlcglib::nlcg_mvp2_cpu_device(energy, smearing, temp, tol, kappa, tau, maxiter, restart);
                    } else {
                        RTE_THROW("invalid processing unit for nlcg given: " + pu);
                    }
                }
#else
                RTE_THROW("SIRIUS was not compiled with NLCG option.");
#endif
            },
            error_code__);
}

/*
@api begin
sirius_nlcg_params:
  doc: Robust wave function optimizer
  arguments:
    handler:
      type: gs_handler
      attr: in, required
      doc: Ground state handler.
    ks_handler:
      type: ks_handler
      attr: in, required
      doc: K-point set handler.
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
    converged:
      type: bool
      attr: out, required
      doc: Boolean variable indicating if the calculation has converged
      doc:
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/

void
sirius_nlcg_params(void* const* handler__, void* const* ks_handler__, double const* temp__, char const* smearing__,
                   double const* kappa__, double const* tau__, double const* tol__, int const* maxiter__,
                   int const* restart__, char const* processing_unit__, bool* converged__, int* error_code__)
{
    call_sirius(
            [&]() {
#if defined(SIRIUS_NLCGLIB)
                PROFILE("sirius::nlcglib");
                // call nlcg solver
                auto& gs        = get_gs(handler__);
                auto& potential = gs.potential();
                auto& density   = gs.density();

                auto& kset = get_ks(ks_handler__);
                auto& ctx  = kset.ctx();

                double temp  = *temp__;
                double kappa = *kappa__;
                double tau   = *tau__;
                double tol   = *tol__;
                int maxiter  = *maxiter__;
                int restart  = *restart__;

                std::string smear(smearing__);
                std::string pu(processing_unit__);

                device_t processing_unit{ctx.processing_unit()};

                if (pu.compare("none") != 0) {
                    processing_unit = get_device_t(pu);
                }

                nlcglib::smearing_type smearing_t;
                if (smear.compare("FD") == 0) {
                    smearing_t = nlcglib::smearing_type::FERMI_DIRAC;
                } else if (smear.compare("GS") == 0) {
                    smearing_t = nlcglib::smearing_type::GAUSSIAN_SPLINE;
                } else if (smear.compare("GAUSS") == 0) {
                    smearing_t = nlcglib::smearing_type::GAUSS;
                } else if (smear.compare("MP") == 0) {
                    smearing_t = nlcglib::smearing_type::METHFESSEL_PAXTON;
                } else if (smear.compare("COLD") == 0) {
                    smearing_t = nlcglib::smearing_type::COLD;
                } else {
                    RTE_THROW("invalid smearing type given: " + smear);
                }

                if (pu.compare("none") == 0) {
                    // use same processing unit as SIRIUS
                    pu = ctx.cfg().control().processing_unit();
                }

                sirius::Energy energy(kset, density, potential);

                sirius::Hamiltonian0<double> H0(potential, false);

                sirius::UltrasoftPrecond us_precond(kset, ctx, H0.Q());
                sirius::Overlap_operators<sirius::S_k<std::complex<double>>> S(kset, ctx, H0.Q());

                // ultrasoft pp
                nlcglib::nlcg_info info;
                switch (processing_unit) {
                    case device_t::CPU: {
                        info = nlcglib::nlcg_us_cpu(energy, us_precond, S, smearing_t, temp, tol, kappa, tau, maxiter,
                                                    restart);
                        break;
                    }
                    case device_t::GPU: {
                        info = nlcglib::nlcg_us_device(energy, us_precond, S, smearing_t, temp, tol, kappa, tau,
                                                       maxiter, restart);
                        break;
                    }
                }

                // return exit state of nlcglib to QE
                *converged__ = info.converged;
#else  /* SIRIUS_NLCGLIB */
                RTE_THROW("SIRIUS was not compiled with NLCG option.");
#endif /* SIRIUS_NLCGLIB */
            },
            error_code__);
}

/*
@api begin
sirius_add_hubbard_atom_pair:
  doc: Add a non-local Hubbard interaction V for a pair of atoms.
  arguments:
    handler:
      type: ctx_handler
      attr: in, required
      doc: Simulation context handler.
    atom_pair:
      type: int
      attr: in, required, dimension(2)
      doc: atom pair for the V term
    translation:
      type: int
      attr: in, required, dimension(3)
      doc: translation vector between the two unit cells containing the atoms
    n:
      type: int
      attr: in, required, dimension(2)
      doc: principal quantum number of the atomic levels involved in the V correction
    l:
      type: int
      attr: in, required, dimension(2)
      doc: angular momentum of the atomic levels
    coupling:
      type: double
      attr: in, required
      doc: value of the V constant
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void
sirius_add_hubbard_atom_pair(void* const* handler__, int* const atom_pair__, int* const translation__, int* const n__,
                             int* const l__, const double* const coupling__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& sim_ctx  = get_sim_ctx(handler__);
                auto conf_dict = sim_ctx.cfg().hubbard();

                json elem;
                std::vector<int> atom_pair(atom_pair__, atom_pair__ + 2);
                /* Fortran indices start from 1 */
                atom_pair[0] -= 1;
                atom_pair[1] -= 1;
                std::vector<int> n(n__, n__ + 2);
                std::vector<int> l(l__, l__ + 2);
                std::vector<int> translation(translation__, translation__ + 3);

                elem["atom_pair"] = atom_pair;
                elem["T"]         = translation;
                elem["n"]         = n;
                elem["l"]         = l;
                elem["V"]         = *coupling__;

                bool test{false};

                for (int idx = 0; idx < conf_dict.nonlocal().size(); idx++) {
                    auto v     = conf_dict.nonlocal(idx);
                    auto at_pr = v.atom_pair();
                    /* search if the pair is already present */
                    if ((at_pr[0] == atom_pair[0]) && (at_pr[1] == atom_pair[1])) {
                        auto tr = v.T();
                        if ((tr[0] = translation[0]) && (tr[1] = translation[1]) && (tr[2] = translation[2])) {
                            auto lvl = v.n();
                            if ((lvl[0] == n[0]) && (lvl[0] == n[1])) {
                                auto li = v.l();
                                if ((li[0] == l[0]) && (li[1] == l[1])) {
                                    test = true;
                                    break;
                                }
                            }
                        }
                    }
                }

                if (!test) {
                    conf_dict.nonlocal().append(elem);
                } else {
                    RTE_THROW("Atom pair for hubbard correction is already present");
                }
            },
            error_code__);
}

/*
@api begin
sirius_set_hubbard_contrained_parameters:
   doc: Set the parameters controlling hubbard constrained calculation
   arguments:
     handler:
       type: ctx_handler
       attr: in, required
       doc: Simulation context handler.
     hubbard_conv_thr:
       type: double
       attr: in, optional
       doc: convergence threhold when the hubbard occupation is constrained
     hubbard_mixing_beta:
       type: double
       attr: in, optional
       doc: mixing parameter for the hubbard constraints
     hubbard_strength:
       type: double
       attr: in, optional
       doc: energy penalty when the effective occupation numbers deviate from the reference numbers
     hubbard_maxstep:
       type: int
       attr: in, optional
       doc: maximum number of constrained iterations
     hubbard_constraint_type:
       type: string
       attr: in, optional
       doc: type of constrain, energy or occupation
     error_code:
       type: int
       attr: out, optional
       doc: Error code.
  @api end
*/
void
sirius_set_hubbard_contrained_parameters(void* const* handler__, double const* hubbard_conv_thr__,
                                         double const* hubbard_mixing_beta__, double const* hubbard_strength__,
                                         int const* hubbard_maxstep__, char const* hubbard_constraint_type__,
                                         int* const error_code__)
{
    call_sirius(
            [&]() {
                auto& sim_ctx = get_sim_ctx(handler__);
                if (hubbard_conv_thr__ != nullptr) {
                    sim_ctx.cfg().hubbard().constraint_error(*hubbard_conv_thr__);
                }

                if (hubbard_mixing_beta__ != nullptr) {
                    sim_ctx.cfg().hubbard().constraint_beta_mixing(*hubbard_mixing_beta__);
                }

                if (hubbard_strength__ != nullptr) {
                    sim_ctx.cfg().hubbard().constraint_strength(*hubbard_strength__);
                }

                if (hubbard_maxstep__ != nullptr) {
                    sim_ctx.cfg().hubbard().constraint_max_iteration(*hubbard_maxstep__);
                }

                if (hubbard_constraint_type__ != nullptr) {
                    sim_ctx.cfg().hubbard().constraint_method(hubbard_constraint_type__);
                }
            },
            error_code__);
}
/*
@api begin
sirius_add_hubbard_atom_constraint:
  doc: Information about the constrained atomic level
  arguments:
    handler:
      type: ctx_handler
      attr: in, required
      doc: Simulation context handler.
    atom_id:
      type: int
      attr: in, required
      doc: atom iindex
    n:
      type: int
      attr: in, required
      doc: principal quantum number of the atomic level for the constrained hubbard correction
    l:
      type: int
      attr: in, required
      doc: angular momentum of the atomic level
    lmax_at:
      type: int
      attr: in, required
      doc: maximum angular momentum
    occ:
      type: double
      attr: in, required, dimension(2 * lmax_at + 1, 2 * lmax_at + 1, 2)
      doc: value of the occupation matrix for this level
    orbital_order:
      type: int
      attr: in, optional, dimension(2 * l + 1)
      doc: order or the Ylm by default it is SIRIUS order for Ylm
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
  @api end
*/
void
sirius_add_hubbard_atom_constraint(void* const* handler__, int* const atom_id__, int* const n__, int* const l__,
                                   int* const lmax_at__, const double* const occ__, int* const orbital_order__,
                                   int* const error_code__)
{
    call_sirius(
            [&]() {
                auto& sim_ctx = get_sim_ctx(handler__);
                json elem;
                elem["atom_index"] = *atom_id__ - 1;
                elem["n"]          = *n__;
                elem["l"]          = *l__;
                const mdarray<double, 3> occ_({2 * *lmax_at__ + 1, 2 * *lmax_at__ + 1, 2}, (double* const)occ__);
                std::vector<std::vector<std::vector<double>>> elem_occ_(2);
                // const int lda_ = 2 * *lmax_at__ + 1;
                for (int s = 0; s < 2; s++) {
                    elem_occ_[s].clear();
                    elem_occ_[s].resize(2 * *l__ + 1);
                    for (int m = 0; m < 2 * *l__ + 1; m++) {
                        elem_occ_[s][m].clear();
                        elem_occ_[s][m].resize(2 * *l__ + 1);
                        for (int mp = 0; mp < 2 * *l__ + 1; mp++) {
                            elem_occ_[s][m][mp] = occ_(mp, m, s);
                        }
                    }
                }
                elem["occupancy"] = elem_occ_;

                if (orbital_order__ != nullptr) {
                    std::vector<int> lm_order_(2 * *lmax_at__ + 1, 0);
                    std::memcpy(lm_order_.data(), orbital_order__, sizeof(int) * (2 * *lmax_at__ + 1));
                    elem["lm_order"] = lm_order_;
                }
                sim_ctx.cfg().hubbard().local_constraint().append(elem);
            },
            error_code__);
}
/*
@api begin
sirius_create_H0:
  doc: Generate H0.
  arguments:
    handler:
      type: gs_handler
      attr: in, required
      doc: Ground state handler.
    error_code:
      type: int
      attr: out, optional
      doc: Error code
@api end
*/
void
sirius_create_H0(void* const* handler__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& gs = get_gs(handler__);
                gs.create_H0();
            },
            error_code__);
}

/*
@api begin
sirius_linear_solver:
  doc: Interface to linear solver.
  arguments:
    handler:
      type: gs_handler
      attr: in, required
      doc: DFT ground state handler.
    vkq:
      type: double
      attr: in, required, dimension(3)
      doc: K+q-point in lattice coordinates
    num_gvec_kq_loc:
      type: int
      attr: in, required
      doc: Local number of G-vectors for k+q-point
    gvec_kq_loc:
      type: int
      attr: in, required, dimension(3, num_gvec_kq_loc)
      doc: Local list of G-vectors for k+q-point.
    dpsi:
      type: complex
      attr: inout, required, dimension(ld, num_spin_comp)
      doc: Left-hand side of the linear equation.
    psi:
      type: complex
      attr: in, required, dimension(ld, num_spin_comp)
      doc: Unperturbed eigenvectors.
    eigvals:
      type: double
      attr: in, required, dimension(*)
      doc: Unperturbed eigenvalues.
    dvpsi:
      type: complex
      attr: inout, required, dimension(ld, num_spin_comp)
      doc: Right-hand side of the linear equation (dV * psi)
    ld:
      type: int
      attr: in, required
      doc: Leading dimension of dpsi, psi, dvpsi.
    num_spin_comp:
      type: int
      attr: in, required
      doc: Number of spin components.
    alpha_pv:
      type: double
      attr: in, required
      doc: Constant for the projector.
    spin:
      type: int
      attr: in, required
      doc: Current spin channel.
    nbnd_occ:
      type: int
      attr: in, required
      doc: Number of occupied bands.
    tol:
      type: double
      attr: in, optional
      doc: Tolerance for the unconverged residuals (residual L2-norm should be below this value).
    niter:
      type: int
      attr: out, optional
      doc: Average number of iterations.
    error_code:
      type: int
      attr: out, optional
      doc: Error code
@api end
*/
void
sirius_linear_solver(void* const* handler__, double const* vkq__, int const* num_gvec_kq_loc__,
                     int const* gvec_kq_loc__, std::complex<double>* dpsi__, std::complex<double>* psi__,
                     double* eigvals__, std::complex<double>* dvpsi__, int const* ld__, int const* num_spin_comp__,
                     double const* alpha_pv__, int const* spin__, int const* nbnd_occ__, double const* tol__,
                     int* niter__, int* error_code__)
{
    using namespace sirius;
    PROFILE("sirius_api::sirius_linear_solver");
    call_sirius(
            [&]() {
                /* works for non-magnetic and collinear cases */
                RTE_ASSERT(*num_spin_comp__ == 1);

                int nbnd_occ = *nbnd_occ__;

                if (nbnd_occ == 0) {
                    return;
                }

                auto& gs   = get_gs(handler__);
                auto& sctx = gs.ctx();

                wf::spin_range sr(0);
                if (sctx.num_mag_dims() == 1) {
                    if (!(*spin__ == 1 || *spin__ == 2)) {
                        RTE_THROW("wrong spin channel");
                    }
                    sr = wf::spin_range(*spin__ - 1);
                }

                std::shared_ptr<fft::Gvec> gvkq_in;
                gvkq_in = std::make_shared<fft::Gvec>(r3::vector<double>(vkq__),
                                                      sctx.unit_cell().reciprocal_lattice_vectors(), *num_gvec_kq_loc__,
                                                      gvec_kq_loc__, sctx.comm_band(), false);

                int num_gvec_kq_loc = *num_gvec_kq_loc__;
                int num_gvec_kq     = num_gvec_kq_loc;
                sctx.comm_band().allreduce(&num_gvec_kq, 1);

                if (num_gvec_kq != gvkq_in->num_gvec()) {
                    RTE_THROW("wrong number of G+k vectors for k");
                }

                auto& H0 = gs.get_H0();

                sirius::K_point<double> kp(const_cast<sirius::Simulation_context&>(sctx), gvkq_in, 1.0);
                kp.initialize();

                auto Hk = H0(kp);

                /* copy eigenvalues (factor 2 for rydberg/hartree) */
                std::vector<double> eigvals_vec(eigvals__, eigvals__ + nbnd_occ);
                for (auto& val : eigvals_vec) {
                    val /= 2;
                }

                // Setup dpsi (unknown), psi (part of projector), and dvpsi (right-hand side)
                mdarray<std::complex<double>, 3> psi({*ld__, *num_spin_comp__, nbnd_occ}, psi__);
                mdarray<std::complex<double>, 3> dpsi({*ld__, *num_spin_comp__, nbnd_occ}, dpsi__);
                mdarray<std::complex<double>, 3> dvpsi({*ld__, *num_spin_comp__, nbnd_occ}, dvpsi__);

                auto dpsi_wf  = sirius::wave_function_factory<double>(sctx, kp, wf::num_bands(nbnd_occ),
                                                                     wf::num_mag_dims(0), false);
                auto psi_wf   = sirius::wave_function_factory<double>(sctx, kp, wf::num_bands(nbnd_occ),
                                                                    wf::num_mag_dims(0), false);
                auto dvpsi_wf = sirius::wave_function_factory<double>(sctx, kp, wf::num_bands(nbnd_occ),
                                                                      wf::num_mag_dims(0), false);
                auto tmp_wf   = sirius::wave_function_factory<double>(sctx, kp, wf::num_bands(nbnd_occ),
                                                                    wf::num_mag_dims(0), false);

                for (int ispn = 0; ispn < *num_spin_comp__; ispn++) {
                    for (int i = 0; i < nbnd_occ; i++) {
                        for (int ig = 0; ig < kp.gkvec().count(); ig++) {
                            psi_wf->pw_coeffs(ig, wf::spin_index(ispn), wf::band_index(i))  = psi(ig, ispn, i);
                            dpsi_wf->pw_coeffs(ig, wf::spin_index(ispn), wf::band_index(i)) = dpsi(ig, ispn, i);
                            // divide by two to account for hartree / rydberg, this is
                            // dv * psi and dv should be 2x smaller in sirius.
                            dvpsi_wf->pw_coeffs(ig, wf::spin_index(ispn), wf::band_index(i)) = dvpsi(ig, ispn, i) / 2.0;
                        }
                    }
                }

                /* check residuals H|psi> - e * S |psi> */
                if (sctx.cfg().control().verification() >= 1) {
                    sirius::K_point<double> kp(const_cast<sirius::Simulation_context&>(sctx), gvkq_in, 1.0);
                    kp.initialize();
                    auto Hk = H0(kp);
                    sirius::check_wave_functions<double, std::complex<double>>(
                            Hk, *psi_wf, sr, wf::band_range(0, nbnd_occ), eigvals_vec.data());
                }

                /* setup auxiliary state vectors for CG */
                auto U = sirius::wave_function_factory<double>(sctx, kp, wf::num_bands(nbnd_occ), wf::num_mag_dims(0),
                                                               false);
                auto C = sirius::wave_function_factory<double>(sctx, kp, wf::num_bands(nbnd_occ), wf::num_mag_dims(0),
                                                               false);

                auto Hphi_wf = sirius::wave_function_factory<double>(sctx, kp, wf::num_bands(nbnd_occ),
                                                                     wf::num_mag_dims(0), false);
                auto Sphi_wf = sirius::wave_function_factory<double>(sctx, kp, wf::num_bands(nbnd_occ),
                                                                     wf::num_mag_dims(0), false);

                auto mem = sctx.processing_unit_memory_t();

                std::vector<wf::device_memory_guard> mg;

                mg.emplace_back(psi_wf->memory_guard(mem, wf::copy_to::device));
                mg.emplace_back(dpsi_wf->memory_guard(mem, wf::copy_to::device | wf::copy_to::host));
                mg.emplace_back(dvpsi_wf->memory_guard(mem, wf::copy_to::device));
                mg.emplace_back(tmp_wf->memory_guard(mem, wf::copy_to::device));

                mg.emplace_back(U->memory_guard(mem, wf::copy_to::device));
                mg.emplace_back(C->memory_guard(mem, wf::copy_to::device));
                mg.emplace_back(Hphi_wf->memory_guard(mem, wf::copy_to::device));
                mg.emplace_back(Sphi_wf->memory_guard(mem, wf::copy_to::device));

                sirius::lr::Linear_response_operator linear_operator(const_cast<sirius::Simulation_context&>(sctx), Hk,
                                                                     eigvals_vec, Hphi_wf.get(), Sphi_wf.get(),
                                                                     psi_wf.get(), tmp_wf.get(),
                                                                     *alpha_pv__ / 2, // rydberg/hartree factor
                                                                     wf::band_range(0, nbnd_occ), sr, mem);
                /* CG state vectors */
                auto X_wrap = sirius::lr::Wave_functions_wrap{dpsi_wf.get(), mem};
                auto B_wrap = sirius::lr::Wave_functions_wrap{dvpsi_wf.get(), mem};
                auto U_wrap = sirius::lr::Wave_functions_wrap{U.get(), mem};
                auto C_wrap = sirius::lr::Wave_functions_wrap{C.get(), mem};

                /* set up the diagonal preconditioner */
                auto h_o_diag = Hk.get_h_o_diag_pw<double, 3>(); // already on the GPU if mem=GPU
                mdarray<double, 1> eigvals_mdarray({eigvals_vec.size()});
                eigvals_mdarray = [&](int i) { return eigvals_vec[i]; };
                /* allocate and copy eigvals_mdarray to GPU if running on GPU */
                if (is_device_memory(mem)) {
                    eigvals_mdarray.allocate(mem).copy_to(mem);
                }

                sirius::lr::Smoothed_diagonal_preconditioner preconditioner{std::move(h_o_diag.first),
                                                                            std::move(h_o_diag.second),
                                                                            std::move(eigvals_mdarray),
                                                                            nbnd_occ,
                                                                            mem,
                                                                            sr};

                // Identity_preconditioner preconditioner{static_cast<size_t>(nbnd_occ)};

                auto tol = get_value(tol__, 1e-13);

                auto result = sirius::cg::multi_cg(linear_operator, preconditioner, X_wrap, B_wrap, U_wrap,
                                                   C_wrap, // state vectors
                                                   100,    // iters
                                                   tol);
                if (niter__) {
                    *niter__ = result.niter_eff;
                }
                mg.clear();

                /* bring wave functions back in order of QE */
                for (int ispn = 0; ispn < *num_spin_comp__; ispn++) {
                    for (int i = 0; i < nbnd_occ; i++) {
                        for (int ig = 0; ig < kp.gkvec().count(); ig++) {
                            dpsi(ig, ispn, i) = dpsi_wf->pw_coeffs(ig, wf::spin_index(ispn), wf::band_index(i));
                        }
                    }
                }
            },
            error_code__);
}

/*
@api begin
sirius_generate_rhoaug_q:
  doc: Generate augmentation charge in case of complex density (linear response)
  arguments:
    handler:
      type: gs_handler
      attr: in, required
      doc: DFT ground state handler.
    iat:
      type: int
      attr: in, required
      doc: Index of atom type.
    num_atoms:
      type: int
      attr: in, required
      doc: Total number of atoms.
    num_gvec_loc:
      type: int
      attr: in, required
      doc: Local number of G-vectors
    num_spin_comp:
      type: int
      attr: in, required
      doc: Number of spin components.
    qpw:
      type: complex
      attr: in, required, dimension(ldq, num_gvec_loc)
      doc: Augmentation operator for a givem atom type.
    ldq:
      type: int
      attr: in, required
      doc: Leading dimension of qpw array.
    phase_factors_q:
      type: complex
      attr: in, required, dimension(num_atoms)
      doc: Phase factors exp(i*q*r_alpha)
    mill:
      type: int
      attr: in, required, dimension(3, num_gvec_loc)
      doc: Miller indices (G-vectors in lattice coordinates)
    dens_mtrx:
      type: complex
      attr: in, required, dimension(ld, num_atoms, num_spin_comp)
      doc: Density matrix
    ldd:
      type: int
      attr: in, required
      doc: Leading dimension of density matrix.
    rho_aug:
      type: complex
      attr: inout, required, dimension(num_gvec_loc, num_spin_comp)
      doc: Resulting augmentation charge.
    error_code:
      type: int
      attr: out, optional
      doc: Error code
@api end
*/
void
sirius_generate_rhoaug_q(void* const* handler__, int const* iat__, int const* num_atoms__, int const* num_gvec_loc__,
                         int const* num_spin_comp__, std::complex<double> const* qpw__, int const* ldq__,
                         std::complex<double> const* phase_factors_q__, int const* mill__,
                         std::complex<double> const* dens_mtrx__, int const* ldd__, std::complex<double>* rho_aug__,
                         int* error_code__)
{
    using namespace sirius;
    PROFILE("sirius_api::sirius_generate_rhoaug_q");
    call_sirius(
            [&]() {
                auto& gs   = get_gs(handler__);
                auto& sctx = gs.ctx();
                /* index of atom type */
                int iat           = *iat__ - 1;
                int num_beta      = sctx.unit_cell().atom_type(iat).mt_basis_size();
                int num_gvec_loc  = *num_gvec_loc__;
                int num_spin_comp = *num_spin_comp__;

                mdarray<std::complex<double>, 2> qpw({*ldq__, num_gvec_loc}, const_cast<std::complex<double>*>(qpw__));
                mdarray<int, 2> mill({3, num_gvec_loc}, const_cast<int*>(mill__));
                mdarray<std::complex<double>, 3> dens_mtrx({*ldd__, *num_atoms__, num_spin_comp},
                                                           const_cast<std::complex<double>*>(dens_mtrx__));
                mdarray<std::complex<double>, 2> rho_aug({num_gvec_loc, num_spin_comp}, rho_aug__);

                /* density matrix for all atoms of a given type */
                mdarray<std::complex<double>, 2> tmp2(
                        {num_beta * (num_beta + 1) / 2, sctx.unit_cell().atom_type(iat).num_atoms()},
                        get_memory_pool(memory_t::host));

                mdarray<std::complex<double>, 2> tmp1({sctx.unit_cell().atom_type(iat).num_atoms(), num_gvec_loc},
                                                      get_memory_pool(memory_t::host));
                for (int is = 0; is < num_spin_comp; is++) {
                    PROFILE_START("sirius_generate_rhoaug_q:tmp2")
                    for (int i = 0; i < sctx.unit_cell().atom_type(iat).num_atoms(); i++) {
                        int ia = sctx.unit_cell().atom_type(iat).atom_id(i);
                        for (int j = 0; j < num_beta * (num_beta + 1) / 2; j++) {
                            tmp2(j, i) = dens_mtrx(j, ia, is);
                        }
                    }
                    PROFILE_STOP("sirius_generate_rhoaug_q:tmp2")

                    PROFILE_START("sirius_generate_rhoaug_q:gemm")
                    la::wrap(la::lib_t::blas)
                            .gemm('T', 'N', sctx.unit_cell().atom_type(iat).num_atoms(), num_gvec_loc,
                                  num_beta * (num_beta + 1) / 2, &la::constant<std::complex<double>>::one(),
                                  tmp2.at(memory_t::host), tmp2.ld(), qpw.at(memory_t::host), qpw.ld(),
                                  &la::constant<std::complex<double>>::zero(), tmp1.at(memory_t::host), tmp1.ld());
                    PROFILE_STOP("sirius_generate_rhoaug_q:gemm")

                    PROFILE_START("sirius_generate_rhoaug_q:sum")
                    #pragma omp parallel for
                    for (int ig = 0; ig < num_gvec_loc; ig++) {
                        std::complex<double> z(0, 0);
                        for (int i = 0; i < sctx.unit_cell().atom_type(iat).num_atoms(); i++) {
                            int ia = sctx.unit_cell().atom_type(iat).atom_id(i);
                            z += tmp1(i, ig) * phase_factors_q__[ia] *
                                 std::conj(sctx.gvec_phase_factor(r3::vector<int>(&mill(0, ig)), ia));
                        }
                        rho_aug(ig, is) += z * 2.0;
                    }
                    PROFILE_STOP("sirius_generate_rhoaug_q:sum")
                }
            },
            error_code__);
}

/*
@api begin
sirius_generate_d_operator_matrix:
  doc: Generate D-operator matrix.
  arguments:
    handler:
      type: gs_handler
      attr: in, required
      doc: Ground state handler.
    error_code:
      type: int
      attr: out, optional
      doc: Error code
@api end
*/
void
sirius_generate_d_operator_matrix(void* const* handler__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& gs = get_gs(handler__);
                gs.potential().generate_D_operator_matrix();
            },
            error_code__);
}

/*
@api begin
sirius_save_state:
  doc: Save DFT ground state (density and potential)
  arguments:
    gs_handler:
      type: gs_handler
      attr: in, required
      doc: Ground-state handler.
    file_name:
      type: string
      attr: in, required
      doc: Name of the file that stores the saved data.
    error_code:
      type: int
      attr: out, optional
      doc: Error code
@api end
*/
void
sirius_save_state(void** handler__, const char* file_name__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& gs = get_gs(handler__);
                std::string file_name(file_name__);
                gs.ctx().create_storage_file(file_name);
                gs.potential().save(file_name);
                gs.density().save(file_name);
            },
            error_code__);
}

/*
@api begin
sirius_load_state:
  doc: Save DFT ground state (density and potential)
  arguments:
    handler:
      type: gs_handler
      attr: in, required
      doc: Ground-state handler.
    file_name:
      type: string
      attr: in, required
      doc: Name of the file that stores the saved data.
    error_code:
      type: int
      attr: out, optional
      doc: Error code
@api end
*/
void
sirius_load_state(void** handler__, const char* file_name__, int* error_code__)
{
    call_sirius(
            [&]() {
                auto& gs = get_gs(handler__);
                std::string file_name(file_name__);
                gs.potential().load(file_name);
                gs.density().load(file_name);
            },
            error_code__);
}

/*
@api begin
sirius_set_density_matrix:
  doc: Set density matrix.
  arguments:
    handler:
      type: gs_handler
      attr: in, required
      doc: Ground-state handler.
    ia:
      type: int
      attr: in, required
      doc: Index of atom.
    dm:
      type: complex
      attr: in, required, dimension(ld, ld, 3)
      doc: Input density matrix.
    ld:
      type: int
      attr: in, required
      doc: Leading dimension of the density matrix.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
@api end
*/
void
sirius_set_density_matrix(void** handler__, int const* ia__, std::complex<double>* dm__, int const* ld__,
                          int* error_code__)
{
    call_sirius(
            [&]() {
                auto& gs = get_gs(handler__);
                mdarray<std::complex<double>, 3> dm({*ld__, *ld__, 3}, dm__);
                int ia       = *ia__ - 1;
                auto& atom   = gs.ctx().unit_cell().atom(ia);
                auto idx_map = atomic_orbital_index_map_QE(atom.type());
                int nbf      = atom.mt_basis_size();
                RTE_ASSERT(nbf <= *ld__);

                for (int icomp = 0; icomp < gs.ctx().num_mag_comp(); icomp++) {
                    for (int xi1 = 0; xi1 < nbf; xi1++) {
                        int p1 = phase_Rlm_QE(atom.type(), xi1);
                        for (int xi2 = 0; xi2 < nbf; xi2++) {
                            int p2 = phase_Rlm_QE(atom.type(), xi2);
                            gs.density().density_matrix(ia)(xi1, xi2, icomp) =
                                    dm(idx_map[xi1], idx_map[xi2], icomp) * static_cast<double>(p1 * p2);
                        }
                    }
                }
            },
            error_code__);
}

} // extern "C"
