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

#include "sirius.h"
#include "utils/any_ptr.hpp"

#define GET_SIM_CTX(h) auto& sim_ctx = static_cast<utils::any_ptr*>(*h)->get<sirius::Simulation_context>();

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
    return std::move(idx_map);
}

static inline int phase_Rlm_QE(sirius::Atom_type const& type__, int xi__)
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

/* @fortran begin function void sirius_finalize         Shut down the SIRIUS library
   @fortran argument in required bool call_mpi_fin      If .true. then MPI_Finalize must be called after the shutdown.
   @fortran end */
void sirius_finalize(bool const* call_mpi_fin__)
{
    sirius::finalize(*call_mpi_fin__);
}

/* @fortran begin function void sirius_start_timer      Start the timer.
   @fortran argument in required string name            Timer label.
   @fortran end */
void sirius_start_timer(char const* name__)
{
    std::string name(name__);
    if (!utils::timer::ftimers().count(name)) {
        utils::timer::ftimers().insert(std::make_pair(name, utils::timer(name)));
    } else {
        std::stringstream s;
        s << "timer " << name__ << " is already active";
        TERMINATE(s);
    }
}

/* @fortran begin function void sirius_stop_timer       Stop the running timer.
   @fortran argument in required string name            Timer label.
   @fortran end */
void sirius_stop_timer(char const* name__)
{
    std::string name(name__);
    if (utils::timer::ftimers().count(name)) {
        utils::timer::ftimers().erase(name);
    }
}

/* @fortran begin function void sirius_print_timers      Print all timers.
   @fortran end */
void sirius_print_timers(void)
{
    utils::timer::print();
}

/* @fortran begin function void sirius_serialize_timers    Save all timers to JSON file.
   @fortran argument in required string fname              Name of the output JSON file.
   @fortran end */
void sirius_serialize_timers(char const* fname__)
{
    json dict;
    dict["flat"] = utils::timer::serialize();
    dict["tree"] = utils::timer::serialize_tree();
    std::ofstream ofs(fname__, std::ofstream::out | std::ofstream::trunc);
    ofs << dict.dump(4);
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
    GET_SIM_CTX(handler__);
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
   @fortran argument in required string json_str                JSON string with parameters or a JSON file.
   @fortran end */
void sirius_import_parameters(void* const* handler__,
                              char  const* str__)
{
    GET_SIM_CTX(handler__);
    sim_ctx.import(std::string(str__));
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
   @fortran argument in optional double aw_cutoff                This is R_{mt} * gk_cutoff.
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
                           double const* aw_cutoff__,
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
                           char   const* hubbard_orbitals__)
{
    GET_SIM_CTX(handler__);
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
        sim_ctx.set_pw_cutoff(*pw_cutoff__);
    }
    if (gk_cutoff__ != nullptr) {
        sim_ctx.set_gk_cutoff(*gk_cutoff__);
    }
    if (aw_cutoff__ != nullptr) {
        sim_ctx.set_aw_cutoff(*aw_cutoff__);
    }
    if (auto_rmt__ != nullptr) {
        sim_ctx.set_auto_rmt(*auto_rmt__);
    }
    if (gamma_point__ != nullptr) {
        sim_ctx.set_gamma_point(*gamma_point__);
    }
    if (use_symmetry__ != nullptr) {
        sim_ctx.use_symmetry(*use_symmetry__);
    }
    if (so_correction__ != nullptr) {
        sim_ctx.set_so_correction(*so_correction__);
    }
    if (valence_rel__ != nullptr) {
        sim_ctx.set_valence_relativity(valence_rel__);
    }
    if (core_rel__ != nullptr) {
        sim_ctx.set_core_relativity(core_rel__);
    }
    if (esm_bc__ != nullptr) {
        sim_ctx.parameters_input().esm_bc_ = std::string(esm_bc__);
        sim_ctx.parameters_input().enable_esm_ = true;
    }
    if (iter_solver_tol__ != nullptr) {
        sim_ctx.set_iterative_solver_tolerance(*iter_solver_tol__);
    }
    if (iter_solver_tol_empty__ != nullptr) {
        sim_ctx.empty_states_tolerance(*iter_solver_tol_empty__);
    }
    if (iter_solver_type__ != nullptr) {
        sim_ctx.set_iterative_solver_type(std::string(iter_solver_type__));
    }
    if (verbosity__ != nullptr) {
        sim_ctx.set_verbosity(*verbosity__);
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
}

/* @fortran begin function void sirius_add_xc_functional         Add one of the XC functionals.
   @fortran argument in required void* handler                   Simulation context handler
   @fortran argument in required string name                     LibXC label of the functional.
   @fortran end */
void sirius_add_xc_functional(void* const* handler__,
                              char  const* name__)
{
    GET_SIM_CTX(handler__);
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
    GET_SIM_CTX(handler__);
    std::vector<int> dims(dims__, dims__ + *ndims__);
    sim_ctx.set_mpi_grid_dims(dims);
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
    GET_SIM_CTX(handler__);
    sim_ctx.unit_cell().set_lattice_vectors(vector3d<double>(a1__), vector3d<double>(a2__), vector3d<double>(a3__));
}

/* @fortran begin function void sirius_initialize_context     Initialize simulation context.
   @fortran argument in required void* handler                Simulation context handler.
   @fortran end */
void sirius_initialize_context(void* const* handler__)
{
    GET_SIM_CTX(handler__)
    sim_ctx.initialize();
}

/* @fortran begin function void sirius_update_context     Update simulation context after changing lattice or atomic positions.
   @fortran argument in required void* handler            Simulation context handler.
   @fortran end */
void sirius_update_context(void* const* handler__)
{
    GET_SIM_CTX(handler__)
    sim_ctx.update();
}

/* @fortran begin function void sirius_print_info      Print basic info
   @fortran argument in required void* handler         Simulation context handler.
   @fortran end */
void sirius_print_info(void* const* handler__)
{
    GET_SIM_CTX(handler__);
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
    auto& dft = static_cast<utils::any_ptr*>(*handler__)->get<sirius::DFT_ground_state>();
    std::string label(label__);

    std::map<std::string, sirius::Periodic_function<double>*> func_map = {
        {"rho",  &dft.density().component(0)},
        {"magz", &dft.density().component(1)},
        {"magx", &dft.density().component(2)},
        {"magy", &dft.density().component(3)},
        {"veff", &dft.potential().component(0)},
        {"bz",   &dft.potential().component(1)},
        {"bx",   &dft.potential().component(2)},
        {"by",   &dft.potential().component(3)}
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
void* sirius_create_kset(void* const*  handler__,
                         int    const* num_kpoints__,
                         double*       kpoints__,
                         double const* kpoint_weights__,
                         bool   const* init_kset__)
{
    GET_SIM_CTX(handler__);

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
                                   int const* k_grid__,
                                   int const* k_shift__,
                                   bool const* use_symmetry)
{
    GET_SIM_CTX(handler__);
    std::vector<int> k_grid(3);
    std::vector<int> k_shift(3);

    k_grid[0] = k_grid__[0];
    k_grid[1] = k_grid__[1];
    k_grid[2] = k_grid__[2];

    k_shift[0] = k_shift__[0];
    k_shift[1] = k_shift__[1];
    k_shift[2] = k_shift__[2];

    sirius::K_point_set* new_kset = new sirius::K_point_set(sim_ctx,
                                                            k_grid,
                                                            k_shift,
                                                            *use_symmetry);

    return new utils::any_ptr(new_kset);
}

/* @fortran begin function void* sirius_create_ground_state    Create a ground state object.
   @fortran argument in  required void*  ks_handler            Handler of the k-point set.
   @fortran end */
void* sirius_create_ground_state(void* const* ks_handler__)
{
    auto& ks = static_cast<utils::any_ptr*>(*ks_handler__)->get<sirius::K_point_set>();

    return new utils::any_ptr(new sirius::DFT_ground_state(ks));
}

/* @fortran begin function void sirius_find_ground_state        Find the ground state
   @fortran argument in required void* gs_handler               Handler of the ground state
   @fortran argument in optional bool  save__                   boolean variable indicating if we want to save the ground state
   @fortran end */
void sirius_find_ground_state(void* const* gs_handler__, bool const *save__)
{
    auto& gs = static_cast<utils::any_ptr*>(*gs_handler__)->get<sirius::DFT_ground_state>();
    auto& ctx = gs.ctx();
    auto& inp = ctx.parameters_input();
    gs.initial_state();

    if (save__ != nullptr) {
        auto result = gs.find(inp.potential_tol_,
                              inp.energy_tol_,
                              inp.num_dft_iter_,
                              true);
    } else {
        auto result = gs.find(inp.potential_tol_,
                              inp.energy_tol_,
                              inp.num_dft_iter_,
                              false);
    }
}

/* @fortran begin function void sirius_update_ground_state   Update a ground state object after change of atomic coordinates or lattice vectors.
   @fortran argument in  required void*  gs_handler          Ground-state handler.
   @fortran end */
void sirius_update_ground_state(void** handler__)
{
    auto& gs = static_cast<utils::any_ptr*>(*handler__)->get<sirius::DFT_ground_state>();
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
    GET_SIM_CTX(handler__);

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
    GET_SIM_CTX(handler__);

    auto& type = sim_ctx.unit_cell().atom_type(std::string(label__));
    type.set_radial_grid(*num_radial_points__, radial_points__);
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
    GET_SIM_CTX(handler__);

    auto& type = sim_ctx.unit_cell().atom_type(std::string(atom_type__));
    std::string label(label__);

    if (label == "beta") {
        if (l__ == nullptr) {
            TERMINATE("orbital quantum number must be provided for beta-projector");
        }
        type.add_beta_radial_function(*l__, std::vector<double>(rf__, rf__ + *num_points__));
    } else if (label == "ps_atomic_wf") {
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
    } else if (label == "q_aug") {
        if (l__ == nullptr) {
            TERMINATE("orbital quantum number must be provided for augmentation charge radial function");
        }
        if (idxrf1__ == nullptr || idxrf2__ == nullptr) {
            TERMINATE("both radial-function indices must be provided for augmentation charge radial function");
        }
        type.add_q_radial_function(*idxrf1__, *idxrf2__, *l__, std::vector<double>(rf__, rf__ + *num_points__));
    } else if (label == "ae_paw_wf") {
        type.add_ae_paw_wf(std::vector<double>(rf__, rf__ + *num_points__));
    } else  if (label == "ps_paw_wf") {
        type.add_ps_paw_wf(std::vector<double>(rf__, rf__ + *num_points__));
    } else if (label == "ae_paw_core") {
        type.paw_ae_core_charge_density(std::vector<double>(rf__, rf__ + *num_points__));
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
    GET_SIM_CTX(handler__);
    auto& type = sim_ctx.unit_cell().atom_type(std::string(label__));
    type.set_hubbard_correction();
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
    GET_SIM_CTX(handler__);
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
    GET_SIM_CTX(handler__);

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
    GET_SIM_CTX(handler__);
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
    GET_SIM_CTX(handler__);
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

    auto& gs = static_cast<utils::any_ptr*>(*handler__)->get<sirius::DFT_ground_state>();

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

    auto& gs = static_cast<utils::any_ptr*>(*handler__)->get<sirius::DFT_ground_state>();

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
    GET_SIM_CTX(handler__);

    int iat = sim_ctx.unit_cell().atom_type(atom_label).id();

    auto make_pw_coeffs = [&](std::function<double(double)> f)
    {
        mdarray<int, 2> gvec(gvl__, 3, *ngv__);

        double fourpi_omega = fourpi / sim_ctx.unit_cell().omega();
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < *ngv__; i++) {
            auto gc = sim_ctx.unit_cell().reciprocal_lattice_vectors() *  vector3d<int>(gvec(0, i), gvec(1, i), gvec(2, i));
            pw_coeffs__[i] = fourpi_omega * f(gc.length());
        }
    };

    // TODO: if radial integrals take considerable time, cache them in Simulation_context

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
        s << "wrong label in sirius_get_pw_coeffs_real()" << std::endl
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
    auto& gs = static_cast<utils::any_ptr*>(*gs_handler__)->get<sirius::DFT_ground_state>();
    auto& ks = static_cast<utils::any_ptr*>(*ks_handler__)->get<sirius::K_point_set>();
    sirius::Band(ks.ctx()).initialize_subspace(ks, gs.hamiltonian());
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
    auto& gs = static_cast<utils::any_ptr*>(*gs_handler__)->get<sirius::DFT_ground_state>();
    auto& ks = static_cast<utils::any_ptr*>(*ks_handler__)->get<sirius::K_point_set>();
    if (iter_solver_tol__ != nullptr) {
        ks.ctx().set_iterative_solver_tolerance(*iter_solver_tol__);
    }
    sirius::Band(ks.ctx()).solve(ks, gs.hamiltonian(), *precompute__);
}

/* @fortran begin function void sirius_generate_d_operator_matrix     Generate D-operator matrix.
   @fortran argument in  required void*   handler                     Ground state handler.
   @fortran end */
void sirius_generate_d_operator_matrix(void* const* handler__)
{
    auto& gs = static_cast<utils::any_ptr*>(*handler__)->get<sirius::DFT_ground_state>();
    gs.potential().generate_D_operator_matrix();
}

/* @fortran begin function void sirius_generate_initial_density     Generate initial density.
   @fortran argument in  required void*   handler                   Ground state handler.
   @fortran end */
void sirius_generate_initial_density(void* const* handler__)
{
    auto& gs = static_cast<utils::any_ptr*>(*handler__)->get<sirius::DFT_ground_state>();
    gs.density().initial_density();
}

/* @fortran begin function void sirius_generate_effective_potential     Generate effective potential and magnetic field.
   @fortran argument in  required void*   handler                       Ground state handler.
   @fortran end */
void sirius_generate_effective_potential(void* const* handler__)
{
    auto& gs = static_cast<utils::any_ptr*>(*handler__)->get<sirius::DFT_ground_state>();
    gs.potential().generate(gs.density());
}

/* @fortran begin function void sirius_generate_density    Generate charge density and magnetization.
   @fortran argument in  required void*   gs_handler       Ground state handler.
   @fortran end */
void sirius_generate_density(void* const* gs_handler__)
{
    auto& gs = static_cast<utils::any_ptr*>(*gs_handler__)->get<sirius::DFT_ground_state>();
    gs.density().generate(gs.k_point_set());
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
    auto& ks = static_cast<utils::any_ptr*>(*ks_handler__)->get<sirius::K_point_set>();
    int ik = *ik__ - 1;
    for (int i = 0; i < ks.ctx().num_bands(); i++) {
        ks[ik]->band_occupancy(i, *ispn__) = band_occupancies__[i];
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
    auto& ks = static_cast<utils::any_ptr*>(*ks_handler__)->get<sirius::K_point_set>();
    int ik = *ik__ - 1;
    for (int i = 0; i < ks.ctx().num_bands(); i++) {
        band_energies__[i] = ks[ik]->band_energy(i, *ispn__);
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
    GET_SIM_CTX(handler__);

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
    GET_SIM_CTX(handler__);

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
    GET_SIM_CTX(handler__);

    auto& type = sim_ctx.unit_cell().atom_type(std::string(label__));
    mdarray<double, 2> q_mtrx(q_mtrx__, *ld__, *ld__);

    auto idx_map = atomic_orbital_index_map_QE(type);
    int nbf = type.mt_basis_size();

    for (int xi1 = 0; xi1 < nbf; xi1++) {
        int p1 = phase_Rlm_QE(type, xi1);
        for (int xi2 = 0; xi2 < nbf; xi2++) {
            int p2 = phase_Rlm_QE(type, xi2);
            sim_ctx.augmentation_op(type.id()).q_mtrx(xi1, xi2) = q_mtrx(idx_map[xi1], idx_map[xi2]) * p1 * p2;
        }
    }
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
    GET_SIM_CTX(handler__);

    auto& type = sim_ctx.unit_cell().atom_type(std::string(label__));
    mdarray<double, 2> q_mtrx(q_mtrx__, *ld__, *ld__);

    auto idx_map = atomic_orbital_index_map_QE(type);
    int nbf = type.mt_basis_size();

    for (int xi1 = 0; xi1 < nbf; xi1++) {
        int p1 = phase_Rlm_QE(type, xi1);
        for (int xi2 = 0; xi2 < nbf; xi2++) {
            int p2 = phase_Rlm_QE(type, xi2);
            q_mtrx(idx_map[xi1], idx_map[xi2]) = sim_ctx.augmentation_op(type.id()).q_mtrx(xi1, xi2) * p1 * p2;
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
                               int            const* ld__)
{
    auto& gs = static_cast<utils::any_ptr*>(*handler__)->get<sirius::DFT_ground_state>();

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
                dm(idx_map[i], idx_map[j], icomp) = gs.density().density_matrix()(i, j, icomp, *ia__ - 1) * static_cast<double>(p1 * p2);
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
    auto& gs = static_cast<utils::any_ptr*>(*handler__)->get<sirius::DFT_ground_state>();

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
    auto& gs = static_cast<utils::any_ptr*>(*handler__)->get<sirius::DFT_ground_state>();

    std::map<std::string, double (sirius::DFT_ground_state::*)(void) const> func = {
        {"total",    &sirius::DFT_ground_state::total_energy},
        {"evalsum",  &sirius::DFT_ground_state::eval_sum},
        {"exc",      &sirius::DFT_ground_state::energy_exc},
        {"vxc",      &sirius::DFT_ground_state::energy_vxc},
        {"bxc",      &sirius::DFT_ground_state::energy_bxc},
        {"veff",     &sirius::DFT_ground_state::energy_veff},
        {"vloc",     &sirius::DFT_ground_state::energy_vloc},
        {"vha",      &sirius::DFT_ground_state::energy_vha},
        {"enuc",     &sirius::DFT_ground_state::energy_enuc},
        {"kin",      &sirius::DFT_ground_state::energy_kin}
    };

    std::string label(label__);

    try {
        *energy__ = (gs.*func.at(label))();
    } catch(...) {
        TERMINATE("wrong label");
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

    auto& gs = static_cast<utils::any_ptr*>(*handler__)->get<sirius::DFT_ground_state>();

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

    auto& gs = static_cast<utils::any_ptr*>(*handler__)->get<sirius::DFT_ground_state>();

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
    GET_SIM_CTX(handler__);

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

    GET_SIM_CTX(handler__);

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
        double x = sim_ctx.augmentation_op(type.id()).q_pw(idx, 2 * ig);
        double y = sim_ctx.augmentation_op(type.id()).q_pw(idx, 2 * ig + 1);
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

    auto& kset = static_cast<utils::any_ptr*>(*ks_handler__)->get<sirius::K_point_set>();
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
                    gvp = std::unique_ptr<Gvec_partition>(new Gvec_partition(gkvec, sim_ctx.comm_fft_coarse(),
                                                                             sim_ctx.comm_band_ortho_fft_coarse()));
                    wf = std::unique_ptr<Wave_functions>(new Wave_functions(*gvp, sim_ctx.num_bands()));
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
    GET_SIM_CTX(handler__);

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
    auto& gs = static_cast<utils::any_ptr*>(*handler__)->get<sirius::DFT_ground_state>();
    gs.hamiltonian().U().hubbard_compute_occupation_numbers(gs.k_point_set());
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
    auto& gs = static_cast<utils::any_ptr*>(*handler__)->get<sirius::DFT_ground_state>();
    gs.hamiltonian().U().access_hubbard_occupancies("set", occ__, ld__);
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
    auto& gs = static_cast<utils::any_ptr*>(*handler__)->get<sirius::DFT_ground_state>();
    gs.hamiltonian().U().access_hubbard_occupancies("get", occ__, ld__);
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
    auto& gs = static_cast<utils::any_ptr*>(*handler__)->get<sirius::DFT_ground_state>();
    gs.hamiltonian().U().access_hubbard_potential("set", pot__, ld__);
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
    auto& gs = static_cast<utils::any_ptr*>(*handler__)->get<sirius::DFT_ground_state>();
    gs.hamiltonian().U().access_hubbard_potential("get", pot__, ld__);
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
    GET_SIM_CTX(handler__);
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
    GET_SIM_CTX(handler__);
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
    GET_SIM_CTX(handler__);
    auto& type = sim_ctx.unit_cell().atom_type(std::string(label__));
    type.set_configuration(*n__, *l__, *k__, *occupancy__, *core__);
}

/* @fortran begin function void sirius_generate_coulomb_potential    Generate Coulomb potential by solving Poisson equation
   @fortran argument in required void*   handler   Ground state handler
   @fortran argument out required double vclmt     Muffin-tin part of potential
   @fortran argument out required double vclrg     Regular-grid part of potential
   @fortran end */
void sirius_generate_coulomb_potential(void* const* handler__,
                                       double*      vclmt__,
                                       double*      vclrg__)
{
    auto& gs = static_cast<utils::any_ptr*>(*handler__)->get<sirius::DFT_ground_state>();

    gs.density().rho().fft_transform(-1);
    gs.potential().poisson(gs.density().rho());
    gs.potential().hartree_potential().copy_to_global_ptr(vclmt__, vclrg__);
}

/* @fortran begin function void sirius_generate_xc_potential    Generate XC potential using LibXC
   @fortran argument in required void*   handler   Ground state handler
   @fortran argument out required double vxcmt     Muffin-tin part of potential
   @fortran argument out required double vxcrg     Regular-grid part of potential
   @fortran argument out required double bxcmt     Muffin-tin part of effective magentic field
   @fortran argument out required double bxcrg     Regular-grid part of effective magnetic field
   @fortran end */
void sirius_generate_xc_potential(void* const* handler__,
                                  double*      vxcmt__,
                                  double*      vxcrg__,
                                  double*      bxcmt__,
                                  double*      bxcrg__)
{
    auto& gs = static_cast<utils::any_ptr*>(*handler__)->get<sirius::DFT_ground_state>();
    gs.potential().xc(gs.density());
    gs.potential().xc_potential().copy_to_global_ptr(vxcmt__, vxcrg__);

    if (gs.ctx().num_mag_dims() == 0) {
        return;
    }

    /* set temporary array wrapper */
    mdarray<double, 4> bxcmt(bxcmt__, gs.ctx().lmmax_pot(), gs.ctx().unit_cell().max_num_mt_points(),
                             gs.ctx().unit_cell().num_atoms(), gs.ctx().num_mag_dims());
    mdarray<double, 2> bxcrg(bxcrg__, gs.ctx().fft().local_size(), gs.ctx().num_mag_dims());

    if (gs.ctx().num_mag_dims() == 1) {
        /* z component */
        gs.potential().effective_magnetic_field(0).copy_to_global_ptr(&bxcmt(0, 0, 0, 0), &bxcrg(0, 0));
    } else {
        /* z component */
        gs.potential().effective_magnetic_field(0).copy_to_global_ptr(&bxcmt(0, 0, 0, 2), &bxcrg(0, 2));
        /* x component */
        gs.potential().effective_magnetic_field(1).copy_to_global_ptr(&bxcmt(0, 0, 0, 0), &bxcrg(0, 0));
        /* y component */
        gs.potential().effective_magnetic_field(2).copy_to_global_ptr(&bxcmt(0, 0, 0, 1), &bxcrg(0, 1));
    }
}

} // extern "C"
