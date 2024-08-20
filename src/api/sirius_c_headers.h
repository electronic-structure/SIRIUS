#include <stdbool.h>
#include <complex.h>

/*
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
*/
void
sirius_initialize(bool const* call_mpi_init__, int* error_code__);

/*
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
*/
void
sirius_finalize(bool const* call_mpi_fin__, bool const* call_device_reset__, bool const* call_fftw_fin__,
                int* error_code__);

/*
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
*/
void
sirius_start_timer(char const* name__, int* error_code__);

/*
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
*/
void
sirius_stop_timer(char const* name__, int* error_code__);

/*
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
*/
void
sirius_print_timers(bool* flatten__, int* error_code__);

/*
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
*/
void
sirius_serialize_timers(char const* fname__, int* error_code__);

/*
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
*/
void
sirius_context_initialized(void* const* handler__, bool* status__, int* error_code__);

/*
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
*/
void
sirius_create_context(int fcomm__, void** handler__, int* fcomm_k__, int* fcomm_band__, int* error_code__);

/*
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
*/
void
sirius_import_parameters(void* const* handler__, char const* str__, int* error_code__);

/*
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
                      char const* electronic_structure_method__, int* error_code__);

/*
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
*/
void
sirius_get_parameters(void* const* handler__, int* lmax_apw__, int* lmax_rho__, int* lmax_pot__, int* num_fv_states__,
                      int* num_bands__, int* num_spins__, int* num_mag_dims__, double* pw_cutoff__, double* gk_cutoff__,
                      int* fft_grid_size__, int* auto_rmt__, bool* gamma_point__, bool* use_symmetry__,
                      bool* so_correction__, double* iter_solver_tol__, double* iter_solver_tol_empty__,
                      int* verbosity__, bool* hubbard_correction__, double* evp_work_count__, int* num_loc_op_applied__,
                      int* num_sym_op__, char* electronic_structure_method__, int* error_code__);

/*
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
*/
void
sirius_add_xc_functional(void* const* handler__, char const* name__, int* error_code__);

/*
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
*/
void
sirius_set_mpi_grid_dims(void* const* handler__, int const* ndims__, int const* dims__, int* error_code__);

/*
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
*/
void
sirius_set_lattice_vectors(void* const* handler__, double const* a1__, double const* a2__, double const* a3__,
                           int* error_code__);

/*
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
*/
void
sirius_initialize_context(void* const* handler__, int* error_code__);

/*
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
*/
void
sirius_update_context(void* const* handler__, int* error_code__);

/*
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
*/
void
sirius_print_info(void* const* handler__, int* error_code__);

/*
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
*/
void
sirius_free_object_handler(void** handler__, int* error_code__);

/*
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
*/
void
sirius_set_periodic_function_ptr(void* const* handler__, char const* label__, double* f_mt__, int const* lmmax__,
                                 int const* nrmtmax__, int const* num_atoms__, double* f_rg__, int const* size_x__,
                                 int const* size_y__, int const* size_z__, int const* offset_z__, int* error_code__);

/*
sirius_set_periodic_function:
  doc: Set values of the periodic function.
  arguments:
    gs_handler:
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
*/
void
sirius_set_periodic_function(void* const* gs_handler__, char const* label__, double* f_mt__, int const* lmmax__,
                             int const* nrmtmax__, int const* num_atoms__, double* f_rg__, int const* size_x__,
                             int const* size_y__, int const* size_z__, int const* offset_z__, int* error_code__);

/*
sirius_get_periodic_function:
  doc: Get values of the periodic function.
  arguments:
    gs_handler:
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
*/
void
sirius_get_periodic_function(void* const* gs_handler__, char const* label__, double* f_mt__, int const* lmmax__,
                             int const* nrmtmax__, int const* num_atoms__, double* f_rg__, int const* size_x__,
                             int const* size_y__, int const* size_z__, int const* offset_z__, int* error_code__);

/*
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
*/
void
sirius_create_kset(void* const* handler__, int const* num_kpoints__, double* kpoints__, double const* kpoint_weights__,
                   bool const* init_kset__, void** kset_handler__, int* error_code__);

/*
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
*/
void
sirius_create_kset_from_grid(void* const* handler__, int const* k_grid__, int const* k_shift__,
                             bool const* use_symmetry, void** kset_handler__, int* error_code__);

/*
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
*/
void
sirius_create_ground_state(void* const* ks_handler__, void** gs_handler__, int* error_code__);

/*
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
*/
void
sirius_initialize_kset(void* const* ks_handler__, int* count__, int* error_code__);

/*
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
*/
void
sirius_find_ground_state(void* const* gs_handler__, double const* density_tol__, double const* energy_tol__,
                         double const* iter_solver_tol__, bool const* initial_guess__, int const* max_niter__,
                         bool const* save_state__, bool* converged__, int* niter__, double* rho_min__,
                         int* error_code__);

/*
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
*/
void
sirius_check_scf_density(void* const* gs_handler__, int* error_code__);

/*
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
*/
void
sirius_update_ground_state(void** gs_handler__, int* error_code__);

/*
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
*/
void
sirius_add_atom_type(void* const* handler__, char const* label__, char const* fname__, int const* zn__,
                     char const* symbol__, double const* mass__, bool const* spin_orbit__, int* error_code__);

/*
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
*/
void
sirius_set_atom_type_radial_grid(void* const* handler__, char const* label__, int const* num_radial_points__,
                                 double const* radial_points__, int* error_code__);

/*
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
*/
void
sirius_set_atom_type_radial_grid_inf(void* const* handler__, char const* label__, int const* num_radial_points__,
                                     double const* radial_points__, int* error_code__);

/*
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
*/
void
sirius_add_atom_type_radial_function(void* const* handler__, char const* atom_type__, char const* label__,
                                     double const* rf__, int const* num_points__, int const* n__, int const* l__,
                                     int const* idxrf1__, int const* idxrf2__, double const* occ__, int* error_code__);

/*
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
*/
void
sirius_set_atom_type_hubbard(void* const* handler__, char const* label__, int const* l__, int const* n__,
                             double const* occ__, double const* U__, double const* J__, double const* alpha__,
                             double const* beta__, double const* J0__, int* error_code__);

/*
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
*/
void
sirius_set_atom_type_dion(void* const* handler__, char const* label__, int const* num_beta__, double* dion__,
                          int* error_code__);

/*
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
*/
void
sirius_set_atom_type_paw(void* const* handler__, char const* label__, double const* core_energy__,
                         double const* occupations__, int const* num_occ__, int* error_code__);

/*
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
*/
void
sirius_add_atom(void* const* handler__, char const* label__, double const* position__, double const* vector_field__,
                int* error_code__);

/*
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
*/
void
sirius_set_atom_position(void* const* handler__, int const* ia__, double const* position__, int* error_code__);

/*
sirius_set_pw_coeffs:
  doc: Set plane-wave coefficients of a periodic function.
  arguments:
    gs_handler:
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
*/
void
sirius_set_pw_coeffs(void* const* gs_handler__, char const* label__, double complex const* pw_coeffs__,
                     bool const* transform_to_rg__, int const* ngv__, int* gvl__, int const* comm__, int* error_code__);

/*
sirius_get_pw_coeffs:
  doc: Get plane-wave coefficients of a periodic function.
  arguments:
    gs_handler:
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
*/
void
sirius_get_pw_coeffs(void* const* gs_handler__, char const* label__, double complex* pw_coeffs__, int const* ngv__,
                     int* gvl__, int const* comm__, int* error_code__);

/*
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
*/
void
sirius_initialize_subspace(void* const* gs_handler__, void* const* ks_handler__, int* error_code__);

/*
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
    iter_solver_steps:
      type: int
      attr: in, optional
      doc: Iterative solver number of steps.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
*/
void
sirius_find_eigen_states(void* const* gs_handler__, void* const* ks_handler__, bool const* precompute_pw__,
                         bool const* precompute_rf__, bool const* precompute_ri__, double const* iter_solver_tol__,
                         int const* iter_solver_steps__, int* error_code__);

/*
sirius_generate_initial_density:
  doc: Generate initial density.
  arguments:
    gs_handler:
      type: gs_handler
      attr: in, required
      doc: Ground state handler.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
*/
void
sirius_generate_initial_density(void* const* gs_handler__, int* error_code__);

/*
sirius_generate_effective_potential:
  doc: Generate effective potential and magnetic field.
  arguments:
    gs_handler:
      type: gs_handler
      attr: in, required
      doc: Ground state handler.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
*/
void
sirius_generate_effective_potential(void* const* gs_handler__, int* error_code__);

/*
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
*/
void
sirius_generate_density(void* const* gs_handler__, bool const* add_core__, bool const* transform_to_rg__,
                        bool const* paw_only__, int* error_code__);

/*
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
*/
void
sirius_set_band_occupancies(void* const* ks_handler__, int const* ik__, int const* ispn__,
                            double const* band_occupancies__, int* error_code__);

/*
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
*/
void
sirius_get_band_occupancies(void* const* ks_handler__, int const* ik__, int const* ispn__, double* band_occupancies__,
                            int* error_code__);

/*
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
*/
void
sirius_get_band_energies(void* const* ks_handler__, int const* ik__, int const* ispn__, double* band_energies__,
                         int* error_code__);

/*
sirius_get_energy:
  doc: Get one of the total energy components.
  arguments:
    gs_handler:
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
*/
void
sirius_get_energy(void* const* gs_handler__, char const* label__, double* energy__, int* error_code__);

/*
sirius_get_forces:
  doc: Get one of the total force components.
  arguments:
    gs_handler:
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
*/
void
sirius_get_forces(void* const* gs_handler__, char const* label__, double* forces__, int* error_code__);

/*
sirius_get_stress_tensor:
  doc: Get one of the stress tensor components.
  arguments:
    gs_handler:
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
*/
void
sirius_get_stress_tensor(void* const* gs_handler__, char const* label__, double* stress_tensor__, int* error_code__);

/*
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
*/
void
sirius_get_num_beta_projectors(void* const* handler__, char const* label__, int* num_bp__, int* error_code__);

/*
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
*/
void
sirius_get_wave_functions(void* const* ks_handler__, double const* vkl__, int const* spin__, int const* num_gvec_loc__,
                          int const* gvec_loc__, double complex* evec__, int const* ld__,
                          int const* num_spin_comp__, int* error_code__);

/*
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
*/
void
sirius_add_atom_type_aw_descriptor(void* const* handler__, char const* label__, int const* n__, int const* l__,
                                   double const* enu__, int const* dme__, bool const* auto_enu__, int* error_code__);

/*
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
*/
void
sirius_add_atom_type_lo_descriptor(void* const* handler__, char const* label__, int const* ilo__, int const* n__,
                                   int const* l__, double const* enu__, int const* dme__, bool const* auto_enu__,
                                   int* error_code__);

/*
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
*/
void
sirius_set_atom_type_configuration(void* const* handler__, char const* label__, int const* n__, int const* l__,
                                   int const* k__, double const* occupancy__, bool const* core__, int* error_code__);

/*
sirius_generate_coulomb_potential:
  doc: Generate Coulomb potential by solving Poisson equation
  arguments:
    gs_handler:
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
*/
void
sirius_generate_coulomb_potential(void* const* gs_handler__, double* vh_el__, int* error_code__);

/*
sirius_generate_xc_potential:
  doc: Generate XC potential using LibXC
  arguments:
    gs_handler:
      type: gs_handler
      attr: in, required
      doc: Ground state handler
    error_code:
      type: int
      attr: out, optional
      doc: Error code
*/
void
sirius_generate_xc_potential(void* const* gs_handler__, int* error_code__);

/*
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
*/
void
sirius_get_kpoint_inter_comm(void* const* handler__, int* fcomm__, int* error_code__);

/*
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
*/
void
sirius_get_kpoint_inner_comm(void* const* handler__, int* fcomm__, int* error_code__);

/*
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
*/
void
sirius_get_fft_comm(void* const* handler__, int* fcomm__, int* error_code__);

/*
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
*/
void
sirius_get_num_gvec(void* const* handler__, int* num_gvec__, int* error_code__);

/*
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
*/
void
sirius_get_gvec_arrays(void* const* handler__, int* gvec__, double* gvec_cart__, double* gvec_len__,
                       int* index_by_gvec__, int* error_code__);

/*
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
*/
void
sirius_get_num_fft_grid_points(void* const* handler__, int* num_fft_grid_points__, int* error_code__);

/*
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
*/
void
sirius_get_fft_index(void* const* handler__, int* fft_index__, int* error_code__);

/*
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
*/
void
sirius_get_max_num_gkvec(void* const* ks_handler__, int* max_num_gkvec__, int* error_code__);

/*
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
*/
void
sirius_get_gkvec_arrays(void* const* ks_handler__, int* ik__, int* num_gkvec__, int* gvec_index__, double* gkvec__,
                        double* gkvec_cart__, double* gkvec_len, double* gkvec_tp__, int* error_code__);

/*
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
*/
void
sirius_get_step_function(void* const* handler__, double complex* cfunig__, double* cfunrg__, int* num_rg_points__,
                         int* error_code__);

/*
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
*/
void
sirius_set_h_radial_integrals(void* const* handler__, int* ia__, int* lmmax__, double* val__, int* l1__, int* o1__,
                              int* ilo1__, int* l2__, int* o2__, int* ilo2__, int* error_code__);

/*
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
*/
void
sirius_set_o_radial_integral(void* const* handler__, int* ia__, double* val__, int* l__, int* o1__, int* ilo1__,
                             int* o2__, int* ilo2__, int* error_code__);

/*
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
*/
void
sirius_set_o1_radial_integral(void* const* handler__, int* ia__, double* val__, int* l1__, int* o1__, int* ilo1__,
                              int* l2__, int* o2__, int* ilo2__, int* error_code__);

/*
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
*/
void
sirius_set_radial_function(void* const* handler__, int const* ia__, int const* deriv_order__, double const* f__,
                           int const* l__, int const* o__, int const* ilo__, int* error_code__);

/*
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
*/
void
sirius_set_equivalent_atoms(void* const* handler__, int* equivalent_atoms__, int* error_code__);

/*
sirius_update_atomic_potential:
  doc: Set the new spherical potential.
  arguments:
    gs_handler:
      type: gs_handler
      attr: in, required
      doc: Ground state handler.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
*/
void
sirius_update_atomic_potential(void* const* gs_handler__, int* error_code__);

/*
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
*/
void
sirius_option_get_number_of_sections(int* length__, int* error_code__);

/*
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
*/
void
sirius_option_get_section_name(int elem__, char* section_name__, int section_name_length__, int* error_code__);

/*
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
*/
void
sirius_option_get_section_length(char const* section__, int* length__, int* error_code__);

/*
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
*/
void
sirius_option_get_info(char const* section__, int elem__, char* key_name__, int key_name_len__, int* type__,
                       int* length__, int* enum_size__, char* title__, int title_len__, char* description__,
                       int description_len__, int* error_code__);

/*
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
*/
void
sirius_option_get(char const* section__, char const* name__, int const* type__, void* data_ptr__,
                  int const* max_length__, int const* enum_idx__, int* error_code__);

/*
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
*/
void
sirius_option_set(void* const* handler__, char const* section__, char const* name__, int const* type__,
                  void const* data_ptr__, int const* max_length__, bool const* append__, int* error_code__);

/*
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
*/
void
sirius_dump_runtime_setup(void* const* handler__, char* filename__, int* error_code__);

/*
sirius_get_fv_eigen_vectors:
  doc: Get the first-variational eigen vectors
  arguments:
    ks_handler:
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
*/
void
sirius_get_fv_eigen_vectors(void* const* ks_handler__, int const* ik__, double complex* fv_evec__, int const* ld__,
                            int const* num_fv_states__, int* error_code__);

/*
sirius_get_fv_eigen_values:
  doc: Get the first-variational eigen values
  arguments:
    ks_handler:
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
*/
void
sirius_get_fv_eigen_values(void* const* ks_handler__, int const* ik__, double* fv_eval__, int const* num_fv_states__,
                           int* error_code__);

/*
sirius_get_sv_eigen_vectors:
  doc: Get the second-variational eigen vectors
  arguments:
    ks_handler:
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
*/
void
sirius_get_sv_eigen_vectors(void* const* ks_handler__, int const* ik__, double complex* sv_evec__,
                            int const* num_bands__, int* error_code__);

/*
sirius_set_rg_values:
  doc: Set the values of the function on the regular grid.
  arguments:
    gs_handler:
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
*/
void
sirius_set_rg_values(void* const* gs_handler__, char const* label__, int const* grid_dims__, int const* local_box_origin__,
                     int const* local_box_size__, int const* fcomm__, double const* values__,
                     bool const* transform_to_pw__, int* error_code__);

/*
sirius_get_rg_values:
  doc: Get the values of the function on the regular grid.
  arguments:
    gs_handler:
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
*/
void
sirius_get_rg_values(void* const* gs_handler__, char const* label__, int const* grid_dims__, int const* local_box_origin__,
                     int const* local_box_size__, int const* fcomm__, double* values__, bool const* transform_to_rg__,
                     int* error_code__);

/*
sirius_get_total_magnetization:
  doc: Get the total magnetization of the system.
  arguments:
    gs_handler:
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
*/
void
sirius_get_total_magnetization(void* const* gs_handler__, double* mag__, int* error_code__);

/*
sirius_get_num_kpoints:
  doc: Get the total number of kpoints
  arguments:
    ks_handler:
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
*/
void
sirius_get_num_kpoints(void* const* ks_handler__, int* num_kpoints__, int* error_code__);

/*
sirius_get_kpoint_properties:
  doc: Get the kpoint properties
  arguments:
    ks_handler:
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
*/
void
sirius_get_kpoint_properties(void* const* ks_handler__, int const* ik__, double* weight__, double* coordinates__,
                             int* error_code__);

/*
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
*/
void
sirius_set_callback_function(void* const* handler__, char const* label__, void (*fptr__)(), int* error_code__);

/*
sirius_nlcg:
  doc: Robust wave function optimizer.
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
*/
void
sirius_nlcg(void* const* gs_handler__, void* const* ks_handler__, int* error_code__);

/*
sirius_nlcg_params:
  doc: Robust wave function optimizer
  arguments:
    gs_handler:
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
*/
void
sirius_nlcg_params(void* const* gs_handler__, void* const* ks_handler__, double const* temp__, char const* smearing__,
                   double const* kappa__, double const* tau__, double const* tol__, int const* maxiter__,
                   int const* restart__, char const* processing_unit__, bool* converged__, int* error_code__);

/*
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
*/
void
sirius_add_hubbard_atom_pair(void* const* handler__, int* const atom_pair__, int* const translation__, int* const n__,
                             int* const l__, const double* const coupling__, int* error_code__);

/*
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
*/
void
sirius_set_hubbard_contrained_parameters(void* const* handler__, double const* hubbard_conv_thr__,
                                         double const* hubbard_mixing_beta__, double const* hubbard_strength__,
                                         int const* hubbard_maxstep__, char const* hubbard_constraint_type__,
                                         int* const error_code__);

/*
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
*/
void
sirius_add_hubbard_atom_constraint(void* const* handler__, int* const atom_id__, int* const n__, int* const l__,
                                   int* const lmax_at__, const double* const occ__, int* const orbital_order__,
                                   int* const error_code__);

/*
sirius_create_H0:
  doc: Generate H0.
  arguments:
    gs_handler:
      type: gs_handler
      attr: in, required
      doc: Ground state handler.
    error_code:
      type: int
      attr: out, optional
      doc: Error code
*/
void
sirius_create_H0(void* const* gs_handler__, int* error_code__);

/*
sirius_linear_solver:
  doc: Interface to linear solver.
  arguments:
    gs_handler:
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
    nbnd_occ_k:
      type: int
      attr: in, required
      doc: Number of occupied bands at k.
    nbnd_occ_kq:
      type: int
      attr: in, required
      doc: Number of occupied bands at k+q.
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
*/
void
sirius_linear_solver(void* const* gs_handler__, double const* vkq__, int const* num_gvec_kq_loc__,
                     int const* gvec_kq_loc__, double complex* dpsi__, double complex* psi__,
                     double* eigvals__, double complex* dvpsi__, int const* ld__, int const* num_spin_comp__,
                     double const* alpha_pv__, int const* spin__, int const* nbnd_occ_k__, int const* nbnd_occ_kq__,
                     double const* tol__, int* niter__, int* error_code__);

/*
sirius_generate_rhoaug_q:
  doc: Generate augmentation charge in case of complex density (linear response)
  arguments:
    gs_handler:
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
      attr: in, required, dimension(ldd, num_atoms, num_spin_comp)
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
*/
void
sirius_generate_rhoaug_q(void* const* gs_handler__, int const* iat__, int const* num_atoms__, int const* num_gvec_loc__,
                         int const* num_spin_comp__, double complex const* qpw__, int const* ldq__,
                         double complex const* phase_factors_q__, int const* mill__,
                         double complex const* dens_mtrx__, int const* ldd__, double complex* rho_aug__,
                         int* error_code__);

/*
sirius_generate_d_operator_matrix:
  doc: Generate D-operator matrix.
  arguments:
    gs_handler:
      type: gs_handler
      attr: in, required
      doc: Ground state handler.
    error_code:
      type: int
      attr: out, optional
      doc: Error code
*/
void
sirius_generate_d_operator_matrix(void* const* gs_handler__, int* error_code__);

/*
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
*/
void
sirius_save_state(void** gs_handler__, const char* file_name__, int* error_code__);

/*
sirius_load_state:
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
*/
void
sirius_load_state(void** gs_handler__, const char* file_name__, int* error_code__);

/*
sirius_set_density_matrix:
  doc: Set density matrix.
  arguments:
    gs_handler:
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
*/
void
sirius_set_density_matrix(void** gs_handler__, int const* ia__, double complex* dm__, int const* ld__,
                          int* error_code__);

/*
sirius_set_local_occupation_matrix:
  doc: Set local occupation matrix of LDA+U+V method.
  arguments:
    handler:
      type: gs_handler
      attr: in, required
      doc: Ground-state handler.
    ia:
      type: int
      attr: in, required
      doc: Index of atom.
    n:
      type: int
      attr: in, required
      doc: Principal quantum number.
    l:
      type: int
      attr: in, required
      doc: Orbital quantum number.
    spin:
      type: int
      attr: in, required
      doc: Spin index.
    occ_mtrx:
      type: complex
      attr: in, required, dimension(ld, ld)
      doc: Local occupation matrix.
    ld:
      type: int
      attr: in, required
      doc: Leading dimension of the occupation matrix.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
*/
void
sirius_set_local_occupation_matrix(void** handler__, int const* ia__, int const* n__, int const* l__, int const* spin__,
                                   double complex* occ_mtrx__, int const* ld__, int* error_code__);

/*
sirius_set_nonlocal_occupation_matrix:
  doc: Set nonlocal part of LDA+U+V occupation matrix.
  arguments:
    handler:
      type: gs_handler
      attr: in, required
      doc: Ground-state handler.
    atom_pair:
      type: int
      attr: in, required, dimension(2)
      doc: Index of two atoms in the non-local V correction.
    n:
      type: int
      attr: in, required, dimension(2)
      doc: Pair of principal quantum numbers.
    l:
      type: int
      attr: in, required, dimension(2)
      doc: Pair of orbital quantum numbers.
    spin:
      type: int
      attr: in, required
      doc: Spin index.
    T:
      type: int
      attr: in, required, dimension(3)
      doc: Translation vector that connects two atoms.
    occ_mtrx:
      type: complex
      attr: in, required, dimension(ld1, ld2)
      doc: Nonlocal occupation matrix.
    ld1:
      type: int
      attr: in, required
      doc: Leading dimension of the occupation matrix.
    ld2:
      type: int
      attr: in, required
      doc: Second dimension of the occupation matrix.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
*/
void
sirius_set_nonlocal_occupation_matrix(void** handler__, int const* atom_pair__, int const* n__, int const* l__,
                                      int const* spin__, int const* T__, double complex* occ_mtrx__,
                                      int const* ld1__, int const* ld2__, int* error_code__);

/*
sirius_get_major_version:
 doc: major version.
 arguments:
   version:
     type: int
     attr: out, required
     doc: version
*/
void
sirius_get_major_version(int* version);

/*
sirius_get_minor_version:
  doc: minor version.
  arguments:
    version:
      type: int
      attr: out, required
      doc: version
*/
void
sirius_get_minor_version(int* version);

/*
sirius_get_revision:
  doc: minor version.
  arguments:
    version:
      type: int
      attr: out, required
      doc: version
*/
void
sirius_get_revision(int* version);

/*
sirius_is_initialized:
   doc: Checks if the library is initialized.
   arguments:
     status:
      type: bool
      attr: out, required
      doc: Status of the library (true if initialized).
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
*/
void
sirius_is_initialized(bool* status__, int* error_code__);

/*
sirius_create_context_from_json:
  doc: Create context of the simulation, from a JSON file or string.
  arguments:
    fcomm:
      type: int
      attr: in, required, value
      doc: Entire communicator of the simulation.
    handler:
      type: ctx_handler
      attr: out, required
      doc: New empty simulation context.
    fname:
      type: string
      attr: in, required
      doc: file name or JSON string.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
*/
void
sirius_create_context_from_json(int fcomm__, void** handler__, char const* fname__, int* error_code__);

/*
sirius_get_num_atoms:
  doc: Get the number of atoms in the simulation
  arguments:
    gs_handler:
      type: gs_handler
      attr: in, required
      doc: Ground-state handler.
    num_atoms:
      type: int
      attr: out, required
      doc: Number of atoms.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
*/
void
sirius_get_num_atoms(void* const* gs_handler__, int* num_atoms__, int* error_code__);

/*
sirius_get_kp_params_from_ctx:
  doc: Get the k-point parameters from the simulation context.
  arguments:
    handler:
      type: ctx_handler
      attr: in, required
      doc: Simulation context handler.
    k_grid:
      type: int
      attr: out, required, dimension(3)
      doc: dimensions of the k points grid.
    k_shift:
      type: int
      attr: out, required, dimension(3)
      doc: k point shifts.
    use_symmetry:
      type: bool
      attr: out, required
      doc: If .true. k-set will be generated using symmetries.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
*/
void
sirius_get_kp_params_from_ctx(void* const* handler__, int* k_grid__, int* k_shift__, bool* use_symmetry__,
                              int* error_code__);

/*
sirius_get_scf_params_from_ctx:
  doc: Get the SCF parameters from the simulation context.
  arguments:
    handler:
      type: ctx_handler
      attr: in, required
      doc: Simulation context handler.
    density_tol__:
      type: double
      attr: out, required
      doc: Tolerance on RMS in density.
    energy_tol__:
      type: double
      attr: out, require
      doc: Tolerance in total energy difference.
    iter_solver_tol:
      type: double
      attr: out, required
      doc: Initial tolerance of the iterative solver.
    max_niter:
      type: int
      attr: out, required
      doc: Maximum number of SCF iterations.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
*/
void
sirius_get_scf_params_from_ctx(void* const* handler__, double* density_tol__, double* energy_tol__,
                               double* iter_solver_tol__, int* max_niter, int* error_code__);

/*
sirius_create_hamiltonian:
  doc: Create an Hamiltonian based on the density stored in the gs_handler.
  arguments:
    gs_handler:
      type: gs_handler
      attr: in, required
      doc: Ground-state context handler.
    H0_handler:
      type: H0_handler
      attr: out, required
      doc: The new handler for the Hamiltonian
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
*/
void
sirius_create_hamiltonian(void* const* gs_handler__, void** H0_handler__, int* error_code__);

/*
sirius_diagonalize_hamiltonian:
  doc: Diagonalizes the Hamiltonian.
  arguments:
    handler:
      type: ctx_handler
      attr: in, required
      doc: Simulation context handler.
    gs_handler:
      type: gs_handler
      attr: in, required
      doc: Ground-state context handler.
    H0_handler:
      type: H0_handler
      attr: in, required
      doc: Hamiltonian contexct handler.
    iter_solver_tol:
      type: double
      attr: in, required
      doc: Tolerance for the iterative solver.
    max_steps:
      type: int
      attr: in, required
      doc: Maximum number of steps for the iterative solver.
    converge_by_energy:
      type: int
      attr: in, optional
      doc: Whether the solver should determine convergence by checking the energy different (1), or the L2 norm of the residual (0). Default is value is 1.
    exact_diagonalization:
      type: bool
      attr: in, optional
      doc: Whether an exact diagonalization should take place (rather than iterative Davidson)
    converged:
      type: bool
      attr: out, required
      doc: Whether the iterative solver converged
    niter:
      type: int
      attr: out, required
      doc: Number of steps for the solver to converge
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
*/
void 
sirius_diagonalize_hamiltonian(void* const* handler__, void* const* gs_handler__,
                               void* const* H0_handler__, double* const iter_solver_tol__,
                               int* const max_steps__, int* converge_by_energy__,
                               bool* const exact_diagonalization__,
                               bool* converged__, int* niter__, int* error_code__);

/*
sirius_find_band_occupancies:
  doc: Internally calculate the band occupancies.
  arguments:
    ks_handler:
      type: ks_handler
      attr: in, required
      doc: Handler for the k-point set.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
*/
void
sirius_find_band_occupancies(void* const* ks_handler__, int* error_code__);

/*
sirius_set_num_bands:
  doc: Sets the number of bands in the simulation context.
  arguments:
    handler:
      type: ctx_handler
      attr: in, required
      doc: Simulation context handler.
    num_bands:
      type: int
      attr: in, required
      doc: Number of bands to set.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
*/
void
sirius_set_num_bands(void* const* handler__, int* const num_bands__, int* error_code__);

/*
sirius_fft_transform:
  doc: Triggers an internal FFT transform of the given field, in the given direction
  arguments:
    gs_handler:
      type: gs_handler
      attr: in, required
      doc: Ground-state handler.
    label:
      type: string
      attr: in, required
      doc: Which field to FFT transform.
    direction:
      type: int
      attr: in, required
      doc: FFT transform direction (1 forward, -1, backward)
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
*/
void
sirius_fft_transform(void* const* gs_handler__, char const* label__, int* direction__, int* error_code__);

/*
sirius_get_psi:
  doc: Gets the wave function for a given k-point and spin (all local, no MPI communication).
  arguments:
    ks_handler:
      type: ks_handler
      attr: in, required
      doc: Handler for the k-point set.
    ik:
      type: int
      attr: in, required
      doc: Index of the k-point.
    ispin:
      type: int
      attr: in, required
      doc: Index of the spin.
    psi:
      type: complex
      attr: in, required, dimension(:)
      doc: Pointer to the wave function coefficients.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
*/
void
sirius_get_psi(void* const* ks_handler__, int* ik__, int* ispin__, double complex* psi__, 
               int* error_code__);

/*
sirius_get_gkvec:
  doc: Gets the G+k integer coordinates for a given k-point.
  arguments:
    ks_handler:
      type: ks_handler
      attr: in, required
      doc: Handler for the k-point set.
    ik:
      type: int
      attr: in, required
      doc: Index of the k-point.
    gvec:
      type: double
      attr: in, required, dimension(:)
      doc: Pointer to the G+k vector coordinates.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
*/
void
sirius_get_gkvec(void* const* ks_handler__, int* ik__, double* gvec__, int* error_code__);

/*
sirius_set_energy_fermi:
  doc: Sets the SIRIUS Fermi energy.
  arguments:
    ks_handler:
      type: ks_handler
      attr: in, required
      doc: Handler for the k-point set.
    energy_fermi:
      type: double
      attr: in, required
      doc: Fermi energy to be set.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
*/
void
sirius_set_energy_fermi(void* const* ks_handler__, double* energy_fermi__, int* error_code__);

/*
sirius_set_atom_vector_field:
  doc: Set new atomic vector field (aka initial magnetization).
  arguments:
    handler:
      type: ctx_handler
      attr: in, required
      doc: Simulation context handler.
    ia:
      type: int
      attr: in, required
      doc: Index of atom; index starts form 1
    vector_field:
      type: double
      attr: in, required, dimension(3)
      doc: Atom vector field.
    error_code:
      type: int
      attr: out, optional
      doc: Error code.
*/
void
sirius_set_atom_vector_field(void* const* handler__, int const* ia__, double const* vector_field__, int* error_code__);

