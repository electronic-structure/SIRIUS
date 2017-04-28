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

/// Simulation context.
std::unique_ptr<sirius::Simulation_context> sim_ctx{nullptr};

/// Pointer to Density class, implicitly used by Fortran side.
sirius::Density* density = nullptr;

/// Pointer to Potential class, implicitly used by Fortran side.
sirius::Potential* potential = nullptr;

/// List of pointers to the sets of k-points.
std::vector<sirius::K_point_set*> kset_list;

/// DFT ground state wrapper
sirius::DFT_ground_state* dft_ground_state = nullptr;

/// List of timers created on the Fortran side.
std::map<std::string, sddk::timer*> ftimers;

std::unique_ptr<sirius::Stress> stress_tensor{nullptr};
std::unique_ptr<sirius::Forces_PS> forces{nullptr};

extern "C" {

/// Initialize the library.
/** \param [in] call_mpi_init .true. if the library needs to call MPI_Init()
 *
 *  Example:
    \code{.F90}
    integer ierr
    call mpi_init(ierr)
    ! initialize low-level stuff and don't call MPI_Init() from SIRIUS
    call sirius_initialize(0)
    \endcode
 */
void sirius_initialize(ftn_int* call_mpi_init__)
{
    bool call_mpi_init = (*call_mpi_init__ != 0) ? true : false;
    sirius::initialize(call_mpi_init);
}

/// Clear global variables and destroy all objects
void sirius_clear(void)
{
    if (density != nullptr) {
        delete density;
        density = nullptr;
    }
    if (potential != nullptr) {
        delete potential;
        potential = nullptr;
    }
    if (dft_ground_state != nullptr) {
        delete dft_ground_state;
        dft_ground_state = nullptr;
    }
    for (size_t i = 0; i < kset_list.size(); i++) {
        if (kset_list[i] != nullptr) {
            delete kset_list[i];
            kset_list[i] = nullptr;
        }
    }
    sim_ctx = nullptr;

    kset_list.clear();
}

void sirius_create_simulation_context(const char* config_file_name__)
{
    std::string config_file_name(config_file_name__);
    if (config_file_name.length() == 0) {
        sim_ctx = std::unique_ptr<sirius::Simulation_context>(new sirius::Simulation_context(mpi_comm_world()));
    } else {
        sim_ctx = std::unique_ptr<sirius::Simulation_context>(new sirius::Simulation_context(config_file_name, mpi_comm_world()));
    }
}

/// Initialize the global variables.
/** The function must be called after setting up the lattice vectors, plane wave-cutoff, autormt flag and loading
 *  atom types and atoms into the unit cell.
 */
void sirius_initialize_simulation_context()
{
    sim_ctx->initialize();
}

void sirius_delete_simulation_context()
{
    sim_ctx = nullptr;
}

/// Initialize the Potential object.
/** \param [in] veffmt pointer to the muffin-tin part of the effective potential
 *  \param [in] veffit pointer to the interstitial part of the effective potential
 *  \param [in] beffmt pointer to the muffin-tin part of effective magnetic field
 *  \param [in] beffit pointer to the interstitial part of the effective magnetic field
 */
void sirius_create_potential(double* veffit__,
                             double* veffmt__,
                             double* beffit__,
                             double* beffmt__)
{
    potential = new sirius::Potential(*sim_ctx);
    potential->set_effective_potential_ptr(veffmt__, veffit__);
    potential->set_effective_magnetic_field_ptr(beffmt__, beffit__);
}

void sirius_delete_potential()
{
    if (potential != nullptr) {
        delete potential;
    }
    potential = nullptr;
}

/// Initialize the Density object.
/** \param [in] rhomt pointer to the muffin-tin part of the density
 *  \param [in] rhoit pointer to the interstitial part of the denssity
 *  \param [in] magmt pointer to the muffin-tin part of the magnetization
 *  \param [in] magit pointer to the interstitial part of the magnetization
 */
void sirius_create_density(double* rhoit__,
                           double* rhomt__,
                           double* magit__,
                           double* magmt__)
{
    density = new sirius::Density(*sim_ctx);
    density->set_charge_density_ptr(rhomt__, rhoit__);
    density->set_magnetization_ptr(magmt__, magit__);
}

void sirius_delete_density()
{
    if (density != nullptr) {
        delete density;
    }
    density = nullptr;
}

/// Create the k-point set from the list of k-points and return it's id
void sirius_create_kset(ftn_int*    num_kpoints__,
                        ftn_double* kpoints__,
                        ftn_double* kpoint_weights__,
                        ftn_int*    init_kset__,
                        ftn_int*    kset_id__,
                        ftn_int*    nk_loc__)
{
    mdarray<double, 2> kpoints(kpoints__, 3, *num_kpoints__);

    sirius::K_point_set* new_kset = new sirius::K_point_set(*sim_ctx);
    new_kset->add_kpoints(kpoints, kpoint_weights__);
    if (*init_kset__) {
        std::vector<int> counts;
        if (nk_loc__ != NULL) {
            for (int i = 0; i < new_kset->comm().size(); i++) {
                counts.push_back(nk_loc__[i]);
            }
        }
        new_kset->initialize(counts);
    }

    kset_list.push_back(new_kset);
    *kset_id__ = (int)kset_list.size() - 1;
}

void sirius_create_irreducible_kset_(int32_t* mesh__, int32_t* is_shift__, int32_t* use_sym__, int32_t* kset_id__)
{
    for (int x = 0; x < 3; x++) {
        if (!(is_shift__[x] == 0 || is_shift__[x] == 1)) {
            std::stringstream s;
            s << "wrong k-shift " << is_shift__[0] << " " << is_shift__[1] << " " << is_shift__[2];
            TERMINATE(s);
        }
    }

    sirius::K_point_set* new_kset = new sirius::K_point_set(*sim_ctx,
                                                            vector3d<int>(mesh__[0], mesh__[1], mesh__[2]),
                                                            vector3d<int>(is_shift__[0], is_shift__[1], is_shift__[2]),
                                                            *use_sym__);

    new_kset->initialize();

    kset_list.push_back(new_kset);
    *kset_id__ = (int)kset_list.size() - 1;
}

void sirius_delete_kset(int32_t* kset_id__)
{
    if (kset_list[*kset_id__] != nullptr) {
        delete kset_list[*kset_id__];
    }
    kset_list[*kset_id__] = nullptr;
}

void sirius_create_ground_state(int32_t* kset_id__)
{
    if (dft_ground_state != nullptr) {
        TERMINATE("dft_ground_state object is already allocate");
    }

    dft_ground_state = new sirius::DFT_ground_state(*sim_ctx, *potential, *density, *kset_list[*kset_id__], 1);
}

void sirius_delete_ground_state()
{
    delete dft_ground_state;
    dft_ground_state = nullptr;
}

void sirius_set_lmax_apw(int32_t* lmax_apw__)
{
    sim_ctx->set_lmax_apw(*lmax_apw__);
}

void sirius_set_lmax_pot(int32_t* lmax_pot__)
{
    sim_ctx->set_lmax_pot(*lmax_pot__);
}

void sirius_set_lmax_rho(int32_t* lmax_rho__)
{
    sim_ctx->set_lmax_rho(*lmax_rho__);
}

void sirius_set_num_mag_dims(int32_t* num_mag_dims__)
{
    sim_ctx->set_num_mag_dims(*num_mag_dims__);
}

/// Set lattice vectors.
/** \param [in] a1 1st lattice vector
 *  \param [in] a2 2nd lattice vector
 *  \param [in] a3 3rd lattice vector
 *
 *  Example:
    \code{.F90}
    real(8) a1(3),a2(3),a3(3)
    a1(:) = (/5.d0, 0.d0, 0.d0/)
    a2(:) = (/0.d0, 5.d0, 0.d0/)
    a3(:) = (/0.d0, 0.d0, 5.d0/)
    call sirius_set_lattice_vectors(a1, a2, a3)
    \endcode
 */
void sirius_set_lattice_vectors(double* a1__,
                                double* a2__,
                                double* a3__)
{
    sim_ctx->unit_cell().set_lattice_vectors(vector3d<double>(a1__[0], a1__[1], a1__[2]),
                                             vector3d<double>(a2__[0], a2__[1], a2__[2]),
                                             vector3d<double>(a3__[0], a3__[1], a3__[2]));
}

/// Set plane-wave cutoff for FFT grid.
/** \param [in] gmaxvr maximum G-vector length

    Example:
    \code{.F90}
        real(8) gmaxvr
        gmaxvr = 20.0
        call sirius_set_pw_cutoff(gmaxvr)
    \endcode
*/
void sirius_set_pw_cutoff(double* pw_cutoff__)
{
    sim_ctx->set_pw_cutoff(*pw_cutoff__);
}

void sirius_set_gk_cutoff(double* gk_cutoff__)
{
    sim_ctx->set_gk_cutoff(*gk_cutoff__);
}

/// Turn on or off the automatic scaling of muffin-tin spheres.
/** \param [in] auto_rmt .true. if muffin-tin spheres must be resized to the maximally allowed radii

    The auto_rmt flag tells the library how to proceed with the muffin-tin spheres in the case when
    library takes the full control over the muffin-tin geometry.

    Example:
    \code{.F90}
        logical autormt
        autormt = .true.
        call sirius_set_auto_rmt(autormt)
    \endcode
*/
void sirius_set_auto_rmt(int32_t* auto_rmt__)
{
    sim_ctx->set_auto_rmt(*auto_rmt__);
}

/// Add atom type to the unit cell.
/** \param [in] label unique label of atom type
 *  \param [in] fname name of .json atomic file
 *
 *  Atom type (species in the terminology of Exciting/Elk) is a class which holds information
 *  common to the atoms of the same element: charge, number of core and valence electrons, muffin-tin
 *  radius, radial grid etc. See sirius::Atom_type class for details.
 *
 *  Example:
    \code{.F90}
    do is = 1, nspecies
      !====================================================
      ! add atom type with label equal to the file name and
      ! read the .json file
      !====================================================
      call sirius_add_atom_type(trim(spfname(is)), trim(spfname(is))
    enddo
    \endcode
 */
void sirius_add_atom_type(ftn_char label__,
                          ftn_char fname__)
{
    std::string fname = (fname__ == NULL) ? std::string("") : std::string(fname__);
    sim_ctx->unit_cell().add_atom_type(std::string(label__), fname);
}

/// Set basic properties of the atom type.
/** \param [in] label unique label of the atom type
 *  \param [in] symbol symbol of the element
 *  \param [in] zn positive integer charge
 *  \param [in] mass atomic mass
 *  \param [in] mt_radius muffin-tin radius
 *  \param [in] num_mt_points number of muffin-tin points
 *  \param [in] radial_grid_origin origin of radial grid
 *  \param [in] radial_grid_infinity effective infinity
 *
 *  Example:
    \code{.F90}
    do is=1,nspecies
      call sirius_set_atom_type_properties(trim(spfname(is)), trim(spsymb(is)),  &
                                          &nint(-spzn(is)), spmass(is), rmt(is), &
                                          &nrmt(is), sprmin(is), sprmax(is))
    enddo
    \endcode
 */
void sirius_set_atom_type_properties(char const* label__,
                                     char const* symbol__,
                                     int32_t* zn__,
                                     double* mass__,
                                     double* mt_radius__,
                                     int32_t* num_mt_points__)
{
    auto& type = sim_ctx->unit_cell().atom_type(std::string(label__));
    type.set_symbol(std::string(symbol__));
    type.set_zn(*zn__);
    type.set_mass(*mass__);
    type.set_num_mt_points(*num_mt_points__);
    type.set_mt_radius(*mt_radius__);
}

/// Set the radial grid of atom type.
/** \param [in] label unique label of the atom type
 *  \param [in] num_radial_points number of radial points
 *  \param [in] radial_points radial points
 *
 *  Example:
    \code{.F90}
    do is=1,nspecies
      call sirius_set_atom_type_radial_grid(trim(spfname(is)), spnr(is), spr(1, is))
    enddo
    \endcode
 */
void sirius_set_atom_type_radial_grid(char const* label__,
                                      int32_t const* num_radial_points__,
                                      double const* radial_points__)
{
    auto& type = sim_ctx->unit_cell().atom_type(std::string(label__));
    type.set_radial_grid(type.num_mt_points(), radial_points__);
    type.set_free_atom_radial_grid(*num_radial_points__, radial_points__);
}

void sirius_set_free_atom_density(char const* label__,
                                  int32_t const* num_points__,
                                  double const* dens__)
{
    auto& type = sim_ctx->unit_cell().atom_type(std::string(label__));
    type.set_free_atom_density(*num_points__, dens__);
}

/// Set the atomic level configuration of the atom type.
/** With each call to the function new atomic level is added to the list of atomic levels of the atom type.
 *
 *  \param [in] label unique label of the atom type
 *  \param [in] n principal quantum number of the atomic level
 *  \param [in] l angular quantum number of the atomic level
 *  \param [in] k kappa quantum number of the atomic level
 *  \param [in] occupancy occupancy of the atomic level
 *  \param [in] core .true. if the atomic level belongs to the core
 *
 *  Example
    \code{.F90}
    do is=1,nspecies
      do ist=1,spnst(is)
        call sirius_set_atom_type_configuration(trim(spfname(is)), spn(ist, is), spl(ist, is),&
                                               &spk(ist, is), spocc(ist, is),&
                                               &spcore(ist, is))
      enddo
    enddo
    \endcode
 */
void sirius_set_atom_type_configuration(char* label__,
                                        int32_t* n__,
                                        int32_t* l__,
                                        int32_t* k__,
                                        double* occupancy__,
                                        int32_t* core__)
{
    auto& type = sim_ctx->unit_cell().atom_type(std::string(label__));
    bool core = *core__;
    type.set_configuration(*n__, *l__, *k__, *occupancy__, core);
}

/// Add atom to the unit cell.
/** \param [in] label unique label of the atom type
 *  \param [in] position atom position in fractional coordinates
 *  \param [in] vector_field vector field associated with the given atom
 *
 *  Example:
    \code{.F90}
    do is = 1, nspecies
      do ia = 1, natoms(is)
        call sirius_add_atom(trim(spfname(is)), atposl(:, ia, is), bfcmt(:, ia, is))
      enddo
    enddo
    \endcode
 */
void sirius_add_atom(char* label__,
                     double* position__,
                     double* vector_field__)
{
    if (vector_field__ != NULL) {
        sim_ctx->unit_cell().add_atom(std::string(label__),
                                      vector3d<double>(position__[0], position__[1], position__[2]),
                                      vector3d<double>(vector_field__[0], vector_field__[1], vector_field__[2]));
    } else {
        sim_ctx->unit_cell().add_atom(std::string(label__),
                                      vector3d<double>(position__[0], position__[1], position__[2]));
    }
}

/// Set the table of equivalent atoms.
/** \param [in] equivalent_atoms table of equivalent atoms

    Equivalent atoms are symmetry related and belong to the same atom symmetry class. If equivalence table is not
    provided by user, \a spglib is called. In case of magnetic symmetry \a spglib is of no use and euivalence table
    must be provided.
*/
void sirius_set_equivalent_atoms(int32_t* equivalent_atoms__)
{
    sim_ctx->unit_cell().set_equivalent_atoms(equivalent_atoms__);
}

/// Set augmented-wave cutoff
/** \param [in] aw_cutoff augmented-wave cutoff

     Augmented wave cutoff is used to setup the |G+k| cutoff which controls the size of the (L)APW basis.
     The following simple relation is used:
     \f[
       |\mathbf{G}+\mathbf{k}| R^{MT}_{min} \leq \textrm{AW cutoff}
     \f]

     Example:
     \code{.F90}
         real(8) rgkmax
         rgkmax = 10.0
         call sirius_set_aw_cutoff(rgkmax)
     \endcode
*/
void sirius_set_aw_cutoff(double* aw_cutoff__)
{
    sim_ctx->set_aw_cutoff(*aw_cutoff__);
}

void sirius_add_xc_functional(char const* name__)
{
    assert(name__ != NULL);
    sim_ctx->add_xc_functional(name__);
}

void sirius_set_esm_type(char const* name__)
{
    assert(name__ != NULL);
    sim_ctx->set_esm_type(name__);
}

void sirius_set_gamma_point(ftn_int* gamma_point__)
{
    sim_ctx->set_gamma_point(*gamma_point__);
}

void sirius_set_valence_relativity(ftn_char str__)
{
    sim_ctx->set_valence_relativity(str__);
}

void sirius_set_core_relativity(ftn_char str__)
{
    sim_ctx->set_core_relativity(str__);
}

/// Get maximum number of muffin-tin radial points.
/** \param [out] max_num_mt_points maximum number of muffin-tin points */
void sirius_get_max_num_mt_points(ftn_int* max_num_mt_points__)
{
    *max_num_mt_points__ = sim_ctx->unit_cell().max_num_mt_points();
}

/// Get number of muffin-tin radial points for a specific atom type.
/** \param [in] label unique label of atom type
 *  \param [out] num_mt_points number of muffin-tin points
 */
void sirius_get_num_mt_points(ftn_char label__,
                              ftn_int* num_mt_points__)
{
    *num_mt_points__ = sim_ctx->unit_cell().atom_type(std::string(label__)).num_mt_points();
}

void sirius_get_mt_points(ftn_char label__,
                          ftn_double* mt_points__)
{
    auto& type = sim_ctx->unit_cell().atom_type(std::string(label__));
    for (int i = 0; i < type.num_mt_points(); i++) mt_points__[i] = type.radial_grid(i);
}

void sirius_get_num_fft_grid_points(int32_t* num_grid_points__)
{
    *num_grid_points__ = sim_ctx->fft().local_size();
}

void sirius_get_num_bands(int32_t* num_bands)
{
    *num_bands = sim_ctx->num_bands();
}

/// Get number of G-vectors within the plane-wave cutoff
void sirius_get_num_gvec(int32_t* num_gvec__)
{
    *num_gvec__ = sim_ctx->gvec().num_gvec();
}

void sirius_find_fft_grid_size(ftn_double* cutoff__,
                               ftn_double* lat_vec__,
                               ftn_int*    grid_size__)
{
    FFT3D_grid grid(find_translations(*cutoff__, {{lat_vec__[0], lat_vec__[3], lat_vec__[6]},
                                                  {lat_vec__[1], lat_vec__[4], lat_vec__[7]},
                                                  {lat_vec__[2], lat_vec__[5], lat_vec__[8]}}));
    for (int x: {0, 1, 2}) {
        grid_size__[x] = grid.size(x);
    }
}

/// Get sizes of FFT grid
void sirius_get_fft_grid_size(ftn_int* grid_size__)
{
    for (int x: {0, 1, 2}) {
        grid_size__[x] = sim_ctx->fft().grid().size(x);
    }
}

/// Get lower and upper limits of the FFT grid dimension
/** \param [in] d index of dimension (1,2, or 3)
 *  \param [out] lower lower (most negative) value
 *  \param [out] upper upper (most positive) value
 *
 *  Example:
    \code{.F90}
    do i=1,3
      call sirius_get_fft_grid_limits(i,intgv(i,1),intgv(i,2))
    enddo
    \endcode
 */
void sirius_get_fft_grid_limits(int32_t const* d, int32_t* lower, int32_t* upper)
{
    assert((*d >= 1) && (*d <= 3));
    *lower = sim_ctx->fft().grid().limits(*d - 1).first;
    *upper = sim_ctx->fft().grid().limits(*d - 1).second;
}

/// Get mapping between G-vector index and FFT index
void sirius_get_fft_index(int32_t* fft_index__)
{
    for (int ig = 0; ig < sim_ctx->gvec().num_gvec(); ig++) {
        auto G = sim_ctx->gvec().gvec(ig);
        fft_index__[ig] = sim_ctx->fft().grid().index_by_gvec(G[0], G[1], G[2]) + 1;
    }
}

/// Get list of G-vectors in fractional corrdinates
void sirius_get_gvec(int32_t* gvec__)
{
    mdarray<int, 2> gvec(gvec__, 3, sim_ctx->gvec().num_gvec());
    for (int ig = 0; ig < sim_ctx->gvec().num_gvec(); ig++) {
        auto gv = sim_ctx->gvec().gvec(ig);
        for (int x: {0, 1, 2}) {
            gvec(x, ig) = gv[x];
        }
    }
}

/// Get list of G-vectors in Cartesian coordinates
void sirius_get_gvec_cart(double* gvec_cart__)
{
    mdarray<double, 2> gvec_cart(gvec_cart__, 3, sim_ctx->gvec().num_gvec());
    for (int ig = 0; ig < sim_ctx->gvec().num_gvec(); ig++) {
        auto gvc = sim_ctx->gvec().gvec_cart(ig);
        for (int x: {0, 1, 2}) {
            gvec_cart(x, ig) = gvc[x];
        }
    }
}

/// Get lengh of G-vectors
void sirius_get_gvec_len(double* gvec_len__)
{
    for (int ig = 0; ig < sim_ctx->gvec().num_gvec(); ig++) {
        gvec_len__[ig] = sim_ctx->gvec().gvec_len(ig);
    }
}

void sirius_get_index_by_gvec(int32_t* index_by_gvec__)
{
    auto d0 = sim_ctx->fft().grid().limits(0);
    auto d1 = sim_ctx->fft().grid().limits(1);
    auto d2 = sim_ctx->fft().grid().limits(2);

    mdarray<int, 3> index_by_gvec(index_by_gvec__, d0, d1, d2);
    std::fill(index_by_gvec.at<CPU>(), index_by_gvec.at<CPU>() + index_by_gvec.size(), -1);

    for (int ig = 0; ig < sim_ctx->gvec().num_gvec(); ig++) {
        auto G = sim_ctx->gvec().gvec(ig);
        index_by_gvec(G[0], G[1], G[2]) = ig + 1;
    }
}

/// Get Ylm spherical harmonics of G-vectors.
void sirius_get_gvec_ylm(double_complex* gvec_ylm__, int* ld__, int* lmax__)
{
    TERMINATE("fix this");

    //==mdarray<double_complex, 2> gvec_ylm(gvec_ylm__, *ld__, sim_ctx->reciprocal_lattice()->num_gvec());
    //==// TODO: can be parallelized
    //==for (int ig = 0; ig < sim_ctx->reciprocal_lattice()->num_gvec(); ig++)
    //=={
    //==    sim_ctx->reciprocal_lattice()->gvec_ylm_array<global>(ig, &gvec_ylm(0, ig), *lmax__);
    //==}
}

void sirius_get_gvec_phase_factors(double_complex* sfacg__)
{
    TERMINATE("fix this");
    //mdarray<double_complex, 2> sfacg(sfacg__, sim_ctx->fft().num_gvec(), sim_ctx->unit_cell().num_atoms());
    //for (int ia = 0; ia < sim_ctx->unit_cell().num_atoms(); ia++)
    //{
    //    for (int ig = 0; ig < sim_ctx->fft().num_gvec(); ig++)
    //        sfacg(ig, ia) = sim_ctx->reciprocal_lattice()->gvec_phase_factor(ig, ia);
    //}
}

void sirius_get_step_function(double_complex* cfunig__, double* cfunir__)
{
    for (int i = 0; i < sim_ctx->fft().local_size(); i++) {
        cfunir__[i] = sim_ctx->step_function().theta_r(i);
    }
    for (int ig = 0; ig < sim_ctx->gvec().num_gvec(); ig++) {
        cfunig__[ig] = sim_ctx->step_function().theta_pw(ig);
    }
}

/// Get the total number of electrons
void sirius_get_num_electrons(double* num_electrons__)
{
    *num_electrons__ = sim_ctx->unit_cell().num_electrons();
}

/// Get the number of valence electrons
void sirius_get_num_valence_electrons(double* num_valence_electrons__)
{
    *num_valence_electrons__ = sim_ctx->unit_cell().num_valence_electrons();
}

/// Get the number of core electrons
void sirius_get_num_core_electrons(double* num_core_electrons__)
{
    *num_core_electrons__ = sim_ctx->unit_cell().num_core_electrons();
}

void sirius_generate_initial_density()
{
    density->initial_density();
}

void sirius_generate_effective_potential()
{
    potential->generate(*density);
}

void sirius_initialize_subspace(ftn_int* kset_id__)
{
    dft_ground_state->band().initialize_subspace(*kset_list[*kset_id__], *potential);
}

void sirius_generate_density(int32_t* kset_id__)
{
    density->generate(*kset_list[*kset_id__]);
}

void sirius_generate_valence_density(int32_t* kset_id__)
{
    density->generate_valence(*kset_list[*kset_id__]);
    /* only PW coeffs have been generated; transfrom them to real space */
    density->fft_transform(1);
}

void sirius_augment_density(int32_t* kset_id__)
{
    density->augment(*kset_list[*kset_id__]);
}

/// Find eigen-states of the k-point set.
/** \param [in] kset_id k-point set id
    \param [in] precompute .true. if the radial integrals and plane-wave coefficients of the interstitial
                potential must be precomputed

    Example:
    \code{.F90}
        ! precompute the necessary data on the Fortran side
          call sirius_update_atomic_potential
          call sirius_generate_radial_functions
          call sirius_generate_radial_integrals
          call sirius_generate_potential_pw_coefs
          call sirius_find_eigen_states(kset_id, 0)
          .
          .
        ! or
          .
          .
        ! ask the library to precompute the necessary data
          call sirius_find_eigen_states(kset_id, 1)
    \endcode
*/
void sirius_find_eigen_states(int32_t* kset_id__,
                              int32_t* precompute__)
{
    bool precompute = (*precompute__) ? true : false;
    dft_ground_state->band().solve_for_kset(*kset_list[*kset_id__], *potential, precompute);
}

void sirius_find_band_occupancies(int32_t* kset_id__)
{
    kset_list[*kset_id__]->find_band_occupancies();
}

void sirius_get_energy_fermi(int32_t* kset_id__, double* efermi__)
{
    *efermi__ = kset_list[*kset_id__]->energy_fermi();
}

void sirius_set_band_occupancies(int32_t* kset_id__,
                                 int32_t* ik__,
                                 double* band_occupancies__)
{
    int ik = *ik__ - 1;
    kset_list[*kset_id__]->set_band_occupancies(ik, band_occupancies__);
}

void sirius_get_band_energies(int32_t* kset_id__,
                              int32_t* ik__,
                              double* band_energies__)
{
    int ik = *ik__ - 1;
    kset_list[*kset_id__]->get_band_energies(ik, band_energies__);
}

void sirius_get_band_occupancies(int32_t* kset_id, int32_t* ik_, double* band_occupancies)
{
    int ik = *ik_ - 1;
    kset_list[*kset_id]->get_band_occupancies(ik, band_occupancies);
}

void sirius_print_timers(void)
{
    sddk::timer::print(0);
}

void sirius_start_timer(char const* name__)
{
    std::string name(name__);
    ftimers[name] = new sddk::timer(name);
}

void sirius_stop_timer(char const* name__)
{
    std::string name(name__);
    if (ftimers.count(name)) {
        delete ftimers[name];
    }
}

void sirius_save_potential(void)
{
    potential->save();
}

void sirius_save_density(void)
{
    density->save();
}

void sirius_load_potential(void)
{
    potential->load();
}

//== void FORTRAN(sirius_save_wave_functions)(int32_t* kset_id)
//== {
//==     kset_list[*kset_id]->save_wave_functions();
//== }
//==
//== void FORTRAN(sirius_load_wave_functions)(int32_t* kset_id)
//== {
//==     kset_list[*kset_id]->load_wave_functions();
//== }

void sirius_save_kset(int32_t* kset_id)
{
    kset_list[*kset_id]->save();
}

void sirius_load_kset(int32_t* kset_id)
{
    kset_list[*kset_id]->load();
}

/*  Relevant block in the input file:

    "bz_path" : {
        "num_steps" : 100,
        "points" : [["G", [0, 0, 0]], ["X", [0.5, 0.0, 0.5]], ["L", [0.5, 0.5, 0.5]]]
    }
*/
//== void FORTRAN(sirius_bands)(void)
//== {
//==     FORTRAN(sirius_read_state)();
//==
//==     std::vector<std::pair<std::string, std::vector<double> > > bz_path;
//==     std::string fname("sirius.json");
//==
//==     int num_steps = 0;
//==     if (Utils::file_exists(fname))
//==     {
//==         JSON_tree parser(fname);
//==         if (!parser["bz_path"].empty())
//==         {
//==             parser["bz_path"]["num_steps"] >> num_steps;
//==
//==             for (int ipt = 0; ipt < parser["bz_path"]["points"].size(); ipt++)
//==             {
//==                 std::pair<std::string, std::vector<double> > pt;
//==                 parser["bz_path"]["points"][ipt][0] >> pt.first;
//==                 parser["bz_path"]["points"][ipt][1] >> pt.second;
//==                 bz_path.push_back(pt);
//==             }
//==         }
//==     }
//==
//==     if (bz_path.size() < 2) TERMINATE("at least two BZ points are required");
//==
//==     // compute length of segments
//==     std::vector<double> segment_length;
//==     double total_path_length = 0.0;
//==     for (int ip = 0; ip < (int)bz_path.size() - 1; ip++)
//==     {
//==         double vf[3];
//==         for (int x = 0; x < 3; x++) vf[x] = bz_path[ip + 1].second[x] - bz_path[ip].second[x];
//==         double vc[3];
//==         sim_ctx->get_coordinates<cartesian, reciprocal>(vf, vc);
//==         double length = Utils::vector_length(vc);
//==         total_path_length += length;
//==         segment_length.push_back(length);
//==     }
//==
//==     std::vector<double> xaxis;
//==
//==     sirius::K_point_set kset_(global_parameters);
//==
//==     double prev_seg_len = 0.0;
//==
//==     // segments
//==     for (int ip = 0; ip < (int)bz_path.size() - 1; ip++)
//==     {
//==         std::vector<double> p0 = bz_path[ip].second;
//==         std::vector<double> p1 = bz_path[ip + 1].second;
//==
//==         int n = int((segment_length[ip] * num_steps) / total_path_length);
//==         int n0 = (ip == (int)bz_path.size() - 2) ? n - 1 : n;
//==
//==         double dvf[3];
//==         for (int x = 0; x < 3; x++) dvf[x] = (p1[x] - p0[x]) / double(n0);
//==
//==         for (int i = 0; i < n; i++)
//==         {
//==             double vf[3];
//==             for (int x = 0; x < 3; x++) vf[x] = p0[x] + dvf[x] * i;
//==             kset_.add_kpoint(vf, 0.0);
//==
//==             xaxis.push_back(prev_seg_len + segment_length[ip] * i / double(n0));
//==         }
//==         prev_seg_len += segment_length[ip];
//==     }
//==
//==     std::vector<double> xaxis_ticks;
//==     std::vector<std::string> xaxis_tick_labels;
//==     prev_seg_len = 0.0;
//==     for (int ip = 0; ip < (int)bz_path.size(); ip++)
//==     {
//==         xaxis_ticks.push_back(prev_seg_len);
//==         xaxis_tick_labels.push_back(bz_path[ip].first);
//==         if (ip < (int)bz_path.size() - 1) prev_seg_len += segment_length[ip];
//==     }
//==
//==     kset_.initialize();
//==
//==     sim_ctx->solve_free_atoms();
//==
//==     potential->update_atomic_potential();
//==     sim_ctx->generate_radial_functions();
//==     sim_ctx->generate_radial_integrals();
//==
//==     // generate plane-wave coefficients of the potential in the interstitial region
//==     potential->generate_pw_coefs();
//==
//==     for (int ikloc = 0; ikloc < kset_.spl_num_kpoints().local_size(); ikloc++)
//==     {
//==         int ik = kset_.spl_num_kpoints(ikloc);
//==         kset_[ik]->find_eigen_states(potential->effective_potential(), potential->effective_magnetic_field());
//==     }
//==     // synchronize eigen-values
//==     kset_.sync_band_energies();
//==
//==     if (sim_ctx->mpi_grid().root())
//==     {
//==         JSON_write jw("bands.json");
//==         jw.single("xaxis", xaxis);
//==         //** jw.single("Ef", sim_ctx->rti().energy_fermi);
//==
//==         jw.single("xaxis_ticks", xaxis_ticks);
//==         jw.single("xaxis_tick_labels", xaxis_tick_labels);
//==
//==         jw.begin_array("plot");
//==         std::vector<double> yvalues(kset_.num_kpoints());
//==         for (int i = 0; i < sim_ctx->num_bands(); i++)
//==         {
//==             jw.begin_set();
//==             for (int ik = 0; ik < kset_.num_kpoints(); ik++) yvalues[ik] = kset_[ik]->band_energy(i);
//==             jw.single("yvalues", yvalues);
//==             jw.end_set();
//==         }
//==         jw.end_array();
//==
//==         //FILE* fout = fopen("bands.dat", "w");
//==         //for (int i = 0; i < sim_ctx->num_bands(); i++)
//==         //{
//==         //    for (int ik = 0; ik < kpoint_set_.num_kpoints(); ik++)
//==         //    {
//==         //        fprintf(fout, "%f %f\n", xaxis[ik], kpoint_set_[ik]->band_energy(i));
//==         //    }
//==         //    fprintf(fout, "\n");
//==         //}
//==         //fclose(fout);
//==     }
//== }

void FORTRAN(sirius_plot_potential)(void)
{
    int N{10000};

    potential->effective_potential()->fft_transform(-1);

    std::vector<double> p(N);
    std::vector<double> x(N);

    vector3d<double> vf1({0.1, 0.1, 0.1});
    vector3d<double> vf2({0.4, 0.4, 0.4});

    #pragma omp parallel for default(shared)
    for (int i = 0; i < N; i++) {
        double t = double(i) / (N - 1);
        auto vf = vf1 + (vf2 - vf1) * t;

        auto vc = sim_ctx->unit_cell().get_cartesian_coordinates(vf);
        p[i] = potential->effective_potential()->value(vc);
        x[i] = vc.length();
    }

    FILE* fout = fopen("potential.dat", "w");
    for (int i = 0; i < N; i++) {
        fprintf(fout, "%.12f %.12f\n", x[i] - x[0], p[i]);
    }
    fclose(fout);
}

void sirius_write_json_output(void)
{
    json dict;
    dict["git_hash"] = git_hash;
    dict["build_date"] = build_date;
    dict["comm_world_size"] = mpi_comm_world().size();
    dict["threads_per_rank"] = omp_get_max_threads();
    dict["ground_state"] = dft_ground_state->serialize();
    dict["timers"] = Utils::serialize_timers();
 
    if (mpi_comm_world().rank() == 0) {
        std::ofstream ofs(std::string("output_") + sim_ctx->start_time_tag() + std::string(".json"),
                          std::ofstream::out | std::ofstream::trunc);
        ofs << dict.dump(4);
    }
}

void FORTRAN(sirius_get_occupation_matrix)(int32_t* atom_id, double_complex* occupation_matrix)
{
    int ia = *atom_id - 1;
    sim_ctx->unit_cell().atom(ia).get_occupation_matrix(occupation_matrix);
}

void FORTRAN(sirius_set_uj_correction_matrix)(int32_t* atom_id, int32_t* l, double_complex* uj_correction_matrix)
{
    int ia = *atom_id - 1;
    sim_ctx->unit_cell().atom(ia).set_uj_correction_matrix(*l, uj_correction_matrix);
}

void FORTRAN(sirius_set_so_correction)(int32_t* so_correction)
{
    if (*so_correction != 0) {
        sim_ctx->set_so_correction(true);
    } else {
        sim_ctx->set_so_correction(false);
    }
}

void FORTRAN(sirius_set_uj_correction)(int32_t* uj_correction)
{
    if (*uj_correction != 0) {
        sim_ctx->set_uj_correction(true);
    } else {
        sim_ctx->set_uj_correction(false);
    }
}

void sirius_get_energy_tot(double* total_energy__)
{
    *total_energy__ = dft_ground_state->total_energy();
}

void sirius_get_energy_ewald(double* ewald_energy__)
{
    *ewald_energy__ = dft_ground_state->energy_ewald();
}


void sirius_add_atom_type_aw_descriptor(char const* label__,
                                        int32_t const* n__,
                                        int32_t const* l__,
                                        double const* enu__,
                                        int32_t const* dme__,
                                        int32_t const* auto_enu__)
{
    auto& type = sim_ctx->unit_cell().atom_type(std::string(label__));
    type.add_aw_descriptor(*n__, *l__, *enu__, *dme__, *auto_enu__);
}

void sirius_add_atom_type_lo_descriptor(char const* label__,
                                        int32_t const* ilo__,
                                        int32_t const* n__,
                                        int32_t const* l__,
                                        double const* enu__,
                                        int32_t const* dme__,
                                        int32_t* auto_enu__)
{
    std::string label(label__);
    auto& type = sim_ctx->unit_cell().atom_type(std::string(label__));
    type.add_lo_descriptor(*ilo__ - 1, *n__, *l__, *enu__, *dme__, *auto_enu__);
}

void sirius_set_aw_enu(int32_t const* ia__,
                       int32_t const* l__,
                       int32_t const* order__,
                       double const* enu__)
{
    sim_ctx->unit_cell().atom(*ia__ - 1).symmetry_class().set_aw_enu(*l__, *order__ - 1, *enu__);
}

void sirius_get_aw_enu(int32_t const* ia__,
                       int32_t const* l__,
                       int32_t const* order__,
                       double* enu__)
{
    *enu__ = sim_ctx->unit_cell().atom(*ia__ - 1).symmetry_class().get_aw_enu(*l__, *order__ - 1);
}

void sirius_set_lo_enu(int32_t const* ia__,
                       int32_t const* idxlo__,
                       int32_t const* order__,
                       double const* enu__)
{
    sim_ctx->unit_cell().atom(*ia__ - 1).symmetry_class().set_lo_enu(*idxlo__ - 1, *order__ - 1, *enu__);
}

void sirius_get_lo_enu(int32_t const* ia__,
                       int32_t const* idxlo__,
                       int32_t const* order__,
                       double* enu__)
{
    *enu__ = sim_ctx->unit_cell().atom(*ia__ - 1).symmetry_class().get_lo_enu(*idxlo__ - 1, *order__ - 1);
}

void sirius_get_local_num_kpoints(int32_t* kset_id, int32_t* nkpt_loc)
{
    *nkpt_loc = (int)kset_list[*kset_id]->spl_num_kpoints().local_size();
}

void sirius_get_local_kpoint_rank_and_offset(int32_t* kset_id, int32_t* ik, int32_t* rank, int32_t* ikloc)
{
    *rank = kset_list[*kset_id]->spl_num_kpoints().local_rank(*ik - 1);
    *ikloc = (int)kset_list[*kset_id]->spl_num_kpoints().local_index(*ik - 1) + 1;
}

void sirius_get_global_kpoint_index(int32_t* kset_id, int32_t* ikloc, int32_t* ik)
{
    *ik = kset_list[*kset_id]->spl_num_kpoints(*ikloc - 1) + 1; // Fortran counts from 1
}

/// Generate radial functions (both aw and lo)
void sirius_generate_radial_functions()
{
    sim_ctx->unit_cell().generate_radial_functions();
}

/// Generate radial integrals
void sirius_generate_radial_integrals()
{
    sim_ctx->unit_cell().generate_radial_integrals();
}

void sirius_get_symmetry_classes(int32_t* ncls, int32_t* icls_by_ia)
{
    *ncls = sim_ctx->unit_cell().num_atom_symmetry_classes();

    for (int ic = 0; ic < sim_ctx->unit_cell().num_atom_symmetry_classes(); ic++)
    {
        for (int i = 0; i < sim_ctx->unit_cell().atom_symmetry_class(ic).num_atoms(); i++)
            icls_by_ia[sim_ctx->unit_cell().atom_symmetry_class(ic).atom_id(i)] = ic + 1; // Fortran counts from 1
    }
}

void sirius_get_max_mt_radial_basis_size(int32_t* max_mt_radial_basis_size)
{
    *max_mt_radial_basis_size = sim_ctx->unit_cell().max_mt_radial_basis_size();
}

void sirius_get_radial_functions(double* radial_functions__)
{
    mdarray<double, 3> radial_functions(radial_functions__,
                                        sim_ctx->unit_cell().max_num_mt_points(),
                                        sim_ctx->unit_cell().max_mt_radial_basis_size(),
                                        sim_ctx->unit_cell().num_atom_symmetry_classes());
    radial_functions.zero();

    for (int ic = 0; ic < sim_ctx->unit_cell().num_atom_symmetry_classes(); ic++)
    {
        for (int idxrf = 0; idxrf < sim_ctx->unit_cell().atom_symmetry_class(ic).atom_type().mt_radial_basis_size(); idxrf++)
        {
            for (int ir = 0; ir < sim_ctx->unit_cell().atom_symmetry_class(ic).atom_type().num_mt_points(); ir++)
                radial_functions(ir, idxrf, ic) = sim_ctx->unit_cell().atom_symmetry_class(ic).radial_function(ir, idxrf);
        }
    }
}

void sirius_get_max_mt_basis_size(int32_t* max_mt_basis_size)
{
    *max_mt_basis_size = sim_ctx->unit_cell().max_mt_basis_size();
}

void sirius_get_basis_functions_index(int32_t* mt_basis_size, int32_t* offset_wf, int32_t* indexb__)
{
    mdarray<int, 3> indexb(indexb__, 4, sim_ctx->unit_cell().max_mt_basis_size(), sim_ctx->unit_cell().num_atoms());

    for (int ia = 0; ia < sim_ctx->unit_cell().num_atoms(); ia++)
    {
        mt_basis_size[ia] = sim_ctx->unit_cell().atom(ia).type().mt_basis_size();
        offset_wf[ia] = sim_ctx->unit_cell().atom(ia).offset_mt_coeffs();

        for (int j = 0; j < sim_ctx->unit_cell().atom(ia).type().mt_basis_size(); j++)
        {
            indexb(0, j, ia) = sim_ctx->unit_cell().atom(ia).type().indexb(j).l;
            indexb(1, j, ia) = sim_ctx->unit_cell().atom(ia).type().indexb(j).lm + 1; // Fortran counts from 1
            indexb(2, j, ia) = sim_ctx->unit_cell().atom(ia).type().indexb(j).idxrf + 1; // Fortran counts from 1
        }
    }
}

/// Get number of G+k vectors for a given k-point in the set
void sirius_get_num_gkvec(ftn_int* kset_id__,
                          ftn_int* ik__,
                          ftn_int* num_gkvec__)
{
    auto ks = kset_list[*kset_id__];
    auto kp = (*kset_list[*kset_id__])[*ik__ - 1];
    /* get rank that stores a given k-point */
    int rank = ks->spl_num_kpoints().local_rank(*ik__ - 1);
    auto& comm_k = sim_ctx->comm_k();
    if (rank == comm_k.rank()) {
        *num_gkvec__ = kp->num_gkvec();
    }
    comm_k.bcast(num_gkvec__, 1, rank);
}

/// Get maximum number of G+k vectors across all k-points in the set
void sirius_get_max_num_gkvec(ftn_int* kset_id__,
                              ftn_int* max_num_gkvec__)
{
    *max_num_gkvec__ = kset_list[*kset_id__]->max_num_gkvec();
}

/// Get all G+k vector related arrays
void sirius_get_gkvec_arrays(ftn_int*    kset_id__,
                             ftn_int*    ik__,
                             ftn_int*    num_gkvec__,
                             ftn_int*    gvec_index__,
                             ftn_double* gkvec__,
                             ftn_double* gkvec_cart__,
                             ftn_double* gkvec_len,
                             ftn_double* gkvec_tp__)
{
    
    auto ks = kset_list[*kset_id__];
    auto kp = (*kset_list[*kset_id__])[*ik__ - 1];

    /* get rank that stores a given k-point */
    int rank = ks->spl_num_kpoints().local_rank(*ik__ - 1);

    auto& comm_k = sim_ctx->comm_k();

    if (rank == comm_k.rank()) {
        *num_gkvec__ = kp->num_gkvec();
        mdarray<double, 2> gkvec(gkvec__, 3, kp->num_gkvec());
        mdarray<double, 2> gkvec_cart(gkvec_cart__, 3, kp->num_gkvec());
        mdarray<double, 2> gkvec_tp(gkvec_tp__, 2, kp->num_gkvec());

        for (int igk = 0; igk < kp->num_gkvec(); igk++) {
            auto gkc = kp->gkvec().gkvec_cart(igk);
            auto G = kp->gkvec().gvec(igk);

            gvec_index__[igk] = sim_ctx->gvec().index_by_gvec(G) + 1; // Fortran counts from 1
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

void sirius_get_matching_coefficients(int32_t const* kset_id__,
                                      int32_t const* ik__,
                                      double_complex* apwalm__,
                                      int32_t const* ngkmax__,
                                      int32_t const* apwordmax__)
{

    TERMINATE_NOT_IMPLEMENTED;

    //int rank = kset_list[*kset_id__]->spl_num_kpoints().local_rank(*ik__ - 1);
    //
    //if (rank == sim_ctx->mpi_grid().coordinate(0))
    //{
    //    auto kp = (*kset_list[*kset_id__])[*ik__ - 1];
    //
    //    mdarray<double_complex, 4> apwalm(apwalm__, *ngkmax__, *apwordmax__, sim_ctx->lmmax_apw(),
    //                                      sim_ctx->unit_cell().num_atoms());


    //    dmatrix<double_complex> alm(kp->num_gkvec_row(), sim_ctx->unit_cell().mt_aw_basis_size(), *blacs_grid, sim_ctx->cyclic_block_size(), sim_ctx->cyclic_block_size());
    //    kp->alm_coeffs_row()->generate<true>(alm);

    //    for (int i = 0; i < sim_ctx->unit_cell().mt_aw_basis_size(); i++)
    //    {
    //        int ia = sim_ctx->unit_cell().mt_aw_basis_descriptor(i).ia;
    //        int xi = sim_ctx->unit_cell().mt_aw_basis_descriptor(i).xi;
    //
    //        int lm = sim_ctx->unit_cell().atom(ia).type().indexb(xi).lm;
    //        int order = sim_ctx->unit_cell().atom(ia).type().indexb(xi).order;

    //        for (int igkloc = 0; igkloc < kp->num_gkvec_row(); igkloc++)
    //        {
    //            int igk = kp->gklo_basis_descriptor_row(igkloc).igk;
    //            apwalm(igk, order, lm, ia) = alm(igkloc, i);
    //        }
    //    }
    //    //== for (int ia = 0; ia < sim_ctx->unit_cell().num_atoms(); ia++)
    //    //== {
    //    //==     Platform::allreduce(&apwalm(0, 0, 0, ia), (int)(apwalm.size(0) * apwalm.size(1) * apwalm.size(2)),
    //    //==                         sim_ctx->mpi_grid().communicator(1 << _dim_row_));
    //    //== }
    //}
}

///// Get first-variational matrices of Hamiltonian and overlap
///** Radial integrals and plane-wave coefficients of the interstitial potential must be calculated prior to
// *  Hamiltonian and overlap matrix construction.
// */
//void sirius_get_fv_h_o(int32_t const* kset_id__,
//                       int32_t const* ik__,
//                       int32_t const* size__,
//                       double_complex* h__,
//                       double_complex* o__)
//{
//    int rank = kset_list[*kset_id__]->spl_num_kpoints().local_rank(*ik__ - 1);
//
//    if (rank == sim_ctx->mpi_grid().coordinate(0))
//    {
//        auto kp = (*kset_list[*kset_id__])[*ik__ - 1];
//
//        if (*size__ != kp->gklo_basis_size())
//        {
//            TERMINATE("wrong matrix size");
//        }
//
//        dmatrix<double_complex> h(h__, kp->gklo_basis_size(), kp->gklo_basis_size(), sim_ctx->blacs_grid(), sim_ctx->cyclic_block_size(), sim_ctx->cyclic_block_size());
//        dmatrix<double_complex> o(o__, kp->gklo_basis_size(), kp->gklo_basis_size(), sim_ctx->blacs_grid(), sim_ctx->cyclic_block_size(), sim_ctx->cyclic_block_size());
//        dft_ground_state->band().set_fv_h_o<CPU, electronic_structure_method_t::full_potential_lapwlo>(kp, *potential, h, o);
//    }
//}

//void sirius_solve_fv(int32_t const* kset_id__,
//                     int32_t const* ik__,
//                     double_complex* h__,
//                     double_complex* o__,
//                     double* eval__,
//                     double_complex* evec__,
//                     int32_t const* evec_ld__)
//{
//    int rank = kset_list[*kset_id__]->spl_num_kpoints().local_rank(*ik__ - 1);
//
//    if (rank == sim_ctx->mpi_grid().coordinate(0))
//    {
//        auto kp = (*kset_list[*kset_id__])[*ik__ - 1];
//
//        dft_ground_state->band().gen_evp_solver().solve(kp->gklo_basis_size(),
//                                                        sim_ctx->num_fv_states(),
//                                                        h__,
//                                                        kp->gklo_basis_size_row(),
//                                                        o__,
//                                                        kp->gklo_basis_size_row(),
//                                                        eval__,
//                                                        evec__,
//                                                        *evec_ld__,
//                                                        kp->gklo_basis_size_row(),
//                                                        kp->gklo_basis_size_col());
//    }
//}

///// Get the total size of wave-function (number of mt coefficients + number of G+k coefficients)
//void sirius_get_mtgk_size(int32_t* kset_id, int32_t* ik, int32_t* mtgk_size)
//{
//    *mtgk_size = (*kset_list[*kset_id])[*ik - 1]->wf_size();
//}

void sirius_get_spinor_wave_functions(int32_t* kset_id, int32_t* ik, double_complex* spinor_wave_functions__)
{
    TERMINATE("fix this for distributed WF storage");
    //== assert(sim_ctx->num_bands() == (int)sim_ctx->spl_spinor_wf().local_size());

    //== sirius::K_point* kp = (*kset_list[*kset_id])[*ik - 1];
    //==
    //== mdarray<double_complex, 3> spinor_wave_functions(spinor_wave_functions__, kp->wf_size(), sim_ctx->num_spins(),
    //==                                             sim_ctx->spl_spinor_wf().local_size());

    //== for (int j = 0; j < (int)sim_ctx->spl_spinor_wf().local_size(); j++)
    //== {
    //==     memcpy(&spinor_wave_functions(0, 0, j), &kp->spinor_wave_function(0, 0, j),
    //==            kp->wf_size() * sim_ctx->num_spins() * sizeof(double_complex));
    //== }
}

//== void FORTRAN(sirius_apply_step_function_gk)(int32_t* kset_id, int32_t* ik, double_complex* wf__)
//== {
//==     int thread_id = Platform::thread_id();
//==
//==     sirius::K_point* kp = (*kset_list[*kset_id])[*ik - 1];
//==     int num_gkvec = kp->num_gkvec();
//==
//==     sim_ctx->reciprocal_lattice()->fft().input(num_gkvec, kp->fft_index(), wf__, thread_id);
//==     sim_ctx->reciprocal_lattice()->fft().transform(1, thread_id);
//==     for (int ir = 0; ir < sim_ctx->reciprocal_lattice()->fft().size(); ir++)
//==         sim_ctx->reciprocal_lattice()->fft().buffer(ir, thread_id) *= sim_ctx->step_function()->theta_it(ir);
//==
//==     sim_ctx->reciprocal_lattice()->fft().transform(-1, thread_id);
//==     sim_ctx->reciprocal_lattice()->fft().output(num_gkvec, kp->fft_index(), wf__, thread_id);
//== }

/// Get Cartesian coordinates of G+k vectors
void sirius_get_gkvec_cart(int32_t* kset_id, int32_t* ik, double* gkvec_cart__)
{
    sirius::K_point* kp = (*kset_list[*kset_id])[*ik - 1];
    mdarray<double, 2> gkvec_cart(gkvec_cart__, 3, kp->num_gkvec());

    for (int igk = 0; igk < kp->num_gkvec(); igk++)
    {
        for (int x = 0; x < 3; x++) gkvec_cart(x, igk) = kp->gkvec().gkvec(igk)[x]; //kp->gkvec<cartesian>(igk)[x];
    }
}

void sirius_get_evalsum(double* evalsum)
{
    *evalsum = dft_ground_state->eval_sum();
}

void sirius_get_energy_exc(double* energy_exc)
{
    *energy_exc = dft_ground_state->energy_exc();
}

void sirius_get_energy_vxc(double* energy_vxc)
{
    *energy_vxc = dft_ground_state->energy_vxc();
}

void sirius_get_energy_bxc(double* energy_bxc)
{
    *energy_bxc = dft_ground_state->energy_bxc();
}

void sirius_get_energy_veff(double* energy_veff)
{
    *energy_veff = dft_ground_state->energy_veff();
}

void sirius_get_energy_vloc(ftn_double* energy_vloc__)
{
    *energy_vloc__ = dft_ground_state->energy_vloc();
}

void sirius_get_energy_vha(double* energy_vha)
{
    *energy_vha = dft_ground_state->energy_vha();
}

void sirius_get_energy_enuc(double* energy_enuc)
{
    *energy_enuc = dft_ground_state->energy_enuc();
}

void sirius_get_energy_kin(double* energy_kin)
{
    *energy_kin = dft_ground_state->energy_kin();
}

/// Generate XC potential and magnetic field
void sirius_generate_xc_potential(ftn_double* vxcmt__,
                                  ftn_double* vxcit__,
                                  ftn_double* bxcmt__,
                                  ftn_double* bxcit__)
{

    potential->xc(density->rho(), density->magnetization(), potential->xc_potential(), potential->effective_magnetic_field(),
                  potential->xc_energy_density());

    potential->xc_potential()->copy_to_global_ptr(vxcmt__, vxcit__);

    if (sim_ctx->num_mag_dims() == 0) {
        return;
    }
    assert(sim_ctx->num_spins() == 2);

    /* set temporary array wrapper */
    mdarray<double, 4> bxcmt(bxcmt__, sim_ctx->lmmax_pot(), sim_ctx->unit_cell().max_num_mt_points(),
                             sim_ctx->unit_cell().num_atoms(), sim_ctx->num_mag_dims());
    mdarray<double, 2> bxcit(bxcit__, sim_ctx->fft().local_size(), sim_ctx->num_mag_dims());

    if (sim_ctx->num_mag_dims() == 1) {
        /* z component */
        potential->effective_magnetic_field(0)->copy_to_global_ptr(&bxcmt(0, 0, 0, 0), &bxcit(0, 0));
    } else {
        /* z component */
        potential->effective_magnetic_field(0)->copy_to_global_ptr(&bxcmt(0, 0, 0, 2), &bxcit(0, 2));
        /* x component */
        potential->effective_magnetic_field(1)->copy_to_global_ptr(&bxcmt(0, 0, 0, 0), &bxcit(0, 0));
        /* y component */
        potential->effective_magnetic_field(2)->copy_to_global_ptr(&bxcmt(0, 0, 0, 1), &bxcit(0, 1));
    }
}

void sirius_generate_rho_multipole_moments(ftn_int*            lmmax__,
                                           ftn_double_complex* qmt__)
{
    mdarray<ftn_double_complex, 2> qmt(qmt__, *lmmax__, sim_ctx->unit_cell().num_atoms());
    qmt.zero();
    
    int lmmax = std::min(*lmmax__, sim_ctx->lmmax_rho());

    auto l_by_lm = Utils::l_by_lm(Utils::lmax_by_lmmax(lmmax));

    for (int ialoc = 0; ialoc < sim_ctx->unit_cell().spl_num_atoms().local_size(); ialoc++) {
        int ia = sim_ctx->unit_cell().spl_num_atoms(ialoc);
        std::vector<double> tmp(lmmax);
        for (int lm = 0; lm < lmmax; lm++) {
            int l = l_by_lm[lm];
            auto s = density->rho()->f_mt(ialoc).component(lm);
            tmp[lm] = s.integrate(l + 2);
        }
        sirius::SHT::convert(Utils::lmax_by_lmmax(lmmax), tmp.data(), &qmt(0, ia));
        qmt(0, ia) -= sim_ctx->unit_cell().atom(ia).zn() * y00;
    }
    sim_ctx->comm().allreduce(&qmt(0, 0), static_cast<int>(qmt.size()));
}

void sirius_generate_coulomb_potential_mt(ftn_int*            ia__,
                                          ftn_int*            lmmax_rho__,
                                          ftn_double_complex* rho__,
                                          ftn_int*            lmmax_pot__,
                                          ftn_double_complex* vmt__)
{
    auto& atom = sim_ctx->unit_cell().atom(*ia__ - 1);

    sirius::Spheric_function<function_domain_t::spectral, double_complex> rho(rho__, *lmmax_rho__, atom.radial_grid());
    sirius::Spheric_function<function_domain_t::spectral, double_complex> vmt(vmt__, *lmmax_pot__, atom.radial_grid());
    potential->poisson_vmt<true>(atom, rho, vmt);
}

void sirius_generate_coulomb_potential(ftn_double* vclmt__,
                                       ftn_double* vclit__)
{
    density->rho()->fft_transform(-1);
    potential->poisson(density->rho(), potential->hartree_potential());
    potential->hartree_potential()->copy_to_global_ptr(vclmt__, vclit__);
}

void sirius_update_atomic_potential()
{
    potential->update_atomic_potential();
}

void sirius_radial_solver(ftn_char    type__,
                          ftn_int*    zn__,
                          ftn_int*    dme__,
                          ftn_int*    l__,
                          ftn_int*    k__, 
                          ftn_double* enu__,
                          ftn_int*    nr__,
                          ftn_double* r__,
                          ftn_double* v__,
                          ftn_int*    nn__,
                          ftn_double* p0__,
                          ftn_double* p1__,
                          ftn_double* q0__,
                          ftn_double* q1__,
                          ftn_int     type_len__)
{

    std::string type(type__, type_len__);
    if (type != "none") {
        TERMINATE_NOT_IMPLEMENTED;
    }

    relativity_t rel = relativity_t::none;

    sirius::Radial_grid_ext<double> rgrid(*nr__, r__);
    std::vector<double> v(v__, v__ + rgrid.num_points());
    sirius::Radial_solver solver(*zn__, v, rgrid);

    auto result = solver.solve(rel, *dme__, *l__, *k__, *enu__);

    *nn__ = std::get<0>(result);
    std::memcpy(p0__, std::get<1>(result).data(), rgrid.num_points() * sizeof(double));
    std::memcpy(p1__, std::get<2>(result).data(), rgrid.num_points() * sizeof(double));
    std::memcpy(q0__, std::get<3>(result).data(), rgrid.num_points() * sizeof(double));
    std::memcpy(q1__, std::get<4>(result).data(), rgrid.num_points() * sizeof(double));
}

void sirius_get_aw_radial_function(ftn_int*    ia__,
                                   ftn_int*    l__,
                                   ftn_int*    io__,
                                   ftn_double* f__)
{
    int ia = *ia__ - 1;
    int io = *io__ - 1;
    auto& atom = sim_ctx->unit_cell().atom(ia);
    int idxrf = atom.type().indexr_by_l_order(*l__, io);
    for (int ir = 0; ir < atom.num_mt_points(); ir++) {
        f__[ir] = atom.symmetry_class().radial_function(ir, idxrf);
    }
}

void sirius_set_aw_radial_function(ftn_int*    ia__,
                                   ftn_int*    l__,
                                   ftn_int*    io__,
                                   ftn_double* f__)
{
    int ia = *ia__ - 1;
    int io = *io__ - 1;
    auto& atom = sim_ctx->unit_cell().atom(ia);
    int idxrf = atom.type().indexr_by_l_order(*l__, io);
    for (int ir = 0; ir < atom.num_mt_points(); ir++) {
        atom.symmetry_class().radial_function(ir, idxrf) = f__[ir];
    }
}

void sirius_set_aw_radial_function_derivative(ftn_int*    ia__,
                                              ftn_int*    l__,
                                              ftn_int*    io__,
                                              ftn_double* f__)
{
    int ia = *ia__ - 1;
    int io = *io__ - 1;
    auto& atom = sim_ctx->unit_cell().atom(ia);
    int idxrf = atom.type().indexr_by_l_order(*l__, io);
    for (int ir = 0; ir < atom.num_mt_points(); ir++) {
        atom.symmetry_class().radial_function_derivative(ir, idxrf) = f__[ir] * atom.type().radial_grid()[ir];
    }
}

//void sirius_get_aw_deriv_radial_function(int32_t* ia__,
//                                         int32_t* l__,
//                                         int32_t* io__,
//                                         double* dfdr__)
//{
//    int ia = *ia__ - 1;
//    int io = *io__ - 1;
//    auto& atom = sim_ctx->unit_cell().atom(ia);
//    int idxrf = atom.type().indexr_by_l_order(*l__, io);
//    for (int ir = 0; ir < atom.num_mt_points(); ir++)
//    {
//        double rinv = atom.type().radial_grid().x_inv(ir);
//        dfdr__[ir] = atom.symmetry_class().r_deriv_radial_function(ir, idxrf) * rinv;
//    }
//}

void sirius_get_aw_surface_derivative(ftn_int*    ia__,
                                      ftn_int*    l__,
                                      ftn_int*    io__,
                                      ftn_int*    dm__,
                                      ftn_double* deriv__)
{
    *deriv__ = sim_ctx->unit_cell().atom(*ia__ - 1).symmetry_class().aw_surface_dm(*l__, *io__ - 1, *dm__);
}

void sirius_set_aw_surface_derivative(ftn_int*    ia__,
                                      ftn_int*    l__,
                                      ftn_int*    io__,
                                      ftn_int*    dm__,
                                      ftn_double* deriv__)
{
    sim_ctx->unit_cell().atom(*ia__ - 1).symmetry_class().set_aw_surface_deriv(*l__, *io__ - 1, *dm__, *deriv__);
}

void sirius_get_lo_radial_function(ftn_int*    ia__,
                                   ftn_int*    idxlo__,
                                   ftn_double* f__)
{
    int ia = *ia__ - 1;
    int idxlo = *idxlo__ - 1;
    auto& atom = sim_ctx->unit_cell().atom(ia);
    int idxrf = atom.type().indexr_by_idxlo(idxlo);
    for (int ir = 0; ir < atom.num_mt_points(); ir++) {
        f__[ir] = atom.symmetry_class().radial_function(ir, idxrf);
    }
}

void sirius_set_lo_radial_function(ftn_int*    ia__,
                                   ftn_int*    idxlo__,
                                   ftn_double* f__)
{
    int ia = *ia__ - 1;
    int idxlo = *idxlo__ - 1;
    auto& atom = sim_ctx->unit_cell().atom(ia);
    int idxrf = atom.type().indexr_by_idxlo(idxlo);
    for (int ir = 0; ir < atom.num_mt_points(); ir++) {
        atom.symmetry_class().radial_function(ir, idxrf) = f__[ir];
    }
}

void sirius_set_lo_radial_function_derivative(ftn_int*    ia__,
                                              ftn_int*    idxlo__,
                                              ftn_double* f__)
{
    int ia = *ia__ - 1;
    int idxlo = *idxlo__ - 1;
    auto& atom = sim_ctx->unit_cell().atom(ia);
    int idxrf = atom.type().indexr_by_idxlo(idxlo);
    for (int ir = 0; ir < atom.num_mt_points(); ir++) {
        atom.symmetry_class().radial_function_derivative(ir, idxrf) = f__[ir] * atom.type().radial_grid()[ir];
    }
}

//void sirius_get_lo_deriv_radial_function(int32_t const* ia__,
//                                         int32_t const* idxlo__,
//                                         double* dfdr__)
//{
//    int ia = *ia__ - 1;
//    int idxlo = *idxlo__ - 1;
//    auto& atom = sim_ctx->unit_cell().atom(ia);
//    int idxrf = atom.type().indexr_by_idxlo(idxlo);
//    for (int ir = 0; ir < atom.num_mt_points(); ir++)
//    {
//        double rinv = atom.type().radial_grid().x_inv(ir);
//        dfdr__[ir] = atom.symmetry_class().r_deriv_radial_function(ir, idxrf) * rinv;
//    }
//}

void sirius_get_aw_lo_o_radial_integral(int32_t* ia__, int32_t* l, int32_t* io1, int32_t* ilo2,
                                        double* oalo)
{
    int ia = *ia__ - 1;

    int idxrf2 = sim_ctx->unit_cell().atom(ia).type().indexr_by_idxlo(*ilo2 - 1);
    int order2 = sim_ctx->unit_cell().atom(ia).type().indexr(idxrf2).order;

    *oalo = sim_ctx->unit_cell().atom(ia).symmetry_class().o_radial_integral(*l, *io1 - 1, order2);
}

void sirius_set_aw_lo_o_radial_integral(int32_t* ia__,
                                        int32_t* l__,
                                        int32_t* io1__,
                                        int32_t* ilo2__,
                                        double* oalo__)
{
    int ia = *ia__ - 1;

    int idxrf2 = sim_ctx->unit_cell().atom(ia).type().indexr_by_idxlo(*ilo2__ - 1);
    int order2 = sim_ctx->unit_cell().atom(ia).type().indexr(idxrf2).order;

    //double d1 = std::abs(*oalo__ - sim_ctx->unit_cell().atom(ia).symmetry_class().o_radial_integral(*l__, *io1__ - 1, order2));
    //double d2 = std::abs(*oalo__ - sim_ctx->unit_cell().atom(ia).symmetry_class().o_radial_integral(*l__, order2, *io1__ - 1));
    //
    //if (d1 > 1e-6) {
    //    printf("ia: %i, oalo diff=%f\n", ia, d1);
    //}
    //if (d2 > 1e-6) {
    //    printf("ia: %i, oloa diff=%f\n", ia, d2);
    //}
    sim_ctx->unit_cell().atom(ia).symmetry_class().set_o_radial_integral(*l__, *io1__ - 1, order2, *oalo__);
    sim_ctx->unit_cell().atom(ia).symmetry_class().set_o_radial_integral(*l__, order2, *io1__ - 1, *oalo__);
}

void sirius_get_lo_lo_o_radial_integral(int32_t* ia__, int32_t* l, int32_t* ilo1, int32_t* ilo2,
                                        double* ololo)
{
    int ia = *ia__ - 1;

    int idxrf1 = sim_ctx->unit_cell().atom(ia).type().indexr_by_idxlo(*ilo1 - 1);
    int order1 = sim_ctx->unit_cell().atom(ia).type().indexr(idxrf1).order;
    int idxrf2 = sim_ctx->unit_cell().atom(ia).type().indexr_by_idxlo(*ilo2 - 1);
    int order2 = sim_ctx->unit_cell().atom(ia).type().indexr(idxrf2).order;

    *ololo = sim_ctx->unit_cell().atom(ia).symmetry_class().o_radial_integral(*l, order1, order2);
}

void sirius_set_lo_lo_o_radial_integral(int32_t* ia__,
                                        int32_t* l__,
                                        int32_t* ilo1__,
                                        int32_t* ilo2__,
                                        double* ololo__)
{
    int ia = *ia__ - 1;

    int idxrf1 = sim_ctx->unit_cell().atom(ia).type().indexr_by_idxlo(*ilo1__ - 1);
    int order1 = sim_ctx->unit_cell().atom(ia).type().indexr(idxrf1).order;
    int idxrf2 = sim_ctx->unit_cell().atom(ia).type().indexr_by_idxlo(*ilo2__ - 1);
    int order2 = sim_ctx->unit_cell().atom(ia).type().indexr(idxrf2).order;

    //double d1 = std::abs(*ololo__ - sim_ctx->unit_cell().atom(ia).symmetry_class().o_radial_integral(*l__, order1, order2));
    //
    //if (d1 > 1e-6) {
    //    printf("ia: %i, ololo diff=%f\n", ia, d1);
    //}

    sim_ctx->unit_cell().atom(ia).symmetry_class().set_o_radial_integral(*l__, order1, order2, *ololo__);
}

void sirius_get_aw_aw_h_radial_integral(int32_t* ia__, int32_t* l1, int32_t* io1, int32_t* l2,
                                        int32_t* io2, int32_t* lm3, double* haa)
{
    int ia = *ia__ - 1;
    int idxrf1 = sim_ctx->unit_cell().atom(ia).type().indexr_by_l_order(*l1, *io1 - 1);
    int idxrf2 = sim_ctx->unit_cell().atom(ia).type().indexr_by_l_order(*l2, *io2 - 1);

    *haa = sim_ctx->unit_cell().atom(ia).h_radial_integrals(idxrf1, idxrf2)[*lm3 - 1];
}

void sirius_set_aw_aw_h_radial_integral(int32_t* ia__,
                                        int32_t* l1__,
                                        int32_t* io1__,
                                        int32_t* l2__,
                                        int32_t* io2__,
                                        int32_t* lm3__,
                                        double* haa__)
{
    int ia = *ia__ - 1;
    int idxrf1 = sim_ctx->unit_cell().atom(ia).type().indexr_by_l_order(*l1__, *io1__ - 1);
    int idxrf2 = sim_ctx->unit_cell().atom(ia).type().indexr_by_l_order(*l2__, *io2__ - 1);
    
    //double d1 = std::abs(*haa__ - sim_ctx->unit_cell().atom(ia).h_radial_integrals(idxrf1, idxrf2)[*lm3__ - 1]);
    //
    //if (d1 > 1e-3) {
    //    printf("ia: %i, l1: %i, io1: %i, l2: %i, io2: %i, lm3: %i, haa diff=%f\n", ia, *l1__, *io1__, *l2__, *io2__, *lm3__, d1);
    //    printf("exciting value: %f, sirius value: %f\n", *haa__, sim_ctx->unit_cell().atom(ia).h_radial_integrals(idxrf1, idxrf2)[*lm3__ - 1]);
    //}

    sim_ctx->unit_cell().atom(ia).h_radial_integrals(idxrf1, idxrf2)[*lm3__ - 1] = *haa__;
}

void sirius_get_lo_aw_h_radial_integral(int32_t* ia__, int32_t* ilo1, int32_t* l2, int32_t* io2, int32_t* lm3,
                                        double* hloa)
{
    int ia = *ia__ - 1;
    int idxrf1 = sim_ctx->unit_cell().atom(ia).type().indexr_by_idxlo(*ilo1 - 1);
    int idxrf2 = sim_ctx->unit_cell().atom(ia).type().indexr_by_l_order(*l2, *io2 - 1);

    *hloa = sim_ctx->unit_cell().atom(ia).h_radial_integrals(idxrf1, idxrf2)[*lm3 - 1];
}

void sirius_set_lo_aw_h_radial_integral(int32_t* ia__,
                                        int32_t* ilo1__,
                                        int32_t* l2__,
                                        int32_t* io2__,
                                        int32_t* lm3__,
                                        double* hloa__)
{
    int ia = *ia__ - 1;
    int idxrf1 = sim_ctx->unit_cell().atom(ia).type().indexr_by_idxlo(*ilo1__ - 1);
    int idxrf2 = sim_ctx->unit_cell().atom(ia).type().indexr_by_l_order(*l2__, *io2__ - 1);

    //double d1 = std::abs(*hloa__ -  sim_ctx->unit_cell().atom(ia).h_radial_integrals(idxrf1, idxrf2)[*lm3__ - 1]);
    //double d2 = std::abs(*hloa__ -  sim_ctx->unit_cell().atom(ia).h_radial_integrals(idxrf2, idxrf1)[*lm3__ - 1]);
    //if (d1 > 1e-6) {
    //    printf("ia: %i, hloa diff=%f\n", ia, d1);
    //}
    //if (d2 > 1e-6) {
    //    printf("ia: %i, halo diff=%f\n", ia, d2);
    //}

    sim_ctx->unit_cell().atom(ia).h_radial_integrals(idxrf1, idxrf2)[*lm3__ - 1] = *hloa__;
    sim_ctx->unit_cell().atom(ia).h_radial_integrals(idxrf2, idxrf1)[*lm3__ - 1] = *hloa__;
}


void sirius_get_lo_lo_h_radial_integral(int32_t* ia__, int32_t* ilo1, int32_t* ilo2, int32_t* lm3,
                                        double* hlolo)
{
    int ia = *ia__ - 1;
    int idxrf1 = sim_ctx->unit_cell().atom(ia).type().indexr_by_idxlo(*ilo1 - 1);
    int idxrf2 = sim_ctx->unit_cell().atom(ia).type().indexr_by_idxlo(*ilo2 - 1);

    *hlolo = sim_ctx->unit_cell().atom(ia).h_radial_integrals(idxrf1, idxrf2)[*lm3 - 1];
}

void sirius_set_lo_lo_h_radial_integral(int32_t* ia__,
                                        int32_t* ilo1__,
                                        int32_t* ilo2__,
                                        int32_t* lm3__,
                                        double* hlolo__)
{
    int ia = *ia__ - 1;
    int idxrf1 = sim_ctx->unit_cell().atom(ia).type().indexr_by_idxlo(*ilo1__ - 1);
    int idxrf2 = sim_ctx->unit_cell().atom(ia).type().indexr_by_idxlo(*ilo2__ - 1);

    //double d1 = std::abs(*hlolo__ -  sim_ctx->unit_cell().atom(ia).h_radial_integrals(idxrf1, idxrf2)[*lm3__ - 1]);
    //if (d1 > 1e-6) {
    //    printf("ia: %i, lo1: %i, lo2: %i, lm3: %i, hlolo diff=%f\n", ia, *ilo1__, *ilo2__, *lm3__, d1);
    //    printf("exciting value: %f, sirius value: %f\n", *hlolo__, sim_ctx->unit_cell().atom(ia).h_radial_integrals(idxrf1, idxrf2)[*lm3__ - 1]);
    //}

    sim_ctx->unit_cell().atom(ia).h_radial_integrals(idxrf1, idxrf2)[*lm3__ - 1] = *hlolo__;
}

void sirius_set_aw_aw_o1_radial_integral(ftn_int* ia__,
                                         ftn_int* l__,
                                         ftn_int* io1__,
                                         ftn_int* io2__,
                                         ftn_double* o1aa__)
{
    int ia = *ia__ - 1;
    int idxrf1 = sim_ctx->unit_cell().atom(ia).type().indexr_by_l_order(*l__, *io1__ - 1);
    int idxrf2 = sim_ctx->unit_cell().atom(ia).type().indexr_by_l_order(*l__, *io2__ - 1);

    sim_ctx->unit_cell().atom(ia).symmetry_class().set_o1_radial_integral(idxrf1, idxrf2, *o1aa__);
    sim_ctx->unit_cell().atom(ia).symmetry_class().set_o1_radial_integral(idxrf2, idxrf1, *o1aa__);
}

void sirius_set_aw_lo_o1_radial_integral(ftn_int* ia__,
                                         ftn_int* io1__,
                                         ftn_int* ilo2__,
                                         ftn_double* o1alo__)
{
    int ia = *ia__ - 1;
    int idxrf2 = sim_ctx->unit_cell().atom(ia).type().indexr_by_idxlo(*ilo2__ - 1);
    int l = sim_ctx->unit_cell().atom(ia).type().indexr(idxrf2).l;
    int idxrf1 = sim_ctx->unit_cell().atom(ia).type().indexr_by_l_order(l, *io1__ - 1);

    sim_ctx->unit_cell().atom(ia).symmetry_class().set_o1_radial_integral(idxrf1, idxrf2, *o1alo__);
    sim_ctx->unit_cell().atom(ia).symmetry_class().set_o1_radial_integral(idxrf2, idxrf1, *o1alo__);
}

void sirius_set_lo_lo_o1_radial_integral(ftn_int* ia__,
                                         ftn_int* ilo1__,
                                         ftn_int* ilo2__,
                                         ftn_double* o1lolo__)
{
    int ia = *ia__ - 1;
    int idxrf1 = sim_ctx->unit_cell().atom(ia).type().indexr_by_idxlo(*ilo1__ - 1);
    int idxrf2 = sim_ctx->unit_cell().atom(ia).type().indexr_by_idxlo(*ilo2__ - 1);

    sim_ctx->unit_cell().atom(ia).symmetry_class().set_o1_radial_integral(idxrf1, idxrf2, *o1lolo__);
    sim_ctx->unit_cell().atom(ia).symmetry_class().set_o1_radial_integral(idxrf2, idxrf1, *o1lolo__);
}

void sirius_generate_potential_pw_coefs()
{
    potential->generate_pw_coefs();
}

void sirius_generate_density_pw_coefs()
{
    density->generate_pw_coefs();
}

/// Get first-variational eigen-vectors
/** Assume that the Fortran side holds the whole array */
void sirius_get_fv_eigen_vectors(int32_t* kset_id__, int32_t* ik__, double_complex* fv_evec__, int32_t* ld__,
                                 int32_t* num_fv_evec__)
{
    mdarray<double_complex, 2> fv_evec(fv_evec__, *ld__, *num_fv_evec__);
    (*kset_list[*kset_id__])[*ik__ - 1]->get_fv_eigen_vectors(fv_evec);
}

/// Get second-variational eigen-vectors
/** Assume that the Fortran side holds the whole array */
void sirius_get_sv_eigen_vectors(int32_t* kset_id, int32_t* ik, double_complex* sv_evec__, int32_t* size)
{
    mdarray<double_complex, 2> sv_evec(sv_evec__, *size, *size);
    (*kset_list[*kset_id])[*ik - 1]->get_sv_eigen_vectors(sv_evec);
}

void sirius_get_num_fv_states(int32_t* num_fv_states__)
{
    *num_fv_states__ = sim_ctx->num_fv_states();
}

void sirius_set_num_fv_states(int32_t* num_fv_states__)
{
    sim_ctx->set_num_fv_states(*num_fv_states__);
}

//void sirius_get_mpi_comm(int32_t* directions__, int32_t* fcomm__)
//{
//    *fcomm__ = MPI_Comm_c2f(sim_ctx->mpi_grid().communicator(*directions__).mpi_comm());
//}

void sirius_get_fft_comm(int32_t* fcomm__)
{
    *fcomm__ = MPI_Comm_c2f(sim_ctx->fft().comm().mpi_comm());
}

void sirius_get_kpoint_inner_comm(int32_t* fcomm__)
{
    *fcomm__ = MPI_Comm_c2f(sim_ctx->comm_band().mpi_comm());
}

void sirius_get_all_kpoints_comm(int32_t* fcomm__)
{
    *fcomm__ = MPI_Comm_c2f(sim_ctx->comm_k().mpi_comm());
}

void sirius_forces(double* forces__)
{
    //mdarray<double, 2> forces(forces__, 3, sim_ctx->unit_cell().num_atoms());
    //dft_ground_state->forces(forces);
}

void sirius_set_atom_pos(int32_t* atom_id, double* pos)
{
    sim_ctx->unit_cell().atom(*atom_id - 1).set_position(vector3d<double>(pos[0], pos[1], pos[2]));
}

void sirius_core_leakage(double* core_leakage)
{
    *core_leakage = density->core_leakage();
}

void sirius_ground_state_print_info()
{
    dft_ground_state->print_info();
}

void sirius_create_storage_file()
{
    sim_ctx->create_storage_file();
}

void sirius_test_spinor_wave_functions(int32_t* kset_id)
{
    sirius::K_point_set* kset = kset_list[*kset_id];
    for (int ikloc = 0; ikloc < (int)kset->spl_num_kpoints().local_size(); ikloc++)
    {
        int ik = kset->spl_num_kpoints(ikloc);
        (*kset)[ik]->test_spinor_wave_functions(0);
    }
}

//void sirius_generate_gq_matrix_elements(int32_t* kset_id, double* vq)
//{
//     kset_list[*kset_id]->generate_Gq_matrix_elements(vector3d<double>(vq[0], vq[1], vq[2]));
//}

void sirius_density_mixer_initialize(void)
{
    density->mixer_init();
}

void sirius_mix_density(double* rms)
{
    *rms = density->mix();
    density->fft_transform(1);
    sim_ctx->comm().bcast(rms, 1, 0);
}

void sirius_set_atom_type_dion(char* label__,
                               int32_t* num_beta__,
                               double* dion__)
{
    auto& type = sim_ctx->unit_cell().atom_type(std::string(label__));
    matrix<double> d_mtrx_ion(dion__, *num_beta__, *num_beta__);
    type.set_d_mtrx_ion(d_mtrx_ion);
}

// This must be called prior to sirius_set_atom_type_q_rf
void sirius_set_atom_type_beta_rf(char* label__,
                                  int32_t* num_beta__,
                                  int32_t* beta_l__,
                                  int32_t* num_mesh_points__,
                                  double* beta_rf__,
                                  int32_t* ld__)
{
    auto& type = sim_ctx->unit_cell().atom_type(std::string(label__));

    mdarray<double, 2> beta_rf(beta_rf__, *ld__, *num_beta__);
    type.pp_desc().lmax_beta_ = 0;
    type.pp_desc().num_beta_radial_functions = *num_beta__;
    type.pp_desc().beta_l = std::vector<int>(*num_beta__);
    type.pp_desc().num_beta_radial_points = std::vector<int>(*num_beta__);
    for (int i = 0; i < *num_beta__; i++)
    {
        type.pp_desc().beta_l[i] = beta_l__[i];
        type.pp_desc().lmax_beta_ = std::max(type.pp_desc().lmax_beta_, beta_l__[i]);
        type.pp_desc().num_beta_radial_points[i] = num_mesh_points__[i];
    }
    type.pp_desc().beta_radial_functions = mdarray<double, 2>(type.num_mt_points(), *num_beta__);
    beta_rf >> type.pp_desc().beta_radial_functions;
}

void sirius_set_atom_type_q_rf(char* label__,
                               double* q_rf__,
                               int32_t* lmax__)
{
    auto& type = sim_ctx->unit_cell().atom_type(std::string(label__));

    int nbeta = type.pp_desc().num_beta_radial_functions;

    /* temporary wrapper */
    mdarray<double, 3> q_rf(q_rf__, type.num_mt_points(), nbeta * (nbeta + 1) / 2, 2 * (*lmax__) + 1);

    /* allocate space for radial functions of Q operator */
    type.pp_desc().q_radial_functions_l = mdarray<double, 3>(type.num_mt_points(), nbeta * (nbeta + 1) / 2, 2 * type.pp_desc().lmax_beta_ + 1);

    for (int nb = 0; nb < nbeta; nb++) {
        for (int mb = nb; mb < nbeta; mb++) {
            /* combined index */
            int ijv = (mb + 1) * mb / 2 + nb;

            if (*lmax__ ==  type.pp_desc().lmax_beta_) {
                for (int l = 0; l <= 2 * type.pp_desc().lmax_beta_; l++) {
                    std::memcpy(&type.pp_desc().q_radial_functions_l(0, ijv, l), &q_rf(0, ijv, l), type.num_mt_points() * sizeof(double));
                }
            } else {
                std::stringstream s;
                s << "wrong lmax for " << std::string(label__) << " " << std::endl
                  << "lmax: " << *lmax__ << std::endl
                  << "lmax_beta: " << type.pp_desc().lmax_beta_;
                TERMINATE(s);
            }
        }
    }
    type.pp_desc().augment = true;
}

void sirius_set_atom_type_rho_core(char const* label__,
                                   int32_t* num_points__,
                                   double* rho_core__)
{
    auto& type = sim_ctx->unit_cell().atom_type(std::string(label__));
    type.pp_desc().core_charge_density = std::vector<double>(*num_points__);
    for (int i = 0; i < *num_points__; i++) type.pp_desc().core_charge_density[i] = rho_core__[i];
}

void sirius_set_atom_type_rho_tot(char const* label__,
                                  int32_t* num_points__,
                                  double* rho_tot__)
{
    auto& type = sim_ctx->unit_cell().atom_type(std::string(label__));
    type.pp_desc().total_charge_density = std::vector<double>(*num_points__);
    for (int i = 0; i < *num_points__; i++) type.pp_desc().total_charge_density[i] = rho_tot__[i];
}

void sirius_set_atom_type_vloc(char const* label__,
                               int32_t* num_points__,
                               double* vloc__)
{
    auto& type = sim_ctx->unit_cell().atom_type(std::string(label__));
    type.pp_desc().vloc = std::vector<double>(*num_points__);
    for (int i = 0; i < *num_points__; i++) type.pp_desc().vloc[i] = vloc__[i];
}

void sirius_symmetrize_density()
{
    dft_ground_state->symmetrize(density->rho(), density->magnetization(0), density->magnetization(1), density->magnetization(2));
}

void sirius_get_gvec_index(int32_t* gvec__, int32_t* ig__)
{
    vector3d<int> gv(gvec__[0], gvec__[1], gvec__[2]);
    *ig__ = sim_ctx->gvec().index_by_gvec(gv) + 1;
}

void sirius_use_internal_mixer(int32_t* use_internal_mixer__)
{
    *use_internal_mixer__ = (sim_ctx->mixer_input().exist_) ? 1 : 0;
}

void sirius_set_iterative_solver_tolerance(double* tol__)
{
    /* convert tolerance to Ha */
    sim_ctx->set_iterative_solver_tolerance(*tol__ / 2);
}

void sirius_get_density_dr2(double* dr2__)
{
    *dr2__ = density->dr2();
}

void sirius_real_gaunt_coeff_(int32_t* lm1__, int32_t* lm2__, int32_t* lm3__, double* coeff__)
{
    std::vector<int> idxlm(100);
    std::vector<int> phase(100, 1);
    int lm = 0;
    for (int l = 0; l < 10; l++)
    {
        idxlm[lm++] = Utils::lm_by_l_m(l, 0);
        for (int m = 1; m <= l; m++)
        {
            idxlm[lm++] = Utils::lm_by_l_m(l, m);
            idxlm[lm] = Utils::lm_by_l_m(l, -m);
            if (m % 2 == 0) phase[lm] = -1;
            lm++;
        }
    }

    int l1(0), m1(0), l2(0), m2(0), l3(0), m3(0);
    int s = 1;

    for (int l = 0; l < 10; l++)
    {
        for (int m = -l; m <= l; m++)
        {
            int lm = Utils::lm_by_l_m(l, m);
            if (lm == idxlm[*lm1__ - 1])
            {
                l1 = l;
                m1 = m;
                s *= phase[*lm1__ - 1];
            }
            if (lm == idxlm[*lm2__ - 1])
            {
                l2 = l;
                m2 = m;
                s *= phase[*lm2__ - 1];
            }
            if (lm == idxlm[*lm3__ - 1])
            {
                l3 = l;
                m3 = m;
                s *= phase[*lm3__ - 1];
            }
        }
    }
    double d = 0;
    for (int k1 = -l1; k1 <= l1; k1++)
    {
        for (int k2 = -l2; k2 <= l2; k2++)
        {
            for (int k3 = -l3; k3 <= l3; k3++)
            {
                d += real(conj(sirius::SHT::ylm_dot_rlm(l1, k1, m1)) *
                          sirius::SHT::ylm_dot_rlm(l2, k2, m2) *
                          sirius::SHT::ylm_dot_rlm(l3, k3, m3)) * sirius::SHT::gaunt_ylm(l1, l2, l3, k1, k2, k3);
            }
        }
    }
    //double d = sirius::SHT::gaunt<double>(l1, l2, l3, m1, m2, m3);

    *coeff__ = d * s;
}

void sirius_ylmr2_(int32_t* lmmax__, int32_t* nr__, double* vr__, double* rlm__)
{
    mdarray<double, 2> rlm(rlm__, *nr__, *lmmax__);
    mdarray<double, 2> vr(vr__, 3, *nr__);

    int lmax = Utils::lmax_by_lmmax(*lmmax__);

    std::vector<int> idxlm(*lmmax__);
    std::vector<int> phase(*lmmax__, 1);
    int lm = 0;
    for (int l = 0; l <= lmax; l++)
    {
        idxlm[lm++] = Utils::lm_by_l_m(l, 0);
        for (int m = 1; m <= l; m++)
        {
            idxlm[lm++] = Utils::lm_by_l_m(l, m);
            idxlm[lm] = Utils::lm_by_l_m(l, -m);
            if (m % 2 == 0) phase[lm] = -1;
            lm++;
        }
    }

    std::vector<double> rlm_tmp(*lmmax__);
    for (int i = 0; i < *nr__; i++)
    {
        auto vs = sirius::SHT::spherical_coordinates(vector3d<double>(vr(0, i), vr(1, i), vr(2, i)));
        sirius::SHT::spherical_harmonics(lmax, vs[1], vs[2], &rlm_tmp[0]);
        for (int lm = 0; lm < *lmmax__; lm++) rlm(i, lm) = rlm_tmp[idxlm[lm]] * phase[lm];
    }
}

//void sirius_get_vloc_(int32_t* size__, double* vloc__)
//{
//    TERMINATE("fix this");
//    //if (!sim_ctx) return;
//
//    //auto fft_coarse = sim_ctx->fft_coarse();
//    //if (*size__ != fft_coarse->size())
//    //{
//    //    TERMINATE("wrong size of coarse FFT mesh");
//    //}
//
//    ///* map effective potential to a corase grid */
//    //std::vector<double> veff_it_coarse(fft_coarse->size());
//    //std::vector<double_complex> veff_pw_coarse(fft_coarse->num_gvec());
//
//    ///* take only first num_gvec_coarse plane-wave harmonics; this is enough to apply V_eff to \Psi */
//    //for (int igc = 0; igc < fft_coarse->num_gvec(); igc++)
//    //{
//    //    int ig = sim_ctx->fft().gvec_index(fft_coarse->gvec(igc));
//    //    veff_pw_coarse[igc] = potential->effective_potential()->f_pw(ig);
//    //}
//    //fft_coarse->input(fft_coarse->num_gvec(), fft_coarse->index_map(), &veff_pw_coarse[0]);
//    //fft_coarse->transform(1);
//    //fft_coarse->output(vloc__);
//    //for (int i = 0; i < fft_coarse->size(); i++) vloc__[i] *= 2; // convert to Ry
//}

void sirius_get_q_operator_matrix(ftn_int*    iat__,
                                  ftn_double* q_mtrx__,
                                  ftn_int*    ld__)
{
    mdarray<double, 2> q_mtrx(q_mtrx__, *ld__, *ld__);

    auto& atom_type = sim_ctx->unit_cell().atom_type(*iat__ - 1);

    int nbf = atom_type.mt_basis_size();

    /* index of Rlm of QE */
    auto idx_Rlm = [](int lm)
    {
        int l = static_cast<int>(std::sqrt(static_cast<double>(lm) + 1e-12));
        int m = lm - l * l - l;
        return (m > 0) ? 2 * m - 1 : -2 * m;
    };

    std::vector<int> idx_map(nbf);
    for (int xi = 0; xi < nbf; xi++) {
        int lm     = atom_type.indexb(xi).lm;
        int idxrf  = atom_type.indexb(xi).idxrf;
        idx_map[xi] = atom_type.indexb().index_by_idxrf(idxrf) + idx_Rlm(lm);
    }
    
    q_mtrx.zero();
    
    if (atom_type.pp_desc().augment) {
        for (int xi1 = 0; xi1 < nbf; xi1++) {
            for (int xi2 = 0; xi2 < nbf; xi2++) {
                q_mtrx(idx_map[xi1], idx_map[xi2]) = sim_ctx->augmentation_op(*iat__ - 1).q_mtrx(xi1, xi2);
            }
        }
    }

    //mdarray<double_complex, 2> sirius_Ylm_to_QE_Rlm(nbf, nbf);
    //sirius_Ylm_to_QE_Rlm.zero();

    //for (int idxrf = 0; idxrf < atom_type.mt_radial_basis_size(); idxrf++) {
    //    int l      = atom_type.indexr(idxrf).l;
    //    int offset = atom_type.indexb().index_by_idxrf(idxrf); 
    //    for (int m1 = -l; m1 <= l; m1++) { // this runs over Ylm index of sirius
    //        for (int m2 = -l; m2 <= l; m2++) { // this runs over Rlm index of sirius
    //            int i{0}; // index of QE Rlm
    //            if (m2 > 0) {
    //                i = m2 * 2 - 1;
    //            }
    //            if (m2 < 0) {
    //                i = (-m2) * 2;
    //            }
    //            double phase{1};
    //            if (m2 < 0 && (-m2) % 2 == 0) {
    //                phase = -1;
    //            }
    //            sirius_Ylm_to_QE_Rlm(offset + i, offset + l + m1) = sirius::SHT::rlm_dot_ylm(l, m2, m1) * phase;
    //        }
    //    }

    //}


    //mdarray<double_complex, 2> z1(nbf, nbf);
    //mdarray<double_complex, 2> z2(nbf, nbf);

    //for (int xi1 = 0; xi1 < nbf; xi1++) {
    //    for (int xi2 = 0; xi2 < nbf; xi2++) {
    //        z1(xi1, xi2) = atom_type.pp_desc().q_mtrx(xi1, xi2);
    //    }
    //}
    //linalg<CPU>::gemm(0, 2, nbf, nbf, nbf, double_complex(1, 0), z1, sirius_Ylm_to_QE_Rlm, double_complex(0, 0), z2);
    //linalg<CPU>::gemm(0, 0, nbf, nbf, nbf, double_complex(1, 0), sirius_Ylm_to_QE_Rlm, z2, double_complex(0, 0), z1);

    //for (int xi1 = 0; xi1 < nbf; xi1++)
    //{
    //    for (int xi2 = 0; xi2 < nbf; xi2++)
    //    {
    //        //== double diff = std::abs(q_mtrx(xi1, xi2) - real(z1(xi1, xi2)));
    //        //== printf("itype=%i, xi1,xi2=%i %i, q_diff=%18.12f\n", *itype__ - 1, xi1, xi2, diff);
    //        q_mtrx(xi1, xi2) = real(z1(xi1, xi2));
    //    }
    //}
}

void sirius_get_d_operator_matrix(ftn_int*    ia__,
                                  ftn_double* d_mtrx__,
                                  ftn_int*    ld__)
{
    mdarray<double, 2> d_mtrx(d_mtrx__, *ld__, *ld__);

    auto& atom = sim_ctx->unit_cell().atom(*ia__ - 1);

    int nbf = atom.mt_basis_size();

    /* index of Rlm of QE */
    auto idx_Rlm = [](int lm) // TODO: move duplicated code to a separate inline function
    {
        int l = static_cast<int>(std::sqrt(static_cast<double>(lm) + 1e-12));
        int m = lm - l * l - l;
        return (m > 0) ? 2 * m - 1 : -2 * m;
    };

    std::vector<int> idx_map(nbf);
    for (int xi = 0; xi < nbf; xi++) {
        int lm      = atom.type().indexb(xi).lm;
        int idxrf   = atom.type().indexb(xi).idxrf;
        idx_map[xi] = atom.type().indexb().index_by_idxrf(idxrf) + idx_Rlm(lm);
    }
    
    d_mtrx.zero();

    for (int xi1 = 0; xi1 < nbf; xi1++) {
        for (int xi2 = 0; xi2 < nbf; xi2++) {
            d_mtrx(idx_map[xi1], idx_map[xi2]) = atom.d_mtrx(xi1, xi2, 0);
        }
    }

    //mdarray<double_complex, 2> sirius_Ylm_to_QE_Rlm(nbf, nbf);
    //sirius_Ylm_to_QE_Rlm.zero();

    //for (int idxrf = 0; idxrf < atom.type().mt_radial_basis_size(); idxrf++) {
    //    int l      = atom.type().indexr(idxrf).l;
    //    int offset = atom.type().indexb().index_by_idxrf(idxrf);

    //    for (int m1 = -l; m1 <= l; m1++) { // this runs over Ylm index of sirius
    //        for (int m2 = -l; m2 <= l; m2++) { // this runs over Rlm index of sirius
    //            int i{0}; // index of QE Rlm
    //            if (m2 > 0) {
    //                i = m2 * 2 - 1;
    //            }
    //            if (m2 < 0) {
    //                i = (-m2) * 2;
    //            }
    //            double phase{1};
    //            if (m2 < 0 && (-m2) % 2 == 0) {
    //                phase = -1;
    //            }
    //            sirius_Ylm_to_QE_Rlm(offset + i, offset + l + m1) = sirius::SHT::rlm_dot_ylm(l, m2, m1) * phase;
    //        }
    //    }
    //}

    //mdarray<double_complex, 2> z1(nbf, nbf);
    //mdarray<double_complex, 2> z2(nbf, nbf);

    //for (int xi1 = 0; xi1 < nbf; xi1++) {
    //    for (int xi2 = 0; xi2 < nbf; xi2++) {
    //        z1(xi1, xi2) = atom.d_mtrx(xi1, xi2, 0);
    //    }
    //}
    //linalg<CPU>::gemm(0, 2, nbf, nbf, nbf, double_complex(1, 0), z1, sirius_Ylm_to_QE_Rlm, double_complex(0, 0), z2);
    //linalg<CPU>::gemm(0, 0, nbf, nbf, nbf, double_complex(1, 0), sirius_Ylm_to_QE_Rlm, z2, double_complex(0, 0), z1);

    //for (int xi1 = 0; xi1 < nbf; xi1++) {
    //    for (int xi2 = 0; xi2 < nbf; xi2++) {
    //        //double diff = std::abs(d_mtrx(xi1, xi2) - real(z1(xi1, xi2) * 2.0));
    //        //if (diff > 1e-8)
    //        //{
    //        //    printf("ia=%2i, xi1,xi2=%2i %2i, D(QE)=%18.12f D(S)=%18.12f\n", *ia__ - 1, xi1, xi2, d_mtrx(xi1, xi2), real(z1(xi1, xi2)) * 2);
    //        //}
    //        d_mtrx(xi1, xi2) = std::real(z1(xi1, xi2)) * 2; // convert to Ry
    //    }
    //}
}

void sirius_set_d_operator_matrix(ftn_int*    ia__,
                                  ftn_double* d_mtrx__,
                                  ftn_int*    ld__)
{
    mdarray<double, 2> d_mtrx(d_mtrx__, *ld__, *ld__);

    auto& atom = sim_ctx->unit_cell().atom(*ia__ - 1);

    int nbf = atom.mt_basis_size();

    /* index of Rlm of QE */
    auto idx_Rlm = [](int lm) // TODO: move duplicated code to a separate inline function
    {
        int l = static_cast<int>(std::sqrt(static_cast<double>(lm) + 1e-12));
        int m = lm - l * l - l;
        return (m > 0) ? 2 * m - 1 : -2 * m;
    };

    std::vector<int> idx_map(nbf);
    for (int xi = 0; xi < nbf; xi++) {
        int lm      = atom.type().indexb(xi).lm;
        int idxrf   = atom.type().indexb(xi).idxrf;
        idx_map[xi] = atom.type().indexb().index_by_idxrf(idxrf) + idx_Rlm(lm);
    }
    
    for (int xi1 = 0; xi1 < nbf; xi1++) {
        for (int xi2 = 0; xi2 < nbf; xi2++) {
            atom.d_mtrx(xi1, xi2, 0) = d_mtrx(idx_map[xi1], idx_map[xi2]);
        }
    }
}

void sirius_get_q_pw_(int32_t* iat__, int32_t* num_gvec__, double_complex* q_pw__)
{
    TERMINATE("fix this");
    //if (*num_gvec__ != sim_ctx->fft().num_gvec())
    //{
    //    TERMINATE("wrong number of G-vectors");
    //}

    //auto atom_type = sim_ctx->unit_cell().atom_type(*iat__ - 1);

    //int nbf = atom_type.mt_basis_size();

    //mdarray<double_complex, 3> q_pw(q_pw__, nbf, nbf, *num_gvec__);

    //mdarray<double_complex, 2> sirius_Ylm_to_QE_Rlm(nbf, nbf);
    //sirius_Ylm_to_QE_Rlm.zero();

    //for (int idxrf = 0; idxrf < atom_type.mt_radial_basis_size(); idxrf++)
    //{
    //    int l = atom_type.indexr(idxrf).l;
    //    int offset = atom_type.indexb().index_by_idxrf(idxrf);

    //    for (int m1 = -l; m1 <= l; m1++) // this runs over Ylm index of sirius
    //    {
    //        for (int m2 = -l; m2 <= l; m2++) // this runs over Rlm index of sirius
    //        {
    //            int i; // index of QE Rlm
    //            if (m2 == 0) i = 0;
    //            if (m2 > 0) i = m2 * 2 - 1;
    //            if (m2 < 0) i = (-m2) * 2;
    //            double phase = 1;
    //            if (m2 < 0 && (-m2) % 2 == 0) phase = -1;
    //            sirius_Ylm_to_QE_Rlm(offset + i, offset + l + m1) = sirius::SHT::rlm_dot_ylm(l, m2, m1) * phase;
    //        }
    //    }

    //}

    //mdarray<double_complex, 2> z1(nbf, nbf);
    //mdarray<double_complex, 2> z2(nbf, nbf);

    //for (int ig = 0; ig < *num_gvec__; ig++)
    //{
    //    for (int xi2 = 0; xi2 < nbf; xi2++)
    //    {
    //        for (int xi1 = 0; xi1 <= xi2; xi1++)
    //        {
    //            int idx12 = xi2 * (xi2 + 1) / 2 + xi1;

    //            z1(xi1, xi2) = atom_type.pp_desc().q_pw(ig, idx12);
    //            z1(xi2, xi1) = conj(z1(xi1, xi2));
    //        }
    //    }

    //    linalg<CPU>::gemm(0, 2, nbf, nbf, nbf, double_complex(1, 0), z1, sirius_Ylm_to_QE_Rlm, double_complex(0, 0), z2);
    //    linalg<CPU>::gemm(0, 0, nbf, nbf, nbf, double_complex(1, 0), sirius_Ylm_to_QE_Rlm, z2, double_complex(0, 0), z1);

    //    for (int xi2 = 0; xi2 < nbf; xi2++)
    //    {
    //        for (int xi1 = 0; xi1 < nbf; xi1++)
    //        {
    //            q_pw(xi1, xi2, ig) = z1(xi1, xi2);
    //        }
    //    }
    //}
}

void sirius_get_fv_states_(int32_t* kset_id__, int32_t* ik__, int32_t* nfv__, int32_t* ngk__, int32_t* gvec_of_k__,
                           double_complex* fv_states__, int32_t* ld__)
{
    TERMINATE("fix this");
    //auto kset = kset_list[*kset_id__];
    //auto kp = (*kset)[*ik__ - 1];

    //if (*ngk__ != kp->num_gkvec())
    //{
    //    std::stringstream s;
    //    s << "wrong number of G+k vectors" << std::endl
    //      << "ik = " << *ik__ - 1 << std::endl
    //      << "ngk = " << *ngk__ << std::endl
    //      << "kp->num_gkvec() = " << kp->num_gkvec();
    //    TERMINATE(s);
    //}
    //if (*nfv__ != sim_ctx->num_fv_states())
    //{
    //    TERMINATE("wrong number of first-variational states");
    //}
    //
    //mdarray<int, 2> gvec_of_k(gvec_of_k__, 3, *ngk__);
    //std::vector<int> igk_map(*ngk__);
    //for (int igk = 0; igk < kp->num_gkvec(); igk++)
    //{
    //    bool found = false;
    //    for (int i = 0; i < kp->num_gkvec(); i++)
    //    {
    //        int ig = kp->gvec_index(i);
    //        /* G-vector of sirius ordering */
    //        auto vg = sim_ctx->fft().gvec(ig);
    //        if (gvec_of_k(0, igk) == vg[0] &&
    //            gvec_of_k(1, igk) == vg[1] &&
    //            gvec_of_k(2, igk) == vg[2])
    //        {
    //            igk_map[igk] = i;
    //            found = true;
    //        }
    //    }
    //    if (!found)
    //    {
    //        TERMINATE("G-vector is not found");
    //    }
    //}

    //mdarray<double_complex, 2> fv_states(fv_states__, *ld__, *nfv__);

    //for (int i = 0; i < sim_ctx->num_fv_states(); i++)
    //{
    //    for (int igk = 0; igk < kp->num_gkvec(); igk++)
    //    {
    //        fv_states(igk, i) = kp->fv_states()(igk_map[igk], i);
    //    }
    //}
}

//void FORTRAN(sirius_scf_loop)()
//{
//    dft_ground_state->find(1e-6, 1e-6, 20);
//}

//void FORTRAN(sirius_potential_checksum)()
//{
//    potential->checksum();
//}

void sirius_set_mpi_grid_dims(int *ndims__, int* dims__)
{
    assert(*ndims__ > 0);
    std::vector<int> dims(*ndims__);
    for (int i = 0; i < *ndims__; i++) dims[i] = dims__[i];
    sim_ctx->set_mpi_grid_dims(dims);
}

void sirius_set_atom_type_paw_data(char* label__,
                                   double* ae_wfc_rf__,
                                   double* ps_wfc_rf__,
                                   int32_t* num_wfc__,
                                   int32_t* ld__,
                                   int32_t* cutoff_radius_index__,
                                   double* core_energy__,
                                   double* ae_core_charge__,
                                   int32_t* num_ae_core_charge__,
                                   double* occupations__,
                                   int32_t* num_occ__)
{
    auto& type = sim_ctx->unit_cell().atom_type(std::string(label__));

    auto& pp_desc = type.pp_desc();

    if (*num_wfc__ != type.pp_desc().num_beta_radial_functions) {
        TERMINATE("PAW error: different number of projectors and wave functions!");
    }

    if (*ld__ != type.num_mt_points()) {
        TERMINATE("PAW error: different number of grid points of projectors and wave functions!");
    }

    if (*num_ae_core_charge__ != type.num_mt_points()) {
        TERMINATE("PAW error: different number of grid points of core charge and wave functions!");
    }

    if (*num_occ__ != type.pp_desc().num_beta_radial_functions) {
        TERMINATE("PAW error: different number of occupations and wave functions!");
    }

    // we load PAW, so we set is_paw to true
    pp_desc.is_paw = true;

    // load parameters
    pp_desc.core_energy = (*core_energy__) * 0.5; // convert Ry to Ha

    pp_desc.cutoff_radius_index = *cutoff_radius_index__;

    // load ae and ps wave functions
    mdarray<double, 2> aewfcs(ae_wfc_rf__, type.num_mt_points(), type.pp_desc().num_beta_radial_functions);
    mdarray<double, 2> pswfcs(ps_wfc_rf__, type.num_mt_points(), type.pp_desc().num_beta_radial_functions);

    pp_desc.all_elec_wfc = mdarray<double, 2>(type.num_mt_points(), type.pp_desc().num_beta_radial_functions);
    pp_desc.pseudo_wfc   = mdarray<double, 2>(type.num_mt_points(), type.pp_desc().num_beta_radial_functions);

    pp_desc.all_elec_wfc.zero();
    pp_desc.pseudo_wfc.zero();

    aewfcs >> pp_desc.all_elec_wfc;
    pswfcs >> pp_desc.pseudo_wfc;

    for (int i = 0; i < type.pp_desc().num_beta_radial_functions; i++) {
        std::memcpy(&pp_desc.all_elec_wfc(0, i), &aewfcs(0, i), pp_desc.cutoff_radius_index * sizeof(double));
        std::memcpy(&pp_desc.pseudo_wfc(0, i),   &pswfcs(0, i), pp_desc.cutoff_radius_index * sizeof(double));
    }

    // read ae core charge
    pp_desc.all_elec_core_charge.resize(type.num_mt_points());

    std::memcpy(pp_desc.all_elec_core_charge.data(), ae_core_charge__, type.num_mt_points() * sizeof(double));

    // read occupations
    pp_desc.occupations.resize(type.pp_desc().num_beta_radial_functions);

    std::memcpy(pp_desc.occupations.data(), occupations__, type.pp_desc().num_beta_radial_functions * sizeof(double));
}

void sirius_get_paw_total_energy(double* tot_en__)
{
    if (potential) {
        *tot_en__ = potential->PAW_total_energy();
    } else {
        TERMINATE("Potential is not allocated");
    }
}

void sirius_get_paw_one_elec_energy(double* one_elec_en__)
{
    if (potential) {
        *one_elec_en__ = potential->PAW_one_elec_energy();
    } else {
        TERMINATE("Potential is not allocated");
    }
}

void sirius_reduce_coordinates(ftn_double* coord__,
                               ftn_double* reduced_coord__, 
                               ftn_int* T__)
{
    vector3d<double> coord(coord__[0], coord__[1], coord__[2]);
    auto result = reduce_coordinates(coord);
    for (int x: {0, 1, 2}) {
        reduced_coord__[x] = result.first[x];
        T__[x] = result.second[x];
    }
}

void sirius_fderiv(ftn_int* m__,
                   ftn_int* np__,
                   ftn_double* x__,
                   ftn_double* f__,
                   ftn_double* g__)
{
    int np = *np__;
    sirius::Radial_grid_ext<double> rgrid(np, x__);
    sirius::Spline<double> s(rgrid);
    for (int i = 0; i < np; i++) {
        s[i] = f__[i];
    }
    s.interpolate();
    switch (*m__) {
        case -1: {
            std::vector<double> g(np);
            s.integrate(g, 0);
            for (int i = 0; i < np; i++) {
                g__[i] = g[i];
            }
            return;
        }
        default: {
             TERMINATE_NOT_IMPLEMENTED;
        }
    }
}

void sirius_integrate_(ftn_int* m__,
                      ftn_int* np__,
                      ftn_double* x__,
                      ftn_double* f__,
                      ftn_double* result__)
{
    int np = *np__;
    sirius::Radial_grid_ext<double> rgrid(np, x__);
    sirius::Spline<double> s(rgrid);
    for (int i = 0; i < np; i++) {
        s[i] = f__[i];
    }
    s.interpolate();
    *result__ = s.integrate(*m__);
}


void sirius_get_wave_functions(ftn_int* kset_id__,
                               ftn_int* ik__,
                               ftn_int* npw__,
                               ftn_int* gvec_k__,
                               ftn_double_complex* evc__,
                               ftn_int* ld__)
{
    auto kset = kset_list[*kset_id__];
    auto kp = (*kset)[*ik__ - 1];

    mdarray<int, 2> gvec_k(gvec_k__, 3, *npw__);
    mdarray<double_complex, 2> evc(evc__, *ld__, sim_ctx->num_bands());
    evc.zero();
    
    std::vector<double_complex> wf_tmp(kp->num_gkvec());
    int gkvec_count = kp->gkvec().gvec_count(kp->comm().rank());
    int gkvec_offset = kp->gkvec().gvec_offset(kp->comm().rank());

    for (int i = 0; i < sim_ctx->num_bands(); i++) {
        std::memcpy(&wf_tmp[gkvec_offset], &kp->spinor_wave_functions(0).pw_coeffs().prime(0, i), gkvec_count * sizeof(double_complex));
        kp->comm().allgather(wf_tmp.data(), gkvec_offset, gkvec_count);

        for (int ig = 0; ig < *npw__; ig++) {
            int ig1 = kp->gkvec().index_by_gvec({gvec_k(0, ig), gvec_k(1, ig), gvec_k(2, ig)});
            if (ig1 < 0 || ig1 >= kp->num_gkvec()) {
                TERMINATE("G-vector is out of range");
            }
            evc(ig, i) = wf_tmp[ig1];
        }
    }
}

void sirius_get_beta_projectors(ftn_int* kset_id__,
                                ftn_int* ik__,
                                ftn_int* npw__,
                                ftn_int* gvec_k__,
                                ftn_double_complex* vkb__,
                                ftn_int* ld__,
                                ftn_int* nkb__)
{
    PROFILE("sirius_api::sirius_get_beta_projectors");

    STOP();

    //if (*nkb__ != sim_ctx->unit_cell().mt_lo_basis_size()) {
    //    TERMINATE("wrong number of beta-projectors");
    //}

    //auto kset = kset_list[*kset_id__];
    //auto kp = (*kset)[*ik__ - 1];
    //auto& beta_gk = kp->beta_projectors().beta_gk_total();

    //mdarray<int, 2> gvec_k(gvec_k__, 3, *npw__);
    //mdarray<double_complex, 2> vkb(vkb__, *ld__, *nkb__);
    //vkb.zero();
    //
    //std::vector<double_complex> wf_tmp(kp->num_gkvec());
    //int gkvec_count = kp->gkvec().gvec_count(kp->comm().rank());
    //int gkvec_offset = kp->gkvec().gvec_offset(kp->comm().rank());

    //for (int ia = 0; ia < sim_ctx->unit_cell().num_atoms(); ia++) {
    //    auto& atom = sim_ctx->unit_cell().atom(ia);
    //    
    //    int nbf = atom.mt_basis_size();

    //    /* index of Rlm of QE */
    //    auto idx_Rlm = [](int lm)
    //    {
    //        int l = static_cast<int>(std::sqrt(static_cast<double>(lm) + 1e-12));
    //        int m = lm - l * l - l;
    //        return (m > 0) ? 2 * m - 1 : -2 * m;
    //    };

    //    for (int xi = 0; xi < nbf; xi++) {
    //        int lm     = atom.type().indexb(xi).lm;
    //        int idxrf  = atom.type().indexb(xi).idxrf;
    //        /* position in QE array */
    //        int xi1 = atom.type().indexb().index_by_idxrf(idxrf) + idx_Rlm(lm);

    //        std::memcpy(&wf_tmp[gkvec_offset], &beta_gk(0, atom.offset_lo() + xi), gkvec_count * sizeof(double_complex));
    //        kp->comm().allgather(wf_tmp.data(), gkvec_offset, gkvec_count);

    //        for (int ig = 0; ig < *npw__; ig++) {
    //            int ig1 = kp->gkvec().index_by_gvec({gvec_k(0, ig), gvec_k(1, ig), gvec_k(2, ig)});
    //            if (ig1 < 0 || ig1 >= kp->num_gkvec()) {
    //                TERMINATE("G-vector is out of range");
    //            }
    //            vkb(ig, atom.offset_lo() + xi1) = wf_tmp[ig1];
    //        }
    //    }
    //}
}

void sirius_get_beta_projectors_by_kp(ftn_int* kset_id__,
                                      ftn_double* vk__,
                                      ftn_int* npw__,
                                      ftn_int* gvec_k__,
                                      ftn_double_complex* vkb__,
                                      ftn_int* ld__,
                                      ftn_int* nkb__)
{
    PROFILE("sirius_api::sirius_get_beta_projectors_by_kp");

    vector3d<double> vk(vk__[0], vk__[1], vk__[2]);

    auto kset = kset_list[*kset_id__];
    for (int ikloc = 0; ikloc < kset->spl_num_kpoints().local_size(); ikloc++) {
        int ik = kset->spl_num_kpoints(ikloc);
        auto kp = (*kset)[ik];
        if ((kp->vk() - vk).length() < 1e-12) {
            sirius_get_beta_projectors(kset_id__, &ik, npw__, gvec_k__, vkb__, ld__, nkb__);
            return;
        }
    }
    std::stringstream s;
    s << "k-point " << vk << " is not found";
    TERMINATE(s);
}

void sirius_get_density_matrix(ftn_int*            ia__,
                               ftn_double_complex* dm__,
                               ftn_int*            ld__)
{
    mdarray<double_complex, 2> dm(dm__, *ld__, *ld__);
    int nbf = sim_ctx->unit_cell().atom(*ia__ - 1).mt_basis_size();
    for (int i = 0; i < nbf; i++) {
        for (int j = 0; j < nbf; j++) {
            dm(i, j) = density->density_matrix()(i, j, 0, *ia__ - 1);
        }
    }
}

void sirius_calc_forces(ftn_int* kset_id__)
{
    auto& kset = *kset_list[*kset_id__];
    forces = std::unique_ptr<sirius::Forces_PS>(new sirius::Forces_PS(*sim_ctx, *density, *potential, kset));
}

void sirius_get_forces(ftn_char label__, ftn_double* forces__)
{
    std::string label(label__);

    auto get_forces = [&](const mdarray<double,2>& sirius_forces__)
        {
            #pragma omp parallel for
            for(size_t i = 0; i < sirius_forces__.size(); i++)
            {
                forces__[i] = sirius_forces__[i];
            }
        };

    if (label == "vloc") {
        get_forces( forces->local_forces() );
    } else if (label == "nlcc") {
        get_forces( forces->nlcc_forces() );
    } else if (label == "ewald") {
        get_forces( forces->ewald_forces() );
    } else if (label == "nl") {
        get_forces( forces->nonlocal_forces() );
    } else if (label == "us") {
        get_forces( forces->ultrasoft_forces() );
    } else if (label == "tot") {
        get_forces( forces->total_forces() );
    } else {
        TERMINATE("wrong label");
    }
}

void sirius_set_verbosity(ftn_int* level__)
{
    sim_ctx->set_verbosity(*level__);
}

void sirius_generate_d_operator_matrix()
{
    potential->generate_D_operator_matrix();
}

/// Set the plane-wave expansion coefficients of a particular function.
/** \param [in] label     label of the target function
 *  \param [in] pw_coeffs array of plane-wave coefficients
 *  \param [in] ngv       number of G-vectors
 *  \param [in] gvl       G-vectors in lattice coordinates (Miller indices)
 *  \param [in] comm      MPI communicator used in distribution of G-vectors
 */
void sirius_set_pw_coeffs(ftn_char label__,
                          double_complex* pw_coeffs__,
                          ftn_int* ngv__,
                          ftn_int* gvl__, 
                          ftn_int* comm__)
{
    std::string label(label__);

    if (sim_ctx->full_potential()) {
        if (label == "veff") {
            potential->set_veff_pw(pw_coeffs__);
        } else if (label == "rm_inv") {
            potential->set_rm_inv_pw(pw_coeffs__);
        } else if (label == "rm2_inv") {
            potential->set_rm2_inv_pw(pw_coeffs__);
        } else {
            TERMINATE("wrong label");
        }
    } else {
        assert(ngv__ != NULL);
        assert(gvl__ != NULL);
        assert(comm__ != NULL);

        Communicator comm(MPI_Comm_f2c(*comm__));
        mdarray<int, 2> gvec(gvl__, 3, *ngv__);

        std::vector<double_complex> v(sim_ctx->gvec().num_gvec(), 0);
        for (int i = 0; i < *ngv__; i++) {
            vector3d<int> G(gvec(0, i), gvec(1, i), gvec(2, i));
            int ig = sim_ctx->gvec().index_by_gvec(G);
            if (ig >= 0) {
                v[ig] = pw_coeffs__[i];
            }
        }
        comm.allreduce(v.data(), sim_ctx->gvec().num_gvec());

        if (label == "rho") {
            density->rho()->scatter_f_pw(v);
            density->rho()->fft_transform(1);
        } else if (label == "veff") {
            potential->effective_potential()->scatter_f_pw(v);
            potential->effective_potential()->fft_transform(1);
        } else if (label == "vxc") {
            potential->xc_potential()->scatter_f_pw(v);
            potential->xc_potential()->fft_transform(1);
        } else {
            TERMINATE("wrong label");
        }
    }
}

void sirius_get_pw_coeffs(ftn_char label__,
                          double_complex* pw_coeffs__,
                          ftn_int* ngv__,
                          ftn_int* gvl__, 
                          ftn_int* comm__)
{
    std::string label(label__);
    if (sim_ctx->full_potential()) {
        STOP();
    } else {
        assert(ngv__ != NULL);
        assert(gvl__ != NULL);
        assert(comm__ != NULL);

        Communicator comm(MPI_Comm_f2c(*comm__));
        mdarray<int, 2> gvec(gvl__, 3, *ngv__);

        std::vector<double_complex> v;

        if (label == "rho") {
            v = density->rho()->gather_f_pw();
        } else if (label == "veff") {
            v = potential->effective_potential()->gather_f_pw();
        } else {
            TERMINATE("wrong label");
        }

        for (int i = 0; i < *ngv__; i++) {
            vector3d<int> G(gvec(0, i), gvec(1, i), gvec(2, i));
            
            bool is_inverse{false};
            int ig = sim_ctx->gvec().index_by_gvec(G);
            if (ig == -1 && sim_ctx->gvec().reduced()) {
                ig = sim_ctx->gvec().index_by_gvec(G * (-1));
                is_inverse = true;
            }
            if (ig == -1) {
                std::stringstream s;
                auto gvc = sim_ctx->unit_cell().reciprocal_lattice_vectors() * vector3d<double>(G[0], G[1], G[2]);
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

void sirius_calculate_stress_tensor(ftn_int* kset_id__)
{
    auto& kset = *kset_list[*kset_id__];
    stress_tensor = std::unique_ptr<sirius::Stress>(new sirius::Stress(*sim_ctx, kset, *density, *potential));
}

void sirius_get_stress_tensor(ftn_char label__, ftn_double* stress_tensor__)
{
    std::string label(label__);
    matrix3d<double> s;
    if (label == "vloc") {
        s = stress_tensor->stress_vloc();
    } else if (label == "har") {
        s = stress_tensor->stress_har();
    } else if (label == "ewald") {
        s = stress_tensor->stress_ewald();
    } else if (label == "kin") {
        s = stress_tensor->stress_kin();
    } else if (label == "nl") {
        s = stress_tensor->stress_nl();
    } else {
        TERMINATE("wrong label");
    }
    for (int mu = 0; mu < 3; mu++) {
        for (int nu = 0; nu < 3; nu++) {
            stress_tensor__[nu + mu * 3] = s(mu, nu);
        }
    }
}

} // extern "C"
