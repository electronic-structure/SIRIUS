// Copyright (c) 2013-2015 Anton Kozhevnikov, Thomas Schulthess
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

/// Parameters of the simulation.
sirius::Simulation_parameters* sim_param = nullptr;

/// Simulation context.
sirius::Simulation_context* sim_ctx = nullptr;

/// Pointer to Density class, implicitly used by Fortran side.
sirius::Density* density = nullptr;

/// Pointer to Potential class, implicitly used by Fortran side.
sirius::Potential* potential = nullptr;

/// List of pointers to the sets of k-points.
std::vector<sirius::K_set*> kset_list;

/// DFT ground state wrapper
sirius::DFT_ground_state* dft_ground_state = nullptr;

/// Charge density and magnetization mixer
sirius::Mixer<double>* mixer_rho = nullptr;

/// Potential and magnetic field mixer
sirius::Mixer<double>* mixer_pot = nullptr;

BLACS_grid* blacs_grid = nullptr;

std::map<std::string, runtime::Timer*> ftimers;

extern "C" 
{

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
void sirius_initialize(int32_t* call_mpi_init__)
{
    bool call_mpi_init = (*call_mpi_init__ != 0) ? true : false; 
    sirius::initialize(call_mpi_init);
}

void sirius_create_global_parameters()
{
    sim_param = (Utils::file_exists("sirius.json")) ? new sirius::Simulation_parameters("sirius.json")
                                                    : new sirius::Simulation_parameters();
}

void sirius_create_simulation_context()
{
    sim_ctx = new sirius::Simulation_context(*sim_param, mpi_comm_world());
}

void sirius_set_lmax_apw(int32_t* lmax_apw__)
{
    sim_param->set_lmax_apw(*lmax_apw__);
}

void sirius_set_lmax_pot(int32_t* lmax_pot__)
{
    sim_param->set_lmax_pot(*lmax_pot__);
}

void sirius_set_lmax_rho(int32_t* lmax_rho__)
{
    sim_param->set_lmax_rho(*lmax_rho__);
}

void sirius_set_num_mag_dims(int32_t* num_mag_dims__)
{
    sim_param->set_num_mag_dims(*num_mag_dims__);
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
    PROFILE();
    sim_ctx->unit_cell().set_lattice_vectors(a1__, a2__, a3__);
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
    PROFILE();
    sim_param->set_pw_cutoff(*pw_cutoff__);
}

void sirius_set_gk_cutoff(double* gk_cutoff__)
{
    PROFILE();
    sim_param->set_gk_cutoff(*gk_cutoff__);
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
    PROFILE();
    sim_ctx->unit_cell().set_auto_rmt(*auto_rmt__);
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
void sirius_add_atom_type(char const* label__,
                          char const* fname__)
{
    PROFILE();
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
    PROFILE();
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

void sirius_set_free_atom_potential(char const* label__,
                                    int32_t const* num_points__,
                                    double const* vs__)
{
    auto& type = sim_ctx->unit_cell().atom_type(std::string(label__));
    type.set_free_atom_potential(*num_points__, vs__);
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
    PROFILE();
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
    PROFILE();
    if (vector_field__ != NULL)
    {
        sim_ctx->unit_cell().add_atom(std::string(label__),
                                      vector3d<double>(position__[0], position__[1], position__[2]),
                                      vector3d<double>(vector_field__[0], vector_field__[1], vector_field__[2]));
    }
    else
    {
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
    PROFILE();
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
    PROFILE();
    sim_param->set_aw_cutoff(*aw_cutoff__);
}

void sirius_add_xc_functional(char const* name__)
{
    assert(name__ != NULL);
    sim_param->add_xc_functional(name__);
}

void sirius_set_esm_type(char const* name__)
{
    assert(name__ != NULL);
    sim_param->set_esm_type(name__);
}

/// Initialize the global variables.
/** The function must be called after setting up the lattice vectors, plane wave-cutoff, autormt flag and loading
 *  atom types and atoms into the unit cell.
 */
void sirius_global_initialize()
{
    PROFILE();

    sim_ctx->initialize();

    blacs_grid = new BLACS_grid(sim_ctx->mpi_grid().communicator(1 << _dim_row_ | 1 << _dim_col_),
                                sim_ctx->mpi_grid().dimension_size(_dim_row_),
                                sim_ctx->mpi_grid().dimension_size(_dim_col_));

}

/// Initialize the Density object.
/** \param [in] rhomt pointer to the muffin-tin part of the density
 *  \param [in] rhoit pointer to the interstitial part of the denssity
 *  \param [in] magmt pointer to the muffin-tin part of the magnetization
 *  \param [in] magit pointer to the interstitial part of the magnetization
 */
void sirius_density_initialize(double* rhoit__,
                               double* rhomt__,
                               double* magit__,
                               double* magmt__)
{
    PROFILE();
    density = new sirius::Density(*sim_ctx);
    density->set_charge_density_ptr(rhomt__, rhoit__);
    density->set_magnetization_ptr(magmt__, magit__);
}

/// Initialize the Potential object.
/** \param [in] veffmt pointer to the muffin-tin part of the effective potential
 *  \param [in] veffit pointer to the interstitial part of the effective potential
 *  \param [in] beffmt pointer to the muffin-tin part of effective magnetic field
 *  \param [in] beffit pointer to the interstitial part of the effective magnetic field
 */
void sirius_potential_initialize(double* veffit__,
                                 double* veffmt__,
                                 double* beffit__,
                                 double* beffmt__)
{
    PROFILE();
    potential = new sirius::Potential(*sim_ctx);
    potential->set_effective_potential_ptr(veffmt__, veffit__);
    potential->set_effective_magnetic_field_ptr(beffmt__, beffit__);
}

/// Get maximum number of muffin-tin radial points.
/** \param [out] max_num_mt_points maximum number of muffin-tin points */
void sirius_get_max_num_mt_points(int32_t* max_num_mt_points__)
{
    PROFILE();
    *max_num_mt_points__ = sim_ctx->unit_cell().max_num_mt_points();
}

/// Get number of muffin-tin radial points for a specific atom type.
/** \param [in] label unique label of atom type
 *  \param [out] num_mt_points number of muffin-tin points
 */
void sirius_get_num_mt_points(char* label__, int32_t* num_mt_points__)
{
    PROFILE();
    *num_mt_points__ = sim_ctx->unit_cell().atom_type(std::string(label__)).num_mt_points();
}

void sirius_get_mt_points(char* label__, double* mt_points__)
{
    PROFILE();
    auto& type = sim_ctx->unit_cell().atom_type(std::string(label__));
    for (int i = 0; i < type.num_mt_points(); i++) mt_points__[i] = type.radial_grid(i);
}

void sirius_get_num_fft_grid_points(int32_t* num_grid_points__)
{
    PROFILE();
    *num_grid_points__ = sim_ctx->fft().local_size();
}

void sirius_get_num_bands(int32_t* num_bands)
{
    PROFILE();
    *num_bands = sim_ctx->num_bands();
}

/// Get number of G-vectors within the plane-wave cutoff
void sirius_get_num_gvec(int32_t* num_gvec__)
{
    PROFILE();
    *num_gvec__ = sim_ctx->gvec().num_gvec();
}

/// Get sizes of FFT grid
void sirius_get_fft_grid_size(int32_t* grid_size__)
{
    PROFILE();
    grid_size__[0] = sim_ctx->fft().grid().size(0);
    grid_size__[1] = sim_ctx->fft().grid().size(1);
    grid_size__[2] = sim_ctx->fft().grid().size(2);
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
    PROFILE();
    assert((*d >= 1) && (*d <= 3));
    *lower = sim_ctx->fft().grid().limits(*d - 1).first;
    *upper = sim_ctx->fft().grid().limits(*d - 1).second;
}

/// Get mapping between G-vector index and FFT index
void sirius_get_fft_index(int32_t* fft_index__)
{
    TERMINATE("fix thix");
    //PROFILE();
    //memcpy(fft_index__, sim_ctx->gvec()->index_map(), sim_ctx->fft().size() * sizeof(int32_t));
    //for (int i = 0; i < sim_ctx->gvec()->size(); i++) fft_index__[i]++;
}

/// Get list of G-vectors in fractional corrdinates
void sirius_get_gvec(int32_t* gvec__)
{
    TERMINATE("fix thix");
    //PROFILE();
    //mdarray<int, 2> gvec(gvec__, 3, sim_ctx->fft().size());
    //for (int ig = 0; ig < sim_ctx->gvec().num_gvec(); ig++)
    //{
    //    vector3d<int> gv = sim_ctx->gvec()[ig];
    //    for (int x = 0; x < 3; x++) gvec(x, ig) = gv[x];
    //}
}

/// Get list of G-vectors in Cartesian coordinates
void sirius_get_gvec_cart(double* gvec_cart__)
{
    TERMINATE("fix thix");
    //PROFILE();
    //mdarray<double, 2> gvec_cart(gvec_cart__, 3, sim_ctx->fft().size());
    //for (int ig = 0; ig < sim_ctx->fft().size(); ig++)
    //{
    //    vector3d<double> gvc = sim_ctx->fft().gvec_cart(ig);
    //    for (int x = 0; x < 3; x++) gvec_cart(x, ig) = gvc[x];
    //}
}

/// Get lengh of G-vectors
void sirius_get_gvec_len(double* gvec_len__)
{
    TERMINATE("fix thix");
    
    //PROFILE();
    //for (int ig = 0; ig < sim_ctx->fft().size(); ig++) gvec_len__[ig] = sim_ctx->fft().gvec_len(ig);
}

void sirius_get_index_by_gvec(int32_t* index_by_gvec__)
{
    TERMINATE("fix thix");
    //PROFILE();
    //auto fft = sim_ctx->fft();
    //std::pair<int, int> d0 = fft->grid_limits(0);
    //std::pair<int, int> d1 = fft->grid_limits(1);
    //std::pair<int, int> d2 = fft->grid_limits(2);

    //mdarray<int, 3> index_by_gvec(index_by_gvec__, 
    //                              mdarray_index_descriptor(d0.first, d0.second), 
    //                              mdarray_index_descriptor(d1.first, d1.second), 
    //                              mdarray_index_descriptor(d2.first, d2.second));

    //for (int i0 = d0.first; i0 <= d0.second; i0++)
    //{
    //    for (int i1 = d1.first; i1 <= d1.second; i1++)
    //    {
    //        for (int i2 = d2.first; i2 <= d2.second; i2++)
    //        {
    //            index_by_gvec(i0, i1, i2) = fft->gvec_index(vector3d<int>(i0, i1, i2)) + 1;
    //        }
    //    }
    //}
}

/// Get Ylm spherical harmonics of G-vectors.
void sirius_get_gvec_ylm(double_complex* gvec_ylm__, int* ld__, int* lmax__)
{
    TERMINATE("fix this");

    //==PROFILE();
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
    //PROFILE();
    //mdarray<double_complex, 2> sfacg(sfacg__, sim_ctx->fft().num_gvec(), sim_ctx->unit_cell().num_atoms());
    //for (int ia = 0; ia < sim_ctx->unit_cell().num_atoms(); ia++)
    //{
    //    for (int ig = 0; ig < sim_ctx->fft().num_gvec(); ig++)
    //        sfacg(ig, ia) = sim_ctx->reciprocal_lattice()->gvec_phase_factor(ig, ia);
    //}
}

void sirius_get_step_function(double_complex* cfunig__, double* cfunir__)
{
    PROFILE();
    for (int i = 0; i < sim_ctx->fft().size(); i++)
    {
        cfunig__[i] = sim_ctx->step_function().theta_pw(i);
        cfunir__[i] = sim_ctx->step_function().theta_r(i);
    }
}

/// Get the total number of electrons
void sirius_get_num_electrons(double* num_electrons__)
{
    PROFILE();
    *num_electrons__ = sim_ctx->unit_cell().num_electrons();
}

/// Get the number of valence electrons
void sirius_get_num_valence_electrons(double* num_valence_electrons__)
{
    PROFILE();
    *num_valence_electrons__ = sim_ctx->unit_cell().num_valence_electrons();
}

/// Get the number of core electrons
void sirius_get_num_core_electrons(double* num_core_electrons__)
{
    PROFILE();
    *num_core_electrons__ = sim_ctx->unit_cell().num_core_electrons();
}

/// Clear global variables and destroy all objects
void sirius_clear(void)
{
    PROFILE();
    
    if (density != nullptr) 
    {
        delete density;
        density = nullptr;
    }
    if (potential != nullptr)
    {
        delete potential;
        potential = nullptr;
    }
    if (dft_ground_state != nullptr)
    {
        delete dft_ground_state;
        dft_ground_state = nullptr;
    }
    if (blacs_grid != nullptr)
    {
        delete blacs_grid;
        blacs_grid = nullptr;
    }
    for (int i = 0; i < (int)kset_list.size(); i++)
    {
        if (kset_list[i] != nullptr) 
        {
            delete kset_list[i];
            kset_list[i] = nullptr;
        }
    }
    if (sim_ctx != nullptr)
    {
        delete sim_ctx;
        sim_ctx = nullptr;
    }

    if (sim_param != nullptr)
    {
        delete sim_param;
        sim_param = nullptr;
    }
    kset_list.clear();
}

void sirius_generate_initial_density()
{
    PROFILE();
    density->initial_density();
}

void sirius_generate_effective_potential()
{
    PROFILE();
    dft_ground_state->generate_effective_potential();
}

void sirius_generate_density(int32_t* kset_id__)
{
    PROFILE();
    density->generate(*kset_list[*kset_id__]);
}

void sirius_generate_valence_density(int32_t* kset_id__)
{
    PROFILE();
    density->generate_valence(*kset_list[*kset_id__]);
}

void sirius_augment_density(int32_t* kset_id__)
{
    PROFILE();
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
    PROFILE();
    bool precompute = (*precompute__) ? true : false;
    kset_list[*kset_id__]->find_eigen_states(potential, precompute);
}

void sirius_find_band_occupancies(int32_t* kset_id__)
{
    PROFILE();
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
    PROFILE();
    int ik = *ik__ - 1;
    kset_list[*kset_id__]->set_band_occupancies(ik, band_occupancies__);
}

void sirius_get_band_energies(int32_t* kset_id__,
                              int32_t* ik__,
                              double* band_energies__)
{
    PROFILE();
    int ik = *ik__ - 1;
    kset_list[*kset_id__]->get_band_energies(ik, band_energies__);
}

void sirius_get_band_occupancies(int32_t* kset_id, int32_t* ik_, double* band_occupancies)
{
    PROFILE();
    int ik = *ik_ - 1;
    kset_list[*kset_id]->get_band_occupancies(ik, band_occupancies);
}

//** void FORTRAN(sirius_integrate_density)(void)
//** {
//**     density->integrate();
//** }

/*
    print info
*/
//== void FORTRAN(sirius_print_info)(void)
//== {
//==     PROFILE();
//==     sim_param->print_info();
//== }

void sirius_print_timers(void)
{
    PROFILE();
#ifdef __TIMER
    runtime::Timer::print();
#endif
}   

void sirius_start_timer(char const* name__)
{
    PROFILE();
    std::string name(name__);
    ftimers[name] = new runtime::Timer(name);
}

void sirius_stop_timer(char const* name__)
{
    PROFILE();
    std::string name(name__);
    if (ftimers.count(name)) delete ftimers[name];
}

void sirius_save_potential(void)
{
    PROFILE();
    potential->save();
}

void sirius_save_density(void)
{
    PROFILE();
    density->save();
}

void sirius_load_potential(void)
{
    PROFILE();
    potential->load();
}

//== void FORTRAN(sirius_save_wave_functions)(int32_t* kset_id)
//== {
//==     PROFILE();
//==     kset_list[*kset_id]->save_wave_functions();
//== }
//==     
//== void FORTRAN(sirius_load_wave_functions)(int32_t* kset_id)
//== {
//==     PROFILE();
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
//==         sim_param->get_coordinates<cartesian, reciprocal>(vf, vc);
//==         double length = Utils::vector_length(vc);
//==         total_path_length += length;
//==         segment_length.push_back(length);
//==     }
//== 
//==     std::vector<double> xaxis;
//== 
//==     sirius::K_set kset_(global_parameters);
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
//==     sim_param->solve_free_atoms();
//== 
//==     potential->update_atomic_potential();
//==     sim_param->generate_radial_functions();
//==     sim_param->generate_radial_integrals();
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
//==     if (sim_param->mpi_grid().root())
//==     {
//==         JSON_write jw("bands.json");
//==         jw.single("xaxis", xaxis);
//==         //** jw.single("Ef", sim_param->rti().energy_fermi);
//==         
//==         jw.single("xaxis_ticks", xaxis_ticks);
//==         jw.single("xaxis_tick_labels", xaxis_tick_labels);
//==         
//==         jw.begin_array("plot");
//==         std::vector<double> yvalues(kset_.num_kpoints());
//==         for (int i = 0; i < sim_param->num_bands(); i++)
//==         {
//==             jw.begin_set();
//==             for (int ik = 0; ik < kset_.num_kpoints(); ik++) yvalues[ik] = kset_[ik]->band_energy(i);
//==             jw.single("yvalues", yvalues);
//==             jw.end_set();
//==         }
//==         jw.end_array();
//== 
//==         //FILE* fout = fopen("bands.dat", "w");
//==         //for (int i = 0; i < sim_param->num_bands(); i++)
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
//    //FORTRAN(sirius_read_state)();
//
//    density->initial_density(1);
//
//    potential->generate_effective_potential(density->rho(), density->magnetization());
//
//    
//    // generate plane-wave coefficients of the potential in the interstitial region
//    sim_ctx->fft().input(potential->effective_potential()->f_it());
//    sim_ctx->fft().transform(-1);
//    sim_ctx->fft().output(sim_param->num_gvec(), sim_ctx->fft_index(), 
//                                   potential->effective_potential()->f_pw());
//
//    int N = 10000;
//    double* p = new double[N];
//    double* x = new double[N];
//
//    double vf1[] = {0.1, 0.1, 0.1};
//    double vf2[] = {0.9, 0.9, 0.9};
//
//    #pragma omp parallel for default(shared)
//    for (int i = 0; i < N; i++)
//    {
//        double vf[3];
//        double vc[3];
//        double t = double(i) / (N - 1);
//        for (int j = 0; j < 3; j++) vf[j] = vf1[j] + t * (vf2[j] - vf1[j]);
//
//        sim_param->get_coordinates<cartesian, direct>(vf, vc);
//        p[i] = potential->value(vc);
//        x[i] = Utils::vector_length(vc);
//    }
//
//    FILE* fout = fopen("potential.dat", "w");
//    for (int i = 0; i < N; i++) fprintf(fout, "%.12f %.12f\n", x[i] - x[0], p[i]);
//    fclose(fout);
//    delete x;
//    delete p;
}

//** void FORTRAN(sirius_print_rti)(void)
//** {
//**     sim_param->print_rti();
//** }

void sirius_write_json_output(void)
{
    PROFILE();

#ifdef __TIMER
    auto ts = runtime::Timer::collect_timer_stats();
    if (mpi_comm_world().rank() == 0)
    {
        std::string fname = std::string("output_") + sim_ctx->start_time_tag() + std::string(".json");
        JSON_write jw(fname);
        
        //== jw.single("git_hash", git_hash);
        //== jw.single("build_date", build_date);
        //== jw.single("num_ranks", ctx.comm().size());
        //== jw.single("max_num_threads", Platform::max_num_threads());
        //== //jw.single("cyclic_block_size", p->cyclic_block_size());
        //== jw.single("mpi_grid", ctx.parameters().mpi_grid_dims());
        //== std::vector<int> fftgrid(3);
        //== for (int i = 0; i < 3; i++) fftgrid[i] = ctx.fft().size(i);
        //== jw.single("fft_grid", fftgrid);
        //== jw.single("chemical_formula", ctx.unit_cell().chemical_formula());
        //== jw.single("num_atoms", ctx.unit_cell().num_atoms());
        //== jw.single("num_fv_states", ctx.parameters().num_fv_states());
        //== jw.single("num_bands", ctx.parameters().num_bands());
        //== jw.single("aw_cutoff", ctx.parameters().aw_cutoff());
        //== jw.single("pw_cutoff", ctx.parameters().pw_cutoff());
        //== jw.single("omega", ctx.unit_cell().omega());

        //== jw.begin_set("energy");
        //== jw.single("total", etot, 8);
        //== jw.single("evxc", evxc, 8);
        //== jw.single("eexc", eexc, 8);
        //== jw.single("evha", evha, 8);
        //== jw.single("enuc", enuc, 8);
        //== jw.end_set();
        //== 
        //== //** if (num_mag_dims())
        //== //** {
        //== //**     std::vector<double> v(3, 0);
        //== //**     v[2] = rti().total_magnetization[0];
        //== //**     if (num_mag_dims() == 3)
        //== //**     {
        //== //**         v[0] = rti().total_magnetization[1];
        //== //**         v[1] = rti().total_magnetization[2];
        //== //**     }
        //== //**     jw.single("total_moment", v);
        //== //**     jw.single("total_moment_len", Utils::vector_length(&v[0]));
        //== //** }
        //== 
        //== //** jw.single("total_energy", total_energy());
        //== //** jw.single("kinetic_energy", kinetic_energy());
        //== //** jw.single("energy_veff", rti_.energy_veff);
        //== //** jw.single("energy_vha", rti_.energy_vha);
        //== //** jw.single("energy_vxc", rti_.energy_vxc);
        //== //** jw.single("energy_bxc", rti_.energy_bxc);
        //== //** jw.single("energy_exc", rti_.energy_exc);
        //== //** jw.single("energy_enuc", rti_.energy_enuc);
        //== //** jw.single("core_eval_sum", rti_.core_eval_sum);
        //== //** jw.single("valence_eval_sum", rti_.valence_eval_sum);
        //== //** jw.single("band_gap", rti_.band_gap);
        //== //** jw.single("energy_fermi", rti_.energy_fermi);
        
        jw.single("timers", ts);
    }
#endif
}

void FORTRAN(sirius_get_occupation_matrix)(int32_t* atom_id, double_complex* occupation_matrix)
{
    PROFILE();
    int ia = *atom_id - 1;
    sim_ctx->unit_cell().atom(ia).get_occupation_matrix(occupation_matrix);
}

void FORTRAN(sirius_set_uj_correction_matrix)(int32_t* atom_id, int32_t* l, double_complex* uj_correction_matrix)
{
    PROFILE();
    int ia = *atom_id - 1;
    sim_ctx->unit_cell().atom(ia).set_uj_correction_matrix(*l, uj_correction_matrix);
}

void FORTRAN(sirius_set_so_correction)(int32_t* so_correction)
{
    PROFILE();
    if (*so_correction != 0) 
    {
        sim_param->set_so_correction(true);
    }
    else
    {
        sim_param->set_so_correction(false);
    }
}

void FORTRAN(sirius_set_uj_correction)(int32_t* uj_correction)
{
    PROFILE();
    if (*uj_correction != 0)
    {
        sim_param->set_uj_correction(true);
    }
    else
    {
        sim_param->set_uj_correction(false);
    }
}

//void FORTRAN(sirius_platform_mpi_rank)(int32_t* rank)
//{
//    PROFILE();
//    *rank = sim_param->comm().rank();
//}
//
//void FORTRAN(sirius_platform_mpi_grid_rank)(int32_t* dimension, int32_t* rank)
//{
//    PROFILE();
//    *rank = sim_param->mpi_grid().coordinate(*dimension);
//}

//== void FORTRAN(sirius_platform_mpi_grid_barrier)(int32_t* dimension)
//== {
//==     PROFILE();
//==     sim_param->mpi_grid().barrier(1 << (*dimension));
//== }

//void FORTRAN(sirius_global_set_sync_flag)(int32_t* flag)
//{
//    PROFILE();
//    sim_param->set_sync_flag(*flag);
//}
//
//void FORTRAN(sirius_global_get_sync_flag)(int32_t* flag)
//{
//    PROFILE();
//    *flag = sim_param->sync_flag();
//}

//void FORTRAN(sirius_platform_barrier)(void)
//{
//    PROFILE();
//    sim_param->comm().barrier();
//}

void sirius_get_energy_tot(double* total_energy__)
{
    PROFILE();
    *total_energy__ = dft_ground_state->total_energy();
}

void sirius_get_energy_ewald(double* ewald_energy__)
{
    PROFILE();
    *ewald_energy__ = dft_ground_state->energy_ewald();
}


void sirius_add_atom_type_aw_descriptor(char const* label__,
                                        int32_t const* n__,
                                        int32_t const* l__,
                                        double const* enu__, 
                                        int32_t const* dme__,
                                        int32_t const* auto_enu__)
{
    PROFILE();
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
    PROFILE();
    std::string label(label__);
    auto& type = sim_ctx->unit_cell().atom_type(std::string(label__));
    type.add_lo_descriptor(*ilo__ - 1, *n__, *l__, *enu__, *dme__, *auto_enu__);
}

void sirius_set_aw_enu(int32_t const* ia__,
                       int32_t const* l__,
                       int32_t const* order__,
                       double const* enu__)
{
    PROFILE();
    sim_ctx->unit_cell().atom(*ia__ - 1).symmetry_class().set_aw_enu(*l__, *order__ - 1, *enu__);
}

void sirius_get_aw_enu(int32_t const* ia__,
                       int32_t const* l__,
                       int32_t const* order__,
                       double* enu__)
{
    PROFILE();
    *enu__ = sim_ctx->unit_cell().atom(*ia__ - 1).symmetry_class().get_aw_enu(*l__, *order__ - 1);
}

void sirius_set_lo_enu(int32_t const* ia__,
                       int32_t const* idxlo__,
                       int32_t const* order__,
                       double const* enu__)
{
    PROFILE();
    sim_ctx->unit_cell().atom(*ia__ - 1).symmetry_class().set_lo_enu(*idxlo__ - 1, *order__ - 1, *enu__);
}

void sirius_get_lo_enu(int32_t const* ia__,
                       int32_t const* idxlo__,
                       int32_t const* order__,
                       double* enu__)
{
    PROFILE();
    *enu__ = sim_ctx->unit_cell().atom(*ia__ - 1).symmetry_class().get_lo_enu(*idxlo__ - 1, *order__ - 1);
}

/// Create the k-point set from the list of k-points and return it's id
void sirius_create_kset(int32_t* num_kpoints__,
                        double* kpoints__,
                        double* kpoint_weights__,
                        int32_t* init_kset__, 
                        int32_t* kset_id__)
{
    PROFILE();
    mdarray<double, 2> kpoints(kpoints__, 3, *num_kpoints__); 
    
    sirius::K_set* new_kset = new sirius::K_set(*sim_ctx, sim_ctx->mpi_grid().communicator(1 << _dim_k_), *blacs_grid);
    new_kset->add_kpoints(kpoints, kpoint_weights__);
    if (*init_kset__) new_kset->initialize();
   
    kset_list.push_back(new_kset);
    *kset_id__ = (int)kset_list.size() - 1;

}

void sirius_create_irreducible_kset_(int32_t* mesh__, int32_t* is_shift__, int32_t* use_sym__, int32_t* kset_id__)
{
    PROFILE();
    for (int x = 0; x < 3; x++)
    {
        if (!(is_shift__[x] == 0 || is_shift__[x] == 1))
        {
            std::stringstream s;
            s << "wrong k-shift " << is_shift__[0] << " " << is_shift__[1] << " " << is_shift__[2]; 
            TERMINATE(s);
        }
    }

    sirius::K_set* new_kset = new sirius::K_set(*sim_ctx,
                                                sim_ctx->mpi_grid().communicator(1 << _dim_k_),
                                                *blacs_grid,
                                                vector3d<int>(mesh__[0], mesh__[1], mesh__[2]),
                                                vector3d<int>(is_shift__[0], is_shift__[1], is_shift__[2]),
                                                *use_sym__);

    new_kset->initialize();
   
    kset_list.push_back(new_kset);
    *kset_id__ = (int)kset_list.size() - 1;
}

void sirius_delete_kset(int32_t* kset_id__)
{
    PROFILE();
    delete kset_list[*kset_id__];
    kset_list[*kset_id__] = nullptr;
}

void sirius_get_local_num_kpoints(int32_t* kset_id, int32_t* nkpt_loc)
{
    PROFILE();
    *nkpt_loc = (int)kset_list[*kset_id]->spl_num_kpoints().local_size();
}

void sirius_get_local_kpoint_rank_and_offset(int32_t* kset_id, int32_t* ik, int32_t* rank, int32_t* ikloc)
{
    PROFILE();
    *rank = kset_list[*kset_id]->spl_num_kpoints().local_rank(*ik - 1);
    *ikloc = (int)kset_list[*kset_id]->spl_num_kpoints().local_index(*ik - 1) + 1;
}

void sirius_get_global_kpoint_index(int32_t* kset_id, int32_t* ikloc, int32_t* ik)
{
    PROFILE();
    *ik = kset_list[*kset_id]->spl_num_kpoints(*ikloc - 1) + 1; // Fortran counts from 1
}

/// Generate radial functions (both aw and lo)
void sirius_generate_radial_functions()
{
    PROFILE();
    sim_ctx->unit_cell().generate_radial_functions();
}

/// Generate radial integrals
void sirius_generate_radial_integrals()
{
    PROFILE();
    sim_ctx->unit_cell().generate_radial_integrals();
}

void sirius_get_symmetry_classes(int32_t* ncls, int32_t* icls_by_ia)
{
    PROFILE();
    *ncls = sim_ctx->unit_cell().num_atom_symmetry_classes();

    for (int ic = 0; ic < sim_ctx->unit_cell().num_atom_symmetry_classes(); ic++)
    {
        for (int i = 0; i < sim_ctx->unit_cell().atom_symmetry_class(ic).num_atoms(); i++)
            icls_by_ia[sim_ctx->unit_cell().atom_symmetry_class(ic).atom_id(i)] = ic + 1; // Fortran counts from 1
    }
}

void sirius_get_max_mt_radial_basis_size(int32_t* max_mt_radial_basis_size)
{
    PROFILE();
    *max_mt_radial_basis_size = sim_ctx->unit_cell().max_mt_radial_basis_size();
}

void sirius_get_radial_functions(double* radial_functions__)
{
    PROFILE();
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
    PROFILE();
    *max_mt_basis_size = sim_ctx->unit_cell().max_mt_basis_size();
}

void sirius_get_basis_functions_index(int32_t* mt_basis_size, int32_t* offset_wf, int32_t* indexb__)
{
    PROFILE();
    mdarray<int, 3> indexb(indexb__, 4, sim_ctx->unit_cell().max_mt_basis_size(), sim_ctx->unit_cell().num_atoms()); 

    for (int ia = 0; ia < sim_ctx->unit_cell().num_atoms(); ia++)
    {
        mt_basis_size[ia] = sim_ctx->unit_cell().atom(ia).type().mt_basis_size();
        offset_wf[ia] = sim_ctx->unit_cell().atom(ia).offset_wf();

        for (int j = 0; j < sim_ctx->unit_cell().atom(ia).type().mt_basis_size(); j++)
        {
            indexb(0, j, ia) = sim_ctx->unit_cell().atom(ia).type().indexb(j).l;
            indexb(1, j, ia) = sim_ctx->unit_cell().atom(ia).type().indexb(j).lm + 1; // Fortran counts from 1
            indexb(2, j, ia) = sim_ctx->unit_cell().atom(ia).type().indexb(j).idxrf + 1; // Fortran counts from 1
        }
    }
}

/// Get number of G+k vectors for a given k-point in the set
void sirius_get_num_gkvec(int32_t* kset_id, int32_t* ik, int32_t* num_gkvec)
{
    PROFILE();
    *num_gkvec = (*kset_list[*kset_id])[*ik - 1]->num_gkvec();
}

/// Get maximum number of G+k vectors across all k-points in the set
void sirius_get_max_num_gkvec(int32_t const* kset_id__,
                              int32_t* max_num_gkvec__)
{
    PROFILE();
    *max_num_gkvec__ = kset_list[*kset_id__]->max_num_gkvec();
}

/// Get all G+k vector related arrays
void sirius_get_gkvec_arrays(int32_t* kset_id,
                             int32_t* ik,
                             int32_t* num_gkvec,
                             int32_t* gvec_index, 
                             double* gkvec__,
                             double* gkvec_cart__,
                             double* gkvec_len,
                             double* gkvec_tp__, 
                             double_complex* gkvec_phase_factors__,
                             int32_t* ld)
{
    PROFILE();

    /* position of processors which store a given k-point */
    int rank = kset_list[*kset_id]->spl_num_kpoints().local_rank(*ik - 1);
    
    auto& comm_r = sim_ctx->mpi_grid().communicator(1 << _dim_row_);
    auto& comm_k = sim_ctx->mpi_grid().communicator(1 << _dim_k_);

    if (rank == comm_k.rank())
    {
        sirius::K_point* kp = (*kset_list[*kset_id])[*ik - 1];
        *num_gkvec = kp->num_gkvec();
        mdarray<double, 2> gkvec(gkvec__, 3, kp->num_gkvec()); 
        mdarray<double, 2> gkvec_cart(gkvec_cart__, 3, kp->num_gkvec());
        mdarray<double, 2> gkvec_tp(gkvec_tp__, 2, kp->num_gkvec()); 

        for (int igk = 0; igk < kp->num_gkvec(); igk++)
        {
            auto gkc = kp->gkvec().cart_shifted(igk);

            //gvec_index[igk] = kp->gvec_index(igk) + 1; // Fortran counts form 1
            gvec_index[igk] = igk + 1; // Fortran counts from 1
            for (int x = 0; x < 3; x++) 
            {
                gkvec(x, igk) = kp->gkvec().gvec_shifted(igk)[x]; //kp->gkvec<fractional>(igk)[x];
                gkvec_cart(x, igk) = gkc[x]; //kp->gkvec().cart_shifted(igk)[x]; //kp->gkvec<cartesian>(igk)[x];
            }
            auto rtp = sirius::SHT::spherical_coordinates(gkc);
            gkvec_len[igk] = rtp[0];
            gkvec_tp(0, igk) = rtp[1];
            gkvec_tp(1, igk) = rtp[2];
        }
        
        mdarray<double_complex, 2> gkvec_phase_factors(gkvec_phase_factors__, *ld, sim_ctx->unit_cell().num_atoms());
        gkvec_phase_factors.zero();
        STOP();
        //for (int igkloc = 0; igkloc < kp->num_gkvec_row(); igkloc++)
        //{
        //    int igk = kp->gklo_basis_descriptor_row(igkloc).igk;
        //    for (int ia = 0; ia < sim_ctx->unit_cell().num_atoms(); ia++)
        //        gkvec_phase_factors(igk, ia) = kp->gkvec_phase_factor(igkloc, ia);
        //}
        comm_r.allreduce(&gkvec_phase_factors(0, 0), (int)gkvec_phase_factors.size()); 
    }
    comm_k.bcast(num_gkvec, 1, rank);
    comm_k.bcast(gvec_index, *num_gkvec, rank);
    comm_k.bcast(gkvec__, *num_gkvec * 3, rank);
    comm_k.bcast(gkvec_cart__, *num_gkvec * 3, rank);
    comm_k.bcast(gkvec_len, *num_gkvec, rank);
    comm_k.bcast(gkvec_tp__, *num_gkvec * 2, rank);
    comm_k.bcast(gkvec_phase_factors__, *ld * sim_ctx->unit_cell().num_atoms(), rank);
}

void sirius_get_matching_coefficients(int32_t const* kset_id__,
                                      int32_t const* ik__,
                                      double_complex* apwalm__, 
                                      int32_t const* ngkmax__,
                                      int32_t const* apwordmax__)
{
    PROFILE();

    TERMINATE_NOT_IMPLEMENTED;

    //int rank = kset_list[*kset_id__]->spl_num_kpoints().local_rank(*ik__ - 1);
    //
    //if (rank == sim_ctx->mpi_grid().coordinate(0))
    //{
    //    auto kp = (*kset_list[*kset_id__])[*ik__ - 1];
    //    
    //    mdarray<double_complex, 4> apwalm(apwalm__, *ngkmax__, *apwordmax__, sim_param->lmmax_apw(), 
    //                                      sim_ctx->unit_cell().num_atoms());


    //    dmatrix<double_complex> alm(kp->num_gkvec_row(), sim_ctx->unit_cell().mt_aw_basis_size(), *blacs_grid, sim_param->cyclic_block_size(), sim_param->cyclic_block_size());
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
    //    //==                         sim_param->mpi_grid().communicator(1 << _dim_row_));
    //    //== }
    //}
}

/// Get first-variational matrices of Hamiltonian and overlap
/** Radial integrals and plane-wave coefficients of the interstitial potential must be calculated prior to
 *  Hamiltonian and overlap matrix construction. 
 */
void sirius_get_fv_h_o(int32_t const* kset_id__,
                       int32_t const* ik__,
                       int32_t const* size__,
                       double_complex* h__,
                       double_complex* o__)
{
    int rank = kset_list[*kset_id__]->spl_num_kpoints().local_rank(*ik__ - 1);
    
    if (rank == sim_ctx->mpi_grid().coordinate(0))
    {
        auto kp = (*kset_list[*kset_id__])[*ik__ - 1];
        
        if (*size__ != kp->gklo_basis_size())
        {
            TERMINATE("wrong matrix size");
        }

        dmatrix<double_complex> h(h__, kp->gklo_basis_size(), kp->gklo_basis_size(), *blacs_grid, sim_param->cyclic_block_size(), sim_param->cyclic_block_size());
        dmatrix<double_complex> o(o__, kp->gklo_basis_size(), kp->gklo_basis_size(), *blacs_grid, sim_param->cyclic_block_size(), sim_param->cyclic_block_size());
        kset_list[*kset_id__]->band()->set_fv_h_o<CPU, full_potential_lapwlo>(kp, potential->effective_potential(), h, o);  
    }
}

void sirius_solve_fv(int32_t const* kset_id__,
                     int32_t const* ik__,
                     double_complex* h__,
                     double_complex* o__,
                     double* eval__,
                     double_complex* evec__,
                     int32_t const* evec_ld__)
{
    int rank = kset_list[*kset_id__]->spl_num_kpoints().local_rank(*ik__ - 1);
    
    if (rank == sim_ctx->mpi_grid().coordinate(0))
    {
        auto kp = (*kset_list[*kset_id__])[*ik__ - 1];
    
        kset_list[*kset_id__]->band()->gen_evp_solver()->solve(kp->gklo_basis_size(),
                                                               kp->gklo_basis_size_row(),
                                                               kp->gklo_basis_size_col(),
                                                               sim_ctx->num_fv_states(),
                                                               h__,
                                                               kp->gklo_basis_size_row(), 
                                                               o__,
                                                               kp->gklo_basis_size_row(),
                                                               eval__, 
                                                               evec__,
                                                               *evec_ld__);
    }
}

/// Get the total size of wave-function (number of mt coefficients + number of G+k coefficients)
void sirius_get_mtgk_size(int32_t* kset_id, int32_t* ik, int32_t* mtgk_size)
{
    PROFILE();
    *mtgk_size = (*kset_list[*kset_id])[*ik - 1]->wf_size();
}

void sirius_get_spinor_wave_functions(int32_t* kset_id, int32_t* ik, double_complex* spinor_wave_functions__)
{
    PROFILE();
    TERMINATE("fix this for distributed WF storage");
    //== assert(sim_param->num_bands() == (int)sim_param->spl_spinor_wf().local_size());

    //== sirius::K_point* kp = (*kset_list[*kset_id])[*ik - 1];
    //== 
    //== mdarray<double_complex, 3> spinor_wave_functions(spinor_wave_functions__, kp->wf_size(), sim_param->num_spins(), 
    //==                                             sim_param->spl_spinor_wf().local_size());

    //== for (int j = 0; j < (int)sim_param->spl_spinor_wf().local_size(); j++)
    //== {
    //==     memcpy(&spinor_wave_functions(0, 0, j), &kp->spinor_wave_function(0, 0, j), 
    //==            kp->wf_size() * sim_param->num_spins() * sizeof(double_complex));
    //== }
}

//== void FORTRAN(sirius_apply_step_function_gk)(int32_t* kset_id, int32_t* ik, double_complex* wf__)
//== {
//==     PROFILE();
//==     int thread_id = Platform::thread_id();
//== 
//==     sirius::K_point* kp = (*kset_list[*kset_id])[*ik - 1];
//==     int num_gkvec = kp->num_gkvec();
//== 
//==     sim_ctx->reciprocal_lattice()->fft().input(num_gkvec, kp->fft_index(), wf__, thread_id);
//==     sim_ctx->reciprocal_lattice()->fft().transform(1, thread_id);
//==     for (int ir = 0; ir < sim_ctx->reciprocal_lattice()->fft().size(); ir++)
//==         sim_ctx->reciprocal_lattice()->fft().buffer(ir, thread_id) *= sim_param->step_function()->theta_it(ir);
//== 
//==     sim_ctx->reciprocal_lattice()->fft().transform(-1, thread_id);
//==     sim_ctx->reciprocal_lattice()->fft().output(num_gkvec, kp->fft_index(), wf__, thread_id);
//== }

/// Get Cartesian coordinates of G+k vectors
void sirius_get_gkvec_cart(int32_t* kset_id, int32_t* ik, double* gkvec_cart__)
{
    PROFILE();
    sirius::K_point* kp = (*kset_list[*kset_id])[*ik - 1];
    mdarray<double, 2> gkvec_cart(gkvec_cart__, 3, kp->num_gkvec());

    for (int igk = 0; igk < kp->num_gkvec(); igk++)
    {
        for (int x = 0; x < 3; x++) gkvec_cart(x, igk) = kp->gkvec().gvec_shifted(igk)[x]; //kp->gkvec<cartesian>(igk)[x];
    }
}

void sirius_get_evalsum(double* evalsum)
{
    PROFILE();
    *evalsum = dft_ground_state->eval_sum();
}

void sirius_get_energy_exc(double* energy_exc)
{
    PROFILE();
    *energy_exc = dft_ground_state->energy_exc();
}

void sirius_get_energy_vxc(double* energy_vxc)
{
    PROFILE();
    *energy_vxc = dft_ground_state->energy_vxc();
}

void sirius_get_energy_bxc(double* energy_bxc)
{
    PROFILE();
    *energy_bxc = dft_ground_state->energy_bxc();
}

void sirius_get_energy_veff(double* energy_veff)
{
    PROFILE();
    *energy_veff = dft_ground_state->energy_veff();
}

void sirius_get_energy_vha(double* energy_vha)
{
    PROFILE();
    *energy_vha = dft_ground_state->energy_vha();
}

void sirius_get_energy_enuc(double* energy_enuc)
{
    PROFILE();
    *energy_enuc = dft_ground_state->energy_enuc();
}

void sirius_get_energy_kin(double* energy_kin)
{
    PROFILE();
    *energy_kin = dft_ground_state->energy_kin();
}

/// Generate XC potential and magnetic field
void sirius_generate_xc_potential(double* vxcmt__, double* vxcit__, double* bxcmt__, double* bxcit__)
{
    PROFILE();

    potential->xc(density->rho(), density->magnetization(), potential->xc_potential(), potential->effective_magnetic_field(), 
                  potential->xc_energy_density());

    potential->xc_potential()->copy_to_global_ptr(vxcmt__, vxcit__);
 
    if (sim_param->num_mag_dims() == 0) return;
    assert(sim_param->num_spins() == 2);

    /* set temporary array wrapper */
    mdarray<double, 4> bxcmt(bxcmt__, sim_param->lmmax_pot(), sim_ctx->unit_cell().max_num_mt_points(), 
                             sim_ctx->unit_cell().num_atoms(), sim_param->num_mag_dims());
    mdarray<double, 2> bxcit(bxcit__, sim_ctx->fft().size(), sim_param->num_mag_dims());

    if (sim_param->num_mag_dims() == 1)
    {
        /* z component */
        potential->effective_magnetic_field(0)->copy_to_global_ptr(&bxcmt(0, 0, 0, 0), &bxcit(0, 0));
    }
    else
    {
        /* z component */
        potential->effective_magnetic_field(0)->copy_to_global_ptr(&bxcmt(0, 0, 0, 2), &bxcit(0, 2));
        /* x component */
        potential->effective_magnetic_field(1)->copy_to_global_ptr(&bxcmt(0, 0, 0, 0), &bxcit(0, 0));
        /* y component */
        potential->effective_magnetic_field(2)->copy_to_global_ptr(&bxcmt(0, 0, 0, 1), &bxcit(0, 1));
    }
}

void sirius_generate_coulomb_potential(double* vclmt__, double* vclit__)
{
    PROFILE();
    
    potential->poisson(density->rho(), potential->hartree_potential());
    potential->hartree_potential()->copy_to_global_ptr(vclmt__, vclit__);

}

void sirius_update_atomic_potential()
{
    PROFILE();
    potential->update_atomic_potential();
}

void sirius_scalar_radial_solver(int32_t* zn, int32_t* l, int32_t* dme, double* enu, int32_t* nr, double* r, 
                                 double* v__, int32_t* nn, double* p0__, double* p1__, double* q0__, double* q1__)
{
    PROFILE();
    sirius::Radial_grid rgrid(*nr, r);
    sirius::Radial_solver solver(false, *zn, rgrid);

    std::vector<double> v(*nr);
    std::vector<double> p0;
    std::vector<double> p1;
    std::vector<double> q0;
    std::vector<double> q1;

    memcpy(&v[0], v__, (*nr) * sizeof(double));

    *nn = solver.solve(*l, *enu, *dme, v, p0, p1, q0, q1);

    memcpy(p0__, &p0[0], (*nr) * sizeof(double));
    memcpy(p1__, &p1[0], (*nr) * sizeof(double));
    memcpy(q0__, &q0[0], (*nr) * sizeof(double));
    memcpy(q1__, &q1[0], (*nr) * sizeof(double));
}

void sirius_get_aw_radial_function(int32_t const* ia__,
                                   int32_t const* l__,
                                   int32_t const* io__,
                                   double* f__)
{
    PROFILE();
    int ia = *ia__ - 1;
    int io = *io__ - 1;
    auto& atom = sim_ctx->unit_cell().atom(ia);
    int idxrf = atom.type().indexr_by_l_order(*l__, io);
    for (int ir = 0; ir < atom.num_mt_points(); ir++) f__[ir] = atom.symmetry_class().radial_function(ir, idxrf);
}
    
void sirius_get_aw_deriv_radial_function(int32_t* ia__,
                                         int32_t* l__,
                                         int32_t* io__,
                                         double* dfdr__)
{
    PROFILE();
    int ia = *ia__ - 1;
    int io = *io__ - 1;
    auto& atom = sim_ctx->unit_cell().atom(ia);
    int idxrf = atom.type().indexr_by_l_order(*l__, io);
    for (int ir = 0; ir < atom.num_mt_points(); ir++)
    {
        double rinv = atom.type().radial_grid().x_inv(ir);
        dfdr__[ir] = atom.symmetry_class().r_deriv_radial_function(ir, idxrf) * rinv;
    }
}
    
void sirius_get_aw_surface_derivative(int32_t* ia__, int32_t* l__, int32_t* io__, double* dawrf__)
{
    PROFILE();
    *dawrf__ = sim_ctx->unit_cell().atom(*ia__ - 1).symmetry_class().aw_surface_dm(*l__, *io__ - 1, 1); 
}

void sirius_get_lo_radial_function(int32_t const* ia__,
                                   int32_t const* idxlo__,
                                   double* f__)
{
    PROFILE();
    int ia = *ia__ - 1;
    int idxlo = *idxlo__ - 1;
    auto& atom = sim_ctx->unit_cell().atom(ia);
    int idxrf = atom.type().indexr_by_idxlo(idxlo);
    for (int ir = 0; ir < atom.num_mt_points(); ir++) f__[ir] = atom.symmetry_class().radial_function(ir, idxrf);
}
    
void sirius_get_lo_deriv_radial_function(int32_t const* ia__,
                                         int32_t const* idxlo__,
                                         double* dfdr__)
{
    PROFILE();
    int ia = *ia__ - 1;
    int idxlo = *idxlo__ - 1;
    auto& atom = sim_ctx->unit_cell().atom(ia);
    int idxrf = atom.type().indexr_by_idxlo(idxlo);
    for (int ir = 0; ir < atom.num_mt_points(); ir++)
    {
        double rinv = atom.type().radial_grid().x_inv(ir);
        dfdr__[ir] = atom.symmetry_class().r_deriv_radial_function(ir, idxrf) * rinv;
    }
}
    
void sirius_get_aw_lo_o_radial_integral(int32_t* ia__, int32_t* l, int32_t* io1, int32_t* ilo2, 
                                        double* oalo)
{
    PROFILE();
    int ia = *ia__ - 1;

    int idxrf2 = sim_ctx->unit_cell().atom(ia).type().indexr_by_idxlo(*ilo2 - 1);
    int order2 = sim_ctx->unit_cell().atom(ia).type().indexr(idxrf2).order;

    *oalo = sim_ctx->unit_cell().atom(ia).symmetry_class().o_radial_integral(*l, *io1 - 1, order2);
}

void sirius_get_lo_lo_o_radial_integral(int32_t* ia__, int32_t* l, int32_t* ilo1, int32_t* ilo2, 
                                        double* ololo)
{
    PROFILE();
    int ia = *ia__ - 1;

    int idxrf1 = sim_ctx->unit_cell().atom(ia).type().indexr_by_idxlo(*ilo1 - 1);
    int order1 = sim_ctx->unit_cell().atom(ia).type().indexr(idxrf1).order;
    int idxrf2 = sim_ctx->unit_cell().atom(ia).type().indexr_by_idxlo(*ilo2 - 1);
    int order2 = sim_ctx->unit_cell().atom(ia).type().indexr(idxrf2).order;

    *ololo = sim_ctx->unit_cell().atom(ia).symmetry_class().o_radial_integral(*l, order1, order2);
}

void sirius_get_aw_aw_h_radial_integral(int32_t* ia__, int32_t* l1, int32_t* io1, int32_t* l2, 
                                        int32_t* io2, int32_t* lm3, double* haa)
{
    PROFILE();
    int ia = *ia__ - 1;
    int idxrf1 = sim_ctx->unit_cell().atom(ia).type().indexr_by_l_order(*l1, *io1 - 1);
    int idxrf2 = sim_ctx->unit_cell().atom(ia).type().indexr_by_l_order(*l2, *io2 - 1);

    *haa = sim_ctx->unit_cell().atom(ia).h_radial_integrals(idxrf1, idxrf2)[*lm3 - 1];
}

void sirius_get_lo_aw_h_radial_integral(int32_t* ia__, int32_t* ilo1, int32_t* l2, int32_t* io2, int32_t* lm3, 
                                        double* hloa)
{
    PROFILE();
    int ia = *ia__ - 1;
    int idxrf1 = sim_ctx->unit_cell().atom(ia).type().indexr_by_idxlo(*ilo1 - 1);
    int idxrf2 = sim_ctx->unit_cell().atom(ia).type().indexr_by_l_order(*l2, *io2 - 1);

    *hloa = sim_ctx->unit_cell().atom(ia).h_radial_integrals(idxrf1, idxrf2)[*lm3 - 1];
}


void sirius_get_lo_lo_h_radial_integral(int32_t* ia__, int32_t* ilo1, int32_t* ilo2, int32_t* lm3, 
                                        double* hlolo)
{
    PROFILE();
    int ia = *ia__ - 1;
    int idxrf1 = sim_ctx->unit_cell().atom(ia).type().indexr_by_idxlo(*ilo1 - 1);
    int idxrf2 = sim_ctx->unit_cell().atom(ia).type().indexr_by_idxlo(*ilo2 - 1);

    *hlolo = sim_ctx->unit_cell().atom(ia).h_radial_integrals(idxrf1, idxrf2)[*lm3 - 1];
}

void sirius_generate_potential_pw_coefs()
{
    PROFILE();
    potential->generate_pw_coefs();
}

void sirius_generate_density_pw_coefs()
{
    PROFILE();
    density->generate_pw_coefs();
}

/// Get first-variational eigen-vectors
/** Assume that the Fortran side holds the whole array */
void sirius_get_fv_eigen_vectors(int32_t* kset_id__, int32_t* ik__, double_complex* fv_evec__, int32_t* ld__, 
                                 int32_t* num_fv_evec__)
{
    PROFILE();
    mdarray<double_complex, 2> fv_evec(fv_evec__, *ld__, *num_fv_evec__);
    (*kset_list[*kset_id__])[*ik__ - 1]->get_fv_eigen_vectors(fv_evec);
}

/// Get second-variational eigen-vectors
/** Assume that the Fortran side holds the whole array */
void sirius_get_sv_eigen_vectors(int32_t* kset_id, int32_t* ik, double_complex* sv_evec__, int32_t* size)
{
    PROFILE();
    mdarray<double_complex, 2> sv_evec(sv_evec__, *size, *size);
    (*kset_list[*kset_id])[*ik - 1]->get_sv_eigen_vectors(sv_evec);
}

void sirius_get_num_fv_states(int32_t* num_fv_states__)
{
    PROFILE();
    *num_fv_states__ = sim_ctx->num_fv_states();
}

void sirius_set_num_fv_states(int32_t* num_fv_states__)
{
    PROFILE();
    sim_param->set_num_fv_states(*num_fv_states__);
}

void sirius_ground_state_initialize(int32_t* kset_id__)
{
    PROFILE();
    if (dft_ground_state != nullptr) TERMINATE("dft_ground_state object is already allocate");

    dft_ground_state = new sirius::DFT_ground_state(*sim_ctx, potential, density, kset_list[*kset_id__], 1);
}

void sirius_ground_state_clear()
{
    PROFILE();
    delete dft_ground_state;
    dft_ground_state = nullptr;
}

void sirius_get_mpi_comm(int32_t* directions__, int32_t* fcomm__)
{
    PROFILE();
    *fcomm__ = MPI_Comm_c2f(sim_ctx->mpi_grid().communicator(*directions__).mpi_comm());
}

void sirius_forces(double* forces__)
{
    PROFILE();
    mdarray<double, 2> forces(forces__, 3, sim_ctx->unit_cell().num_atoms()); 
    dft_ground_state->forces(forces);
}

void sirius_set_atom_pos(int32_t* atom_id, double* pos)
{
    PROFILE();
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
    sirius::K_set* kset = kset_list[*kset_id];
    for (int ikloc = 0; ikloc < (int)kset->spl_num_kpoints().local_size(); ikloc++)
    {
        int ik = kset->spl_num_kpoints(ikloc);
        (*kset)[ik]->test_spinor_wave_functions(0);
    }
}

void sirius_generate_gq_matrix_elements(int32_t* kset_id, double* vq)
{
     kset_list[*kset_id]->generate_Gq_matrix_elements(vector3d<double>(vq[0], vq[1], vq[2]));
}

void sirius_density_mixer_initialize(void)
{
    //if (sim_param->mixer_input_section_.type_ == "broyden")
    //{
    //    mixer_rho = new sirius::Broyden_mixer(density->size(), sim_param->mixer_input_section_.max_history_, 
    //                                          sim_param->mixer_input_section_.beta_, sim_param->comm());
    //}
    //else if (sim_param->mixer_input_section_.type_ == "linear")
    //{
    //    mixer_rho = new sirius::Linear_mixer(density->size(), sim_param->mixer_input_section_.beta_, sim_param->comm());
    //}
    //else
    //{
    //    error_global(__FILE__, __LINE__, "Wrong mixer type");
    //}
    //
    ///* initialize density mixer with starting density */
    //density->pack(mixer_rho);
    //mixer_rho->initialize();
    density->mixer_init();
}

void sirius_potential_mixer_initialize(void)
{
    if (sim_param->mixer_input_section().type_ == "linear")
    {
        mixer_pot = new sirius::Linear_mixer<double>(potential->size(), sim_param->mixer_input_section().gamma_, sim_ctx->comm());

        /* initialize potential mixer */
        potential->pack(mixer_pot);
        mixer_pot->initialize();
    }
}

void sirius_mix_density(double* rms)
{
    //density->pack(mixer_rho);
    //*rms = mixer_rho->mix();
    //density->unpack(mixer_rho->output_buffer());
    *rms = density->mix();
    sim_ctx->comm().bcast(rms, 1, 0);
}

void sirius_mix_potential(void)
{
    if (mixer_pot)
    {
        potential->pack(mixer_pot);
        mixer_pot->mix();
        potential->unpack(mixer_pot->output_buffer());
    }
}

void sirius_set_atom_type_dion(char* label__,
                               int32_t* num_beta__,
                               double* dion__)
{
    PROFILE();
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
    PROFILE();
    auto& type = sim_ctx->unit_cell().atom_type(std::string(label__));

    mdarray<double, 2> beta_rf(beta_rf__, *ld__, *num_beta__);
    type.uspp().lmax = 0;
    type.uspp().num_beta_radial_functions = *num_beta__;
    type.uspp().beta_l = std::vector<int>(*num_beta__);
    type.uspp().num_beta_radial_points = std::vector<int>(*num_beta__);
    for (int i = 0; i < *num_beta__; i++)
    {
        type.uspp().beta_l[i] = beta_l__[i];
        type.uspp().lmax = std::max(type.uspp().lmax, 2 * beta_l__[i]);
        type.uspp().num_beta_radial_points[i] = num_mesh_points__[i];
    }
    type.uspp().beta_radial_functions = mdarray<double, 2>(type.num_mt_points(), *num_beta__);
    beta_rf >> type.uspp().beta_radial_functions;
}
    
void sirius_set_atom_type_q_rf(char* label__,
                               int32_t* num_q_coefs__,
                               int32_t* lmax_q__,
                               double* q_coefs__,
                               double* rinner__,
                               double* q_rf__,
                               int32_t* lmax__)
{
    PROFILE();
    auto& type = sim_ctx->unit_cell().atom_type(std::string(label__));
    
    type.uspp().num_q_coefs = *num_q_coefs__;
    int nbeta = type.uspp().num_beta_radial_functions;
    
    /* copy interpolating coefficients */
    if (*num_q_coefs__ > 0)
    {
        mdarray<double, 4> q_coefs(q_coefs__, *num_q_coefs__, *lmax_q__ + 1, nbeta, nbeta);
        type.uspp().q_coefs = mdarray<double, 4>(*num_q_coefs__, *lmax_q__ + 1, nbeta, nbeta);
        q_coefs >> type.uspp().q_coefs;
    }
    
    type.uspp().q_functions_inner_radii = std::vector<double>(*lmax_q__ + 1);
    for (int l = 0; l <= *lmax_q__; l++) type.uspp().q_functions_inner_radii[l] = rinner__[l];
    
    /* temporary wrapper */
    mdarray<double, 3> q_rf(q_rf__, type.num_mt_points(), nbeta * (nbeta + 1) / 2, 2 * (*lmax__) + 1);

    /* allocate space for radial functions of Q operator */
    type.uspp().q_radial_functions_l = mdarray<double, 3>(type.num_mt_points(), nbeta * (nbeta + 1) / 2, 2 * type.uspp().lmax + 1);

    for (int nb = 0; nb < nbeta; nb++)
    {
        for (int mb = nb; mb < nbeta; mb++)
        {
            /* combined index */
            int ijv = (mb + 1) * mb / 2 + nb;

            if (*lmax__ == 0)
            {
                for (int l = 0; l <= 2 * type.uspp().lmax; l++)
                    memcpy(&type.uspp().q_radial_functions_l(0, ijv, l), &q_rf(0, ijv, 0), type.num_mt_points() * sizeof(double));
            } 
            else if (*lmax__ ==  type.uspp().lmax)
            {
                for (int l = 0; l <= 2 * type.uspp().lmax; l++)
                    memcpy(&type.uspp().q_radial_functions_l(0, ijv, l), &q_rf(0, ijv, l), type.num_mt_points() * sizeof(double));
            }
            else
            {
                std::stringstream s;
                s << "wrong lmax" << std::endl
                  << "lmax: " << *lmax__ << std::endl 
                  << "lmax_beta: " << type.uspp().lmax;
                TERMINATE(s);
            }
        }
    }
}
    
void sirius_set_atom_type_rho_core(char const* label__,
                                   int32_t* num_points__,
                                   double* rho_core__)
{
    auto& type = sim_ctx->unit_cell().atom_type(std::string(label__));
    type.uspp().core_charge_density = std::vector<double>(*num_points__);
    for (int i = 0; i < *num_points__; i++) type.uspp().core_charge_density[i] = rho_core__[i];
}

void sirius_set_atom_type_rho_tot(char const* label__,
                                  int32_t* num_points__,
                                  double* rho_tot__)
{
    auto& type = sim_ctx->unit_cell().atom_type(std::string(label__));
    type.uspp().total_charge_density = std::vector<double>(*num_points__);
    for (int i = 0; i < *num_points__; i++) type.uspp().total_charge_density[i] = rho_tot__[i];
}

void sirius_set_atom_type_vloc(char const* label__,
                               int32_t* num_points__,
                               double* vloc__)
{
    auto& type = sim_ctx->unit_cell().atom_type(std::string(label__));
    type.uspp().vloc = std::vector<double>(*num_points__);
    for (int i = 0; i < *num_points__; i++) type.uspp().vloc[i] = vloc__[i];
}

void sirius_symmetrize_density()
{
    dft_ground_state->symmetrize_density();
}

void sirius_get_rho_pw(int32_t* num_gvec__, 
                       double_complex* rho_pw__)
{
    PROFILE();

    int num_gvec = sim_ctx->gvec().num_gvec();
    assert(*num_gvec__ == num_gvec);
    memcpy(rho_pw__, &density->rho()->f_pw(0), num_gvec * sizeof(double_complex));
}

void sirius_set_rho_pw(int32_t* num_gvec__,
                       double_complex* rho_pw__)
{
    PROFILE();

    int num_gvec = sim_ctx->gvec().num_gvec();
    assert(*num_gvec__ == num_gvec);
    memcpy(&density->rho()->f_pw(0), rho_pw__, num_gvec * sizeof(double_complex));
    density->rho()->fft_transform(1);
}

void sirius_get_gvec_index(int32_t* gvec__, int32_t* ig__)
{
    vector3d<int> gv(gvec__[0], gvec__[1], gvec__[2]);
    *ig__ = sim_ctx->gvec().index_by_gvec(gv) + 1;
}

void sirius_use_internal_mixer(int32_t* use_internal_mixer__)
{
    *use_internal_mixer__ = (sim_param->mixer_input_section().exist_) ? 1 : 0;
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

void sirius_get_beta_projectors_(int32_t* kset_id__, int32_t* ik__, int32_t* ngk__, int32_t* nbeta__, 
                                 double_complex* beta_gk__, int32_t* ld__, int32_t* gvec_of_k__)
{
    TERMINATE("fix this");
    //sirius::Timer t("sirius_get_beta_projectors");

    //auto kset = kset_list[*kset_id__];
    //auto kp = (*kset)[*ik__ - 1];

    //if (*ngk__ != kp->num_gkvec())
    //{
    //    TERMINATE("wrong number of G+k vectors");
    //}
    //if (*nbeta__ != sim_ctx->unit_cell().mt_basis_size())
    //{
    //    TERMINATE("wrong number of beta-projectors");
    //}
    //mdarray<double_complex, 2> beta_gk(beta_gk__, *ld__, *nbeta__);

    //auto& beta_gk_sirius = (*kset)[*ik__ - 1]->beta_gk();

    //int lmax = 10;

    //std::vector<int> idxlm(Utils::lmmax(lmax));
    //std::vector<int> phase(Utils::lmmax(lmax), 1);
    //int lm = 0;
    //for (int l = 0; l <= lmax; l++)
    //{
    //    idxlm[lm++] = Utils::lm_by_l_m(l, 0);
    //    for (int m = 1; m <= l; m++)
    //    {
    //        idxlm[lm++] = Utils::lm_by_l_m(l, m);
    //        idxlm[lm] = Utils::lm_by_l_m(l, -m);
    //        if (m % 2 == 0) phase[lm] = -1;
    //        lm++;
    //    }
    //}

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
    //auto l_m_by_lm = Utils::l_m_by_lm(10);

    //for (int igk = 0; igk < kp->num_gkvec(); igk++)
    //{
    //    for (int ia = 0; ia < sim_ctx->unit_cell().num_atoms(); ia++)
    //    {
    //        auto atom = sim_ctx->unit_cell().atom(ia);
    //        int nbf = atom.mt_basis_size();
    //        /* cycle through QE beta projectors in R_lm */
    //        for (int xi = 0; xi < nbf; xi++)
    //        {
    //            int lm = atom.type().indexb(xi).lm;
    //            int order = atom.type().indexb(xi).order;
    //            /* this is lm componet of R_lm in sirius order */
    //            int lm1 = idxlm[lm];
    //            int l = l_m_by_lm[lm1].first;
    //            int m = l_m_by_lm[lm1].second;

    //            double_complex z;
    //            if (m == 0)
    //            {
    //                int xi1 = atom.type().indexb_by_lm_order(lm1, order);
    //                z = beta_gk_sirius(igk_map[igk], atom.offset_lo() + xi1);
    //            }
    //            else
    //            {
    //                int j1 = Utils::lm_by_l_m(l, m); 
    //                int xi1 = atom.type().indexb_by_lm_order(j1, order);
    //                int j2 = Utils::lm_by_l_m(l, -m); 
    //                int xi2 = atom.type().indexb_by_lm_order(j2, order);

    //                z = sirius::SHT::ylm_dot_rlm(l,  m, m) * beta_gk_sirius(igk_map[igk], atom.offset_lo() + xi1) + 
    //                    sirius::SHT::ylm_dot_rlm(l, -m, m) * beta_gk_sirius(igk_map[igk], atom.offset_lo() + xi2); 
    //            }
    //            z = z * double(phase[lm]);
    //            
    //            //== if (std::abs(beta_gk(igk, atom.offset_lo() + xi) - z) > 1e-4)
    //            //== {
    //            //==     printf("large diff for beta-projectors for ig: %i ia: %i xi: %i\n", igk, ia, xi);
    //            //==     std::cout << beta_gk(igk, atom.offset_lo() + xi) << " " << z << std::endl;
    //            //== }

    //            beta_gk(igk, atom.offset_lo() + xi) = z;
    //        }
    //    }
    //}
}

void sirius_get_vloc_(int32_t* size__, double* vloc__)
{
    TERMINATE("fix this");
    //if (!sim_ctx) return;

    //auto fft_coarse = sim_ctx->fft_coarse();
    //if (*size__ != fft_coarse->size())
    //{
    //    TERMINATE("wrong size of coarse FFT mesh");
    //}

    ///* map effective potential to a corase grid */
    //std::vector<double> veff_it_coarse(fft_coarse->size());
    //std::vector<double_complex> veff_pw_coarse(fft_coarse->num_gvec());

    ///* take only first num_gvec_coarse plane-wave harmonics; this is enough to apply V_eff to \Psi */
    //for (int igc = 0; igc < fft_coarse->num_gvec(); igc++)
    //{
    //    int ig = sim_ctx->fft().gvec_index(fft_coarse->gvec(igc));
    //    veff_pw_coarse[igc] = potential->effective_potential()->f_pw(ig);
    //}
    //fft_coarse->input(fft_coarse->num_gvec(), fft_coarse->index_map(), &veff_pw_coarse[0]);
    //fft_coarse->transform(1);
    //fft_coarse->output(vloc__);
    //for (int i = 0; i < fft_coarse->size(); i++) vloc__[i] *= 2; // convert to Ry
}

void sirius_get_q_mtrx_(int32_t* itype__, double* q_mtrx__, int32_t* ld__)
{
    if (!sim_ctx) return;

    mdarray<double, 2> q_mtrx(q_mtrx__, *ld__, *ld__);

    auto& atom_type = sim_ctx->unit_cell().atom_type(*itype__ - 1);

    int nbf = atom_type.mt_basis_size();

    mdarray<double_complex, 2> sirius_Ylm_to_QE_Rlm(nbf, nbf);
    sirius_Ylm_to_QE_Rlm.zero();

    for (int idxrf = 0; idxrf < atom_type.mt_radial_basis_size(); idxrf++)
    {
        int l = atom_type.indexr(idxrf).l;
        int offset = atom_type.indexb().index_by_idxrf(idxrf);

        for (int m1 = -l; m1 <= l; m1++) // this runs over Ylm index of sirius
        {
            for (int m2 = -l; m2 <= l; m2++) // this runs over Rlm index of sirius
            {
                int i = 0; // index of QE Rlm
                if (m2 > 0) i = m2 * 2 - 1;
                if (m2 < 0) i = (-m2) * 2;
                double phase = 1;
                if (m2 < 0 && (-m2) % 2 == 0) phase = -1;
                sirius_Ylm_to_QE_Rlm(offset + i, offset + l + m1) = sirius::SHT::rlm_dot_ylm(l, m2, m1) * phase;
            }
        }

    }

    mdarray<double_complex, 2> z1(nbf, nbf);
    mdarray<double_complex, 2> z2(nbf, nbf);
    STOP();

    //for (int xi1 = 0; xi1 < nbf; xi1++)
    //{
    //    for (int xi2 = 0; xi2 < nbf; xi2++) z1(xi1, xi2) = atom_type.uspp().q_mtrx(xi1, xi2);
    //}
    //linalg<CPU>::gemm(0, 2, nbf, nbf, nbf, double_complex(1, 0), z1, sirius_Ylm_to_QE_Rlm, double_complex(0, 0), z2);
    //linalg<CPU>::gemm(0, 0, nbf, nbf, nbf, double_complex(1, 0), sirius_Ylm_to_QE_Rlm, z2, double_complex(0, 0), z1);

    for (int xi1 = 0; xi1 < nbf; xi1++)
    {
        for (int xi2 = 0; xi2 < nbf; xi2++)
        {
            //== double diff = std::abs(q_mtrx(xi1, xi2) - real(z1(xi1, xi2)));
            //== printf("itype=%i, xi1,xi2=%i %i, q_diff=%18.12f\n", *itype__ - 1, xi1, xi2, diff);
            q_mtrx(xi1, xi2) = real(z1(xi1, xi2));
        }
    }
}

void sirius_get_d_mtrx_(int32_t* ia__, double* d_mtrx__, int32_t* ld__)
{
    if (!sim_ctx) return;

    mdarray<double, 2> d_mtrx(d_mtrx__, *ld__, *ld__);

    auto& atom = sim_ctx->unit_cell().atom(*ia__ - 1);

    int nbf = atom.mt_basis_size();

    mdarray<double_complex, 2> sirius_Ylm_to_QE_Rlm(nbf, nbf);
    sirius_Ylm_to_QE_Rlm.zero();

    for (int idxrf = 0; idxrf < atom.type().mt_radial_basis_size(); idxrf++)
    {
        int l = atom.type().indexr(idxrf).l;
        int offset = atom.type().indexb().index_by_idxrf(idxrf);

        for (int m1 = -l; m1 <= l; m1++) // this runs over Ylm index of sirius
        {
            for (int m2 = -l; m2 <= l; m2++) // this runs over Rlm index of sirius
            {
                int i = 0; // index of QE Rlm
                if (m2 > 0) i = m2 * 2 - 1;
                if (m2 < 0) i = (-m2) * 2;
                double phase = 1;
                if (m2 < 0 && (-m2) % 2 == 0) phase = -1;
                sirius_Ylm_to_QE_Rlm(offset + i, offset + l + m1) = sirius::SHT::rlm_dot_ylm(l, m2, m1) * phase;
            }
        }

    }

    mdarray<double_complex, 2> z1(nbf, nbf);
    mdarray<double_complex, 2> z2(nbf, nbf);
    
    STOP();
    //for (int xi1 = 0; xi1 < nbf; xi1++)
    //{
    //    for (int xi2 = 0; xi2 < nbf; xi2++) z1(xi1, xi2) = atom.d_mtrx(xi1, xi2);
    //}
    linalg<CPU>::gemm(0, 2, nbf, nbf, nbf, double_complex(1, 0), z1, sirius_Ylm_to_QE_Rlm, double_complex(0, 0), z2);
    linalg<CPU>::gemm(0, 0, nbf, nbf, nbf, double_complex(1, 0), sirius_Ylm_to_QE_Rlm, z2, double_complex(0, 0), z1);

    for (int xi1 = 0; xi1 < nbf; xi1++)
    {
        for (int xi2 = 0; xi2 < nbf; xi2++)
        {
            //double diff = std::abs(d_mtrx(xi1, xi2) - real(z1(xi1, xi2) * 2.0));
            //if (diff > 1e-8)
            //{
            //    printf("ia=%2i, xi1,xi2=%2i %2i, D(QE)=%18.12f D(S)=%18.12f\n", *ia__ - 1, xi1, xi2, d_mtrx(xi1, xi2), real(z1(xi1, xi2)) * 2);
            //}
            d_mtrx(xi1, xi2) = real(z1(xi1, xi2)) * 2; // convert to Ry
        }
    }
}

void sirius_get_h_o_diag_(int32_t* kset_id__, int32_t* ik__, double* h_diag__, double* o_diag__, int32_t* ngk__, int32_t* gvec_of_k__)
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
    //
    //auto pw_ekin = kp->get_pw_ekin();

    //double v0 = real(potential->effective_potential()->f_pw(0));

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

    //std::vector<double> h_diag, o_diag;
    //kset->band()->get_h_o_diag<true>(kp, v0, pw_ekin, h_diag, o_diag);

    //for (int igk = 0; igk < kp->num_gkvec(); igk++)
    //{
    //    h_diag__[igk] = h_diag[igk_map[igk]] * 2; // convert to Ry
    //    o_diag__[igk] = o_diag[igk_map[igk]];
    //}
}

void sirius_generate_augmented_density_(double* rhoit__)
{



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

    //            z1(xi1, xi2) = atom_type.uspp().q_pw(ig, idx12);
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
    //if (*nfv__ != sim_param->num_fv_states())
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

    //for (int i = 0; i < sim_param->num_fv_states(); i++)
    //{
    //    for (int igk = 0; igk < kp->num_gkvec(); igk++)
    //    {
    //        fv_states(igk, i) = kp->fv_states()(igk_map[igk], i);
    //    }
    //}
}

void FORTRAN(sirius_scf_loop)()
{
    dft_ground_state->scf_loop(1e-6, 1e-6, 20);
}

//void FORTRAN(sirius_potential_checksum)()
//{
//    potential->checksum();
//}

void sirius_set_mpi_grid_dims(int *ndims__, int* dims__)
{
    assert(*ndims__ > 0);
    std::vector<int> dims(*ndims__);
    for (int i = 0; i < *ndims__; i++) dims[i] = dims__[i];
    sim_param->set_mpi_grid_dims(dims);
}


} // extern "C"
