// Copyright (c) 2013-2014 Anton Kozhevnikov, Thomas Schulthess
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

/** \file sirius.cpp
 *   
 *  \brief Fortran API.
 */

#include "sirius.h"

/// Pointer to Density class, implicitly used by Fortran side.
sirius::Density* density = NULL;

/// Pointer to Potential class, implicitly used by Fortran side.
sirius::Potential* potential = NULL;

/// Set of global parameters
sirius::Global* global_parameters = nullptr;

/// List of pointers to the sets of k-points.
std::vector<sirius::K_set*> kset_list;

/// DFT ground state wrapper
sirius::DFT_ground_state* dft_ground_state = NULL;

/// Charge density and magnetization mixer
sirius::Mixer* mixer_rho = NULL;

/// Potential and magnetic field mixer
sirius::Mixer* mixer_pot = NULL;

BLACS_grid* blacs_grid = nullptr;

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
    call sirius_platform_initialize(0)
    \endcode
 */
void FORTRAN(sirius_platform_initialize)(int32_t* call_mpi_init_)
{
    bool call_mpi_init = (*call_mpi_init_ != 0) ? true : false; 
    Platform::initialize(call_mpi_init);
    global_parameters = new sirius::Global(MPI_COMM_WORLD);
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
void FORTRAN(sirius_set_lattice_vectors)(double* a1, double* a2, double* a3)
{
    log_function_enter(__func__);
    global_parameters->unit_cell()->set_lattice_vectors(a1, a2, a3);
    log_function_exit(__func__);
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
void FORTRAN(sirius_set_pw_cutoff)(double* pw_cutoff)
{
    log_function_enter(__func__);
    global_parameters->set_pw_cutoff(*pw_cutoff);
    log_function_exit(__func__);
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
void FORTRAN(sirius_set_auto_rmt)(int32_t* auto_rmt)
{
    log_function_enter(__func__);
    global_parameters->unit_cell()->set_auto_rmt(*auto_rmt);
    log_function_exit(__func__);
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
void FORTRAN(sirius_add_atom_type)(char* label, char* fname, int32_t label_len, int32_t fname_len)
{
    log_function_enter(__func__);
    global_parameters->unit_cell()->add_atom_type(std::string(label, label_len), std::string(fname, fname_len), 
                                                  global_parameters->esm_type());
    log_function_exit(__func__);
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
void FORTRAN(sirius_set_atom_type_properties)(char* label, char* symbol, int32_t* zn, double* mass, 
                                              double* mt_radius, int32_t* num_mt_points, double* radial_grid_origin, 
                                              double* radial_grid_infinity, int32_t label_len, int32_t symbol_len)
{
    log_function_enter(__func__);
    sirius::Atom_type* type = global_parameters->unit_cell()->atom_type(std::string(label, label_len));
    type->set_symbol(std::string(symbol, symbol_len));
    type->set_zn(*zn);
    type->set_mass(*mass);
    type->set_num_mt_points(*num_mt_points);
    //type->set_radial_grid_origin(*radial_grid_origin);
    //type->set_radial_grid_infinity(*radial_grid_infinity);
    type->set_mt_radius(*mt_radius);
    log_function_exit(__func__);
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
void FORTRAN(sirius_set_atom_type_radial_grid)(char* label, int32_t* num_radial_points, 
                                               double* radial_points, int32_t label_len)
{
    sirius::Atom_type* type = global_parameters->unit_cell()->atom_type(std::string(label, label_len));
    type->set_radial_grid(type->num_mt_points(), radial_points);
    type->set_free_atom_radial_grid(*num_radial_points, radial_points);
}

void FORTRAN(sirius_set_free_atom_potential)(char* label__, int32_t* num_points__, double* vs__, int32_t label_len__)
{
    sirius::Atom_type* type = global_parameters->unit_cell()->atom_type(std::string(label__, label_len__));
    type->set_free_atom_potential(*num_points__, vs__);
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
void FORTRAN(sirius_set_atom_type_configuration)(char* label, int32_t* n, int32_t* l, int32_t* k, 
                                                 double* occupancy, int32_t* core_, int32_t label_len)
{
    log_function_enter(__func__);
    sirius::Atom_type* type = global_parameters->unit_cell()->atom_type(std::string(label, label_len));
    bool core = *core_;
    type->set_configuration(*n, *l, *k, *occupancy, core);
    log_function_exit(__func__);
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
void FORTRAN(sirius_add_atom)(char* label, double* position, double* vector_field, int32_t label_len)
{
    log_function_enter(__func__);
    global_parameters->unit_cell()->add_atom(std::string(label, label_len), position, vector_field);
    log_function_exit(__func__);
}

/// Set the table of equivalent atoms.
/** \param [in] equivalent_atoms table of equivalent atoms
    
    Equivalent atoms are symmetry related and belong to the same atom symmetry class. If equivalence table is not
    provided by user, \a spglib is called. In case of magnetic symmetry \a spglib is of no use and euivalence table 
    must be provided.
*/
void FORTRAN(sirius_set_equivalent_atoms)(int32_t* equivalent_atoms)
{
    log_function_enter(__func__);
    global_parameters->unit_cell()->set_equivalent_atoms(equivalent_atoms);
    log_function_exit(__func__);
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
void FORTRAN(sirius_set_aw_cutoff)(double* aw_cutoff)
{
    log_function_enter(__func__);
    global_parameters->set_aw_cutoff(*aw_cutoff);
    log_function_exit(__func__);
}

/// Initialize the global variables.
/** The function must be called after setting up the lattice vectors, plane wave-cutoff, autormt flag and loading
 *  atom types and atoms into the unit cell.
 *
 *  \param [in] lmax_apw maximum \f$ \ell \f$ for APW functions
 *  \param [in] lmax_rho maximum \f$ \ell \f$ for charge density and magnetization
 *  \param [in] lmax_pot maximum \f$ \ell \f$ for potential and effective magnetic field
 *  \param [in] num_mag_dims number of magnetic dimensions (0, 1 or 3)
 *
 *  Example:
    \code{.F90}
    integer lmaxapw, lmaxvr, ndmag
    lmaxapw = 10
    lmaxvr = 8
    ndmag = 0
    ! initialize global variables
    call sirius_global_initialize(lmaxapw, lmaxvr, lmaxvr, ndmag)
    \endcode
 */
void FORTRAN(sirius_global_initialize)(int32_t* lmax_apw, int32_t* lmax_rho, int32_t* lmax_pot, int32_t* num_mag_dims)
{
    log_function_enter(__func__);
    int num_spins = (*num_mag_dims == 0) ? 1 : 2;
    global_parameters->set_lmax_apw(*lmax_apw);
    global_parameters->set_lmax_rho(*lmax_rho);
    global_parameters->set_lmax_pot(*lmax_pot);
    global_parameters->set_num_spins(num_spins);
    global_parameters->set_num_mag_dims(*num_mag_dims);
    global_parameters->initialize();
    blacs_grid = new BLACS_grid(global_parameters->mpi_grid().communicator(1 << _dim_row_ | 1 << _dim_col_),
                                global_parameters->mpi_grid().dimension_size(_dim_row_),
                                global_parameters->mpi_grid().dimension_size(_dim_col_));
    log_function_exit(__func__);
}

/// Initialize the Density object.
/** \param [in] rhomt pointer to the muffin-tin part of the density
 *  \param [in] rhoit pointer to the interstitial part of the denssity
 *  \param [in] magmt pointer to the muffin-tin part of the magnetization
 *  \param [in] magit pointer to the interstitial part of the magnetization
 */
void FORTRAN(sirius_density_initialize)(double* rhomt, double* rhoit, double* magmt, double* magit)
{
    log_function_enter(__func__);
    density = new sirius::Density(*global_parameters);
    density->set_charge_density_ptr(rhomt, rhoit);
    density->set_magnetization_ptr(magmt, magit);
    log_function_exit(__func__);
}

/// Initialize the Potential object.
/** \param [in] veffmt pointer to the muffin-tin part of the effective potential
 *  \param [in] veffit pointer to the interstitial part of the effective potential
 *  \param [in] beffmt pointer to the muffin-tin part of effective magnetic field
 *  \param [in] beffit pointer to the interstitial part of the effective magnetic field
 */
void FORTRAN(sirius_potential_initialize)(double* veffmt, double* veffit, double* beffmt, double* beffit)
{
    log_function_enter(__func__);
    potential = new sirius::Potential(*global_parameters);
    potential->set_effective_potential_ptr(veffmt, veffit);
    potential->set_effective_magnetic_field_ptr(beffmt, beffit);
    log_function_exit(__func__);
}

/// Get maximum number of muffin-tin radial points.
/** \param [out] max_num_mt_points maximum number of muffin-tin points */
void FORTRAN(sirius_get_max_num_mt_points)(int32_t* max_num_mt_points)
{
    log_function_enter(__func__);
    *max_num_mt_points = global_parameters->unit_cell()->max_num_mt_points();
    log_function_exit(__func__);
}

/// Get number of muffin-tin radial points for a specific atom type.
/** \param [in] label unique label of atom type
 *  \param [out] num_mt_points number of muffin-tin points
 */
void FORTRAN(sirius_get_num_mt_points)(char* label, int32_t* num_mt_points, int32_t label_len)
{
    log_function_enter(__func__);
    *num_mt_points = global_parameters->unit_cell()->atom_type(std::string(label, label_len))->num_mt_points();
    log_function_exit(__func__);
}

void FORTRAN(sirius_get_mt_points)(char* label, double* mt_points, int32_t label_len)
{
    log_function_enter(__func__);
    sirius::Atom_type* type = global_parameters->unit_cell()->atom_type(std::string(label, label_len));
    for (int i = 0; i < type->num_mt_points(); i++) mt_points[i] = type->radial_grid(i);
    log_function_exit(__func__);
}

void FORTRAN(sirius_get_num_grid_points)(int32_t* num_grid_points)
{
    log_function_enter(__func__);
    *num_grid_points = global_parameters->reciprocal_lattice()->fft()->size();
    log_function_exit(__func__);
}

void FORTRAN(sirius_get_num_bands)(int32_t* num_bands)
{
    log_function_enter(__func__);
    *num_bands = global_parameters->num_bands();
    log_function_exit(__func__);
}

/// Get number of G-vectors within the plane-wave cutoff
void FORTRAN(sirius_get_num_gvec)(int32_t* num_gvec)
{
    log_function_enter(__func__);
    *num_gvec = global_parameters->reciprocal_lattice()->num_gvec();
    log_function_exit(__func__);
}

/// Get sizes of FFT grid
void FORTRAN(sirius_get_fft_grid_size)(int32_t* grid_size)
{
    log_function_enter(__func__);
    grid_size[0] = global_parameters->reciprocal_lattice()->fft()->size(0);
    grid_size[1] = global_parameters->reciprocal_lattice()->fft()->size(1);
    grid_size[2] = global_parameters->reciprocal_lattice()->fft()->size(2);
    log_function_exit(__func__);
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
void FORTRAN(sirius_get_fft_grid_limits)(int32_t* d, int32_t* lower, int32_t* upper)
{
    log_function_enter(__func__);
    assert((*d >= 1) && (*d <= 3));
    *lower = global_parameters->reciprocal_lattice()->fft()->grid_limits(*d - 1).first;
    *upper = global_parameters->reciprocal_lattice()->fft()->grid_limits(*d - 1).second;
    log_function_exit(__func__);
}

/// Get mapping between G-vector index and FFT index
void FORTRAN(sirius_get_fft_index)(int32_t* fft_index)
{
    log_function_enter(__func__);
    memcpy(fft_index, global_parameters->reciprocal_lattice()->fft_index(),  global_parameters->reciprocal_lattice()->fft()->size() * sizeof(int32_t));
    for (int i = 0; i < global_parameters->reciprocal_lattice()->fft()->size(); i++) fft_index[i]++;
    log_function_exit(__func__);
}

/// Get list of G-vectors in fractional corrdinates
void FORTRAN(sirius_get_gvec)(int32_t* gvec__)
{
    log_function_enter(__func__);
    mdarray<int, 2> gvec(gvec__, 3, global_parameters->reciprocal_lattice()->fft()->size());
    for (int ig = 0; ig < global_parameters->reciprocal_lattice()->fft()->size(); ig++)
    {
        vector3d<int> gv = global_parameters->reciprocal_lattice()->gvec(ig);
        for (int x = 0; x < 3; x++) gvec(x, ig) = gv[x];
    }
    //memcpy(gvec, global_parameters->gvec(0), 3 * global_parameters->fft()->size() * sizeof(int32_t));
    log_function_exit(__func__);
}

/// Get list of G-vectors in Cartesian coordinates
void FORTRAN(sirius_get_gvec_cart)(double* gvec_cart__)
{
    log_function_enter(__func__);
    mdarray<double, 2> gvec_cart(gvec_cart__, 3,  global_parameters->reciprocal_lattice()->fft()->size());
    for (int ig = 0; ig < global_parameters->reciprocal_lattice()->fft()->size(); ig++)
    {
        vector3d<double> gvc = global_parameters->reciprocal_lattice()->gvec_cart(ig);
        for (int x = 0; x < 3; x++) gvec_cart(x, ig) = gvc[x];
    }
    log_function_exit(__func__);
}

/// Get lengh of G-vectors
void FORTRAN(sirius_get_gvec_len)(double* gvec_len)
{
    log_function_enter(__func__);
    for (int ig = 0; ig <  global_parameters->reciprocal_lattice()->fft()->size(); ig++) gvec_len[ig] = global_parameters->reciprocal_lattice()->gvec_len(ig);
    log_function_exit(__func__);
}

void FORTRAN(sirius_get_index_by_gvec)(int32_t* index_by_gvec__)
{
    log_function_enter(__func__);
    sirius::FFT3D<cpu>* fft = global_parameters->reciprocal_lattice()->fft();
    std::pair<int, int> d0 = fft->grid_limits(0);
    std::pair<int, int> d1 = fft->grid_limits(1);
    std::pair<int, int> d2 = fft->grid_limits(2);

    mdarray<int, 3> index_by_gvec(index_by_gvec__, 
                                  mdarray_index_descriptor(d0.first, d0.second), 
                                  mdarray_index_descriptor(d1.first, d1.second), 
                                  mdarray_index_descriptor(d2.first, d2.second));

    for (int i0 = d0.first; i0 <= d0.second; i0++)
    {
        for (int i1 = d1.first; i1 <= d1.second; i1++)
        {
            for (int i2 = d2.first; i2 <= d2.second; i2++)
            {
                index_by_gvec(i0, i1, i2) = global_parameters->reciprocal_lattice()->index_by_gvec(i0, i1, i2) + 1;
            }
        }
    }
    log_function_exit(__func__);
}

/// Get Ylm spherical harmonics of G-vectors.
void FORTRAN(sirius_get_gvec_ylm)(double_complex* gvec_ylm__, int* ld__, int* lmax__)
{
    log_function_enter(__func__);
    mdarray<double_complex, 2> gvec_ylm(gvec_ylm__, *ld__, global_parameters->reciprocal_lattice()->num_gvec());
    // TODO: can be parallelized
    for (int ig = 0; ig < global_parameters->reciprocal_lattice()->num_gvec(); ig++)
    {
        global_parameters->reciprocal_lattice()->gvec_ylm_array<global>(ig, &gvec_ylm(0, ig), *lmax__);
    }
    log_function_exit(__func__);
}

void FORTRAN(sirius_get_gvec_phase_factors)(double_complex* sfacg__)
{
    log_function_enter(__func__);
    mdarray<double_complex, 2> sfacg(sfacg__, global_parameters->reciprocal_lattice()->num_gvec(), global_parameters->unit_cell()->num_atoms());
    for (int ia = 0; ia < global_parameters->unit_cell()->num_atoms(); ia++)
    {
        for (int ig = 0; ig < global_parameters->reciprocal_lattice()->num_gvec(); ig++)
            sfacg(ig, ia) = global_parameters->reciprocal_lattice()->gvec_phase_factor<global>(ig, ia);
    }
    log_function_exit(__func__);
}

void FORTRAN(sirius_get_step_function)(double_complex* cfunig, double* cfunir)
{
    log_function_enter(__func__);
    for (int i = 0; i < global_parameters->reciprocal_lattice()->fft()->size(); i++)
    {
        cfunig[i] = global_parameters->step_function()->theta_pw(i);
        cfunir[i] = global_parameters->step_function()->theta_it(i);
    }
    log_function_exit(__func__);
}

/// Get the total number of electrons
void FORTRAN(sirius_get_num_electrons)(double* num_electrons)
{
    log_function_enter(__func__);
    *num_electrons = global_parameters->unit_cell()->num_electrons();
    log_function_exit(__func__);
}

/// Get the number of valence electrons
void FORTRAN(sirius_get_num_valence_electrons)(double* num_valence_electrons)
{
    log_function_enter(__func__);
    *num_valence_electrons = global_parameters->unit_cell()->num_valence_electrons();
    log_function_exit(__func__);
}

/// Get the number of core electrons
void FORTRAN(sirius_get_num_core_electrons)(double* num_core_electrons)
{
    log_function_enter(__func__);
    *num_core_electrons = global_parameters->unit_cell()->num_core_electrons();
    log_function_exit(__func__);
}

/// Clear global variables and destroy all objects
void FORTRAN(sirius_clear)(void)
{
    log_function_enter(__func__);
    
    global_parameters->clear();

    if (density) 
    {
        delete density;
        density = NULL;
    }
    if (potential)
    {
        delete potential;
        potential = NULL;
    }
    if (dft_ground_state)
    {
        delete dft_ground_state;
        dft_ground_state = NULL;
    }
    if (blacs_grid != nullptr)
    {
        delete blacs_grid;
        blacs_grid = nullptr;
    }
    for (int i = 0; i < (int)kset_list.size(); i++)
    {
        if (kset_list[i] != NULL) 
        {
            delete kset_list[i];
            kset_list[i] = NULL;
        }
    }
    kset_list.clear();
    log_function_exit(__func__);
}

void FORTRAN(sirius_initial_density)(void)
{
    log_function_enter(__func__);
    density->initial_density();
    log_function_exit(__func__);
}

void FORTRAN(sirius_generate_effective_potential)(void)
{
    log_function_enter(__func__);
    potential->generate_effective_potential(density->rho(), density->magnetization());
    log_function_exit(__func__);
}

void FORTRAN(sirius_generate_density)(int32_t* kset_id__)
{
    log_function_enter(__func__);
    density->generate(*kset_list[*kset_id__]);
    log_function_exit(__func__);
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
void FORTRAN(sirius_find_eigen_states)(int32_t* kset_id__, int32_t* precompute__)
{
    log_function_enter(__func__);
    bool precompute = (*precompute__) ? true : false;
    kset_list[*kset_id__]->find_eigen_states(potential, precompute);
    log_function_exit(__func__);
}

void FORTRAN(sirius_find_band_occupancies)(int32_t* kset_id__)
{
    log_function_enter(__func__);
    kset_list[*kset_id__]->find_band_occupancies();
    log_function_exit(__func__);
}

void FORTRAN(sirius_set_band_occupancies)(int32_t* kset_id, int32_t* ik_, double* band_occupancies)
{
    log_function_enter(__func__);
    int ik = *ik_ - 1;
    kset_list[*kset_id]->set_band_occupancies(ik, band_occupancies);
    log_function_exit(__func__);
}

void FORTRAN(sirius_get_band_energies)(int32_t* kset_id__, int32_t* ik__, double* band_energies__)
{
    log_function_enter(__func__);
    int ik = *ik__ - 1;
    kset_list[*kset_id__]->get_band_energies(ik, band_energies__);
    log_function_exit(__func__);
}

void FORTRAN(sirius_get_band_occupancies)(int32_t* kset_id, int32_t* ik_, double* band_occupancies)
{
    log_function_enter(__func__);
    int ik = *ik_ - 1;
    kset_list[*kset_id]->get_band_occupancies(ik, band_occupancies);
    log_function_exit(__func__);
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
//==     log_function_enter(__func__);
//==     global_parameters->print_info();
//==     log_function_exit(__func__);
//== }

void FORTRAN(sirius_print_timers)(void)
{
    log_function_enter(__func__);
    sirius::Timer::print();
    log_function_exit(__func__);
}   

void FORTRAN(sirius_start_timer)(char* name_, int32_t name_len)
{
    log_function_enter(__func__);
    extern std::map<std::string, sirius::Timer*> ftimers;
    std::string name(name_, name_len);
    ftimers[name] = new sirius::Timer(name);
    log_function_exit(__func__);
}

void FORTRAN(sirius_stop_timer)(char* name_, int32_t name_len)
{
    log_function_enter(__func__);
    extern std::map<std::string, sirius::Timer*> ftimers;
    std::string name(name_, name_len);
    if (ftimers.count(name)) delete ftimers[name];
    log_function_exit(__func__);
}

void FORTRAN(sirius_save_potential)(void)
{
    log_function_enter(__func__);
    potential->save();
    log_function_exit(__func__);
}

void FORTRAN(sirius_save_density)(void)
{
    log_function_enter(__func__);
    density->save();
    log_function_exit(__func__);
}

void FORTRAN(sirius_load_potential)(void)
{
    log_function_enter(__func__);
    potential->load();
    log_function_exit(__func__);
}

//== void FORTRAN(sirius_save_wave_functions)(int32_t* kset_id)
//== {
//==     log_function_enter(__func__);
//==     kset_list[*kset_id]->save_wave_functions();
//==     log_function_exit(__func__);
//== }
//==     
//== void FORTRAN(sirius_load_wave_functions)(int32_t* kset_id)
//== {
//==     log_function_enter(__func__);
//==     kset_list[*kset_id]->load_wave_functions();
//==     log_function_exit(__func__);
//== }

void FORTRAN(sirius_save_kset)(int32_t* kset_id)
{
    kset_list[*kset_id]->save();
}

void FORTRAN(sirius_load_kset)(int32_t* kset_id)
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
//==     if (bz_path.size() < 2) error_local(__FILE__, __LINE__, "at least two BZ points are required");
//==    
//==     // compute length of segments
//==     std::vector<double> segment_length;
//==     double total_path_length = 0.0;
//==     for (int ip = 0; ip < (int)bz_path.size() - 1; ip++)
//==     {
//==         double vf[3];
//==         for (int x = 0; x < 3; x++) vf[x] = bz_path[ip + 1].second[x] - bz_path[ip].second[x];
//==         double vc[3];
//==         global_parameters->get_coordinates<cartesian, reciprocal>(vf, vc);
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
//==     global_parameters->solve_free_atoms();
//== 
//==     potential->update_atomic_potential();
//==     global_parameters->generate_radial_functions();
//==     global_parameters->generate_radial_integrals();
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
//==     if (global_parameters->mpi_grid().root())
//==     {
//==         JSON_write jw("bands.json");
//==         jw.single("xaxis", xaxis);
//==         //** jw.single("Ef", global_parameters->rti().energy_fermi);
//==         
//==         jw.single("xaxis_ticks", xaxis_ticks);
//==         jw.single("xaxis_tick_labels", xaxis_tick_labels);
//==         
//==         jw.begin_array("plot");
//==         std::vector<double> yvalues(kset_.num_kpoints());
//==         for (int i = 0; i < global_parameters->num_bands(); i++)
//==         {
//==             jw.begin_set();
//==             for (int ik = 0; ik < kset_.num_kpoints(); ik++) yvalues[ik] = kset_[ik]->band_energy(i);
//==             jw.single("yvalues", yvalues);
//==             jw.end_set();
//==         }
//==         jw.end_array();
//== 
//==         //FILE* fout = fopen("bands.dat", "w");
//==         //for (int i = 0; i < global_parameters->num_bands(); i++)
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
//    global_parameters->fft()->input(potential->effective_potential()->f_it());
//    global_parameters->fft()->transform(-1);
//    global_parameters->fft()->output(global_parameters->num_gvec(), global_parameters->fft_index(), 
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
//        global_parameters->get_coordinates<cartesian, direct>(vf, vc);
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
//**     global_parameters->print_rti();
//** }

void FORTRAN(sirius_write_json_output)(void)
{
    log_function_enter(__func__);
    global_parameters->write_json_output();
    log_function_exit(__func__);
}

void FORTRAN(sirius_get_occupation_matrix)(int32_t* atom_id, double_complex* occupation_matrix)
{
    log_function_enter(__func__);
    int ia = *atom_id - 1;
    global_parameters->unit_cell()->atom(ia)->get_occupation_matrix(occupation_matrix);
    log_function_exit(__func__);
}

void FORTRAN(sirius_set_uj_correction_matrix)(int32_t* atom_id, int32_t* l, double_complex* uj_correction_matrix)
{
    log_function_enter(__func__);
    int ia = *atom_id - 1;
    global_parameters->unit_cell()->atom(ia)->set_uj_correction_matrix(*l, uj_correction_matrix);
    log_function_exit(__func__);
}

void FORTRAN(sirius_set_so_correction)(int32_t* so_correction)
{
    log_function_enter(__func__);
    if (*so_correction != 0) 
    {
        global_parameters->set_so_correction(true);
    }
    else
    {
        global_parameters->set_so_correction(false);
    }
    log_function_exit(__func__);
}

void FORTRAN(sirius_set_uj_correction)(int32_t* uj_correction)
{
    log_function_enter(__func__);
    if (*uj_correction != 0)
    {
        global_parameters->set_uj_correction(true);
    }
    else
    {
        global_parameters->set_uj_correction(false);
    }
    log_function_exit(__func__);
}

void FORTRAN(sirius_platform_mpi_rank)(int32_t* rank)
{
    log_function_enter(__func__);
    *rank = global_parameters->comm().rank();
    log_function_exit(__func__);
}

void FORTRAN(sirius_platform_mpi_grid_rank)(int32_t* dimension, int32_t* rank)
{
    log_function_enter(__func__);
    *rank = global_parameters->mpi_grid().coordinate(*dimension);
    log_function_exit(__func__);
}

//== void FORTRAN(sirius_platform_mpi_grid_barrier)(int32_t* dimension)
//== {
//==     log_function_enter(__func__);
//==     global_parameters->mpi_grid().barrier(1 << (*dimension));
//==     log_function_exit(__func__);
//== }

//void FORTRAN(sirius_global_set_sync_flag)(int32_t* flag)
//{
//    log_function_enter(__func__);
//    global_parameters->set_sync_flag(*flag);
//    log_function_exit(__func__);
//}
//
//void FORTRAN(sirius_global_get_sync_flag)(int32_t* flag)
//{
//    log_function_enter(__func__);
//    *flag = global_parameters->sync_flag();
//    log_function_exit(__func__);
//}

void FORTRAN(sirius_platform_barrier)(void)
{
    log_function_enter(__func__);
    global_parameters->comm().barrier();
    log_function_exit(__func__);
}

void FORTRAN(sirius_get_total_energy)(double* total_energy)
{
    log_function_enter(__func__);
    *total_energy = dft_ground_state->total_energy();
    log_function_exit(__func__);
}


void FORTRAN(sirius_add_atom_type_aw_descriptor)(char* label, int32_t* n, int32_t* l, double* enu, 
                                                 int32_t* dme, int32_t* auto_enu, int32_t label_len)
{
    log_function_enter(__func__);
    sirius::Atom_type* type = global_parameters->unit_cell()->atom_type(std::string(label, label_len));
    type->add_aw_descriptor(*n, *l, *enu, *dme, *auto_enu);
    log_function_exit(__func__);
}

void FORTRAN(sirius_add_atom_type_lo_descriptor)(char* label, int32_t* ilo, int32_t* n, int32_t* l, 
                                                 double* enu, int32_t* dme, int32_t* auto_enu, int32_t label_len)
{
    log_function_enter(__func__);
    sirius::Atom_type* type = global_parameters->unit_cell()->atom_type(std::string(label, label_len));
    type->add_lo_descriptor(*ilo - 1, *n, *l, *enu, *dme, *auto_enu);
    log_function_exit(__func__);
}

void FORTRAN(sirius_set_aw_enu)(int32_t* ia, int32_t* l, int32_t* order, double* enu)
{
    log_function_enter(__func__);
    global_parameters->unit_cell()->atom(*ia - 1)->symmetry_class()->set_aw_enu(*l, *order - 1, *enu);
    log_function_exit(__func__);
}

void FORTRAN(sirius_get_aw_enu)(int32_t* ia, int32_t* l, int32_t* order, double* enu)
{
    log_function_enter(__func__);
    *enu = global_parameters->unit_cell()->atom(*ia - 1)->symmetry_class()->get_aw_enu(*l, *order - 1);
    log_function_exit(__func__);
}

void FORTRAN(sirius_set_lo_enu)(int32_t* ia, int32_t* idxlo, int32_t* order, double* enu)
{
    log_function_enter(__func__);
    global_parameters->unit_cell()->atom(*ia - 1)->symmetry_class()->set_lo_enu(*idxlo - 1, *order - 1, *enu);
    log_function_exit(__func__);
}

void FORTRAN(sirius_get_lo_enu)(int32_t* ia, int32_t* idxlo, int32_t* order, double* enu)
{
    log_function_enter(__func__);
    *enu = global_parameters->unit_cell()->atom(*ia - 1)->symmetry_class()->get_lo_enu(*idxlo - 1, *order - 1);
    log_function_exit(__func__);
}

/// Create the k-point set from the list of k-points and return it's id
void FORTRAN(sirius_create_kset)(int32_t* num_kpoints, double* kpoints__, double* kpoint_weights, int32_t* init_kset, 
                                 int32_t* kset_id)
{
    log_function_enter(__func__);
    mdarray<double, 2> kpoints(kpoints__, 3, *num_kpoints); 
    
    sirius::K_set* new_kset = new sirius::K_set(*global_parameters, global_parameters->mpi_grid().communicator(1 << _dim_k_), *blacs_grid);
    new_kset->add_kpoints(kpoints, kpoint_weights);
    if (*init_kset) new_kset->initialize();
    
    for (int i = 0; i < (int)kset_list.size(); i++)
    {
        if (kset_list[i] == NULL) 
        {
            kset_list[i] = new_kset;
            *kset_id = i;
            return;
        }
    }

    kset_list.push_back(new_kset);
    *kset_id = (int)kset_list.size() - 1;
    log_function_exit(__func__);
}

void FORTRAN(sirius_delete_kset)(int32_t* kset_id)
{
    log_function_enter(__func__);
    delete kset_list[*kset_id];
    kset_list[*kset_id] = NULL;
    log_function_exit(__func__);
}

void FORTRAN(sirius_get_local_num_kpoints)(int32_t* kset_id, int32_t* nkpt_loc)
{
    log_function_enter(__func__);
    *nkpt_loc = (int)kset_list[*kset_id]->spl_num_kpoints().local_size();
    log_function_exit(__func__);
}

void FORTRAN(sirius_get_local_kpoint_rank_and_offset)(int32_t* kset_id, int32_t* ik, int32_t* rank, int32_t* ikloc)
{
    log_function_enter(__func__);
    *rank = kset_list[*kset_id]->spl_num_kpoints().local_rank(*ik - 1);
    *ikloc = (int)kset_list[*kset_id]->spl_num_kpoints().local_index(*ik - 1) + 1;
    log_function_exit(__func__);
}

void FORTRAN(sirius_get_global_kpoint_index)(int32_t* kset_id, int32_t* ikloc, int32_t* ik)
{
    log_function_enter(__func__);
    *ik = kset_list[*kset_id]->spl_num_kpoints(*ikloc - 1) + 1; // Fortran counts from 1
    log_function_exit(__func__);
}

/// Generate radial functions (both aw and lo)
void FORTRAN(sirius_generate_radial_functions)()
{
    log_function_enter(__func__);
    global_parameters->unit_cell()->generate_radial_functions();
    log_function_exit(__func__);
}

/// Generate radial integrals
void FORTRAN(sirius_generate_radial_integrals)()
{
    log_function_enter(__func__);
    global_parameters->unit_cell()->generate_radial_integrals();
    log_function_exit(__func__);
}

void FORTRAN(sirius_get_symmetry_classes)(int32_t* ncls, int32_t* icls_by_ia)
{
    log_function_enter(__func__);
    *ncls = global_parameters->unit_cell()->num_atom_symmetry_classes();

    for (int ic = 0; ic < global_parameters->unit_cell()->num_atom_symmetry_classes(); ic++)
    {
        for (int i = 0; i < global_parameters->unit_cell()->atom_symmetry_class(ic)->num_atoms(); i++)
            icls_by_ia[global_parameters->unit_cell()->atom_symmetry_class(ic)->atom_id(i)] = ic + 1; // Fortran counts from 1
    }
    log_function_exit(__func__);
}

void FORTRAN(sirius_get_max_mt_radial_basis_size)(int32_t* max_mt_radial_basis_size)
{
    log_function_enter(__func__);
    *max_mt_radial_basis_size = global_parameters->unit_cell()->max_mt_radial_basis_size();
    log_function_exit(__func__);
}

void FORTRAN(sirius_get_radial_functions)(double* radial_functions__)
{
    log_function_enter(__func__);
    mdarray<double, 3> radial_functions(radial_functions__, 
                                        global_parameters->unit_cell()->max_num_mt_points(), 
                                        global_parameters->unit_cell()->max_mt_radial_basis_size(),
                                        global_parameters->unit_cell()->num_atom_symmetry_classes());
    radial_functions.zero();

    for (int ic = 0; ic < global_parameters->unit_cell()->num_atom_symmetry_classes(); ic++)
    {
        for (int idxrf = 0; idxrf < global_parameters->unit_cell()->atom_symmetry_class(ic)->atom_type()->mt_radial_basis_size(); idxrf++)
        {
            for (int ir = 0; ir < global_parameters->unit_cell()->atom_symmetry_class(ic)->atom_type()->num_mt_points(); ir++)
                radial_functions(ir, idxrf, ic) = global_parameters->unit_cell()->atom_symmetry_class(ic)->radial_function(ir, idxrf);
        }
    }
    log_function_exit(__func__);
}

void FORTRAN(sirius_get_max_mt_basis_size)(int32_t* max_mt_basis_size)
{
    log_function_enter(__func__);
    *max_mt_basis_size = global_parameters->unit_cell()->max_mt_basis_size();
    log_function_exit(__func__);
}

void FORTRAN(sirius_get_basis_functions_index)(int32_t* mt_basis_size, int32_t* offset_wf, int32_t* indexb__)
{
    log_function_enter(__func__);
    mdarray<int, 3> indexb(indexb__, 4, global_parameters->unit_cell()->max_mt_basis_size(), global_parameters->unit_cell()->num_atoms()); 

    for (int ia = 0; ia < global_parameters->unit_cell()->num_atoms(); ia++)
    {
        mt_basis_size[ia] = global_parameters->unit_cell()->atom(ia)->type()->mt_basis_size();
        offset_wf[ia] = global_parameters->unit_cell()->atom(ia)->offset_wf();

        for (int j = 0; j < global_parameters->unit_cell()->atom(ia)->type()->mt_basis_size(); j++)
        {
            indexb(0, j, ia) = global_parameters->unit_cell()->atom(ia)->type()->indexb(j).l;
            indexb(1, j, ia) = global_parameters->unit_cell()->atom(ia)->type()->indexb(j).lm + 1; // Fortran counts from 1
            indexb(2, j, ia) = global_parameters->unit_cell()->atom(ia)->type()->indexb(j).idxrf + 1; // Fortran counts from 1
        }
    }
    log_function_exit(__func__);
}

/// Get number of G+k vectors for a given k-point in the set
void FORTRAN(sirius_get_num_gkvec)(int32_t* kset_id, int32_t* ik, int32_t* num_gkvec)
{
    log_function_enter(__func__);
    *num_gkvec = (*kset_list[*kset_id])[*ik - 1]->num_gkvec();
    log_function_exit(__func__);
}

/// Get maximum number of G+k vectors across all k-points in the set
void FORTRAN(sirius_get_max_num_gkvec)(int32_t* kset_id, int32_t* max_num_gkvec)
{
    log_function_enter(__func__);
    *max_num_gkvec = kset_list[*kset_id]->max_num_gkvec();
    log_function_exit(__func__);
}

/// Get all G+k vector related arrays
void FORTRAN(sirius_get_gkvec_arrays)(int32_t* kset_id, int32_t* ik, int32_t* num_gkvec, int32_t* gvec_index, 
                                      double* gkvec__, double* gkvec_cart__, double* gkvec_len, double* gkvec_tp__, 
                                      double_complex* gkvec_phase_factors__, int32_t* ld)
{
    log_function_enter(__func__);

    /* position of processors which store a given k-point */
    int rank = kset_list[*kset_id]->spl_num_kpoints().local_rank(*ik - 1);
    
    Communicator comm_r(global_parameters->mpi_grid().communicator(1 << _dim_row_));
    Communicator comm_k(global_parameters->mpi_grid().communicator(1 << _dim_k_));

    if (rank == comm_k.rank())
    {
        sirius::K_point* kp = (*kset_list[*kset_id])[*ik - 1];
        *num_gkvec = kp->num_gkvec();
        mdarray<double, 2> gkvec(gkvec__, 3, kp->num_gkvec()); 
        mdarray<double, 2> gkvec_cart(gkvec_cart__, 3, kp->num_gkvec());
        mdarray<double, 2> gkvec_tp(gkvec_tp__, 2, kp->num_gkvec()); 

        for (int igk = 0; igk < kp->num_gkvec(); igk++)
        {
            gvec_index[igk] = kp->gvec_index(igk) + 1; //Fortran counts form 1
            for (int x = 0; x < 3; x++) 
            {
                gkvec(x, igk) = kp->gkvec(igk)[x];
                gkvec_cart(x, igk) = kp->gkvec_cart(igk)[x];
            }
            auto rtp = sirius::SHT::spherical_coordinates(kp->gkvec_cart(igk));
            gkvec_len[igk] = rtp[0];
            gkvec_tp(0, igk) = rtp[1];
            gkvec_tp(1, igk) = rtp[2];
        }
        
        mdarray<double_complex, 2> gkvec_phase_factors(gkvec_phase_factors__, *ld, global_parameters->unit_cell()->num_atoms());
        gkvec_phase_factors.zero();
        for (int igkloc = 0; igkloc < kp->num_gkvec_row(); igkloc++)
        {
            int igk = kp->gklo_basis_descriptor_row(igkloc).igk;
            for (int ia = 0; ia < global_parameters->unit_cell()->num_atoms(); ia++)
                gkvec_phase_factors(igk, ia) = kp->gkvec_phase_factor(igkloc, ia);
        }
        comm_r.allreduce(&gkvec_phase_factors(0, 0), (int)gkvec_phase_factors.size()); 
    }
    comm_k.bcast(num_gkvec, 1, rank);
    comm_k.bcast(gvec_index, *num_gkvec, rank);
    comm_k.bcast(gkvec__, *num_gkvec * 3, rank);
    comm_k.bcast(gkvec_cart__, *num_gkvec * 3, rank);
    comm_k.bcast(gkvec_len, *num_gkvec, rank);
    comm_k.bcast(gkvec_tp__, *num_gkvec * 2, rank);
    comm_k.bcast(gkvec_phase_factors__, *ld * global_parameters->unit_cell()->num_atoms(), rank);
    log_function_exit(__func__);
}

void FORTRAN(sirius_get_matching_coefficients)(int32_t* kset_id__, int32_t* ik__, double_complex* apwalm__, 
                                               int32_t* ngkmax__, int32_t* apwordmax__)
{
    log_function_enter(__func__);

    int rank = kset_list[*kset_id__]->spl_num_kpoints().local_rank(*ik__ - 1);
    
    if (rank == global_parameters->mpi_grid().coordinate(0))
    {
        auto kp = (*kset_list[*kset_id__])[*ik__ - 1];
        
        mdarray<double_complex, 4> apwalm(apwalm__, *ngkmax__, *apwordmax__, global_parameters->lmmax_apw(), 
                                          global_parameters->unit_cell()->num_atoms());


        dmatrix<double_complex> alm(kp->num_gkvec_row(), global_parameters->unit_cell()->mt_aw_basis_size(), *blacs_grid);
        kp->alm_coeffs_row()->generate<true>(alm);

        for (int i = 0; i < global_parameters->unit_cell()->mt_aw_basis_size(); i++)
        {
            int ia = global_parameters->unit_cell()->mt_aw_basis_descriptor(i).ia;
            int xi = global_parameters->unit_cell()->mt_aw_basis_descriptor(i).xi;
            
            int lm = global_parameters->unit_cell()->atom(ia)->type()->indexb(xi).lm;
            int order = global_parameters->unit_cell()->atom(ia)->type()->indexb(xi).order;

            for (int igkloc = 0; igkloc < kp->num_gkvec_row(); igkloc++) 
            {
                int igk = kp->gklo_basis_descriptor_row(igkloc).igk;
                apwalm(igk, order, lm, ia) = alm(igkloc, i);
            }
        }
        //== for (int ia = 0; ia < global_parameters->unit_cell()->num_atoms(); ia++)
        //== {
        //==     Platform::allreduce(&apwalm(0, 0, 0, ia), (int)(apwalm.size(0) * apwalm.size(1) * apwalm.size(2)),
        //==                         global_parameters->mpi_grid().communicator(1 << _dim_row_));
        //== }
    }
    log_function_exit(__func__);
}

/// Get first-variational matrices of Hamiltonian and overlap
/** Radial integrals and plane-wave coefficients of the interstitial potential must be calculated prior to
 *  Hamiltonian and overlap matrix construction. 
 */
void FORTRAN(sirius_get_fv_h_o)(int32_t* kset_id__, int32_t* ik__, int32_t* size__, double_complex* h__, double_complex* o__)
{
    int rank = kset_list[*kset_id__]->spl_num_kpoints().local_rank(*ik__ - 1);
    
    if (rank == global_parameters->mpi_grid().coordinate(0))
    {
        auto kp = (*kset_list[*kset_id__])[*ik__ - 1];
        
        if (*size__ != kp->gklo_basis_size())
        {
            error_local(__FILE__, __LINE__, "wrong matrix size");
        }

        dmatrix<double_complex> h(h__, kp->gklo_basis_size(), kp->gklo_basis_size(), *blacs_grid);
        dmatrix<double_complex> o(o__, kp->gklo_basis_size(), kp->gklo_basis_size(), *blacs_grid);
        kset_list[*kset_id__]->band()->set_fv_h_o<cpu, full_potential_lapwlo>(kp, potential->effective_potential(), h, o);  
    }
}

void FORTRAN(sirius_solve_fv)(int32_t* kset_id__, int32_t* ik__, double_complex* h__, double_complex* o__, 
                              double* eval__, double_complex* evec__, int32_t* evec_ld__)
{
    int rank = kset_list[*kset_id__]->spl_num_kpoints().local_rank(*ik__ - 1);
    
    if (rank == global_parameters->mpi_grid().coordinate(0))
    {
        auto kp = (*kset_list[*kset_id__])[*ik__ - 1];
    
        kset_list[*kset_id__]->band()->gen_evp_solver()->solve(kp->gklo_basis_size(),
                                                               kp->gklo_basis_size_row(),
                                                               kp->gklo_basis_size_col(),
                                                               global_parameters->num_fv_states(),
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
void FORTRAN(sirius_get_mtgk_size)(int32_t* kset_id, int32_t* ik, int32_t* mtgk_size)
{
    log_function_enter(__func__);
    *mtgk_size = (*kset_list[*kset_id])[*ik - 1]->wf_size();
    log_function_exit(__func__);
}

void FORTRAN(sirius_get_spinor_wave_functions)(int32_t* kset_id, int32_t* ik, double_complex* spinor_wave_functions__)
{
    log_function_enter(__func__);
    TERMINATE("fix this for distributed WF storage");
    //== assert(global_parameters->num_bands() == (int)global_parameters->spl_spinor_wf().local_size());

    //== sirius::K_point* kp = (*kset_list[*kset_id])[*ik - 1];
    //== 
    //== mdarray<double_complex, 3> spinor_wave_functions(spinor_wave_functions__, kp->wf_size(), global_parameters->num_spins(), 
    //==                                             global_parameters->spl_spinor_wf().local_size());

    //== for (int j = 0; j < (int)global_parameters->spl_spinor_wf().local_size(); j++)
    //== {
    //==     memcpy(&spinor_wave_functions(0, 0, j), &kp->spinor_wave_function(0, 0, j), 
    //==            kp->wf_size() * global_parameters->num_spins() * sizeof(double_complex));
    //== }
    log_function_exit(__func__);
}

void FORTRAN(sirius_apply_step_function_gk)(int32_t* kset_id, int32_t* ik, double_complex* wf__)
{
    log_function_enter(__func__);
    int thread_id = Platform::thread_id();

    sirius::K_point* kp = (*kset_list[*kset_id])[*ik - 1];
    int num_gkvec = kp->num_gkvec();

    global_parameters->reciprocal_lattice()->fft()->input(num_gkvec, kp->fft_index(), wf__, thread_id);
    global_parameters->reciprocal_lattice()->fft()->transform(1, thread_id);
    for (int ir = 0; ir < global_parameters->reciprocal_lattice()->fft()->size(); ir++)
        global_parameters->reciprocal_lattice()->fft()->buffer(ir, thread_id) *= global_parameters->step_function()->theta_it(ir);

    global_parameters->reciprocal_lattice()->fft()->transform(-1, thread_id);
    global_parameters->reciprocal_lattice()->fft()->output(num_gkvec, kp->fft_index(), wf__, thread_id);
    log_function_exit(__func__);
}

/// Get Cartesian coordinates of G+k vectors
void FORTRAN(sirius_get_gkvec_cart)(int32_t* kset_id, int32_t* ik, double* gkvec_cart__)
{
    log_function_enter(__func__);
    sirius::K_point* kp = (*kset_list[*kset_id])[*ik - 1];
    mdarray<double, 2> gkvec_cart(gkvec_cart__, 3, kp->num_gkvec());

    for (int igk = 0; igk < kp->num_gkvec(); igk++)
    {
        for (int x = 0; x < 3; x++) gkvec_cart(x, igk) = kp->gkvec_cart(igk)[x];
    }
    log_function_exit(__func__);
}

void FORTRAN(sirius_get_evalsum)(double* evalsum)
{
    log_function_enter(__func__);
    *evalsum = dft_ground_state->eval_sum();
    log_function_exit(__func__);
}

void FORTRAN(sirius_get_energy_exc)(double* energy_exc)
{
    log_function_enter(__func__);
    *energy_exc = dft_ground_state->energy_exc();
    log_function_exit(__func__);
}

void FORTRAN(sirius_get_energy_vxc)(double* energy_vxc)
{
    log_function_enter(__func__);
    *energy_vxc = dft_ground_state->energy_vxc();
    log_function_exit(__func__);
}

void FORTRAN(sirius_get_energy_bxc)(double* energy_bxc)
{
    log_function_enter(__func__);
    *energy_bxc = dft_ground_state->energy_bxc();
    log_function_exit(__func__);
}

void FORTRAN(sirius_get_energy_veff)(double* energy_veff)
{
    log_function_enter(__func__);
    *energy_veff = dft_ground_state->energy_veff();
    log_function_exit(__func__);
}

void FORTRAN(sirius_get_energy_vha)(double* energy_vha)
{
    log_function_enter(__func__);
    *energy_vha = dft_ground_state->energy_vha();
    log_function_exit(__func__);
}

void FORTRAN(sirius_get_energy_enuc)(double* energy_enuc)
{
    log_function_enter(__func__);
    *energy_enuc = dft_ground_state->energy_enuc();
    log_function_exit(__func__);
}

void FORTRAN(sirius_get_energy_kin)(double* energy_kin)
{
    log_function_enter(__func__);
    *energy_kin = dft_ground_state->energy_kin();
    log_function_exit(__func__);
}

/// Generate XC potential and magnetic field
void FORTRAN(sirius_generate_xc_potential)(double* vxcmt__, double* vxcit__, double* bxcmt__, double* bxcit__)
{
    log_function_enter(__func__);

    potential->xc(density->rho(), density->magnetization(), potential->xc_potential(), potential->effective_magnetic_field(), 
                  potential->xc_energy_density());

    potential->xc_potential()->copy_to_global_ptr(vxcmt__, vxcit__);
 
    if (global_parameters->num_mag_dims() == 0) return;
    assert(global_parameters->num_spins() == 2);

    /* set temporary array wrapper */
    mdarray<double,4> bxcmt(bxcmt__, global_parameters->lmmax_pot(), global_parameters->unit_cell()->max_num_mt_points(), 
                            global_parameters->unit_cell()->num_atoms(), global_parameters->num_mag_dims());
    mdarray<double,2> bxcit(bxcit__, global_parameters->reciprocal_lattice()->fft()->size(), global_parameters->num_mag_dims());

    if (global_parameters->num_mag_dims() == 1)
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
    log_function_exit(__func__);
}

void FORTRAN(sirius_generate_coulomb_potential)(double* vclmt__, double* vclit__)
{
    log_function_enter(__func__);

    potential->poisson(density->rho(), potential->hartree_potential());
    potential->hartree_potential()->copy_to_global_ptr(vclmt__, vclit__);

    log_function_exit(__func__);
}

void FORTRAN(sirius_update_atomic_potential)()
{
    log_function_enter(__func__);
    potential->update_atomic_potential();
    log_function_exit(__func__);
}

void FORTRAN(sirius_scalar_radial_solver)(int32_t* zn, int32_t* l, int32_t* dme, double* enu, int32_t* nr, double* r, 
                                          double* v__, int32_t* nn, double* p0__, double* p1__, double* q0__, double* q1__)
{
    log_function_enter(__func__);
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
    log_function_exit(__func__);
}

void FORTRAN(sirius_get_aw_radial_function)(int32_t* ia__, int32_t* l__, int32_t* io__, double* f__)
{
    log_function_enter(__func__);
    int ia = *ia__ - 1;
    int io = *io__ - 1;
    auto atom = global_parameters->unit_cell()->atom(ia);
    int idxrf = atom->type()->indexr_by_l_order(*l__, io);
    for (int ir = 0; ir < atom->num_mt_points(); ir++) f__[ir] = atom->symmetry_class()->radial_function(ir, idxrf);
    log_function_exit(__func__);
}
    
void FORTRAN(sirius_get_aw_deriv_radial_function)(int32_t* ia__, int32_t* l__, int32_t* io__, double* dfdr__)
{
    log_function_enter(__func__);
    int ia = *ia__ - 1;
    int io = *io__ - 1;
    auto atom = global_parameters->unit_cell()->atom(ia);
    int idxrf = atom->type()->indexr_by_l_order(*l__, io);
    for (int ir = 0; ir < atom->num_mt_points(); ir++)
    {
        double rinv = atom->type()->radial_grid().x_inv(ir);
        dfdr__[ir] = atom->symmetry_class()->r_deriv_radial_function(ir, idxrf) * rinv;
    }
    log_function_exit(__func__);
}
    
void FORTRAN(sirius_get_aw_surface_derivative)(int32_t* ia__, int32_t* l__, int32_t* io__, double* dawrf__)
{
    log_function_enter(__func__);
    *dawrf__ = global_parameters->unit_cell()->atom(*ia__ - 1)->symmetry_class()->aw_surface_dm(*l__, *io__ - 1, 1); 
    log_function_exit(__func__);
}

void FORTRAN(sirius_get_lo_radial_function)(int32_t* ia__, int32_t* idxlo__, double* f__)
{
    log_function_enter(__func__);
    int ia = *ia__ - 1;
    int idxlo = *idxlo__ - 1;
    auto atom = global_parameters->unit_cell()->atom(ia);
    int idxrf = atom->type()->indexr_by_idxlo(idxlo);
    for (int ir = 0; ir < atom->num_mt_points(); ir++) f__[ir] = atom->symmetry_class()->radial_function(ir, idxrf);
    log_function_exit(__func__);
}
    
void FORTRAN(sirius_get_lo_deriv_radial_function)(int32_t* ia__, int32_t* idxlo__, double* dfdr__)
{
    log_function_enter(__func__);
    int ia = *ia__ - 1;
    int idxlo = *idxlo__ - 1;
    auto atom = global_parameters->unit_cell()->atom(ia);
    int idxrf = atom->type()->indexr_by_idxlo(idxlo);
    for (int ir = 0; ir < atom->num_mt_points(); ir++)
    {
        double rinv = atom->type()->radial_grid().x_inv(ir);
        dfdr__[ir] = atom->symmetry_class()->r_deriv_radial_function(ir, idxrf) * rinv;
    }
    log_function_exit(__func__);
}
    
void FORTRAN(sirius_get_aw_lo_o_radial_integral)(int32_t* ia__, int32_t* l, int32_t* io1, int32_t* ilo2, 
                                                 double* oalo)
{
    log_function_enter(__func__);
    int ia = *ia__ - 1;

    int idxrf2 = global_parameters->unit_cell()->atom(ia)->type()->indexr_by_idxlo(*ilo2 - 1);
    int order2 = global_parameters->unit_cell()->atom(ia)->type()->indexr(idxrf2).order;

    *oalo = global_parameters->unit_cell()->atom(ia)->symmetry_class()->o_radial_integral(*l, *io1 - 1, order2);
    log_function_exit(__func__);
}

void FORTRAN(sirius_get_lo_lo_o_radial_integral)(int32_t* ia__, int32_t* l, int32_t* ilo1, int32_t* ilo2, 
                                                 double* ololo)
{
    log_function_enter(__func__);
    int ia = *ia__ - 1;

    int idxrf1 = global_parameters->unit_cell()->atom(ia)->type()->indexr_by_idxlo(*ilo1 - 1);
    int order1 = global_parameters->unit_cell()->atom(ia)->type()->indexr(idxrf1).order;
    int idxrf2 = global_parameters->unit_cell()->atom(ia)->type()->indexr_by_idxlo(*ilo2 - 1);
    int order2 = global_parameters->unit_cell()->atom(ia)->type()->indexr(idxrf2).order;

    *ololo = global_parameters->unit_cell()->atom(ia)->symmetry_class()->o_radial_integral(*l, order1, order2);
    log_function_exit(__func__);
}

void FORTRAN(sirius_get_aw_aw_h_radial_integral)(int32_t* ia__, int32_t* l1, int32_t* io1, int32_t* l2, 
                                                 int32_t* io2, int32_t* lm3, double* haa)
{
    log_function_enter(__func__);
    int ia = *ia__ - 1;
    int idxrf1 = global_parameters->unit_cell()->atom(ia)->type()->indexr_by_l_order(*l1, *io1 - 1);
    int idxrf2 = global_parameters->unit_cell()->atom(ia)->type()->indexr_by_l_order(*l2, *io2 - 1);

    *haa = global_parameters->unit_cell()->atom(ia)->h_radial_integrals(idxrf1, idxrf2)[*lm3 - 1];
    log_function_exit(__func__);
}

void FORTRAN(sirius_get_lo_aw_h_radial_integral)(int32_t* ia__, int32_t* ilo1, int32_t* l2, int32_t* io2, int32_t* lm3, 
                                                 double* hloa)
{
    log_function_enter(__func__);
    int ia = *ia__ - 1;
    int idxrf1 = global_parameters->unit_cell()->atom(ia)->type()->indexr_by_idxlo(*ilo1 - 1);
    int idxrf2 = global_parameters->unit_cell()->atom(ia)->type()->indexr_by_l_order(*l2, *io2 - 1);

    *hloa = global_parameters->unit_cell()->atom(ia)->h_radial_integrals(idxrf1, idxrf2)[*lm3 - 1];
    log_function_exit(__func__);
}


void FORTRAN(sirius_get_lo_lo_h_radial_integral)(int32_t* ia__, int32_t* ilo1, int32_t* ilo2, int32_t* lm3, 
                                                 double* hlolo)
{
    log_function_enter(__func__);
    int ia = *ia__ - 1;
    int idxrf1 = global_parameters->unit_cell()->atom(ia)->type()->indexr_by_idxlo(*ilo1 - 1);
    int idxrf2 = global_parameters->unit_cell()->atom(ia)->type()->indexr_by_idxlo(*ilo2 - 1);

    *hlolo = global_parameters->unit_cell()->atom(ia)->h_radial_integrals(idxrf1, idxrf2)[*lm3 - 1];
    log_function_exit(__func__);
}

void FORTRAN(sirius_generate_potential_pw_coefs)(void)
{
    log_function_enter(__func__);
    potential->generate_pw_coefs();
    log_function_exit(__func__);
}

/// Get first-variational eigen-vectors
/** Assume that the Fortran side holds the whole array */
void FORTRAN(sirius_get_fv_eigen_vectors)(int32_t* kset_id__, int32_t* ik__, double_complex* fv_evec__, int32_t* ld__, 
                                          int32_t* num_fv_evec__)
{
    log_function_enter(__func__);
    mdarray<double_complex, 2> fv_evec(fv_evec__, *ld__, *num_fv_evec__);
    (*kset_list[*kset_id__])[*ik__ - 1]->get_fv_eigen_vectors(fv_evec);
    log_function_exit(__func__);
}

/// Get second-variational eigen-vectors
/** Assume that the Fortran side holds the whole array */
void FORTRAN(sirius_get_sv_eigen_vectors)(int32_t* kset_id, int32_t* ik, double_complex* sv_evec__, int32_t* size)
{
    log_function_enter(__func__);
    mdarray<double_complex, 2> sv_evec(sv_evec__, *size, *size);
    (*kset_list[*kset_id])[*ik - 1]->get_sv_eigen_vectors(sv_evec);
    log_function_exit(__func__);
}

void FORTRAN(sirius_get_num_fv_states)(int32_t* num_fv_states__)
{
    log_function_enter(__func__);
    *num_fv_states__ = global_parameters->num_fv_states();
    log_function_exit(__func__);
}

void FORTRAN(sirius_ground_state_initialize)(int32_t* kset_id)
{
    log_function_enter(__func__);
    if (dft_ground_state) error_local(__FILE__, __LINE__, "dft_ground_state object is already allocate");

    dft_ground_state = new sirius::DFT_ground_state(*global_parameters, potential, density, kset_list[*kset_id]);
    log_function_exit(__func__);
}

void FORTRAN(sirius_ground_state_clear)()
{
    log_function_enter(__func__);
    delete dft_ground_state;
    dft_ground_state = NULL;
    log_function_exit(__func__);
}

void FORTRAN(sirius_get_mpi_comm)(int32_t* directions__, int32_t* fcomm__)
{
    log_function_enter(__func__);
    *fcomm__ = MPI_Comm_c2f(global_parameters->mpi_grid().communicator(*directions__).mpi_comm());
    log_function_exit(__func__);
}

void FORTRAN(sirius_forces)(double* forces__)
{
    log_function_enter(__func__);
    mdarray<double, 2> forces(forces__, 3, global_parameters->unit_cell()->num_atoms()); 
    dft_ground_state->forces(forces);
    log_function_exit(__func__);
}

void FORTRAN(sirius_set_atom_pos)(int32_t* atom_id, double* pos)
{
    log_function_enter(__func__);
    global_parameters->unit_cell()->atom(*atom_id - 1)->set_position(pos);
    log_function_exit(__func__);
}

void FORTRAN(sirius_update)(int32_t* kset_id)
{
    log_function_enter(__func__);
    global_parameters->update();
    potential->update();
    kset_list[*kset_id]->update();
    log_function_exit(__func__);
}

void FORTRAN(sirius_core_leakage)(double* core_leakage)
{
    *core_leakage = density->core_leakage();
}

void FORTRAN(sirius_ground_state_print_info)(void)
{
    dft_ground_state->print_info();
}

void FORTRAN(sirius_create_storage_file)(void)
{
    global_parameters->create_storage_file();
}

void FORTRAN(sirius_test_spinor_wave_functions)(int32_t* kset_id)
{
    sirius::K_set* kset = kset_list[*kset_id];
    for (int ikloc = 0; ikloc < (int)kset->spl_num_kpoints().local_size(); ikloc++)
    {
        int ik = kset->spl_num_kpoints(ikloc);
        (*kset)[ik]->test_spinor_wave_functions(0);
    }
}

void FORTRAN(sirius_generate_gq_matrix_elements)(int32_t* kset_id, double* vq)
{
     kset_list[*kset_id]->generate_Gq_matrix_elements(vq);
}

void FORTRAN(sirius_density_mixer_initialize)(void)
{
    if (global_parameters->mixer_input_section_.type_ == "broyden")
    {
        mixer_rho = new sirius::Broyden_mixer(density->size(), global_parameters->mixer_input_section_.max_history_, 
                                              global_parameters->mixer_input_section_.beta_, global_parameters->comm());
    }
    else if (global_parameters->mixer_input_section_.type_ == "linear")
    {
        mixer_rho = new sirius::Linear_mixer(density->size(), global_parameters->mixer_input_section_.beta_, global_parameters->comm());
    }
    else
    {
        error_global(__FILE__, __LINE__, "Wrong mixer type");
    }
    
    /* initialize density mixer with starting density */
    density->pack(mixer_rho);
    mixer_rho->initialize();
}

void FORTRAN(sirius_potential_mixer_initialize)(void)
{
    if (global_parameters->mixer_input_section_.type_ == "linear")
    {
        mixer_pot = new sirius::Linear_mixer(potential->size(), global_parameters->mixer_input_section_.gamma_, global_parameters->comm());

        /* initialize potential mixer */
        potential->pack(mixer_pot);
        mixer_pot->initialize();
    }
}

void FORTRAN(sirius_mix_density)(double* rms)
{
    density->pack(mixer_rho);
    *rms = mixer_rho->mix();
    density->unpack(mixer_rho->output_buffer());
    global_parameters->comm().bcast(rms, 1, 0);
}

void FORTRAN(sirius_mix_potential)(void)
{
    if (mixer_pot)
    {
        potential->pack(mixer_pot);
        mixer_pot->mix();
        potential->unpack(mixer_pot->output_buffer());
    }
}
    

} // extern "C"
