#include "sirius.h"

/** \file sirius.cpp
    \brief Fortran API
*/

/// pointer to Density class, implicitly used by Fortran side
sirius::Density* density = NULL;

/// pointer to Potential class, implicitly used by Fortran side
sirius::Potential* potential = NULL;

/// set of global parameters
sirius::Global global_parameters;

/// list of pointers to the sets of k-points
std::vector<sirius::K_set*> kset_list;

extern "C" 
{

/// Set lattice vectors
/** \param [in] a1 1st lattice vector
    \param [in] a2 2nd lattice vector
    \param [in] a3 3rd lattice vector

    Example:
    \code{.F90}
        real(8) a1(3),a2(3),a3(3)
        a1(:) = (/5.d0, 0.d0, 0.d0/)
        a2(:) = (/0.d0, 5.d0, 0.d0/)
        a3(:) = (/0.d0, 0.d0, 5.d0/)
        call sirius_set_lattice_vectors(a1, a2, a3)
    \endcode
*/
void FORTRAN(sirius_set_lattice_vectors)(real8* a1, real8* a2, real8* a3)
{
    global_parameters.set_lattice_vectors(a1, a2, a3);
}

/// Set maximum l-value for augmented waves
/** \param [in] lmax_apw maximum l-value for augmented waves

    Example:
    \code{.F90}
        integer lmaxapw
        lmaxapw = 10
        call sirius_set_lmax_apw(lmaxapw)
    \endcode
*/
void FORTRAN(sirius_set_lmax_apw)(int32_t* lmax_apw)
{
    global_parameters.set_lmax_apw(*lmax_apw);
}

/// Set maximum l-value for density expansion
/** \param [in] lmax_rho maximum l-value for density expansion
    
    Example:
    \code{.F90}
        integer lmaxrho
        lmaxrho = 8
        call sirius_set_lmax_rho(lmaxrho)
    \endcode
*/
void FORTRAN(sirius_set_lmax_rho)(int32_t* lmax_rho)
{
    global_parameters.set_lmax_rho(*lmax_rho);
}

/// Set maximum l-value for potential expansion
/** \param [in] lmax_pot maximum l-value for potential expansion
    
    Example:
    \code{.F90}
        integer lmaxpot
        lmaxrho = 8
        call sirius_set_lmax_pot(lmaxpot)
    \endcode
*/
void FORTRAN(sirius_set_lmax_pot)(int32_t* lmax_pot)
{
    global_parameters.set_lmax_pot(*lmax_pot);
}

/// Set plane-wave cutoff for FFT grid
/** \param [in] gmaxvr maximum G-vector length 

    Example:
    \code{.F90}
        real(8) gmaxvr
        gmaxvr = 20.0
        call sirius_set_pw_cutoff(gmaxvr)
    \endcode
*/
void FORTRAN(sirius_set_pw_cutoff)(real8* pw_cutoff)
{
    global_parameters.set_pw_cutoff(*pw_cutoff);
}

/// Set the number of spins
/** \param [in] num_spins number of spins (1 or 2)
    
    The default number of spins is 1 (non-magnetic treatment of electrons).

    Example:
    \code{.F90}
        if (spinpol) call sirius_set_num_spins(2)
    \endcode
*/
void FORTRAN(sirius_set_num_spins)(int32_t* num_spins)
{
    global_parameters.set_num_spins(*num_spins);
}

/// Set the number of magnetic dimensions
/** \param [in] num_mag_dims number of magnetic dimensions (0, 1, or 3)
    
    In case of spin-polarized calculation magnetization density may have only one (z) component 
    (num_mag_dims = 1, collinear case) or all three components (num_mag_dims = 3, non-collinear case).
    For non magnetic calcualtions num_mag_dims = 0.

    Example:
    \code{.F90}
        integer ndmag
        ndmag = 3
        call sirius_set_num_mag_dims(ndmag)
    \endcode
*/
void FORTRAN(sirius_set_num_mag_dims)(int32_t* num_mag_dims)
{
    global_parameters.set_num_mag_dims(*num_mag_dims);
}

/// Turn on or off the automatic scaling of muffin-tin spheres
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
    global_parameters.set_auto_rmt(*auto_rmt);
}

/// Add atom type to the library
/** \param [in] atom_type_id unique id of atom type
    \param [in] label atom type label

    Atom type (species in the terminology of Exciting/Elk) is a class which holds information 
    common to the atoms of the same element: charge, number of core and valence electrons, muffin-tin
    radius, radial grid etc. See Atom_type class for details.

    Example:
    \code{.F90}
        do is = 1, nspecies
          !======================================================
          ! add atom type with ID=is and read the .json file with 
          ! the symbol name if it exists
          !======================================================
          call sirius_add_atom_type(is, trim(spsymb(is))
        enddo
    \endcode
*/
void FORTRAN(sirius_add_atom_type)(int32_t* atom_type_id, char* label, int32_t label_len)
{
    global_parameters.add_atom_type(*atom_type_id, std::string(label, label_len));
}

/// Set basic properties of the atom type
/** \param [in] atom_type_id id of the atom type
*/ 
void FORTRAN(sirius_set_atom_type_properties)(int32_t* atom_type_id, char* symbol, int32_t* zn, real8* mass, 
                                              real8* mt_radius, int32_t* num_mt_points, real8* radial_grid_origin, 
                                              real8* radial_grid_infinity, int32_t symbol_len)
{
    sirius::Atom_type* type = global_parameters.atom_type_by_id(*atom_type_id);
    type->set_symbol(std::string(symbol, symbol_len));
    type->set_zn(*zn);
    type->set_mass(*mass);
    type->set_num_mt_points(*num_mt_points);
    type->set_radial_grid_origin(*radial_grid_origin);
    type->set_radial_grid_infinity(*radial_grid_infinity);
    type->set_mt_radius(*mt_radius);
}


/// Add atom to the library
/** \param [in] atom_type_id id of the atom type
    \param [in] position atom position in fractional coordinates
    \param [in] vector_field vector field associated with the given atom

    Example:
    \code{.F90}
        do is = 1, nspecies
          do ia = 1, natoms(is)
            call sirius_add_atom(is, atposl(:, ia, is), bfcmt(:, ia, is))
          enddo
        enddo
    \endcode
*/
void FORTRAN(sirius_add_atom)(int32_t* atom_type_id, real8* position, real8* vector_field)
{
    global_parameters.add_atom(*atom_type_id, position, vector_field);
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
void FORTRAN(sirius_set_aw_cutoff)(real8* aw_cutoff)
{
    global_parameters.set_aw_cutoff(*aw_cutoff);
}

void FORTRAN(sirius_set_charge_density_ptr)(real8* rhomt, real8* rhoit)
{
    density->set_charge_density_ptr(rhomt, rhoit);
}

void FORTRAN(sirius_set_magnetization_ptr)(real8* magmt, real8* magit)
{
    density->set_magnetization_ptr(magmt, magit);
}

void FORTRAN(sirius_set_effective_potential_ptr)(real8* veffmt, real8* veffir)
{
    potential->set_effective_potential_ptr(veffmt, veffir);
}

void FORTRAN(sirius_set_effective_magnetic_field_ptr)(real8* beffmt, real8* beffir)
{
    potential->set_effective_magnetic_field_ptr(beffmt, beffir);
}

void FORTRAN(sirius_set_equivalent_atoms)(int32_t* equivalent_atoms)
{
    global_parameters.set_equivalent_atoms(equivalent_atoms);
}


/*
    primitive get functions
*/
void FORTRAN(sirius_get_max_num_mt_points)(int32_t* max_num_mt_points)
{
    *max_num_mt_points = global_parameters.max_num_mt_points();
}

void FORTRAN(sirius_get_num_mt_points)(int32_t* atom_type_id, int32_t* num_mt_points)
{
    *num_mt_points = global_parameters.atom_type_by_id(*atom_type_id)->num_mt_points();
}

void FORTRAN(sirius_get_mt_points)(int32_t* atom_type_id, real8* mt_points)
{
    memcpy(mt_points, global_parameters.atom_type_by_id(*atom_type_id)->radial_grid().get_ptr(),
        global_parameters.atom_type_by_id(*atom_type_id)->num_mt_points() * sizeof(real8));
}

void FORTRAN(sirius_get_num_grid_points)(int32_t* num_grid_points)
{
    *num_grid_points = global_parameters.fft().size();
}

void FORTRAN(sirius_get_num_bands)(int32_t* num_bands)
{
    *num_bands = global_parameters.num_bands();
}

/// Get number of G-vectors within the plane-wave cutoff
void FORTRAN(sirius_get_num_gvec)(int32_t* num_gvec)
{
    *num_gvec = global_parameters.num_gvec();
}

/// Get sizes of FFT grid
void FORTRAN(sirius_get_fft_grid_size)(int32_t* grid_size)
{
    grid_size[0] = global_parameters.fft().size(0);
    grid_size[1] = global_parameters.fft().size(1);
    grid_size[2] = global_parameters.fft().size(2);
}

/// Get lower and upper limits of the FFT grid dimension
/** \param [in] d index of dimension (1,2, or 3)
    \param [out] lower lower (most negative) value
    \param [out] upper upper (most positive) value

    Example:
    \code{.F90}
        do i=1,3
          call sirius_get_fft_grid_limits(i,intgv(i,1),intgv(i,2))
        enddo
    \endcode
*/
void FORTRAN(sirius_get_fft_grid_limits)(int32_t* d, int32_t* lower, int32_t* upper)
{
    *lower = global_parameters.fft().grid_limits(*d - 1, 0);
    *upper = global_parameters.fft().grid_limits(*d - 1, 1);
}

/// Get mapping between G-vector index and FFT index
void FORTRAN(sirius_get_fft_index)(int32_t* fft_index)
{
    memcpy(fft_index, global_parameters.fft_index(),  global_parameters.fft().size() * sizeof(int32_t));
    for (int i = 0; i < global_parameters.fft().size(); i++) fft_index[i]++;
}

/// Get list of G-vectors in fractional corrdinates
void FORTRAN(sirius_get_gvec)(int32_t* gvec)
{
    memcpy(gvec, global_parameters.gvec(0), 3 * global_parameters.fft().size() * sizeof(int32_t));
}

/// Get list of G-vectors in Cartesian coordinates
void FORTRAN(sirius_get_gvec_cart)(real8* gvec_cart__)
{
    mdarray<double, 2> gvec_cart(gvec_cart__, 3,  global_parameters.fft().size());
    for (int ig = 0; ig <  global_parameters.fft().size(); ig++)
        global_parameters.get_coordinates<cartesian, reciprocal>(global_parameters.gvec(ig), &gvec_cart(0, ig));
}

/// Get lengh of G-vectors
void FORTRAN(sirius_get_gvec_len)(real8* gvec_len)
{
    for (int ig = 0; ig <  global_parameters.fft().size(); ig++) gvec_len[ig] = global_parameters.gvec_len(ig);
}

void FORTRAN(sirius_get_index_by_gvec)(int32_t* index_by_gvec)
{
    memcpy(index_by_gvec, global_parameters.index_by_gvec(), global_parameters.fft().size() * sizeof(int32_t));
    for (int i = 0; i < global_parameters.fft().size(); i++) index_by_gvec[i]++;
}

void FORTRAN(sirius_get_gvec_ylm)(complex16* gvec_ylm__, int* ld, int* lmax)
{
    mdarray<complex16, 2> gvec_ylm(gvec_ylm__, *ld, global_parameters.num_gvec());
    for (int ig = 0; ig < global_parameters.num_gvec(); ig++)
    {
        global_parameters.gvec_ylm_array<global>(ig, &gvec_ylm(0, ig), *lmax);
    }
}

void FORTRAN(sirius_get_gvec_phase_factors)(complex16* sfacg__)
{
    mdarray<complex16, 2> sfacg(sfacg__, global_parameters.num_gvec(), global_parameters.num_atoms());
    for (int ia = 0; ia < global_parameters.num_atoms(); ia++)
    {
        for (int ig = 0; ig < global_parameters.num_gvec(); ig++)
            sfacg(ig, ia) = global_parameters.gvec_phase_factor<global>(ig, ia);
    }
}

void FORTRAN(sirius_get_step_function)(complex16* cfunig, real8* cfunir)
{
    for (int i = 0; i < global_parameters.fft().size(); i++)
    {
        cfunig[i] = global_parameters.step_function_pw(i);
        cfunir[i] = global_parameters.step_function(i);
    }
}

/// Get the total number of electrons
void FORTRAN(sirius_get_num_electrons)(real8* num_electrons)
{
    *num_electrons = global_parameters.num_electrons();
}

/// Get the number of valence electrons
void FORTRAN(sirius_get_num_valence_electrons)(real8* num_valence_electrons)
{
    *num_valence_electrons = global_parameters.num_valence_electrons();
}

/// Get the number of core electrons
void FORTRAN(sirius_get_num_core_electrons)(real8* num_core_electrons)
{
    *num_core_electrons = global_parameters.num_core_electrons();
}


/// Initialize the low-level of the library
void FORTRAN(sirius_platform_initialize)(int32_t* call_mpi_init_)
{
    bool call_mpi_init = (*call_mpi_init_ != 0) ? true : false; 
    Platform::initialize(call_mpi_init);
}

/// Initialize the global variables
void FORTRAN(sirius_global_initialize)(int32_t* init_aw_descriptors)
{
    global_parameters.initialize(*init_aw_descriptors);
}

void FORTRAN(sirius_potential_initialize)(void)
{
    potential = new sirius::Potential(global_parameters);
}

void FORTRAN(sirius_density_initialize)(void)
{
    density = new sirius::Density(global_parameters);
}

/// Clear the global variables and destroy all objects
void FORTRAN(sirius_clear)(void)
{
    global_parameters.clear();
    
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
}

void FORTRAN(sirius_initial_density)(void)
{
    density->initial_density(0);
}

void FORTRAN(sirius_generate_effective_potential)(void)
{
    potential->generate_effective_potential(density->rho(), density->magnetization());
}

void FORTRAN(sirius_generate_density)(int32_t* kset_id)
{
    density->generate(*kset_list[*kset_id]);
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
void FORTRAN(sirius_find_eigen_states)(int32_t* kset_id, int32_t* precompute__)
{
    bool precompute = (*precompute__) ? true : false;
    kset_list[*kset_id]->find_eigen_states(potential, precompute);
}

void FORTRAN(sirius_find_band_occupancies)(int32_t* kset_id)
{
    kset_list[*kset_id]->find_band_occupancies();
}

void FORTRAN(sirius_set_band_occupancies)(int32_t* kset_id, int32_t* ik_, real8* band_occupancies)
{
    int ik = *ik_ - 1;
    kset_list[*kset_id]->set_band_occupancies(ik, band_occupancies);
}

void FORTRAN(sirius_get_band_energies)(int32_t* kset_id, int32_t* ik__, real8* band_energies)
{
    int ik = *ik__ - 1;
    kset_list[*kset_id]->get_band_energies(ik, band_energies);
}

void FORTRAN(sirius_get_band_occupancies)(int32_t* kset_id, int32_t* ik_, real8* band_occupancies)
{
    int ik = *ik_ - 1;
    kset_list[*kset_id]->get_band_occupancies(ik, band_occupancies);
}

void FORTRAN(sirius_integrate_density)(void)
{
    density->integrate();
}

/*
    print info
*/
void FORTRAN(sirius_print_info)(void)
{
    global_parameters.print_info();
    for (int i = 0; i < (int)kset_list.size(); i++) if (kset_list[i]) kset_list[i]->print_info();
}

void FORTRAN(sirius_print_timers)(void)
{
   sirius::Timer::print();
}   

void FORTRAN(sirius_start_timer)(char* name_, int32_t name_len)
{
    std::string name(name_, name_len);
    sirius::ftimers[name] = new sirius::Timer(name);
}

void FORTRAN(sirius_stop_timer)(char* name_, int32_t name_len)
{
    std::string name(name_, name_len);
    if (sirius::ftimers.count(name)) delete sirius::ftimers[name];
}

void FORTRAN(sirius_read_state)()
{
    // TODO: save and load the potential of free atoms
    global_parameters.solve_free_atoms();
    potential->hdf5_read();
    potential->update_atomic_potential();
    sirius:: hdf5_tree fout("sirius.h5", false);
    fout.read("energy_fermi", &global_parameters.rti().energy_fermi);
}

void FORTRAN(sirius_save_potential)()
{
    if (Platform::mpi_rank() == 0) 
    {
        // create new hdf5 file
        sirius::hdf5_tree fout("sirius.h5", true);
        fout.create_node("parameters");
        fout.create_node("kpoints");
        fout.create_node("effective_potential");
        fout.create_node("effective_magnetic_field");
        
        // write Fermi energy
        fout.write("energy_fermi", &global_parameters.rti().energy_fermi);
        
        // write potential
        potential->effective_potential()->hdf5_write(fout["effective_potential"]);

        // write magnetic field
        for (int j = 0; j < global_parameters.num_mag_dims(); j++)
            potential->effective_magnetic_field(j)->hdf5_write(fout["effective_magnetic_field"].create_node(j));
        
    }
    Platform::barrier();
}

void FORTRAN(sirius_save_wave_functions)(int32_t* kset_id)
{
    kset_list[*kset_id]->save_wave_functions();
}
    
void FORTRAN(sirius_load_wave_functions)(int32_t* kset_id)
{
    kset_list[*kset_id]->load_wave_functions();
}

/*  Relevant block in the input file:

    "bz_path" : {
        "num_steps" : 100,
        "points" : [["G", [0, 0, 0]], ["X", [0.5, 0.0, 0.5]], ["L", [0.5, 0.5, 0.5]]]
    }
*/
void FORTRAN(sirius_bands)(void)
{
    FORTRAN(sirius_read_state)();

    std::vector<std::pair<std::string, std::vector<double> > > bz_path;
    std::string fname("sirius.json");
            
    int num_steps = 0;
    if (Utils::file_exists(fname))
    {
        JsonTree parser(fname);
        if (!parser["bz_path"].empty())
        {
            num_steps = parser["bz_path"]["num_steps"].get<int>();

            for (int ipt = 0; ipt < parser["bz_path"]["points"].size(); ipt++)
            {
                std::pair<std::string, std::vector<double> > pt;
                pt.first = parser["bz_path"]["points"][ipt][0].get<std::string>();
                pt.second = parser["bz_path"]["points"][ipt][1].get<std::vector<double> >();
                bz_path.push_back(pt);
            }
        }
    }

    if (bz_path.size() < 2) error(__FILE__, __LINE__, "at least two BZ points are required");
   
    // compute length of segments
    std::vector<double> segment_length;
    double total_path_length = 0.0;
    for (int ip = 0; ip < (int)bz_path.size() - 1; ip++)
    {
        double vf[3];
        for (int x = 0; x < 3; x++) vf[x] = bz_path[ip + 1].second[x] - bz_path[ip].second[x];
        double vc[3];
        global_parameters.get_coordinates<cartesian, reciprocal>(vf, vc);
        double length = Utils::vector_length(vc);
        total_path_length += length;
        segment_length.push_back(length);
    }

    std::vector<double> xaxis;

    sirius::K_set kset_(global_parameters);
    
    double prev_seg_len = 0.0;

    // segments 
    for (int ip = 0; ip < (int)bz_path.size() - 1; ip++)
    {
        std::vector<double> p0 = bz_path[ip].second;
        std::vector<double> p1 = bz_path[ip + 1].second;

        int n = int((segment_length[ip] * num_steps) / total_path_length);
        int n0 = (ip == (int)bz_path.size() - 2) ? n - 1 : n;
        
        double dvf[3];
        for (int x = 0; x < 3; x++) dvf[x] = (p1[x] - p0[x]) / double(n0);
        
        for (int i = 0; i < n; i++)
        {
            double vf[3];
            for (int x = 0; x < 3; x++) vf[x] = p0[x] + dvf[x] * i;
            kset_.add_kpoint(vf, 0.0);

            xaxis.push_back(prev_seg_len + segment_length[ip] * i / double(n0));
        }
        prev_seg_len += segment_length[ip];
    }

    std::vector<double> xaxis_ticks;
    std::vector<std::string> xaxis_tick_labels;
    prev_seg_len = 0.0;
    for (int ip = 0; ip < (int)bz_path.size(); ip++)
    {
        xaxis_ticks.push_back(prev_seg_len);
        xaxis_tick_labels.push_back(bz_path[ip].first);
        if (ip < (int)bz_path.size() - 1) prev_seg_len += segment_length[ip];
    }

    kset_.initialize();

    global_parameters.solve_free_atoms();

    potential->update_atomic_potential();
    global_parameters.generate_radial_functions();
    global_parameters.generate_radial_integrals();

    // generate plane-wave coefficients of the potential in the interstitial region
    potential->generate_pw_coefs();

    sirius::Band* band = kset_.band();
    
    for (int ikloc = 0; ikloc < kset_.spl_num_kpoints().local_size(); ikloc++)
    {
        int ik = kset_.spl_num_kpoints(ikloc);
        kset_[ik]->find_eigen_states(band, potential->effective_potential(), potential->effective_magnetic_field());
    } 
    // synchronize eigen-values
    kset_.sync_band_energies();

    if (global_parameters.mpi_grid().root())
    {
        JSON_write jw("bands.json");
        jw.single("xaxis", xaxis);
        jw.single("Ef", global_parameters.rti().energy_fermi);
        
        jw.single("xaxis_ticks", xaxis_ticks);
        jw.single("xaxis_tick_labels", xaxis_tick_labels);
        
        jw.begin_array("plot");
        std::vector<double> yvalues(kset_.num_kpoints());
        for (int i = 0; i < global_parameters.num_bands(); i++)
        {
            jw.begin_set();
            for (int ik = 0; ik < kset_.num_kpoints(); ik++) yvalues[ik] = kset_[ik]->band_energy(i);
            jw.single("yvalues", yvalues);
            jw.end_set();
        }
        jw.end_array();

        //FILE* fout = fopen("bands.dat", "w");
        //for (int i = 0; i < global_parameters.num_bands(); i++)
        //{
        //    for (int ik = 0; ik < kpoint_set_.num_kpoints(); ik++)
        //    {
        //        fprintf(fout, "%f %f\n", xaxis[ik], kpoint_set_[ik]->band_energy(i));
        //    }
        //    fprintf(fout, "\n");
        //}
        //fclose(fout);
    }
}

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
//    global_parameters.fft().input(potential->effective_potential()->f_it());
//    global_parameters.fft().transform(-1);
//    global_parameters.fft().output(global_parameters.num_gvec(), global_parameters.fft_index(), 
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
//        global_parameters.get_coordinates<cartesian, direct>(vf, vc);
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

void FORTRAN(sirius_print_rti)(void)
{
    global_parameters.print_rti();
}

void FORTRAN(sirius_write_json_output)(void)
{
    global_parameters.write_json_output();
}

void FORTRAN(sirius_get_occupation_matrix)(int32_t* atom_id, complex16* occupation_matrix)
{
    int ia = *atom_id - 1;
    global_parameters.atom(ia)->get_occupation_matrix(occupation_matrix);
}

void FORTRAN(sirius_set_uj_correction_matrix)(int32_t* atom_id, int32_t* l, complex16* uj_correction_matrix)
{
    int ia = *atom_id - 1;
    global_parameters.atom(ia)->set_uj_correction_matrix(*l, uj_correction_matrix);
}

void FORTRAN(sirius_set_so_correction)(int32_t* so_correction)
{
    if (*so_correction != 0) 
    {
        global_parameters.set_so_correction(true);
    }
    else
    {
        global_parameters.set_so_correction(false);
    }
}

void FORTRAN(sirius_set_uj_correction)(int32_t* uj_correction)
{
    if (*uj_correction != 0)
    {
        global_parameters.set_uj_correction(true);
    }
    else
    {
        global_parameters.set_uj_correction(false);
    }
}

void FORTRAN(sirius_platform_mpi_rank)(int32_t* rank)
{
    *rank = Platform::mpi_rank();
}

void FORTRAN(sirius_platform_mpi_grid_rank)(int32_t* dimension, int32_t* rank)
{
    *rank = global_parameters.mpi_grid().coordinate(*dimension);
}

void FORTRAN(sirius_platform_mpi_grid_barrier)(int32_t* dimension)
{
    global_parameters.mpi_grid().barrier(1 << (*dimension));
}

void FORTRAN(sirius_global_set_sync_flag)(int32_t* flag)
{
    global_parameters.set_sync_flag(*flag);
}

void FORTRAN(sirius_global_get_sync_flag)(int32_t* flag)
{
    *flag = global_parameters.sync_flag();
}

void FORTRAN(sirius_platform_barrier)(void)
{
    Platform::barrier();
}

void FORTRAN(sirius_get_total_energy)(real8* total_energy)
{
    *total_energy = global_parameters.total_energy();
}


void FORTRAN(sirius_set_atom_type_radial_grid)(int32_t* atom_type_id, int32_t* num_radial_points, 
                                               int32_t* num_mt_points, real8* radial_points)
{
    sirius::Atom_type* type = global_parameters.atom_type_by_id(*atom_type_id);
    type->radial_grid().set_radial_points(*num_radial_points, *num_mt_points, radial_points);
}

void FORTRAN(sirius_set_atom_type_configuration)(int32_t* atom_type_id, int32_t* n, int32_t* l, int32_t* k, 
                                                 real8* occupancy, int32_t* core_)
{
    sirius::Atom_type* type = global_parameters.atom_type_by_id(*atom_type_id);
    bool core = *core_;
    type->set_configuration(*n, *l, *k, *occupancy, core);
}

void FORTRAN(sirius_add_atom_type_aw_descriptor)(int32_t* atom_type_id, int32_t* n, int32_t* l, real8* enu, 
                                                 int32_t* dme, int32_t* auto_enu__)
{
    sirius::Atom_type* type = global_parameters.atom_type_by_id(*atom_type_id);
    int auto_enu = (*auto_enu__) ? 2 : 0;
    type->add_aw_descriptor(*n, *l, *enu, *dme, auto_enu);
}

void FORTRAN(sirius_add_atom_type_lo_descriptor)(int32_t* atom_type_id, int32_t* ilo, int32_t* n, int32_t* l, 
                                                 real8* enu, int32_t* dme, int32_t* auto_enu__)
{
    sirius::Atom_type* type = global_parameters.atom_type_by_id(*atom_type_id);
    int auto_enu = (*auto_enu__) ? 2 : 0;
    type->add_lo_descriptor(*ilo - 1, *n, *l, *enu, *dme, auto_enu);
}

void FORTRAN(sirius_set_aw_enu)(int32_t* ia, int32_t* l, int32_t* order, real8* enu)
{
    global_parameters.atom(*ia - 1)->symmetry_class()->set_aw_enu(*l, *order - 1, *enu);
}

void FORTRAN(sirius_set_lo_enu)(int32_t* ia, int32_t* idxlo, int32_t* order, real8* enu)
{
    global_parameters.atom(*ia - 1)->symmetry_class()->set_lo_enu(*idxlo - 1, *order - 1, *enu);
}

/// Create the k-point set from the list of k-points and return it's id
void FORTRAN(sirius_create_kset)(int32_t* num_kpoints, double* kpoints__, double* kpoint_weights, int32_t* init_kset, 
                                 int32_t* kset_id)
{
    mdarray<double, 2> kpoints(kpoints__, 3, *num_kpoints); 
    
    sirius::K_set* new_kset = new sirius::K_set(global_parameters);
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
}

//** void FORTRAN(sirius_initialize_kset)(int32_t* kset_id)
//** {
//**     kset_list[*kset_id]->initialize();
//** }

void FORTRAN(sirius_delete_kset)(int32_t* kset_id)
{
    delete kset_list[*kset_id];
    kset_list[*kset_id] = NULL;
}


void FORTRAN(sirius_get_local_num_kpoints)(int32_t* kset_id, int32_t* nkpt_loc)
{
    *nkpt_loc = kset_list[*kset_id]->spl_num_kpoints().local_size();
}

void FORTRAN(sirius_get_local_kpoint_rank_and_offset)(int32_t* kset_id, int32_t* ik, int32_t* rank, int32_t* ikloc)
{
    *rank = kset_list[*kset_id]->spl_num_kpoints().location(_splindex_rank_, *ik - 1);
    *ikloc = kset_list[*kset_id]->spl_num_kpoints().location(_splindex_offs_, *ik - 1) + 1;
}

void FORTRAN(sirius_get_global_kpoint_index)(int32_t* kset_id, int32_t* ikloc, int32_t* ik)
{
    *ik = kset_list[*kset_id]->spl_num_kpoints(*ikloc - 1) + 1; // Fortran counts from 1
}

/// Generate radial functions (both aw and lo)
void FORTRAN(sirius_generate_radial_functions)()
{
    global_parameters.generate_radial_functions();
}

/// Generate radial integrals
void FORTRAN(sirius_generate_radial_integrals)()
{
    global_parameters.generate_radial_integrals();
}

void FORTRAN(sirius_get_symmetry_classes)(int32_t* ncls, int32_t* icls_by_ia)
{
    *ncls = global_parameters.num_atom_symmetry_classes();

    for (int ic = 0; ic < global_parameters.num_atom_symmetry_classes(); ic++)
    {
        for (int i = 0; i < global_parameters.atom_symmetry_class(ic)->num_atoms(); i++)
            icls_by_ia[global_parameters.atom_symmetry_class(ic)->atom_id(i)] = ic + 1; // Fortran counts from 1
    }
}

void FORTRAN(sirius_get_max_mt_radial_basis_size)(int32_t* max_mt_radial_basis_size)
{
    *max_mt_radial_basis_size = global_parameters.max_mt_radial_basis_size();
}

void FORTRAN(sirius_get_radial_functions)(double* radial_functions__)
{
    mdarray<double, 3> radial_functions(radial_functions__, 
                                        global_parameters.max_num_mt_points(), 
                                        global_parameters.max_mt_radial_basis_size(),
                                        global_parameters.num_atom_symmetry_classes());
    radial_functions.zero();

    for (int ic = 0; ic < global_parameters.num_atom_symmetry_classes(); ic++)
    {
        for (int idxrf = 0; idxrf < global_parameters.atom_symmetry_class(ic)->atom_type()->mt_radial_basis_size(); idxrf++)
        {
            for (int ir = 0; ir < global_parameters.atom_symmetry_class(ic)->atom_type()->num_mt_points(); ir++)
                radial_functions(ir, idxrf, ic) = global_parameters.atom_symmetry_class(ic)->radial_function(ir, idxrf);
        }
    }
}

void FORTRAN(sirius_get_max_mt_basis_size)(int32_t* max_mt_basis_size)
{
    *max_mt_basis_size = global_parameters.max_mt_basis_size();
}

void FORTRAN(sirius_get_basis_functions_index)(int32_t* mt_basis_size, int32_t* offset_wf, int32_t* indexb__)
{
    mdarray<int, 3> indexb(indexb__, 4, global_parameters.max_mt_basis_size(), global_parameters.num_atoms()); 

    for (int ia = 0; ia < global_parameters.num_atoms(); ia++)
    {
        mt_basis_size[ia] = global_parameters.atom(ia)->type()->mt_basis_size();
        offset_wf[ia] = global_parameters.atom(ia)->offset_wf();

        for (int j = 0; j < global_parameters.atom(ia)->type()->mt_basis_size(); j++)
        {
            indexb(0, j, ia) = global_parameters.atom(ia)->type()->indexb(j).l;
            indexb(1, j, ia) = global_parameters.atom(ia)->type()->indexb(j).lm + 1; // Fortran counts from 1
            indexb(2, j, ia) = global_parameters.atom(ia)->type()->indexb(j).idxrf + 1; // Fortran counts from 1
        }
    }
}

/// Get number of G+k vectors for a given k-point in the set
void FORTRAN(sirius_get_num_gkvec)(int32_t* kset_id, int32_t* ik, int32_t* num_gkvec)
{
    *num_gkvec = (*kset_list[*kset_id])[*ik - 1]->num_gkvec();
}

/// Get maximum number of G+k vectors across all k-points in the set
void FORTRAN(sirius_get_max_num_gkvec)(int32_t* kset_id, int32_t* max_num_gkvec)
{
    *max_num_gkvec = kset_list[*kset_id]->max_num_gkvec();
}

/// Get all G+k vector related arrays
void FORTRAN(sirius_get_gkvec_arrays)(int32_t* kset_id, int32_t* ik, int32_t* num_gkvec, int32_t* gvec_index, 
                                      real8* gkvec__, real8* gkvec_cart__, real8* gkvec_len, real8* gkvec_tp__, 
                                      complex16* gkvec_phase_factors__, int32_t* ld)
{
    int rank = kset_list[*kset_id]->spl_num_kpoints().location(_splindex_rank_, *ik - 1);
    
    if (rank == global_parameters.mpi_grid().coordinate(0))
    {
        sirius::K_point* kp = (*kset_list[*kset_id])[*ik - 1];
        *num_gkvec = kp->num_gkvec();
        mdarray<double, 2> gkvec(gkvec__, 3, kp->num_gkvec()); 
        mdarray<double, 2> gkvec_cart(gkvec_cart__, 3, kp->num_gkvec());
        mdarray<double, 2> gkvec_tp(gkvec_tp__, 2, kp->num_gkvec()); 
        mdarray<complex16, 2> gkvec_phase_factors(gkvec_phase_factors__, *ld, global_parameters.num_atoms());

        for (int igk = 0; igk < kp->num_gkvec(); igk++)
        {
            gvec_index[igk] = kp->gvec_index(igk) + 1; //Fortran counst form 1
            for (int x = 0; x < 3; x++) gkvec(x, igk) = kp->gkvec(igk)[x];
            global_parameters.get_coordinates<cartesian, reciprocal>(&gkvec(0, igk), &gkvec_cart(0, igk));
            double rtp[3];
            sirius::SHT::spherical_coordinates(&gkvec_cart(0, igk), rtp);
            gkvec_len[igk] = rtp[0];
            gkvec_tp(0, igk) = rtp[1];
            gkvec_tp(1, igk) = rtp[2];

            for (int ia = 0; ia < global_parameters.num_atoms(); ia++)
                gkvec_phase_factors(igk, ia) = kp->gkvec_phase_factor(igk, ia);
        }
    }
    Platform::bcast(num_gkvec, 1, rank);
}

void FORTRAN(sirius_get_matching_coefficients)(int32_t* kset_id, int32_t* ik, complex16* apwalm__, 
                                               int32_t* ngkmax, int32_t* apwordmax)
{

    int rank = kset_list[*kset_id]->spl_num_kpoints().location(_splindex_rank_, *ik - 1);
    
    if (rank == global_parameters.mpi_grid().coordinate(0))
    {
        sirius::K_point* kp = (*kset_list[*kset_id])[*ik - 1];
        
        mdarray<complex16, 4> apwalm(apwalm__, *ngkmax, *apwordmax, global_parameters.lmmax_apw(), 
                                     global_parameters.num_atoms());
        mdarray<complex16, 2> alm(kp->num_gkvec(), global_parameters.max_mt_aw_basis_size());

        for (int ia = 0; ia < global_parameters.num_atoms(); ia++)
        {
            sirius::Atom* atom = global_parameters.atom(ia);
            kp->generate_matching_coefficients(kp->num_gkvec(), ia, alm);

            for (int l = 0; l <= global_parameters.lmax_apw(); l++)
            {
                for (int order = 0; order < (int)atom->type()->aw_descriptor(l).size(); order++)
                {
                    for (int m = -l; m <= l; m++)
                    {
                        int lm = Utils::lm_by_l_m(l, m);
                        int i = atom->type()->indexb_by_lm_order(lm, order);
                        for (int ig = 0; ig < kp->num_gkvec(); ig++) apwalm(ig, order, lm, ia) = conj(alm(ig, i));
                    }
                }
            }
        }
    }
}

/// Get first-variational matrices of Hamiltonian and overlap
/** Radial integrals and plane-wave coefficients of the interstitial potential must be calculated prior to
    Hamiltonian and overlap matrix construction. */
void FORTRAN(sirius_get_fv_h_o)(int32_t* kset_id, int32_t* ik, int32_t* size, complex16* h__, complex16* o__)
{
    int rank = kset_list[*kset_id]->spl_num_kpoints().location(_splindex_rank_, *ik - 1);
    
    if (rank == global_parameters.mpi_grid().coordinate(0))
    {
        sirius::K_point* kp = (*kset_list[*kset_id])[*ik - 1];
        
        if (*size != kp->apwlo_basis_size())
        {
            error(__FILE__, __LINE__, "wrong matrix size");
        }

        mdarray<complex16, 2> h(h__, kp->apwlo_basis_size(), kp->apwlo_basis_size());
        mdarray<complex16, 2> o(o__, kp->apwlo_basis_size(), kp->apwlo_basis_size());
        kp->set_fv_h_o<cpu, apwlo>(potential->effective_potential(), kset_list[*kset_id]->band()->num_ranks(), h, o);
    }
}

/// Get the total size of wave-function (number of mt coefficients + number of G+k coefficients)
void FORTRAN(sirius_get_mtgk_size)(int32_t* kset_id, int32_t* ik, int32_t* mtgk_size)
{
    *mtgk_size = (*kset_list[*kset_id])[*ik - 1]->mtgk_size();
}

void FORTRAN(sirius_get_spinor_wave_functions)(int32_t* kset_id, int32_t* ik, complex16* spinor_wave_functions__)
{
    assert(global_parameters.num_bands() == kset_list[*kset_id]->band()->spl_spinor_wf_col().local_size());

    sirius::K_point* kp = (*kset_list[*kset_id])[*ik - 1];
    sirius::Band* band = kset_list[*kset_id]->band();
    
    mdarray<complex16, 3> spinor_wave_functions(spinor_wave_functions__, kp->mtgk_size(), global_parameters.num_spins(), 
                                                band->spl_spinor_wf_col().local_size());

    for (int j = 0; j < band->spl_spinor_wf_col().local_size(); j++)
    {
        memcpy(&spinor_wave_functions(0, 0, j), &kp->spinor_wave_function(0, 0, j), 
               kp->mtgk_size() * global_parameters.num_spins() * sizeof(complex16));
    }
}

void FORTRAN(sirius_apply_step_function_gk)(int32_t* kset_id, int32_t* ik, complex16* wf__)
{
    int thread_id = Platform::thread_id();

    sirius::K_point* kp = (*kset_list[*kset_id])[*ik - 1];
    int num_gkvec = kp->num_gkvec();

    global_parameters.fft().input(num_gkvec, kp->fft_index(), wf__, thread_id);
    global_parameters.fft().transform(1, thread_id);
    for (int ir = 0; ir < global_parameters.fft().size(); ir++)
        global_parameters.fft().output_buffer(ir, thread_id) *= global_parameters.step_function(ir);

    global_parameters.fft().input(&global_parameters.fft().output_buffer(0, thread_id));
    global_parameters.fft().transform(-1, thread_id);
    global_parameters.fft().output(num_gkvec, kp->fft_index(), wf__, thread_id);
}

/// Get Cartesian coordinates of G+k vectors
void FORTRAN(sirius_get_gkvec_cart)(int32_t* kset_id, int32_t* ik, double* gkvec_cart__)
{
    sirius::K_point* kp = (*kset_list[*kset_id])[*ik - 1];
    mdarray<double, 2> gkvec_cart(gkvec_cart__, 3, kp->num_gkvec());

    for (int ig = 0; ig < kp->num_gkvec(); ig++)
        global_parameters.get_coordinates<cartesian, reciprocal>(kp->gkvec(ig), &gkvec_cart(0, ig));
}

void FORTRAN(sirius_get_evalsum)(real8* evalsum)
{
    *evalsum = global_parameters.rti().core_eval_sum + global_parameters.rti().valence_eval_sum;
}

void FORTRAN(sirius_get_energy_exc)(real8* energy_exc)
{
    *energy_exc = global_parameters.rti().energy_exc;
}

void FORTRAN(sirius_get_energy_vxc)(real8* energy_vxc)
{
    *energy_vxc = global_parameters.rti().energy_vxc;
}

void FORTRAN(sirius_get_energy_bxc)(real8* energy_bxc)
{
    *energy_bxc = global_parameters.rti().energy_bxc;
}

void FORTRAN(sirius_get_energy_veff)(real8* energy_veff)
{
    *energy_veff = global_parameters.rti().energy_veff;
}

void FORTRAN(sirius_get_energy_vha)(real8* energy_vha)
{
    *energy_vha = global_parameters.rti().energy_vha;
}

void FORTRAN(sirius_get_energy_enuc)(real8* energy_enuc)
{
    *energy_enuc = global_parameters.rti().energy_enuc;
}

void FORTRAN(sirius_generate_xc_potential)(real8* rhomt, real8* rhoit, real8* vxcmt, real8* vxcit)
{
    using namespace sirius;
    Periodic_function<double>* rho = 
        new Periodic_function<double>(global_parameters, Argument(arg_lm, global_parameters.lmmax_rho()),
                                                        Argument(arg_radial, global_parameters.max_num_mt_points()));
    rho->set_mt_ptr(rhomt);
    rho->set_it_ptr(rhoit);
    
    Periodic_function<double>* vxc = 
        new Periodic_function<double>(global_parameters, Argument(arg_lm, global_parameters.lmmax_pot()),
                                                        Argument(arg_radial, global_parameters.max_num_mt_points()));
    vxc->set_mt_ptr(vxcmt);
    vxc->set_it_ptr(vxcit);

    Periodic_function<double>* exc = 
        new Periodic_function<double>(global_parameters, Argument(arg_lm, global_parameters.lmmax_pot()),
                                                        Argument(arg_radial, global_parameters.max_num_mt_points()));
    exc->allocate(false);
    
    //Periodic_function<double>* bxc[3];
    //for (int j = 0; j < parameters_.num_mag_dims(); j++)
    //{
    //    bxc[j] = new Periodic_function<double>(parameters_, parameters_.lmax_pot());
    //    bxc[j]->split(rlm_component | it_component);
    //    bxc[j]->allocate(rlm_component | it_component);
    //    //bxc[j]->zero();
    //}

    potential->xc(rho, NULL, vxc, NULL, exc);
    //vxc->zero();
    //exc->zero();
   
    global_parameters.rti().energy_vxc = inner(global_parameters, rho, vxc);
    global_parameters.rti().energy_exc = inner(global_parameters, rho, exc);

    delete vxc;
    delete exc;
    delete rho;
}

void FORTRAN(sirius_generate_coulomb_potential)(real8* rhomt, real8* rhoit, real8* vclmt, real8* vclit)
{
    using namespace sirius;
    Periodic_function<double>* rho = 
        new Periodic_function<double>(global_parameters, Argument(arg_lm, global_parameters.lmmax_rho()),
                                                        Argument(arg_radial, global_parameters.max_num_mt_points()),
                                                        global_parameters.num_gvec());
    rho->set_mt_ptr(rhomt);
    rho->set_it_ptr(rhoit);
    
    Periodic_function<double>* vcl = 
        new Periodic_function<double>(global_parameters, Argument(arg_lm, global_parameters.lmmax_pot()),
                                                        Argument(arg_radial, global_parameters.max_num_mt_points()),
                                                        global_parameters.num_gvec());
    vcl->set_mt_ptr(vclmt);
    vcl->set_it_ptr(vclit);

    potential->poisson(rho, vcl);
   
    global_parameters.rti().energy_vha = inner(global_parameters, rho, vcl);

    delete vcl;
    delete rho;
}

void FORTRAN(sirius_update_atomic_potential)()
{
    potential->update_atomic_potential();
}

void FORTRAN(sirius_scalar_radial_solver)(int32_t* zn, int32_t* l, int32_t* dme, real8* enu, int32_t* nr, real8* r, 
                                          real8* v__, int32_t* nn, real8* p0__, real8* p1__, real8* q0__, real8* q1__)
{
    sirius::RadialGrid rgrid;
    rgrid.set_radial_points(*nr, *nr, r);
    sirius::RadialSolver solver(false, *zn, rgrid);

    std::vector<real8> v(*nr);
    std::vector<real8> p0;
    std::vector<real8> p1;
    std::vector<real8> q0;
    std::vector<real8> q1;

    memcpy(&v[0], v__, (*nr) * sizeof(real8));

    *nn = solver.solve_in_mt(*l, *enu, *dme, v, p0, p1, q0, q1);

    memcpy(p0__, &p0[0], (*nr) * sizeof(real8));
    memcpy(p1__, &p1[0], (*nr) * sizeof(real8));
    memcpy(q0__, &q0[0], (*nr) * sizeof(real8));
    memcpy(q1__, &q1[0], (*nr) * sizeof(real8));
}

void FORTRAN(sirius_get_aw_radial_function)(int32_t* ia__, int32_t* l, int32_t* io__, real8* awrf)
{
    int ia = *ia__ - 1;
    int io = *io__ - 1;
    int idxrf = global_parameters.atom(ia)->type()->indexr_by_l_order(*l, io);
    for (int ir = 0; ir < global_parameters.atom(ia)->num_mt_points(); ir++)
        awrf[ir] = global_parameters.atom(ia)->symmetry_class()->radial_function(ir, idxrf);
}
    
void FORTRAN(sirius_get_aw_h_radial_function)(int32_t* ia__, int32_t* l, int32_t* io__, real8* hawrf)
{
    int ia = *ia__ - 1;
    int io = *io__ - 1;
    int idxrf = global_parameters.atom(ia)->type()->indexr_by_l_order(*l, io);
    for (int ir = 0; ir < global_parameters.atom(ia)->num_mt_points(); ir++)
    {
        double rinv = global_parameters.atom(ia)->type()->radial_grid().rinv(ir);
        hawrf[ir] = global_parameters.atom(ia)->symmetry_class()->h_radial_function(ir, idxrf) * rinv;
    }
}
    
void FORTRAN(sirius_get_aw_surface_derivative)(int32_t* ia, int32_t* l, int32_t* io, real8* dawrf)
{
    *dawrf = global_parameters.atom(*ia - 1)->symmetry_class()->aw_surface_dm(*l, *io - 1, 1); 
}

void FORTRAN(sirius_get_lo_radial_function)(int32_t* ia__, int32_t* idxlo__, real8* lorf)
{
    int ia = *ia__ - 1;
    int idxlo = *idxlo__ - 1;
    int idxrf = global_parameters.atom(ia)->type()->indexr_by_idxlo(idxlo);
    for (int ir = 0; ir < global_parameters.atom(ia)->num_mt_points(); ir++)
        lorf[ir] = global_parameters.atom(ia)->symmetry_class()->radial_function(ir, idxrf);
}
    
void FORTRAN(sirius_get_lo_h_radial_function)(int32_t* ia__, int32_t* idxlo__, real8* hlorf)
{
    int ia = *ia__ - 1;
    int idxlo = *idxlo__ - 1;
    int idxrf = global_parameters.atom(ia)->type()->indexr_by_idxlo(idxlo);
    for (int ir = 0; ir < global_parameters.atom(ia)->num_mt_points(); ir++)
    {
        double rinv = global_parameters.atom(ia)->type()->radial_grid().rinv(ir);
        hlorf[ir] = global_parameters.atom(ia)->symmetry_class()->h_radial_function(ir, idxrf) * rinv;
    }
}
    
void FORTRAN(sirius_get_aw_lo_o_radial_integral)(int32_t* ia__, int32_t* l, int32_t* io1, int32_t* ilo2, 
                                                 real8* oalo)
{
    int ia = *ia__ - 1;

    int idxrf2 = global_parameters.atom(ia)->type()->indexr_by_idxlo(*ilo2 - 1);
    int order2 = global_parameters.atom(ia)->type()->indexr(idxrf2).order;

    *oalo = global_parameters.atom(ia)->symmetry_class()->o_radial_integral(*l, *io1 - 1, order2);
}

void FORTRAN(sirius_get_lo_lo_o_radial_integral)(int32_t* ia__, int32_t* l, int32_t* ilo1, int32_t* ilo2, 
                                                 real8* ololo)
{
    int ia = *ia__ - 1;

    int idxrf1 = global_parameters.atom(ia)->type()->indexr_by_idxlo(*ilo1 - 1);
    int order1 = global_parameters.atom(ia)->type()->indexr(idxrf1).order;
    int idxrf2 = global_parameters.atom(ia)->type()->indexr_by_idxlo(*ilo2 - 1);
    int order2 = global_parameters.atom(ia)->type()->indexr(idxrf2).order;

    *ololo = global_parameters.atom(ia)->symmetry_class()->o_radial_integral(*l, order1, order2);
}

void FORTRAN(sirius_get_aw_aw_h_radial_integral)(int32_t* ia__, int32_t* l1, int32_t* io1, int32_t* l2, 
                                                 int32_t* io2, int32_t* lm3, real8* haa)
{
    int ia = *ia__ - 1;
    int idxrf1 = global_parameters.atom(ia)->type()->indexr_by_l_order(*l1, *io1 - 1);
    int idxrf2 = global_parameters.atom(ia)->type()->indexr_by_l_order(*l2, *io2 - 1);

    *haa = global_parameters.atom(ia)->h_radial_integrals(idxrf1, idxrf2)[*lm3 - 1];
}

void FORTRAN(sirius_get_lo_aw_h_radial_integral)(int32_t* ia__, int32_t* ilo1, int32_t* l2, int32_t* io2, int32_t* lm3, 
                                                 real8* hloa)
{
    int ia = *ia__ - 1;
    int idxrf1 = global_parameters.atom(ia)->type()->indexr_by_idxlo(*ilo1 - 1);
    int idxrf2 = global_parameters.atom(ia)->type()->indexr_by_l_order(*l2, *io2 - 1);

    *hloa = global_parameters.atom(ia)->h_radial_integrals(idxrf1, idxrf2)[*lm3 - 1];
}


void FORTRAN(sirius_get_lo_lo_h_radial_integral)(int32_t* ia__, int32_t* ilo1, int32_t* ilo2, int32_t* lm3, 
                                                 real8* hlolo)
{
    int ia = *ia__ - 1;
    int idxrf1 = global_parameters.atom(ia)->type()->indexr_by_idxlo(*ilo1 - 1);
    int idxrf2 = global_parameters.atom(ia)->type()->indexr_by_idxlo(*ilo2 - 1);

    *hlolo = global_parameters.atom(ia)->h_radial_integrals(idxrf1, idxrf2)[*lm3 - 1];
}

void FORTRAN(sirius_generate_potential_pw_coefs)(void)
{
    potential->generate_pw_coefs();
}

void FORTRAN(sirius_get_fv_eigen_vectors)(int32_t* kset_id, int32_t* ik, complex16* fv_evec__, int32_t* ld, 
                                          int32_t* num_fv_evec)
{
    mdarray<complex16, 2> fv_evec(fv_evec__, *ld, *num_fv_evec);
    (*kset_list[*kset_id])[*ik - 1]->get_fv_eigen_vectors(fv_evec);
}

void FORTRAN(sirius_get_sv_eigen_vectors)(int32_t* kset_id, int32_t* ik, complex16* sv_evec__, int32_t* size)
{
    mdarray<complex16, 2> sv_evec(sv_evec__, *size, *size);
    (*kset_list[*kset_id])[*ik - 1]->get_sv_eigen_vectors(sv_evec);
}

void FORTRAN(sirius_get_num_fv_states)(int32_t* num_fv_states)
{
    *num_fv_states = global_parameters.num_fv_states();
}

} // extern "C"
