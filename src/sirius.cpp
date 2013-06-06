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
std::vector<sirius::kset*> kset_list;

extern "C" 
{

/// Set lattice vectors
/** Fortran example:
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
/** Fortran example:
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

void FORTRAN(sirius_set_lmax_rho)(int32_t* lmax_rho)
{
    global_parameters.set_lmax_rho(*lmax_rho);
}

void FORTRAN(sirius_set_lmax_pot)(int32_t* lmax_pot)
{
    global_parameters.set_lmax_pot(*lmax_pot);
}

void FORTRAN(sirius_set_pw_cutoff)(real8* pw_cutoff)
{
    global_parameters.set_pw_cutoff(*pw_cutoff);
}

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

void FORTRAN(sirius_set_num_spins)(int32_t* num_spins)
{
    global_parameters.set_num_spins(*num_spins);
}

void FORTRAN(sirius_set_num_mag_dims)(int32_t* num_mag_dims)
{
    global_parameters.set_num_mag_dims(*num_mag_dims);
}

void FORTRAN(sirius_set_auto_rmt)(int32_t* auto_rmt)
{
    global_parameters.set_auto_rmt(*auto_rmt);
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

/// Get lower or upper limit of each FFT grid dimension
void FORTRAN(sirius_get_fft_grid_limits)(int32_t* d, int32_t* lu, int32_t* val)
{
    *val = global_parameters.fft().grid_limits(*d, *lu);
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

void FORTRAN(sirius_add_atom_type)(int32_t* atom_type_id, char* label, int32_t label_len)
{
    global_parameters.add_atom_type(*atom_type_id, std::string(label, label_len));
}

void FORTRAN(sirius_add_atom)(int32_t* atom_type_id, real8* position, real8* vector_field)
{
    global_parameters.add_atom(*atom_type_id, position, vector_field);
}

/// Initialize the low-level of the library
void FORTRAN(sirius_platform_initialize)(int32_t* call_mpi_init_)
{
    bool call_mpi_init = (*call_mpi_init_ != 0) ? true : false; 
    Platform::initialize(call_mpi_init);
}

/// Initialize the global variables
/** All _set_ functions must be called before.*/
void FORTRAN(sirius_global_initialize)(int32_t* init_radial_grid, int32_t* init_aw_descriptors)
{
    global_parameters.initialize(*init_radial_grid, *init_aw_descriptors);
}

void FORTRAN(sirius_potential_initialize)(void)
{
    potential = new sirius::Potential(global_parameters);
}

void FORTRAN(sirius_density_initialize)(int32_t* num_kpoints, double* kpoints_, double* kpoint_weights)
{
    mdarray<double, 2> kpoints(kpoints_, 3, *num_kpoints); 
    density = new sirius::Density(global_parameters, potential);
}

void FORTRAN(sirius_clear)(void)
{
    global_parameters.clear();
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

void FORTRAN(sirius_find_eigen_states)(int32_t* kset_id)
{
    kset_list[*kset_id]->find_eigen_states(potential);
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

    sirius::kset kset_(global_parameters);
    
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
    for (int ir = 0; ir < global_parameters.fft().size(); ir++)
         potential->effective_potential()->f_it(ir) *= global_parameters.step_function(ir);

    global_parameters.fft().input(potential->effective_potential()->f_it());
    global_parameters.fft().transform(-1);
    global_parameters.fft().output(global_parameters.num_gvec(), global_parameters.fft_index(), potential->effective_potential()->f_pw());
    
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
        json_write jw("bands.json");
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
    //FORTRAN(sirius_read_state)();

    density->initial_density(1);

    potential->generate_effective_potential(density->rho(), density->magnetization());

    
    // generate plane-wave coefficients of the potential in the interstitial region
    global_parameters.fft().input(potential->effective_potential()->f_it());
    global_parameters.fft().transform(-1);
    global_parameters.fft().output(global_parameters.num_gvec(), global_parameters.fft_index(), 
                                   potential->effective_potential()->f_pw());

    int N = 10000;
    double* p = new double[N];
    double* x = new double[N];

    double vf1[] = {0.1, 0.1, 0.1};
    double vf2[] = {0.9, 0.9, 0.9};

    #pragma omp parallel for default(shared)
    for (int i = 0; i < N; i++)
    {
        double vf[3];
        double vc[3];
        double t = double(i) / (N - 1);
        for (int j = 0; j < 3; j++) vf[j] = vf1[j] + t * (vf2[j] - vf1[j]);

        global_parameters.get_coordinates<cartesian, direct>(vf, vc);
        p[i] = potential->value(vc);
        x[i] = Utils::vector_length(vc);
    }

    FILE* fout = fopen("potential.dat", "w");
    for (int i = 0; i < N; i++) fprintf(fout, "%.12f %.12f\n", x[i] - x[0], p[i]);
    fclose(fout);
    delete x;
    delete p;
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

void FORTRAN(sirius_set_atom_type_properties)(int32_t* atom_type_id, char* symbol, int32_t* zn, real8* mass, 
                                              real8* mt_radius, int32_t* num_mt_points, real8* radial_grid_origin, 
                                              real8* radial_grid_infinity, int32_t symbol_len)
{
    sirius::AtomType* type = global_parameters.atom_type_by_id(*atom_type_id);
    type->set_symbol(std::string(symbol, symbol_len));
    type->set_zn(*zn);
    type->set_mass(*mass);
    type->set_num_mt_points(*num_mt_points);
    type->set_radial_grid_origin(*radial_grid_origin);
    type->set_radial_grid_infinity(*radial_grid_infinity);
    type->set_mt_radius(*mt_radius);
}

void FORTRAN(sirius_set_atom_type_radial_grid)(int32_t* atom_type_id, int32_t* num_radial_points, 
                                               int32_t* num_mt_points, real8* radial_points)
{
    sirius::AtomType* type = global_parameters.atom_type_by_id(*atom_type_id);
    type->radial_grid().set_radial_points(*num_radial_points, *num_mt_points, radial_points);
}

void FORTRAN(sirius_set_atom_type_configuration)(int32_t* atom_type_id, int32_t* n, int32_t* l, int32_t* k, 
                                                 real8* occupancy, int32_t* core_)
{
    sirius::AtomType* type = global_parameters.atom_type_by_id(*atom_type_id);
    bool core = *core_;
    type->set_configuration(*n, *l, *k, *occupancy, core);
}

void FORTRAN(sirius_add_atom_type_aw_descriptor)(int32_t* atom_type_id, int32_t* n, int32_t* l, real8* enu, 
                                                 int32_t* dme, int32_t* auto_enu)
{
    sirius::AtomType* type = global_parameters.atom_type_by_id(*atom_type_id);
    type->add_aw_descriptor(*n, *l, *enu, *dme, *auto_enu);
}

void FORTRAN(sirius_add_atom_type_lo_descriptor)(int32_t* atom_type_id, int32_t* ilo, int32_t* n, int32_t* l, 
                                                 real8* enu, int32_t* dme, int32_t* auto_enu)
{
    sirius::AtomType* type = global_parameters.atom_type_by_id(*atom_type_id);
    type->add_lo_descriptor(*ilo - 1, *n, *l, *enu, *dme, *auto_enu);
}

/// Create the k-point set from the list of k-points and return it's id
void FORTRAN(sirius_create_kset)(int32_t* num_kpoints, double* kpoints__, double* kpoint_weights, int32_t* kset_id)
{
    mdarray<double, 2> kpoints(kpoints__, 3, *num_kpoints); 
    
    sirius::kset* new_kset = new sirius::kset(global_parameters);
    new_kset->add_kpoints(kpoints, kpoint_weights);
    new_kset->initialize();
    
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

void FORTRAN(sirius_initialize_kset)(int32_t* kset_id)
{
    kset_list[*kset_id]->initialize();
}

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
        sirius::kpoint* kp = (*kset_list[*kset_id])[*ik - 1];
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
}

void FORTRAN(sirius_get_matching_coefficients)(int32_t* kset_id, int32_t* ik, complex16* apwalm__, 
                                               int32_t* ngkmax, int32_t* apwordmax)
{

    int rank = kset_list[*kset_id]->spl_num_kpoints().location(_splindex_rank_, *ik - 1);
    
    if (rank == global_parameters.mpi_grid().coordinate(0))
    {
        sirius::kpoint* kp = (*kset_list[*kset_id])[*ik - 1];
        
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
                        int i = atom->type()->indexb().index_by_lm_order(lm, order);
                        for (int ig = 0; ig < kp->num_gkvec(); ig++) apwalm(ig, order, lm, ia) = conj(alm(ig, i));
                    }
                }
            }
        }
    }
}

/// Get first-variational matrices of Hamiltonian and overlap
void FORTRAN(sirius_get_fv_h_o)(int32_t* kset_id, int32_t* ik, int32_t* size, complex16* h__, complex16* o__)
{
    potential->generate_pw_coefs();

    int rank = kset_list[*kset_id]->spl_num_kpoints().location(_splindex_rank_, *ik - 1);
    
    if (rank == global_parameters.mpi_grid().coordinate(0))
    {
        sirius::kpoint* kp = (*kset_list[*kset_id])[*ik - 1];
        
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

    sirius::kpoint* kp = (*kset_list[*kset_id])[*ik - 1];
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
    sirius::kpoint* kp = (*kset_list[*kset_id])[*ik - 1];
    int num_gkvec = kp->num_gkvec();

    global_parameters.fft().input(num_gkvec, kp->fft_index(), wf__);
    global_parameters.fft().transform(1);
    for (int ir = 0; ir < global_parameters.fft().size(); ir++)
        global_parameters.fft().output_buffer(ir) *= global_parameters.step_function(ir);

    global_parameters.fft().input(global_parameters.fft().output_buffer_ptr());
    global_parameters.fft().transform(-1);
    global_parameters.fft().output(num_gkvec, kp->fft_index(), wf__);
}

/// Get Cartesian coordinates of G+k vectors
void FORTRAN(sirius_get_gkvec_cart)(int32_t* kset_id, int32_t* ik, double* gkvec_cart__)
{
    sirius::kpoint* kp = (*kset_list[*kset_id])[*ik - 1];
    mdarray<double, 2> gkvec_cart(gkvec_cart__, 3, kp->num_gkvec());

    for (int ig = 0; ig < kp->num_gkvec(); ig++)
        global_parameters.get_coordinates<cartesian, reciprocal>(kp->gkvec(ig), &gkvec_cart(0, ig));
}

void FORTRAN(sirius_get_evalsum)(real8* evalsum)
{
    *evalsum = global_parameters.rti().core_eval_sum + global_parameters.rti().valence_eval_sum;
}

void FORTRAN(sirius_get_energy_exc)(real8* exc)
{
    *exc = global_parameters.rti().energy_exc;
}

void FORTRAN(sirius_generate_xc_potential)(real8* rhomt, real8* rhoit, real8* vxcmt, real8* vxcit)
{
    sirius::PeriodicFunction<double>* rho = new sirius::PeriodicFunction<double>(global_parameters, global_parameters.lmax_rho()); 
    rho->set_rlm_ptr(rhomt);
    rho->set_it_ptr(rhoit);
    
    sirius::PeriodicFunction<double>* vxc = new sirius::PeriodicFunction<double>(global_parameters, global_parameters.lmax_pot());
    vxc->set_rlm_ptr(vxcmt);
    vxc->set_it_ptr(vxcit);

    sirius::PeriodicFunction<double>* exc = new sirius::PeriodicFunction<double>(global_parameters, global_parameters.lmax_pot());     
    exc->allocate(sirius::rlm_component | sirius::it_component);
    
    //PeriodicFunction<double>* bxc[3];
    //for (int j = 0; j < parameters_.num_mag_dims(); j++)
    //{
    //    bxc[j] = new PeriodicFunction<double>(parameters_, parameters_.lmax_pot());
    //    bxc[j]->split(rlm_component | it_component);
    //    bxc[j]->allocate(rlm_component | it_component);
    //    //bxc[j]->zero();
    //}

    potential->xc(rho, NULL, vxc, NULL, exc);
    //vxc->zero();
    //exc->zero();
   
    global_parameters.rti().energy_vxc = rho->inner(vxc, sirius::rlm_component | sirius::it_component);
    global_parameters.rti().energy_exc = rho->inner(exc, sirius::rlm_component | sirius::it_component);

    delete vxc;
    delete exc;
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
    int idxrf = global_parameters.atom(ia)->type()->indexr().index_by_l_order(*l, io);
    for (int ir = 0; ir < global_parameters.atom(ia)->num_mt_points(); ir++)
        awrf[ir] = global_parameters.atom(ia)->symmetry_class()->radial_function(ir, idxrf);
}
    
void FORTRAN(sirius_get_aw_h_radial_function)(int32_t* ia__, int32_t* l, int32_t* io__, real8* hawrf)
{
    int ia = *ia__ - 1;
    int io = *io__ - 1;
    int idxrf = global_parameters.atom(ia)->type()->indexr().index_by_l_order(*l, io);
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

void FORTRAN(sirius_get_aw_lo_o_radial_integral)(int32_t* ia__, int32_t* l, int32_t* io1, int32_t* ilo2, 
                                                 real8* oalo)
{
    int ia = *ia__ - 1;

    int idxrf2 = global_parameters.atom(ia)->type()->indexr().index_by_idxlo(*ilo2 - 1);
    int order2 = global_parameters.atom(ia)->type()->indexr(idxrf2).order;

    *oalo = global_parameters.atom(ia)->symmetry_class()->o_radial_integral(*l, *io1 - 1, order2);
}

void FORTRAN(sirius_get_lo_lo_o_radial_integral)(int32_t* ia__, int32_t* l, int32_t* ilo1, int32_t* ilo2, 
                                                 real8* ololo)
{
    int ia = *ia__ - 1;

    int idxrf1 = global_parameters.atom(ia)->type()->indexr().index_by_idxlo(*ilo1 - 1);
    int order1 = global_parameters.atom(ia)->type()->indexr(idxrf1).order;
    int idxrf2 = global_parameters.atom(ia)->type()->indexr().index_by_idxlo(*ilo2 - 1);
    int order2 = global_parameters.atom(ia)->type()->indexr(idxrf2).order;

    *ololo = global_parameters.atom(ia)->symmetry_class()->o_radial_integral(*l, order1, order2);
}

void FORTRAN(sirius_get_aw_aw_h_radial_integral)(int32_t* ia__, int32_t* l1, int32_t* io1, int32_t* l2, 
                                                 int32_t* io2, int32_t* lm3, real8* haa)
{
    int ia = *ia__ - 1;
    int idxrf1 = global_parameters.atom(ia)->type()->indexr().index_by_l_order(*l1, *io1 - 1);
    int idxrf2 = global_parameters.atom(ia)->type()->indexr().index_by_l_order(*l2, *io2 - 1);

    *haa = global_parameters.atom(ia)->h_radial_integrals(idxrf1, idxrf2)[*lm3 - 1];
}

void FORTRAN(sirius_get_lo_aw_h_radial_integral)(int32_t* ia__, int32_t* ilo1, int32_t* l2, int32_t* io2, int32_t* lm3, 
                                                 real8* hloa)
{
    int ia = *ia__ - 1;
    int idxrf1 = global_parameters.atom(ia)->type()->indexr().index_by_idxlo(*ilo1 - 1);
    int idxrf2 = global_parameters.atom(ia)->type()->indexr().index_by_l_order(*l2, *io2 - 1);

    *hloa = global_parameters.atom(ia)->h_radial_integrals(idxrf1, idxrf2)[*lm3 - 1];
}


void FORTRAN(sirius_get_lo_lo_h_radial_integral)(int32_t* ia__, int32_t* ilo1, int32_t* ilo2, int32_t* lm3, 
                                                 real8* hlolo)
{
    int ia = *ia__ - 1;
    int idxrf1 = global_parameters.atom(ia)->type()->indexr().index_by_idxlo(*ilo1 - 1);
    int idxrf2 = global_parameters.atom(ia)->type()->indexr().index_by_idxlo(*ilo2 - 1);

    *hlolo = global_parameters.atom(ia)->h_radial_integrals(idxrf1, idxrf2)[*lm3 - 1];
}

void FORTRAN(sirius_generate_potential_pw_coefs)()
{
    potential->generate_pw_coefs();
}


} // extern "C"
