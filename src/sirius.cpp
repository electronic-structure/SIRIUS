#include "sirius.h"

/** \file sirius.cpp
    \brief Fortran API
*/

sirius::Density* density = NULL;

sirius::Potential* potential = NULL;

sirius::Global global;

extern "C" 
{

/*
    primitive set functions
*/

/// set lattice vectors

/** Fortran example:
    \code{.F90}
        call sirius_set_lattice_vectors(avec(1,1), avec(1,2), avec(1,3))
    \endcode
*/
void FORTRAN(sirius_set_lattice_vectors)(real8* a1, real8* a2, real8* a3)
{
    global.set_lattice_vectors(a1, a2, a3);
}

void FORTRAN(sirius_set_lmax_apw)(int32_t* lmax_apw)
{
    global.set_lmax_apw(*lmax_apw);
}

void FORTRAN(sirius_set_lmax_rho)(int32_t* lmax_rho)
{
    global.set_lmax_rho(*lmax_rho);
}

void FORTRAN(sirius_set_lmax_pot)(int32_t* lmax_pot)
{
    global.set_lmax_pot(*lmax_pot);
}

void FORTRAN(sirius_set_pw_cutoff)(real8* pw_cutoff)
{
    global.set_pw_cutoff(*pw_cutoff);
}

void FORTRAN(sirius_set_aw_cutoff)(real8* aw_cutoff)
{
    global.set_aw_cutoff(*aw_cutoff);
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
    global.set_equivalent_atoms(equivalent_atoms);
}

void FORTRAN(sirius_set_num_spins)(int32_t* num_spins)
{
    global.set_num_spins(*num_spins);
}

void FORTRAN(sirius_set_num_mag_dims)(int32_t* num_mag_dims)
{
    global.set_num_mag_dims(*num_mag_dims);
}

/*
    primitive get functions
*/
void FORTRAN(sirius_get_max_num_mt_points)(int32_t* max_num_mt_points)
{
    *max_num_mt_points = global.max_num_mt_points();
}

void FORTRAN(sirius_get_num_mt_points)(int32_t* atom_type_id, int32_t* num_mt_points)
{
    *num_mt_points = global.atom_type_by_id(*atom_type_id)->num_mt_points();
}

void FORTRAN(sirius_get_mt_points)(int32_t* atom_type_id, real8* mt_points)
{
    memcpy(mt_points, global.atom_type_by_id(*atom_type_id)->radial_grid().get_ptr(),
        global.atom_type_by_id(*atom_type_id)->num_mt_points() * sizeof(real8));
}

void FORTRAN(sirius_get_num_grid_points)(int32_t* num_grid_points)
{
    *num_grid_points = global.fft().size();
}

void FORTRAN(sirius_get_num_bands)(int32_t* num_bands)
{
    *num_bands = global.num_bands();
}

void FORTRAN(sirius_get_num_gvec)(int32_t* num_gvec)
{
    *num_gvec = global.num_gvec();
}

void FORTRAN(sirius_get_fft_grid_size)(int32_t* grid_size)
{
    grid_size[0] = global.fft().size(0);
    grid_size[1] = global.fft().size(1);
    grid_size[2] = global.fft().size(2);
}

void FORTRAN(sirius_get_fft_grid_limits)(int32_t* d, int32_t* ul, int32_t* val)
{
    *val = global.fft().grid_limits(*d, *ul);
}

void FORTRAN(sirius_get_fft_index)(int32_t* fft_index)
{
    memcpy(fft_index, global.fft_index(),  global.fft().size() * sizeof(int32_t));
    for (int i = 0; i < global.fft().size(); i++) fft_index[i]++;
}

void FORTRAN(sirius_get_gvec)(int32_t* gvec)
{
    memcpy(gvec, global.gvec(0), 3 * global.fft().size() * sizeof(int32_t));
}

void FORTRAN(sirius_get_index_by_gvec)(int32_t* index_by_gvec)
{
    memcpy(index_by_gvec, global.index_by_gvec(), global.fft().size() * sizeof(int32_t));
    for (int i = 0; i < global.fft().size(); i++) index_by_gvec[i]++;
}

void FORTRAN(sirius_get_num_electrons)(real8* num_electrons)
{
    *num_electrons = global.num_electrons();
}

void FORTRAN(sirius_get_num_valence_electrons)(real8* num_valence_electrons)
{
    *num_valence_electrons = global.num_valence_electrons();
}

void FORTRAN(sirius_get_num_core_electrons)(real8* num_core_electrons)
{
    *num_core_electrons = global.num_core_electrons();
}

void FORTRAN(sirius_add_atom_type)(int32_t* atom_type_id, char* _label, int32_t label_len)
{
    std::string label(_label, label_len);
    global.add_atom_type(*atom_type_id, label);
}

void FORTRAN(sirius_add_atom)(int32_t* atom_type_id, real8* position, real8* vector_field)
{
    global.add_atom(*atom_type_id, position, vector_field);
}

/*
    main functions
*/
void FORTRAN(sirius_platform_initialize)(int32_t* call_mpi_init_)
{
    bool call_mpi_init = (*call_mpi_init_ != 0) ? true : false; 
    Platform::initialize(call_mpi_init);
}

void FORTRAN(sirius_global_initialize)()
{
    global.initialize();
}

void FORTRAN(sirius_potential_initialize)(void)
{
    potential = new sirius::Potential(global);
}

void FORTRAN(sirius_density_initialize)(int32_t* num_kpoints, double* kpoints_, double* kpoint_weights)
{
    mdarray<double, 2> kpoints(kpoints_, 3, *num_kpoints); 
    density = new sirius::Density(global, potential, kpoints, kpoint_weights);
}

void FORTRAN(sirius_clear)(void)
{
    global.clear();
}

void FORTRAN(sirius_initial_density)(void)
{
    density->initial_density();
}

void FORTRAN(sirius_generate_effective_potential)(void)
{
    potential->generate_effective_potential(density->rho(), density->magnetization());
}

void FORTRAN(sirius_generate_density)(void)
{
    density->generate();
}

void FORTRAN(sirius_density_find_eigen_states)(void)
{
    density->find_eigen_states();
}

void FORTRAN(sirius_density_find_band_occupancies)(void)
{
    density->find_band_occupancies();
}

void FORTRAN(sirius_density_set_band_occupancies)(int32_t* ik_, real8* band_occupancies)
{
    int ik = *ik_ - 1;
    density->set_band_occupancies(ik, band_occupancies);
}

void FORTRAN(sirius_density_get_band_energies)(int32_t* ik_, real8* band_energies)
{
    int ik = *ik_ - 1;
    density->get_band_energies(ik, band_energies);
}

void FORTRAN(sirius_density_get_band_occupancies)(int32_t* ik_, real8* band_occupancies)
{
    int ik = *ik_ - 1;
    density->get_band_occupancies(ik, band_occupancies);
}

void FORTRAN(sirius_density_integrate)(void)
{
    density->integrate();
}

/*
    print info
*/
void FORTRAN(sirius_print_info)(void)
{
    global.print_info();
    if (density) density->print_info();
}

void FORTRAN(sirius_print_timers)(void)
{
   sirius::Timer::print();
}   

void FORTRAN(sirius_timer_start)(char* name_, int32_t name_len)
{
    std::string name(name_, name_len);
    sirius::ftimers[name] = new sirius::Timer(name);
}

void FORTRAN(sirius_timer_stop)(char* name_, int32_t name_len)
{
    std::string name(name_, name_len);
    if (sirius::ftimers.count(name)) delete sirius::ftimers[name];
}

void FORTRAN(sirius_read_state)()
{
    potential->hdf5_read();
    sirius:: hdf5_tree fout("sirius.h5", false);
    fout.read("energy_fermi", &global.rti().energy_fermi);
}

void FORTRAN(sirius_write_state)()
{
    potential->hdf5_write();
    if (Platform::mpi_rank() == 0)
    {
        sirius::hdf5_tree fout("sirius.h5", false);
        fout.write("energy_fermi", &global.rti().energy_fermi);
    }
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
        global.get_coordinates<cartesian, reciprocal>(vf, vc);
        double length = Utils::vector_length(vc);
        total_path_length += length;
        segment_length.push_back(length);
    }

    std::vector<double> xaxis;

    sirius::kpoint_set kpoint_set_(global.mpi_grid());
    
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
            kpoint_set_.add_kpoint(vf, 0.0, global);

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

    // distribute k-points along the 1-st direction of the MPI grid
    splindex<block> spl_num_kpoints_(kpoint_set_.num_kpoints(), global.mpi_grid().dimension_size(0), 
                                     global.mpi_grid().coordinate(0));

    global.solve_free_atoms();

    potential->set_spherical_potential();
    potential->set_nonspherical_potential();
    global.generate_radial_functions();
    global.generate_radial_integrals();

    // generate plane-wave coefficients of the potential in the interstitial region
    for (int ir = 0; ir < global.fft().size(); ir++)
         potential->effective_potential()->f_it(ir) *= global.step_function(ir);

    global.fft().input(potential->effective_potential()->f_it());
    global.fft().transform(-1);
    global.fft().output(global.num_gvec(), global.fft_index(), potential->effective_potential()->f_pw());
    
    sirius::Band* band = new sirius::Band(global);
    for (int ikloc = 0; ikloc < spl_num_kpoints_.local_size(); ikloc++)
    {
        int ik = spl_num_kpoints_[ikloc];
        kpoint_set_[ik]->initialize(band);
        kpoint_set_[ik]->find_eigen_states(band, potential->effective_potential(),
                                           potential->effective_magnetic_field());
    } 
    // synchronize eigen-values
    kpoint_set_.sync_band_energies(global.num_bands(), spl_num_kpoints_);

    if (global.mpi_grid().root())
    {
        json_write jw("bands.json");
        jw.single("xaxis", xaxis);
        jw.single("Ef", global.rti().energy_fermi);
        
        jw.single("xaxis_ticks", xaxis_ticks);
        jw.single("xaxis_tick_labels", xaxis_tick_labels);
        
        jw.begin_array("plot");
        std::vector<double> yvalues(kpoint_set_.num_kpoints());
        for (int i = 0; i < global.num_bands(); i++)
        {
            jw.begin_set();
            for (int ik = 0; ik < kpoint_set_.num_kpoints(); ik++) yvalues[ik] = kpoint_set_[ik]->band_energy(i);
            jw.single("yvalues", yvalues);
            jw.end_set();
        }
        jw.end_array();



        //FILE* fout = fopen("bands.dat", "w");
        //for (int i = 0; i < global.num_bands(); i++)
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

void FORTRAN(sirius_print_rti)(void)
{
    global.print_rti();
}

void FORTRAN(sirius_write_json_output)(void)
{
    global.write_json_output();
}

void FORTRAN(sirius_get_occupation_matrix)(int32_t* atom_id, complex16* occupation_matrix)
{
    int ia = *atom_id - 1;
    global.atom(ia)->get_occupation_matrix(occupation_matrix);
}

void FORTRAN(sirius_set_uj_correction_matrix)(int32_t* atom_id, int32_t* l, complex16* uj_correction_matrix)
{
    int ia = *atom_id - 1;
    global.atom(ia)->set_uj_correction_matrix(*l, uj_correction_matrix);
}

void FORTRAN(sirius_set_so_correction)(int32_t* so_correction)
{
    if (*so_correction != 0) 
    {
        global.set_so_correction(true);
    }
    else
    {
        global.set_so_correction(false);
    }
}

void FORTRAN(sirius_set_uj_correction)(int32_t* uj_correction)
{
    if (*uj_correction != 0)
    {
        global.set_uj_correction(true);
    }
    else
    {
        global.set_uj_correction(false);
    }
}

void FORTRAN(sirius_platform_mpi_rank)(int32_t* rank)
{
    *rank = Platform::mpi_rank();
}

void FORTRAN(sirius_global_set_sync_flag)(int32_t* flag)
{
    global.set_sync_flag(*flag);
}

void FORTRAN(sirius_global_get_sync_flag)(int32_t* flag)
{
    *flag = global.sync_flag();
}

void FORTRAN(sirius_platform_barrier)(void)
{
    Platform::barrier();
}

void FORTRAN(sirius_get_total_energy)(real8* total_energy)
{
    *total_energy = global.total_energy();
}


} // extern "C"
