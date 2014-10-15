#include <sirius.h>

using namespace sirius;

void write_json_output(Global* p, DFT_ground_state* gs)
{
    //double evalsum1 = gs->kset_->valence_eval_sum();
    //double evalsum2 = gs->core_eval_sum();
    //double ekin = gs->energy_kin();
    double evxc = gs->energy_vxc();
    double eexc = gs->energy_exc();
    //double ebxc = gs->energy_bxc();
    double evha = gs->energy_vha();
    double etot = gs->total_energy();
    //double gap = kset_->band_gap() * ha2ev;
    //double ef = kset_->energy_fermi();
    //double core_leak = density_->core_leakage();
    double enuc = gs->energy_enuc();

    auto ts = Timer::collect_timer_stats();
    if (p->comm().rank() == 0)
    {
        std::string fname = std::string("output_") + p->start_time("%Y%m%d%H%M%S") + std::string(".json");
        JSON_write jw(fname);
        
        jw.single("git_hash", git_hash);
        jw.single("build_date", build_date);
        jw.single("num_ranks", p->comm().size());
        jw.single("max_num_threads", Platform::max_num_threads());
        jw.single("num_fft_threads", Platform::num_fft_threads());
        //jw.single("cyclic_block_size", p->cyclic_block_size());
        jw.single("mpi_grid", p->mpi_grid_dims());
        std::vector<int> fftgrid(3);
        for (int i = 0; i < 3; i++) fftgrid[i] = p->reciprocal_lattice()->fft()->size(i);
        jw.single("fft_grid", fftgrid);
        jw.single("chemical_formula", p->unit_cell()->chemical_formula());
        jw.single("num_atoms", p->unit_cell()->num_atoms());
        jw.single("num_fv_states", p->num_fv_states());
        jw.single("num_bands", p->num_bands());
        jw.single("aw_cutoff", p->aw_cutoff());
        jw.single("pw_cutoff", p->pw_cutoff());
        jw.single("omega", p->unit_cell()->omega());

        jw.begin_set("energy");
        jw.single("total", etot, 8);
        jw.single("evxc", evxc, 8);
        jw.single("eexc", eexc, 8);
        jw.single("evha", evha, 8);
        jw.single("enuc", enuc, 8);
        jw.end_set();
        
        //** if (num_mag_dims())
        //** {
        //**     std::vector<double> v(3, 0);
        //**     v[2] = rti().total_magnetization[0];
        //**     if (num_mag_dims() == 3)
        //**     {
        //**         v[0] = rti().total_magnetization[1];
        //**         v[1] = rti().total_magnetization[2];
        //**     }
        //**     jw.single("total_moment", v);
        //**     jw.single("total_moment_len", Utils::vector_length(&v[0]));
        //** }
        
        //** jw.single("total_energy", total_energy());
        //** jw.single("kinetic_energy", kinetic_energy());
        //** jw.single("energy_veff", rti_.energy_veff);
        //** jw.single("energy_vha", rti_.energy_vha);
        //** jw.single("energy_vxc", rti_.energy_vxc);
        //** jw.single("energy_bxc", rti_.energy_bxc);
        //** jw.single("energy_exc", rti_.energy_exc);
        //** jw.single("energy_enuc", rti_.energy_enuc);
        //** jw.single("core_eval_sum", rti_.core_eval_sum);
        //** jw.single("valence_eval_sum", rti_.valence_eval_sum);
        //** jw.single("band_gap", rti_.band_gap);
        //** jw.single("energy_fermi", rti_.energy_fermi);
        
        jw.single("timers", ts);
    }

}

void dft_loop(cmd_args args)
{
    std::string task_name = args.value<std::string>("task");

    if (!(task_name == "gs_new" || task_name == "gs_restart" || task_name == "gs_relax" || task_name == "test_init"))
        error_global(__FILE__, __LINE__, "wrong task name");
    
    std::vector<int> mpi_grid_dims;
    mpi_grid_dims = args.value< std::vector<int> >("mpi_grid", mpi_grid_dims);
    Global parameters(MPI_COMM_WORLD, mpi_grid_dims);

    JSON_tree parser("sirius.json");

    parameters.read_unit_cell_input();

    parameters.set_lmax_apw(parser["lmax_apw"].get(10));
    parameters.set_lmax_pot(parser["lmax_pot"].get(10));
    parameters.set_lmax_rho(parser["lmax_rho"].get(10));
    parameters.set_pw_cutoff(parser["pw_cutoff"].get(20.0));
    parameters.set_aw_cutoff(parser["aw_cutoff"].get(7.0));
    parameters.set_gk_cutoff(parser["gk_cutoff"].get(7.0));
    
    parameters.unit_cell()->set_auto_rmt(parser["auto_rmt"].get(0));
    int num_mag_dims = parser["num_mag_dims"].get(0);
    int num_spins = (num_mag_dims == 0) ? 1 : 2;

    int bs = parser["cyclic_block_size"].get(64);
    
    parameters.set_num_mag_dims(num_mag_dims);
    parameters.set_num_spins(num_spins);

    parameters.initialize();
    
    BLACS_grid blacs_grid(parameters.mpi_grid().communicator(1 << _dim_row_ | 1 << _dim_col_), 
                          parameters.mpi_grid().dimension_size(_dim_row_),
                          parameters.mpi_grid().dimension_size(_dim_col_), bs);

    #ifdef _MEMORY_USAGE_INFO_
    MEMORY_USAGE_INFO();
    #endif
    
    Potential* potential = new Potential(parameters);
    potential->allocate();

    #ifdef _MEMORY_USAGE_INFO_
    MEMORY_USAGE_INFO();
    #endif

    auto ngridk = parser["ngridk"].get(std::vector<int>(3, 1));
    
    K_set ks(parameters, parameters.mpi_grid().communicator(1 << _dim_k_), blacs_grid, ngridk, parser["use_symmetry"].get(0));

    ks.initialize();
    
    #ifdef _MEMORY_USAGE_INFO_
    MEMORY_USAGE_INFO();
    #endif
    
    Density* density = new Density(parameters);
    density->allocate();
    
    #ifdef _MEMORY_USAGE_INFO_
    MEMORY_USAGE_INFO();
    #endif
    
    if (task_name == "gs_restart")
    {
        if (!Utils::file_exists(storage_file_name)) error_global(__FILE__, __LINE__, "storage file is not found");
        density->load();
        potential->load();
    }
    else
    {
        density->initial_density();

        switch(parameters.esm_type())
        {
            case full_potential_lapwlo:
            case full_potential_pwlo:
            {
                potential->generate_effective_potential(density->rho(), density->magnetization());
                break;
            }
            case ultrasoft_pseudopotential:
            case norm_conserving_pseudopotential:
            {
                potential->generate_effective_potential(density->rho(), density->rho_pseudo_core(), density->magnetization());
                break;
            }
        }

    }
    
    DFT_ground_state dft(parameters, potential, density, &ks);
    double potential_tol = parser["potential_tol"].get(1e-4);
    double energy_tol = parser["energy_tol"].get(1e-4);

    if (task_name == "test_init")
    {
        potential->update_atomic_potential();
        parameters.unit_cell()->generate_radial_functions();
        dft.print_info();
        parameters.create_storage_file();
        density->save();
    }
    if (task_name == "gs_new" || task_name == "gs_restart")
    {
        dft.scf_loop(potential_tol, energy_tol, parser["num_dft_iter"].get(100));
    }
    if (task_name == "gs_relax")
    {
        dft.relax_atom_positions();
    }

    write_json_output(&parameters, &dft);

    delete density;
    delete potential;
    
    parameters.clear();

    Timer::print();

}

int main(int argn, char** argv)
{
    Platform::initialize(1);

    cmd_args args;
    args.register_key("--task=", "{string} name of the task");
    args.register_key("--mpi_grid=", "{vector int} MPI grid dimensions");
    args.parse_args(argn, argv);

    if (argn == 1)
    {
        printf("Usage: ./dft_loop [options] \n");
        args.print_help();
        exit(0);
    }

    dft_loop(args);
    
    Platform::finalize();
}
