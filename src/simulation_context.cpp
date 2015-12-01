#include "simulation_context.h"

namespace sirius {

void Simulation_context::init_fft()
{
    auto rlv = unit_cell_.reciprocal_lattice_vectors();

    int nfft_threads = parameters_.num_fft_threads();
    int nfft_workers = parameters_.num_fft_workers();

    bool do_parallel_fft = true; //(mpi_grid_->dimension_size(_dim_row_) > 1 && !parameters_.full_potential());

    auto& comm = mpi_grid_->communicator(1 << _dim_col_ | 1 << _dim_row_);

    if (do_parallel_fft)
    {
        mpi_grid_fft_ = new MPI_grid({mpi_grid_->dimension_size(_dim_col_), mpi_grid_->dimension_size(_dim_row_)}, comm);
    }
    else
    {
        mpi_grid_fft_ = new MPI_grid({comm.size(), 1}, comm);
    }

    if (do_parallel_fft)
    {
        nfft_workers *= nfft_threads;
        nfft_threads = 1;
    }

    gpu_thread_id_ = -1;
    if (nfft_threads > 1) gpu_thread_id_ = nfft_threads - 1;

    FFT3D_grid fft_grid(parameters_.pw_cutoff(), rlv);
    FFT3D_grid fft_coarse_grid(2 * parameters_.gk_cutoff(), rlv);

    fft_ctx_ = new FFT3D_context(*mpi_grid_fft_, fft_grid, nfft_threads, nfft_workers, parameters_.processing_unit());
    if (!parameters_.full_potential())
    {
        fft_coarse_ctx_ = new FFT3D_context(*mpi_grid_fft_, fft_coarse_grid, nfft_threads, nfft_workers,
                                            parameters_.processing_unit());
    }

    //for (int tid = 0; tid < nfft_threads; tid++)
    //{
    //    /* in case of parallel FFT */
    //    if (do_parallel_fft)
    //    {
    //        fft_.push_back(new FFT3D(Utils::find_translations(parameters_.pw_cutoff(), rlv),
    //                                 nfft_workers, mpi_grid_fft_->communicator(1 << 1), CPU));
    //        if (!parameters_.full_potential())
    //        {
    //            fft_coarse_.push_back(new FFT3D(Utils::find_translations(2 * parameters_.gk_cutoff(), rlv),
    //                                            nfft_workers, mpi_grid_fft_->communicator(1 << 1), parameters_.processing_unit()));
    //        }
    //    }
    //    else /* serial FFT driver */
    //    {
    //        if (tid == gpu_thread_id_)
    //        {
    //            fft_.push_back(new FFT3D(Utils::find_translations(parameters_.pw_cutoff(), rlv),
    //                                     nfft_workers, mpi_comm_self, parameters_.processing_unit()));
    //        }
    //        else
    //        {
    //            fft_.push_back(new FFT3D(Utils::find_translations(parameters_.pw_cutoff(), rlv),
    //                                     nfft_workers, mpi_comm_self, CPU));
    //        }
    //        if (!parameters_.full_potential())
    //        {
    //            if (tid == gpu_thread_id_)
    //            {
    //                fft_coarse_.push_back(new FFT3D(Utils::find_translations(2 * parameters_.gk_cutoff(), rlv),
    //                                                nfft_workers, mpi_comm_self, parameters_.processing_unit()));
    //            }
    //            else
    //            {
    //                fft_coarse_.push_back(new FFT3D(Utils::find_translations(2 * parameters_.gk_cutoff(), rlv),
    //                                                nfft_workers, mpi_comm_self, CPU));
    //            }
    //        }
    //    }
    //}

    /* create a list of G-vectors for dense FFT grid */
    gvec_ = Gvec(vector3d<double>(0, 0, 0), rlv, parameters_.pw_cutoff(), fft_grid,
                 mpi_grid_fft_->communicator(1 << 1), mpi_grid_fft_->dimension_size(0), true, false);

    if (!parameters_.full_potential())
    {
        /* create a list of G-vectors for corase FFT grid */
        gvec_coarse_ = Gvec(vector3d<double>(0, 0, 0), rlv, parameters_.gk_cutoff() * 2, fft_coarse_grid,
                            mpi_grid_fft_->communicator(1 << 1), mpi_grid_fft_->dimension_size(0), false, false);
    }
}

void Simulation_context::initialize()
{
    PROFILE();

    if (initialized_) TERMINATE("Simulation context is already initialized.");
    
    /* check if we can use a GPU device */
    if (parameters_.processing_unit() == GPU)
    {
        #ifndef __GPU
        TERMINATE_NO_GPU
        #endif
    }

    switch (parameters_.esm_type())
    {
        case full_potential_lapwlo:
        {
            break;
        }
        case full_potential_pwlo:
        {
            parameters_.set_lmax_pw(parameters_.lmax_apw());
            parameters_.set_lmax_apw(-1);
            break;
        }
        case ultrasoft_pseudopotential:
        case norm_conserving_pseudopotential:
        {
            parameters_.set_lmax_apw(-1);
            parameters_.set_lmax_rho(-1);
            parameters_.set_lmax_pot(-1);
            break;
        }
    }

    /* check MPI grid dimensions and set a default grid if needed */
    auto mpi_grid_dims = parameters_.mpi_grid_dims();
    if (!mpi_grid_dims.size()) 
    {
        mpi_grid_dims = std::vector<int>(1);
        mpi_grid_dims[0] = comm_.size();
    }
    parameters_.set_mpi_grid_dims(mpi_grid_dims);

    /* setup MPI grid */
    mpi_grid_ = new MPI_grid(mpi_grid_dims, comm_);

    /* initialize variables, related to the unit cell */
    unit_cell_.initialize();

    /* initialize FFT subsystem */
    init_fft();

    #ifdef __PRINT_MEMORY_USAGE
    MEMORY_USAGE_INFO();
    #endif

    if (comm_.rank() == 0)
    {
        unit_cell_.write_cif();
        unit_cell_.write_json();
    }

    parameters_.set_lmax_beta(unit_cell_.lmax_beta());

    #ifdef __PRINT_MEMORY_USAGE
    MEMORY_USAGE_INFO();
    #endif

    if (unit_cell_.num_atoms() != 0) unit_cell_.symmetry()->check_gvec_symmetry(gvec_);

    /* create a reciprocal lattice */
    int lmax = -1;
    switch (parameters_.esm_type())
    {
        case full_potential_lapwlo:
        {
            lmax = parameters_.lmax_pot();
            break;
        }
        case full_potential_pwlo:
        {
            STOP();
        }
        case ultrasoft_pseudopotential:
        case norm_conserving_pseudopotential:
        {
            lmax = 2 * parameters_.lmax_beta();
            break;
        }
    }
    
    reciprocal_lattice_ = new Reciprocal_lattice(unit_cell_, parameters_.esm_type(), gvec_, lmax, comm_);

    #ifdef __PRINT_MEMORY_USAGE
    MEMORY_USAGE_INFO();
    #endif

    if (parameters_.full_potential()) step_function_ = new Step_function(unit_cell_, reciprocal_lattice_, fft_ctx_->fft(0), gvec_, comm_);

    if (parameters_.iterative_solver_input_section().real_space_prj_) 
    {
        real_space_prj_ = new Real_space_prj(unit_cell_, comm_, parameters_.iterative_solver_input_section().R_mask_scale_,
                                             parameters_.iterative_solver_input_section().mask_alpha_,
                                             parameters_.gk_cutoff(), parameters_.num_fft_threads(),
                                             parameters_.num_fft_workers());
    }

    /* take 20% of empty non-magnetic states */
    if (parameters_.num_fv_states() < 0) 
    {
        int nfv = static_cast<int>(1e-8 + unit_cell_.num_valence_electrons() / 2.0) +
                                   std::max(10, static_cast<int>(0.1 * unit_cell_.num_valence_electrons()));
        parameters_.set_num_fv_states(nfv);
    }
    
    if (parameters_.num_fv_states() < int(unit_cell_.num_valence_electrons() / 2.0))
        TERMINATE("not enough first-variational states");
    
    /* total number of bands */
    parameters_.set_num_bands(parameters_.num_fv_states() * parameters_.num_spins());

    std::map<std::string, ev_solver_t> str_to_ev_solver_t;

    str_to_ev_solver_t["lapack"]    = ev_lapack;
    str_to_ev_solver_t["scalapack"] = ev_scalapack;
    str_to_ev_solver_t["elpa1"]     = ev_elpa1;
    str_to_ev_solver_t["elpa2"]     = ev_elpa2;
    str_to_ev_solver_t["magma"]     = ev_magma;
    str_to_ev_solver_t["plasma"]    = ev_plasma;
    str_to_ev_solver_t["rs_cpu"]    = ev_rs_cpu;
    str_to_ev_solver_t["rs_gpu"]    = ev_rs_gpu;

    std::string evsn[] = {parameters_.std_evp_solver_name(), parameters_.gen_evp_solver_name()};

    if (evsn[0] == "")
    {
        if (mpi_grid_->size(1 << _dim_row_ | 1 << _dim_col_) == 1)
        {
            evsn[0] = "lapack";
        }
        else
        {
            evsn[0] = "scalapack";
        }
    }

    if (evsn[1] == "")
    {
        if (mpi_grid_->size(1 << _dim_row_ | 1 << _dim_col_) == 1)
        {
            evsn[1] = "lapack";
        }
        else
        {
            evsn[1] = "elpa1";
        }
    }

    ev_solver_t* evst[] = {&std_evp_solver_type_, &gen_evp_solver_type_};

    for (int i = 0; i < 2; i++)
    {
        auto name = evsn[i];

        if (str_to_ev_solver_t.count(name) == 0)
        {
            std::stringstream s;
            s << "wrong eigen value solver " << name;
            TERMINATE(s);
        }
        *evst[i] = str_to_ev_solver_t[name];
    }

    #if (__VERBOSITY > 0)
    if (comm_.rank() == 0) print_info();
    #endif

    time_active_ = -Utils::current_time();

    initialized_ = true;
}

};
