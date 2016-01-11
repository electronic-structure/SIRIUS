#include "simulation_context.h"

namespace sirius {

void Simulation_context::init_fft()
{
    auto rlv = unit_cell_.reciprocal_lattice_vectors();

    int nfft_streams = num_fft_streams();
    int nfft_workers = num_fft_workers();

    /* for now, use parallel fft only in pseudopotential part of the code */
    bool do_parallel_fft = !full_potential();

    auto& comm = mpi_grid_->communicator(1 << _dim_col_ | 1 << _dim_row_);

    mpi_grid_fft_ = (do_parallel_fft) ? new MPI_grid({mpi_grid_->dimension_size(_dim_col_), mpi_grid_->dimension_size(_dim_row_)}, comm)
                                      : new MPI_grid({comm.size(), 1}, comm);

    FFT3D_grid fft_grid(pw_cutoff(), rlv);
    FFT3D_grid fft_coarse_grid(2 * gk_cutoff(), rlv);

    fft_ctx_ = new FFT3D_context(*mpi_grid_fft_, fft_grid, nfft_streams, nfft_workers, processing_unit());
    if (!full_potential())
    {
        fft_coarse_ctx_ = new FFT3D_context(*mpi_grid_fft_, fft_coarse_grid, nfft_streams, nfft_workers,
                                            processing_unit());
    }

    /* create a list of G-vectors for dense FFT grid */
    gvec_ = Gvec(vector3d<double>(0, 0, 0), rlv, pw_cutoff(), fft_grid,
                 mpi_grid_fft_->communicator(1 << 1), mpi_grid_fft_->dimension_size(0), true, false);

    if (!full_potential())
    {
        /* create a list of G-vectors for corase FFT grid */
        gvec_coarse_ = Gvec(vector3d<double>(0, 0, 0), rlv, gk_cutoff() * 2, fft_coarse_grid,
                            mpi_grid_fft_->communicator(1 << 1), mpi_grid_fft_->dimension_size(0), false, false);
    }
}

void Simulation_context::initialize()
{
    PROFILE();

    if (initialized_) TERMINATE("Simulation context is already initialized.");
    
    /* check if we can use a GPU device */
    if (processing_unit() == GPU)
    {
        #ifndef __GPU
        TERMINATE_NO_GPU
        #endif
    }

    /* check MPI grid dimensions and set a default grid if needed */
    if (!mpi_grid_dims_.size()) mpi_grid_dims_ = {comm_.size()};

    /* setup MPI grid */
    mpi_grid_ = new MPI_grid(mpi_grid_dims_, comm_);

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

    //parameters_.set_lmax_beta(unit_cell_.lmax_beta());

    #ifdef __PRINT_MEMORY_USAGE
    MEMORY_USAGE_INFO();
    #endif

    if (unit_cell_.num_atoms() != 0) unit_cell_.symmetry()->check_gvec_symmetry(gvec_);
    
    if (!full_potential())
    {
        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++)
            augmentation_op_.push_back(new Augmentation_operator(comm_, unit_cell_.atom_type(iat), gvec_, unit_cell_.omega()));
    }
    
    #ifdef __PRINT_MEMORY_USAGE
    MEMORY_USAGE_INFO();
    #endif

    if (full_potential()) step_function_ = new Step_function(unit_cell_, fft_ctx_->fft(0), gvec_, comm_);

    if (iterative_solver_input_section().real_space_prj_) 
    {
        real_space_prj_ = new Real_space_prj(unit_cell_, comm_, iterative_solver_input_section().R_mask_scale_,
                                             iterative_solver_input_section().mask_alpha_,
                                             gk_cutoff(), num_fft_streams(), num_fft_workers());
    }

    /* take 10% of empty non-magnetic states */
    if (num_fv_states_ < 0) 
    {
        num_fv_states_ = static_cast<int>(1e-8 + unit_cell_.num_valence_electrons() / 2.0) +
                                          std::max(10, static_cast<int>(0.1 * unit_cell_.num_valence_electrons()));
    }
    
    if (num_fv_states() < int(unit_cell_.num_valence_electrons() / 2.0))
        TERMINATE("not enough first-variational states");
    
    std::map<std::string, ev_solver_t> str_to_ev_solver_t;

    str_to_ev_solver_t["lapack"]    = ev_lapack;
    str_to_ev_solver_t["scalapack"] = ev_scalapack;
    str_to_ev_solver_t["elpa1"]     = ev_elpa1;
    str_to_ev_solver_t["elpa2"]     = ev_elpa2;
    str_to_ev_solver_t["magma"]     = ev_magma;
    str_to_ev_solver_t["plasma"]    = ev_plasma;
    str_to_ev_solver_t["rs_cpu"]    = ev_rs_cpu;
    str_to_ev_solver_t["rs_gpu"]    = ev_rs_gpu;

    std::string evsn[] = {std_evp_solver_name(), gen_evp_solver_name()};

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

    for (int i: {0, 1})
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
