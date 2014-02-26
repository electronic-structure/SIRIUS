#include "global.h"

namespace sirius {

void Global::read_input()
{
    std::string fname("sirius.json");
    
    int num_fft_threads = Platform::num_fft_threads();
    if (num_fft_threads == -1) num_fft_threads = Platform::max_num_threads();

    if (Utils::file_exists(fname))
    {
        JSON_tree parser(fname);
        mpi_grid_dims_ = parser["mpi_grid_dims"].get(mpi_grid_dims_); 
        cyclic_block_size_ = parser["cyclic_block_size"].get(cyclic_block_size_);
        num_fft_threads = parser["num_fft_threads"].get(num_fft_threads);
        num_fv_states_ = parser["num_fv_states"].get(num_fv_states_);
        smearing_width_ = parser["smearing_width"].get(smearing_width_);
        
        if (parser.exist("eigen_value_solver"))
        {
            std::string ev_solver_name;
            parser["eigen_value_solver"] >> ev_solver_name;
            if (ev_solver_name == "lapack") 
            {
                eigen_value_solver_ = lapack;
            }
            else if (ev_solver_name == "scalapack") 
            {
                eigen_value_solver_ = scalapack;
            }
            else if (ev_solver_name == "elpa") 
            {
                eigen_value_solver_ = elpa;
            }
            else if (ev_solver_name == "magma") 
            {
                eigen_value_solver_ = magma;
            }
            else if (ev_solver_name == "plasma")
            {
                eigen_value_solver_ = plasma;
            }
            else
            {
                error_local(__FILE__, __LINE__, "wrong eigen value solver");
            }
        }

        if (parser.exist("processing_unit"))
        {
            std::string pu;
            parser["processing_unit"] >> pu;
            if (pu == "cpu")
            {
                processing_unit_ = cpu;
            }
            else if (pu == "gpu")
            {
                processing_unit_ = gpu;
            }
            else
            {
                error_local(__FILE__, __LINE__, "wrong processing unit");
            }
        }

        if (parser.exist("electronic_structure_method"))
        {
            std::string str;
            parser["electronic_structure_method"] >> str;
            if (str == "full_potential_lapwlo")
            {
                esm_type_ = full_potential_lapwlo;
            }
            else if (str == "full_potential_pwlo")
            {
                esm_type_ = full_potential_pwlo;
            }
            else if (str == "ultrasoft_pseudopotential")
            {
                esm_type_ = ultrasoft_pseudopotential;
            }
            else
            {
                error_local(__FILE__, __LINE__, "wrong type of electronic structure method");
            }
        }

        mixer_input_section_.read(parser);
    }


    Platform::set_num_fft_threads(std::min(num_fft_threads, Platform::max_num_threads()));
}

void Global::read_unit_cell_input()
{
    std::string fname("sirius.json");
    JSON_tree parser(fname);
    unit_cell_input_section_.read(parser);
        
    for (int iat = 0; iat < (int)unit_cell_input_section_.labels_.size(); iat++)
    {
        unit_cell()->add_atom_type(iat, unit_cell_input_section_.labels_[iat], esm_type());
        for (int ia = 0; ia < (int)unit_cell_input_section_.coordinates_[iat].size(); ia++)
        {
            std::vector<double> v = unit_cell_input_section_.coordinates_[iat][ia];
            unit_cell()->add_atom(iat, &v[0], &v[3]);
        }
    }

    unit_cell()->set_lattice_vectors(unit_cell_input_section_.lattice_vectors_[0], 
                                     unit_cell_input_section_.lattice_vectors_[1], 
                                     unit_cell_input_section_.lattice_vectors_[2]);
}

std::string Global::start_time(const char* fmt)
{
    char buf[100]; 
    
    tm* ptm = localtime(&start_time_.tv_sec); 
    strftime(buf, sizeof(buf), fmt, ptm); 
    return std::string(buf);
}

void Global::initialize()
{
    if (initialized_) error_local(__FILE__, __LINE__, "Can't initialize global variables more than once.");

    switch (esm_type())
    {
        case full_potential_lapwlo:
        {
            break;
        }
        case full_potential_pwlo:
        {
            lmax_pw_ = lmax_apw_;
            lmax_apw_ = -1;
            break;
        }
        case ultrasoft_pseudopotential:
        {
            lmax_apw_ = lmax_rho_ = lmax_pot_ = -1;
            break;
        }
    }

    // initialize variables, related to the unit cell
    unit_cell_->initialize(lmax_apw(), lmax_pot(), num_mag_dims());

    // create a reciprocal lattice
    int lmax = -1;
    switch (esm_type())
    {
        case full_potential_lapwlo:
        {
            lmax = lmax_pot_;
            break;
        }
        case full_potential_pwlo:
        {
            stop_here
        }
        case ultrasoft_pseudopotential:
        {
            lmax = 2 * unit_cell_->lmax_beta();
            break;
        }
    }
    reciprocal_lattice_ = new Reciprocal_lattice(unit_cell_, esm_type(), pw_cutoff(), gk_cutoff(), lmax);

    if (unit_cell_->full_potential()) step_function_ = new Step_function(unit_cell_, reciprocal_lattice_);

    // check MPI grid dimensions and set a default grid if needed
    if (!mpi_grid_dims_.size()) 
    {
        mpi_grid_dims_ = std::vector<int>(1);
        mpi_grid_dims_[0] = Platform::num_mpi_ranks();
    }

    // setup MPI grid
    mpi_grid_.initialize(mpi_grid_dims_);
    
    if (num_fv_states_ < 0) num_fv_states_ = int(unit_cell_->num_valence_electrons() / 2.0) + 20;
    if (num_fv_states_ < int(unit_cell_->num_valence_electrons() / 2.0))
        error_global(__FILE__, __LINE__, "not enough first-variational states");

    int nrow = mpi_grid().dimension_size(_dim_row_);
    int ncol = mpi_grid().dimension_size(_dim_col_);
    
    int irow = mpi_grid().coordinate(_dim_row_);
    int icol = mpi_grid().coordinate(_dim_col_);

    if (eigen_value_solver() == scalapack || eigen_value_solver() == elpa)
    {
        int n = num_fv_states_ / (ncol * cyclic_block_size()) + 
                std::min(1, num_fv_states_ % (ncol * cyclic_block_size()));

        while ((n * ncol) % nrow) n++;
        
        num_fv_states_ = n * ncol * cyclic_block_size();

        #if defined(_SCALAPACK_) || defined(_ELPA_)
        int rc = (1 << _dim_row_) | (1 << _dim_col_);
        MPI_Comm comm = mpi_grid().communicator(rc);
        blacs_context_ = linalg<scalapack>::create_blacs_context(comm);

        mdarray<int, 2> map_ranks(nrow, ncol);
        for (int i = 0; i < nrow; i++)
        {
            for (int j = 0; j < ncol; j++)
            {
                std::vector<int> xy(2);
                xy[0] = j;
                xy[1] = i;
                map_ranks(i, j) = mpi_grid().cart_rank(comm, xy);
            }
        }
        linalg<scalapack>::gridmap(&blacs_context_, map_ranks.get_ptr(), map_ranks.ld(), nrow, ncol);

        // check the grid
        int nrow1, ncol1, irow1, icol1;
        linalg<scalapack>::gridinfo(blacs_context_, &nrow1, &ncol1, &irow1, &icol1);

        if (irow != irow1 || icol != icol1 || nrow != nrow1 || ncol != ncol1) 
        {
            std::stringstream s;
            s << "wrong grid" << std::endl
              << "            row | col | nrow | ncol " << std::endl
              << " mpi_grid " << irow << " " << icol << " " << nrow << " " << ncol << std::endl  
              << " blacs    " << irow1 << " " << icol1 << " " << nrow1 << " " << ncol1;
            error_local(__FILE__, __LINE__, s);
        }
        #endif
    }

    num_bands_ = num_fv_states_ * num_spins_;

    // distribue first-variational states along columns
    spl_fv_states_col_.split(num_fv_states(), ncol, icol, cyclic_block_size());

    // distribue first-variational states along rows
    spl_fv_states_row_.split(num_fv_states(), nrow, irow, cyclic_block_size());

    // distribue spinor wave-functions along columns
    spl_spinor_wf_col_.split(num_bands(), ncol, icol, cyclic_block_size());
    
    // additionally split along rows 
    sub_spl_spinor_wf_.split(spl_spinor_wf_col_.local_size(), nrow, irow);
    
    sub_spl_fv_states_col_.split(spl_fv_states_col().local_size(), nrow, irow);

    // check if the distribution of fv states is consistent with the distribtion of spinor wave functions
    for (int ispn = 0; ispn < num_spins(); ispn++)
    {
        for (int i = 0; i < spl_fv_states_col_.local_size(); i++)
        {
            if (spl_spinor_wf_col_[i + ispn * spl_fv_states_col_.local_size()] != 
                spl_fv_states_col_[i] + ispn * num_fv_states())
            {
                error_local(__FILE__, __LINE__, "Wrong distribution of wave-functions");
            }
        }
    }

    if (verbosity_level >= 3 && Platform::mpi_rank() == 0 && nrow * ncol > 1)
    {
        printf("\n");
        printf("table of column distribution of first-variational states\n");
        printf("(columns of the table correspond to column MPI ranks)\n");
        for (int i0 = 0; i0 < spl_fv_states_col_.local_size(0); i0++)
        {
            for (int i1 = 0; i1 < ncol; i1++) printf("%6i", spl_fv_states_col_.global_index(i0, i1));
            printf("\n");
        }
        
        printf("\n");
        printf("table of row distribution of first-variational states\n");
        printf("(columns of the table correspond to row MPI ranks)\n");
        for (int i0 = 0; i0 < spl_fv_states_row_.local_size(0); i0++)
        {
            for (int i1 = 0; i1 < nrow; i1++) printf("%6i", spl_fv_states_row_.global_index(i0, i1));
            printf("\n");
        }

        printf("\n");
        printf("First-variational states index -> (local index, rank) for column distribution\n");
        for (int i = 0; i < num_fv_states(); i++)
        {
            printf("%6i -> (%6i %6i)\n", i, spl_fv_states_col_.location(_splindex_offs_, i), 
                                            spl_fv_states_col_.location(_splindex_rank_, i));
        }
        
        printf("\n");
        printf("First-variational states index -> (local index, rank) for row distribution\n");
        for (int i = 0; i < num_fv_states(); i++)
        {
            printf("%6i -> (%6i %6i)\n", i, spl_fv_states_row_.location(_splindex_offs_, i), 
                                            spl_fv_states_row_.location(_splindex_rank_, i));
        }
        
        printf("\n");
        printf("table of column distribution of spinor wave functions\n");
        printf("(columns of the table correspond to MPI ranks)\n");
        for (int i0 = 0; i0 < spl_spinor_wf_col_.local_size(0); i0++)
        {
            for (int i1 = 0; i1 < ncol; i1++) printf("%6i", spl_spinor_wf_col_.global_index(i0, i1));
            printf("\n");
        }
    }
    
    if (Platform::mpi_rank() == 0 && verbosity_level >= 1) print_info();
    initialized_ = true;
}

void Global::clear()
{
    if (initialized_)
    {
        unit_cell_->clear();
        delete reciprocal_lattice_;
        delete step_function_;
        mpi_grid_.finalize();
        initialized_ = false;
    }
}


void Global::print_info()
{
    printf("\n");
    printf("SIRIUS version : %2i.%02i\n", major_version, minor_version);
    printf("git hash       : %s\n", git_hash);
    printf("build date     : %s\n", build_date);
    printf("start time     : %s\n", start_time("%c").c_str());
    printf("\n");
    printf("number of MPI ranks           : %i\n", Platform::num_mpi_ranks());
    printf("MPI grid                      :");
    for (int i = 0; i < mpi_grid_.num_dimensions(); i++) printf(" %i", mpi_grid_.size(1 << i));
    printf("\n");
    printf("maximum number of OMP threads : %i\n", Platform::max_num_threads()); 
    printf("number of OMP threads for FFT : %i\n", Platform::num_fft_threads()); 

    unit_cell_->print_info();
    reciprocal_lattice_->print_info();

    for (int i = 0; i < unit_cell_->num_atom_types(); i++) unit_cell_->atom_type(i)->print_info();

    printf("\n");
    printf("total number of aw basis functions : %i\n", unit_cell_->mt_aw_basis_size());
    printf("total number of lo basis functions : %i\n", unit_cell_->mt_lo_basis_size());
    printf("number of first-variational states : %i\n", num_fv_states());
    printf("number of bands                    : %i\n", num_bands());
    printf("number of spins                    : %i\n", num_spins());
    printf("number of magnetic dimensions      : %i\n", num_mag_dims());
    printf("lmax_apw                           : %i\n", lmax_apw());
    printf("lmax_pw                            : %i\n", lmax_pw());
    printf("lmax_rho                           : %i\n", lmax_rho());
    printf("lmax_pot                           : %i\n", lmax_pot());
    printf("lmax_beta                          : %i\n", lmax_beta());

    printf("\n");
    printf("eigen-value solver: ");
    switch (eigen_value_solver())
    {
        case lapack:
        {
            printf("LAPACK\n");
            break;
        }
        case scalapack:
        {
            printf("ScaLAPACK, block size %i\n", cyclic_block_size());
            break;
        }
        case elpa:
        {
            printf("ELPA, block size %i\n", cyclic_block_size());
            break;
        }
        case magma:
        {
            printf("MAGMA\n");
            break;
        }
        case plasma:
        {
            printf("PLASMA\n");
            break;
        }
    }
    printf("processing unit : ");
    switch (processing_unit())
    {
        case cpu:
        {
            printf("CPU\n");
            break;
        }
        case gpu:
        {
            printf("GPU\n");
            break;
        }
    }
    unit_cell_->write_cif();
}
//==         
//==         //** /// Print run-time information.
//==         //** void print_rti()
//==         //** {
//==         //**     if (Platform::mpi_rank() == 0)
//==         //**     {
//==         //**         double total_core_leakage = 0.0;
//== 
//==         //**         printf("\n");
//==         //**         printf("Charges and magnetic moments\n");
//==         //**         for (int i = 0; i < 80; i++) printf("-");
//==         //**         printf("\n"); 
//==         //**         printf("atom      charge    core leakage");
//==         //**         if (num_mag_dims()) printf("              moment              |moment|");
//==         //**         printf("\n");
//==         //**         for (int i = 0; i < 80; i++) printf("-");
//==         //**         printf("\n"); 
//== 
//==         //**         for (int ia = 0; ia < num_atoms(); ia++)
//==         //**         {
//==         //**             double core_leakage = atom(ia)->symmetry_class()->core_leakage();
//==         //**             total_core_leakage += core_leakage;
//==         //**             printf("%4i  %10.6f  %10.8e", ia, rti().mt_charge[ia], core_leakage);
//==         //**             if (num_mag_dims())
//==         //**             {
//==         //**                 double v[] = {0, 0, 0};
//==         //**                 v[2] = rti().mt_magnetization[0][ia];
//==         //**                 if (num_mag_dims() == 3)
//==         //**                 {
//==         //**                     v[0] = rti().mt_magnetization[1][ia];
//==         //**                     v[1] = rti().mt_magnetization[2][ia];
//==         //**                 }
//==         //**                 printf("  (%8.4f %8.4f %8.4f)  %10.6f", v[0], v[1], v[2], Utils::vector_length(v));
//==         //**             }
//==         //**             printf("\n");
//==         //**         }
//==         //**         
//==         //**         printf("\n");
//==         //**         printf("interstitial charge   : %10.6f\n", rti().it_charge);
//==         //**         if (num_mag_dims())
//==         //**         {
//==         //**             double v[] = {0, 0, 0};
//==         //**             v[2] = rti().it_magnetization[0];
//==         //**             if (num_mag_dims() == 3)
//==         //**             {
//==         //**                 v[0] = rti().it_magnetization[1];
//==         //**                 v[1] = rti().it_magnetization[2];
//==         //**             }
//==         //**             printf("interstitial moment   : (%8.4f %8.4f %8.4f)\n", v[0], v[1], v[2]);
//==         //**             printf("interstitial |moment| : %10.6f\n", Utils::vector_length(v));
//==         //**         }
//==         //**         
//==         //**         printf("\n");
//==         //**         printf("total charge          : %10.6f\n", rti().total_charge);
//==         //**         printf("total core leakage    : %10.8e\n", total_core_leakage);
//==         //**         if (num_mag_dims())
//==         //**         {
//==         //**             double v[] = {0, 0, 0};
//==         //**             v[2] = rti().total_magnetization[0];
//==         //**             if (num_mag_dims() == 3)
//==         //**             {
//==         //**                 v[0] = rti().total_magnetization[1];
//==         //**                 v[1] = rti().total_magnetization[2];
//==         //**             }
//==         //**             printf("total moment          : (%8.4f %8.4f %8.4f)\n", v[0], v[1], v[2]);
//==         //**             printf("total |moment|        : %10.6f\n", Utils::vector_length(v));
//==         //**         }
//==         //**         printf("pseudo charge error : %18.12f\n", rti().pseudo_charge_error);
//==         //**         
//==         //**         printf("\n");
//==         //**         printf("Energy\n");
//==         //**         for (int i = 0; i < 80; i++) printf("-");
//==         //**         printf("\n"); 
//== 
//==         //**         printf("valence_eval_sum : %18.8f\n", rti().valence_eval_sum);
//==         //**         printf("core_eval_sum    : %18.8f\n", rti().core_eval_sum);
//== 
//==         //**         printf("kinetic energy   : %18.8f\n", kinetic_energy());
//==         //**         printf("<rho|V^{XC}>     : %18.8f\n", rti().energy_vxc);
//==         //**         printf("<rho|E^{XC}>     : %18.8f\n", rti().energy_exc);
//==         //**         printf("<mag|B^{XC}>     : %18.8f\n", rti().energy_bxc);
//==         //**         printf("<rho|V^{H}>      : %18.8f\n", rti().energy_vha);
//==         //**         printf("Total energy     : %18.8f\n", total_energy());
//== 
//==         //**         printf("\n");
//==         //**         printf("band gap (eV) : %18.8f\n", rti().band_gap * ha2ev);
//==         //**         printf("Efermi        : %18.8f\n", rti().energy_fermi);
//==         //**     }
//==         //** }
//== 
void Global::write_json_output()
{
    if (Platform::mpi_rank() == 0)
    {
        std::string fname = std::string("output_") + start_time("%Y%m%d%H%M%S") + std::string(".json");
        JSON_write jw(fname);
        
        jw.single("git_hash", git_hash);
        jw.single("build_date", build_date);
        jw.single("num_ranks", Platform::num_mpi_ranks());
        jw.single("max_num_threads", Platform::max_num_threads());
        jw.single("num_fft_threads", Platform::num_fft_threads());
        jw.single("cyclic_block_size", cyclic_block_size());
        jw.single("mpi_grid", mpi_grid_dims_);
        std::vector<int> fftgrid(3);
        for (int i = 0; i < 3; i++) fftgrid[i] = reciprocal_lattice_->fft()->size(i);
        jw.single("fft_grid", fftgrid);
        jw.single("chemical_formula", unit_cell()->chemical_formula());
        jw.single("num_atoms", unit_cell()->num_atoms());
        jw.single("num_fv_states", num_fv_states());
        jw.single("num_bands", num_bands());
        jw.single("aw_cutoff", aw_cutoff());
        jw.single("pw_cutoff", pw_cutoff());
        jw.single("omega", unit_cell()->omega());
        
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
        
        jw.single("timers", Timer::timers());
    }
}

void Global::create_storage_file()
{
    if (Platform::mpi_rank() == 0) 
    {
        // create new hdf5 file
        HDF5_tree fout(storage_file_name, true);
        fout.create_node("parameters");
        fout.create_node("effective_potential");
        fout.create_node("effective_magnetic_field");
        fout.create_node("density");
        fout.create_node("magnetization");
        
        fout["parameters"].write("num_spins", num_spins());
        fout["parameters"].write("num_mag_dims", num_mag_dims());
        fout["parameters"].write("num_bands", num_bands());
    }
    Platform::barrier();
}

void Global::update()
{
    unit_cell_->update();
    reciprocal_lattice_->update();
    delete step_function_;
    step_function_ = new Step_function(unit_cell_, reciprocal_lattice_);
}

//==        
//==         inline int blacs_context()
//==         {
//==             #ifdef _SCALAPACK_
//==             return blacs_context_;
//==             #else
//==             return -1;
//==             #endif
//==         }
//== 
//==         //inline potential_t potential_type()
//==         //{
//==         //    return potential_type_;
//==         //}
//== 
//==         //inline basis_t basis_type()
//==         //{
//==         //    return basis_type_;
//==         //}
//== 
//==         inline electronic_structure_method_t esm_type()
//==         {
//==             return esm_type_;
//==         }
//== 
//==         inline Step_function* step_function()
//==         {
//==             return step_function_;
//==         }
//== 
//==         inline double step_function(int ir)
//==         {
//==             return step_function_->theta_it(ir);
//==         }
//== 
//==         inline Reciprocal_lattice* reciprocal_lattice()
//==         {
//==             return reciprocal_lattice_;
//==         }
//== 
//==         inline Unit_cell* unit_cell()
//==         {
//==             return unit_cell_;
//==         }
//==         
//==         struct mixer_input_section
//==         {
//==             double beta_;
//==             std::string type_;
//==             int max_history_;
//== 
//==             mixer_input_section() : beta_(0.9), type_("broyden"), max_history_(5)
//==             {
//==             }
//== 
//==             void read(JSON_tree parser)
//==             {
//==                 if (parser.exist("mixer"))
//==                 {
//==                     JSON_tree section = parser["mixer"];
//==                     beta_ = section["beta"].get(beta_);
//==                     max_history_ = section["max_history"].get(max_history_);
//==                     type_ = section["type"].get(type_);
//==                 }
//==             }
//==         } mixer_input_section_;
//== 
//==         inline double iterative_solver_tolerance()
//==         {
//==             return iterative_solver_tolerance_;
//==         }
//== 
//==         inline void set_iterative_solver_tolerance(double tol)
//==         {
//==             iterative_solver_tolerance_ = tol;
//==         }
//== };
//== 
}

