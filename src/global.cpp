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

/** \file global.cpp
 *   
 *  \brief Contains remaining implementation of sirius::Global class.
 */

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
        iterative_solver_tolerance_ = parser["iterative_solver_tolerance"].get(iterative_solver_tolerance_);

        std::string evsn[] = {"std_evp_solver_type", "gen_evp_solver_type"};
        ev_solver_t* evst[] = {&std_evp_solver_type_, &gen_evp_solver_type_};

        for (int i = 0; i < 2; i++)
        {
            if (parser.exist(evsn[i]))
            {
                std::string name;
                parser[evsn[i]] >> name;
                if (name == "lapack") 
                {
                    *evst[i] = ev_lapack;
                }
                else if (name == "scalapack") 
                {
                    *evst[i] = ev_scalapack;
                }
                else if (name == "elpa1") 
                {
                    *evst[i] = ev_elpa1;
                }
                else if (name == "elpa2") 
                {
                    *evst[i] = ev_elpa2;
                }
                else if (name == "magma") 
                {
                    *evst[i] = ev_magma;
                }
                else if (name == "plasma")
                {
                    *evst[i] = ev_plasma;
                }
                else if (name == "rs_gpu")
                {
                    *evst[i] = ev_rs_gpu;
                }
                else if (name == "rs_cpu")
                {
                    *evst[i] = ev_rs_cpu;
                }
                else
                {
                    error_local(__FILE__, __LINE__, "wrong eigen value solver");
                }
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
        xc_functionals_input_section_.read(parser);
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
        std::string label = unit_cell_input_section_.labels_[iat];
        std::string fname = unit_cell_input_section_.atom_files_[label];
        unit_cell()->add_atom_type(label, fname, esm_type());
        for (int ia = 0; ia < (int)unit_cell_input_section_.coordinates_[iat].size(); ia++)
        {
            std::vector<double> v = unit_cell_input_section_.coordinates_[iat][ia];
            unit_cell()->add_atom(label, &v[0], &v[3]);
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

    /* initialize variables, related to the unit cell */
    unit_cell_->initialize(lmax_apw(), lmax_pot(), num_mag_dims());

    /* create a reciprocal lattice */
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

    /* check MPI grid dimensions and set a default grid if needed */
    if (!mpi_grid_dims_.size()) 
    {
        mpi_grid_dims_ = std::vector<int>(1);
        mpi_grid_dims_[0] = Platform::num_mpi_ranks();
    }

    /* setup MPI grid */
    mpi_grid_.initialize(mpi_grid_dims_);
    
    /* take 20% of empty non-magnetic states */
    if (num_fv_states_ < 0) 
    {
        num_fv_states_ = int(1e-8 + unit_cell_->num_valence_electrons() / 2.0) +
                         std::max(10, int(0.1 * unit_cell_->num_valence_electrons()));
    }

    if (num_fv_states_ < int(unit_cell_->num_valence_electrons() / 2.0))
        error_global(__FILE__, __LINE__, "not enough first-variational states");

    #ifdef _SCALAPACK_
    create_blacs_context();
    #endif
    
    int nrow = mpi_grid().dimension_size(_dim_row_);
    int ncol = mpi_grid().dimension_size(_dim_col_);
    
    int irow = mpi_grid().coordinate(_dim_row_);
    int icol = mpi_grid().coordinate(_dim_col_);

    /* create standard eigen-value solver */
    switch (std_evp_solver_type_)
    {
        case ev_lapack:
        {
            std_evp_solver_ = new standard_evp_lapack();
            break;
        }
        case ev_scalapack:
        {
            std_evp_solver_ = new standard_evp_scalapack(nrow, ncol, blacs_context_); 
            break;
        }
        case ev_plasma:
        {
            std_evp_solver_ = new standard_evp_plasma();
            break;
        }
        default:
        {
            error_local(__FILE__, __LINE__, "wrong standard eigen-value solver");
        }
    }
    
    /* create generalized eign-value solver */
    switch (gen_evp_solver_type_)
    {
        case ev_lapack:
        {
            gen_evp_solver_ = new generalized_evp_lapack(1e-15);
            break;
        }
        case ev_scalapack:
        {
            gen_evp_solver_ = new generalized_evp_scalapack(nrow, ncol, blacs_context_, 1e-15);
            break;
        }
        case ev_elpa1:
        {
            gen_evp_solver_ = new generalized_evp_elpa1(nrow, irow, ncol, icol, blacs_context_, 
                                                        mpi_grid().communicator(1 << _dim_row_),
                                                        mpi_grid().communicator(1 << _dim_col_));
            break;
        }
        case ev_elpa2:
        {
            gen_evp_solver_ = new generalized_evp_elpa2(nrow, irow, ncol, icol, blacs_context_, 
                                                        mpi_grid().communicator(1 << _dim_row_),
                                                        mpi_grid().communicator(1 << _dim_col_),
                                                        mpi_grid().communicator(1 << _dim_col_ | 1 << _dim_row_));
            break;
        }
        case ev_magma:
        {
            gen_evp_solver_ = new generalized_evp_magma();
            break;
        }
        case ev_rs_gpu:
        {
            gen_evp_solver_ = new generalized_evp_rs_gpu(nrow, irow, ncol, icol, blacs_context_);
            break;
        }
        case ev_rs_cpu:
        {
            gen_evp_solver_ = new generalized_evp_rs_cpu(nrow, irow, ncol, icol, blacs_context_);
            break;
        }
        default:
        {
            error_local(__FILE__, __LINE__, "wrong generalized eigen-value solver");
        }
    }

    if (std_evp_solver_->parallel() != gen_evp_solver_->parallel())
        error_global(__FILE__, __LINE__, "both eigen-value solvers must be serial or parallel");

    /* total number of bands */
    num_bands_ = num_fv_states_ * num_spins_;

    /* distribue first-variational states along columns */
    spl_fv_states_ = splindex<block_cyclic>(num_fv_states(), ncol, icol, cyclic_block_size_);

    // distribue spinor wave-functions along columns
    spl_spinor_wf_ = splindex<block_cyclic>(num_bands(), ncol, icol, cyclic_block_size_);
    
    // additionally split along rows 
    sub_spl_spinor_wf_ = splindex<block>(spl_spinor_wf_.local_size(), nrow, irow);
    
    sub_spl_fv_states_ = splindex<block>(spl_fv_states().local_size(), nrow, irow);

    //== if (verbosity_level >= 3 && Platform::mpi_rank() == 0 && nrow * ncol > 1)
    //== {
    //==     printf("\n");
    //==     printf("table of column distribution of first-variational states\n");
    //==     printf("(columns of the table correspond to column MPI ranks)\n");
    //==     for (int i0 = 0; i0 < spl_fv_states_col_.local_size(0); i0++)
    //==     {
    //==         for (int i1 = 0; i1 < ncol; i1++) printf("%6i", spl_fv_states_col_.global_index(i0, i1));
    //==         printf("\n");
    //==     }
    //==     
    //==     printf("\n");
    //==     printf("table of row distribution of first-variational states\n");
    //==     printf("(columns of the table correspond to row MPI ranks)\n");
    //==     for (int i0 = 0; i0 < spl_fv_states_row_.local_size(0); i0++)
    //==     {
    //==         for (int i1 = 0; i1 < nrow; i1++) printf("%6i", spl_fv_states_row_.global_index(i0, i1));
    //==         printf("\n");
    //==     }

    //==     printf("\n");
    //==     printf("First-variational states index -> (local index, rank) for column distribution\n");
    //==     for (int i = 0; i < num_fv_states(); i++)
    //==     {
    //==         printf("%6i -> (%6i %6i)\n", i, spl_fv_states_col_.location(_splindex_offs_, i), 
    //==                                         spl_fv_states_col_.location(_splindex_rank_, i));
    //==     }
    //==     
    //==     printf("\n");
    //==     printf("First-variational states index -> (local index, rank) for row distribution\n");
    //==     for (int i = 0; i < num_fv_states(); i++)
    //==     {
    //==         printf("%6i -> (%6i %6i)\n", i, spl_fv_states_row_.location(_splindex_offs_, i), 
    //==                                         spl_fv_states_row_.location(_splindex_rank_, i));
    //==     }
    //==     
    //==     printf("\n");
    //==     printf("table of column distribution of spinor wave functions\n");
    //==     printf("(columns of the table correspond to MPI ranks)\n");
    //==     for (int i0 = 0; i0 < spl_spinor_wf_col_.local_size(0); i0++)
    //==     {
    //==         for (int i1 = 0; i1 < ncol; i1++) printf("%6i", spl_spinor_wf_col_.global_index(i0, i1));
    //==         printf("\n");
    //==     }
    //== }
    
    if (Platform::mpi_rank() == 0 && verbosity_level >= 1) print_info();
    initialized_ = true;
}

void Global::clear()
{
    if (initialized_)
    {
        #ifdef _SCALAPACK_
        linalg<scalapack>::gridexit(blacs_context_);
        linalg<scalapack>::free_blacs_handler(blacs_handler_);
        #endif
        unit_cell_->clear();
        delete reciprocal_lattice_;
        delete step_function_;
        delete std_evp_solver_;
        delete gen_evp_solver_;
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

    std::string evsn[] = {"standard eigen-value solver: ", "generalized eigen-value solver: "};
    ev_solver_t evst[] = {std_evp_solver_->type(), gen_evp_solver_->type()};
    for (int i = 0; i < 2; i++)
    {
        printf("\n");
        printf("%s", evsn[i].c_str());
        switch (evst[i])
        {
            case ev_lapack:
            {
                printf("LAPACK\n");
                break;
            }
            #ifdef _SCALAPACK_
            case ev_scalapack:
            {
                printf("ScaLAPACK, block size %i\n", linalg<scalapack>::cyclic_block_size());
                break;
            }
            case ev_elpa1:
            {
                printf("ELPA1, block size %i\n", linalg<scalapack>::cyclic_block_size());
                break;
            }
            case ev_elpa2:
            {
                printf("ELPA2, block size %i\n", linalg<scalapack>::cyclic_block_size());
                break;
            }
            case ev_rs_gpu:
            {
                printf("RS_gpu\n");
                break;
            }
            case ev_rs_cpu:
            {
                printf("RS_cpu\n");
                break;
            }
            #endif
            case ev_magma:
            {
                printf("MAGMA\n");
                break;
            }
            case ev_plasma:
            {
                printf("PLASMA\n");
                break;
            }
            default:
            {
                error_local(__FILE__, __LINE__, "wrong eigen-value solver");
            }
        }
    }

    printf("\n");
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
    
    printf("\n");
    printf("XC functionals : \n");
    for (int i = 0; i < (int)xc_functionals_input_section_.xc_functional_names_.size(); i++)
    {
        std::string xc_label = xc_functionals_input_section_.xc_functional_names_[i];
        XC_functional xc(xc_label, num_spins());
        printf("\n");
        printf("%s\n", xc_label.c_str());
        printf("%s\n", xc.name().c_str());
        printf("%s\n", xc.refs().c_str());
    }
}

void Global::write_json_output()
{
    auto ts = Timer::collect_timer_stats();
    if (Platform::mpi_rank() == 0)
    {
        std::string fname = std::string("output_") + start_time("%Y%m%d%H%M%S") + std::string(".json");
        JSON_write jw(fname);
        
        jw.single("git_hash", git_hash);
        jw.single("build_date", build_date);
        jw.single("num_ranks", Platform::num_mpi_ranks());
        jw.single("max_num_threads", Platform::max_num_threads());
        jw.single("num_fft_threads", Platform::num_fft_threads());
        jw.single("cyclic_block_size", cyclic_block_size_);
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
        
        jw.single("timers", ts);
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
    step_function_->update();
}

#ifdef _SCALAPACK_
void Global::create_blacs_context()
{
    int nrow = mpi_grid().dimension_size(_dim_row_);
    int ncol = mpi_grid().dimension_size(_dim_col_);
    
    int irow = mpi_grid().coordinate(_dim_row_);
    int icol = mpi_grid().coordinate(_dim_col_);

    int rc = (1 << _dim_row_) | (1 << _dim_col_);

    /* create handler first */
    blacs_handler_ = linalg<scalapack>::create_blacs_handler(mpi_grid().communicator(rc));

    mdarray<int, 2> map_ranks(nrow, ncol);
    for (int i = 0; i < nrow; i++)
    {
        for (int j = 0; j < ncol; j++)
        {
            std::vector<int> xy(2);
            xy[0] = j;
            xy[1] = i;
            map_ranks(i, j) = mpi_grid().cart_rank(mpi_grid().communicator(rc), xy);
        }
    }

    /* create context */
    blacs_context_ = blacs_handler_;
    linalg<scalapack>::gridmap(&blacs_context_, map_ranks.ptr(), map_ranks.ld(), nrow, ncol);

    /* check the grid */
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
}
#endif

}

