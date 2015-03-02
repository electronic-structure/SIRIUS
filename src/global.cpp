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
#include "real_space_prj.h"

namespace sirius {

void Global::parse_input()
{
    mpi_grid_dims_ = iip_.common_input_section_.mpi_grid_dims_;
    num_fv_states_ = iip_.common_input_section_.num_fv_states_;
    smearing_width_ = iip_.common_input_section_.smearing_width_;
    
    std::string evsn[] = {iip_.common_input_section_.std_evp_solver_type_, iip_.common_input_section_.gen_evp_solver_type_};
    ev_solver_t* evst[] = {&std_evp_solver_type_, &gen_evp_solver_type_};

    for (int i = 0; i < 2; i++)
    {
        std::string name = evsn[i];
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
            TERMINATE("wrong eigen value solver");
        }
    }

    std::string pu = iip_.common_input_section_.processing_unit_;
    if (pu == "cpu" || pu == "CPU")
    {
        processing_unit_ = CPU;
    }
    else if (pu == "gpu" || pu == "GPU")
    {
        processing_unit_ = GPU;
    }
    else
    {
        TERMINATE("wrong processing unit");
    }

    std::string esm = iip_.common_input_section_.electronic_structure_method_;
    if (esm == "full_potential_lapwlo")
    {
        esm_type_ = full_potential_lapwlo;
    }
    else if (esm == "full_potential_pwlo")
    {
        esm_type_ = full_potential_pwlo;
    }
    else if (esm == "ultrasoft_pseudopotential")
    {
        esm_type_ = ultrasoft_pseudopotential;
    } 
    else if (esm == "norm_conserving_pseudopotential")
    {
        esm_type_ = norm_conserving_pseudopotential;
    }
    else
    {
        TERMINATE("wrong type of electronic structure method");
    }
}

void Global::read_unit_cell_input()
{
    auto unit_cell_input_section = iip_.unit_cell_input_section_;

    for (int iat = 0; iat < (int)unit_cell_input_section.labels_.size(); iat++)
    {
        std::string label = unit_cell_input_section.labels_[iat];
        std::string fname = unit_cell_input_section.atom_files_[label];
        unit_cell()->add_atom_type(label, fname, esm_type());
        for (int ia = 0; ia < (int)unit_cell_input_section.coordinates_[iat].size(); ia++)
        {
            std::vector<double> v = unit_cell_input_section.coordinates_[iat][ia];
            unit_cell()->add_atom(label, &v[0], &v[3]);
        }
    }

    unit_cell()->set_lattice_vectors(unit_cell_input_section.lattice_vectors_[0], 
                                     unit_cell_input_section.lattice_vectors_[1], 
                                     unit_cell_input_section.lattice_vectors_[2]);
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
        case norm_conserving_pseudopotential:
        {
            lmax_apw_ = lmax_rho_ = lmax_pot_ = -1;
            break;
        }
    }

    /* initialize variables, related to the unit cell */
    unit_cell_->initialize(lmax_apw(), lmax_pot(), num_mag_dims());
    
    /* create FFT interface */
    fft_ = new FFT3D<CPU>(Utils::find_translation_limits(pw_cutoff_, unit_cell_->reciprocal_lattice_vectors()),
                          iip_.common_input_section_.num_fft_threads_,
                          iip_.common_input_section_.num_fft_workers_);
    
    fft_->init_gvec(pw_cutoff_, unit_cell_->reciprocal_lattice_vectors());

    #ifdef _GPU_
    fft_gpu_ = new FFT3D<GPU>(fft_->grid_size(), 2);
    #endif
    
    if (esm_type_ == ultrasoft_pseudopotential || esm_type_ == norm_conserving_pseudopotential)
    {
        /* create FFT interface for coarse grid */
        fft_coarse_ = new FFT3D<CPU>(Utils::find_translation_limits(gk_cutoff_ * 2, unit_cell_->reciprocal_lattice_vectors()),
                                     iip_.common_input_section_.num_fft_threads_,
                                     iip_.common_input_section_.num_fft_workers_);
        
        fft_coarse_->init_gvec(gk_cutoff_ * 2, unit_cell_->reciprocal_lattice_vectors());

        #ifdef _GPU_
        fft_gpu_coarse_ = new FFT3D<GPU>(fft_coarse_->grid_size(), 2);
        #endif
    }

    unit_cell_->symmetry()->check_gvec_symmetry(fft_);

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
            STOP();
        }
        case ultrasoft_pseudopotential:
        case norm_conserving_pseudopotential:
        {
            lmax = 2 * unit_cell_->lmax_beta();
            break;
        }
    }
    reciprocal_lattice_ = new Reciprocal_lattice(unit_cell_, esm_type(), fft_, lmax, comm_);

    if (unit_cell_->full_potential()) step_function_ = new Step_function(reciprocal_lattice_, fft_, comm_);

    if (iip_.iterative_solver_input_section_.real_space_prj_) real_space_prj_ = new Real_space_prj(unit_cell_, fft_coarse_, comm_);

    /* check MPI grid dimensions and set a default grid if needed */
    if (!mpi_grid_dims_.size()) 
    {
        mpi_grid_dims_ = std::vector<int>(1);
        mpi_grid_dims_[0] = comm_.size();
    }

    /* setup MPI grid */
    mpi_grid_ = MPI_grid(mpi_grid_dims_, comm_);

    /* take 20% of empty non-magnetic states */
    if (num_fv_states_ < 0) 
    {
        num_fv_states_ = int(1e-8 + unit_cell_->num_valence_electrons() / 2.0) +
                         std::max(10, int(0.1 * unit_cell_->num_valence_electrons()));
    }

    if (num_fv_states_ < int(unit_cell_->num_valence_electrons() / 2.0))
        error_global(__FILE__, __LINE__, "not enough first-variational states");

    /* total number of bands */
    num_bands_ = num_fv_states_ * num_spins_;

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
    
    if (comm_.rank() == 0 && verbosity_level >= 1) print_info();
    initialized_ = true;
}

void Global::clear()
{
    if (initialized_)
    {
        unit_cell_->clear();
        delete reciprocal_lattice_;
        delete step_function_;
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
    printf("number of MPI ranks           : %i\n", comm_.size());
    printf("MPI grid                      :");
    for (int i = 0; i < mpi_grid_.num_dimensions(); i++) printf(" %i", mpi_grid_.size(1 << i));
    printf("\n");
    printf("maximum number of OMP threads   : %i\n", Platform::max_num_threads()); 
    printf("number of OMP threads for FFT   : %i\n", iip_.common_input_section_.num_fft_threads_); 
    printf("number of pthreads for each FFT : %i\n", iip_.common_input_section_.num_fft_workers_); 
    printf("cyclic block size               : %i\n", iip_.common_input_section_.cyclic_block_size_);

    unit_cell_->print_info();

    printf("\n");
    printf("plane wave cutoff : %f\n", pw_cutoff_);
    printf("number of G-vectors within the cutoff : %i\n", fft_->num_gvec());
    printf("number of G-shells : %i\n", fft_->num_gvec_shells_inner());
    printf("FFT grid size : %i %i %i   total : %i\n", fft_->size(0), fft_->size(1), fft_->size(2), fft_->size());
    printf("FFT grid limits : %i %i   %i %i   %i %i\n", fft_->grid_limits(0).first, fft_->grid_limits(0).second,
                                                        fft_->grid_limits(1).first, fft_->grid_limits(1).second,
                                                        fft_->grid_limits(2).first, fft_->grid_limits(2).second);
    
    if (esm_type_ == ultrasoft_pseudopotential || esm_type_ == norm_conserving_pseudopotential)
    {
        printf("number of G-vectors on the coarse grid within the cutoff : %i\n", fft_coarse_->num_gvec());
        printf("FFT coarse grid size : %i %i %i   total : %i\n", fft_coarse_->size(0), fft_coarse_->size(1), fft_coarse_->size(2), fft_coarse_->size());
        printf("FFT coarse grid limits : %i %i   %i %i   %i %i\n", fft_coarse_->grid_limits(0).first, fft_coarse_->grid_limits(0).second,
                                                                   fft_coarse_->grid_limits(1).first, fft_coarse_->grid_limits(1).second,
                                                                   fft_coarse_->grid_limits(2).first, fft_coarse_->grid_limits(2).second);
    }

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

    //== std::string evsn[] = {"standard eigen-value solver: ", "generalized eigen-value solver: "};
    //== ev_solver_t evst[] = {std_evp_solver_->type(), gen_evp_solver_->type()};
    //== for (int i = 0; i < 2; i++)
    //== {
    //==     printf("\n");
    //==     printf("%s", evsn[i].c_str());
    //==     switch (evst[i])
    //==     {
    //==         case ev_lapack:
    //==         {
    //==             printf("LAPACK\n");
    //==             break;
    //==         }
    //==         #ifdef _SCALAPACK_
    //==         case ev_scalapack:
    //==         {
    //==             printf("ScaLAPACK, block size %i\n", linalg<scalapack>::cyclic_block_size());
    //==             break;
    //==         }
    //==         case ev_elpa1:
    //==         {
    //==             printf("ELPA1, block size %i\n", linalg<scalapack>::cyclic_block_size());
    //==             break;
    //==         }
    //==         case ev_elpa2:
    //==         {
    //==             printf("ELPA2, block size %i\n", linalg<scalapack>::cyclic_block_size());
    //==             break;
    //==         }
    //==         case ev_rs_gpu:
    //==         {
    //==             printf("RS_gpu\n");
    //==             break;
    //==         }
    //==         case ev_rs_cpu:
    //==         {
    //==             printf("RS_cpu\n");
    //==             break;
    //==         }
    //==         #endif
    //==         case ev_magma:
    //==         {
    //==             printf("MAGMA\n");
    //==             break;
    //==         }
    //==         case ev_plasma:
    //==         {
    //==             printf("PLASMA\n");
    //==             break;
    //==         }
    //==         default:
    //==         {
    //==             error_local(__FILE__, __LINE__, "wrong eigen-value solver");
    //==         }
    //==     }
    //== }

    printf("\n");
    printf("processing unit : ");
    switch (processing_unit())
    {
        case CPU:
        {
            printf("CPU\n");
            break;
        }
        case GPU:
        {
            printf("GPU\n");
            break;
        }
    }
    
    printf("\n");
    printf("XC functionals : \n");
    for (int i = 0; i < (int)iip_.xc_functionals_input_section_.xc_functional_names_.size(); i++)
    {
        std::string xc_label = iip_.xc_functionals_input_section_.xc_functional_names_[i];
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
    if (comm_.rank() == 0)
    {
        std::string fname = std::string("output_") + start_time("%Y%m%d%H%M%S") + std::string(".json");
        JSON_write jw(fname);
        
        jw.single("git_hash", git_hash);
        jw.single("build_date", build_date);
        jw.single("num_ranks", comm_.size());
        jw.single("max_num_threads", Platform::max_num_threads());
        jw.single("cyclic_block_size", iip_.common_input_section_.cyclic_block_size_);
        jw.single("mpi_grid", mpi_grid_dims_);
        std::vector<int> fftgrid(3);
        for (int i = 0; i < 3; i++) fftgrid[i] = fft_->size(i);
        jw.single("fft_grid", fftgrid);
        jw.single("chemical_formula", unit_cell()->chemical_formula());
        jw.single("num_atoms", unit_cell()->num_atoms());
        jw.single("num_fv_states", num_fv_states());
        jw.single("num_bands", num_bands());
        jw.single("aw_cutoff", aw_cutoff());
        jw.single("pw_cutoff", pw_cutoff());
        jw.single("omega", unit_cell()->omega());
        
        jw.single("timers", ts);
    }
}

void Global::create_storage_file()
{
    if (comm_.rank() == 0)
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
    comm_.barrier();
}

void Global::update()
{
    unit_cell_->update();
    reciprocal_lattice_->update();
    step_function_->update();
}

}
