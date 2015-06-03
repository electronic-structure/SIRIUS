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
#include "simulation.h"

namespace sirius {

//==     //if (comm_.rank() == 0 && verbosity_level >= 1) print_info();
//==     initialized_ = true;
//== }
//== 
//== void Global::clear()
//== {
//==     if (initialized_)
//==     {
//==         //unit_cell_->clear();
//==         delete reciprocal_lattice_;
//==         delete step_function_;
//==         initialized_ = false;
//==     }
//== }
//== 
//== 
//== void Global::print_info()
//== {
//==     printf("\n");
//==     printf("SIRIUS version : %2i.%02i\n", major_version, minor_version);
//==     printf("git hash       : %s\n", git_hash);
//==     printf("build date     : %s\n", build_date);
//==     printf("start time     : %s\n", start_time("%c").c_str());
//==     printf("\n");
//==     printf("number of MPI ranks           : %i\n", comm_.size());
//==     printf("MPI grid                      :");
//==     for (int i = 0; i < mpi_grid_.num_dimensions(); i++) printf(" %i", mpi_grid_.size(1 << i));
//==     printf("\n");
//==     printf("maximum number of OMP threads   : %i\n", Platform::max_num_threads()); 
//==     //printf("number of OMP threads for FFT   : %i\n", iip_.common_input_section_.num_fft_threads_); 
//==     //printf("number of pthreads for each FFT : %i\n", iip_.common_input_section_.num_fft_workers_); 
//==     //printf("cyclic block size               : %i\n", iip_.common_input_section_.cyclic_block_size_);
//== 
//==     //unit_cell_->print_info();
//== 
//==     printf("\n");
//==     printf("plane wave cutoff : %f\n", pw_cutoff_);
//==     printf("number of G-vectors within the cutoff : %i\n", fft_->num_gvec());
//==     printf("number of G-shells : %i\n", fft_->num_gvec_shells_inner());
//==     printf("FFT grid size : %i %i %i   total : %i\n", fft_->size(0), fft_->size(1), fft_->size(2), fft_->size());
//==     printf("FFT grid limits : %i %i   %i %i   %i %i\n", fft_->grid_limits(0).first, fft_->grid_limits(0).second,
//==                                                         fft_->grid_limits(1).first, fft_->grid_limits(1).second,
//==                                                         fft_->grid_limits(2).first, fft_->grid_limits(2).second);
//==     
//==     if (esm_type_ == ultrasoft_pseudopotential || esm_type_ == norm_conserving_pseudopotential)
//==     {
//==         printf("number of G-vectors on the coarse grid within the cutoff : %i\n", fft_coarse_->num_gvec());
//==         printf("FFT coarse grid size : %i %i %i   total : %i\n", fft_coarse_->size(0), fft_coarse_->size(1), fft_coarse_->size(2), fft_coarse_->size());
//==         printf("FFT coarse grid limits : %i %i   %i %i   %i %i\n", fft_coarse_->grid_limits(0).first, fft_coarse_->grid_limits(0).second,
//==                                                                    fft_coarse_->grid_limits(1).first, fft_coarse_->grid_limits(1).second,
//==                                                                    fft_coarse_->grid_limits(2).first, fft_coarse_->grid_limits(2).second);
//==     }
//== 
//==     //for (int i = 0; i < unit_cell_->num_atom_types(); i++) unit_cell_->atom_type(i)->print_info();
//== 
//==     printf("\n");
//==     //printf("total number of aw basis functions : %i\n", unit_cell_->mt_aw_basis_size());
//==     //printf("total number of lo basis functions : %i\n", unit_cell_->mt_lo_basis_size());
//==     printf("number of first-variational states : %i\n", num_fv_states());
//==     printf("number of bands                    : %i\n", num_bands());
//==     printf("number of spins                    : %i\n", num_spins());
//==     printf("number of magnetic dimensions      : %i\n", num_mag_dims());
//==     printf("lmax_apw                           : %i\n", lmax_apw());
//==     printf("lmax_pw                            : %i\n", lmax_pw());
//==     printf("lmax_rho                           : %i\n", lmax_rho());
//==     printf("lmax_pot                           : %i\n", lmax_pot());
//==     printf("lmax_beta                          : %i\n", lmax_beta());
//== 
//==     //== std::string evsn[] = {"standard eigen-value solver: ", "generalized eigen-value solver: "};
//==     //== ev_solver_t evst[] = {std_evp_solver_->type(), gen_evp_solver_->type()};
//==     //== for (int i = 0; i < 2; i++)
//==     //== {
//==     //==     printf("\n");
//==     //==     printf("%s", evsn[i].c_str());
//==     //==     switch (evst[i])
//==     //==     {
//==     //==         case ev_lapack:
//==     //==         {
//==     //==             printf("LAPACK\n");
//==     //==             break;
//==     //==         }
//==     //==         #ifdef _SCALAPACK_
//==     //==         case ev_scalapack:
//==     //==         {
//==     //==             printf("ScaLAPACK, block size %i\n", linalg<scalapack>::cyclic_block_size());
//==     //==             break;
//==     //==         }
//==     //==         case ev_elpa1:
//==     //==         {
//==     //==             printf("ELPA1, block size %i\n", linalg<scalapack>::cyclic_block_size());
//==     //==             break;
//==     //==         }
//==     //==         case ev_elpa2:
//==     //==         {
//==     //==             printf("ELPA2, block size %i\n", linalg<scalapack>::cyclic_block_size());
//==     //==             break;
//==     //==         }
//==     //==         case ev_rs_gpu:
//==     //==         {
//==     //==             printf("RS_gpu\n");
//==     //==             break;
//==     //==         }
//==     //==         case ev_rs_cpu:
//==     //==         {
//==     //==             printf("RS_cpu\n");
//==     //==             break;
//==     //==         }
//==     //==         #endif
//==     //==         case ev_magma:
//==     //==         {
//==     //==             printf("MAGMA\n");
//==     //==             break;
//==     //==         }
//==     //==         case ev_plasma:
//==     //==         {
//==     //==             printf("PLASMA\n");
//==     //==             break;
//==     //==         }
//==     //==         default:
//==     //==         {
//==     //==             error_local(__FILE__, __LINE__, "wrong eigen-value solver");
//==     //==         }
//==     //==     }
//==     //== }
//== 
//==     printf("\n");
//==     printf("processing unit : ");
//==     switch (processing_unit())
//==     {
//==         case CPU:
//==         {
//==             printf("CPU\n");
//==             break;
//==         }
//==         case GPU:
//==         {
//==             printf("GPU\n");
//==             break;
//==         }
//==     }
//==     
//==     //printf("\n");
//==     //printf("XC functionals : \n");
//==     //for (int i = 0; i < (int)iip_.xc_functionals_input_section_.xc_functional_names_.size(); i++)
//==     //{
//==     //    std::string xc_label = iip_.xc_functionals_input_section_.xc_functional_names_[i];
//==     //    XC_functional xc(xc_label, num_spins());
//==     //    printf("\n");
//==     //    printf("%s\n", xc_label.c_str());
//==     //    printf("%s\n", xc.name().c_str());
//==     //    printf("%s\n", xc.refs().c_str());
//==     //}
//== }
//== 
//== void Global::write_json_output()
//== {
//==     auto ts = Timer::collect_timer_stats();
//==     if (comm_.rank() == 0)
//==     {
//==         std::string fname = std::string("output_") + start_time("%Y%m%d%H%M%S") + std::string(".json");
//==         JSON_write jw(fname);
//==         
//==         jw.single("git_hash", git_hash);
//==         jw.single("build_date", build_date);
//==         jw.single("num_ranks", comm_.size());
//==         jw.single("max_num_threads", Platform::max_num_threads());
//==         //jw.single("cyclic_block_size", iip_.common_input_section_.cyclic_block_size_);
//==         jw.single("mpi_grid", mpi_grid_dims_);
//==         std::vector<int> fftgrid(3);
//==         for (int i = 0; i < 3; i++) fftgrid[i] = fft_->size(i);
//==         jw.single("fft_grid", fftgrid);
//==         //jw.single("chemical_formula", unit_cell()->chemical_formula());
//==         //jw.single("num_atoms", unit_cell()->num_atoms());
//==         jw.single("num_fv_states", num_fv_states());
//==         jw.single("num_bands", num_bands());
//==         jw.single("aw_cutoff", aw_cutoff());
//==         jw.single("pw_cutoff", pw_cutoff());
//==         //jw.single("omega", unit_cell()->omega());
//==         
//==         jw.single("timers", ts);
//==     }
//== }
//== 
//== void Global::create_storage_file()
//== {
//==     if (comm_.rank() == 0)
//==     {
//==         // create new hdf5 file
//==         HDF5_tree fout(storage_file_name, true);
//==         fout.create_node("parameters");
//==         fout.create_node("effective_potential");
//==         fout.create_node("effective_magnetic_field");
//==         fout.create_node("density");
//==         fout.create_node("magnetization");
//==         
//==         fout["parameters"].write("num_spins", num_spins());
//==         fout["parameters"].write("num_mag_dims", num_mag_dims());
//==         fout["parameters"].write("num_bands", num_bands());
//==     }
//==     comm_.barrier();
//== }

}
