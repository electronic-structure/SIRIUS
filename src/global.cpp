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
//#include "simulation.h"

namespace sirius {

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

}
