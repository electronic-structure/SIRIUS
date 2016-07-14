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

/** \file sirius_io.h
 *   
 *  \brief Contains declaration and implementation of sirius::pstdout and sirius::sirius_io classes.
 */

#ifndef __SIRIUS_IO_H__
#define __SIRIUS_IO_H__

#include <cstdarg>
#include "hdf5_tree.h"
#include "mdarray.h"
#include "communicator.h"

namespace sirius
{

/// Input / output interface.
class sirius_io
{
    public:

        static void hdf5_write_matrix(std::string const& fname, mdarray<double_complex, 2>& matrix)
        {
            static int icount = 0;

            icount++;
            std::stringstream s;
            s << icount;
            std::string full_name = s.str() + "_" + fname;
            
            HDF5_tree fout(full_name, true);
            int size0 = (int)matrix.size(0);
            int size1 = (int)matrix.size(1);
            fout.write("nrow", size0); 
            fout.write("ncol", size1);
            fout.write("matrix", matrix);
        }


};

        //== void write_periodic_function()
        //== {
        //==     //== mdarray<double, 3> vloc_3d_map(&vloc_it[0], fft_->size(0), fft_->size(1), fft_->size(2));
        //==     //== int nx = fft_->size(0);
        //==     //== int ny = fft_->size(1);
        //==     //== int nz = fft_->size(2);

        //==     //== auto p = parameters_.unit_cell()->unit_cell_parameters();

        //==     //== FILE* fout = fopen("potential.ted", "w");
        //==     //== fprintf(fout, "%s\n", parameters_.unit_cell()->chemical_formula().c_str());
        //==     //== fprintf(fout, "%16.10f %16.10f %16.10f  %16.10f %16.10f %16.10f\n", p.a, p.b, p.c, p.alpha, p.beta, p.gamma);
        //==     //== fprintf(fout, "%i %i %i\n", nx + 1, ny + 1, nz + 1);
        //==     //== for (int i0 = 0; i0 <= nx; i0++)
        //==     //== {
        //==     //==     for (int i1 = 0; i1 <= ny; i1++)
        //==     //==     {
        //==     //==         for (int i2 = 0; i2 <= nz; i2++)
        //==     //==         {
        //==     //==             fprintf(fout, "%14.8f\n", vloc_3d_map(i0 % nx, i1 % ny, i2 % nz));
        //==     //==         }
        //==     //==     }
        //==     //== }
        //==     //== fclose(fout);
        //== }
}

#endif
