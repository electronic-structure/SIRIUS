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

namespace sirius
{

/// Parallel standard output.
/** Proveides an ordered standard output from multiple MPI ranks. */
class pstdout
{
    private:
        
        std::vector<char> buffer_;

        int offset_;

    public:

        pstdout()
        {
            buffer_.resize(8129);
            offset_ = 0;
        }

        void printf(const char* fmt, ...)
        {
            std::vector<char> str(1024); // assume that one printf will not output more than this

            std::va_list arg;
            va_start(arg, fmt);
            int n = vsnprintf(&str[0], str.size(), fmt, arg);
            va_end(arg);

            n = std::min(n, (int)str.size());
            
            if ((int)buffer_.size() - offset_ < n) buffer_.resize(buffer_.size() + str.size());
            memcpy(&buffer_[offset_], &str[0], n);
            offset_ += n;
        }

        void flush(int rank)
        {
            mdarray<int, 2> offsets(Platform::num_mpi_ranks(), 2);
            offsets.zero();
            Platform::allgather(&offset_, &offsets(0, 0), Platform::mpi_rank(), 1); 
            
            for (int i = 1; i < Platform::num_mpi_ranks(); i++) offsets(i, 1) = offsets(i - 1, 1) + offsets(i - 1, 0);
            
            // total size of the output buffer
            int sz = 0;
            for (int i = 0; i < Platform::num_mpi_ranks(); i++) sz += offsets(i, 0);

            std::vector<char> outb(sz + 1);
            Platform::allgather(&buffer_[0], &outb[0], offsets(Platform::mpi_rank(), 1), offset_);
            outb[sz] = 0;

            if (Platform::mpi_rank() == rank) std::printf("%s", &outb[0]);

            offset_ = 0;
        }
};

/// Input / output interface.
class sirius_io
{
    public:

        static void hdf5_write_matrix(const std::string& fname, mdarray<double_complex, 2>& matrix)
        {
            static int icount = 0;

            icount++;
            std::stringstream s;
            s << icount;
            std::string full_name = s.str() + "_" + fname;
            
            HDF5_tree fout(full_name, true);
            int size0 = (int)matrix.size(0);
            int size1 = (int)matrix.size(1);
            fout.write("nrow", &size0); 
            fout.write("ncol", &size1);
            fout.write("matrix", matrix);
        }


};

}

#endif
