// Copyright (c) 2013-2022 Anton Kozhevnikov, Thomas Schulthess
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

/** \file pstdout.hpp
 *
 *  \brief Contains implementation of the parallel standard output.
 */

#ifndef __PSTDOUT_HPP__
#define __PSTDOUT_HPP__

#include "communicator.hpp"

namespace mpi {

/// Parallel standard output.
/** Proveides an ordered standard output from multiple MPI ranks.
 *  pstdout pout(comm);
 *  pout << "Hello from rank " << comm.rank() << std::end;
 *  // print from root rank (id=0) and flush the internal buffer
 *  std::cout << pout.flush(0);
 */
class pstdout : public std::stringstream
{
  private:
    Communicator const& comm_;

  public:
    pstdout(Communicator const& comm__)
        : comm_(comm__)
    {
    }

    std::string flush(int root__)
    {
        std::stringstream s;

        std::vector<int> counts(comm_.size());
        int count = this->str().length();
        comm_.allgather(&count, counts.data(), 1, comm_.rank());

        int offset{0};
        for (int i = 0; i < comm_.rank(); i++) {
            offset += counts[i];
        }

        int sz = count;
        /* total size of the output buffer */
        comm_.allreduce(&sz, 1);

        if (sz != 0) {
            std::vector<char> outb(sz);
            comm_.allgather(this->str().c_str(), &outb[0], count, offset);
            s.write(outb.data(), sz);
        }
        /* reset the internal string */
        this->str("");
        if (comm_.rank() == root__) {
            return s.str();
        } else {
            return std::string("");
        }
    }
};

}

#endif
