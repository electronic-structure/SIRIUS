/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file pstdout.hpp
 *
 *  \brief Contains implementation of the parallel standard output.
 */

#ifndef __PSTDOUT_HPP__
#define __PSTDOUT_HPP__

#include "communicator.hpp"

namespace sirius {

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

    std::string
    flush(int root__)
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

} // namespace mpi

} // namespace sirius

#endif
