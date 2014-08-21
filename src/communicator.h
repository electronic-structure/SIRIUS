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

/** \file communicator.h
 *
 *  \brief Contains declaration and implementation of Communicator class.
 */

#ifndef __COMMUNICATOR_H__
#define __COMMUNICATOR_H__

#include <mpi.h>
#include "error_handling.h"

#define CALL_MPI(func__, args__)      \
{                                     \
    if (func__ args__ != MPI_SUCCESS) \
    {                                 \
        std::stringstream s;          \
        s << "error in " << #func__;  \
        TERMINATE(s);                 \
    }                                 \
}

/// MPI communicator wrapper.
class Communicator
{
    private:

        MPI_Comm mpi_comm_;

        inline void set_comm(MPI_Comm mpi_comm_orig__)
        {
            assert(mpi_comm_orig__ != MPI_COMM_NULL);
            CALL_MPI(MPI_Comm_dup, (mpi_comm_orig__, &mpi_comm_));
        }

    public:
    
        Communicator() : mpi_comm_(MPI_COMM_NULL)
        {
        }

        Communicator(MPI_Comm const& mpi_comm__)
        {
            set_comm(mpi_comm__);
        }

        Communicator(Communicator const& comm__)
        {
            set_comm(comm__.mpi_comm_);
        }

        Communicator& operator=(Communicator const& comm__)
        {
            set_comm(comm__.mpi_comm_);
            return *this;
        }

        ~Communicator()
        {
            if (mpi_comm_ != MPI_COMM_NULL) 
            {
                CALL_MPI(MPI_Comm_free, (&mpi_comm_));
            }
        }

        inline MPI_Comm mpi_comm()
        {
            return mpi_comm_;
        }

        inline int rank()
        {
            assert(mpi_comm_ != MPI_COMM_NULL);

            int r;
            CALL_MPI(MPI_Comm_rank, (mpi_comm_, &r));
            return r;
        }

        inline int size()
        {
            assert(mpi_comm_ != MPI_COMM_NULL);

            int s;
            CALL_MPI(MPI_Comm_size, (mpi_comm_, &s));
            return s;
        }

        inline void barrier()
        {
            assert(mpi_comm_ != MPI_COMM_NULL);
            CALL_MPI(MPI_Barrier, (mpi_comm_));
        }

        template<typename T, mpi_op_t mpi_op = op_sum>
        inline void allreduce(T* buffer__, int count__)
        {
            MPI_Op op;
            switch (mpi_op)
            {
                case op_sum:
                {
                    op = MPI_SUM;
                    break;
                }
                case op_max:
                {
                    op = MPI_MAX;
                    break;
                }
                default:
                {
                    TERMINATE("wrong operation");
                }
            }

            CALL_MPI(MPI_Allreduce, (MPI_IN_PLACE, buffer__, count__, type_wrapper<T>::mpi_type_id(), op, mpi_comm_));
        }

        template <typename T, mpi_op_t op = op_sum>
        inline void allreduce(std::vector<T>& buffer__)
        {
            allreduce<T, op>(&buffer__[0], (int)buffer__.size());
        }

        template <typename T>
        inline void bcast(T* buffer__, int count__, int root__) const
        {
            CALL_MPI(MPI_Bcast, (buffer__, count__, type_wrapper<T>::mpi_type_id(), root__, mpi_comm_));
        }

        template<typename T>
        void allgather(T* buffer__, int offset__, int count__)
        {
        
            std::vector<int> v(size() * 2);
            v[2 * rank()] = count__;
            v[2 * rank() + 1] = offset__;
        
            CALL_MPI(MPI_Allgather, (MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &v[0], 2, type_wrapper<int>::mpi_type_id(), mpi_comm_));
        
            std::vector<int> counts(size());
            std::vector<int> offsets(size());
        
            for (int i = 0; i < size(); i++)
            {
                counts[i] = v[2 * i];
                offsets[i] = v[2 * i + 1];
            }
        
            CALL_MPI(MPI_Allgatherv, (MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, buffer__, &counts[0], &offsets[0],
                           type_wrapper<T>::mpi_type_id(), mpi_comm_));
        }
};

#endif // __COMMUNICATOR_H__
