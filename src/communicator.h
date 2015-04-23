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
#include "typedefs.h"

#define CALL_MPI(func__, args__)                                                    \
{                                                                                   \
    if (func__ args__ != MPI_SUCCESS)                                               \
    {                                                                               \
        printf("error in %s at line %i of file %s\n", #func__, __LINE__, __FILE__); \
        MPI_Abort(MPI_COMM_WORLD, -1);                                              \
    }                                                                               \
}

/// MPI communicator wrapper.
class Communicator
{
    private:

        MPI_Comm mpi_comm_;

        inline void set_comm(MPI_Comm const& mpi_comm_orig__)
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
                mpi_comm_ = MPI_COMM_NULL;
            }
        }

        inline MPI_Comm mpi_comm() const
        {
            return mpi_comm_;
        }

        inline int rank() const
        {
            assert(mpi_comm_ != MPI_COMM_NULL);

            int r;
            CALL_MPI(MPI_Comm_rank, (mpi_comm_, &r));
            return r;
        }

        inline int cart_rank(std::vector<int> const& coords__) const
        {
            if (mpi_comm_ == MPI_COMM_SELF) return 0;
            
            int r;
            CALL_MPI(MPI_Cart_rank, (mpi_comm_, &coords__[0], &r));
            return r;
        }

        inline int size() const
        {
            assert(mpi_comm_ != MPI_COMM_NULL);

            int s;
            CALL_MPI(MPI_Comm_size, (mpi_comm_, &s));
            return s;
        }

        inline void barrier() const
        {
            assert(mpi_comm_ != MPI_COMM_NULL);
            CALL_MPI(MPI_Barrier, (mpi_comm_));
        }

        template <typename T>
        inline void reduce(T* buffer__, int count__, int root__) const
        {
            if (root__ == rank())
            {
                CALL_MPI(MPI_Reduce, (MPI_IN_PLACE, buffer__, count__, type_wrapper<T>::mpi_type_id(), MPI_SUM, root__, mpi_comm_));
            }
            else
            {
                CALL_MPI(MPI_Reduce, (buffer__, NULL, count__, type_wrapper<T>::mpi_type_id(), MPI_SUM, root__, mpi_comm_));
            }
        }

        template <typename T>
        void reduce(T const* sendbuf__, T* recvbuf__, int count__, int root__) const
        {
            CALL_MPI(MPI_Reduce, (sendbuf__, recvbuf__, count__, type_wrapper<T>::mpi_type_id(), MPI_SUM, root__, mpi_comm_));
        }


        /// Perform the in-place (the output buffer is used as the input buffer) all-to-all reduction.
        template<typename T, mpi_op_t mpi_op__ = op_sum>
        inline void allreduce(T* buffer__, int count__) const
        {
            MPI_Op op;
            switch (mpi_op__)
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
                    printf("wrong operation\n");
                    MPI_Abort(MPI_COMM_WORLD, -2);
                }
            }

            CALL_MPI(MPI_Allreduce, (MPI_IN_PLACE, buffer__, count__, type_wrapper<T>::mpi_type_id(), op, mpi_comm_));
        }

        /// Perform the in-place (the output buffer is used as the input buffer) all-to-all reduction.
        template <typename T, mpi_op_t op__ = op_sum>
        inline void allreduce(std::vector<T>& buffer__) const
        {
            allreduce<T, op__>(&buffer__[0], (int)buffer__.size());
        }
        
        /// Perform buffer broadcast.
        template <typename T>
        inline void bcast(T* buffer__, int count__, int root__) const
        {
            CALL_MPI(MPI_Bcast, (buffer__, count__, type_wrapper<T>::mpi_type_id(), root__, mpi_comm_));
        }

        template<typename T>
        void allgather(T* buffer__, int const* recvcounts__, int const* displs__) const
        {
            CALL_MPI(MPI_Allgatherv, (MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, buffer__, recvcounts__, displs__,
                     type_wrapper<T>::mpi_type_id(), mpi_comm_));
        }

        template<typename T>
        void allgather(T* buffer__, int offset__, int count__) const
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
            allgather(buffer__, &counts[0], &offsets[0]);
        }

        template <typename T>
        void allgather(T const* sendbuf__, T* recvbuf__, int offset__, int count__) const
        {
            std::vector<int> counts(size());
            counts[rank()] = count__;
            CALL_MPI(MPI_Allgather, (MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &counts[0], 1, type_wrapper<int>::mpi_type_id(), mpi_comm_));
            
            std::vector<int> offsets(size());
            offsets[rank()] = offset__;
            CALL_MPI(MPI_Allgather, (MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &offsets[0], 1, type_wrapper<int>::mpi_type_id(), mpi_comm_));
        
            CALL_MPI(MPI_Allgatherv, (sendbuf__, count__, type_wrapper<T>::mpi_type_id(), recvbuf__, &counts[0], &offsets[0],
                           type_wrapper<T>::mpi_type_id(), mpi_comm_));
        }

        template <typename T>
        void allgather(T* const sendbuf__, int sendcount__, T* recvbuf__, int const* recvcounts__, int const* displs__) const
        {
            CALL_MPI(MPI_Allgatherv, (sendbuf__, sendcount__, type_wrapper<T>::mpi_type_id(), 
                                      recvbuf__, recvcounts__, displs__, type_wrapper<T>::mpi_type_id(), mpi_comm_));
        }

        template <typename T>
        void isend(T const* buffer__, int count__, int dest__, int tag__) const
        {
            MPI_Request request;
        
            CALL_MPI(MPI_Isend, (buffer__, count__, type_wrapper<T>::mpi_type_id(), dest__, tag__, mpi_comm_, &request));
        }
        
        template <typename T>
        void recv(T* buffer__, int count__, int source__, int tag__) const
        {
            CALL_MPI(MPI_Recv, (buffer__, count__, type_wrapper<T>::mpi_type_id(), source__, tag__, mpi_comm_, MPI_STATUS_IGNORE));
        }

        template <typename T>
        void gather(T const* sendbuf__, T* recvbuf__, int const* recvcounts__, int const* displs__, int root__) const
        {
            int sendcount = recvcounts__[rank()];
         
            CALL_MPI(MPI_Gatherv, (sendbuf__, sendcount, type_wrapper<T>::mpi_type_id(), recvbuf__, recvcounts__, displs__, 
                                   type_wrapper<T>::mpi_type_id(), root__, mpi_comm_));
        }
        
        template <typename T>
        void scatter(T const* sendbuf__, T* recvbuf__, int const* sendcounts__, int const* displs__, int root__) const
        {
            int recvcount = sendcounts__[rank()];
            CALL_MPI(MPI_Scatterv, (sendbuf__, sendcounts__, displs__, type_wrapper<T>::mpi_type_id(), recvbuf__, recvcount,
                                    type_wrapper<T>::mpi_type_id(), root__, mpi_comm_));
        }

        template <typename T>
        void alltoall(T const* sendbuf__, int const* sendcounts__, int const* sdispls__, 
                      T* recvbuf__, int const* recvcounts__, int const* rdispls__) const
        {
            CALL_MPI(MPI_Alltoallv, (sendbuf__, sendcounts__, sdispls__, type_wrapper<T>::mpi_type_id(),
                                     recvbuf__, recvcounts__, rdispls__, type_wrapper<T>::mpi_type_id(), mpi_comm_));
        }
};

#endif // __COMMUNICATOR_H__
