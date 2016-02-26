// Copyright (c) 2013-2016 Anton Kozhevnikov, Thomas Schulthess
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
#include <cassert>
#include <vector>
#include <complex>

#define CALL_MPI(func__, args__)                                                    \
{                                                                                   \
    if (func__ args__ != MPI_SUCCESS)                                               \
    {                                                                               \
        printf("error in %s at line %i of file %s\n", #func__, __LINE__, __FILE__); \
        MPI_Abort(MPI_COMM_WORLD, -1);                                              \
    }                                                                               \
}

enum mpi_op_t {op_sum, op_max};

template <typename T> 
class mpi_type_wrapper;

template<> 
class mpi_type_wrapper<double>
{
    public:
        static MPI_Datatype kind() { return MPI_DOUBLE; }
};

template<> 
class mpi_type_wrapper<long double>
{
    public:
        static MPI_Datatype kind() { return MPI_LONG_DOUBLE; }
};

template<> 
class mpi_type_wrapper< std::complex<double> >
{
    public:
        static MPI_Datatype kind() { return MPI_COMPLEX16; }
};

template<> 
class mpi_type_wrapper<int>
{
    public:
        static MPI_Datatype kind() { return MPI_INT; }
};

template<> 
class mpi_type_wrapper<int16_t>
{
    public:
        static MPI_Datatype kind() { return MPI_SHORT; }
};

template<> 
class mpi_type_wrapper<char>
{
    public:
        static MPI_Datatype kind() { return MPI_CHAR; }
};

struct alltoall_descriptor
{
    std::vector<int> sendcounts;
    std::vector<int> sdispls;
    std::vector<int> recvcounts;
    std::vector<int> rdispls;
};

/// MPI communicator wrapper.
class Communicator
{
    private:

        MPI_Comm mpi_comm_;

        Communicator(Communicator const& src__) = delete;

    public:
    
        Communicator() : mpi_comm_(MPI_COMM_NULL)
        {
        }

        Communicator(MPI_Comm mpi_comm__) : mpi_comm_(mpi_comm__)
        {
        }

        ~Communicator()
        {
            if (!(mpi_comm_ == MPI_COMM_NULL || mpi_comm_ == MPI_COMM_WORLD || mpi_comm_ == MPI_COMM_SELF))
            {
                CALL_MPI(MPI_Comm_free, (&mpi_comm_));
                mpi_comm_ = MPI_COMM_NULL;
            }
        }

        static void initialize()
        {
            int provided;

            MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided);

            MPI_Query_thread(&provided);
            if (provided < MPI_THREAD_FUNNELED)
            {
                printf("Warning! MPI_THREAD_FUNNELED level of thread support is not provided.\n");
            }
        }

        static void finalize()
        {
            MPI_Finalize();
        }

        inline MPI_Comm& mpi_comm()
        {
            return mpi_comm_;
        }

        inline MPI_Comm const& mpi_comm() const
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

        inline std::pair<int, int> p()
        {
            assert(mpi_comm_ != MPI_COMM_NULL);
            std::pair<int, int> p;
            CALL_MPI(MPI_Comm_size, (mpi_comm_, &p.first));
            CALL_MPI(MPI_Comm_rank, (mpi_comm_, &p.second));
            return p;
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
                CALL_MPI(MPI_Reduce, (MPI_IN_PLACE, buffer__, count__, mpi_type_wrapper<T>::kind(), MPI_SUM, root__, mpi_comm_));
            }
            else
            {
                CALL_MPI(MPI_Reduce, (buffer__, NULL, count__, mpi_type_wrapper<T>::kind(), MPI_SUM, root__, mpi_comm_));
            }
        }

        template <typename T>
        void reduce(T const* sendbuf__, T* recvbuf__, int count__, int root__) const
        {
            CALL_MPI(MPI_Reduce, (sendbuf__, recvbuf__, count__, mpi_type_wrapper<T>::kind(), MPI_SUM, root__, mpi_comm_));
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

            CALL_MPI(MPI_Allreduce, (MPI_IN_PLACE, buffer__, count__, mpi_type_wrapper<T>::kind(), op, mpi_comm_));
        }

        /// Perform the in-place (the output buffer is used as the input buffer) all-to-all reduction.
        template <typename T, mpi_op_t op__ = op_sum>
        inline void allreduce(std::vector<T>& buffer__) const
        {
            allreduce<T, op__>(&buffer__[0], static_cast<int>(buffer__.size()));
        }
        
        /// Perform buffer broadcast.
        template <typename T>
        inline void bcast(T* buffer__, int count__, int root__) const
        {
            CALL_MPI(MPI_Bcast, (buffer__, count__, mpi_type_wrapper<T>::kind(), root__, mpi_comm_));
        }

        template<typename T>
        void allgather(T* buffer__, int const* recvcounts__, int const* displs__) const
        {
            CALL_MPI(MPI_Allgatherv, (MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, buffer__, recvcounts__, displs__,
                     mpi_type_wrapper<T>::kind(), mpi_comm_));
        }

        template <typename T>
        void allgather(T* const sendbuf__, int sendcount__, T* recvbuf__, int const* recvcounts__, int const* displs__) const
        {
            CALL_MPI(MPI_Allgatherv, (sendbuf__, sendcount__, mpi_type_wrapper<T>::kind(), 
                                      recvbuf__, recvcounts__, displs__, mpi_type_wrapper<T>::kind(), mpi_comm_));
        }

        template<typename T>
        void allgather(T* buffer__, int offset__, int count__) const
        {
            std::vector<int> v(size() * 2);
            v[2 * rank()] = count__;
            v[2 * rank() + 1] = offset__;
        
            CALL_MPI(MPI_Allgather, (MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &v[0], 2, mpi_type_wrapper<int>::kind(), mpi_comm_));
        
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
            // TODO: pack in one call
            std::vector<int> counts(size());
            counts[rank()] = count__;
            CALL_MPI(MPI_Allgather, (MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &counts[0], 1, mpi_type_wrapper<int>::kind(), mpi_comm_));
            
            std::vector<int> offsets(size());
            offsets[rank()] = offset__;
            CALL_MPI(MPI_Allgather, (MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &offsets[0], 1, mpi_type_wrapper<int>::kind(), mpi_comm_));

            // Why in hell this doesn't work???
            //allgather((T const*)sendbuf__, count__, (T*)recvbuf__, (int*)&counts[0], (int*)&offsets[0]);
        
            CALL_MPI(MPI_Allgatherv, (sendbuf__, count__, mpi_type_wrapper<T>::kind(), recvbuf__, &counts[0], &offsets[0],
                                      mpi_type_wrapper<T>::kind(), mpi_comm_));
        }

        template <typename T>
        void isend(T const* buffer__, int count__, int dest__, int tag__) const
        {
            MPI_Request request;
        
            CALL_MPI(MPI_Isend, (buffer__, count__, mpi_type_wrapper<T>::kind(), dest__, tag__, mpi_comm_, &request));
        }
        
        template <typename T>
        void recv(T* buffer__, int count__, int source__, int tag__) const
        {
            CALL_MPI(MPI_Recv, (buffer__, count__, mpi_type_wrapper<T>::kind(), source__, tag__, mpi_comm_, MPI_STATUS_IGNORE));
        }

        template <typename T>
        void irecv(T* buffer__, int count__, int source__, int tag__, MPI_Request* request__) const
        {
            CALL_MPI(MPI_Irecv, (buffer__, count__, mpi_type_wrapper<T>::kind(), source__, tag__, mpi_comm_, request__));
        }

        template <typename T>
        void gather(T const* sendbuf__, T* recvbuf__, int const* recvcounts__, int const* displs__, int root__) const
        {
            int sendcount = recvcounts__[rank()];
         
            CALL_MPI(MPI_Gatherv, (sendbuf__, sendcount, mpi_type_wrapper<T>::kind(), recvbuf__, recvcounts__, displs__, 
                                   mpi_type_wrapper<T>::kind(), root__, mpi_comm_));
        }
        
        template <typename T>
        void scatter(T const* sendbuf__, T* recvbuf__, int const* sendcounts__, int const* displs__, int root__) const
        {
            int recvcount = sendcounts__[rank()];
            CALL_MPI(MPI_Scatterv, (sendbuf__, sendcounts__, displs__, mpi_type_wrapper<T>::kind(), recvbuf__, recvcount,
                                    mpi_type_wrapper<T>::kind(), root__, mpi_comm_));
        }

        template <typename T>
        void alltoall(T const* sendbuf__, int sendcounts__, T* recvbuf__, int recvcounts__) const
        {
            CALL_MPI(MPI_Alltoall, (sendbuf__, sendcounts__, mpi_type_wrapper<T>::kind(),
                                    recvbuf__, recvcounts__, mpi_type_wrapper<T>::kind(), mpi_comm_));
        }

        template <typename T>
        void alltoall(T const* sendbuf__, int const* sendcounts__, int const* sdispls__, 
                      T* recvbuf__, int const* recvcounts__, int const* rdispls__) const
        {
            CALL_MPI(MPI_Alltoallv, (sendbuf__, sendcounts__, sdispls__, mpi_type_wrapper<T>::kind(),
                                     recvbuf__, recvcounts__, rdispls__, mpi_type_wrapper<T>::kind(), mpi_comm_));
        }

        //==alltoall_descriptor map_alltoall(std::vector<int> local_sizes_in, std::vector<int> local_sizes_out) const
        //=={
        //==    alltoall_descriptor a2a;
        //==    a2a.sendcounts = std::vector<int>(size(), 0);
        //==    a2a.sdispls    = std::vector<int>(size(), -1);
        //==    a2a.recvcounts = std::vector<int>(size(), 0);
        //==    a2a.rdispls    = std::vector<int>(size(), -1);

        //==    std::vector<int> offs_in(size(), 0);
        //==    std::vector<int> offs_out(size(), 0);

        //==    for (int i = 1; i < size(); i++)
        //==    {
        //==        offs_in[i] = offs_in[i - 1] + local_sizes_in[i - 1];
        //==        offs_out[i] = offs_out[i - 1] + local_sizes_out[i - 1];
        //==    }

        //==    /* loop over sending ranks */
        //==    for (int sr = 0; sr < size(); sr++)
        //==    {
        //==        if (!local_sizes_in[sr]) continue;

        //==        /* beginning of index */
        //==        int i0 = offs_in[sr];
        //==        /* end of index */
        //==        int i1 = offs_in[sr] + local_sizes_in[sr] - 1;

        //==        /* loop over receiving ranks */
        //==        for (int rr = 0; rr < size(); rr++)
        //==        {
        //==            if (!local_sizes_out[rr]) continue;

        //==            int j0 = offs_out[rr];
        //==            int j1 = offs_out[rr] + local_sizes_out[rr] - 1;

        //==            /* rank rr recieves nothing from rank sr*/
        //==            if (j1 < i0 || i1 < j0) continue;

        //==            int s_ofs = std::max(j0 - i0, 0);
        //==            int r_ofs = std::max(i0 - j0, 0);
        //==            int sz = std::min(i1, j1) - std::max(i0, j0) + 1;
        //==            
        //==            if (rank() == sr)
        //==            {
        //==                a2a.sendcounts[rr] = sz;
        //==                a2a.sdispls[rr] = s_ofs;
        //==            }
        //==            if (rank() == rr)
        //==            {
        //==                a2a.recvcounts[sr] = sz;
        //==                a2a.rdispls[sr] = r_ofs;
        //==            }
        //==        }
        //==    }

        //==    int n1 = 0;
        //==    int n2 = 0;
        //==    for (int i = 0; i < size(); i++)
        //==    {
        //==        n1 += a2a.sendcounts[i];
        //==        n2 += a2a.recvcounts[i];
        //==    }
        //==    if (n1 != local_sizes_in[rank()] || n2 != local_sizes_out[rank()])
        //==    {
        //==        printf("wrong sizes");
        //==        MPI_Abort(MPI_COMM_WORLD, -1);
        //==    }

        //==    return a2a;
        //==}
};

inline Communicator const& mpi_comm_self()
{
    static Communicator comm(MPI_COMM_SELF);
    return comm;
}

inline Communicator const& mpi_comm_world()
{
    static Communicator comm(MPI_COMM_WORLD);
    return comm;
}

#endif // __COMMUNICATOR_H__
