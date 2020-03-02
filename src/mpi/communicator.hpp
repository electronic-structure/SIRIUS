// Copyright (c) 2013-2018 Anton Kozhevnikov, Thomas Schulthess
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

/** \file communicator.hpp
 *
 *  \brief Contains declaration and implementation of sddk::Communicator class.
 */

#ifndef __COMMUNICATOR_HPP__
#define __COMMUNICATOR_HPP__

#include <mpi.h>
#include <cassert>
#include <vector>
#include <complex>
#include <cstdarg>
#include <functional>
#include <memory>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <map>

namespace sddk {

#define CALL_MPI(func__, args__)                                                     \
{                                                                                    \
    if (func__ args__ != MPI_SUCCESS) {                                              \
        std::printf("error in %s at line %i of file %s\n", #func__, __LINE__, __FILE__);  \
        MPI_Abort(MPI_COMM_WORLD, -1);                                               \
    }                                                                                \
}

enum class mpi_op_t
{
    sum,
    max,
    min
};

template <mpi_op_t op>
struct mpi_op_wrapper;

template <>
struct mpi_op_wrapper<mpi_op_t::sum>
{
    static MPI_Op kind()
    {
        return MPI_SUM;
    }
};

template <>
struct mpi_op_wrapper<mpi_op_t::max>
{
    static MPI_Op kind()
    {
        return MPI_MAX;
    }
};

template <>
struct mpi_op_wrapper<mpi_op_t::min>
{
    static MPI_Op kind()
    {
        return MPI_MIN;
    }
};

template <typename T>
struct mpi_type_wrapper;

template <>
struct mpi_type_wrapper<double>
{
    static MPI_Datatype kind()
    {
        return MPI_DOUBLE;
    }
};


    template <>
struct mpi_type_wrapper<long double>
{
    static MPI_Datatype kind()
    {
        return MPI_LONG_DOUBLE;
    }
};

template <>
struct mpi_type_wrapper<std::complex<double>>
{
    static MPI_Datatype kind()
    {
        return MPI_CXX_DOUBLE_COMPLEX;
    }
};

template <>
struct mpi_type_wrapper<int>
{
    static MPI_Datatype kind()
    {
        return MPI_INT;
    }
};

template <>
struct mpi_type_wrapper<int16_t>
{
    static MPI_Datatype kind()
    {
        return MPI_SHORT;
    }
};

template <>
struct mpi_type_wrapper<char>
{
    static MPI_Datatype kind()
    {
        return MPI_CHAR;
    }
};

template <>
struct mpi_type_wrapper<unsigned char>
{
    static MPI_Datatype kind()
    {
        return MPI_UNSIGNED_CHAR;
    }
};

template <>
struct mpi_type_wrapper<unsigned long long>
{
    static MPI_Datatype kind()
    {
        return MPI_UNSIGNED_LONG_LONG;
    }
};

template <>
struct mpi_type_wrapper<unsigned long>
{
    static MPI_Datatype kind()
    {
        return MPI_UNSIGNED_LONG;
    }
};

template <>
struct mpi_type_wrapper<bool>
{
    static MPI_Datatype kind()
    {
        return MPI_CXX_BOOL;
    }
};

template <>
struct mpi_type_wrapper<uint32_t>
{
    static MPI_Datatype kind()
    {
        return MPI_UINT32_T;
    }
};

struct alltoall_descriptor
{
    std::vector<int> sendcounts;
    std::vector<int> sdispls;
    std::vector<int> recvcounts;
    std::vector<int> rdispls;
};

struct block_data_descriptor
{
    int num_ranks{-1};
    std::vector<int> counts;
    std::vector<int> offsets;

    block_data_descriptor()
    {
    }

    block_data_descriptor(int num_ranks__)
        : num_ranks(num_ranks__)
    {
        counts  = std::vector<int>(num_ranks, 0);
        offsets = std::vector<int>(num_ranks, 0);
    }

    void calc_offsets()
    {
        for (int i = 1; i < num_ranks; i++) {
            offsets[i] = offsets[i - 1] + counts[i - 1];
        }
    }

    inline int size() const
    {
        return counts.back() + offsets.back();
    }
};

class Request
{
  private:
    MPI_Request handler_;
  public:
    ~Request()
    {
        //CALL_MPI(MPI_Request_free, (&handler_));
    }
    void wait()
    {
        CALL_MPI(MPI_Wait, (&handler_, MPI_STATUS_IGNORE));
    }

    MPI_Request& handler()
    {
        return handler_;
    }
};

struct mpi_comm_deleter
{
    void operator()(MPI_Comm* comm__) const
    {
        int mpi_finalized_flag;
        MPI_Finalized(&mpi_finalized_flag);
        if (!mpi_finalized_flag) {
            CALL_MPI(MPI_Comm_free, (comm__));
        }
        delete comm__;
    }
};

/// MPI communicator wrapper.
class Communicator
{
  private:
    /// Raw MPI communicator.
    MPI_Comm mpi_comm_raw_{MPI_COMM_NULL};
    /// Smart pointer to allocated MPI communicator.
    std::unique_ptr<MPI_Comm, mpi_comm_deleter> mpi_comm_;
    /* copy is not allowed */
    Communicator(Communicator const& src__) = delete;
    /* assigment is not allowed */
    Communicator operator=(Communicator const& src__) = delete;

  public:
    /// Default constructor.
    Communicator()
    {
    }

    /// Constructor for existing communicator.
    explicit Communicator(MPI_Comm mpi_comm__)
        : mpi_comm_raw_(mpi_comm__)
    {
    }

    /// Move constructor.
    Communicator(Communicator&& src__)
    {
        *this = std::move(src__);
    }

    /// Move assigment operator.
    Communicator& operator=(Communicator&& src__)
    {
        if (this != &src__) {
            this->mpi_comm_     = std::move(src__.mpi_comm_);
            this->mpi_comm_raw_ = src__.mpi_comm_raw_;
        }
        return *this;
    }

    /// MPI initialization.
    static void initialize(int required__)
    {
        int provided;

        MPI_Init_thread(NULL, NULL, required__, &provided);

        MPI_Query_thread(&provided);
        if (provided < required__) {
            std::printf("Warning! Required level of thread support is not provided.\nprovided: %d \nrequired: %d\n", provided, required__);
        }
    }

    /// MPI shut down.
    static void finalize()
    {
        MPI_Finalize();
    }

    static bool is_finalized()
    {
        int mpi_finalized_flag;
        MPI_Finalized(&mpi_finalized_flag);
        return mpi_finalized_flag == true;
    }

    static Communicator const& self()
    {
        static Communicator comm(MPI_COMM_SELF);
        return comm;
    }

    static Communicator const& world()
    {
        static Communicator comm(MPI_COMM_WORLD);
        return comm;
    }

    static Communicator const& null()
    {
        static Communicator comm(MPI_COMM_NULL);
        return comm;
    }

    void abort(int errcode__) const
    {
        CALL_MPI(MPI_Abort, (mpi_comm(), errcode__));
    }

    inline Communicator cart_create(int ndims__, int const* dims__, int const* periods__) const
    {
        Communicator new_comm;
        new_comm.mpi_comm_ = std::unique_ptr<MPI_Comm, mpi_comm_deleter>(new MPI_Comm);
        CALL_MPI(MPI_Cart_create, (mpi_comm(), ndims__, dims__, periods__, 0, new_comm.mpi_comm_.get()));
        new_comm.mpi_comm_raw_ = *new_comm.mpi_comm_;
        return new_comm;
    }

    inline Communicator cart_sub(int const* remain_dims__) const
    {
        Communicator new_comm;
        new_comm.mpi_comm_ = std::unique_ptr<MPI_Comm, mpi_comm_deleter>(new MPI_Comm);
        CALL_MPI(MPI_Cart_sub, (mpi_comm(), remain_dims__, new_comm.mpi_comm_.get()));
        new_comm.mpi_comm_raw_ = *new_comm.mpi_comm_;
        return new_comm;
    }

    inline Communicator split(int color__) const
    {
        Communicator new_comm;
        new_comm.mpi_comm_ = std::unique_ptr<MPI_Comm, mpi_comm_deleter>(new MPI_Comm);
        CALL_MPI(MPI_Comm_split, (mpi_comm(), color__, rank(), new_comm.mpi_comm_.get()));
        new_comm.mpi_comm_raw_ = *new_comm.mpi_comm_;
        return new_comm;
    }

    inline Communicator duplicate() const
    {
        Communicator new_comm;
        new_comm.mpi_comm_ = std::unique_ptr<MPI_Comm, mpi_comm_deleter>(new MPI_Comm);
        CALL_MPI(MPI_Comm_dup, (mpi_comm(), new_comm.mpi_comm_.get()));
        new_comm.mpi_comm_raw_ = *new_comm.mpi_comm_;
        return new_comm;
    }

    /// Mapping between Fortran and SIRIUS MPI communicators.
    static Communicator const& map_fcomm(int fcomm__)
    {
        //static std::map<int, std::unique_ptr<Communicator>> fcomm_map;
        static std::map<int, Communicator> fcomm_map;
        if (!fcomm_map.count(fcomm__)) {
            //fcomm_map[fcomm__] = std::unique_ptr<Communicator>(new Communicator(MPI_Comm_f2c(fcomm__)));
            fcomm_map[fcomm__] = Communicator(MPI_Comm_f2c(fcomm__));
        }

        auto& comm = fcomm_map[fcomm__];
        return comm;
    }

    /// Return raw MPI communicator handler.
    MPI_Comm mpi_comm() const
    {
        return mpi_comm_raw_;
    }

    static int get_tag(int i__, int j__)
    {
        if (i__ > j__) {
            std::swap(i__, j__);
        }
        return (j__ * (j__ + 1) / 2 + i__ + 1) << 6;
    }

    /// Rank of MPI process inside communicator.
    inline int rank() const
    {
        assert(mpi_comm() != MPI_COMM_NULL);

        int r;
        CALL_MPI(MPI_Comm_rank, (mpi_comm(), &r));
        return r;
    }

    /// Size of the communicator (number of ranks).
    inline int size() const
    {
        assert(mpi_comm() != MPI_COMM_NULL);

        int s;
        CALL_MPI(MPI_Comm_size, (mpi_comm(), &s));
        return s;
    }

    /// Rank of MPI process inside communicator with associated Cartesian partitioning.
    inline int cart_rank(std::vector<int> const& coords__) const
    {
        if (mpi_comm() == MPI_COMM_SELF) {
            return 0;
        }

        int r;
        CALL_MPI(MPI_Cart_rank, (mpi_comm(), &coords__[0], &r));
        return r;
    }

    inline void barrier() const
    {
#if defined(__PROFILE_MPI)
        PROFILE("MPI_Barrier");
#endif
        assert(mpi_comm() != MPI_COMM_NULL);
        CALL_MPI(MPI_Barrier, (mpi_comm()));
    }

    template <typename T, mpi_op_t mpi_op__ = mpi_op_t::sum>
    inline void reduce(T* buffer__, int count__, int root__) const
    {
        if (root__ == rank()) {
            CALL_MPI(MPI_Reduce, (MPI_IN_PLACE, buffer__, count__, mpi_type_wrapper<T>::kind(),
                                  mpi_op_wrapper<mpi_op__>::kind(), root__, mpi_comm()));
        } else {
            CALL_MPI(MPI_Reduce, (buffer__, NULL, count__, mpi_type_wrapper<T>::kind(),
                                  mpi_op_wrapper<mpi_op__>::kind(), root__, mpi_comm()));
        }
    }

    template <typename T, mpi_op_t mpi_op__ = mpi_op_t::sum>
    inline void reduce(T* buffer__, int count__, int root__, MPI_Request* req__) const
    {
        if (root__ == rank()) {
            CALL_MPI(MPI_Ireduce, (MPI_IN_PLACE, buffer__, count__, mpi_type_wrapper<T>::kind(),
                                   mpi_op_wrapper<mpi_op__>::kind(), root__, mpi_comm(), req__));
        } else {
            CALL_MPI(MPI_Ireduce, (buffer__, NULL, count__, mpi_type_wrapper<T>::kind(),
                                   mpi_op_wrapper<mpi_op__>::kind(), root__, mpi_comm(), req__));
        }
    }

    template <typename T, mpi_op_t mpi_op__ = mpi_op_t::sum>
    void reduce(T const* sendbuf__, T* recvbuf__, int count__, int root__) const
    {
        CALL_MPI(MPI_Reduce, (sendbuf__, recvbuf__, count__, mpi_type_wrapper<T>::kind(),
                              mpi_op_wrapper<mpi_op__>::kind(), root__, mpi_comm()));
    }

    template <typename T, mpi_op_t mpi_op__ = mpi_op_t::sum>
    void reduce(T const* sendbuf__, T* recvbuf__, int count__, int root__, MPI_Request* req__) const
    {
        CALL_MPI(MPI_Ireduce, (sendbuf__, recvbuf__, count__, mpi_type_wrapper<T>::kind(),
                               mpi_op_wrapper<mpi_op__>::kind(), root__, mpi_comm(), req__));
    }

    /// Perform the in-place (the output buffer is used as the input buffer) all-to-all reduction.
    template <typename T, mpi_op_t mpi_op__ = mpi_op_t::sum>
    inline void allreduce(T* buffer__, int count__) const
    {
        CALL_MPI(MPI_Allreduce, (MPI_IN_PLACE, buffer__, count__, mpi_type_wrapper<T>::kind(),
                                 mpi_op_wrapper<mpi_op__>::kind(), mpi_comm()));
    }

    /// Perform the in-place (the output buffer is used as the input buffer) all-to-all reduction.
    template <typename T, mpi_op_t op__ = mpi_op_t::sum>
    inline void allreduce(std::vector<T>& buffer__) const
    {
        allreduce<T, op__>(buffer__.data(), static_cast<int>(buffer__.size()));
    }

    template <typename T, mpi_op_t mpi_op__ = mpi_op_t::sum>
    inline void iallreduce(T* buffer__, int count__, MPI_Request* req__) const
    {
#if defined(__PROFILE_MPI)
        PROFILE("MPI_Iallreduce");
#endif
        CALL_MPI(MPI_Iallreduce, (MPI_IN_PLACE, buffer__, count__, mpi_type_wrapper<T>::kind(),
                                  mpi_op_wrapper<mpi_op__>::kind(), mpi_comm(), req__));
    }

    /// Perform buffer broadcast.
    template <typename T>
    inline void bcast(T* buffer__, int count__, int root__) const
    {
#if defined(__PROFILE_MPI)
        PROFILE("MPI_Bcast");
#endif
        CALL_MPI(MPI_Bcast, (buffer__, count__, mpi_type_wrapper<T>::kind(), root__, mpi_comm()));
    }

    inline void bcast(std::string& str__, int root__) const
    {
        int sz;
        if (rank() == root__) {
            sz = static_cast<int>(str__.size());
        }
        bcast(&sz, 1, root__);
        char* buf = new char[sz + 1];
        if (rank() == root__) {
            std::copy(str__.c_str(), str__.c_str() + sz + 1, buf);
        }
        bcast(buf, sz + 1, root__);
        str__ = std::string(buf);
        delete[] buf;
    }

    /// In-place MPI_Allgatherv.
    template <typename T>
    void allgather(T* buffer__, int const* recvcounts__, int const* displs__) const
    {
#if defined(__PROFILE_MPI)
        PROFILE("MPI_Allgatherv");
#endif
        CALL_MPI(MPI_Allgatherv, (MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, buffer__, recvcounts__, displs__,
                                  mpi_type_wrapper<T>::kind(), mpi_comm()));
    }

    /// Out-of-place MPI_Allgatherv.
    template <typename T>
    void
    allgather(T* const sendbuf__, int sendcount__, T* recvbuf__, int const* recvcounts__, int const* displs__) const
    {
#if defined(__PROFILE_MPI)
        PROFILE("MPI_Allgatherv");
#endif
        CALL_MPI(MPI_Allgatherv, (sendbuf__, sendcount__, mpi_type_wrapper<T>::kind(), recvbuf__, recvcounts__,
                                  displs__, mpi_type_wrapper<T>::kind(), mpi_comm()));
    }

    template <typename T>
    void allgather(T const* sendbuf__, T* recvbuf__, int offset__, int count__) const
    {
        std::vector<int> v(size() * 2);
        v[2 * rank()]     = count__;
        v[2 * rank() + 1] = offset__;

        CALL_MPI(MPI_Allgather,
                 (MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, v.data(), 2, mpi_type_wrapper<int>::kind(), mpi_comm()));

        std::vector<int> counts(size());
        std::vector<int> offsets(size());

        for (int i = 0; i < size(); i++) {
            counts[i]  = v[2 * i];
            offsets[i] = v[2 * i + 1];
        }

        CALL_MPI(MPI_Allgatherv, (sendbuf__, count__, mpi_type_wrapper<T>::kind(), recvbuf__, counts.data(),
                                  offsets.data(), mpi_type_wrapper<T>::kind(), mpi_comm()));
    }

    /// In-place MPI_Allgatherv.
    template <typename T>
    void allgather(T* buffer__, int offset__, int count__) const
    {
        std::vector<int> v(size() * 2);
        v[2 * rank()]     = count__;
        v[2 * rank() + 1] = offset__;

        CALL_MPI(MPI_Allgather,
                 (MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, v.data(), 2, mpi_type_wrapper<int>::kind(), mpi_comm()));

        std::vector<int> counts(size());
        std::vector<int> offsets(size());

        for (int i = 0; i < size(); i++) {
            counts[i]  = v[2 * i];
            offsets[i] = v[2 * i + 1];
        }
        allgather(buffer__, counts.data(), offsets.data());
    }

    template <typename T>
    void send(T const* buffer__, int count__, int dest__, int tag__) const
    {
#if defined(__PROFILE_MPI)
        PROFILE("MPI_Send");
#endif
        CALL_MPI(MPI_Send, (buffer__, count__, mpi_type_wrapper<T>::kind(), dest__, tag__, mpi_comm()));
    }

    template <typename T>
    Request isend(T const* buffer__, int count__, int dest__, int tag__) const
    {
        Request req;
#if defined(__PROFILE_MPI)
        PROFILE("MPI_Isend");
#endif
        CALL_MPI(MPI_Isend, (buffer__, count__, mpi_type_wrapper<T>::kind(), dest__, tag__, mpi_comm(), &req.handler()));
        return req;
    }

    template <typename T>
    void recv(T* buffer__, int count__, int source__, int tag__) const
    {
#if defined(__PROFILE_MPI)
        PROFILE("MPI_Recv");
#endif
        CALL_MPI(MPI_Recv,
                 (buffer__, count__, mpi_type_wrapper<T>::kind(), source__, tag__, mpi_comm(), MPI_STATUS_IGNORE));
    }

    template <typename T>
    Request irecv(T* buffer__, int count__, int source__, int tag__) const
    {
        Request req;
#if defined(__PROFILE_MPI)
        PROFILE("MPI_Irecv");
#endif
        CALL_MPI(MPI_Irecv, (buffer__, count__, mpi_type_wrapper<T>::kind(), source__, tag__, mpi_comm(), &req.handler()));
        return req;
    }

    template <typename T>
    void gather(T const* sendbuf__, T* recvbuf__, int const* recvcounts__, int const* displs__, int root__) const
    {
        int sendcount = recvcounts__[rank()];

#if defined(__PROFILE_MPI)
        PROFILE("MPI_Gatherv");
#endif
        CALL_MPI(MPI_Gatherv, (sendbuf__, sendcount, mpi_type_wrapper<T>::kind(), recvbuf__, recvcounts__, displs__,
                               mpi_type_wrapper<T>::kind(), root__, mpi_comm()));
    }

    /// Gather data on a given rank.
    template <typename T>
    void gather(T const* sendbuf__, T* recvbuf__, int offset__, int count__, int root__) const
    {

#if defined(__PROFILE_MPI)
        PROFILE("MPI_Gatherv");
#endif
        std::vector<int> v(size() * 2);
        v[2 * rank()]     = count__;
        v[2 * rank() + 1] = offset__;

        CALL_MPI(MPI_Allgather,
                 (MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, v.data(), 2, mpi_type_wrapper<int>::kind(), mpi_comm()));

        std::vector<int> counts(size());
        std::vector<int> offsets(size());

        for (int i = 0; i < size(); i++) {
            counts[i]  = v[2 * i];
            offsets[i] = v[2 * i + 1];
        }
        CALL_MPI(MPI_Gatherv, (sendbuf__, count__, mpi_type_wrapper<T>::kind(), recvbuf__, counts.data(),
                               offsets.data(), mpi_type_wrapper<T>::kind(), root__, mpi_comm()));
    }

    template <typename T>
    void scatter(T const* sendbuf__, T* recvbuf__, int const* sendcounts__, int const* displs__, int root__) const
    {
#if defined(__PROFILE_MPI)
        PROFILE("MPI_Scatterv");
#endif
        int recvcount = sendcounts__[rank()];
        CALL_MPI(MPI_Scatterv, (sendbuf__, sendcounts__, displs__, mpi_type_wrapper<T>::kind(), recvbuf__, recvcount,
                                mpi_type_wrapper<T>::kind(), root__, mpi_comm()));
    }

    template <typename T>
    void alltoall(T const* sendbuf__, int sendcounts__, T* recvbuf__, int recvcounts__) const
    {
#if defined(__PROFILE_MPI)
        PROFILE("MPI_Alltoall");
#endif
        CALL_MPI(MPI_Alltoall, (sendbuf__, sendcounts__, mpi_type_wrapper<T>::kind(), recvbuf__, recvcounts__,
                                mpi_type_wrapper<T>::kind(), mpi_comm()));
    }

    template <typename T>
    void alltoall(T const* sendbuf__,
                  int const* sendcounts__,
                  int const* sdispls__,
                  T* recvbuf__,
                  int const* recvcounts__,
                  int const* rdispls__) const
    {
#if defined(__PROFILE_MPI)
        PROFILE("MPI_Alltoallv");
#endif
        CALL_MPI(MPI_Alltoallv, (sendbuf__, sendcounts__, sdispls__, mpi_type_wrapper<T>::kind(), recvbuf__,
                                 recvcounts__, rdispls__, mpi_type_wrapper<T>::kind(), mpi_comm()));
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
    //==        std::printf("wrong sizes");
    //==        MPI_Abort(MPI_COMM_WORLD, -1);
    //==    }

    //==    return a2a;
    //==}
};

/// Get number of ranks per node.
int num_ranks_per_node();

int get_device_id(int num_devices__);

/// Parallel standard output.
/** Proveides an ordered standard output from multiple MPI ranks. */
class pstdout
{
  private:
    std::vector<char> buffer_;

    int count_{0};

    Communicator const& comm_;

  public:
    pstdout(Communicator const& comm__)
        : comm_(comm__)
    {
    }

    ~pstdout()
    {
        flush();
    }

    void printf(const char* fmt, ...);

    void flush();
};

} // namespace sddk

#endif // __COMMUNICATOR_HPP__
