/** \file platform.h
    
    \brief Contains definition and implementation of Platform class.
*/

#ifndef __PLATFORM_H__
#define __PLATFORM_H__

#include <mpi.h>
#include <omp.h>
#include <signal.h>
#ifdef _GPU_
#include "gpu_interface.h"
#endif
#include <vector>
#include "typedefs.h"

class Platform
{
    private:

        static int num_fft_threads_;
    
    public:

        static void initialize(bool call_mpi_init, bool call_cublas_init = true);

        static void finalize();

        static int mpi_rank(MPI_Comm comm = MPI_COMM_WORLD);

        static int num_mpi_ranks(MPI_Comm comm = MPI_COMM_WORLD);

        static void abort();

        /// Broadcast array 
        template <typename T>
        static void bcast(T* buffer, int count, const MPI_Comm& comm, int root);
        
        /// Broadcast array 
        template <typename T>
        static void bcast(T* buffer, int count, int root);
       
        /// Perform all-to-one in-place reduction
        template <typename T>
        static void reduce(T* buf, int count, const MPI_Comm& comm, int root);

        /// Perform all-to-one out-of-place reduction 
        template <typename T>
        static void reduce(T* sendbuf, T* recvbuf, int count, const MPI_Comm& comm, int root);
        
        /// Perform the in-place (the output buffer is used as the input buffer) all-to-all reduction 
        template <typename T>
        static void allreduce(T* buffer, int count, const MPI_Comm& comm);

        /// Perform the in-place (the output buffer is used as the input buffer) all-to-all reduction 
        template<mpi_op_t op, typename T>
        static void allreduce(T* buffer, int count, const MPI_Comm& comm);
        
        /// Perform the in-place (the output buffer is used as the input buffer) all-to-all reduction 
        template <typename T>
        static void allreduce(T* buffer, int count);
        
        /// Perform the in-place (the output buffer is used as the input buffer) all-to-all reduction 
        template<mpi_op_t op, typename T>
        static void allreduce(T* buffer, int count);

        template <typename T>
        static void allgather(T* sendbuf, T* recvbuf, int offset, int count);
        
        template <typename T>
        static void allgather(T* buf, int offset, int count, MPI_Comm comm = MPI_COMM_WORLD);

        template <typename T> 
        static void gather(T* sendbuf, T* recvbuf, int *recvcounts, int *displs, int root, MPI_Comm comm);

        template <typename T>
        static void scatter(T* sendbuf, T* recvbuf, int* sendcounts, int* displs, int root, MPI_Comm comm);

        /// Non-blocking send.
        template <typename T>
        static void isend(T* buf, int count, int dest, int tag, MPI_Comm comm);

        /// Blocking recieve.
        template <typename T>
        static void recv(T* buf, int count, int source, int tag, MPI_Comm comm);
        
        /// Returm maximum number of OMP threads.
        /** Maximum number of OMP threads is controlled by environment variable OMP_NUM_THREADS */
        static inline int max_num_threads()
        {
            return omp_get_max_threads();
        }

        /// Returm number of actually running OMP threads. 
        static inline int num_threads()
        {
            return omp_get_num_threads();
        }
        
        /// Return thread id.
        static inline int thread_id()
        {
            return omp_get_thread_num();
        }
        
        /// Return number of threads for independent FFT transformations.
        static inline int num_fft_threads()
        {
            return num_fft_threads_;
        }
        
        /// Set the number of FFT threads
        static inline void set_num_fft_threads(int num_fft_threads__)
        {
            num_fft_threads_ = num_fft_threads__;
        }
        
        /// Global barrier
        static void barrier(MPI_Comm comm = MPI_COMM_WORLD)
        {
            MPI_Barrier(comm);
        }
};

#include "platform.hpp"

#endif
