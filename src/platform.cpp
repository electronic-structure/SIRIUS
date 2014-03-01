#include "platform.h"

int Platform::num_fft_threads_ = -1;

#ifdef _PLASMA_
extern "C" void plasma_init(int num_cores);
#endif

void Platform::initialize(bool call_mpi_init, bool call_cublas_init)
{
    if (call_mpi_init) MPI_Init(NULL, NULL);

    #ifdef _GPU_
    cuda_initialize();
    if (call_cublas_init) cublas_init();
    if (mpi_rank() == 0) cuda_device_info();
    cuda_create_streams(max_num_threads());
    #endif
    #ifdef _MAGMA_
    magma_init_wrapper();
    #endif
    #ifdef _PLASMA_
    plasma_init();
    #endif

    assert(sizeof(int) == 4);
    assert(sizeof(double) == 8);
}

void Platform::finalize()
{
    MPI_Finalize();
    #ifdef _MAGMA_
    magma_finalize_wrapper();
    #endif
    #ifdef _GPU_
    cuda_destroy_streams(max_num_threads());
    cuda_device_reset();
    #endif
}

int Platform::mpi_rank(MPI_Comm comm)
{
    int rank;
    MPI_Comm_rank(comm, &rank);
    return rank;
}

int Platform::num_mpi_ranks(MPI_Comm comm)
{
    int size;
    MPI_Comm_size(comm, &size);
    return size;
}

void Platform::abort()
{
    if (num_mpi_ranks() == 1)
    {
        raise(SIGTERM);
    }
    else
    {   
        MPI_Abort(MPI_COMM_WORLD, -13);
    }
    exit(-13);
}
