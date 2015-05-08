#include <map>
#include <string>
#include <vector>
#include <stdio.h>
#include <execinfo.h>
#include <unistd.h>
#include <signal.h>
#include <assert.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cufft.h>
#include <stdint.h>

inline void stack_backtrace()
{
    void *array[10];
    char **strings;
    int size = backtrace(array, 10);
    strings = backtrace_symbols(array, size);
    printf ("Stack backtrace:\n");
    for (size_t i = 0; i < size; i++) printf ("%s\n", strings[i]);
    raise(SIGQUIT);
}

#ifdef NDEBUG
#define CALL_CUDA(func__, args__)                                                                                  \
{                                                                                                                  \
    cudaError_t error = func__ args__;                                                                             \
    if (error != cudaSuccess)                                                                                      \
    {                                                                                                              \
        char nm[1024];                                                                                             \
        gethostname(nm, 1024);                                                                                     \
        printf("hostname: %s\n", nm);                                                                              \
        printf("Error in %s at line %i of file %s: %s\n", #func__, __LINE__, __FILE__, cudaGetErrorString(error)); \
        stack_backtrace();                                                                                         \
    }                                                                                                              \
}
#else
#define CALL_CUDA(func__, args__)                                                                                  \
{                                                                                                                  \
    cudaError_t error;                                                                                             \
    func__ args__;                                                                                                 \
    cudaDeviceSynchronize();                                                                                       \
    error = cudaGetLastError();                                                                                    \
    if (error != cudaSuccess)                                                                                      \
    {                                                                                                              \
        char nm[1024];                                                                                             \
        gethostname(nm, 1024);                                                                                     \
        printf("hostname: %s\n", nm);                                                                              \
        printf("Error in %s at line %i of file %s: %s\n", #func__, __LINE__, __FILE__, cudaGetErrorString(error)); \
        stack_backtrace();                                                                                         \
    }                                                                                                              \
}
#endif

extern "C" {

void cuda_initialize();

void cuda_device_info();

void* cuda_malloc(size_t size);

void* cuda_malloc_host(size_t size);

void cuda_free_host(void* ptr);

void cuda_free(void* ptr);

void cuda_copy_to_device(void* target, void const* source, size_t size);

void cuda_copy_to_host(void* target, void const* source, size_t size);

void cuda_copy_device_to_device(void* target, void const* source, size_t size);

void cuda_memset(void *ptr, int value, size_t size);

void cuda_host_register(void* ptr, size_t size);

void cuda_host_unregister(void* ptr);

void cuda_device_synchronize();

void cuda_create_streams(int num_streams);

void cuda_destroy_streams(int num_streams);

void cuda_stream_synchronize(int stream_id);

void cuda_async_copy_to_device(void *target, void *source, size_t size, int stream_id);

void cuda_async_copy_to_host(void *target, void *source, size_t size, int stream_id);

size_t cuda_get_free_mem();

void cuda_device_reset();

void cuda_check_last_error();

void cublas_zgemm(int transa, int transb, int32_t m, int32_t n, int32_t k, 
                  cuDoubleComplex* alpha, cuDoubleComplex* a, int32_t lda, cuDoubleComplex* b, 
                  int32_t ldb, cuDoubleComplex* beta, cuDoubleComplex* c, int32_t ldc, int stream_id);

}

inline __device__ size_t array2D_offset(int i0, int i1, int ld0)
{
    return i0 + i1 * ld0;
}

// TODO: can be optimized in terms of multiplication
inline __device__ size_t array3D_offset(int i0, int i1, int i2, int ld0, int ld1)
{
    return i0 + i1 * ld0 + i2 * ld0 * ld1;
}

// TODO: can be optimized in terms of multiplication
inline __device__ size_t array4D_offset(int i0, int i1, int i2, int i3, int ld0, int ld1, int ld2)
{
    return i0 + i1 * ld0 + i2 * ld0 * ld1 + i3 * ld0 * ld1 * ld2;
}

inline __host__ __device__ int num_blocks(int length, int block_size)
{
    return (length / block_size) + min(length % block_size, 1);
}

class CUDA_timers_wrapper
{
    private:

        std::map<std::string, std::vector<float> > cuda_timers_;

    public:

        void add_measurment(std::string const& label, float value)
        {
            cuda_timers_[label].push_back(value / 1000);
        }

        void print()
        {
            printf("\n");
            printf("CUDA timers \n");
            for (int i = 0; i < 115; i++) printf("-");
            printf("\n");
            printf("name                                                              count      total        min        max    average\n");
            for (int i = 0; i < 115; i++) printf("-");
            printf("\n");

            std::map<std::string, std::vector<float> >::iterator it;
            for (it = cuda_timers_.begin(); it != cuda_timers_.end(); it++)
            {
                int count = (int)it->second.size();
                double total = 0.0;
                float minval = 1e10;
                float maxval = 0.0;
                for (int i = 0; i < count; i++)
                {
                    total += it->second[i];
                    minval = std::min(minval, it->second[i]);
                    maxval = std::max(maxval, it->second[i]);
                }
                double average = (count == 0) ? 0.0 : total / count;
                if (count == 0) minval = 0.0;

                printf("%-60s :    %5i %10.4f %10.4f %10.4f %10.4f\n", it->first.c_str(), count, total, minval, maxval, average);
            }
        }
};

class CUDA_timer
{
    private:

        cudaEvent_t e_start_;
        cudaEvent_t e_stop_;
        bool active_;
        std::string label_;

        void start()
        {
            cudaEventCreate(&e_start_);
            cudaEventCreate(&e_stop_);
            cudaEventRecord(e_start_, 0);
        }

        void stop()
        {
            float time;
            cudaEventRecord(e_stop_, 0);
            cudaEventSynchronize(e_stop_);
            cudaEventElapsedTime(&time, e_start_, e_stop_);
            cudaEventDestroy(e_start_);
            cudaEventDestroy(e_stop_);
            cuda_timers_wrapper().add_measurment(label_, time);
            active_ = false;
        }

    public:

        CUDA_timer(std::string const& label__) : label_(label__), active_(false)
        {
            start();
        }

        ~CUDA_timer()
        {
            stop();
        }

        static CUDA_timers_wrapper& cuda_timers_wrapper()
        {
            static CUDA_timers_wrapper cuda_timers_wrapper_;
            return cuda_timers_wrapper_;
        }
};


