#include <sirius.h>
#include <thread>
#ifdef __PAPI
#include <papi.h>
#endif
#include "kiss_fft.h"

using namespace sirius;

#define NOW std::chrono::high_resolution_clock::now()

#ifdef __PAPI
unsigned long int static thread_id(void)
{
    return omp_get_thread_num();
}
#endif

void kernel_fractal(int size, double_complex* in, double_complex* out)
{
    for (int i = 0; i < size; i++)
    {
        out[i] = double_complex(0, 0);
        for (int j = 0; j < 100; j++)
        {
            out[i] = std::pow(out[i], 2) + in[i];
        }
    }
}

void kernel_v2(int size, double_complex* in, double_complex* out)
{
    for (int i = 0; i < size; i++)
    {
        out[i] = double_complex(0, 0);
    }
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            out[(i + j) % size] = out[i] + in[(i + j) % size];
        }
    }
}

void kernel_fft(int size, double_complex* buf)
{
    int depth = int(log2(size) + 1e-10);
    if (size != std::pow(2, depth))
    {
        printf("wrong FFT size");
        exit(0);
    }
    
    for (int i = 0; i < size / 2; i++)
    {
        int j = 0;
        for (int k = 0; k < depth; k++)
        {
            if (i & (1 << (depth - k - 1))) j += (1 << k);
        }
        if (i != j) std::swap(buf[i], buf[j]);
    }
    
    int nb = size / 2;
    for (int s = 0; s < depth; s++)
    {
        int bs = 1 << (s + 1);
        for (int ib = 0; ib < nb; ib++)
        {
            for (int k = 0; k < (1 << s); k++)
            {
                double_complex w = std::exp(-double_complex(0, twopi * k / size));
                auto z1 = buf[ib * bs + k];
                auto z2 = buf[ib * bs + k + (1<<s)];
    
                buf[ib * bs + k] = z1 + w * z2;
                buf[ib * bs + k + (1<<s)] = z1 - w * z2;
            }
        }
    
        nb >>= 1;
    }
}

void kernel_zgemm(int size, double_complex* in, double_complex* out)
{
    linalg<device_t::CPU>::gemm(0, 0, size, size, size, double_complex(1, 0), in, size, in, size, double_complex(0, 0), out, size);
}

void kernel_memcpy(int size, double_complex* in, double_complex* out)
{
    memcpy(out, in, size * sizeof(double_complex));
}


template <int kernel_id>
void test_fft_1d(int size, int num_tasks, int repeat)
{
    int num_threads = omp_get_max_threads();

    std::vector<double_complex*> in_buf(num_threads);
    std::vector<double_complex*> out_buf(num_threads);

    int N = (kernel_id == 5) ? size * size : size;

    for (int i = 0; i < num_threads; i++)
    {
        in_buf[i] = (double_complex*)fftw_malloc(N * sizeof(double_complex));
        out_buf[i] = (double_complex*)fftw_malloc(N * sizeof(double_complex));
    }

    kiss_fft_cfg cfg;
    if (kernel_id == 1) cfg = kiss_fft_alloc(size, false, 0, 0);

    std::vector<fftw_plan> plan(num_threads);
    fftw_plan_with_nthreads(1);
    if (kernel_id == 0)
    {
        for (int i = 0; i < num_threads; i++)
        {
            plan[i] = fftw_plan_dft_1d(size, (fftw_complex*)in_buf[i], (fftw_complex*)in_buf[i],
                                       FFTW_BACKWARD, FFTW_ESTIMATE);
        }
    }

    int M = (kernel_id == 0 || kernel_id == 1 || kernel_id == 2) ? num_tasks : num_threads;

    mdarray<double_complex, 2> A(N, M);
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++) A(j, i) = type_wrapper<double_complex>::random();
    }

    mdarray<double, 2> times(num_threads, repeat);
    times.zero();
    mdarray<int, 2> counts(num_threads, repeat);
    counts.zero();
    
    #ifdef __PAPI
    mdarray<long long, 2> values(2, num_threads);
    values.zero();
    mdarray<long long, 2> tmp_values(2, num_threads);

    int Events[2] = {PAPI_TOT_CYC, PAPI_TOT_INS};
    int num_hwcntrs = 0;

    int retval;
    /* Initialize the library */
    retval = PAPI_library_init(PAPI_VER_CURRENT);
    if (retval != PAPI_VER_CURRENT && retval > 0) 
    {
        TERMINATE("PAPI library version mismatch!\n");
    }

    if (PAPI_thread_init(thread_id) != PAPI_OK)
    {
        TERMINATE("PAPI_thread_init error");
    }

    /* Initialize the PAPI library and get the number of counters available */
    if ((num_hwcntrs = PAPI_num_counters()) <= PAPI_OK)
    {
        TERMINATE("PAPI error");
    }
    printf("number of available counters: %i\n", num_hwcntrs);
       
    if (num_hwcntrs > 2) num_hwcntrs = 2;
    
    #pragma omp parallel num_threads(num_fft_workers)
    {
        /* Start counting events */
        if (PAPI_start_counters(Events, num_hwcntrs) != PAPI_OK)
        {
            TERMINATE("PAPI error");
        }
    }
    #endif
            
    auto t0 = NOW;
    for (int i = 0; i < repeat; i++)
    {
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();

            #pragma omp for schedule(static)
            for (int j = 0; j < num_tasks; j++)
            {
                if (kernel_id == 0 || kernel_id == 1 || kernel_id == 2)
                {
                    memcpy(in_buf[tid], &A(0, j), N * sizeof(double_complex));
                }
                else
                {
                    memcpy(in_buf[tid], &A(0, tid), N * sizeof(double_complex));
                }
                auto tt = NOW;
                #ifdef __PAPI
                PAPI_read_counters(&tmp_values(0, tid), num_hwcntrs);
                #endif
                if (kernel_id == 0) fftw_execute(plan[tid]);
                if (kernel_id == 1) kiss_fft(cfg, (kiss_fft_cpx*)in_buf[tid], (kiss_fft_cpx*)out_buf[tid]);
                if (kernel_id == 2) kernel_fft(size, in_buf[tid]);
                if (kernel_id == 3) kernel_memcpy(size, in_buf[tid], out_buf[tid]);
                if (kernel_id == 4) kernel_fractal(size, in_buf[tid], out_buf[tid]);
                if (kernel_id == 5) kernel_zgemm(size, in_buf[tid], out_buf[tid]);
                #ifdef __PAPI
                PAPI_accum_counters(&values(0, tid), num_hwcntrs);
                #endif
                times(tid, i) += std::chrono::duration_cast< std::chrono::duration<double> >(NOW - tt).count();
                counts(tid, i)++;
            }
        }
    }
    double tot_time = std::chrono::duration_cast< std::chrono::duration<double> >(NOW - t0).count();

    #ifdef __PAPI
    #pragma omp parallel num_threads(num_fft_workers)
    {
        int tid = omp_get_thread_num();
        /* Stop counting events */
        if (PAPI_stop_counters(&tmp_values(0, tid), num_hwcntrs) != PAPI_OK)
        {
            TERMINATE("PAPI error");
        }
    }
    #endif
    
    double avg_thread_perf = 0;
    runtime::pstdout pout(mpi_comm_world());
    pout.printf("\n");
    pout.printf("rank: %2i\n", mpi_comm_world().rank());
    pout.printf("---------\n");
    for (int tid = 0; tid < num_threads; tid++)
    {
        std::vector<double> x(repeat);
        double avg = 0;
        double tot_time_thread = 0;
        for (int i = 0; i < repeat; i++)
        {
            x[i] = (counts(tid, i) == 0) ? 0 : counts(tid, i) / times(tid, i);
            avg += x[i];
            tot_time_thread += times(tid, i);
        }
        avg /= repeat;
        avg_thread_perf += avg;
        double variance = 0;
        for (int i = 0; i < repeat; i++) variance += std::pow(x[i] - avg, 2);
        variance /= repeat;
        double sigma = std::sqrt(variance);
        pout.printf("                            tid : %i\n", tid);
        pout.printf("            average performance : %.4f\n", avg);
        pout.printf("                          sigma : %.4f\n", sigma);
        pout.printf("       coefficient of variation : %.2f%%\n", 100 * (sigma / avg));
        pout.printf("                    kernel time : %.2f%%\n", 100 * tot_time_thread / tot_time);
        #ifdef __PAPI
        pout.printf("                   total cycles : %lld\n", values(0, tid));
        pout.printf("         instructions completed : %lld\n", values(1, tid));
        pout.printf(" completed instructions / cycle : %f\n", double(values(1, tid)) / values(0, tid));
        #endif
        pout.printf("                             -------\n");
    }
    double perf = repeat * num_tasks / tot_time;
    pout.printf("---------\n");
    pout.printf("average thread performance: %.4f kernels/sec./thread\n", avg_thread_perf / num_threads);
    pout.printf("MPI rank effective performance: %.4f kernels/sec./rank\n", perf);
    pout.printf("---------\n");
    pout.flush();

    mpi_comm_world().allreduce(&perf, 1);
    mpi_comm_world().allreduce(&avg_thread_perf, 1);
    if (mpi_comm_world().rank() == 0)
    {
        printf("\n");
        printf("Average MPI rank performance: %.4f kernels/sec.\n", avg_thread_perf / mpi_comm_world().size());
        printf("Aggregate effective performance: %.4f kernels/sec.\n", perf);
        printf("\n");
    }

    for (int i = 0; i < num_threads; i++)
    {
        fftw_free(in_buf[i]);
        fftw_free(out_buf[i]);
        if (kernel_id == 0) fftw_destroy_plan(plan[i]);
    }
}

int main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--size=", "{int} kernel size (length of FFT buffer or matrix size of a zgemm kernel(");
    args.register_key("--num_tasks=", "{int} number of kernels inside one measurment");
    args.register_key("--repeat=", "{int} number of measurments");
    args.register_key("--kernel=", "{int} 0: fftw, 1: kiss_fft, 2: custom_fft, 3: memcpy, 4: fractal progression, 5: zgemm");

    args.parse_args(argn, argv);
    if (argn == 1)
    {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        exit(0);
    }

    int size = args.value<int>("size", 128);
    int num_tasks = args.value<int>("num_tasks", 64);
    int repeat = args.value<int>("repeat", 100);
    int kernel_id = args.value<int>("kernel", 0);

    sirius::initialize(1);

    if (kernel_id == 0) test_fft_1d<0>(size, num_tasks, repeat);
    if (kernel_id == 1) test_fft_1d<1>(size, num_tasks, repeat);
    if (kernel_id == 2) test_fft_1d<2>(size, num_tasks, repeat);
    if (kernel_id == 3) test_fft_1d<3>(size, num_tasks, repeat);
    if (kernel_id == 4) test_fft_1d<4>(size, num_tasks, repeat);
    if (kernel_id == 5) test_fft_1d<5>(size, num_tasks, repeat);

    sirius::finalize();
}
