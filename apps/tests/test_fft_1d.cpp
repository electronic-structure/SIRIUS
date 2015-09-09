#include <sirius.h>
#include <thread>
#include <papi.h>
#include "kiss_fft.h"

using namespace sirius;

#define NOW std::chrono::high_resolution_clock::now()

unsigned long int static thread_id(void)
{
    return omp_get_thread_num();
}

void kernel_v1(int size, double_complex* in, double_complex* out)
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

template <int fft_kernel>
void test_fft_1d(int fft_size, int num_fft, int repeat)
{
    kiss_fft_cfg cfg = kiss_fft_alloc( fft_size, false ,0,0 );

    int num_fft_workers = Platform::max_num_threads();

    std::vector<fftw_plan> plan_backward_z(num_fft_workers);
    std::vector<double_complex*> fftw_buffer_z(num_fft_workers);
    std::vector<double_complex*> fftw_out_buffer_z(num_fft_workers);

    fftw_plan_with_nthreads(1);
    for (int i = 0; i < num_fft_workers; i++)
    {
        fftw_buffer_z[i] = (double_complex*)fftw_malloc(fft_size * sizeof(double_complex));

        fftw_out_buffer_z[i] = (double_complex*)fftw_malloc(fft_size * sizeof(double_complex));

        plan_backward_z[i] = fftw_plan_dft_1d(fft_size, (fftw_complex*)fftw_buffer_z[i], (fftw_complex*)fftw_buffer_z[i],
                                              FFTW_BACKWARD, FFTW_ESTIMATE);
    }

    mdarray<double_complex, 2> psi(fft_size, num_fft);
    for (int i = 0; i < num_fft; i++)
    {
        for (int j = 0; j < fft_size; j++) psi(j, i) = type_wrapper<double_complex>::random();
    }

    mdarray<double, 2> times(num_fft_workers, repeat);
    times.zero();
    mdarray<int, 2> counts(num_fft_workers, repeat);
    counts.zero();

    mdarray<long long, 2> values(2, num_fft_workers);
    values.zero();
    mdarray<long long, 2> tmp_values(2, num_fft_workers);

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
            
    auto t0 = NOW;
    for (int i = 0; i < repeat; i++)
    {
        #pragma omp parallel num_threads(num_fft_workers)
        {
            int tid = omp_get_thread_num();

            #pragma omp for schedule(static)
            for (int j = 0; j < num_fft; j++)
            {
                memcpy(fftw_buffer_z[tid], &psi(0, j), fft_size * sizeof(double_complex));
                auto tt = NOW;
                PAPI_read_counters(&tmp_values(0, tid), num_hwcntrs);
                if (fft_kernel == 0) fftw_execute(plan_backward_z[tid]);
                if (fft_kernel == 1) kiss_fft(cfg, (kiss_fft_cpx*)fftw_buffer_z[tid], (kiss_fft_cpx*)fftw_out_buffer_z[tid]);
                if (fft_kernel == 2) kernel_fft(fft_size, fftw_buffer_z[tid]);
                PAPI_accum_counters(&values(0, tid), num_hwcntrs);
                times(tid, i) += std::chrono::duration_cast< std::chrono::duration<double> >(NOW - tt).count();
                counts(tid, i)++;
            }
        }
    }
    double tot_time = std::chrono::duration_cast< std::chrono::duration<double> >(NOW - t0).count();
    #pragma omp parallel num_threads(num_fft_workers)
    {
        int tid = omp_get_thread_num();
        /* Stop counting events */
        if (PAPI_stop_counters(&tmp_values(0, tid), num_hwcntrs) != PAPI_OK)
        {
            TERMINATE("PAPI error");
        }
    }

    Communicator comm_world(MPI_COMM_WORLD);
    pstdout pout(comm_world);
    pout.printf("\n");
    pout.printf("rank: %2i\n", comm_world.rank());
    pout.printf("---------\n");
    for (int tid = 0; tid < num_fft_workers; tid++)
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
        double variance = 0;
        for (int i = 0; i < repeat; i++) variance += std::pow(x[i] - avg, 2);
        variance /= repeat;
        double sigma = std::sqrt(variance);
        pout.printf("                            tid : %i\n", tid);
        pout.printf("            average performance : %.4f\n", avg);
        pout.printf("                          sigma : %.4f\n", sigma);
        pout.printf("       coefficient of variation : %.2f%%\n", 100 * (sigma / avg));
        pout.printf("                    kernel time : %.2f%%\n", 100 * tot_time_thread / tot_time);
        pout.printf("                   total cycles : %lld\n", values(0, tid));
        pout.printf("         instructions completed : %lld\n", values(1, tid));
        pout.printf(" completed instructions / cycle : %f\n", double(values(1, tid)) / values(0, tid));
        pout.printf("                             -------\n");
    }
    double perf = repeat * num_fft / tot_time;
    pout.printf("---------\n");
    pout.printf("performance: %.4f FFTs/sec/rank\n", perf);
    pout.printf("---------\n");
    pout.flush();

    comm_world.allreduce(&perf, 1);
    if (comm_world.rank() == 0)
    {
        printf("\n");
        printf("Aggregate performance: %.4f FFTs/sec\n", perf);
        printf("\n");
    }

    for (int i = 0; i < num_fft_workers; i++)
    {
        fftw_free(fftw_buffer_z[i]);
        fftw_free(fftw_out_buffer_z[i]);
        fftw_destroy_plan(plan_backward_z[i]);
    }
}

int main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--fft_size=", "{int} size of 1D FFT");
    args.register_key("--num_fft=", "{int} number of FFTs inside one measurment");
    args.register_key("--repeat=", "{int} number of measurments");
    args.register_key("--kernel=", "{int} 0: fftw, 1: kiss_fft, 2: custom_fft");

    args.parse_args(argn, argv);
    if (argn == 1)
    {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        exit(0);
    }

    int fft_size = args.value<int>("fft_size", 128);
    int num_fft = args.value<int>("num_fft", 64);
    int repeat = args.value<int>("repeat", 100);
    int ikernel = args.value<int>("kernel", 0);

    Platform::initialize(1);

    if (ikernel == 0) test_fft_1d<0>(fft_size, num_fft, repeat);
    if (ikernel == 1) test_fft_1d<1>(fft_size, num_fft, repeat);
    if (ikernel == 2) test_fft_1d<2>(fft_size, num_fft, repeat);

    Platform::finalize();
}
