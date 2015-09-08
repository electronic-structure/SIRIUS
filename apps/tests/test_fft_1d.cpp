#include <sirius.h>

using namespace sirius;

void test_fft_1d(int fft_size, int num_fft, int repeat)
{
    int num_fft_workers = Platform::max_num_threads();

    std::vector<fftw_plan> plan_backward_z(num_fft_workers);
    std::vector<double_complex*> fftw_buffer_z(num_fft_workers);

    fftw_plan_with_nthreads(1);
    for (int i = 0; i < num_fft_workers; i++)
    {
        fftw_buffer_z[i] = (double_complex*)fftw_malloc(fft_size * sizeof(double_complex));

        plan_backward_z[i] = fftw_plan_dft_1d(fft_size, (fftw_complex*)fftw_buffer_z[i], (fftw_complex*)fftw_buffer_z[i],
                                              FFTW_BACKWARD, FFTW_ESTIMATE);
    }

    mdarray<double_complex, 2> psi(fft_size, num_fft_workers);
    for (int i = 0; i < num_fft_workers; i++)
    {
        for (int j = 0; j < fft_size; j++) psi(j, i) = type_wrapper<double_complex>::random();
    }

    mdarray<double, 2> times(num_fft_workers, repeat);
    times.zero();
    mdarray<int, 2> counts(num_fft_workers, repeat);
    counts.zero();
    
    double tot_time = -omp_get_wtime();
    for (int i = 0; i < repeat; i++)
    {
        #pragma omp parallel num_threads(num_fft_workers)
        {
            int tid = omp_get_thread_num();

            #pragma omp for
            for (int j = 0; j < num_fft; j++)
            {
                memcpy(fftw_buffer_z[tid], &psi(0, tid), fft_size * sizeof(double_complex));
                double tt = omp_get_wtime();
                fftw_execute(plan_backward_z[tid]);
                times(tid, i) += (omp_get_wtime() - tt);
                counts(tid, i)++;
            }
        }
    }
    tot_time += omp_get_wtime();
    
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
        pout.printf("tid: %2i, mean: %16.4f, sigma: %16.4f, cv: %6.2f%%, kernel time: %6.2f%%\n",
                    tid, avg, sigma, 100 * (sigma / avg), 100 * tot_time_thread / tot_time);
    }
    double perf = repeat * num_fft / tot_time;
    pout.printf("performance: %.4f FFTs/sec/rank\n", perf);
    pout.flush();

    comm_world.allreduce(&perf, 1);
    if (comm_world.rank() == 0)
    {
        printf("\n");
        printf("Aggregate performance: %.4f FFTs/sec\n", perf);
    }

    for (int i = 0; i < num_fft_workers; i++)
    {
        fftw_free(fftw_buffer_z[i]);
        fftw_destroy_plan(plan_backward_z[i]);
    }
}

int main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--fft_size=", "{int} size of 1D FFT");
    args.register_key("--num_fft=", "{int} number of FFTs inside one measurment");
    args.register_key("--repeat=", "{int} number of measurments");

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

    Platform::initialize(1);

    test_fft_1d(fft_size, num_fft, repeat);

    Platform::finalize();
}
