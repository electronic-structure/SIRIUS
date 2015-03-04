#include <sirius.h>

using namespace sirius;

void test_fft(vector3d<int>& dims__)
{
    printf("test of threaded FFTs\n");

    int max_num_threads = Platform::max_num_threads();
    std::vector< std::pair<int, int> > threads_conf;

    for (int i = 1; i <= max_num_threads; i++)
    {
        for (int j = 1; j <= max_num_threads; j++)
        {
            if (i * j < 2 * max_num_threads) threads_conf.push_back(std::pair<int, int>(i, j));
        }
    }

    matrix3d<double> reciprocal_lattice_vectors;
    for (int i = 0; i < 3; i++) reciprocal_lattice_vectors(i, i) = 1.0;

    for (int k = 0; k < (int)threads_conf.size(); k++)
    {
        printf("\n");
        printf("number of fft threads: %i, number of fft workers: %i\n", threads_conf[k].first, threads_conf[k].second);

        FFT3D<CPU> fft(dims__, threads_conf[k].first, threads_conf[k].second);
        fft.init_gvec(20.0, reciprocal_lattice_vectors);

        int num_phi = 160;
        mdarray<double_complex, 2> phi(fft.num_gvec(), num_phi);
        for (int i = 0; i < num_phi; i++)
        {
            for (int ig = 0; ig < fft.num_gvec(); ig++) phi(ig, i) = type_wrapper<double_complex>::random();
        }

        Timer t("fft_loop");
        #pragma omp parallel num_threads(fft.num_fft_threads())
        {
            int thread_id = Platform::thread_id();
            
            #pragma omp for
            for (int i = 0; i < num_phi; i++)
            {
                fft.input(fft.num_gvec(), fft.index_map(), &phi(0, i), thread_id);
                fft.transform(1, thread_id);

                for (int ir = 0; ir < fft.size(); ir++) fft.buffer(ir, thread_id) += 1.0;

                fft.transform(-1, thread_id);
                fft.output(fft.num_gvec(), fft.index_map(), &phi(0, i), thread_id);
            }
        }
        double tval = t.stop();

        printf("performance: %f, (FFT/sec.)\n", 2 * num_phi / tval);
    }
}

int main(int argn, char **argv)
{
    cmd_args args;
    args.register_key("--dims=", "{vector3d<int>} FFT dimensions");

    args.parse_args(argn, argv);
    if (argn == 1)
    {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        exit(0);
    }

    vector3d<int> dims = args.value< vector3d<int> >("dims");

    Platform::initialize(1);

    test_fft(dims);

    Platform::finalize();
}
