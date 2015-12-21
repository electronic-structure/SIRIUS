#include <sirius.h>
#include <thread>

using namespace sirius;

void test_fft_performance(vector3d<int> const& dims__, double cutoff__, int num_bands__)
{
    Communicator comm(MPI_COMM_WORLD);

    matrix3d<double> M;
    M(0, 0) = M(1, 1) = M(2, 2) = 1.0;

    FFT3D<CPU> fft(dims__, 1, Platform::max_num_threads(), comm);
    
    Gvec gvec(vector3d<double>(0, 0, 0), cutoff__, M, &fft, false);

    if (comm.rank() == 0)
    {
        printf("num_gvec: %i\n", gvec.num_gvec());
        printf("num_xy_packed: %i\n", gvec.num_xy_packed());
    }

    mdarray<double_complex, 2> psi(gvec.num_gvec_loc(), num_bands__);
    for (int i = 0; i < num_bands__; i++)
    {
        for (int j = 0; j < gvec.num_gvec_loc(); j++)
        {
            psi(j, i) = type_wrapper<double_complex>::random();
        }
    }
    
    Timer t("fft_loop");
    for (int i = 0; i < num_bands__; i++)
    {
        
        fft.input_custom(gvec.num_gvec_loc(), gvec.index_map_xy(), &psi(0, i));
        fft.transform_custom(1, gvec.num_xy_packed(), gvec.xy_packed_idx());
        fft.transform_custom(-1, gvec.num_xy_packed(), gvec.xy_packed_idx());
        fft.output_custom(gvec.num_gvec_loc(), gvec.index_map_xy(), &psi(0, i));
    }
    double tval = t.stop();
    if (comm.rank() == 0)
    {
        printf("performance: %8.4f FFTs/sec\n", 2 * num_bands__ / tval);
    }
}

int main(int argn, char **argv)
{
    cmd_args args;
    args.register_key("--dims=", "{vector3d<int>} FFT dimensions");
    args.register_key("--cutoff=", "{double} cutoff radius in G-space");
    args.register_key("--num_bands=", "{int} number of bands");

    args.parse_args(argn, argv);
    if (argn == 1)
    {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        exit(0);
    }

    vector3d<int> dims = args.value< vector3d<int> >("dims");
    double cutoff = args.value<double>("cutoff", 1);
    int num_bands = args.value<int>("num_bands", 1);

    Platform::initialize(1);

    test_fft_performance(dims, cutoff, num_bands);

    Timer::print();

    Platform::finalize();
}
