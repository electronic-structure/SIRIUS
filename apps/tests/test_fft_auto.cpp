#include <sirius.h>
#include <thread>

using namespace sirius;

void test_fft_auto(vector3d<int>& dims__)
{
    matrix3d<double> reciprocal_lattice_vectors;
    for (int i = 0; i < 3; i++) reciprocal_lattice_vectors(i, i) = 1.0;

    FFT3D<CPU> fft(dims__, 1, -1);
    fft.init_gvec(20.0, reciprocal_lattice_vectors);

    int num_phi = 160;
    mdarray<double_complex, 2> phi(fft.num_gvec(), num_phi);
    for (int i = 0; i < num_phi; i++)
    {
        for (int ig = 0; ig < fft.num_gvec(); ig++) phi(ig, i) = type_wrapper<double_complex>::random();
    }

    Timer t("fft_loop");
    for (int i = 0; i < num_phi; i++)
    {
        fft.input(fft.num_gvec(), fft.index_map(), &phi(0, i));
        fft.transform(1);

        for (int ir = 0; ir < fft.size(); ir++) fft.buffer(ir) += 1.0;

        fft.transform(-1);
        fft.output(fft.num_gvec(), fft.index_map(), &phi(0, i));
    }
    double tval = t.stop();

    printf("performance: %f, (FFT/sec.)\n", 2 * num_phi / tval);
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

    test_fft_auto(dims);

    Platform::finalize();
}
