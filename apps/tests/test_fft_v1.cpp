#include <sirius.h>

using namespace sirius;

void test_fft(vector3d<int> dims, double cutoff, std::vector<int> mpi_grid)
{
    FFT3D fft(dims, mpi_comm_self(), GPU);
    
    //Gvec gvec({0, 0, 0}, {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}, cutoff, fft.grid(), 1, mpi_comm_self(), false);

    //printf("num_gvec: %i\n", gvec.num_gvec());
    //printf("num_z_col: %i\n", gvec.num_zcol());

    //MPI_grid mpi_fft_grid({1, 1}, mpi_comm_self());
    //
    //mdarray<double_complex, 1> v(gvec.num_gvec());
    //v.zero();
    //v[0] = 1;
    //v.allocate_on_device();
    //v.copy_to_device();
    //
    //fft.prepare(gvec.partition());
    //for (int i = 0; i < 100; i++) {
    //    fft.transform<1>(gvec.partition(), v.at<GPU>());
    //    fft.transform<-1>(gvec.partition(), v.at<GPU>());
    //}
    //for (int i = 0; i < 100; i++) {
    //    fft.transform<1>(gvec.partition(), v.at<CPU>());
    //    fft.transform<-1>(gvec.partition(), v.at<CPU>());
    //}
    //v.copy_to_host();
    //std::cout << v[0] << std::endl;
    //fft.dismiss();
}

int main(int argn, char **argv)
{
    cmd_args args;
    args.register_key("--dims=", "{vector3d<int>} FFT dimensions");
    args.register_key("--cutoff=", "{double} cutoff radius in G-space");
    args.register_key("--mpi_grid=", "{vector2d<int>} MPI grid");

    args.parse_args(argn, argv);
    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }

    vector3d<int> dims = args.value< vector3d<int> >("dims");
    double cutoff = args.value<double>("cutoff", 1);
    std::vector<int> mpi_grid = args.value< std::vector<int> >("mpi_grid", {1, 1});

    sirius::initialize(1);

    test_fft(dims, cutoff, mpi_grid);

    sirius::finalize();
}
