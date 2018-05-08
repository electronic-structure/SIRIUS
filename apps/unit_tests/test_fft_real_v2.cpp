#include <sirius.h>

using namespace sirius;

void test1(vector3d<int> const& dims__, double cutoff__, device_t pu__)
{
    printf("test1\n");
    matrix3d<double> M;
    M(0, 0) = M(1, 1) = M(2, 2) = 1.0;

    FFT3D fft(find_translations(cutoff__, M), mpi_comm_world(), pu__);

    Gvec gvec(M, cutoff__, mpi_comm_world(), mpi_comm_world(), false);
    Gvec gvec_r(M, cutoff__, mpi_comm_world(), mpi_comm_world(), true);

    if (gvec_r.num_gvec() != gvec.num_gvec() / 2 + 1) {
        printf("wrong number of reduced G-vectors");
        exit(1);
    }

    fft.prepare(gvec_r.partition());

    printf("num_gvec: %i, num_gvec_reduced: %i\n", gvec.num_gvec(), gvec_r.num_gvec());
    printf("num_gvec_loc: %i %i\n", gvec.gvec_count(mpi_comm_world().rank()), gvec_r.gvec_count(mpi_comm_world().rank()));
    printf("num_z_col: %i, num_z_col_reduced: %i\n", gvec.num_zcol(), gvec_r.num_zcol());

    mdarray<double_complex, 1> phi(gvec_r.partition().gvec_count_fft(), memory_t::host | memory_t::device);
    for (int i = 0; i < gvec_r.partition().gvec_count_fft(); i++) {
        phi(i) = double_complex(1.0 / (i + 1), 1.0 / (i + 1) + 0.345);
    }
    phi(0) = 1.0;
    phi.copy<memory_t::host, memory_t::device>();
    if (pu__ == CPU) {
        fft.transform<1, CPU>(gvec_r.partition(), &phi[0]);
    } else {
        fft.transform<1, GPU>(gvec_r.partition(), phi.at<GPU>());
    }

    #ifdef __GPU
    if (pu__ == GPU) {
        fft.buffer().copy_to_host();
    }
    #endif

    for (int i = 0; i < fft.local_size(); i++) {
        if (fft.buffer(i).imag() > 1e-10) {
            printf("function is not real at idx = %i, image value: %18.12f\n", i, fft.buffer(i).imag());
            exit(1);
        }
    }
    //mdarray<double_complex, 1> phi1(gvec_r.partition().gvec_count_fft());
    //fft.transform<-1>(gvec_r.partition(), &phi1[0]);

    //double rms = 0;
    //for (int i = 0; i < gvec_r.partition().gvec_count_fft(); i++) {
    //    rms += std::pow(std::abs(phi(i) - phi1(i)), 2);
    //}
    //rms = std::sqrt(rms / gvec_r.partition().gvec_count_fft());
    //printf("rms: %18.12f\n", rms);
    //if (rms > 1e-13) {
    //    printf("functions are different\n");
    //    exit(1);
    //}

    fft.dismiss();
}

int main(int argn, char **argv)
{
    cmd_args args;
    args.register_key("--dims=", "{vector3d<int>} FFT dimensions");
    args.register_key("--cutoff=", "{double} cutoff radius in G-space");

    args.parse_args(argn, argv);
    if (args.exist("help"))
    {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }

    std::vector<int> vd = args.value<std::vector<int>>("dims", {132, 132, 132});
    vector3d<int> dims(vd[0], vd[1], vd[2]); 
    double cutoff = args.value<double>("cutoff", 50);

    sirius::initialize(1);

    test1(dims, cutoff, CPU);
    #ifdef __GPU
    test1(dims, cutoff, GPU);
    #endif
    
    sirius::finalize();

    return 0;
}
