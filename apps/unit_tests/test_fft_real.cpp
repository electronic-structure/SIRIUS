#include <sirius.h>
#include <thread>

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

    mdarray<double_complex, 1> phi(gvec_r.partition().gvec_count_fft());
    for (int i = 0; i < gvec_r.partition().gvec_count_fft(); i++) {
        phi(i) = type_wrapper<double_complex>::random();
    }
    phi(0) = 1.0;
    fft.transform<1>(&phi[0]);
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
    mdarray<double_complex, 1> phi1(gvec_r.partition().gvec_count_fft());
    fft.transform<-1>(&phi1[0]);

    double rms = 0;
    for (int i = 0; i < gvec_r.partition().gvec_count_fft(); i++) {
        rms += std::pow(std::abs(phi(i) - phi1(i)), 2);
    }
    rms = std::sqrt(rms / gvec_r.partition().gvec_count_fft());
    printf("rms: %18.12f\n", rms);
    if (rms > 1e-13) {
        printf("functions are different\n");
        exit(1);
    }

    fft.dismiss();
}

void test2(vector3d<int> const& dims__, double cutoff__, device_t pu__)
{
    printf("test2\n");
    matrix3d<double> M;
    M(0, 0) = M(1, 1) = M(2, 2) = 1.0;

    FFT3D fft(find_translations(cutoff__, M), mpi_comm_world(), pu__);

    Gvec gvec_r(M, cutoff__, mpi_comm_world(), mpi_comm_world(), true);

    fft.prepare(gvec_r.partition());

    mdarray<double_complex, 1> phi1(gvec_r.partition().gvec_count_fft());
    mdarray<double_complex, 1> phi2(gvec_r.partition().gvec_count_fft());
    mdarray<double_complex, 1> phi1_rg(fft.local_size());
    mdarray<double_complex, 1> phi2_rg(fft.local_size());

    for (int i = 0; i < gvec_r.partition().gvec_count_fft(); i++) {
        phi1(i) = type_wrapper<double_complex>::random();
        phi2(i) = type_wrapper<double_complex>::random();
    }
    phi1(0) = 1.0;
    phi2(0) = 1.0;

    fft.transform<1>(&phi1(0));
    fft.output(&phi1_rg(0));

    fft.transform<1>(&phi2(0));
    fft.output(&phi2_rg(0));

    for (int i = 0; i < fft.local_size(); i++) {
        if (phi1_rg(i).imag() > 1e-10) {
            printf("phi1 is not real at idx = %i, image value: %18.12f\n", i, phi1_rg(i).imag());
            exit(1);
        }
        if (phi2_rg(i).imag() > 1e-10) {
            printf("phi2 is not real at idx = %i, image value: %18.12f\n", i, phi2_rg(i).imag());
            exit(1);
        }
    }

    fft.transform<1>(&phi1(0), &phi2(0));

    mdarray<double_complex, 1> phi12_rg(fft.local_size());
    fft.output(&phi12_rg(0));

    //printf("phi1(0)=%18.10f\n", phi1_rg(0));
    //printf("phi2(0)=%18.10f\n", phi2_rg(0));

    for (int i = 0; i < fft.local_size(); i++) {
        if (std::abs(double_complex(phi1_rg(i).real(), phi2_rg(i).real()) - phi12_rg(i)) > 1e-10) {
            printf("functions don't match\n");
            printf("phi1: %18.10f\n", phi1_rg(i).real());
            printf("phi2: %18.10f\n", phi2_rg(i).real());
            printf("complex phi: %18.10f %18.10f\n", fft.buffer(i).real(), fft.buffer(i).imag());
            exit(1);
        }
    }

    mdarray<double_complex, 1> phi1_bt(gvec_r.partition().gvec_count_fft());
    mdarray<double_complex, 1> phi2_bt(gvec_r.partition().gvec_count_fft());
    fft.transform<-1>(&phi1_bt(0), &phi2_bt(0));

    double diff = 0;
    for (int i = 0; i < gvec_r.partition().gvec_count_fft(); i++) {
        diff += std::abs(phi1(i) - phi1_bt(i));
        diff += std::abs(phi2(i) - phi2_bt(i));
    }
    diff /= gvec_r.partition().gvec_count_fft();
    printf("diff: %18.10f\n", diff);
    if (diff > 1e-13) {
        printf("functions are different\n");
        exit(1);
    }

    fft.dismiss();
}

#ifdef __GPU
void test3(vector3d<int> const& dims__, double cutoff__)
{
    printf("test3\n");
    matrix3d<double> M;
    M(0, 0) = M(1, 1) = M(2, 2) = 1.0;

    FFT3D fft(find_translations(cutoff__, M), mpi_comm_world(), GPU);

    Gvec gvec_r(M, cutoff__, mpi_comm_world(), mpi_comm_world(), true);

    fft.prepare(gvec_r.partition());

    mdarray<double_complex, 1> phi1(gvec_r.partition().gvec_count_fft(), memory_t::host | memory_t::device);
    mdarray<double_complex, 1> phi2(gvec_r.partition().gvec_count_fft(), memory_t::host | memory_t::device);
    mdarray<double_complex, 1> phi1_rg(fft.local_size());
    mdarray<double_complex, 1> phi2_rg(fft.local_size());

    for (int i = 0; i < gvec_r.partition().gvec_count_fft(); i++) {
        phi1(i) = type_wrapper<double_complex>::random();
        phi2(i) = type_wrapper<double_complex>::random();
    }
    phi1(0) = 1.0;
    phi2(0) = 1.0;

    phi1.copy<memory_t::host, memory_t::device>();
    phi2.copy<memory_t::host, memory_t::device>();

    fft.transform<1, GPU>(phi1.at<GPU>());
    fft.output(&phi1_rg(0));

    fft.transform<1, GPU>(phi2.at<GPU>());
    fft.output(&phi2_rg(0));

    for (int i = 0; i < fft.local_size(); i++) {
        if (phi1_rg(i).imag() > 1e-10) {
            printf("phi1 is not real at idx = %i, image value: %18.12f\n", i, phi1_rg(i).imag());
            exit(1);
        }
        if (phi2_rg(i).imag() > 1e-10) {
            printf("phi2 is not real at idx = %i, image value: %18.12f\n", i, phi2_rg(i).imag());
            exit(1);
        }
    }

    fft.transform<1, GPU>(phi1.at<GPU>(), phi2.at<GPU>());

    mdarray<double_complex, 1> phi12_rg(fft.local_size());
    fft.output(&phi12_rg(0));

    //printf("phi1(0)=%18.10f\n", phi1_rg(0));
    //printf("phi2(0)=%18.10f\n", phi2_rg(0));

    for (int i = 0; i < fft.local_size(); i++) {
        if (std::abs(double_complex(phi1_rg(i).real(), phi2_rg(i).real()) - phi12_rg(i)) > 1e-10) {
            printf("functions don't match\n");
            printf("phi1: %18.10f\n", phi1_rg(i).real());
            printf("phi2: %18.10f\n", phi2_rg(i).real());
            printf("complex phi: %18.10f %18.10f\n", fft.buffer(i).real(), fft.buffer(i).imag());
            exit(1);
        }
    }

    mdarray<double_complex, 1> phi1_bt(gvec_r.partition().gvec_count_fft(), memory_t::host | memory_t::device);
    mdarray<double_complex, 1> phi2_bt(gvec_r.partition().gvec_count_fft(), memory_t::host | memory_t::device);
    fft.transform<-1, GPU>(phi1_bt.at<GPU>(), phi2_bt.at<GPU>());

    phi1_bt.copy<memory_t::device, memory_t::host>();
    phi2_bt.copy<memory_t::device, memory_t::host>();

    double diff{0};
    for (int i = 0; i < gvec_r.partition().gvec_count_fft(); i++) {
        diff += std::abs(phi1(i) - phi1_bt(i));
        diff += std::abs(phi2(i) - phi2_bt(i));
    }
    diff /= gvec_r.partition().gvec_count_fft();
    printf("diff: %18.10f\n", diff);
    if (diff > 1e-13) {
        printf("functions are different\n");
        exit(1);
    }

    fft.dismiss();
}
#endif

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

    std::vector<int> vd = args.value< std::vector<int> >("dims", {132, 132, 132});
    vector3d<int> dims(vd[0], vd[1], vd[2]); 
    double cutoff = args.value<double>("cutoff", 50);

    sirius::initialize(1);

    test1(dims, cutoff, CPU);
    test2(dims, cutoff, CPU);
    #ifdef __GPU
    test1(dims, cutoff, GPU);
    test2(dims, cutoff, GPU);
    test3(dims, cutoff);
    #endif
    
    sirius::finalize();

    return 0;
}
