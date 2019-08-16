#include <sirius.h>
#include <thread>

/* test transformation of real function */

using namespace sirius;

int run_test(cmd_args& args, device_t pu__)
{
    //std::vector<int> vd = args.value< std::vector<int> >("dims", {132, 132, 132});
    //vector3d<int> dims(vd[0], vd[1], vd[2]); 
    //double cutoff = args.value<double>("cutoff", 50);

    //matrix3d<double> M;
    //M(0, 0) = M(1, 1) = M(2, 2) = 1.0;

    //FFT3D fft(find_translations(cutoff, M), Communicator::world(), pu__);

    ////Gvec gvec(M, cutoff, Communicator::world(), false);
    ////Gvec_partition gvecp(gvec, Communicator::world(), Communicator::self());

    //Gvec gvec_r(M, cutoff, Communicator::world(), true);
    //Gvec_partition gvecp_r(gvec_r, Communicator::world(), Communicator::self());

    //spfft::Grid spfft_grid(fft.size(0), fft.size(1), fft.size(2), gvecp.zcol_count_fft(), fft.local_size_z(),
    //                       SPFFT_PU_HOST, -1, fft.comm().mpi_comm(), SPFFT_EXCH_DEFAULT);

    //const auto fft_type = gvec.reduced() ? SPFFT_TRANS_R2C : SPFFT_TRANS_C2C;

    ////spfft::Transform spfft(spfft_grid.create_transform(SPFFT_PU_HOST, fft_type, fft.size(0), fft.size(1), fft.size(2),
    ////    fft.local_size_z(), gvecp.gvec_count_fft(), SPFFT_INDEX_TRIPLETS,
    ////    gvecp.gvec_coord().at(memory_t::host)));

    //spfft::Transform spfft_r(spfft_grid.create_transform(SPFFT_PU_HOST, fft_type, fft.size(0), fft.size(1), fft.size(2),
    //    fft.local_size_z(), gvecp_r.gvec_count_fft(), SPFFT_INDEX_TRIPLETS,
    //    gvecp_r.gvec_coord().at(memory_t::host)));


    //mdarray<double_complex, 1> phi(gvecp_r.gvec_count_fft());
    //for (int i = 0; i < gvecp_r.gvec_count_fft(); i++) {
    //    phi(i) = utils::random<double_complex>();
    //}
    ///phi(0) = 1.0;
    ////fft.transform<1>(&phi[0]);
    ////if (pu__ == device_t::GPU) {
    ////    fft.buffer().copy_to(memory_t::host);
    ////}

    ////for (int i = 0; i < fft.local_size(); i++) {
    ////    if (fft.buffer(i).imag() > 1e-10) {
    ////        return 1;
    ////    }
    ////}

    ////fft.dismiss();
    return 0;
}

int main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--dims=", "{vector3d<int>} FFT dimensions");
    args.register_key("--cutoff=", "{double} cutoff radius in G-space");

    args.parse_args(argn, argv);
    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }

    sirius::initialize(true);
    printf("running %-30s : ", argv[0]);
    int result = run_test(args, device_t::CPU);
    if (result) {
        printf("\x1b[31m" "Failed" "\x1b[0m" "\n");
    } else {
        printf("\x1b[32m" "OK" "\x1b[0m" "\n");
    }
    sirius::finalize();

    return result;
}
