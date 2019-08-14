#include <sirius.h>

using namespace sirius;

//void test()
//{
//    FFT3D fft({48, 48, 48}, Communicator::world(), device_t::CPU);
//    matrix3d<double> M = {{1, 0, 0},
//                          {0, 1, 0},
//                          {0, 0, 1}};
//    double Gmax{20};
//    Gvec gv1(M, Gmax, Communicator::world(), true);
//    Gvec gv2(M, 1e10, fft, Communicator::world(), false);
//    if (gv2.num_gvec() != fft.size()) {
//        TERMINATE("wrong number of G-vectors");
//    }
//
//    Gvec_partition gvp1(gv1, fft.comm(), Communicator::self());
//    Gvec_partition gvp2(gv2, fft.comm(), Communicator::self());
//
//    std::cout << "grid size: " << fft.size(0) << " " << fft.size(1) << " " << fft.size(2) << "\n";
//    std::cout << "ngv1: " << gv1.num_gvec() << ", ngv2: " << gv2.num_gvec() << "\n";
//
//    Smooth_periodic_function<double> f1(fft, gvp1);
//    Smooth_periodic_function<double> f2(fft, gvp2);
//
//    for (int igloc = 0; igloc < gv1.count(); igloc++) {
//        f1.f_pw_local(igloc) = utils::random<double_complex>();
//    }
//    f1.f_pw_local(0) = 1.0;
//
//    fft.prepare(gvp1);
//    f1.fft_transform(1);
//    fft.dismiss();
//
//    for (int ir = 0; ir < fft.local_size(); ir++) {
//        f2.f_rg(ir) = f1.f_rg(ir);
//    }
//
//    fft.prepare(gvp2);
//    f2.fft_transform(-1);
//
//    for (int igloc = 9; igloc < gv2.count(); igloc++) {
//        auto gvc = gv2.gvec_cart<index_domain_t::local>(igloc);
//        if (gvc.length() > Gmax && std::abs(f2.f_pw_local(igloc)) > 1e-12) {
//            std::cout << "g=" << gvc.length() << ", fpw=" << std::abs(f2.f_pw_local(igloc)) << "\n";
//        }
//    }
//}
//
//void test2()
//{
//    FFT3D fft({45, 45, 45}, Communicator::world(), device_t::CPU);
//    matrix3d<double> M = {{1, 0, 0},
//                          {0, 1, 0},
//                          {0, 0, 1}};
//    Gvec gv(M, 1e10, fft, Communicator::world(), true);
//    if (gv.num_gvec() * 2 - 1 != fft.size()) {
//        std::stringstream s;
//        s << "wrong number of G-vectors\n"
//          << "  expected: " << fft.size() << "\n"
//          << "  number of reduced: " << gv.num_gvec();
//        TERMINATE(s);
//    }
//}

int main(int argn, char** argv)
{
    cmd_args args;

    args.parse_args(argn, argv);
    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }

    sirius::initialize(1);
    STOP();
    //test();
    //test2();
    sirius::finalize();
}
