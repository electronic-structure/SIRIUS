#include <sirius.hpp>
#include <thread>

/* test G-vectors */

using namespace sirius;

int run_test(cmd_args& args)
{
    auto vd = args.value("dims", std::vector<int>({132, 132, 132}));
    r3::vector<int> dims(vd[0], vd[1], vd[2]); 
    double cutoff = args.value<double>("cutoff", 50);

    r3::matrix<double> M;
    M(0, 0) = M(1, 1) = M(2, 2) = 1.0;
    M(0, 1) = 0.1;
    M(0, 2) = 0.2;
    M(2, 0) = 0.3;

    sddk::Gvec gvec(M, cutoff, sddk::Communicator::world(), false);
    sddk::Gvec gvec_r(M, cutoff, sddk::Communicator::world(), true);

    if (gvec_r.num_gvec() * 2 != gvec.num_gvec() + 1) {
        return 1;
    }
    return 0;
}

int main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--dims=", "{vector<int>} FFT dimensions");
    args.register_key("--cutoff=", "{double} cutoff radius in G-space");

    args.parse_args(argn, argv);
    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }

    sirius::initialize(true);
    printf("running %-30s : ", argv[0]);
    int result = run_test(args);
    if (result) {
        printf("\x1b[31m" "Failed" "\x1b[0m" "\n");
    } else {
        printf("\x1b[32m" "OK" "\x1b[0m" "\n");
    }
    sirius::finalize();

    return result;
}
