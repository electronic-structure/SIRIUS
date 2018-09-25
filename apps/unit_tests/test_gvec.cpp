#include <sirius.h>
#include <thread>

/* test G-vectors */

using namespace sirius;

int run_test(cmd_args& args)
{
    std::vector<int> vd = args.value< std::vector<int> >("dims", {132, 132, 132});
    vector3d<int> dims(vd[0], vd[1], vd[2]); 
    double cutoff = args.value<double>("cutoff", 50);

    matrix3d<double> M;
    M(0, 0) = M(1, 1) = M(2, 2) = 1.0;
    M(0, 1) = 0.1;
    M(0, 2) = 0.2;
    M(2, 0) = 0.3;

    Gvec gvec(M, cutoff, Communicator::world(), false);
    Gvec gvec_r(M, cutoff, Communicator::world(), true);

    if (gvec_r.num_gvec() * 2 != gvec.num_gvec() + 1) {
        return 1;
    }
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
    int result = run_test(args);
    if (result) {
        printf("\x1b[31m" "Failed" "\x1b[0m" "\n");
    } else {
        printf("\x1b[32m" "OK" "\x1b[0m" "\n");
    }
    sirius::finalize();

    return result;
}
