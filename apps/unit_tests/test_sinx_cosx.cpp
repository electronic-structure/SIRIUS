#include <sirius.h>
#include <math.h>
#include <complex.h>

using namespace sirius;

int run_test(cmd_args& args)
{
    int n{10};

    for (int i = 0; i < 20; i++) {
        double phi = utils::random<double>() * fourpi;
        auto cosxn = sf::cosxn(n, phi);
        auto sinxn = sf::sinxn(n, phi);
        for (int l = 0; l < n; l++) {
            if (std::abs(cosxn[l] - std::cos((l + 1) * phi)) > 1e-12) {
                return 1;
            }
            if (std::abs(sinxn[l] - std::sin((l + 1) * phi)) > 1e-12) {
                return 2;
            }
        }
    }
    return 0;
}

int main(int argn, char** argv)
{
    cmd_args args;

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
