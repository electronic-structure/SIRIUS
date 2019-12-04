#include <sirius.h>
#include <math.h>
#include <complex.h>

using namespace sirius;

int run_test(cmd_args& args)
{
    int num_points = 500;
    mdarray<double, 2> tp(2, num_points);

    tp(0, 0) = pi;
    tp(1, 0) = 0;

    for (int k = 1; k < num_points - 1; k++) {
        double hk = -1.0 + double(2 * k) / double(num_points - 1);
        tp(0, k) = std::acos(hk);
        double t = tp(1, k - 1) + 3.80925122745582 / std::sqrt(double(num_points)) / std::sqrt(1 - hk * hk);
        tp(1, k) = std::fmod(t, twopi);
    }

    tp(0, num_points - 1) = 0;
    tp(1, num_points - 1) = 0;

    int lmax{10};
    std::vector<double> rlm((lmax + 1) * (lmax + 1));
    std::vector<double> rlm_ref((lmax + 1) * (lmax + 1));

    for (int k = 0; k < num_points; k++) {
        double theta = tp(0, k);
        double phi = tp(1, k);
        /* generate spherical harmonics */
        sht::spherical_harmonics(lmax, theta, phi, &rlm[0]);
        sht::spherical_harmonics_ref(lmax, theta, phi, &rlm_ref[0]);

        double diff{0};
        for (int l = 0; l <= lmax; l++) {
            for (int m = -l; m <= l; m++) {
                diff += std::abs(rlm[utils::lm(l, m)] - rlm_ref[utils::lm(l, m)]);
            }
        }
        if (diff > 1e-10) {
            return 1;
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
