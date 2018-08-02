#include <sirius.h>

using namespace sirius;

void test1()
{
    int N = 10000000;
    std::vector<vector3d<double>> a(N);

    for (int i = 0; i < N; i++) {
        double r = type_wrapper<double>::random();
        a[i] = {r, r, r};
    }
    std::vector<double_complex> phase(N, 0);
    utils::timer t1("phase");
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        phase[i] = std::exp(double_complex(0, dot(a[i], a[i])));
    }
    double tval = t1.stop();
    printf("speed: %f million phase-factors / sec.\n", N / tval / 1000000);

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

    sirius::initialize(1);
    test1();
    utils::timer::print();
    sirius::finalize();
}
