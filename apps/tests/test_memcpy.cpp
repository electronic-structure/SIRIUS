#include <sirius.h>

using namespace sirius;

int main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--help", "print this help and exit");

    args.parse_args(argn, argv);
    if (args.exist("help"))
    {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        exit(0);
    }

    sirius::initialize(1);
    int n = 10000000;
    std::vector<double_complex> v1(n, 1.0);
    std::vector<double_complex> v2(n, 2.0);

    double t = omp_get_wtime();
    sirius::memcpy_simple(&v1[0], &v2[0], n);
    t = (omp_get_wtime() - t);
    printf("bandwidth: %f GB/s \n", double(n * sizeof(double_complex)) / t / (1 << 30));


    sirius::finalize();
}
