#include <sirius.h>

using namespace sirius;

double vlen(vector3d<double> v)
{
    return v.length();
}

int run_test()
{
    #pragma omp parallel 
    {
    vector3d<double> a(1, 2, 3);
    vector3d<double> b = a;
    vector3d<double> c(b);
    vector3d<double> d;
    d = c;
    matrix3d<double> R = {{1,0,0},{0,1,0},{0,0,1}};
    auto e = R * a + b;
    printf("%f\n", vlen(e));
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

    sirius::initialize(1);
    printf("%-30s", "testing geometry3d: ");
    int result = run_test();
    if (result) {
        printf("\x1b[31m" "Failed" "\x1b[0m" "\n");
    } else {
        printf("\x1b[32m" "OK" "\x1b[0m" "\n");
    }
    sirius::finalize();

    return 0;
}
