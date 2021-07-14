#include <sirius.hpp>

using namespace sirius;

int run_test()
{
    vector3d<double> a(1.1, 2.2, 3.3);
    vector3d<double> b = a;
    vector3d<double> c(b);
    vector3d<double> d;
    d = c;
    matrix3d<int> R = {{1,0,0},{0,1,0},{0,0,1}};
    auto e = dot(R, a) + b;
    vector3d<double> ref(2.2, 4.4, 6.6);
    if ((ref - e).length() > 1e-16) {
        return 1;
    }
    auto x = a + vector3d<int>(4, 5, 6);
    vector3d<double> ref1(5.1, 7.2, 9.3);
    if ((ref1 - x).length() > 1e-16) {
        return 2;
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
