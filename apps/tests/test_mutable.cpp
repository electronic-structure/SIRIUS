#include <sirius.h>

using namespace sirius;

void f(mdarray<int, 1> const& a)
{
    a.allocate_on_device();

}

int main(int argn, char** argv)
{
    cmd_args args;

    args.parse_args(argn, argv);
    if (args.exist("help"))
    {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        exit(0);
    }

    sirius::initialize(1);

    mdarray<int, 1> a(100);
    f(a);

    sirius::finalize();
}
