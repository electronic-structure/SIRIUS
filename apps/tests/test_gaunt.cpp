#include <sirius.h>

using namespace sirius;

int main(int argn, char** argv)
{
    cmd_args args;

    args.parse_args(argn, argv);
    if (args.exist("help"))
    {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }

    sirius::initialize(1);
    printf("%f\n", SHT::gaunt_rlm(2, 2, 2, 0, 0, 0));
    sirius::finalize();
}
