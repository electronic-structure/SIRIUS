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

    Platform::initialize(1);
    Platform::finalize();
}
