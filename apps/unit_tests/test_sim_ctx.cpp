#include <sirius.h>

using namespace sirius;

int run_test(cmd_args& args)
{
    Simulation_context ctx;
    ctx.import(args);
    std::cout << ctx.control().verbosity_ << "\n";
    return 0;
}

int main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--control.verbosity=", "{int} verbosity level");

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
