#include <cmath>
#include "utils/cmd_args.hpp"

/* test for NaN and IEEE arithmetics */

int run_test(cmd_args& args)
{
    double val = std::nan("");
    if (val != val) {
        return 0;
    } else {
        return 1;
    }
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

    printf("running %-30s : ", argv[0]);
    int result = run_test(args);
    if (result) {
        printf("\x1b[31m" "Failed" "\x1b[0m" "\n");
    } else {
        printf("\x1b[32m" "OK" "\x1b[0m" "\n");
    }

    return result;
}
