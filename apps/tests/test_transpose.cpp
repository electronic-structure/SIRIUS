#include <sirius.h>

using namespace sirius;

void test_transpose(int num_gkvec, int num_bands)
{
    matrix<double_complex> a(num_gkvec, num_bands);

    for (int i = 0; i < num_bands; i++)
    {
        for (int j = 0; j < num_gkvec; j++) a(j, i) = type_wrapper<double_complex>::random();
    }
    
    Timer t("transpose");
    matrix<double_complex> b(num_bands, num_gkvec);
    for (int j = 0; j < num_gkvec; j++)
        for (int i = 0; i < num_bands; i++) 
            b(i, j) = a(j, i);
    double tval = t.stop();

    printf("time %12.8f (sec) \n", tval);
}

int main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--num_gkvec=", "{int} number of Gk-vectors");
    args.register_key("--num_bands=", "{int} number of bands");

    args.parse_args(argn, argv);
    if (argn == 1)
    {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        exit(0);
    }

    int num_gkvec = args.value<int>("num_gkvec");
    int num_bands = args.value<int>("num_bands");

    Platform::initialize(1);
    for (int i = 0; i < 10; i++) test_transpose(num_gkvec, num_bands);
    Platform::finalize();
}
