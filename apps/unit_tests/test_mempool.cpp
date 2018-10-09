#include <sirius.h>

using namespace sirius;

void test1()
{
    memory_pool mp;
}

void test2()
{
    memory_pool mp;
    mp.allocate<double_complex, memory_t::host>(1024);
    mp.reset<memory_t::host>();
}

void test3()
{
    memory_pool mp;
    mp.allocate<double_complex, memory_t::host>(1024);
    mp.allocate<double_complex, memory_t::host>(2024);
    mp.allocate<double_complex, memory_t::host>(3024);
    mp.reset<memory_t::host>();
}

void test4()
{
    memory_pool mp;
    mp.allocate<double_complex, memory_t::host>(1024);
    mp.reset<memory_t::host>();
    mp.allocate<double_complex, memory_t::host>(1024);
    mp.allocate<double_complex, memory_t::host>(1024);
    mp.reset<memory_t::host>();
    mp.allocate<double_complex, memory_t::host>(1024);
    mp.allocate<double_complex, memory_t::host>(1024);
    mp.allocate<double_complex, memory_t::host>(1024);
    mp.reset<memory_t::host>();
}

void test5()
{
    memory_pool mp;

    for (int k = 0; k < 2; k++) {
        std::vector<double*> vp;
        for (size_t i = 1; i < 20; i++) {
            size_t sz = 1 << i;
            double* ptr = mp.allocate<double, memory_t::host>(sz);
            ptr[0] = 0;
            ptr[sz - 1] = 0;
            vp.push_back(ptr);
        }
        for (auto& e: vp) {
            mp.free<memory_t::host>(e);
        }
    }
}

int run_test()
{
    test1();
    test2();
    test3();
    test4();
    test5();
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
    printf("%-30s", "testing memory pool: ");
    int result = run_test();
    if (result) {
        printf("\x1b[31m" "Failed" "\x1b[0m" "\n");
    } else {
        printf("\x1b[32m" "OK" "\x1b[0m" "\n");
    }
    sirius::finalize();

    return 0;
}
