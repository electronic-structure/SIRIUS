#include <sirius.h>

using namespace sirius;

template <int touch, int pin, processing_unit_t pu>
void test_alloc(int size__)
{
    runtime::Timer t("alloc");
    if (pu == CPU) {
        mdarray<char, 1> a(1024 * 1024 * size__);
        #ifdef __GPU
        if (pin) {
            a.pin_memory();
        }
        #endif
        if (touch) {
            a.zero();
        }
    }
    #ifdef __GPU
    if (pu == GPU) {
        mdarray<char, 1> a(nullptr, 1024 * 1024 * size__);
        a.allocate_on_device();
        if (touch) {
            a.zero_on_device();
        }
    }
    #endif
    double tval = t.stop();
    printf("time: %f microseconds\n", tval * 1e6);
    printf("effective speed: %f GB/sec.\n", size__ / 1024.0 / tval);
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
    printf("--- allocate on host, don't pin, don't touch\n");
    test_alloc<0, 0, CPU>(1024);
    printf("--- allocate on host, don't pin, touch\n");
    test_alloc<1, 0, CPU>(1024);
    printf("--- allocate on host, pin, don't touch\n");
    test_alloc<0, 1, CPU>(1024);
    printf("--- allocate on host, pin, touch\n");
    test_alloc<1, 1, CPU>(1024);
    #ifdef __GPU
    printf("--- allocate on device, don't touch\n");
    test_alloc<0, 0, GPU>(512);
    printf("--- allocate on device, touch\n");
    test_alloc<1, 0, GPU>(512);
    #endif
    sirius::finalize();
}
