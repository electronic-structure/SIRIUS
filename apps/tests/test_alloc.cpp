#include <sirius.hpp>

using namespace sirius;

template <int touch, int pin, sddk::device_t pu>
void test_alloc(int size__)
{
    auto t0 = utils::time_now();
    if (pu == sddk::device_t::CPU) {
        sddk::mdarray<char, 1> a(1024 * 1024 * size__, pin ? sddk::memory_t::host_pinned : sddk::memory_t::host);
        if (touch) {
            a.zero();
        }
    }
#if defined(SIRIUS_GPU)
    if (pu == sddk::device_t::GPU) {
        sddk::mdarray<char, 1> a(nullptr, 1024 * 1024 * size__);
        a.allocate(sddk::memory_t::device);
        if (touch) {
            a.zero(sddk::memory_t::device);
        }
    }
    #endif
    double tval = utils::time_interval(t0);
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
    test_alloc<0, 0, sddk::device_t::CPU>(1024);
    printf("--- allocate on host, don't pin, touch\n");
    test_alloc<1, 0, sddk::device_t::CPU>(1024);
    printf("--- allocate on host, pin, don't touch\n");
    test_alloc<0, 1, sddk::device_t::CPU>(1024);
    printf("--- allocate on host, pin, touch\n");
    test_alloc<1, 1, sddk::device_t::CPU>(1024);
#if defined(SIRIUS_GPU)
    printf("--- allocate on device, don't touch\n");
    test_alloc<0, 0, sddk::device_t::GPU>(512);
    printf("--- allocate on device, touch\n");
    test_alloc<1, 0, sddk::device_t::GPU>(512);
#endif
    sirius::finalize();
}
