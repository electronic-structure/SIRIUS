#include <sirius.h>

using namespace sirius;

#ifdef __GPU
void test_gpu(int N)
{
    mdarray<char, 1> buf(N * 1024);

    for (size_t i = 0; i < buf.size(); i++) buf(i) = char(i % 255);

    DUMP("hash(buf): %llX", buf.hash());

    buf.allocate_on_device();
    buf.copy_to_device();
    buf.zero();
    buf.copy_to_host();

    DUMP("hash(buf): %llX", buf.hash());
}
#endif

int main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--N=", "{int} buffer size (Kb)");

    args.parse_args(argn, argv);
    if (argn == 1)
    {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        exit(0);
    }

    int N = args.value<int>("N");

    Platform::initialize(1);
    cuda_device_info();
    
    #ifdef __GPU
    test_gpu(N);
    #endif

    Platform::finalize();
}
