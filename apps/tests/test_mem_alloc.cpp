#include <sirius.h>

using namespace sirius;

void test(std::vector<int> sizes, memory_t M__)
{
    std::vector<char*> ptrs;
    for (auto sm: sizes) {
        auto s = sm * (size_t(1) << 20);
        auto t0 = utils::wtime();
        auto ptr = sddk::allocate<char>(s, M__);
        ptrs.push_back(ptr);
        if (is_host_memory(M__)) {
            std::fill(ptr, ptr + s, 0);
        } else {
#ifdef __GPU
            acc::zero(ptr, s);
#endif
        }
        auto t1 = utils::wtime();
        if (is_host_memory(M__)) {
            std::fill(ptr, ptr + s, 0);
        } else {
#ifdef __GPU
            acc::zero(ptr, s);
#endif
        }
        auto t2 = utils::wtime();
        //sddk::deallocate(ptr, M__);
        //auto t3 = utils::wtime();

        std::cout << "block size (Mb) : " << sm << ", alloc time : " << (t1 - t0) - (t2 - t1) << "\n";
        print_memory_usage(__FILE__, __LINE__);
    }
    for (auto p: ptrs) {
        sddk::deallocate(p, M__);
    }
}

int main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--memory_t=", "{string} type of the memory");
    args.register_key("--sizes=", "{vector} list of chunk sizes in Mb");

    args.parse_args(argn, argv);
    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }

    sirius::initialize(1);
    auto sizes = args.value<std::vector<int>>("sizes", std::vector<int>({1024}));
    test(sizes, get_memory_t(args.value<std::string>("memory_t", "host")));
    sirius::finalize();
}
