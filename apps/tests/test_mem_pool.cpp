#include <sirius.h>

using namespace sirius;

void test(int nGb, int gran, memory_t M__)
{
    std::unique_ptr<memory_pool> mp;
    size_t initial_size{0};
    if (M__ == memory_t::host_pinned) {
        initial_size = size_t(4) * (1 << 30);
    } else {
        initial_size = size_t(0) * (1 << 30);
    }
    memory_pool mpool(M__, initial_size);

    std::vector<uint32_t> sizes;
    size_t tot_size{0};

    while (tot_size < size_t(nGb) * (1 << 30)) {
        auto s = std::max(utils::rnd() % (size_t(gran) * (1 << 20)), size_t(1));
        sizes.push_back(s);
        tot_size += s;
    }

    std::cout << "number of memory blocks: " << sizes.size() << "\n";
    std::cout << "total size: " << tot_size << "\n";

    std::vector<mdarray<char, 1>> v;
    for (int k = 0; k < 4; k++) {
        auto t = -utils::wtime();
        v.clear();
        for (auto s: sizes) {
            v.push_back(std::move(mdarray<char, 1>(s, mpool)));
            v.back().zero(M__);
        }
        std::random_shuffle(v.begin(), v.end());
        for (auto& e: v) {
            e.deallocate(M__);
        }

        if (initial_size == 0 && (mpool.total_size() != tot_size)) {
            throw std::runtime_error("wrong total size");
        }
        if (mpool.free_size() != mpool.total_size()) {
            throw std::runtime_error("wrong free size");
        }
        if (mpool.num_blocks() != 1) {
            throw std::runtime_error("wrong number of blocks");
        }
        if (mpool.num_stored_ptr() != 0) {
            throw std::runtime_error("wrong number of stored pointers");
        }
        t += utils::wtime();
        std::cout << "pass : " << k << ", time : " << t << "\n";
    }
}

int main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--memory_t=", "{string} type of the memory");
    args.register_key("--nGb=", "{int} total number of Gigabytes to allocate");
    args.register_key("--gran=", "{int} block granularity in Mb");

    args.parse_args(argn, argv);
    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }

    sirius::initialize(1);
    test(args.value<int>("nGb", 2), args.value<int>("gran", 32), get_memory_t(args.value<std::string>("memory_t", "host")));
    sirius::finalize();
}
