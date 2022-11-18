#include <sirius.hpp>
#include <testing.hpp>
#include "gpu/cusolver.hpp"

/* template for unit tests */

using namespace sirius;

int run_test(cmd_args const& args)
{
    std::vector<int> sizes({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 100, 1000});

    using type = std::complex<double>;

    for (auto n : sizes) {
        auto M = random_positive_definite<type>(n);
        M.allocate(get_memory_pool(sddk::memory_t::device)).copy_to(sddk::memory_t::device);
        std::cout << "n = " << n << std::endl;
        auto info = cusolver::potrf(n, M.at(sddk::memory_t::device), M.ld());
        if (info) {
            return info;
        }
    }
    return 0;
}

int main(int argn, char** argv)
{
    cmd_args args;

    args.parse_args(argn, argv);

    sirius::initialize(true);
    auto result = call_test("test_potrf", run_test, args);
    sirius::finalize();
    return result;
}
