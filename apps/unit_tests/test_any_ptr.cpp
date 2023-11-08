#include <array>
#include "core/any_ptr.hpp"
#include "core/memory.hpp"
#include "testing.hpp"

using namespace sirius;

int run_test(cmd_args const& args)
{
    void* ptr = new any_ptr(new mdarray<int, 1>({100}, get_memory_pool(memory_t::host)));
    delete static_cast<any_ptr*>(ptr);
    return 0;
}

int main(int argn, char** argv)
{
    cmd_args args(argn, argv, { });

    return sirius::call_test(argv[0], run_test, args);
}
