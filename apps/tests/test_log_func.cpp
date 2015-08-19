#include <sirius.h>

using namespace sirius;

void f2()
{
    LOG_FUNC();
    
    debug::log_function::stack_trace();
}

void f1()
{
    LOG_FUNC();
    f2();
}

int main(int argn, char** argv)
{
    Platform::initialize(1);
    f1();
    Platform::finalize();
}
