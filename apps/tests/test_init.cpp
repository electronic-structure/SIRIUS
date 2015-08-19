#include <sirius.h>

using namespace sirius;

int main(int argn, char** argv)
{
    Platform::initialize(1);
    Platform::finalize();
}
