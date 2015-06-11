#include <sirius.h>

using namespace sirius;

int main(int argn, char** argv)
{
    Platform::initialize(1);

    std::stringstream s;
    for (int i = 0; i <= Platform::mpi_rank(); i++) s << "Hello from rank : " << Platform::mpi_rank() << std::endl;
    pstdout(s.str());

    Platform::finalize();
}
