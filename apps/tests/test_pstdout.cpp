#include <sirius.h>

using namespace sirius;

int main(int argn, char** argv)
{
    sirius::initialize(1);
    
    {
        pstdout pout(mpi_comm_world());
        pout.printf("Hello from rank : %i\n", mpi_comm_world().rank());
    }

    sirius::finalize();

    return 0;
}
