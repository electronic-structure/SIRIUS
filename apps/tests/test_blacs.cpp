#include <sirius.hpp>

using namespace sirius;

int main(int argn, char** argv)
{
    sirius::initialize(true);

#if defined(SIRIUS_SCALAPACK)
    std::cout << Communicator::self().size() << " " << Communicator::self().rank() << std::endl;
    std::cout << Communicator::world().size() << " " << Communicator::world().rank() << std::endl;

    auto blacs_handler = linalg_base::create_blacs_handler(Communicator::self().mpi_comm());
    blacs_handler = linalg_base::create_blacs_handler(Communicator::world().mpi_comm());

    sirius::finalize(true);
#endif

    return 0;
}
