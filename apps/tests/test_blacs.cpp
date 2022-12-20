#include <sirius.hpp>

int main(int argn, char** argv)
{
    sirius::initialize(true);

#if defined(SIRIUS_SCALAPACK)
    std::cout << mpi::Communicator::self().size() << " " << mpi::Communicator::self().rank() << std::endl;
    std::cout << mpi::Communicator::world().size() << " " << mpi::Communicator::world().rank() << std::endl;

    auto blacs_handler = sddk::linalg_base::create_blacs_handler(mpi::Communicator::self().mpi_comm());
    blacs_handler = sddk::linalg_base::create_blacs_handler(mpi::Communicator::world().mpi_comm());
    std::cout << blacs_handler << std::endl;

    sirius::finalize(true);
#endif
    return 0;
}
