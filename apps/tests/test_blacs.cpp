#include <sirius.hpp>

int main(int argn, char** argv)
{
    sirius::initialize(true);

#if defined(SIRIUS_SCALAPACK)
    std::cout << sddk::Communicator::self().size() << " " << sddk::Communicator::self().rank() << std::endl;
    std::cout << sddk::Communicator::world().size() << " " << sddk::Communicator::world().rank() << std::endl;

    auto blacs_handler = sddk::linalg_base::create_blacs_handler(sddk::Communicator::self().mpi_comm());
    blacs_handler = sddk::linalg_base::create_blacs_handler(sddk::Communicator::world().mpi_comm());
    std::cout << blacs_handler << std::endl;

    sirius::finalize(true);
#endif
    return 0;
}
