#include <sirius.hpp>

using namespace sirius;

int main(int argn, char** argv)
{
    sirius::initialize(1);
    pstdout pout(Communicator::world());
    pout << "Hello from rank : " << Communicator::world().rank() << std::endl;

    /* this is a collective operation */
    auto s = pout.get().str();

    if (Communicator::world().rank() == 0) {
        std::cout << s;
    }

    sirius::finalize();

    return 0;
}
