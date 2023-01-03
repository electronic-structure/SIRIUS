#include <sirius.hpp>

using namespace sirius;
using namespace mpi;
using namespace sddk;

int main(int argn, char** argv)
{
    sirius::initialize(true);

    bool pr = (Communicator::world().rank() == 0);

    sddk::mdarray<std::complex<double>, 2> tmp_d(100, 200);
    sddk::mdarray<std::complex<double>, 2> tmp(100, 200);

    std::cout << Communicator::world().rank() << " tmp_d " << tmp_d.at(sddk::memory_t::host) << std::endl;
    std::cout << Communicator::world().rank() << " tmp " << tmp.at(sddk::memory_t::host) << std::endl;

    if (pr) {
        std::cout << "allreduce<double> " << std::endl;
    }
    Communicator::world().allreduce(tmp_d.at(memory_t::host), static_cast<int>(tmp.size()));
    Communicator::world().barrier();
    if (pr) {
        std::cout << "allreduce<double> : OK" << std::endl;
    }

    if (pr) {
        std::cout << "reduce<double> " << std::endl;
    }
    Communicator::world().reduce(tmp_d.at(memory_t::host), static_cast<int>(tmp.size()), 1);
    Communicator::world().barrier();
    if (pr) {
        std::cout << "reduce<double> : OK" << std::endl;
    }

    if (pr) {
        std::cout << "allreduce<reinterpret_cast<std::complex<double>>> " << std::endl;
    }
    Communicator::world().allreduce(reinterpret_cast<double*>(tmp.at(memory_t::host)), 2 * static_cast<int>(tmp.size()));

    if (pr) {
        std::cout << "reduce<reinterpret_cast<std::complex<double>>> " << std::endl;
    }
    Communicator::world().reduce(reinterpret_cast<double*>(tmp.at(memory_t::host)), 2 * static_cast<int>(tmp.size()), 1);

    if (pr) {
        std::cout << "allreduce<std::complex<double>> " << std::endl;
    }
    Communicator::world().allreduce(tmp.at(memory_t::host), static_cast<int>(tmp.size()));

    if (pr) {
        std::cout << "reduce<std::complex<double>> " << std::endl;
    }
    Communicator::world().reduce(tmp.at(memory_t::host), static_cast<int>(tmp.size()), 1);

    sirius::finalize(true);

    return 0;
}
