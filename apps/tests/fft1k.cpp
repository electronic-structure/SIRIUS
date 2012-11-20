#include "sirius.h"

using namespace sirius;

void test1()
{
    std::cout << "test1" << std::endl;
    double a1[] = {42, 42, 0};
    double a2[] = {42, 0, 42};
    double a3[] = {0, 42, 42};
    sirius::global.set_lattice_vectors(a1, a2, a3);
    sirius::global.set_pw_cutoff(12.0);
    sirius::global.reciprocal_lattice::init();
    sirius::global.reciprocal_lattice::print_info();
}

void test2()
{
    std::cout << "test2" << std::endl;
    double a1[] = {21, 21, 0};
    double a2[] = {21, 0, 21};
    double a3[] = {0, 21, 21};
    sirius::global.set_lattice_vectors(a1, a2, a3);
    sirius::global.set_pw_cutoff(12.0);
    sirius::global.reciprocal_lattice::init();
    sirius::global.reciprocal_lattice::print_info();
}

int main(int argn, char **argv)
{
    test1();
    sirius::Timer::print();
}
