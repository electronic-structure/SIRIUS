#include <sirius.h>

using namespace sirius;

int main(int argn, char** argv)
{
    sirius::initialize(1);

    printf("sizeof(double): %lu\n", sizeof(double));
    printf("sizeof(long double): %lu\n", sizeof(long double));
    
    long double PI = 3.141592653589793238462643383279502884197L;

    printf("diff (in long double): %40.30Lf\n", std::abs(PI - std::acos(static_cast<long double>(-1))));
    printf("diff (in double): %40.30f\n", std::abs(static_cast<double>(PI) - std::acos(static_cast<double>(-1))));

    sirius::finalize();
    return 0;
}
