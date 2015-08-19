#include <sirius.h>

using namespace sirius;

void test_enu()
{
    Radial_grid rgrid(exponential_grid, 300, 1e-7, 1.0);
    int zn = 29;
    std::vector<double> v(rgrid.num_points());
    for (int ir = 0; ir < rgrid.num_points(); ir++) v[ir] = -double(zn) / rgrid[ir];

    Enu_finder e(zn, 3, 2, rgrid, v, -0.1);

    printf("band energies (bottom, top, enu): %12.6f %12.6f %12.6f\n", e.ebot(), e.etop(), e.enu());
}


int main(int argn, char** argv)
{
    Platform::initialize(1);
    test_enu();
    Platform::finalize();
}
