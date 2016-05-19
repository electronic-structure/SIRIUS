#include <sirius.h>

using namespace sirius;

void test_enu()
{
    Radial_grid rgrid(exponential_grid, 1500, 1e-7, 2.0);
    int zn{29};
    int n{5};
    int l{2};
    std::vector<double> v(rgrid.num_points());
    for (int ir = 0; ir < rgrid.num_points(); ir++) v[ir] = -double(zn) / rgrid[ir];

    Enu_finder e(relativity_t::none, zn, n, l, rgrid, v, -0.1);

    printf("band energies (bottom, top, enu): %12.6f %12.6f %12.6f\n", e.ebot(), e.etop(), e.enu());

    Radial_solver solver(zn, v, rgrid);
    
    std::vector<double> p1, p2, p3, rdudr;
    std::array<double, 2> uderiv;

    solver.solve(relativity_t::none, 0, l, e.ebot(), p1, rdudr, uderiv);
    solver.solve(relativity_t::none, 0, l, e.etop(), p2, rdudr, uderiv);
    solver.solve(relativity_t::none, 0, l, e.enu(), p3, rdudr, uderiv);

    printf("uderiv: %12.6f %12.6f\n", uderiv[0], uderiv[1]);

    FILE* fout = fopen("radial_solution.dat", "w");
    for (int i = 0; i < rgrid.num_points(); i++)
    {
        double x = rgrid[i];
        fprintf(fout, "%18.12f %18.12f %18.12f %18.12f\n", x, p1[i] / x, p2[i] / x, p3[i] / x);
    }
    fclose(fout);
}


int main(int argn, char** argv)
{
    sirius::initialize(1);
    test_enu();
    sirius::finalize();
}
