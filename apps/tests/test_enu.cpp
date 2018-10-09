#include <sirius.h>

using namespace sirius;

void test_enu(int zn__, int n__, int l__, double R__)
{
    auto rgrid = Radial_grid_factory<double>(radial_grid_t::lin_exp, 1500, 1e-7, R__, 6.0);
    std::vector<double> v(rgrid.num_points());
    for (int ir = 0; ir < rgrid.num_points(); ir++) {
        v[ir] = -double(zn__) / rgrid[ir];
    }

    Enu_finder e(relativity_t::none, zn__, n__, l__, rgrid, v, -0.1);

    printf("Z: %i n: %i l: %i band energies (bottom, top, enu): %12.6f %12.6f %12.6f\n", zn__, n__, l__, e.ebot(), e.etop(), e.enu());

    Radial_solver solver(zn__, v, rgrid);
    
    std::vector<double> p1, p2, p3, rdudr;
    std::array<double, 2> uderiv;

    solver.solve(relativity_t::none, 0, l__, e.ebot(), p1, rdudr, uderiv);
    solver.solve(relativity_t::none, 0, l__, e.etop(), p2, rdudr, uderiv);
    solver.solve(relativity_t::none, 0, l__, e.enu(), p3, rdudr, uderiv);

    //printf("uderiv: %12.6f %12.6f\n", uderiv[0], uderiv[1]);

    //FILE* fout = fopen("radial_solution.dat", "w");
    //for (int i = 0; i < rgrid.num_points(); i++)
    //{
    //    double x = rgrid[i];
    //    fprintf(fout, "%18.12f %18.12f %18.12f %18.12f\n", x, p1[i] / x, p2[i] / x, p3[i] / x);
    //}
    //fclose(fout);
}


int main(int argn, char** argv)
{
    sirius::initialize(1);
    for (int zn = 1; zn < 100; zn++) {
        #pragma omp parallel for
        for (int n = 1; n < 7; n++) {
            for (int l = 0; l < n; l++) {
                test_enu(zn, n, l, 1.5);
            }
        }
    }
    sirius::finalize();
}
