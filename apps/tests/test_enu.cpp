#include "sirius.hpp"
#include "testing.hpp"

using namespace sirius;

int
test_enu(cmd_args const& args__)
{
    auto rel = get_relativity_t(args__.value<std::string>("rel", "none"));
    auto zn  = args__.value<int>("zn", 1);
    auto l   = args__.value<int>("l", 0);
    auto n   = args__.value<int>("n", 1);
    auto R   = args__.value<double>("R", 2.2);

    auto rgrid = Radial_grid_factory<double>(radial_grid_t::lin_exp, 1500, 1e-7, R, 6.0);
    std::vector<double> v(rgrid.num_points());
    for (int ir = 0; ir < rgrid.num_points(); ir++) {
        v[ir] = -double(zn) / rgrid[ir];
    }

    Enu_finder e(rel, zn, n, l, rgrid, v, -0.1);
    auto enu_ref = e.enu();

    #pragma omp parallel for
    for (int i = 0; i < 100; i++) {
        auto enu1 = Enu_finder(rel, zn, n, l, rgrid, v, -0.1).enu();
        if (enu1 != enu_ref) {
            std::cout << "wrong enu : " << enu1 << " " << enu_ref << std::endl;
        }
    }

    printf("Z: %i n: %i l: %i band energies (bottom, top, enu): %12.6f %12.6f %12.6f\n", zn, n, l, e.ebot(), e.etop(),
           e.enu());

    Radial_solver solver(zn, v, rgrid);

    std::vector<double> p1, p2, p3, rdudr;
    std::array<double, 2> uderiv;

    int dme{0};

    solver.solve(rel, dme, l, e.ebot(), p1, rdudr, uderiv);
    solver.solve(rel, dme, l, e.etop(), p2, rdudr, uderiv);
    solver.solve(rel, dme, l, e.enu(), p3, rdudr, uderiv);

    printf("uderiv: %12.6f %12.6f\n", uderiv[0], uderiv[1]);

    FILE* fout = fopen("radial_solution.dat", "w");
    for (int i = 0; i < rgrid.num_points(); i++) {
        double x = rgrid[i];
        fprintf(fout, "%18.12f %18.12f %18.12f %18.12f\n", x, p1[i] / x, p2[i] / x, p3[i] / x);
    }
    fclose(fout);
    return 0;
}

int
main(int argn, char** argv)
{
    cmd_args args(argn, argv,
                  {{"rel=", "(string) type of scalar-relativistic equation"},
                   {"zn=", "(int) nuclear charge"},
                   {"l=", "(int) orbital quantum number"},
                   {"n=", "(int) principal quantum number"},
                   {"R=", "(double) muffin-tin radius"}});

    sirius::initialize(1);
    call_test("test_enu", test_enu, args);
    sirius::finalize();
}
