#include <sirius.h>

using namespace sirius;

void test_radial_solver()
{
    Radial_grid_lin_exp<double> rgrid(1500, 1e-7, 2.0);
    int zn{38};
    std::vector<double> v(rgrid.num_points());
    for (int ir = 0; ir < rgrid.num_points(); ir++) {
        v[ir] = -zn * rgrid.x_inv(ir);
    }

    Radial_solver rsolver(zn, v, rgrid);
    std::vector<double> p, rdudr;
    std::array<double, 2> uderiv;
    rsolver.solve(relativity_t::iora, 1, 2, -0.524233, p, rdudr, uderiv);

    std::stringstream s;
    s << "radial_functions.dat";
    FILE* fout = fopen(s.str().c_str(), "w");

    for (int ir = 0; ir < rgrid.num_points(); ir++) {
        fprintf(fout, "%f ", rgrid[ir]);
        fprintf(fout, "%f ", p[ir]);
        fprintf(fout, "%f ", rdudr[ir]);
        fprintf(fout, "\n");
    }
    fclose(fout);
}

int main(int argn, char** argv)
{
    cmd_args args;

    args.parse_args(argn, argv);
    if (args.exist("help"))
    {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }

    sirius::initialize(1);
    test_radial_solver();
    sirius::finalize();
}
