#include <sirius.h>

using namespace sirius;

int main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--grid_type=","{int} type of the radial grid");
    args.register_key("--num_points=","{int} number of grid points");
    args.register_key("--rmin=","{double} first grid point");
    args.register_key("--zn=","{int} element number");
    args.parse_args(argn, argv);
    if (args.exist("help"))
    {
        printf("Usage: %s [options]", argv[0]);
        args.print_help();
        return 0;
    }

    printf("\n");
    printf("Test of radial solver for bare nuclear potential V(r) = -z / r \n");
    printf("\n");

    sirius::initialize(1);

    int num_points = args.value<int>("num_points", 20000);

    radial_grid_t grid_type = static_cast<radial_grid_t>(args.value<int>("grid_type"));

    double rmin = args.value<double>("rmin", 1e-7);

    int zn = args.value<int>("zn", 1);


    //== {
    //==     Radial_grid r(grid_type, num_points, rmin, 200.0);
    //==     std::string fname = "grid_" + r.grid_type_name() + ".txt";
    //==     FILE* fout = fopen(fname.c_str(), "w");
    //==     for (int i = 0; i < r.num_points(); i++) fprintf(fout,"%i %16.12e\n", i, r[i]);
    //==     fclose(fout);

    //==     printf("radial grid: %s\n", r.grid_type_name().c_str());
    //== }

    std::vector<double> err(28);

    Radial_grid radial_grid(grid_type, num_points, rmin, 50.0);

    printf("grid name: %s\n", radial_grid.grid_type_name().c_str());
    
    std::vector<double> v(radial_grid.num_points());
    for (int i = 0; i < radial_grid.num_points(); i++) v[i] = -zn / radial_grid[i];

    #pragma omp parallel for schedule(dynamic)
    for (int j = 0; j < 28; j++)
    {
        int n = atomic_conf[zn - 1][j][0];
        int l = atomic_conf[zn - 1][j][1];
        int k = atomic_conf[zn - 1][j][2];

        if (n == -1) continue;

        double kappa = (k == l) ? k : -k;

        double b = std::sqrt(std::pow(kappa, 2) - std::pow(zn / speed_of_light, 2));

        double enu_exact = std::pow(speed_of_light, 2) / std::sqrt(1 + std::pow(zn / speed_of_light, 2) / std::pow(n - std::abs(kappa) + b, 2)) - std::pow(speed_of_light, 2);
        
        Bound_state bound_state(relativity_t::dirac, zn, n, l, k, radial_grid, v, enu_exact);

        double enu = bound_state.enu();

        double abs_err = std::abs(enu - enu_exact);

        #pragma omp critical
        printf("z = %2i n = %2i l = %2i, k = %2i, enu: %18.12f, enu_exact: %18.12f, error: %18.12e\n", zn, n, l, k, enu, enu_exact, abs_err);

        err[j] = abs_err;
    }

    return 0;
}
