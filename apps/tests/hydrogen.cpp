#include <sirius.h>

using namespace sirius;

struct level_conf
{
    int n;
    int l;
    int z;

    level_conf(int n__, int l__, int z__) : n(n__), l(l__), z(z__)
    {
    }
};

int main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--grid_type=","{int} type of the radial grid");
    args.register_key("--num_points=","{int} number of grid points");
    args.register_key("--rmin=","{double} first grid point");
    args.parse_args(argn, argv);
    if (argn == 1)
    {
        printf("Usage: %s [options]", argv[0]);
        args.print_help();
        exit(0);
    }

    printf("\n");
    printf("Test of radial solver for bare nuclear potential V(r) = -z / r \n");
    printf("Energy of {n,l} states is En = -(1/2) * z^2 / n^2 \n");
    printf("\n");

    Platform::initialize(1);

    int num_points = 20000;
    if (args.exist("num_points")) num_points = args.value<int>("num_points");
    radial_grid_t grid_type = static_cast<radial_grid_t>(args.value<int>("grid_type"));

    double rmin = 1e-7;
    if (args.exist("rmin")) rmin = args.value<double>("rmin");

    std::vector<level_conf> levels;
    for (int k = 0; k < 10; k++)
    {
        int z = 1 + k * 10;

        for (int n = 1; n <= 5 + k; n++)
        {
            for (int l = 0; l <= n - 1; l++) levels.push_back(level_conf(n, l, z));
        }
    }

    {
        Radial_grid r(grid_type, num_points, rmin, 200.0);
        std::string fname = "grid_" + r.grid_type_name() + ".txt";
        FILE* fout = fopen(fname.c_str(), "w");
        for (int i = 0; i < r.num_points(); i++) fprintf(fout,"%i %16.12e\n", i, r[i]);
        fclose(fout);
    }

    std::vector<double> err(levels.size());
    
    #pragma omp parallel for
    for (int j = 0; j < (int)levels.size(); j++)
    {
        int n = levels[j].n;
        int l = levels[j].l;
        int z = levels[j].z;
        
        Radial_grid radial_grid(grid_type, num_points, rmin, 200.0 + z * 3.0);
        Radial_solver solver(false, -double(z), radial_grid);

        std::vector<double> v(radial_grid.num_points());
        std::vector<double> p(radial_grid.num_points());

        solver.set_tolerance(1e-12);

        for (int i = 0; i < radial_grid.num_points(); i++) v[i] = -z / radial_grid[i];
        
        double enu_exact = -0.5 * pow(double(z) / n, 2);
        double enu = solver.bound_state(n, l, enu_exact, v, p);
        double rel_err =  fabs(enu  - enu_exact) / fabs(enu_exact); 
        if (rel_err > 1e-10) 
        {
            printf("Fail! z = %i n = %i l = %i, relative error = %12.6e\n", z, n, l, rel_err);
        }
        err[j] = rel_err;
    }

    FILE* fout = fopen("err.dat", "w");
    for (int j = 0; j < (int)err.size(); j++)
    {
        fprintf(fout, "%i %20.16f\n", j, err[j]);
    }
    fclose(fout);


    



    
    //JSON_write jw("out.json");
    //std::vector<int> xaxis;
    //
    //int j = 0;
    //for (int n = 1; n <= 20; n++)
    //{
    //    for (int l = 0; l <= n - 1; l++)
    //    {
    //        xaxis.push_back(j);
    //        j++;
    //    }
    //}
    //jw.single("xaxis", xaxis);

    //std::vector<int> xaxis_ticks;
    //std::vector<std::string> xaxis_tick_labels;

    //j = 0;
    //for (int n = 1; n <= 20; n++)
    //{
    //    std::stringstream s;
    //    s << "n=" << n;
    //    xaxis_ticks.push_back(j);
    //    xaxis_tick_labels.push_back(s.str());
    //    j += n;
    //}

    //jw.single("xaxis_ticks", xaxis_ticks);
    //jw.single("xaxis_tick_labels", xaxis_tick_labels);
    //jw.begin_array("plot");

    //for (int k = 0; k < 10; k++)
    //{
    //    int z = 1 + k * 10;
    //    std::stringstream s;
    //    s << "z=" << z;
    //    
    //    jw.begin_set();
    //    jw.single("label", s.str());
    //    
    //    Radial_grid radial_grid(scaled_pow_grid, 20000, 1e-7, 200.0 + z * 3.0);
    //    //radial_grid.print_info();

    //    std::vector<double> v(radial_grid.num_points());
    //    std::vector<double> p(radial_grid.num_points());

    //    Radial_solver solver(false, -double(z), radial_grid);
    //    solver.set_tolerance(1e-13 * (k + 1));

    //    for (int i = 0; i < radial_grid.num_points(); i++) v[i] = -z / radial_grid[i];
    //    
    //    double enu = -0.1;

    //    j = 0;
    //    xaxis.clear();
    //    for (int n = 1; n <= 5 + k; n++)
    //    {
    //        for (int l = 0; l <= n - 1; l++)
    //        {
    //            xaxis.push_back(j);
    //            j++;
    //        }
    //    }
    //    jw.single("xaxis", xaxis);
    //    
    //    std::vector<double> yvalues;

    //    j = 0;
    //    for (int n = 1; n <= 5 + k; n++)
    //    {
    //        for (int l = 0; l <= n - 1; l++)
    //        {
    //            solver.bound_state(n, l, v, enu, p);
    //            double enu_exact = -0.5 * pow(double(z) / n, 2);
    //            double rel_err =  fabs(enu  - enu_exact) / fabs(enu_exact); 
    //            printf("z = %i n = %i l = %i, relative error = %12.6e", z, n, l, rel_err);
    //            if (rel_err < 1e-10) 
    //            {
    //                
    //                printf("  OK\n");
    //            }
    //            else
    //            {
    //                printf("  Fail\n");
    //            }
    //            yvalues.push_back(fabs(enu  - enu_exact) / fabs(enu_exact));
    //            j++;
    //        }
    //    }
    //    
    //    jw.single("yvalues", yvalues);
    //    jw.end_set();
    //}
    //jw.end_array();

    //printf("\n");
    //printf("Run 'python hydrogen_plot.py out.json' to produce a PDF plot with relative errors.\n");
}
