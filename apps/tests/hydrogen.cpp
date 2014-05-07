#include <sirius.h>

using namespace sirius;

int main(int argn, char **argv)
{
    printf("\n");
    printf("Test of radial solver for bare nuclear potential V(r) = -z / r \n");
    printf("Energy of {n,l} states is En = -(1/2) * z^2 / n^2 \n");
    printf("\n");

    Platform::initialize(1);
    
    JSON_write jw("out.json");
    std::vector<int> xaxis;
    
    int j = 0;
    for (int n = 1; n <= 20; n++)
    {
        for (int l = 0; l <= n - 1; l++)
        {
            xaxis.push_back(j);
            j++;
        }
    }
    jw.single("xaxis", xaxis);

    std::vector<int> xaxis_ticks;
    std::vector<std::string> xaxis_tick_labels;

    j = 0;
    for (int n = 1; n <= 20; n++)
    {
        std::stringstream s;
        s << "n=" << n;
        xaxis_ticks.push_back(j);
        xaxis_tick_labels.push_back(s.str());
        j += n;
    }

    jw.single("xaxis_ticks", xaxis_ticks);
    jw.single("xaxis_tick_labels", xaxis_tick_labels);
    jw.begin_array("plot");

    for (int k = 0; k < 10; k++)
    {
        int z = 1 + k * 10;
        std::stringstream s;
        s << "z=" << z;
        
        jw.begin_set();
        jw.single("label", s.str());
        
        //Radial_grid radial_grid(linear_exponential_grid, 20000 + z * 300,  1e-7 / z, 200.0, 200.0 + z * 3.0);
        Radial_grid radial_grid(linear_exponential_grid, 20000,  1e-7 / z, 200.0, 200.0 + z * 3.0);
        radial_grid.print_info();

        std::vector<double> v(radial_grid.num_points());
        std::vector<double> p(radial_grid.num_points());

        Radial_solver solver(false, -double(z), radial_grid);
        solver.set_tolerance(1e-13 * (k + 1));

        for (int i = 0; i < radial_grid.num_points(); i++) v[i] = -z / radial_grid[i];
        
        double enu = -0.1;

        j = 0;
        xaxis.clear();
        for (int n = 1; n <= 5 + k; n++)
        {
            for (int l = 0; l <= n - 1; l++)
            {
                xaxis.push_back(j);
                j++;
            }
        }
        jw.single("xaxis", xaxis);
        
        std::vector<double> yvalues;

        j = 0;
        for (int n = 1; n <= 5 + k; n++)
        {
            for (int l = 0; l <= n - 1; l++)
            {
                solver.bound_state(n, l, v, enu, p);
                double enu_exact = -0.5 * pow(double(z) / n, 2);
                double rel_err =  fabs(enu  - enu_exact) / fabs(enu_exact); 
                printf("z = %i n = %i l = %i, relative error = %12.6e", z, n, l, rel_err);
                if (rel_err < 1e-10) 
                {
                    
                    printf("  OK\n");
                }
                else
                {
                    printf("  Fail\n");
                }
                yvalues.push_back(fabs(enu  - enu_exact) / fabs(enu_exact));
                j++;
            }
        }
        
        jw.single("yvalues", yvalues);
        jw.end_set();
    }
    jw.end_array();

    printf("\n");
    printf("Run 'python hydrogen_plot.py out.json' to produce a PDF plot with relative errors.\n");
}
