#include <sirius.h>

int main(int argn, char **argv)
{
    Platform::initialize(true);
    
    json_write jw("out.json");
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
        
        sirius::RadialGrid radial_grid(sirius::exponential_grid, 20000 + z * 300,  1e-7 / z, 200.0 + z * 3.0);
        radial_grid.print_info();

        std::vector<double> v(radial_grid.size());
        std::vector<double> p(radial_grid.size());

        sirius::RadialSolver solver(false, -double(z), radial_grid);
        solver.set_tolerance(1e-13 * (k + 1));

        for (int i = 0; i < radial_grid.size(); i++) v[i] = -z / radial_grid[i];
        
        double enu = -0.1;

        std::vector<double> yvalues;

        j = 0;
        for (int n = 1; n <= 5 + k; n++)
        {
            for (int l = 0; l <= n - 1; l++)
            {
                solver.bound_state(n, l, v, enu, p);
                double enu_exact = -0.5 * (z * z) / pow(double(n), 2);
                printf("z = %i n = %i l = %i  err = %12.6e\n", z, n, l, fabs(enu  - enu_exact) / fabs(enu_exact));
                yvalues.push_back(fabs(enu  - enu_exact) / fabs(enu_exact));
                j++;
            }
        }
        
        jw.single("yvalues", yvalues);
        jw.end_set();
    }
    jw.end_array();
}
