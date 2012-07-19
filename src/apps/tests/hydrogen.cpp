#include "../../lib/sirius.h"

int main(int argn, char **argv)
{
    sirius::RadialGrid radial_grid(sirius::linear_exponential_grid, 30000, 1e-8, 100.0);
    radial_grid.print_info();

    std::vector<double> v(radial_grid.size());
    std::vector<double> p(radial_grid.size());

    sirius::RadialSolver solver(false, -1.0, radial_grid);

    for (int i = 0; i < radial_grid.size(); i++)
    {
        v[i] = -1.0 / radial_grid[i];
    }
    
    double enu = -0.1;
    for (int n = 1; n <=5; n++)
    {
        for (int l = 0; l <= n - 1; l++)
        {
            solver.bound_state(n, l, v, enu, p);
            std::cout << "energy - exact " << enu + 1.0 / pow(n, 2) / 2 << std::endl;
        }
    }
    
}
