#include "../../lib/sirius.h"

int main(int argn, char **argv)
{
    sirius::radial_grid r(sirius::linear_exponential_grid, 30000, 1e-8, 100.0);
    r.print_info();

    std::vector<double> v(r.size());
    std::vector<double> p(r.size());
    std::vector<double> hp(r.size());

    sirius::radial_solver solver(false, -1.0, r);

    for (int i = 0; i < r.size(); i++)
    {
        v[i] = -1.0 / r[i];
    }
    
    double enu = -0.1;
    for (int n = 1; n <=5; n++)
    {
        for (int l = 0; l <= n - 1; l++)
        {
            solver.bound_state(n, l, enu, v, p);
            std::cout << "energy - exact " << enu + 1.0 / pow(n, 2) / 2 << std::endl;
        }
    }
}
