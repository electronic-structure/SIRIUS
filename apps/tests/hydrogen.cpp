#include <sirius.h>

int main(int argn, char **argv)
{
    FILE* fout = fopen("hydrogen.json", "w");
    fprintf(fout, "{\n");
    
    fprintf(fout, "  \"labels\" : [");
    int j = 0;
    for (int n = 1; n <= 20; n++)
    {
        for (int l = 0; l <= n - 1; l++)
        {
            if (j) fprintf(fout, ", ");
            fprintf(fout, "\"n=%i l=%i\"", n, l);
            j++;
        }
    }
    fprintf(fout,"], \n");

    fprintf(fout,"  \"plot\" : [\n");

    for (int k = 0; k < 9; k++)
    {
        int z = 1 + k * 10;

        sirius::RadialGrid radial_grid(sirius::exponential_grid, 15000 + z * 150,  1e-6 / z, 150.0 + z * 2.0);
        radial_grid.print_info();

        std::vector<double> v(radial_grid.size());
        std::vector<double> p(radial_grid.size());

        sirius::RadialSolver solver(false, -double(z), radial_grid);
        //solver.set_tolerance(1e-11);

        for (int i = 0; i < radial_grid.size(); i++)
        {
            v[i] = -z / radial_grid[i];
        }
        
        double enu = -0.1;
        if (k) fprintf(fout, ", \n");
        fprintf(fout,"    {\"z\" : %i, \"values\" : [", z);
        j = 0;
        for (int n = 1; n <= 5 + k; n++)
        {
            for (int l = 0; l <= n - 1; l++)
            {
                solver.bound_state(n, l, v, enu, p);
                double enu_exact = -0.5 * (z * z) / pow(double(n), 2);
                if (j) fprintf(fout, ", ");
                printf("z = %i n = %i l = %i  err = %12.6e\n", z, n, l, fabs(enu  - enu_exact) / fabs(enu_exact));
                fprintf(fout, "%18.12e", fabs(enu - enu_exact) / fabs(enu_exact));
                j++;
            }
        }
        fprintf(fout, "]}");
    }
    fprintf(fout, "\n"); 
    fprintf(fout, "  ]\n");
    fprintf(fout, "}\n");
    
    fclose(fout);
}
