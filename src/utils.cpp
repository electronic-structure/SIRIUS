#include "utils.h"

void Utils::write_matrix(const std::string& fname, mdarray<double_complex, 2>& matrix, int nrow, int ncol,
                         bool write_upper_only, bool write_abs_only, std::string fmt)
{
    static int icount = 0;

    if (nrow < 0 || nrow > matrix.size(0) || ncol < 0 || ncol > matrix.size(1))
        error_local(__FILE__, __LINE__, "wrong number of rows or columns");

    icount++;
    std::stringstream s;
    s << icount;
    std::string full_name = s.str() + "_" + fname;

    FILE* fout = fopen(full_name.c_str(), "w");

    for (int icol = 0; icol < ncol; icol++)
    {
        fprintf(fout, "column : %4i\n", icol);
        for (int i = 0; i < 80; i++) fprintf(fout, "-");
        fprintf(fout, "\n");
        if (write_abs_only)
        {
            fprintf(fout, " row, absolute value\n");
        }
        else
        {
            fprintf(fout, " row, real part, imaginary part, absolute value\n");
        }
        for (int i = 0; i < 80; i++) fprintf(fout, "-");
        fprintf(fout, "\n");
        
        int max_row = (write_upper_only) ? std::min(icol, nrow - 1) : (nrow - 1);
        for (int j = 0; j <= max_row; j++)
        {
            if (write_abs_only)
            {
                std::string s = "%4i  " + fmt + "\n";
                fprintf(fout, s.c_str(), j, abs(matrix(j, icol)));
            }
            else
            {
                fprintf(fout, "%4i  %18.12f %18.12f %18.12f\n", j, real(matrix(j, icol)), imag(matrix(j, icol)), 
                                                                abs(matrix(j, icol)));
            }
        }
        fprintf(fout,"\n");
    }

    fclose(fout);
}
        
void Utils::write_matrix(const std::string& fname, bool write_all, mdarray<double, 2>& matrix)
{
    static int icount = 0;

    icount++;
    std::stringstream s;
    s << icount;
    std::string full_name = s.str() + "_" + fname;

    FILE* fout = fopen(full_name.c_str(), "w");

    for (int icol = 0; icol < matrix.size(1); icol++)
    {
        fprintf(fout, "column : %4i\n", icol);
        for (int i = 0; i < 80; i++) fprintf(fout, "-");
        fprintf(fout, "\n");
        fprintf(fout, " row\n");
        for (int i = 0; i < 80; i++) fprintf(fout, "-");
        fprintf(fout, "\n");
        
        int max_row = (write_all) ? (matrix.size(0) - 1) : std::min(icol, matrix.size(0) - 1);
        for (int j = 0; j <= max_row; j++)
        {
            fprintf(fout, "%4i  %18.12f\n", j, matrix(j, icol));
        }
        fprintf(fout,"\n");
    }

    fclose(fout);
}

void Utils::check_hermitian(const std::string& name, mdarray<double_complex, 2>& mtrx)
{
    assert(mtrx.size(0) == mtrx.size(1));

    double maxdiff = 0.0;
    int i0 = -1;
    int j0 = -1;

    for (int i = 0; i < mtrx.size(0); i++)
    {
        for (int j = 0; j < mtrx.size(1); j++)
        {
            double diff = abs(mtrx(i, j) - conj(mtrx(j, i)));
            if (diff > maxdiff)
            {
                maxdiff = diff;
                i0 = i;
                j0 = j;
            }
        }
    }

    if (maxdiff > 1e-10)
    {
        std::stringstream s;
        s << name << " is not a hermitian matrix" << std::endl
          << "  maximum error: i, j : " << i0 << " " << j0 << " diff : " << maxdiff;

        warning_local(__FILE__, __LINE__, s);
    }
}


double Utils::confined_polynomial(double r, double R, int p1, int p2, int dm)
{
    double t = 1.0 - pow(r / R, 2);
    switch (dm)
    {
        case 0:
        {
            return (pow(r, p1) * pow(t, p2));
        }
        case 2:
        {
            return (-4 * p1 * p2 * pow(r, p1) * pow(t, p2 - 1) / pow(R, 2) +
                    p1 * (p1 - 1) * pow(r, p1 - 2) * pow(t, p2) + 
                    pow(r, p1) * (4 * (p2 - 1) * p2 * pow(r, 2) * pow(t, p2 - 2) / pow(R, 4) - 
                                  2 * p2 * pow(t, p2 - 1) / pow(R, 2)));
        }
        default:
        {
            error_local(__FILE__, __LINE__, "wrong derivative order");
            return 0.0;
        }
    }
}

std::vector<int> Utils::l_by_lm(int lmax)
{
    std::vector<int> l_by_lm__(lmmax(lmax));
    for (int l = 0; l <= lmax; l++)
    {
        for (int m = -l; m <= l; m++) l_by_lm__[lm_by_l_m(l, m)] = l;
    }
    return l_by_lm__;
}

std::pair< vector3d<double>, vector3d<int> > Utils::reduce_coordinates(vector3d<double> coord)
{
    std::pair< vector3d<double>, vector3d<int> > v; 
    
    v.first = coord;
    for (int i = 0; i < 3; i++)
    {
        v.second[i] = (int)floor(v.first[i]);
        v.first[i] -= v.second[i];
        if (v.first[i] < 0.0 || v.first[i] >= 1.0) error_local(__FILE__, __LINE__, "wrong fractional coordinates");
    }
    return v;
}

vector3d<int> Utils::find_translation_limits(double radius, double lattice_vectors[3][3])
{
    sirius::Timer t("sirius::Utils::find_translation_limits");

    vector3d<int> limits;

    int n = 0;
    while(true)
    {
        bool found = false;
        for (int i0 = -n; i0 <= n; i0++)
        {
            for (int i1 = -n; i1 <= n; i1++)
            {
                for (int i2 = -n; i2 <= n; i2++)
                {
                    if (abs(i0) == n || abs(i1) == n || abs(i2) == n)
                    {
                        vector3d<int> vf(i0, i1, i2);
                        vector3d<double> vc;
                        for (int x = 0; x < 3; x++)
                        {
                            vc[x] += (vf[0] * lattice_vectors[0][x] + 
                                      vf[1] * lattice_vectors[1][x] + 
                                      vf[2] * lattice_vectors[2][x]);
                        }
                        double len = vc.length();
                        if (len <= radius)
                        {
                            found = true;
                            for (int j = 0; j < 3; j++) limits[j] = std::max(2 * abs(vf[j]) + 1, limits[j]);
                        }
                    }
                }
            }
        }

        if (found) 
        {
            n++;
        }
        else 
        {
            return limits;
        }
    }
}

