inline double vector_length(double v[3])
{
    return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

template <typename Ta, typename Tb>
inline double scalar_product(Ta a[3], Tb b[3])
{
    return (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]);
}

inline int lmmax_by_lmax(int lmax)
{
    return (lmax + 1) * (lmax + 1);
}

inline int lm_by_l_m(int l, int m)
{
    return (l * l + l + m);
}

double fermi_dirac_distribution(double e)
{
    double kT = 0.01;
    if (e > 100 * kT) return 0.0;
    if (e < -100 * kT) return 1.0;
    return (1.0 / (exp(e / kT) + 1.0));
}

double gaussian_smearing(double e)
{
    double delta = 0.01;

    return 0.5 * (1 - gsl_sf_erf(e / delta));
}

void write_matrix(const std::string& fname, mdarray<complex16, 2>& matrix)
{
    FILE* fout = fopen(fname.c_str(), "w");

    for (int icol = 0; icol < matrix.size(1); icol++)
    {
        fprintf(fout, "column : %4i\n", icol);
        for (int i = 0; i < 80; i++) fprintf(fout, "-");
        fprintf(fout, "\n");
        fprintf(fout, " row                real               imag                abs \n");
        for (int i = 0; i < 80; i++) fprintf(fout, "-");
        fprintf(fout, "\n");
        for (int j = 0; j < matrix.size(0); j++)
        {
            fprintf(fout, "%4i  %18.12f %18.12f %18.12f\n", j, real(matrix(j, icol)), imag(matrix(j, icol)), 
                                                            abs(matrix(j, icol)));
        }
        fprintf(fout,"\n");

    }

    fclose(fout);
}
