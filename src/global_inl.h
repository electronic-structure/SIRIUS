inline int lmmax_by_lmax(int lmax)
{
    return (lmax + 1) * (lmax + 1);
}

inline int lm_by_l_m(int l, int m)
{
    return (l * l + l + m);
}

void write_matrix(const std::string& fname, bool write_all, mdarray<complex16, 2>& matrix)
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
        fprintf(fout, " row                real               imag                abs \n");
        for (int i = 0; i < 80; i++) fprintf(fout, "-");
        fprintf(fout, "\n");
        
        int max_row = (write_all) ? (matrix.size(0) - 1) : std::min(icol, matrix.size(0) - 1);
        for (int j = 0; j <= max_row; j++)
        {
            fprintf(fout, "%4i  %18.12f %18.12f %18.12f\n", j, real(matrix(j, icol)), imag(matrix(j, icol)), 
                                                            abs(matrix(j, icol)));
        }
        fprintf(fout,"\n");
    }

    fclose(fout);
}

void write_matrix(const std::string& fname, bool write_all, mdarray<double, 2>& matrix)
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


