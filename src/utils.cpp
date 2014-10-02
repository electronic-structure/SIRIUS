// Copyright (c) 2013-2014 Anton Kozhevnikov, Thomas Schulthess
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that 
// the following conditions are met:
// 
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the 
//    following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions 
//    and the following disclaimer in the documentation and/or other materials provided with the distribution.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED 
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR 
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/** \file utils.cpp
 *   
 *  \brief Contains remaining implementation of sirius::Utils class.
 */

#include "utils.h"

void Utils::write_matrix(const std::string& fname, mdarray<double_complex, 2>& matrix, int nrow, int ncol,
                         bool write_upper_only, bool write_abs_only, std::string fmt)
{
    static int icount = 0;

    if (nrow < 0 || nrow > (int)matrix.size(0) || ncol < 0 || ncol > (int)matrix.size(1))
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

    for (int icol = 0; icol < (int)matrix.size(1); icol++)
    {
        fprintf(fout, "column : %4i\n", icol);
        for (int i = 0; i < 80; i++) fprintf(fout, "-");
        fprintf(fout, "\n");
        fprintf(fout, " row\n");
        for (int i = 0; i < 80; i++) fprintf(fout, "-");
        fprintf(fout, "\n");
        
        int max_row = (write_all) ? ((int)matrix.size(0) - 1) : std::min(icol, (int)matrix.size(0) - 1);
        for (int j = 0; j <= max_row; j++)
        {
            fprintf(fout, "%4i  %18.12f\n", j, matrix(j, icol));
        }
        fprintf(fout,"\n");
    }

    fclose(fout);
}

void Utils::write_matrix(std::string const& fname, bool write_all, matrix<double_complex> const& mtrx)
{
    static int icount = 0;

    icount++;
    std::stringstream s;
    s << icount;
    std::string full_name = s.str() + "_" + fname;

    FILE* fout = fopen(full_name.c_str(), "w");

    for (int icol = 0; icol < (int)mtrx.size(1); icol++)
    {
        fprintf(fout, "column : %4i\n", icol);
        for (int i = 0; i < 80; i++) fprintf(fout, "-");
        fprintf(fout, "\n");
        fprintf(fout, " row\n");
        for (int i = 0; i < 80; i++) fprintf(fout, "-");
        fprintf(fout, "\n");
        
        int max_row = (write_all) ? ((int)mtrx.size(0) - 1) : std::min(icol, (int)mtrx.size(0) - 1);
        for (int j = 0; j <= max_row; j++)
        {
            fprintf(fout, "%4i  %18.12f %18.12f\n", j, real(mtrx(j, icol)), imag(mtrx(j, icol)));
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

    for (int i = 0; i < (int)mtrx.size(0); i++)
    {
        for (int j = 0; j < (int)mtrx.size(1); j++)
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
    const double eps = 1e-10;

    std::pair< vector3d<double>, vector3d<int> > v; 
    
    v.first = coord;
    for (int i = 0; i < 3; i++)
    {
        v.second[i] = (int)floor(v.first[i]);
        v.first[i] -= v.second[i];
        if (v.first[i] < -eps || v.first[i] > 1.0 + eps)
        {
            std::stringstream s;
            s << "wrong fractional coordinates" << std::endl
              << v.first[0] << " " << v.first[1] << " " << v.first[2];
            error_local(__FILE__, __LINE__, s);
        }
        if (v.first[i] < 0) v.first[i] = 0;
        if (v.first[i] >= (1 - eps))
        {
            v.first[i] = 0;
            v.second[i] += 1;
        }
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

