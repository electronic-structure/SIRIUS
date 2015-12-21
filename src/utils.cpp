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

void Utils::check_hermitian(const std::string& name, mdarray<double_complex, 2>& mtrx, int n)
{
    assert(mtrx.size(0) == mtrx.size(1));

    double maxdiff = 0.0;
    int i0 = -1;
    int j0 = -1;

    if (n == -1) n = (int)mtrx.size(0);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
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
    const double eps = 1e-6;

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

vector3d<int> Utils::find_translations(double radius__, matrix3d<double> const& lattice_vectors__)
{
    sirius::Timer t("sirius::Utils::find_translations");

    double theta = pi;
    double phi = 0;

    auto minv = inverse(lattice_vectors__);
    vector3d<int> limits(0, 0, 0);
    double max_lim[] = {0, 0, 0};
    double min_lim[] = {0, 0, 0};

    auto test_point = [&minv, radius__, &limits, &max_lim, &min_lim](double th, double ph)
    {
        vector3d<double> pos(radius__ * std::sin(th) * std::cos(ph),
                             radius__ * std::sin(th) * std::sin(ph),
                             radius__ * std::cos(th));
        auto f = minv * pos;
        for (int x: {0, 1, 2})
        {
            max_lim[x] = std::max(max_lim[x], std::floor(f[x]));
            min_lim[x] = std::min(min_lim[x], std::ceil(f[x]));
        }
    };

    int num_points = 5000;

    test_point(0, 0);
    test_point(theta, phi);
    for (int k = 1; k < num_points - 1; k++)
    {
        double hk = -1.0 + 2.0 * k / (num_points - 1);
        theta = std::acos(hk);
        double t = phi + 3.80925122745582 / std::sqrt(double(num_points)) / std::sqrt(1 - hk * hk);
        phi = std::fmod(t, twopi);
        test_point(theta, phi);
    }
    
    for (int x: {0, 1, 2}) limits[x] = static_cast<int>(max_lim[x] - min_lim[x] + 0.001) + 1;

    return limits;
}

