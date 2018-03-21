// Copyright (c) 2013-2018 Anton Kozhevnikov, Thomas Schulthess
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

/** \file utils.h
 *
 *  \brief Contains definition and partial implementation of sirius::Utils class.
 */

#ifndef __UTILS_H__
#define __UTILS_H__

#include <gsl/gsl_sf_erf.h>
#include <fstream>
#include <string>
#include <complex>
#include "typedefs.h"
#include "constants.h"
#include "sddk.hpp"

using namespace sddk;

/// Utility class.
class Utils
{
  public:
    /// Maximum number of \f$ \ell, m \f$ combinations for a given \f$ \ell_{max} \f$
    static inline int lmmax(int lmax)
    {
        return (lmax + 1) * (lmax + 1);
    }

    static inline int lm_by_l_m(int l, int m)
    {
        return (l * l + l + m);
    }

    static inline int lmax_by_lmmax(int lmmax__)
    {
        int lmax = int(std::sqrt(double(lmmax__)) + 1e-8) - 1;
        if (lmmax(lmax) != lmmax__) {
            TERMINATE("wrong lmmax");
        }
        return lmax;
    }

    static inline bool file_exists(const std::string file_name)
    {
        std::ifstream ifs(file_name.c_str());
        return ifs.is_open();
    }

    static inline double fermi_dirac_distribution(double e)
    {
        double kT = 0.001;
        if (e > 100 * kT) {
            return 0.0;
        }
        if (e < -100 * kT) {
            return 1.0;
        }
        return (1.0 / (std::exp(e / kT) + 1.0));
    }

    static inline double gaussian_smearing(double e, double delta)
    {
        return 0.5 * (1 - gsl_sf_erf(e / delta));
    }

    static inline double cold_smearing(double e)
    {
        double a = -0.5634;

        if (e < -10.0) {
            return 1.0;
        }
        if (e > 10.0) {
            return 0.0;
        }

        return 0.5 * (1 - gsl_sf_erf(e)) - 1 - 0.25 * std::exp(-e * e) * (a + 2 * e - 2 * a * e * e) / std::sqrt(pi);
    }

    static std::string double_to_string(double val, int precision = -1)
    {
        char buf[100];

        double abs_val = std::abs(val);

        if (precision == -1) {
            if (abs_val > 1.0) {
                precision = 6;
            } else if (abs_val > 1e-14) {
                precision = int(-std::log(abs_val) / std::log(10.0)) + 7;
            } else {
                return std::string("0.0");
            }
        }

        std::stringstream fmt;
        fmt << "%." << precision << "f";

        int len = snprintf(buf, 100, fmt.str().c_str(), val);
        for (int i = len - 1; i >= 1; i--) {
            if (buf[i] == '0' && buf[i - 1] == '0') {
                buf[i] = 0;
            } else {
                break;
            }
        }
        return std::string(buf);
    }

    static inline double phi_by_sin_cos(double sinp, double cosp)
    {
        double phi = std::atan2(sinp, cosp);
        if (phi < 0) {
            phi += twopi;
        }
        return phi;
    }

    static inline long double factorial(int n)
    {
        assert(n >= 0);

        long double result = 1.0L;
        for (int i = 1; i <= n; i++) {
            result *= i;
        }
        return result;
    }

    /// Simple hash function.
    /** Example: printf("hash: %16llX\n", hash()); */
    static uint64_t hash(void const* buff, size_t size, uint64_t h = 5381)
    {
        unsigned char const* p = static_cast<unsigned char const*>(buff);
        for (size_t i = 0; i < size; i++) {
            h = ((h << 5) + h) + p[i];
        }
        return h;
    }

    static void write_matrix(const std::string& fname,
                             mdarray<double_complex, 2>& matrix,
                             int nrow,
                             int ncol,
                             bool write_upper_only = true,
                             bool write_abs_only   = false,
                             std::string fmt       = "%18.12f")
    {
        static int icount = 0;

        if (nrow < 0 || nrow > (int)matrix.size(0) || ncol < 0 || ncol > (int)matrix.size(1))
            TERMINATE("wrong number of rows or columns");

        icount++;
        std::stringstream s;
        s << icount;
        std::string full_name = s.str() + "_" + fname;

        FILE* fout = fopen(full_name.c_str(), "w");

        for (int icol = 0; icol < ncol; icol++) {
            fprintf(fout, "column : %4i\n", icol);
            for (int i = 0; i < 80; i++)
                fprintf(fout, "-");
            fprintf(fout, "\n");
            if (write_abs_only) {
                fprintf(fout, " row, absolute value\n");
            } else {
                fprintf(fout, " row, real part, imaginary part, absolute value\n");
            }
            for (int i = 0; i < 80; i++)
                fprintf(fout, "-");
            fprintf(fout, "\n");

            int max_row = (write_upper_only) ? std::min(icol, nrow - 1) : (nrow - 1);
            for (int j = 0; j <= max_row; j++) {
                if (write_abs_only) {
                    std::string s = "%4i  " + fmt + "\n";
                    fprintf(fout, s.c_str(), j, abs(matrix(j, icol)));
                } else {
                    fprintf(fout, "%4i  %18.12f %18.12f %18.12f\n", j, real(matrix(j, icol)), imag(matrix(j, icol)),
                            abs(matrix(j, icol)));
                }
            }
            fprintf(fout, "\n");
        }

        fclose(fout);
    }

    static void write_matrix(std::string const& fname, bool write_all, mdarray<double, 2>& matrix)
    {
        static int icount = 0;

        icount++;
        std::stringstream s;
        s << icount;
        std::string full_name = s.str() + "_" + fname;

        FILE* fout = fopen(full_name.c_str(), "w");

        for (int icol = 0; icol < (int)matrix.size(1); icol++) {
            fprintf(fout, "column : %4i\n", icol);
            for (int i = 0; i < 80; i++)
                fprintf(fout, "-");
            fprintf(fout, "\n");
            fprintf(fout, " row\n");
            for (int i = 0; i < 80; i++)
                fprintf(fout, "-");
            fprintf(fout, "\n");

            int max_row = (write_all) ? ((int)matrix.size(0) - 1) : std::min(icol, (int)matrix.size(0) - 1);
            for (int j = 0; j <= max_row; j++) {
                fprintf(fout, "%4i  %18.12f\n", j, matrix(j, icol));
            }
            fprintf(fout, "\n");
        }

        fclose(fout);
    }

    static void write_matrix(std::string const& fname, bool write_all, matrix<double_complex> const& mtrx)
    {
        static int icount = 0;

        icount++;
        std::stringstream s;
        s << icount;
        std::string full_name = s.str() + "_" + fname;

        FILE* fout = fopen(full_name.c_str(), "w");

        for (int icol = 0; icol < (int)mtrx.size(1); icol++) {
            fprintf(fout, "column : %4i\n", icol);
            for (int i = 0; i < 80; i++)
                fprintf(fout, "-");
            fprintf(fout, "\n");
            fprintf(fout, " row\n");
            for (int i = 0; i < 80; i++)
                fprintf(fout, "-");
            fprintf(fout, "\n");

            int max_row = (write_all) ? ((int)mtrx.size(0) - 1) : std::min(icol, (int)mtrx.size(0) - 1);
            for (int j = 0; j <= max_row; j++) {
                fprintf(fout, "%4i  %18.12f %18.12f\n", j, real(mtrx(j, icol)), imag(mtrx(j, icol)));
            }
            fprintf(fout, "\n");
        }

        fclose(fout);
    }

    template <typename T>
    static void check_hermitian(const std::string& name, matrix<T> const& mtrx, int n = -1)
    {
        assert(mtrx.size(0) == mtrx.size(1));

        double maxdiff = 0.0;
        int i0         = -1;
        int j0         = -1;

        if (n == -1) {
            n = static_cast<int>(mtrx.size(0));
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double diff = std::abs(mtrx(i, j) - type_wrapper<T>::conjugate(mtrx(j, i)));
                if (diff > maxdiff) {
                    maxdiff = diff;
                    i0      = i;
                    j0      = j;
                }
            }
        }

        if (maxdiff > 1e-10) {
            std::stringstream s;
            s << name << " is not a symmetric or hermitian matrix" << std::endl
              << "  maximum error: i, j : " << i0 << " " << j0 << " diff : " << maxdiff;

            WARNING(s);
        }
    }

    static double confined_polynomial(double r, double R, int p1, int p2, int dm)
    {
        double t = 1.0 - std::pow(r / R, 2);
        switch (dm) {
            case 0: {
                return (std::pow(r, p1) * std::pow(t, p2));
            }
            case 2: {
                return (-4 * p1 * p2 * std::pow(r, p1) * std::pow(t, p2 - 1) / std::pow(R, 2) +
                        p1 * (p1 - 1) * std::pow(r, p1 - 2) * std::pow(t, p2) +
                        std::pow(r, p1) * (4 * (p2 - 1) * p2 * std::pow(r, 2) * std::pow(t, p2 - 2) / std::pow(R, 4) -
                                           2 * p2 * std::pow(t, p2 - 1) / std::pow(R, 2)));
            }
            default: {
                TERMINATE("wrong derivative order");
                return 0.0;
            }
        }
    }

    static mdarray<int, 1> l_by_lm(int lmax)
    {
        mdarray<int, 1> v(lmmax(lmax));
        for (int l = 0; l <= lmax; l++) {
            for (int m = -l; m <= l; m++) {
                v[lm_by_l_m(l, m)] = l;
            }
        }
        return std::move(v);
    }

    static std::vector<std::pair<int, int>> l_m_by_lm(int lmax)
    {
        std::vector<std::pair<int, int>> v(lmmax(lmax));
        for (int l = 0; l <= lmax; l++) {
            for (int m = -l; m <= l; m++) {
                int lm       = lm_by_l_m(l, m);
                v[lm].first  = l;
                v[lm].second = m;
            }
        }
        return std::move(v);
    }

    inline static double round(double a__, int n__)
    {
        double a0 = std::floor(a__);
        double b  = std::round((a__ - a0) * std::pow(10, n__)) / std::pow(10, n__);
        return a0 + b;
    }

    inline static double_complex round(double_complex a__, int n__)
    {
        return double_complex(round(a__.real(), n__), round(a__.imag(), n__));
    }

    template <typename T>
    inline static int sign(T val)
    {
        return (T(0) < val) - (val < T(0));
    }

    /// Pack two indices into one for symmetric matrices.
    inline static int packed_index(int i__, int j__)
    {
        /* suppose we have a symmetric matrix: M_{ij} = M_{ji}
               j
           +-------+
           | + + + |
          i|   + + |   -> idx = j * (j + 1) / 2 + i  for  i <= j
           |     + |
           +-------+

           i, j are row and column indices 
        */

        if (i__ > j__) {
            std::swap(i__, j__);
        }
        return j__ * (j__ + 1) / 2 + i__;
    }
};

#endif
