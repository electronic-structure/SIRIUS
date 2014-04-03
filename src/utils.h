#ifndef __UTILS_H__
#define __UTILS_H__

#include <gsl/gsl_sf_erf.h>
#include <fstream>
#include <string>
#include <complex>
#include "typedefs.h"
#include "constants.h"
#include "mdarray.h"
#include "timer.h"
#include "vector3d.h"

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
            int lmax = int(sqrt(double(lmmax__)) + 1e-8) - 1;
            if (lmmax(lmax) != lmmax__) error_local(__FILE__, __LINE__, "wrong lmmax");
            return lmax;
        }

        static inline bool file_exists(const std::string file_name)
        {
            std::ifstream ifs(file_name.c_str());
            if (ifs.is_open()) return true;
            return false;
        }

        template <typename U, typename V>
        static inline double scalar_product(vector3d<U> a, vector3d<V> b)
        {
            return (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]);
        }

        static inline double fermi_dirac_distribution(double e)
        {
            double kT = 0.001;
            if (e > 100 * kT) return 0.0;
            if (e < -100 * kT) return 1.0;
            return (1.0 / (exp(e / kT) + 1.0));
        }
        
        static inline double gaussian_smearing(double e, double delta)
        {
            return 0.5 * (1 - gsl_sf_erf(e / delta));
        }
        
        static inline double cold_smearing(double e)
        {
            double a = -0.5634;

            if (e < -10.0) return 1.0;
            if (e > 10.0) return 0.0;

            return 0.5 * (1 - gsl_sf_erf(e)) - 1 - 0.25 * exp(-e * e) * (a + 2 * e - 2 * a * e * e) / sqrt(pi);
        }

        static std::string double_to_string(double val, int precision = -1)
        {
            char buf[100];

            double abs_val = std::abs(val);

            if (precision == -1)
            {
                if (abs_val > 1.0) 
                {
                    precision = 6;
                }
                else if (abs_val > 1e-14)
                {
                    precision = int(-std::log(abs_val) / std::log(10.0)) + 7;
                }
                else
                {
                    return std::string("0.0");
                }
            }

            std::stringstream fmt;
            fmt << "%." << precision << "f";
        
            int len = snprintf(buf, 100, fmt.str().c_str(), val);
            for (int i = len - 1; i >= 1; i--) 
            {
                if (buf[i] == '0' && buf[i - 1] == '0') 
                {
                    buf[i] = 0;
                }
                else
                {
                    break;
                }
            }
            return std::string(buf);
        }

        static inline double phi_by_sin_cos(double sinp, double cosp)
        {
            double phi = atan2(sinp, cosp);
            if (phi < 0) phi += twopi;
            return phi;
        }

        static inline long double factorial(int n)
        {
            assert(n >= 0);

            long double result = 1.0L;
            for (int i = 1; i <= n; i++) result *= i;
            return result;
        }

        static void write_matrix(const std::string& fname, mdarray<double_complex, 2>& matrix, int nrow, int ncol,
                                 bool write_upper_only = true, bool write_abs_only = false, std::string fmt = "%18.12f");
        
        static void write_matrix(const std::string& fname, bool write_all, mdarray<double, 2>& matrix);

        static void check_hermitian(const std::string& name, mdarray<double_complex, 2>& mtrx);

        static double confined_polynomial(double r, double R, int p1, int p2, int dm);

        static std::vector<int> l_by_lm(int lmax);

        static std::pair< vector3d<double>, vector3d<int> > reduce_coordinates(vector3d<double> coord);

        static vector3d<int> find_translation_limits(double radius, double lattice_vectors[3][3]);
};

#endif

