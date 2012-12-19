#ifndef __UTILS_H__
#define __UTILS_H__

class Utils
{
    public:

        static bool file_exists(const std::string& file_name)
        {
            std::ifstream ifs(file_name.c_str());
            if (ifs.is_open()) return true;
            return false;
        }

        static inline double vector_length(double v[3])
        {
            return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        }

        template <typename U, typename V>
        static inline double scalar_product(U a[3], V b[3])
        {
            return (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]);
        }


        static inline double fermi_dirac_distribution(double e)
        {
            double kT = 0.01;
            if (e > 100 * kT) return 0.0;
            if (e < -100 * kT) return 1.0;
            return (1.0 / (exp(e / kT) + 1.0));
        }
        
        static inline double gaussian_smearing(double e)
        {
            double delta = 0.01;
        
            return 0.5 * (1 - gsl_sf_erf(e / delta));
        }

};

#endif

