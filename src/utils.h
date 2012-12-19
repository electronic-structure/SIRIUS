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
};

#endif

