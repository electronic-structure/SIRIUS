#ifndef __LIBXC_INTERFACE_H__
#define __LIBXC_INTERFACE_H__

#include <xc.h>
#include <string.h>
#include "error_handling.h"
#include "utils.h"

namespace sirius
{

extern std::map<std::string, int> libxc_functionals;

class XC_functional
{
    private:
        
        std::string libxc_name_;

        int num_spins_;
        
        xc_func_type handler_;

    public:

        XC_functional(const std::string libxc_name__, int num_spins__) 
            : libxc_name_(libxc_name__), num_spins_(num_spins__)
        {
            /* check if functional name is in list */
            if (libxc_functionals.count(libxc_name_) == 0)
            {
                error_local(__FILE__, __LINE__, "XC functional is unknown");
            }

            /* init xc functional handler */
            if (xc_func_init(&handler_, libxc_functionals[libxc_name_], num_spins_) != 0) 
                error_local(__FILE__, __LINE__, "xc_func_init() failed");
        }

        ~XC_functional()
        {
            xc_func_end(&handler_);
        }

        std::string name()
        {
            return std::string(handler_.info->name);
        }
        
        std::string refs()
        {
            return std::string(handler_.info->refs);
        }

        int family()
        {
            return handler_.info->family;
        }

        bool lda()
        {
            return family() == XC_FAMILY_LDA;
        }

        bool gga()
        {
            return family() == XC_FAMILY_GGA;
        }
        
        /// Add LDA contribution.
        void add(int size, const double* rho, double* v, double* e)
        {
            std::vector<double> v_tmp(size);
            std::vector<double> e_tmp(size);

            /* check density */
            for (int i = 0; i < size; i++)
            {
                if (rho[i] < 0.0)
                {
                    std::stringstream s;
                    s << "rho is negative : " << rho[i];
                    error_local(__FILE__, __LINE__, s);
                }
            }

            if (family() != XC_FAMILY_LDA) error_local(__FILE__, __LINE__, "wrong XC");

            xc_lda_exc_vxc(&handler_, size, &rho[0], &e_tmp[0], &v_tmp[0]);
       
            for (int i = 0; i < size; i++)
            {
                v[i] += v_tmp[i];
                e[i] += e_tmp[i];
            }
        }
};

/// Simple interface to libxc functionals.
class libxc_interface
{
    private:

        /// String id of exchanage functional.
        std::string libxc_x_func_;
        
        /// String id of correlation functional.
        std::string libxc_c_func_;
        
        /// Spin-unpolarized XC functional descriptor.
        xc_func_type unpolarized_func_[2];

        /// Spin-polarized XC functional descriptor.
        xc_func_type polarized_func_[2];

    public:
        
        libxc_interface(const std::string libxc_x_func__, const std::string libxc_c_func__) 
            : libxc_x_func_(libxc_x_func__), libxc_c_func_(libxc_c_func__)
        {
            //int xc_id[] = {XC_LDA_X, XC_LDA_C_VWN};
            //int xc_id[] = {XC_LDA_X, XC_LDA_C_PZ};
            //int xc_id[] = {XC_LDA_X, XC_LDA_C_PW};
            
            /* id of exchange and correlation functionals */
            int xc_id[] = {libxc_functionals[libxc_x_func_], libxc_functionals[libxc_c_func_]};
            
            /* init xc functional descriptors */
            for (int i = 0; i < 2; i++)
            {
                if (xc_func_init(&unpolarized_func_[i], xc_id[i], XC_UNPOLARIZED) != 0) 
                    error_local(__FILE__, __LINE__, "functional is not found");
                
                if (xc_func_init(&polarized_func_[i], xc_id[i], XC_POLARIZED) != 0)
                    error_local(__FILE__, __LINE__, "functional is not found");
            }

        }

        ~libxc_interface()
        {
            for (int i = 0; i < 2; i++)
            {
                xc_func_end(&unpolarized_func_[i]);
                xc_func_end(&polarized_func_[i]);
            }
        }

        void getxc(int size, const double* rho, double* vxc, double* exc)
        {
            memset(vxc, 0, size * sizeof(double)); 
            memset(exc, 0, size * sizeof(double));
    
            std::vector<double> vxc_tmp(size);
            std::vector<double> exc_tmp(size);

            // check rho
            for (int i = 0; i < size; i++)
            {
                if (rho[i] < 0.0)
                {
                    std::stringstream s;
                    s << "rho is negative : " << rho[i];
                    error_local(__FILE__, __LINE__, s);
                }
            }

            for (int i = 0; i < 2; i++)
            {
                xc_lda_exc_vxc(&unpolarized_func_[i], size, &rho[0], &exc_tmp[0], &vxc_tmp[0]);
       
                for (int j = 0; j < size; j++)
                {
                    vxc[j] += vxc_tmp[j];
                    exc[j] += exc_tmp[j];
                }
            }
        }

        void getxc(int size, const double* rho, const double* mag, double* vxc, double* bxc, double* exc)
        {
            memset(vxc, 0, size * sizeof(double)); 
            memset(bxc, 0, size * sizeof(double)); 
            memset(exc, 0, size * sizeof(double));

            std::vector<double> rhoud(size * 2);
            for (int i = 0; i < size; i++)
            {
                rhoud[2 * i] = 0.5 * (rho[i] + mag[i]);
                rhoud[2 * i + 1] = 0.5 * (rho[i] - mag[i]);

                if (rhoud[2 * i] < 0.0) error_local(__FILE__, __LINE__, "rho_up is negative");

                if (rhoud[2 * i + 1] < 0.0)
                {
                    if (fabs(rhoud[2 * i + 1]) > 1e-8)
                    {
                        std::stringstream s;
                        s << "rho_dn is negative : " << Utils::double_to_string(rhoud[2 * i + 1]) << std::endl
                          << "  rho : " << Utils::double_to_string(rho[i]) << "   mag : " << Utils::double_to_string(mag[i]);
                        error_local(__FILE__, __LINE__, s);
                    }
                    else
                    {
                        rhoud[2 * i + 1] = 0.0;
                    }
                }
            }

            std::vector<double> vxc_tmp(size * 2);
            std::vector<double> exc_tmp(size);
            
            for (int i = 0; i < 2; i++)
            {
                xc_lda_exc_vxc(&polarized_func_[i], size, &rhoud[0], &exc_tmp[0], &vxc_tmp[0]);

                for (int j = 0; j < size; j++)
                {
                    exc[j] += exc_tmp[j];
                    vxc[j] += 0.5 * (vxc_tmp[2 * j] + vxc_tmp[2 * j + 1]);
                    bxc[j] += 0.5 * (vxc_tmp[2 * j] - vxc_tmp[2 * j + 1]);
                }
            }

            for (int i = 0; i < size; i++)
            {
                if (vxc[i] > 0.0) error_local(__FILE__, __LINE__, "vxc > 0");
                
                if (bxc[i] > 0.0)
                {
                    //if (bxc[i] > 1e-7)
                    //{
                    //    std::stringstream s;
                    //    s << "bxc is positive : " << bxc[i];
                    //    error(__FILE__, __LINE__, s);
                    //}
                    //else
                    //{
                        bxc[i] = 0.0;
                    //}
                }
            }
         }
};

};

#endif // __LIBXC_INTERFACE_H__
