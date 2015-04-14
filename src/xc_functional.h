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

/** \file xc_functional.h
 *   
 *  \brief Contains implementation of sirius::XC_functional class.
 */

#ifndef __XC_FUNCTIONAL_H__
#define __XC_FUNCTIONAL_H__

#include <xc.h>
#include <string.h>
#include "error_handling.h"
#include "utils.h"

namespace sirius
{

extern std::map<std::string, int> libxc_functionals;

/// Interface class to Libxc.
class XC_functional
{
    private:
        
        std::string libxc_name_;

        int num_spins_;
        
        xc_func_type handler_;

        /* forbid copy constructor */
        XC_functional(const XC_functional& src) = delete;

        /* forbid assigment operator */
        XC_functional& operator=(const XC_functional& src) = delete;

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

        /// Get LDA contribution.
        void get_lda(const int size, 
                     const double* rho, 
                     double* v, 
                     double* e)
        {
            if (family() != XC_FAMILY_LDA) error_local(__FILE__, __LINE__, "wrong XC");

            /* check density */
            for (int i = 0; i < size; i++)
            {
                if (rho[i] < 0)
                {
                    std::stringstream s;
                    s << "rho is negative : " << Utils::double_to_string(rho[i]);
                    error_local(__FILE__, __LINE__, s);
                }
            }

            xc_lda_exc_vxc(&handler_, size, rho, e, v);
        }

        /// Get LSDA contribution.
        void get_lda(const int size,
                     const double* rho_up,
                     const double* rho_dn,
                     double* v_up,
                     double* v_dn,
                     double* e)
        {
            if (family() != XC_FAMILY_LDA) error_local(__FILE__, __LINE__, "wrong XC");

            std::vector<double> rho_ud(size * 2);
            /* check and rearrange density */
            for (int i = 0; i < size; i++)
            {
                if (rho_up[i] < 0 || rho_dn[i] < 0)
                {
                    std::stringstream s;
                    s << "rho is negative : " << Utils::double_to_string(rho_up[i]) 
                      << " " << Utils::double_to_string(rho_dn[i]);
                    error_local(__FILE__, __LINE__, s);
                }
                
                rho_ud[2 * i] = rho_up[i];
                rho_ud[2 * i + 1] = rho_dn[i];
            }
            
            std::vector<double> v_ud(size * 2);

            xc_lda_exc_vxc(&handler_, size, &rho_ud[0], &e[0], &v_ud[0]);
            
            /* extract potential */
            for (int i = 0; i < size; i++)
            {
                v_up[i] = v_ud[2 * i];
                v_dn[i] = v_ud[2 * i + 1];
            }
        }

        void add_lda(const int size,
                     const double* rho_up,
                     const double* rho_dn,
                     double* v_up,
                     double* v_dn,
                     double* e)
        {
            if (family() != XC_FAMILY_LDA) error_local(__FILE__, __LINE__, "wrong XC");

            std::vector<double> rho_ud(size * 2);
            /* check and rearrange density */
            for (int i = 0; i < size; i++)
            {
                if (rho_up[i] < 0 || rho_dn[i] < 0)
                {
                    std::stringstream s;
                    s << "rho is negative : " << Utils::double_to_string(rho_up[i]) 
                      << " " << Utils::double_to_string(rho_dn[i]);
                    error_local(__FILE__, __LINE__, s);
                }
                
                rho_ud[2 * i] = rho_up[i];
                rho_ud[2 * i + 1] = rho_dn[i];
            }
            
            std::vector<double> v_ud(size * 2);
            std::vector<double> e_tmp(size);

            xc_lda_exc_vxc(&handler_, size, &rho_ud[0], &e_tmp[0], &v_ud[0]);
            
            /* extract potential */
            for (int i = 0; i < size; i++)
            {
                v_up[i] += v_ud[2 * i];
                v_dn[i] += v_ud[2 * i + 1];
                e[i] += e_tmp[i];
            }
        }

        /// Get GGA contribution.
        void get_gga(const int size,
                     const double* rho,
                     const double* sigma,
                     double* vrho,
                     double* vsigma,
                     double* e)
        {
            if (family() != XC_FAMILY_GGA) error_local(__FILE__, __LINE__, "wrong XC");

            /* check density */
            for (int i = 0; i < size; i++)
            {
                if (rho[i] < 0.0)
                {
                    std::stringstream s;
                    s << "rho is negative : " << Utils::double_to_string(rho[i]);
                    error_local(__FILE__, __LINE__, s);
                }
            }

            xc_gga_exc_vxc(&handler_, size, rho, sigma, e, vrho, vsigma);
        }

        /// Get spin-resolved GGA contribution.
        void get_gga(const int size, 
                     const double* rho_up, 
                     const double* rho_dn, 
                     const double* sigma_uu, 
                     const double* sigma_ud, 
                     const double* sigma_dd, 
                     double* vrho_up,
                     double* vrho_dn, 
                     double* vsigma_uu, 
                     double* vsigma_ud,
                     double* vsigma_dd,
                     double* e)
        {
            if (family() != XC_FAMILY_GGA) error_local(__FILE__, __LINE__, "wrong XC");

            std::vector<double> rho(2 * size);
            std::vector<double> sigma(3 * size);
            /* check and rearrange density */
            /* rearrange sigma as well */
            for (int i = 0; i < size; i++)
            {
                if (rho_up[i] < 0 || rho_dn[i] < 0)
                {
                    std::stringstream s;
                    s << "rho is negative : " << Utils::double_to_string(rho_up[i]) 
                      << " " << Utils::double_to_string(rho_dn[i]);
                    error_local(__FILE__, __LINE__, s);
                }
                
                rho[2 * i] = rho_up[i];
                rho[2 * i + 1] = rho_dn[i];

                sigma[3 * i] = sigma_uu[i];
                sigma[3 * i + 1] = sigma_ud[i];
                sigma[3 * i + 2] = sigma_dd[i];
            }
            
            std::vector<double> vrho(2 * size);
            std::vector<double> vsigma(3 * size);

            xc_gga_exc_vxc(&handler_, size, &rho[0], &sigma[0], e, &vrho[0], &vsigma[0]);

            /* extract vrho and vsigma */
            for (int i = 0; i < size; i++)
            {
                vrho_up[i] = vrho[2 * i];
                vrho_dn[i] = vrho[2 * i + 1];

                vsigma_uu[i] = vsigma[3 * i];
                vsigma_ud[i] = vsigma[3 * i + 1];
                vsigma_dd[i] = vsigma[3 * i + 2];
            }
        }
};

};

#endif // __XC_FUNCTIONAL_H__
