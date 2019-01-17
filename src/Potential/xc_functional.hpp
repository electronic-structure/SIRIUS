// Copyright (c) 2013-2016 Anton Kozhevnikov, Thomas Schulthess
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

/** \file xc_functional.hpp
 *
 *  \brief Contains implementation of sirius::XC_functional class.
 */

#ifndef __XC_FUNCTIONAL_HPP__
#define __XC_FUNCTIONAL_HPP__

#include <xc.h>
#include <string.h>
#include <xc_functional_base.hpp>

namespace sirius {

/// Interface class to Libxc.
    class XC_functional : public XC_functional_base
{
    private:
        // I can not use a generic void pointer because xc_func_type is a structure
        // while wdv_functional_ is a pointer over structure.

        vdwxc_data handler_wdv_{nullptr};
        bool wdv_functional_{false};

        /* forbid copy constructor */
        XC_functional(const XC_functional& src) = delete;

        /* forbid assigment operator */
        XC_functional& operator=(const XC_functional& src) = delete;

        bool initialized_{false};

    public:
    /* we need the context because libvdwxc asks for lattice vectors and fft parameters */

    XC_functional(const simulation_context &ctx__, const std::string libxc_name__, int num_spins__)
        :  XC_functional_base(libxc_name__, num_spins_);
        {

            /* return immediately if the functional_base class is initialized */
            if (initialized) {
                return;
            }

            /* test if have van der walls functionals types */

            bool test = (libxc_name_ == "XC_FUNC_VDWDF");
            test = test || (libxc_name_ == "XC_FUNC_VDWDF2");
            test = test || (libxc_name_ == "XC_FUNC_VDWDFCX");

            if (test) {
                if (num_spins__ == 1) {
                    handler_vdw_ = vdwxc_new(xc_name_.c_string());
                } else {
                    handler_vdw_ = vdwxc_new_spin(xc_name_.c_string());
                }

                if (!handler_vdw_) {
                    s << "VDW functional lib could not be initialized";
                    TERMINATE(s);
                }

                if (num_spins__ == 1) {
                    handler_vdw_ = vdwxc_new(libxc_name__.c_string());
                } else {
                    handler_vdw_ = vdwxc_new_spin(libxc_name__.c_string());
                }

                auto &v1 = ctx__.unit_cell().lattice_vector(0);
                auto &v2 = ctx__.unit_cell().lattice_vector(1);
                auto &v3 = ctx__.unit_cell().lattice_vector(2);

                vdwxc_set_unit_cell(handler_vdw_,
                                    ctx__.fft().size(0),
                                    ctx__.fft().size(1),
                                    ctx__.fft().size(2),
                                    v1(0), v1(1), v1(2),
                                    v2(0), v2(1), v2(2),
                                    v3(0), v3(1), v3(2));

                if (ctx_.fft().comm().size() == 1) {
                    vdwxc_init_serial(handler_vdw_);
                } else {
                    vdwxc_set_communicator(handler_vdw_, ctx__.fft().comm());
                    vdwxc_init_mpi(handler_vdw_, ctx__.fft().comm());
                }
                vdw_functional_ = true;
            } else {
                /* it means that the functional does not exist either in vdw or xc libraries */
                std::stringstream s;
                s << "XC functional " << libxc_name__ << " is unknown";
                TERMINATE(s);
            }
            initialized_ = true;
        }

        XC_functional(XC_functional&& src__)
        {
            this->libxc_name_  = src__.libxc_name_;
            this->num_spins_   = src__.num_spins_;
            this->handler_     = src__.handler_;
            this->handler_vdw_ = src__.handler_vdw_;
            this->vdw_functional_ = src__.vdw_functional_;
            this->initialized_ = true;
            src__.initialized_ = false;
        }

        ~XC_functional()
        {
            if (initialized_) {
                if (this->vdw_functional_) {
                    vdwxc_finalize(&this->handler_vdw_);
                    return;
                }
                xc_func_end(&handler_);
            }
        }

        const std::string refs() const
        {
            std::stringstream s;
            if (wdv_functional_) {
                s << "A. H. Larsen, M. Kuisma, J. Löfgren, Y. Pouillon, P. Erhart, and P. Hyldgaard,\n";
                s << "libvdwxc: a library for exchange–correlation functionals in the vdW-DF family,\n";
                s << "Modelling Simul. Mater. Sci. Eng. 25, 065004 (2017),\n";
                s << "doi : 10.1088/1361-651X/aa7320\n";
                return s.str();
            }
            return XC_functional_base::refs();
        }

        int family() const
        {
            if (this->vdw_functional_ == true) {
                return XC_FAMILY_UNKNOWN;
            }
            return XC_functional_base::family();
        }

        bool is_vdw() const
        {
            return this->vdw_functional_;
        }

        int kind() const
        {
            if (this->vdw_functional_ == true) {
                return -1;
            }
            return XC_functional_base::kind();
        }

    /// get wan der walls contribution to the exchange term
    void get_vdw(const double* rho,
                 const double* sigma,
                 double* vrho,
                 double* vsigma,
                 double* energy__)
        {
            if (!is_vdw()) {
                TERMINATE("Error wrong vdw XC");
            }
            energy__[0] = vdwxc_calculate(handler_vdw_, rho, sigma, v);
        }

    /// get wan der walls contribution to the exchange term magnetic case
    void get_vdw(const double *rho_up, const double *rho_down,
                 const double *sigma_up, const double *sigma_down,
                 double *vrho_up, double *vrho_down,
                 double *vsigma_up, double *vsigma_down,
                 double *energy__)
        {
            if (!is_vdw()) {
                TERMINATE("Error wrong XC");
            }

            energy__[0] = vdwxc_calculate(handler_vdw_, rho_up, rho_down,
                                          sigma_up, , sigma_down,
                                          vrho_up, vrho_down,
                                          vsigma_up, vsigma_down);
        }
};

}

#endif // __XC_FUNCTIONAL_H__
