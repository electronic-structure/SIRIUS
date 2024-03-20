/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file xc_functional.hpp
 *
 *  \brief Contains implementation of sirius::XC_functional class.
 */

#ifndef __XC_FUNCTIONAL_HPP__
#define __XC_FUNCTIONAL_HPP__

#include <xc.h>
#include <string.h>
#include "core/r3/r3.hpp"
#include "xc_functional_base.hpp"
#include "core/fft/fft.hpp"
#if defined(SIRIUS_USE_VDWXC)
#include <vdwxc.h>
#if SIRIUS_HAVE_VDWXC_MPI
#include <vdwxc_mpi.h>
#endif
#endif

namespace sirius {

/// Interface class to Libxc.
class XC_functional : public XC_functional_base
{
  private:
    // I can not use a generic void pointer because xc_func_type is a structure
    // while wdv_functional_ is a pointer over structure.
#if defined(SIRIUS_USE_VDWXC)
    vdwxc_data handler_vdw_{nullptr};
    bool vdw_functional_{false};
#endif
    /* forbid copy constructor */
    XC_functional(const XC_functional& src) = delete;

    /* forbid assignment operator */
    XC_functional&
    operator=(const XC_functional& src) = delete;

  public:
    /* we need the context because libvdwxc asks for lattice vectors and fft parameters */
    XC_functional(spfft::Transform const& fft__, r3::matrix<double> const& lattice_vectors__,
                  const std::string libxc_name__, int num_spins__)
        : XC_functional_base(libxc_name__, num_spins__)
    {

#if defined(SIRIUS_USE_VDWXC)
        /* return immediately if the functional_base class is initialized */
        if (this->libxc_initialized_) {
            return;
        }

        /* test if have van der walls functionals types */

        bool test = (libxc_name_ == "XC_FUNC_VDWDF");
        test      = test || (libxc_name_ == "XC_FUNC_VDWDF2");
        test      = test || (libxc_name_ == "XC_FUNC_VDWDFCX");

        int func_ = -1;

        if (libxc_name__ == "XC_FUNC_VDWDF") {
            func_ = FUNC_VDWDF;
        }

        if (libxc_name__ == "XC_FUNC_VDWDF2") {
            func_ = FUNC_VDWDF2;
        }

        if (libxc_name__ == "XC_FUNC_VDWDFCX") {
            func_ = FUNC_VDWDFCX;
        }

        if (test) {
            if (num_spins__ == 1) {
                // non magnetic case
                handler_vdw_ = vdwxc_new(func_);
            } else {
                // magnetic case
                handler_vdw_ = vdwxc_new_spin(func_);
            }

            if (!handler_vdw_) {
                std::stringstream s;
                s << "VDW functional lib could not be initialized";
                RTE_THROW(s);
            }

            double v1[3] = {lattice_vectors__(0, 0), lattice_vectors__(1, 0), lattice_vectors__(2, 0)};
            double v2[3] = {lattice_vectors__(0, 1), lattice_vectors__(1, 1), lattice_vectors__(2, 1)};
            double v3[3] = {lattice_vectors__(0, 2), lattice_vectors__(1, 2), lattice_vectors__(2, 2)};
            vdwxc_set_unit_cell(handler_vdw_, fft__.dim_x(), fft__.dim_y(), fft__.dim_z(), v1[0], v1[1], v1[2], v2[0],
                                v2[1], v2[2], v3[0], v3[1], v3[2]);
            if (mpi::Communicator(fft__.communicator()).size() == 1) {
                vdwxc_init_serial(handler_vdw_);
            } else {
#if SIRIUS_HAVE_VDWXC_MPI
                vdwxc_init_mpi(handler_vdw_, fft__.communicator());
#else
                vdwxc_init_serial(handler_vdw_);
#endif
            }
            vdw_functional_ = true;
            return;
        } else {
            /* it means that the functional does not exist either in vdw or xc libraries */
            std::stringstream s;
            s << "XC functional " << libxc_name__ << " is unknown";
            RTE_THROW(s);
        }
#else
        if (this->libxc_initialized_) {
            return;
        } else {
            /* it means that the functional does not exist either in vdw or xc libraries */
            std::stringstream s;
            s << "XC functional " << libxc_name__ << " is unknown";
            RTE_THROW(s);
        }
#endif /* SIRIUS_USE_VDWXC */
    }

    XC_functional(XC_functional&& src__)
        : XC_functional_base(std::move(src__))
    {
#if defined(SIRIUS_USE_VDWXC)
        this->handler_vdw_    = src__.handler_vdw_;
        this->vdw_functional_ = src__.vdw_functional_;
        src__.vdw_functional_ = false;
#endif
    }

    ~XC_functional()
    {
#if defined(SIRIUS_USE_VDWXC)
        if (this->vdw_functional_) {
            vdwxc_finalize(&this->handler_vdw_);
            this->vdw_functional_ = false;
            return;
        }
#endif
    }

    const std::string
    refs() const
    {
#if defined(SIRIUS_USE_VDWXC)
        std::stringstream s;
        if (vdw_functional_) {
            s << "==============================================================================\n";
            s << "                                                                              \n";
            s << "Warning : these functionals should be used in combination with GGA functionals\n";
            s << "                                                                              \n";
            s << "==============================================================================\n";
            s << "\n";
            s << "A. H. Larsen, M. Kuisma, J. Löfgren, Y. Pouillon, P. Erhart, and P. Hyldgaard, ";
            s << "Modelling Simul. Mater. Sci. Eng. 25, 065004 (2017) (10.1088/1361-651X/aa7320)\n";
            return s.str();
        }
#endif
        return XC_functional_base::refs();
    }

    int
    family() const
    {
#if defined(SIRIUS_USE_VDWXC)
        if (this->vdw_functional_ == true) {
            return XC_FAMILY_UNKNOWN;
        }
#endif
        return XC_functional_base::family();
    }

    bool
    is_vdw() const
    {
#if defined(SIRIUS_USE_VDWXC)
        return this->vdw_functional_;
#else
        return false;
#endif
    }

    void
    vdw_update_unit_cell(spfft::Transform const& fft__, r3::matrix<double> const& lattice_vectors__)
    {
#ifdef SIRIUS_USE_VDWXC
        if (is_vdw()) {
            double v1[3] = {lattice_vectors__(0, 0), lattice_vectors__(1, 0), lattice_vectors__(2, 0)};
            double v2[3] = {lattice_vectors__(0, 1), lattice_vectors__(1, 1), lattice_vectors__(2, 1)};
            double v3[3] = {lattice_vectors__(0, 2), lattice_vectors__(1, 2), lattice_vectors__(2, 2)};

            vdwxc_set_unit_cell(handler_vdw_, fft__.dim_x(), fft__.dim_y(), fft__.dim_z(), v1[0], v1[1], v1[2], v2[0],
                                v2[1], v2[2], v3[0], v3[1], v3[2]);
        }
#endif
    }

    int
    kind() const
    {

#if defined(SIRIUS_USE_VDWXC)
        if (this->vdw_functional_ == true) {
            return -1;
        }
#endif
        return XC_functional_base::kind();
    }

#if defined(SIRIUS_USE_VDWXC)
    /// get van der walls contribution for the exchange term
    void
    get_vdw(double* rho, double* sigma, double* vrho, double* vsigma, double* energy__)
    {
        if (!is_vdw()) {
            RTE_THROW("Error wrong vdw XC");
        }
        energy__[0] = vdwxc_calculate(handler_vdw_, rho, sigma, vrho, vsigma);
    }

    /// get van der walls contribution to the exchange term magnetic case
    void
    get_vdw(double* rho_up, double* rho_down, double* sigma_up, double* sigma_down, double* vrho_up, double* vrho_down,
            double* vsigma_up, double* vsigma_down, double* energy__)
    {
        if (!is_vdw()) {
            RTE_THROW("Error wrong XC");
        }

        energy__[0] = vdwxc_calculate_spin(handler_vdw_, rho_up, rho_down, sigma_up, sigma_down, vrho_up, vrho_down,
                                           vsigma_up, vsigma_down);
    }
#endif
};

} // namespace sirius

#endif // __XC_FUNCTIONAL_H__
