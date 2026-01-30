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
#include <string>
#include "core/r3/r3.hpp"
#include "xc_functional_base.hpp"
#include "core/mpi/communicator.hpp"
#include <spfft/transform.hpp>
#if defined(SIRIUS_USE_VDWXC)
#include <vdwxc.h>
#include <vdwxc_mpi.h>
#endif

namespace sirius {
static bool has_c_lda_term_{false};

/// Interface class to Libxc.
class XC_functional : public XC_functional_base
{
  private:
    // I can not use a generic void pointer because xc_func_type is a structure
    // while wdv_functional_ is a pointer over structure.
#if defined(SIRIUS_USE_VDWXC)
    vdwxc_data handler_vdw_{nullptr};
    bool vdw_functional_{false};
    int func_{-1};
    int num_spins_{-1};
    MPI_Comm comm_{MPI_COMM_WORLD};
#endif

    /* forbid copy constructor */
    XC_functional(const XC_functional& src) = delete;

    /* forbid assignment operator */
    XC_functional&
    operator=(const XC_functional& src) = delete;

    void
    create_handler(spfft::Transform const& fft__, r3::matrix<double> const& lattice_vectors__);

  public:
    /* we need the context because libvdwxc asks for lattice vectors and fft parameters */
    XC_functional(spfft::Transform const& fft__, r3::matrix<double> const& lattice_vectors__,
                  const std::string libxc_name__, const double weight__, int num_spins__);

    XC_functional(XC_functional&& src__);

    ~XC_functional();

    const std::string
    refs() const;

    int
    family() const;

    bool
    is_vdw() const;

    void
    vdw_update_unit_cell(spfft::Transform const& fft__, r3::matrix<double> const& lattice_vectors__);

    int
    kind() const;

#if defined(SIRIUS_USE_VDWXC)
    /// get van der walls contribution for the exchange term
    void
    get_vdw(const bool calculate_vdw_stress__, double* rho, double* sigma, double* vrho, double* vsigma,
            double* energy__, std::array<double, 9>& vdw_stress__);

    /// get van der walls contribution to the exchange term magnetic case
    void
    get_vdw(const bool calculate_vdw_stress__, double* rho_up, double* rho_down, double* sigma_up, double* sigma_down,
            double* vrho_up, double* vrho_down, double* vsigma_up, double* vsigma_down, double* energy__,
            std::array<double, 9>& vdw_stress__);

    void
    vdw_calculate_stress_kernel(const double vdw_energy__, const double volume__,
                                const std::array<double, 9>& stress_kernel__, r3::matrix<double>& vdw_stress_kernel__);

    void
    vdw_calculate_stress_potential(const double vdw_energy__, const double stress_vdwxc_pot__, const double volume__,
                                   r3::matrix<double>& vdw_stress_potential__);

    void
    vdw_calculate_stress_gradient(const std::array<double, 9>& stress_gradient__,
                                  r3::matrix<double>& vdw_stress_gradient__);
#endif
};

//=============================================================================
// Implementation
//=============================================================================

inline XC_functional::XC_functional(spfft::Transform const& fft__, r3::matrix<double> const& lattice_vectors__,
                                    const std::string libxc_name__, const double weight__, int num_spins__)
    : XC_functional_base(libxc_name__, weight__, num_spins__)
{

#if defined(SIRIUS_USE_VDWXC)
    /* return immediately if the functional_base class is initialized */
    if (this->libxc_initialized_) {
        /* the libvdwxc library only computes the non-local term. It DOES
         * not include the necessary LDA term by default so check that one
         * of the functional is indeed of the C_LDA type
         */
        if (libxc_name_.find("C_LDA") != std::string::npos) {
            has_c_lda_term_ = true;
        }
        return;
    }

    /* test if have van der walls functionals types */

    bool test = (libxc_name_ == "XC_FUNC_VDWDF");
    test      = test || (libxc_name_ == "XC_FUNC_VDWDF2");
    test      = test || (libxc_name_ == "XC_FUNC_VDWDFCX");

    if (libxc_name__ == "XC_FUNC_VDWDF") {
        func_ = FUNC_VDWDF;
    }

    if (libxc_name__ == "XC_FUNC_VDWDF2") {
        func_ = FUNC_VDWDF2;
    }

    if (libxc_name__ == "XC_FUNC_VDWDFCX") {
        func_ = FUNC_VDWDFCX;
    }

    if (!has_c_lda_term_ && (libxc_name__ != "XC_FUNC_VDWDF")) {
        std::stringstream s;
        s << "vdw xc: the vdw functional requires a correction LDA term to give meaningfull results\n";
        s << "        Please add it to the list of required functionals (typical value is XC_C_LDA_PW)\n";
        RTE_THROW(s);
    }

    num_spins_ = num_spins__;

    if (test) {
        create_handler(fft__, lattice_vectors__);
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
        /* it means that the functional does not exist either in vdw or xc
           libraries or that SIRIUS is not compiled with vdwxc support */
        std::stringstream s;
        s << "XC functional " << libxc_name__ << " is unknown";

        RTE_THROW(s);
    }
#endif /* SIRIUS_USE_VDWXC */
}

inline XC_functional::XC_functional(XC_functional&& src__)
    : XC_functional_base(std::move(src__))
{
#if defined(SIRIUS_USE_VDWXC)
    this->handler_vdw_    = std::move(src__.handler_vdw_);
    this->vdw_functional_ = src__.vdw_functional_;
    this->func_           = src__.func_;
    this->num_spins_      = src__.num_spins_;
    src__.handler_vdw_    = nullptr;
    src__.vdw_functional_ = false;
#endif
}

inline XC_functional::~XC_functional()
{
#if defined(SIRIUS_USE_VDWXC)
    if (handler_vdw_) {
        vdwxc_finalize(&this->handler_vdw_);
        this->vdw_functional_ = false;
        this->handler_vdw_    = nullptr;
        return;
    }
#endif
}

inline const std::string
XC_functional::refs() const
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

inline int
XC_functional::family() const
{
#if defined(SIRIUS_USE_VDWXC)
    if (this->vdw_functional_ == true) {
        return XC_FAMILY_UNKNOWN;
    }
#endif
    return XC_functional_base::family();
}

inline bool
XC_functional::is_vdw() const
{
#if defined(SIRIUS_USE_VDWXC)
    return this->vdw_functional_;
#else
    return false;
#endif
}

inline void
XC_functional::vdw_update_unit_cell(spfft::Transform const& fft__, r3::matrix<double> const& lattice_vectors__)
{
#ifdef SIRIUS_USE_VDWXC
    if (is_vdw()) {
        // we need to add a test for checking that the fft parameters have
        // not changed. Right now I assume that fft box can change and I
        // recreate the handler from scratch

        if (this->handler_vdw_ != nullptr) {
            vdwxc_finalize(&this->handler_vdw_);
            this->handler_vdw_ = nullptr;
        }
        create_handler(fft__, lattice_vectors__);
    }
#endif
}

inline int
XC_functional::kind() const
{

#if defined(SIRIUS_USE_VDWXC)
    if (this->vdw_functional_ == true) {
        return -1;
    }
#endif
    return XC_functional_base::kind();
}

#if defined(SIRIUS_USE_VDWXC)
inline void
XC_functional::get_vdw(const bool calculate_vdw_stress__, double* rho, double* sigma, double* vrho, double* vsigma,
                       double* energy__, std::array<double, 9>& vdw_stress__)
{
    if (!is_vdw()) {
        RTE_THROW("Error wrong vdw XC");
    }

    if (rho != nullptr) {
        // vdwxc will raise an exception if any input is a nullpointer
        if (!calculate_vdw_stress__) {
            energy__[0] = vdwxc_calculate(handler_vdw_, rho, sigma, vrho, vsigma);
        } else {
            std::fill(vdw_stress__.begin(), vdw_stress__.end(), 0.0);
            energy__[0] = vdwxc_stress(handler_vdw_, vdw_stress__.data(), rho, sigma, vrho, vsigma);
        }
    }

    auto comm = mpi::Communicator(this->comm_);
    comm.allreduce<double, mpi::op_t::sum>(energy__, 1);

    if (calculate_vdw_stress__) {
        comm.allreduce<double, mpi::op_t::sum>(vdw_stress__.data(), 9);
    }
}

inline void
XC_functional::get_vdw(const bool calculate_vdw_stress__, double* rho_up, double* rho_down, double* sigma_up,
                       double* sigma_down, double* vrho_up, double* vrho_down, double* vsigma_up, double* vsigma_down,
                       double* energy__, std::array<double, 9>& vdw_stress__)
{
    if (!is_vdw()) {
        RTE_THROW("Error wrong XC");
    }

    if (!calculate_vdw_stress__) {
        energy__[0] = vdwxc_calculate_spin(handler_vdw_, rho_up, rho_down, sigma_up, sigma_down, vrho_up, vrho_down,
                                           vsigma_up, vsigma_down);
    } else {
        std::fill(vdw_stress__.begin(), vdw_stress__.end(), 0.0);
        energy__[0] = vdwxc_stress_spin(handler_vdw_, vdw_stress__.data(), rho_up, rho_down, sigma_up, sigma_down,
                                        vrho_up, vrho_down, vsigma_up, vsigma_down);
    }

    auto comm = mpi::Communicator(this->comm_);

    comm.allreduce<double, mpi::op_t::sum>(energy__, 1);
    if (calculate_vdw_stress__) {
        comm.allreduce<double, mpi::op_t::sum>(vdw_stress__.data(), 9);
    }
}

inline void
XC_functional::vdw_calculate_stress_kernel(const double vdw_energy__, const double volume__,
                                           const std::array<double, 9>& stress_kernel__,
                                           r3::matrix<double>& vdw_stress_kernel__)
{
    // libvdwxc adds $E^{nl}_c$ to the kernel term but the energy
    // correction is already included in the xc stress tensor before
    // calling this function.

    vdw_stress_kernel__(0, 0) = -vdw_energy__ / volume__;
    vdw_stress_kernel__(1, 1) = -vdw_energy__ / volume__;
    vdw_stress_kernel__(2, 2) = -vdw_energy__ / volume__;

    for (int nu = 0; nu < 3; nu++) {
        for (int mu = 0; mu < 3; mu++) {
            /* Specific to the non-local corrections  */
            vdw_stress_kernel__(mu, nu) += stress_kernel__[3 * nu + mu] / volume__;
            vdw_stress_kernel__(mu, nu) *= weight();
        }
    }
}

inline void
XC_functional::vdw_calculate_stress_potential(const double vdw_energy__, const double stress_vdwxc_pot__,
                                              const double volume__, r3::matrix<double>& vdw_stress_potential__)
{
    /*
     * Compute 1/\Omega \int E - v d^3 r. vdw_energy_ is negative in
     * libvdwxc but positive in QE
     */
    vdw_stress_potential__(0, 0) = (vdw_energy__ - stress_vdwxc_pot__) * weight() / volume__;
    vdw_stress_potential__(1, 1) = (vdw_energy__ - stress_vdwxc_pot__) * weight() / volume__;
    vdw_stress_potential__(2, 2) = (vdw_energy__ - stress_vdwxc_pot__) * weight() / volume__;
}

inline void
XC_functional::vdw_calculate_stress_gradient(const std::array<double, 9>& stress_gradient__,
                                             r3::matrix<double>& vdw_stress_gradient__)
{
    for (int nu = 0; nu < 3; nu++) {
        for (int mu = 0; mu < 3; mu++) {
            /*
             * The factor -2.0 comes from using |\nabla n|^2 during the
             * calculations of exchange-correlation potential, instead
             * of |\nabla n| in Sabatini paper.
             */
            vdw_stress_gradient__(mu, nu) = -2.0 * stress_gradient__[3 * nu + mu] * weight();
        }
    }
}
#endif

inline void
XC_functional::create_handler(spfft::Transform const& fft__, r3::matrix<double> const& lattice_vectors__)
{
#if defined(SIRIUS_USE_VDWXC)
    if (num_spins_ == 1) {
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

    vdwxc_set_unit_cell(handler_vdw_, fft__.dim_z(), fft__.dim_y(), fft__.dim_x(), v3[0], v3[1], v3[2], v2[0], v2[1],
                        v2[2], v1[0], v1[1], v1[2]);

    if (mpi::Communicator(fft__.communicator()).size() == 1) {
        vdwxc_init_serial(handler_vdw_);
    } else {
        vdwxc_init_mpi(handler_vdw_, fft__.communicator());
    }
    comm_           = fft__.communicator();
    vdw_functional_ = true;
#endif /* SIRIUS_USE_VDWXC */
    return;
}

} // namespace sirius

#endif // __XC_FUNCTIONAL_H__
