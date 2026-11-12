/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file xc_functional.hpp
 *
 *  \brief Contains implementation of sirius::XC_functional_base class.
 */

#ifndef __XC_FUNCTIONAL_BASE_HPP__
#define __XC_FUNCTIONAL_BASE_HPP__

#include <xc.h>

/* libxc >= 7 split the functional definition in a different file from xc.h */

#if (XC_MAJOR_VERSION >= 7)
#include <xc_funcs.h>
#endif

#include <string.h>
#include <memory>
#include <map>
#include <stdexcept>
#include <iostream>
#include "core/rte/rte.hpp"
#include "core/ostream_tools.hpp"

namespace sirius {

/// Map from the name of a libxc functional to its id (only functionals known to the compiled libxc).
extern const std::map<std::string, int> libxc_functionals;

/// Interface class to Libxc.
class XC_functional_base
{
  protected:
    std::string libxc_name_;
    double weight_{1.0};
    int num_spins_;

    std::unique_ptr<xc_func_type> handler_{nullptr};

    bool libxc_initialized_{false};

  private:
    /* forbid copy constructor */
    XC_functional_base(const XC_functional_base& src) = delete;

    /* forbid assignment operator */
    XC_functional_base&
    operator=(const XC_functional_base& src) = delete;

  public:
    XC_functional_base(const std::string libxc_name__, const double weight__, int num_spins__)
        : libxc_name_(libxc_name__)
        , weight_(weight__)
        , num_spins_(num_spins__)
    {
        /* check if functional name is in list */
        if (libxc_functionals.count(libxc_name_) == 0 && libxc_name_ != "XC_GGA_DEBUG" &&
            libxc_name_ != "XC_LDA_DEBUG") {
            /* if not just return since van der walls functionals can be
             * used */
            libxc_initialized_ = false;
            return;
        }

        auto ns = (num_spins__ == 1) ? XC_UNPOLARIZED : XC_POLARIZED;

        if (libxc_name_ != "XC_GGA_DEBUG" && libxc_name_ != "XC_LDA_DEBUG") {
            handler_ = std::make_unique<xc_func_type>();

            /* init xc functional handler */
            if (xc_func_init(handler_.get(), libxc_functionals.at(libxc_name_), ns) != 0) {
                RTE_THROW("xc_func_init() failed");
            }
            if (handler_->info->family == XC_FAMILY_MGGA) {
                // Fermi-hole clipping does not include its chain rule in Libxc's Vxc.
                xc_func_set_fhc_enforcement(handler_.get(), 0);
            }
        }
        libxc_initialized_ = true;
    }

    XC_functional_base(XC_functional_base&& src__)
    {
        this->libxc_name_        = src__.libxc_name_;
        this->weight_            = src__.weight_;
        this->num_spins_         = src__.num_spins_;
        this->handler_           = std::move(src__.handler_);
        this->libxc_initialized_ = src__.libxc_initialized_;
        src__.libxc_initialized_ = false;
    }

    ~XC_functional_base()
    {
        if (handler_) {
            xc_func_end(handler_.get());
        }
    }

    const std::string
    name() const
    {
        if (handler_) {
            return std::string(handler_->info->name);
        } else {
            return libxc_name_;
        }
    }

    const std::string
    refs() const
    {
        if (handler_) {
            std::stringstream s;
            for (int i = 0; handler_->info->refs[i] != NULL; i++) {
                s << std::string(handler_->info->refs[i]->ref);
                if (strlen(handler_->info->refs[i]->doi) > 0) {
                    s << " (" << std::string(handler_->info->refs[i]->doi) << ")";
                }
                s << std::endl;
            }
            return s.str();
        } else {
            return "";
        }
    }

    int
    family() const
    {
        if (handler_) {
            return handler_->info->family;
        } else {
            if (libxc_name_ == "XC_GGA_DEBUG") {
                return XC_FAMILY_GGA;
            } else {
                return XC_FAMILY_LDA;
            }
        }
    }

    xc_func_type const*
    handler() const
    {
        if (handler_) {
            return handler_.get();
        }

        throw std::runtime_error("attempt to access nullptr in xc_functional_base::handler");
    }

    xc_func_type*
    handler()
    {
        return const_cast<xc_func_type*>(static_cast<XC_functional_base const&>(*this).handler());
    }

    bool
    is_lda() const
    {
        return family() == XC_FAMILY_LDA;
    }

    bool
    is_gga() const
    {
        return family() == XC_FAMILY_GGA;
    }

    bool
    is_meta_gga() const
    {
        return family() == XC_FAMILY_MGGA || family() == XC_FAMILY_HYB_MGGA;
    }

    /// Positive-tau meta-GGA evaluation in Libxc's point-major spin layout.
    /** Laplacian-dependent and hybrid functionals require additional operators. */
    void
    get_meta(int size__, double const* rho__, double const* sigma__, double const* tau__, double* vrho__,
             double* vsigma__, double* vtau__, double* exc__) const
    {
        if (size__ < 0 || family() != XC_FAMILY_MGGA || !handler_ ||
            (handler_->info->flags & XC_FLAGS_NEEDS_LAPLACIAN) || !(handler_->info->flags & XC_FLAGS_HAVE_EXC) ||
            !(handler_->info->flags & XC_FLAGS_HAVE_VXC)) {
            RTE_THROW("meta-GGA evaluation requires a nonhybrid energy functional without a density Laplacian");
        }
        if (size__ == 0) {
            return;
        }
        for (int i = 0; i < size__ * num_spins_; i++) {
            if (!std::isfinite(rho__[i]) || !std::isfinite(tau__[i]) || rho__[i] < 0 || tau__[i] < 0) {
                RTE_THROW("meta-GGA evaluation requires finite nonnegative spin densities and positive tau");
            }
        }
        for (int i = 0; i < size__ * (num_spins_ == 1 ? 1 : 3); i++) {
            if (!std::isfinite(sigma__[i])) {
                RTE_THROW("meta-GGA evaluation requires finite gradient invariants");
            }
        }
        std::vector<double> lapl(size__ * num_spins_, 0), vlapl(lapl.size());
        xc_mgga_exc_vxc(handler_.get(), size__, rho__, sigma__, lapl.data(), tau__, exc__, vrho__, vsigma__,
                        vlapl.data(), vtau__);
    }

    int
    kind() const
    {
        if (handler_) {
            return handler_->info->kind;
        } else {
            return XC_EXCHANGE_CORRELATION;
        }
    }

    bool
    is_exchange() const
    {
        return kind() == XC_EXCHANGE;
    }

    bool
    is_correlation() const
    {
        return kind() == XC_CORRELATION;
    }

    bool
    is_exchange_correlation() const
    {
        return kind() == XC_EXCHANGE_CORRELATION;
    }

    const double
    weight() const
    {
        return weight_;
    }

    /// Get LDA contribution.
    void
    get_lda(const int size, const double* rho, double* v, double* e) const
    {
        if (family() != XC_FAMILY_LDA) {
            RTE_THROW("wrong XC");
        }

        /* check density */
        for (int i = 0; i < size; i++) {
            if (rho[i] < 0) {
                std::stringstream s;
                s << "rho is negative : " << double_to_string(rho[i]);
                RTE_THROW(s);
            }
        }

        if (handler_) {
            xc_lda_exc_vxc(handler_.get(), size, rho, e, v);
        } else {
            for (int i = 0; i < size; i++) {
                /* E = \int e * rho * dr */
                e[i] = -0.001 * (rho[i] * rho[i]);
                /* var E / var rho = (de/drho) rho + e */
                v[i] = -0.002 * rho[i] * rho[i] + e[i];
            }
        }
    }

    /// Get LSDA contribution.
    void
    get_lda(const int size, const double* rho_up, const double* rho_dn, double* v_up, double* v_dn, double* e) const
    {
        if (family() != XC_FAMILY_LDA) {
            RTE_THROW("wrong XC");
        }

        std::vector<double> rho_ud(size * 2);
        /* check and rearrange density */
        for (int i = 0; i < size; i++) {
            if (rho_up[i] < 0 || rho_dn[i] < 0) {
                std::stringstream s;
                s << "rho is negative : " << double_to_string(rho_up[i]) << " " << double_to_string(rho_dn[i]);
                RTE_THROW(s);
            }

            rho_ud[2 * i]     = rho_up[i];
            rho_ud[2 * i + 1] = rho_dn[i];
        }

        if (handler_) {
            std::vector<double> v_ud(size * 2);

            xc_lda_exc_vxc(handler_.get(), size, &rho_ud[0], &e[0], &v_ud[0]);

            /* extract potential */
            for (int i = 0; i < size; i++) {
                v_up[i] = v_ud[2 * i];
                v_dn[i] = v_ud[2 * i + 1];
            }
        } else {
            for (int i = 0; i < size; i++) {
                e[i]    = -0.001 * (rho_up[i] * rho_up[i] + rho_dn[i] * rho_dn[i]);
                v_up[i] = -0.002 * rho_up[i] * (rho_up[i] + rho_dn[i]) + e[i];
                v_dn[i] = -0.002 * rho_dn[i] * (rho_up[i] + rho_dn[i]) + e[i];
            }
        }
    }

    /// Get GGA contribution.
    void
    get_gga(const int size, const double* rho, const double* sigma, double* vrho, double* vsigma, double* e) const
    {
        if (family() != XC_FAMILY_GGA)
            RTE_THROW("wrong XC");

        /* check density */
        for (int i = 0; i < size; i++) {
            if (rho[i] < 0.0) {
                std::stringstream s;
                s << "rho is negative : " << double_to_string(rho[i]);
                RTE_THROW(s);
            }
        }

        if (handler_) {
            xc_gga_exc_vxc(handler_.get(), size, rho, sigma, e, vrho, vsigma);
        } else {
            for (int i = 0; i < size; i++) {
                e[i]      = -0.001 * (rho[i] * sigma[i]);
                vrho[i]   = -0.001 * sigma[i];
                vsigma[i] = -0.001 * rho[i];
            }
        }
    }

    /// Get spin-resolved GGA contribution.
    void
    get_gga(const int size, const double* rho_up, const double* rho_dn, const double* sigma_uu, const double* sigma_ud,
            const double* sigma_dd, double* vrho_up, double* vrho_dn, double* vsigma_uu, double* vsigma_ud,
            double* vsigma_dd, double* e) const
    {
        if (family() != XC_FAMILY_GGA) {
            RTE_THROW("wrong XC");
        }

        std::vector<double> rho(2 * size);
        std::vector<double> sigma(3 * size);
        /* check and rearrange density */
        /* rearrange sigma as well */
        for (int i = 0; i < size; i++) {
            if (rho_up[i] < 0 || rho_dn[i] < 0) {
                std::stringstream s;
                s << "rho is negative : " << double_to_string(rho_up[i]) << " " << double_to_string(rho_dn[i]);
                RTE_THROW(s);
            }

            rho[2 * i]     = rho_up[i];
            rho[2 * i + 1] = rho_dn[i];

            sigma[3 * i]     = sigma_uu[i];
            sigma[3 * i + 1] = sigma_ud[i];
            sigma[3 * i + 2] = sigma_dd[i];
        }

        std::vector<double> vrho(2 * size);
        std::vector<double> vsigma(3 * size);

        if (handler_) {
            xc_gga_exc_vxc(handler_.get(), size, &rho[0], &sigma[0], e, &vrho[0], &vsigma[0]);

            /* extract vrho and vsigma */
            for (int i = 0; i < size; i++) {
                vrho_up[i] = vrho[2 * i];
                vrho_dn[i] = vrho[2 * i + 1];

                vsigma_uu[i] = vsigma[3 * i];
                vsigma_ud[i] = vsigma[3 * i + 1];
                vsigma_dd[i] = vsigma[3 * i + 2];
            }
        } else {
            auto h1 = std::make_unique<xc_func_type>();

            /* init xc functional handler */
            if (xc_func_init(h1.get(), XC_LDA_C_PZ, 2) != 0) {
                RTE_THROW("xc_func_init() failed");
            }

            xc_lda_exc_vxc(h1.get(), size, &rho[0], e, &vrho[0]);
            /* extract vrho and vsigma */
            for (int i = 0; i < size; i++) {
                vrho_up[i] = vrho[2 * i];
                vrho_dn[i] = vrho[2 * i + 1];

                vsigma_uu[i] = 0;
                vsigma_ud[i] = 0;
                vsigma_dd[i] = 0;
            }

            for (int i = 0; i < size; i++) {
                e[i] += 0.001 * (sigma_uu[i] + sigma_ud[i] + sigma_dd[i]);
                vsigma_uu[i] = -0.001;
                vsigma_ud[i] = -0.001;
                vsigma_dd[i] = -0.001;
            }
        }
    }

    /// set density threshold of libxc, if density is below tre, all xc output will be set to 0.
    void
    set_dens_threshold(double tre)
    {
#if XC_MAJOR_VERSION >= 4
        xc_func_set_dens_threshold(this->handler(), tre);
#else
        std::cout << "set_dens_threshold not available in old libxc versions, install at least 4.2.3"
                  << "\n";
#endif
    }
};
} // namespace sirius
#endif // __XC_FUNCTIONAL_BASE_H__
