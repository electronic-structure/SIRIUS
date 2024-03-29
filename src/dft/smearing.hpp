/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file smearing.hpp
 *
 *  \brief Smearing functions used in finding the band occupancies.
 */

#ifndef __SMEARING_HPP__
#define __SMEARING_HPP__

#include <functional>
#include <string>
#include <stdexcept>
#include <map>
#include <sstream>
#include <algorithm>

namespace sirius {

/// Occupancy smearing functions.
namespace smearing {

enum class smearing_t
{
    gaussian,
    fermi_dirac,
    cold,
    methfessel_paxton
};

inline smearing_t
get_smearing_t(std::string name__)
{
    std::transform(name__.begin(), name__.end(), name__.begin(), ::tolower);
    std::map<std::string, smearing_t> const m = {
            {"gaussian", smearing_t::gaussian},
            {"fermi_dirac", smearing_t::fermi_dirac},
            {"cold", smearing_t::cold},
            {"methfessel_paxton", smearing_t::methfessel_paxton},
    };

    if (m.count(name__) == 0) {
        std::stringstream s;
        s << "get_smearing_t(): wrong label of the smearing_t enumerator: " << name__;
        throw std::runtime_error(s.str());
    }
    return m.at(name__);
}

struct gaussian
{
    static double
    delta(double x__, double width__);
    static double
    occupancy(double x__, double width__);
    static double
    entropy(double x__, double width__);
};

struct fermi_dirac
{
    static double
    dxdelta(double x__, double width__);
    static double
    delta(double x__, double width__);
    static double
    occupancy(double x__, double width__);
    static double
    entropy(double x__, double width__);
};

struct cold
{
    static double
    delta(double x__, double width__);
    static double
    occupancy(double x__, double width__);
    static double
    entropy(double x__, double width__);

    /** Second derivative of the occupation function \f$f(x,w)\f$.
     *   \f[
     *     \frac{\partial^2 f(x,w)}{\partial x^2} = \frac{e^{-y^2} \left(2 \sqrt{2} y^2-2 y-\sqrt{2}\right)}{\sqrt{\pi }
     * w^2}, \qquad y=\frac{x}{w} - \frac{1}{\sqrt{2}} \f]
     */
    static double
    dxdelta(double x__, double width__);
};

/** Methfessel-Paxton smearing.
 *
 *  Methfessel, M., & Paxton, High-precision sampling for Brillouin-zone
 *  integration in metals. , 40(6), 3616–3621.
 *  http://dx.doi.org/10.1103/PhysRevB.40.3616
 */
struct methfessel_paxton
{
    static double
    dxdelta(double x__, double width__, int n__);
    static double
    delta(double x__, double width__, int n__);
    static double
    occupancy(double x__, double width__, int n__);
    static double
    entropy(double x__, double width__, int n__);
};

inline std::function<double(double)>
occupancy(smearing_t type__, double width__)
{
    switch (type__) {
        case smearing_t::gaussian: {
            return [width__](double x__) { return gaussian::occupancy(x__, width__); };
        }
        case smearing_t::fermi_dirac: {
            return [width__](double x__) { return fermi_dirac::occupancy(x__, width__); };
        }
        case smearing_t::cold: {
            return [width__](double x__) { return cold::occupancy(x__, width__); };
        }
        case smearing_t::methfessel_paxton: {
            return [width__](double x__) { return methfessel_paxton::occupancy(x__, width__, 1); };
        }
        default: {
            throw std::runtime_error("wrong type of smearing");
        }
    }
}

inline std::function<double(double)>
entropy(smearing_t type__, double width__)
{
    switch (type__) {
        case smearing_t::gaussian: {
            return [width__](double x__) { return gaussian::entropy(x__, width__); };
        }
        case smearing_t::fermi_dirac: {
            return [width__](double x__) { return fermi_dirac::entropy(x__, width__); };
        }
        case smearing_t::cold: {
            return [width__](double x__) { return cold::entropy(x__, width__); };
        }
        case smearing_t::methfessel_paxton: {
            return [width__](double x__) { return methfessel_paxton::entropy(x__, width__, 1); };
        }
        default: {
            throw std::runtime_error("wrong type of smearing");
        }
    }
}

inline std::function<double(double)>
delta(smearing_t type__, double width__)
{
    switch (type__) {
        case smearing_t::gaussian: {
            throw std::runtime_error("not available");
        }
        case smearing_t::fermi_dirac: {
            return [width__](double x__) { return fermi_dirac::delta(x__, width__); };
        }
        case smearing_t::cold: {
            return [width__](double x__) { return cold::delta(x__, width__); };
        }
        case smearing_t::methfessel_paxton: {
            return [width__](double x__) { return methfessel_paxton::delta(x__, width__, 1); };
        }
        default: {
            throw std::runtime_error("wrong type of smearing");
        }
    }
}

inline std::function<double(double)>
dxdelta(smearing_t type__, double width__)
{
    switch (type__) {
        case smearing_t::gaussian: {
            throw std::runtime_error("not available");
        }
        case smearing_t::fermi_dirac: {
            return [width__](double x__) { return fermi_dirac::dxdelta(x__, width__); };
        }
        case smearing_t::cold: {
            return [width__](double x__) { return cold::dxdelta(x__, width__); };
        }
        case smearing_t::methfessel_paxton: {
            return [width__](double x__) { return methfessel_paxton::dxdelta(x__, width__, 1); };
        }
        default: {
            throw std::runtime_error("wrong type of smearing");
        }
    }
}

} // namespace smearing

} // namespace sirius

#endif
