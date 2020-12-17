// Copyright (c) 2013-2018 Anton Kozhevnikov, Thomas Schulthess
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

/** \file smearing.hpp
 *
 *  \brief Smearing functions used in finding the band occupancies.
 */

#ifndef __SMEARING_HPP__
#define __SMEARING_HPP__

#include <cmath>
#include <functional>
#include <string>
#include <stdexcept>
#include <map>
#include <sstream>
#include <algorithm>

namespace smearing {

const double pi = 3.1415926535897932385;

enum class smearing_t
{
    gaussian,
    fermi_dirac
};

inline smearing_t get_smearing_t(std::string name__)
{
    std::transform(name__.begin(), name__.end(), name__.begin(), ::tolower);
    std::map<std::string, smearing_t> const m = {
        {"gaussian", smearing_t::gaussian},
        {"fermi_dirac", smearing_t::fermi_dirac}
    };

    if (m.count(name__) == 0) {
        std::stringstream s;
        s << "get_smearing_t(): wrong label of the smearing_t enumerator: " << name__;
        throw std::runtime_error(s.str());
     }
     return m.at(name__);
}

namespace gaussian {

inline double delta(double x__, double width__)
{
    double t = std::pow(x__ / width__, 2);
    return std::exp(-t) / std::sqrt(pi) / width__;
}

inline double occupancy(double x__, double width__)
{
    return 0.5 * (1 + std::erf(x__ / width__));
}

inline double entropy(double x__, double width__)
{
    double t = std::pow(x__ / width__, 2);
    return -std::exp(-t) * width__ / 2.0 / std::sqrt(pi);

}

} // namespace "gaussian"

namespace fermi_dirac {

inline double delta(double x__, double width__)
{
    double t = x__ / 2.0 / width__;
    return 1.0 / std::pow(std::exp(t) + std::exp(-t), 2) / width__;
}

inline double occupancy(double x__, double width__)
{
    return 1.0 - 1.0 / (1.0 + std::exp(x__ / width__));
}

inline double entropy(double x__, double width__)
{
    double t = x__ / width__;
    if (std::abs(t) > 50) {
        return 0.0;
    }
    double f = 1.0 / (1.0 + std::exp(t));
    return width__ * ((1 - f) * std::log(1 - f) + f * std::log(f));
}

} // namespace "fermi_dirac"

inline std::function<double(double)> occupancy(smearing_t type__, double width__)
{
    switch (type__) {
        case smearing_t::gaussian: {
            return [width__](double x__){return gaussian::occupancy(x__, width__);};
        }
        case smearing_t::fermi_dirac: {
            return [width__](double x__){return fermi_dirac::occupancy(x__, width__);};
        }
        default: {
            throw std::runtime_error("wrong type of smearing");
        }
    }
}

inline std::function<double(double)> entropy(smearing_t type__, double width__)
{
    switch (type__) {
        case smearing_t::gaussian: {
            return [width__](double x__){return gaussian::entropy(x__, width__);};
        }
        case smearing_t::fermi_dirac: {
            return [width__](double x__){return fermi_dirac::entropy(x__, width__);};
        }
        default: {
            throw std::runtime_error("wrong type of smearing");
        }
    }
}


//inline double fermi_dirac(double e)
//{
//    double kT = 0.001;
//    if (e > 100 * kT) {
//        return 0.0;
//    }
//    if (e < -100 * kT) {
//        return 1.0;
//    }
//    return (1.0 / (std::exp(e / kT) + 1.0));
//}
//
//inline double gaussian(double e, double delta)
//{
//    return 0.5 * (1 + std::erf(e / delta));
//}
//
//inline double gaussian_entropy(double e, double delta)
//{
//    const double pi = 3.1415926535897932385;
//    return std::exp(-std::pow(e / delta, 2)) * delta / 2 / std::sqrt(pi);
//}
//
//inline double cold(double e)
//{
//    const double pi = 3.1415926535897932385;
//
//    double a = -0.5634;
//
//    if (e < -10.0) {
//        return 1.0;
//    }
//    if (e > 10.0) {
//        return 0.0;
//    }
//
//    return 0.5 * (1 - std::erf(e)) - 1 - 0.25 * std::exp(-e * e) * (a + 2 * e - 2 * a * e * e) / std::sqrt(pi);
//}

}

#endif
