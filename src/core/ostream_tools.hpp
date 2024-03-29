/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file ostream_tools.hpp
 *
 *  \brief Output stream tools.
 */

#ifndef __OSTREAM_TOOLS_HPP__
#define __OSTREAM_TOOLS_HPP__

#include <ostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <sstream>

namespace sirius {

class null_stream_t : public std::ostream
{
  public:
    null_stream_t()
        : std::ostream(nullptr)
    {
    }
    null_stream_t(null_stream_t&&)
        : std::ostream(nullptr){};
};

null_stream_t&
null_stream();

inline std::string
boolstr(bool b__)
{
    if (b__) {
        return "true";
    } else {
        return "false";
    }
}

/// Horisontal bar.
class hbar
{
  private:
    int w_;
    char c_;

  public:
    hbar(int w__, char c__)
        : w_(w__)
        , c_(c__)
    {
    }
    int
    w() const
    {
        return w_;
    }
    char
    c() const
    {
        return c_;
    }
};

/// Inject horisontal bar to ostream.
inline std::ostream&
operator<<(std::ostream& out, hbar&& b)
{
    char prev = out.fill();
    out << std::setfill(b.c()) << std::setw(b.w()) << b.c() << std::setfill(prev);
    return out;
}

/// Floating-point formatting (precision and width).
class ffmt
{
  private:
    int w_;
    int p_;

  public:
    ffmt(int w__, int p__)
        : w_(w__)
        , p_(p__)
    {
    }
    int
    w() const
    {
        return w_;
    }
    int
    p() const
    {
        return p_;
    }
};

/// Inject floating point format to ostream.
inline std::ostream&
operator<<(std::ostream& out, ffmt&& f)
{
    out.precision(f.p());
    out.width(f.w());
    out.setf(std::ios_base::fixed, std::ios_base::floatfield);
    return out;
}

/// Print std::vector to ostream.
template <typename T>
inline std::ostream&
operator<<(std::ostream& out, std::vector<T>& v)
{
    if (v.size() == 0) {
        out << "{}";
    } else {
        out << "{";
        for (size_t i = 0; i < v.size() - 1; i++) {
            out << v[i] << ", ";
        }
        out << v.back() << "}";
    }
    return out;
}

/// Convert double to a string with a given precision.
inline std::string
double_to_string(double val, int precision = -1)
{
    char buf[100];

    double abs_val = std::abs(val);

    if (precision == -1) {
        if (abs_val > 1.0) {
            precision = 6;
        } else if (abs_val > 1e-14) {
            precision = int(-std::log(abs_val) / std::log(10.0)) + 7;
        } else {
            return std::string("0.0");
        }
    }

    std::stringstream fmt;
    fmt << "%." << precision << "f";

    int len = snprintf(buf, 100, fmt.str().c_str(), val);
    for (int i = len - 1; i >= 1; i--) {
        if (buf[i] == '0' && buf[i - 1] == '0') {
            buf[i] = 0;
        } else {
            break;
        }
    }
    return std::string(buf);
}

template <typename T, typename OUT>
inline void
print_checksum(std::string label__, T value__, OUT&& out__)
{
    out__ << "checksum(" << label__ << ") : " << ffmt(16, 8) << value__ << std::endl;
}

template <typename OUT>
inline void
print_hash(std::string label__, unsigned long long int hash__, OUT&& out__)
{
    out__ << "hashsum(" << label__ << ") : " << std::hex << hash__ << std::endl;
}

} // namespace sirius

#endif
