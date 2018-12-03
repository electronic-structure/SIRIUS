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

/** \file utils.hpp
 *
 *  \brief A collection of utility functions.
 *
 *  General purpose header file containing various helper utility functions. This file should only include
 *  standard headers without any code-specific headers.
 */

#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <cassert>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <sys/time.h>
#include <unistd.h>
#include <complex>
#include "json.hpp"

/// Namespace for simple utility functions.
namespace utils {

/// Terminate the execution and print the info message.
inline void terminate(const char* file_name__, int line_number__, const std::string& message__)
{
    std::stringstream s;
    s << "\n=== Fatal error at line " << line_number__ << " of file " << file_name__ << " ===\n";
    s << message__ << "\n\n";
    throw std::runtime_error(s.str());
}

/// Terminate the execution and print the info message.
inline void terminate(const char* file_name__, int line_number__, const std::stringstream& message__)
{
    terminate(file_name__, line_number__, message__.str());
}

/// Issue a warning message.
inline void warning(const char* file_name__, int line_number__, const std::string& message__)
{
    printf("\n=== Warning at line %i of file %s ===\n", line_number__, file_name__);
    printf("%s\n\n", message__.c_str());
}

/// Issue a warning message.
inline void warning(const char* file_name__, int line_number__, const std::stringstream& message__)
{
    warning(file_name__, line_number__, message__.str());
}

#define TERMINATE(msg) utils::terminate(__FILE__, __LINE__, msg);

#define WARNING(msg) utils::warning(__FILE__, __LINE__, msg);

#define STOP() TERMINATE("terminated by request")

inline void print_checksum(std::string label__, double cs__)
{
    printf("checksum(%s): %18.12f\n", label__.c_str(), cs__);
}

inline void print_checksum(std::string label__, std::complex<double> cs__)
{
    printf("checksum(%s): %18.12f %18.12f\n", label__.c_str(), cs__.real(), cs__.imag());
}

inline void print_hash(std::string label__, unsigned long long int hash__)
{
    printf("hash(%s): %llx\n", label__.c_str(), hash__);
}

/// Maximum number of \f$ \ell, m \f$ combinations for a given \f$ \ell_{max} \f$
inline int lmmax(int lmax)
{
    return (lmax + 1) * (lmax + 1);
}

/// Get composite lm index by angular index l and azimuthal index m.
inline int lm(int l, int m)
{
    return (l * l + l + m);
}

/// Get maximum orbital quantum number by the maximum lm index.
inline int lmax(int lmmax__)
{
    assert(lmmax__ >= 0);
    int lmax = static_cast<int>(std::sqrt(static_cast<double>(lmmax__)) + 1e-8) - 1;
    if (lmmax(lmax) != lmmax__) {
        std::stringstream s;
        s << "wrong lmmax: " << lmmax__;
        TERMINATE(s);
    }
    return lmax;
}

/// Get array of orbital quantum numbers for each lm component.
inline std::vector<int> l_by_lm(int lmax__)
{
    std::vector<int> v(lmmax(lmax__));
    for (int l = 0; l <= lmax__; l++) {
        for (int m = -l; m <= l; m++) {
            v[lm(l, m)] = l;
        }
    }
    return std::move(v);
}

/// Check if file exists.
/** \param[in] file_name Full path to the file being checked.
 *  \return True if file exists, false otherwise. 
 */
inline bool file_exists(std::string file_name)
{
    std::ifstream ifs(file_name.c_str());
    return ifs.is_open();
}

/// Return the timestamp string in a specified format.
/** Typical format strings: "%Y%m%d_%H%M%S", "%Y-%m-%d %H:%M:%S", "%H:%M:%S" 
 */
inline std::string timestamp(std::string fmt)
{
    timeval t;
    gettimeofday(&t, NULL);

    char buf[128];

    tm* ptm = localtime(&t.tv_sec);
    strftime(buf, sizeof(buf), fmt.c_str(), ptm);
    return std::string(buf);
}

/// Wall-clock time in seconds.
inline double wtime()
{
    timeval t;
    gettimeofday(&t, NULL);
    return double(t.tv_sec) + double(t.tv_usec) / 1e6;
}

/// Sign of the variable.
template <typename T>
inline int sign(T val)
{
    return (T(0) < val) - (val < T(0));
}

/// Pack two indices into one for symmetric matrices.
inline int packed_index(int i__, int j__)
{
    /* suppose we have a symmetric matrix: M_{ij} = M_{ji}
           j
       +-------+
       | + + + |
      i|   + + |   -> idx = j * (j + 1) / 2 + i  for  i <= j
       |     + |
       +-------+

       i, j are row and column indices 
    */

    if (i__ > j__) {
        std::swap(i__, j__);
    }
    return j__ * (j__ + 1) / 2 + i__;
}

/// Convert double to a string with a given precision.
inline std::string double_to_string(double val, int precision = -1)
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

/// Return angle phi in the range [0, 2Pi) by its values of sin(phi) and cos(phi).
inline double phi_by_sin_cos(double sinp, double cosp)
{
    const double twopi = 6.2831853071795864769;
    double phi = std::atan2(sinp, cosp);
    if (phi < 0) {
        phi += twopi;
    }
    return phi;
}

/// Compute a factorial.
template <typename T>
inline T factorial(int n)
{
    assert(n >= 0);

    T result{1};
    for (int i = 1; i <= n; i++) {
        result *= i;
    }
    return result;
}

inline int num_blocks(int length__, int block_size__)
{
    return (length__ / block_size__) + std::min(length__ % block_size__, 1);
}

inline double round(double a__, int n__)
{
    double a0 = std::floor(a__);
    double b  = std::round((a__ - a0) * std::pow(10, n__)) / std::pow(10, n__);
    return a0 + b;
}

inline std::complex<double> round(std::complex<double> a__, int n__)
{
    return std::complex<double>(round(a__.real(), n__), round(a__.imag(), n__));
}

/// Simple hash function.
/** Example: printf("hash: %16llX\n", hash()); */
inline uint64_t hash(void const* buff, size_t size, uint64_t h = 5381)
{
    unsigned char const* p = static_cast<unsigned char const*>(buff);
    for (size_t i = 0; i < size; i++) {
        h = ((h << 5) + h) + p[i];
    }
    return h;
}

/// Simple pseudo-random generator.
inline uint32_t rand()
{
    static uint32_t a = 123456;

    a = (a ^ 61) ^ (a >> 16);
    a = a + (a << 3);
    a = a ^ (a >> 4);
    a = a * 0x27d4eb2d;
    a = a ^ (a >> 15);
    return a;
}

inline double confined_polynomial(double r, double R, int p1, int p2, int dm)
{
    double t = 1.0 - std::pow(r / R, 2);
    switch (dm) {
        case 0: {
            return (std::pow(r, p1) * std::pow(t, p2));
        }
        case 2: {
            return (-4 * p1 * p2 * std::pow(r, p1) * std::pow(t, p2 - 1) / std::pow(R, 2) +
                    p1 * (p1 - 1) * std::pow(r, p1 - 2) * std::pow(t, p2) +
                    std::pow(r, p1) * (4 * (p2 - 1) * p2 * std::pow(r, 2) * std::pow(t, p2 - 2) / std::pow(R, 4) -
                                       2 * p2 * std::pow(t, p2 - 1) / std::pow(R, 2)));
        }
        default: {
            TERMINATE("wrong derivative order");
            return 0.0;
        }
    }
}

/// Read json dictionary from file or string.
/** Terminate if file doesn't exist. */
inline nlohmann::json read_json_from_file_or_string(std::string const& str__)
{
    nlohmann::json dict = {};
    if (str__.size() == 0) {
        return std::move(dict);
    }

    if (str__.find("{") == std::string::npos) { /* this is a file */
        if (file_exists(str__)) {
            try {
                std::ifstream(str__) >> dict;
            } catch(std::exception& e) {
                std::stringstream s;
                s << "wrong input json file" << std::endl
                  << e.what();
                TERMINATE(s);
            }
        } 
        else {
            std::stringstream s;
            s << "file " << str__ << " doesn't exist";
            TERMINATE(s);
        }
    } else { /* this is a json string */
        try {
            std::istringstream(str__) >> dict;
        } catch (std::exception& e) {
            std::stringstream s;
            s << "wrong input json string" << std::endl
              << e.what();
            TERMINATE(s);
        }
    }

    return std::move(dict);
}

/// Get high water mark and resident space size values of a given process.
inline void get_proc_status(size_t* VmHWM__, size_t* VmRSS__)
{
    *VmHWM__ = 0;
    *VmRSS__ = 0;

    std::ifstream ifs("/proc/self/status");
    if (ifs.is_open()) {
        size_t tmp;
        std::string str; 
        std::string units;
        while (std::getline(ifs, str)) {
            auto p = str.find("VmHWM:");
            if (p != std::string::npos) {
                std::stringstream s(str.substr(p + 7));
                s >> tmp;
                s >> units;

                if (units != "kB") {
                    printf("runtime::get_proc_status(): wrong units");
                } else {
                    *VmHWM__ = tmp * 1024;
                }
            }

            p = str.find("VmRSS:");
            if (p != std::string::npos) {
                std::stringstream s(str.substr(p + 7));
                s >> tmp;
                s >> units;

                if (units != "kB") {
                    printf("runtime::get_proc_status(): wrong units");
                } else {
                    *VmRSS__ = tmp * 1024;
                }
            }
        } 
    }
}

/// Get number of threads currently running for this process.
inline int get_proc_threads()
{
    int num_threads{-1};

    std::ifstream ifs("/proc/self/status");
    if (ifs.is_open()) {
        std::string str;
        while (std::getline(ifs, str)) {
            auto p = str.find("Threads:");
            if (p != std::string::npos) {
                std::stringstream s(str.substr(p + 9));
                s >> num_threads;
                break;
            }
        }
    }

    return num_threads;
}

/// Get a host name.
inline std::string hostname()
{
    const int len{1024};
    char nm[len];
    gethostname(nm, len);
    nm[len - 1] = 0;
    return std::string(nm);
}

/// Return complex conjugate of a number. For a real value this is the number itself.
inline double conj(double x__)
{
    /* std::conj() will return complex for a double value input; this is not what we want */
    return x__;
}

/// Return complex conjugate of a number.
inline std::complex<double> conj(std::complex<double> x__)
{
    return std::conj(x__);
}

template <typename T>
inline T zero_if_not_complex(std::complex<double> x__);

template<>
inline double zero_if_not_complex<double>(std::complex<double> x__)
{
    return 0;
}

template<>
inline std::complex<double> zero_if_not_complex<std::complex<double>>(std::complex<double> x__)
{
    return x__;
}

/// Simple random number generator.
inline uint32_t rnd()
{
    static uint32_t a = 123456;
    a                 = (a ^ 61) ^ (a >> 16);
    a                 = a + (a << 3);
    a                 = a ^ (a >> 4);
    a                 = a * 0x27d4eb2d;
    a                 = a ^ (a >> 15);
    return a;
}

template <typename T>
inline T random();

template <>
inline double random<double>()
{
    return static_cast<double>(rnd()) / std::numeric_limits<uint32_t>::max();
}

template <>
inline std::complex<double> random<std::complex<double>>()
{
    return std::complex<double>(random<double>(), random<double>());
}

inline long get_page_size()
{
    return sysconf(_SC_PAGESIZE);
}

inline long get_num_pages()
{
    return sysconf(_SC_PHYS_PAGES);
}

inline long get_total_memory()
{
    return get_page_size() * get_num_pages();
}

} // namespace

template <typename T>
inline std::ostream& operator<<(std::ostream& out, std::vector<T>& v)
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

inline std::string& ltrim(std::string& str, const std::string& chars = "\t\n\v\f\r ")
{
    str.erase(0, str.find_first_not_of(chars));
    return str;
}
 
inline std::string& rtrim(std::string& str, const std::string& chars = "\t\n\v\f\r ")
{
    str.erase(str.find_last_not_of(chars) + 1);
    return str;
}

inline std::string& trim(std::string& str, const std::string& chars = "\t\n\v\f\r ")
{
    return ltrim(rtrim(str, chars), chars);
}

#endif
