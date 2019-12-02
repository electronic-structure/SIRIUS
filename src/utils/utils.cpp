// Copyright (c) 2013-2017 Anton Kozhevnikov, Thomas Schulthess
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

/** \file utils.cpp
 *
 *  \brief Definitions.
 *
 */
#include "utils/utils.hpp"

namespace utils {
std::string timestamp(std::string fmt)
{
    timeval t;
    gettimeofday(&t, NULL);

    char buf[128];

    tm* ptm = localtime(&t.tv_sec);
    strftime(buf, sizeof(buf), fmt.c_str(), ptm);
    return std::string(buf);
}

std::string double_to_string(double val, int precision)
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

double confined_polynomial(double r, double R, int p1, int p2, int dm)
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

nlohmann::json read_json_from_file_or_string(std::string const& str__)
{
    nlohmann::json dict = {};
    if (str__.size() == 0) {
        return dict;
    }

    if (str__.find("{") == std::string::npos) { /* this is a file */
        if (file_exists(str__)) {
            try {
                std::ifstream(str__) >> dict;
            } catch (std::exception& e) {
                std::stringstream s;
                s << "wrong input json file" << std::endl << e.what();
                TERMINATE(s);
            }
        } else {
            std::stringstream s;
            s << "file " << str__ << " doesn't exist";
            TERMINATE(s);
        }
    } else { /* this is a json string */
        try {
            std::istringstream(str__) >> dict;
        } catch (std::exception& e) {
            std::stringstream s;
            s << "wrong input json string" << std::endl << e.what();
            TERMINATE(s);
        }
    }

    return dict;
}

void get_proc_status(size_t* VmHWM__, size_t* VmRSS__)
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
                    std::printf("runtime::get_proc_status(): wrong units");
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
                    std::printf("runtime::get_proc_status(): wrong units");
                } else {
                    *VmRSS__ = tmp * 1024;
                }
            }
        }
    }
}

int get_proc_threads()
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
} // namespace utils
