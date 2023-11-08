// Copyright (c) 2013-2023 Anton Kozhevnikov, Thomas Schulthess
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

/** \file system_tools.hpp
 *
 *  \brief System-level helper functions.
 */

#ifndef __SYSTEM_TOOLS_HPP__
#define __SYSTEM_TOOLS_HPP__

namespace sirius {

/// Check if file exists.
/** \param[in] file_name Full path to the file being checked.
 *  \return True if file exists, false otherwise.
 */
inline bool
file_exists(std::string file_name)
{
    std::ifstream ifs(file_name.c_str());
    return ifs.is_open();
}

/// Get host name.
inline auto
hostname()
{
    const int len{1024};
    char nm[len];
    gethostname(nm, len);
    nm[len - 1] = 0;
    return std::string(nm);
}

inline long
get_page_size()
{
    return sysconf(_SC_PAGESIZE);
}

inline long
get_num_pages()
{
    return sysconf(_SC_PHYS_PAGES);
}

inline long
get_total_memory()
{
    return get_page_size() * get_num_pages();
}

inline auto
get_proc_status()
{
    /* virtul memory high water mark */
    size_t VmHWM{0};
    /* virtual memory resident set size */
    size_t VmRSS{0};

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
                    VmHWM = tmp * 1024;
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
                    VmRSS = tmp * 1024;
                }
            }
        }
    }
    struct
    {
        size_t VmHWM;
        size_t VmRSS;
    } res{VmHWM, VmRSS};

    return res;
}

inline int
get_proc_threads()
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

} // namespace sirius

#endif
