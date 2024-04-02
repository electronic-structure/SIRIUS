/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

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
