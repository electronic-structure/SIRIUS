/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file power.hpp
 *
 *  \brief Read power counters on Cray.
 */

#ifndef __POWER_HPP__
#define __POWER_HPP__

#include <fstream>
#include <string>
#include <iomanip>
#include <array>
#include <vector>
#include <iterator>
#include <iostream>
#include "core/mpi/communicator.hpp"

namespace sirius {
namespace power {

inline double
read_pm_file(const std::string& fname)
{
    double result = 0.;
#if defined(SIRIUS_USE_POWER_COUNTER)
    std::ifstream fid(fname.c_str());
    fid >> result;
#endif
    return result;
}

inline double
node_energy()
{
    return read_pm_file("/sys/cray/pm_counters/energy");
}

inline double
accel_energy()
{
    return read_pm_file("/sys/cray/pm_counters/accel0_energy") + read_pm_file("/sys/cray/pm_counters/accel1_energy") +
           read_pm_file("/sys/cray/pm_counters/accel2_energy") + read_pm_file("/sys/cray/pm_counters/accel3_energy");
}

inline double
cpu_energy()
{
    return read_pm_file("/sys/cray/pm_counters/cpu_energy");
}

struct Event
{
    double node_energy_;
    double accel_energy_;
    double cpu_energy_;
    const char* label;
    int id; // 0 - stop, 1 - start
    Event(const char* label, int id)
        : label{label}
        , id{id}
    {
        node_energy_  = node_energy();
        accel_energy_ = accel_energy();
        cpu_energy_   = cpu_energy();
    }
    Event()
    {
    }
};

class Profile
{
  private:
    const char* label_;

  public:
    Profile(const char* label)
        : label_{label}
    {
        events_.emplace_back(label_, 1);
    }
    ~Profile()
    {
        events_.emplace_back(label_, 0);
    }
    inline static std::vector<Event> events_;
};

inline std::string
format_energy(double energy)
{
    const char* units[]   = {"J", "kJ", "MJ", "GJ"};
    const double scales[] = {1.0, 1e3, 1e6, 1e9};

    int idx = 0;
    while (idx < 3 && energy >= scales[idx + 1]) {
        ++idx;
    }

    double value = energy / scales[idx];

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << value << " " << units[idx];
    return oss.str();
}

inline void
report(mpi::Communicator const& comm__)
{
    // vector of results of range measurments for a given label
    std::map<std::string, std::vector<std::array<double, 3>>> results;
    for (size_t i = 0; i < Profile::events_.size(); i++) {
        auto const& e = Profile::events_[i];
        if (e.id == 1) {
            std::string label(e.label);
            for (size_t j = i + 1; j < Profile::events_.size(); j++) {
                auto const& e1 = Profile::events_[j];
                if (e1.id == 0 && std::string(e1.label) == label) {
                    std::array<double, 3> d({e1.node_energy_ - e.node_energy_, e1.accel_energy_ - e.accel_energy_,
                                             e1.cpu_energy_ - e.cpu_energy_});
                    results[label].push_back(d);
                    break;
                }
            }
        }
    }
    size_t len{0};
    for (auto& e : results) {
        len = std::max(len, e.first.length());
    }
    if (comm__.rank() == 0) {
        std::cout << "=== Energy consumption report ===" << std::endl;
        std::fill_n(std::ostream_iterator<char>(std::cout), len + 57, '-');
        std::cout << std::endl
                  << std::right << std::setw(len) << "name" << " : " << std::right << std::setw(6) << "count"
                  << std::right << std::setw(12) << "nodes" << std::right << std::setw(12) << "GPUs" << std::right
                  << std::setw(12) << "CPUs" << std::right << std::setw(12) << "nodes avg." << std::endl;
        std::fill_n(std::ostream_iterator<char>(std::cout), len + 57, '-');
        std::cout << std::endl;
    }

    for (auto& e : results) {
        //auto [minIt, maxIt] = std::minmax_element(e.second.begin(), e.second.end());
        //double sum = std::accumulate(e.second.begin(), e.second.end(), 0.0);
        double total_node_energy{0};
        double total_accel_energy{0};
        double total_cpu_energy{0};
        for (auto& d : e.second) {
            total_node_energy += d[0];
            total_accel_energy += d[1];
            total_cpu_energy += d[2];
        }
        comm__.allreduce(&total_node_energy, 1);
        comm__.allreduce(&total_accel_energy, 1);
        comm__.allreduce(&total_cpu_energy, 1);
        if (comm__.rank() == 0) {
            std::cout << std::setfill(' ') << std::right << std::setw(len) << e.first << " : " << std::right
                      << std::setw(6) << e.second.size() << std::right << std::setw(12)
                      << format_energy(total_node_energy) << std::right << std::setw(12)
                      << format_energy(total_accel_energy) << std::right << std::setw(12)
                      << format_energy(total_cpu_energy) << std::right << std::setw(12)
                      << format_energy(total_node_energy / e.second.size()) << std::endl;
        }
    }
}

} // namespace power
} // namespace sirius

#endif
