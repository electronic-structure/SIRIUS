/** \file power.hpp
 *
 *  \brief Read power counters on Cray.
 */

#include <fstream>
#include <string>

namespace utils {
namespace power {

static double read_pm_file(const std::string& fname)
{
    double result = 0.;
    std::ifstream fid(fname.c_str());

    fid >> result;
    // std::cout << fname << " :: " << result << std::endl;
    return result;
}

static double device_energy(void)
{
    return read_pm_file("/sys/cray/pm_counters/accel_energy");
}

static double energy()
{
    return read_pm_file("/sys/cray/pm_counters/energy");
}

static double device_power()
{
    return read_pm_file("/sys/cray/pm_counters/accel_power");
}

static double power()
{
    return read_pm_file("/sys/cray/pm_counters/power");
}

static int num_nodes()
{
    // find out the number of nodes
    char* ptr = std::getenv("SLURM_JOB_NUM_NODES");
    if (ptr) {
        return atoi(ptr);
    } else {
        return -1;
    }
}

} // namespace power
} // namespace utils
