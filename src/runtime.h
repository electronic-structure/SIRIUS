#ifndef __RUNTIME_H__
#define __RUNTIME_H__

// include "config.h" or define varaibles here
//#define __TIMER
//#define __TIMER_CHRONO

#include <signal.h>
#include <sys/time.h>
#include <map>
#include <fstream>
#include <chrono>
#include <omp.h>
#include <unistd.h>
#include <cinttypes.h>
#include <cstring>
#include <cstdarg>
#include "config.h"
#include "communicator.hpp"
#include "json.hpp"
#ifdef __GPU
#include "GPU/cuda.hpp"
#endif

using json = nlohmann::json;
using namespace sddk;

namespace runtime {

/// Parallel standard output.
/** Proveides an ordered standard output from multiple MPI ranks. */
class pstdout
{
    private:
        
        std::vector<char> buffer_;

        int count_{0};

        Communicator const& comm_;

    public:

        pstdout(Communicator const& comm__) : comm_(comm__)
        {
            buffer_.resize(10240);
        }

        ~pstdout()
        {
            flush();
        }

        void printf(const char* fmt, ...)
        {
            std::vector<char> str(1024); // assume that one printf will not output more than this

            std::va_list arg;
            va_start(arg, fmt);
            int n = vsnprintf(&str[0], str.size(), fmt, arg);
            va_end(arg);

            n = std::min(n, (int)str.size());
            
            if ((int)buffer_.size() - count_ < n) {
                buffer_.resize(buffer_.size() + str.size());
            }
            std::memcpy(&buffer_[count_], &str[0], n);
            count_ += n;
        }

        void flush()
        {
            std::vector<int> counts(comm_.size());
            comm_.allgather(&count_, counts.data(), comm_.rank(), 1); 
            
            int offset{0};
            for (int i = 0; i < comm_.rank(); i++) {
                offset += counts[i];
            }
            
            /* total size of the output buffer */
            int sz = count_;
            comm_.allreduce(&sz, 1);
            
            if (sz != 0) {
                std::vector<char> outb(sz + 1);
                comm_.allgather(&buffer_[0], &outb[0], offset, count_);
                outb[sz] = 0;

                if (comm_.rank() == 0) {
                    std::printf("%s", &outb[0]);
                }
            }
            count_ = 0;
        }
};


//inline void terminate(const char* file_name__, int line_number__, const std::string& message__)
//{
//    printf("\n=== Fatal error at line %i of file %s ===\n", line_number__, file_name__);
//    printf("%s\n\n", message__.c_str());
//    raise(SIGTERM);
//    throw std::runtime_error("terminating...");
//}
//
//inline void terminate(const char* file_name__, int line_number__, const std::stringstream& message__)
//{
//    terminate(file_name__, line_number__, message__.str());
//}
//
//inline void warning(const char* file_name__, int line_number__, const std::string& message__)
//{
//    printf("\n=== Warning at line %i of file %s ===\n", line_number__, file_name__);
//    printf("%s\n\n", message__.c_str());
//}
//
//inline void warning(const char* file_name__, int line_number__, const std::stringstream& message__)
//{
//    warning(file_name__, line_number__, message__.str());
//}

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

inline int get_num_threads()
{
    int num_threads = -1;
    
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

//inline double wtime()
//{
//    timeval t;
//    gettimeofday(&t, NULL);
//    return double(t.tv_sec) + double(t.tv_usec) / 1e6;
//}

}

#define DUMP(...)                                                                     \
{                                                                                     \
    char str__[1024];                                                                 \
    /* int x__ = snprintf(str__, 1024, "[%s:%04i] ", __func__, mpi_comm_world().rank()); */ \
    int x__ = snprintf(str__, 1024, "[rank%04i] ", mpi_comm_world().rank()); \
    x__ += snprintf(&str__[x__], 1024, __VA_ARGS__ );                                 \
    printf("%s\n", str__);                                                            \
}

inline void print_checksum(std::string label__, double cs__)
{
    printf("checksum(%s): %18.12f\n", label__.c_str(), cs__);
}

inline void print_checksum(std::string label__, std::complex<double> cs__)
{
    printf("checksum(%s): %18.12f %18.12f\n", label__.c_str(), cs__.real(), cs__.imag());
}

inline void print_hash(std::string label__, uint64_t hash__)
{
    printf("hash(%s): %" PRIx64 "\n", label__.c_str(), hash__);
}

inline void print_memory_usage(const char* file__, int line__)
{
    size_t VmRSS, VmHWM;
    runtime::get_proc_status(&VmHWM, &VmRSS);

    std::vector<char> str(2048);
    int n = snprintf(&str[0], 2048, "[rank%04i at line %i of file %s]", mpi_comm_world().rank(), line__, file__);

    n += snprintf(&str[n], 2048, " VmHWM: %i Mb, VmRSS: %i Mb", static_cast<int>(VmHWM >> 20), static_cast<int>(VmRSS >> 20));

    #ifdef __GPU
    size_t gpu_mem = acc::get_free_mem();
    n += snprintf(&str[n], 2048, ", GPU free memory: %i Mb", static_cast<int>(gpu_mem >> 20));
    #endif

    printf("%s\n", &str[0]);
}

#define MEMORY_USAGE_INFO() print_memory_usage(__FILE__, __LINE__);

#endif // __RUNTIME_H__
