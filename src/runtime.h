#ifndef __RUNTIME_H__
#define __RUNTIME_H__

#include <signal.h>
#include <sys/time.h>
#include <map>
#include <fstream>
#include <chrono>
#include <omp.h>
#include <unistd.h>
#include <cstring>
#include <cstdarg>
#include "config.h"
#include "communicator.hpp"
#include "utils/json.hpp"
#include "utils/utils.hpp"
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
    pstdout(Communicator const& comm__)
        : comm_(comm__)
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

} // namespace runtime

#define DUMP(...)                                                                      \
{                                                                                  \
    char str__[1024];                                                              \
    int  x__ = snprintf(str__, 1024, "[rank%04i] ", Communicator::world().rank()); \
    x__ += snprintf(&str__[x__], 1024, __VA_ARGS__);                               \
    printf("%s\n", str__);                                                         \
}

inline void print_memory_usage(const char* file__, int line__)
{
    size_t VmRSS, VmHWM;
    utils::get_proc_status(&VmHWM, &VmRSS);

    std::vector<char> str(2048);
    int               n = snprintf(&str[0], 2048, "[rank%04i at line %i of file %s]", Communicator::world().rank(), line__, file__);

    n += snprintf(&str[n], 2048, " VmHWM: %i Mb, VmRSS: %i Mb", static_cast<int>(VmHWM >> 20), static_cast<int>(VmRSS >> 20));

#ifdef __GPU
    size_t gpu_mem = acc::get_free_mem();
    n += snprintf(&str[n], 2048, ", GPU free memory: %i Mb", static_cast<int>(gpu_mem >> 20));
#endif

    printf("%s\n", &str[0]);
}

#define MEMORY_USAGE_INFO() print_memory_usage(__FILE__, __LINE__);

#endif // __RUNTIME_H__
