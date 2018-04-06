#ifndef __SDDK_INTERNAL_HPP__
#define __SDDK_INTERNAL_HPP__

#define __PROFILE
#define __PROFILE_TIME
//#define __PROFILE_STACK
//#define __PROFILE_FUNC

#include <omp.h>
#include <string>
#include <sstream>
#include <chrono>
#include <map>
#include <vector>
#include <memory>
#include <complex>
#include <algorithm>
#include "communicator.hpp"
#ifdef __GPU
#include "GPU/cuda.hpp"
#endif
#include "json.hpp"

/// Namespace of the Slab Data Distribution Kit.
namespace sddk {

#include "timer.hpp"

class Profiler
{
  private:
    /// Label of the profiler.
    std::string label_;

    /// Name of the function in which the profiler is created.
    std::string function_name_;

    /// Name of the file.
    std::string file_;

    /// Line number.
    int line_;

    /// Profiler's timer.
    std::unique_ptr<timer> timer_;

// std::string timestamp()
//{
//    timeval t;
//    gettimeofday(&t, NULL);
//
//    char buf[100];
//
//    tm* ptm = localtime(&t.tv_sec);
//    //strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", ptm);
//    strftime(buf, sizeof(buf), "%H:%M:%S", ptm);
//    return std::string(buf);
//}

    #ifdef __PROFILE_STACK
    static std::vector<std::string>& call_stack()
    {
        static std::vector<std::string> call_stack_;
        return call_stack_;
    }
    #endif

  public:
    Profiler(char const* function_name__, char const* file__, int line__, char const* label__)
        : label_(std::string(label__))
        , function_name_(std::string(function_name__))
        , file_(std::string(file__))
        , line_(line__)

    {
        if (omp_get_num_threads() != 1) {
            std::stringstream s;
            s << "sddk::Profiler does not support threads";
            throw std::runtime_error(s.str());
        }

        #if defined(__PROFILE_STACK) || defined(__PROFILE_FUNC)
        char str[2048];
        snprintf(str, 2048, "%s at %s:%i", function_name__, file__, line__);
        #endif

        #if defined(__PROFILE_STACK)
        call_stack().push_back(std::string(str));
        #endif

        #if defined(__PROFILE_FUNC)
        int tab{0};
        #if defined(__PROFILE_STACK)
        tab = static_cast<int>(call_stack().size()) - 1;
        #endif
        for (int i = 0; i < tab; i++) {
            printf(" ");
        }
        printf("[rank%04i] + %s\n", mpi_comm_world().rank(), label_.c_str());
        #endif

        #if defined(__PROFILE_TIME)
        timer_ = std::unique_ptr<timer>(new timer(label_));
        #endif

        #if defined(__GPU) && defined(__GPU_NVTX)
        acc::begin_range_marker(label_.c_str());
        #endif
    }

    ~Profiler()
    {
        #ifdef __PROFILE_FUNC
        int tab{0};
        #ifdef __PROFILE_STACK
        tab = static_cast<int>(call_stack().size()) - 1;
        #endif
        for (int i = 0; i < tab; i++) {
            printf(" ");
        }
        printf("[rank%04i] - %s\n", mpi_comm_world().rank(), label_.c_str());
        #endif

        #ifdef __PROFILE_STACK
        call_stack().pop_back();
        #endif

        #if defined(__GPU) && defined(__GPU_NVTX)
        acc::end_range_marker();
        #endif
    }

    static void stack_trace()
    {
        #ifdef __PROFILE_STACK
        int t{0};
        for (auto it = call_stack().rbegin(); it != call_stack().rend(); it++) {
            for (int i = 0; i < t; i++) {
                printf(" ");
            }
            printf("[%s]\n", it->c_str());
            t++;
        }
        #endif
    }
};

#ifdef __GNUC__
    #define __function_name__ __PRETTY_FUNCTION__
#else
    #define __function_name__ __func__
#endif

#ifdef __PROFILE
    #define PROFILE(name) sddk::Profiler profiler__(__function_name__, __FILE__, __LINE__, name);
#else
    #define PROFILE(...)
#endif

using double_complex = std::complex<double>;

inline void terminate(const char* file_name__, int line_number__, const std::string& message__)
{
    std::stringstream s;
    s << "\n=== Fatal error at line " << line_number__ << " of file " << file_name__ << " ===\n";
    s << message__ << "\n\n";
    // raise(SIGTERM);
    throw std::runtime_error(s.str());
}

inline void terminate(const char* file_name__, int line_number__, const std::stringstream& message__)
{
    terminate(file_name__, line_number__, message__.str());
}

inline void warning(const char* file_name__, int line_number__, const std::string& message__)
{
    printf("\n=== Warning at line %i of file %s ===\n", line_number__, file_name__);
    printf("%s\n\n", message__.c_str());
}

inline void warning(const char* file_name__, int line_number__, const std::stringstream& message__)
{
    warning(file_name__, line_number__, message__.str());
}

} // namespace sddk

#define TERMINATE(msg) sddk::terminate(__FILE__, __LINE__, msg);

#define WARNING(msg) sddk::warning(__FILE__, __LINE__, msg);

#define STOP() TERMINATE("terminated by request")

#define TERMINATE_NO_GPU TERMINATE("not compiled with GPU support");

#define TERMINATE_NO_SCALAPACK TERMINATE("not compiled with ScaLAPACK support");

#define TERMINATE_NOT_IMPLEMENTED TERMINATE("feature is not implemented");

#endif // __SDDK_INTERNAL_HPP__
