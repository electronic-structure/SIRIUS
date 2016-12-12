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
#include <cstring>
#include <cstdarg>
#include "config.h"
#include "communicator.hpp"
#include "json.hpp"
#ifdef __GPU
#include "gpu.h"
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
            comm_.allgather(&count_, &counts[0], comm_.rank(), 1); 
            
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
    if (ifs.is_open())
    {
        std::string str; 
        while (std::getline(ifs, str))
        {
            auto p = str.find("Threads:");
            if (p != std::string::npos)
            {
                std::stringstream s(str.substr(p + 9));
                s >> num_threads;
                break;
            }
        }
    }

    return num_threads;
}

///// Simple timer interface.
//class Timer
//{
//    private:
//
//        #ifdef __TIMER
//        /// String label of the timer.
//        std::string label_;
//        
//        /// Starting time.
//        #if defined(__TIMER_TIMEOFDAY)
//        timeval starting_time_;
//        #elif defined(__TIMER_MPI_WTIME)
//        double starting_time_;
//        #elif defined(__TIMER_CHRONO)
//        std::chrono::high_resolution_clock::time_point starting_time_;
//        #endif
//
//        Communicator const* comm_;
//
//        /// true if timer is running
//        bool active_;
//        #endif
//    
//    public:
//
//        struct timer_stats
//        {
//            int count;
//            double min_value;
//            double max_value;
//            double total_value;
//            double average_value;
//        };
//        
//        #ifdef __TIMER
//        Timer(std::string const& label__) 
//            : label_(label__),
//              comm_(nullptr),
//              active_(false)
//        {
//            #pragma omp critical
//            if (timers().count(label_) == 0) {
//                timers()[label_] = std::vector<double>();
//            }
//
//            start();
//        }
//        #else
//        Timer(std::string const& label__)
//        {
//        }
//        #endif
//        
//        #ifdef __TIMER
//        Timer(std::string const& label__, Communicator const& comm__) 
//            : label_(label__),
//              comm_(&comm__),
//              active_(false)
//        {
//            if (timers().count(label_) == 0) timers()[label_] = std::vector<double>();
//
//            start();
//        }
//        #else
//        Timer(std::string const& label__, Communicator const& comm__)
//        {
//        }
//        #endif
//
//        ~Timer()
//        {
//            #ifdef __TIMER
//            if (active_) stop();
//            #endif
//        }
//
//        void start()
//        {
//            #if defined (__GPU) && defined (__GPU_NVTX)
//            cuda_begin_range_marker(label_.c_str());
//            #endif
//
//            #ifdef __TIMER
//            //#ifndef NDEBUG
//            //if (omp_get_num_threads() != 1) {
//            //    printf("std::map used by Timer is not thread-safe\n");
//            //    printf("timer name: %s\n", label_.c_str());
//            //    exit(-1);
//            //}
//            //#endif
//
//            if (active_) {
//                printf("timer %s is already running\n", label_.c_str());
//                exit(-2);
//            }
//            if (comm_ != nullptr) {
//                comm_->barrier();
//            }
//            #if defined(__TIMER_TIMEOFDAY)
//            gettimeofday(&starting_time_, NULL);
//            #elif defined(__TIMER_MPI_WTIME)
//            starting_time_ = MPI_Wtime();
//            #elif defined (__TIMER_CHRONO)
//            starting_time_ = std::chrono::high_resolution_clock::now();
//            #endif
//            active_ = true;
//            #endif
//        }
//
//        double stop()
//        {
//            #if defined (__GPU) && defined(__GPU_NVTX)
//            cuda_end_range_marker();
//            #endif
//
//            #ifdef __TIMER
//            if (!active_)
//            {
//                printf("timer %s was not running\n", label_.c_str());
//                exit(-2);
//            }
//            if (comm_ != nullptr) comm_->barrier();
//
//            #if defined(__TIMER_TIMEOFDAY)
//            timeval end;
//            gettimeofday(&end, NULL);
//            double val = double(end.tv_sec - starting_time_.tv_sec) + 
//                         double(end.tv_usec - starting_time_.tv_usec) / 1e6;
//            #elif defined(__TIMER_MPI_WTIME)
//            double val = MPI_Wtime() - starting_time_;
//            #elif defined(__TIMER_CHRONO)
//            auto t2 = std::chrono::high_resolution_clock::now();
//            std::chrono::duration<double> tdiff = std::chrono::duration_cast< std::chrono::duration<double> >(t2 - starting_time_);
//            double val = tdiff.count();
//            #endif
//            
//            #pragma omp critical
//            timers()[label_].push_back(val);
//
//            active_ = false;
//
//            return val;
//            #else
//            return 0;
//            #endif
//        }
//
//        static void clear()
//        {
//            #ifdef __TIMER
//            timers().clear();
//            #endif
//        }
//        
//        #ifdef __TIMER
//        static std::map< std::string, std::vector<double> >& timers()
//        {
//            static std::map< std::string, std::vector<double> > timers_;
//            return timers_;
//        }
//
//        static std::map<std::string, timer_stats> collect_timer_stats()
//        {
//            std::map<std::string, timer_stats> tstats;
//
//            /* collect local timers */
//            for (auto& it: timers()) {
//                timer_stats ts;
//
//                ts.count = static_cast<int>(it.second.size());
//                ts.total_value = 0.0;
//                ts.min_value = 1e100;
//                ts.max_value = 0.0;
//                for (int i = 0; i < ts.count; i++) {
//                    ts.total_value += it.second[i];
//                    ts.min_value = std::min(ts.min_value, it.second[i]);
//                    ts.max_value = std::max(ts.max_value, it.second[i]);
//                }
//                ts.average_value = (ts.count == 0) ? 0.0 : ts.total_value / ts.count;
//                if (ts.count == 0) {
//                    ts.min_value = 0.0;
//                }
//
//                tstats[it.first] = ts;
//            }
//
//            return tstats;
//        }
//
//        static json serialize()
//        {
//            json dict;
//
//            /* collect local timers */
//            for (auto& it: timers()) {
//                timer_stats ts;
//
//                ts.count = static_cast<int>(it.second.size());
//                ts.total_value = 0.0;
//                ts.min_value = 1e100;
//                ts.max_value = 0.0;
//                for (int i = 0; i < ts.count; i++) {
//                    ts.total_value += it.second[i];
//                    ts.min_value = std::min(ts.min_value, it.second[i]);
//                    ts.max_value = std::max(ts.max_value, it.second[i]);
//                }
//                ts.average_value = (ts.count == 0) ? 0.0 : ts.total_value / ts.count;
//                if (ts.count == 0) {
//                    ts.min_value = 0.0;
//                }
//
//                dict[it.first] = {ts.total_value, ts.average_value, ts.min_value, ts.max_value};
//            }
//            return std::move(dict);
//        }
//
//        static double value(std::string const& label__)
//        {
//            auto values = timers()[label__];
//
//            double d = 0;
//            for (double v: values) d += v;
//            return d;
//        }
//
//        static void print()
//        {
//            auto tstats = collect_timer_stats();
//
//            if (mpi_comm_world().rank() == 0)
//            {
//                printf("\n");
//                printf("Timers\n");
//                for (int i = 0; i < 115; i++) printf("-");
//                printf("\n");
//                printf("name                                                              count      total        min        max    average\n");
//                for (int i = 0; i < 115; i++) printf("-");
//                printf("\n");
//
//                for (auto it = tstats.begin(); it != tstats.end(); it++)
//                {
//                    auto ts = it->second;
//                    printf("%-60s :    %5i %10.4f %10.4f %10.4f %10.4f\n", it->first.c_str(), ts.count, ts.total_value, 
//                           ts.min_value, ts.max_value, ts.average_value);
//                }
//                
//                //#ifdef __GPU
//                //print_cuda_timers();
//                //#endif
//            }
//        }
//
//        static void print_all()
//        {
//            char host_name[1024];
//            gethostname(host_name, 1024);
//
//            auto tstats = collect_timer_stats();
//
//            std::vector<std::string> timer_names;
//
//            if (mpi_comm_world().rank() == 0)
//            {
//                for (auto it = tstats.begin(); it != tstats.end(); it++)
//                {
//                    timer_names.push_back(it->first);
//                }
//            }
//            int nt = static_cast<int>(timer_names.size());
//            mpi_comm_world().bcast(&nt, 1, 0);
//            if (mpi_comm_world().rank() != 0) timer_names = std::vector<std::string>(nt);
//
//            for (int i = 0; i < nt; i++)
//                mpi_comm_world().bcast(timer_names[i], 0);
//
//            if (mpi_comm_world().rank() == 0)
//            {
//                printf("\n");
//                printf("Timers\n");
//                for (int i = 0; i < 115; i++) printf("-");
//                printf("\n");
//                printf("name                                                              count      total        min        max    average\n");
//                for (int i = 0; i < 115; i++) printf("-");
//                printf("\n");
//            }
//            pstdout pout(mpi_comm_world());
//
//            for (int i = 0; i < nt; i++)
//            {
//                if (mpi_comm_world().rank() == 0)
//                    pout.printf("---- %s ----\n", timer_names[i].c_str());
//
//                if (tstats.count(timer_names[i]) != 0)
//                {
//                    auto ts = tstats[timer_names[i]];
//                    pout.printf("%-60s :    %5i %10.4f %10.4f %10.4f %10.4f\n", host_name, ts.count, ts.total_value, 
//                                ts.min_value, ts.max_value, ts.average_value);
//                }
//                pout.flush();
//            }
//
//
//
//            //std::vector<std::string> host_name(comm__.size());
//            //host_name[comm__.rank()] = std::string(buf);
//
//            //for (int i = 0; i < comm__.size(); i++)
//            //{
//            //    int sz;
//            //    if (i = comm__.rank())
//            //    {
//            //        sz = static_cast<int>(host_name[i].size());
//            //        std::memcpy(buf, host_name[i].str().c_str(), sz);
//            //    }
//            //    comm__.bcast(&sz, 1, i);
//            //    comm__.bcast(buf, sz, i);
//            //    buf[sz] = 0;
//            //    host_name[i] = std::string(buf);
//            //}
//
//            //for (int i = 0; i < comm__.size(); i++)
//            //{
//            //    if (i = comm__.rank());
//
//            //}
//        }
//        #else
//        static void print()
//        {
//        }
//        #endif
//};
//
//inline double wtime()
//{
//    timeval t;
//    gettimeofday(&t, NULL);
//    return double(t.tv_sec) + double(t.tv_usec) / 1e6;
//}
//
//class Profiler
//{
//    private:
//
//        std::string name_;
//        std::string file_;
//        int line_;
//        Timer* timer_;
//
//        std::string timestamp()
//        {
//            timeval t;
//            gettimeofday(&t, NULL);
//        
//            char buf[100]; 
//        
//            tm* ptm = localtime(&t.tv_sec); 
//            //strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", ptm); 
//            strftime(buf, sizeof(buf), "%H:%M:%S", ptm); 
//            return std::string(buf);
//        }
//
//        #ifdef __PROFILE_STACK
//        static std::vector<std::string>& call_stack()
//        {
//            static std::vector<std::string> call_stack_;
//            return call_stack_;
//        }
//        #endif
//
//        inline void init(char const* name__, char const* file__, int line__)
//        {
//            #if defined(__PROFILE_STACK) || defined(__PROFILE_FUNC)
//            name_ = std::string(name__);
//            file_ = std::string(file__);
//            line_ = line__;
//
//            char str[1024];
//            snprintf(str, 1024, "%s at %s:%i", name__, file__, line__);
//            #endif
//            
//            #ifdef __PROFILE_STACK
//            call_stack().push_back(std::string(str));
//            #endif
//
//            #ifdef __PROFILE_FUNC
//            int tab = 0;
//            #ifdef __PROFILE_STACK
//            tab = static_cast<int>(call_stack().size()) - 1;
//            #endif
//            for (int i = 0; i < tab; i++) printf(" ");
//            printf("[rank%04i] + %s\n", mpi_comm_world().rank(), name_.c_str());
//            #endif
//        }
//
//    public:
//
//        Profiler(char const* name__, char const* file__, int line__) : timer_(nullptr)
//        {
//            init(name__, file__, line__);
//        }
//
//        Profiler(char const* name__, char const* file__, int line__, char const* timer_name_)
//        {
//            init(name__, file__, line__);
//            
//            #ifdef __PROFILE_TIME
//            timer_ = new Timer(timer_name_);
//            #endif
//        }
//
//        ~Profiler()
//        {
//            #ifdef __PROFILE_TIME
//            if (timer_ != nullptr) delete timer_;
//            #endif
//
//            #ifdef __PROFILE_FUNC
//            int tab = 0;
//            #ifdef __PROFILE_STACK
//            tab = static_cast<int>(call_stack().size()) - 1;
//            #endif
//            for (int i = 0; i < tab; i++) printf(" ");
//            printf("[rank%04i] - %s\n", mpi_comm_world().rank(), name_.c_str());
//            #endif
//
//            #ifdef __PROFILE_STACK
//            call_stack().pop_back();
//            #endif
//        }
//
//        static void stack_trace()
//        {
//            #ifdef __PROFILE_STACK
//            int t = 0;
//            for (auto it = call_stack().rbegin(); it != call_stack().rend(); it++)
//            {
//                for (int i = 0; i < t; i++) printf(" ");
//                printf("[%s]\n", it->c_str());
//                t++;
//            }
//            #endif
//        }
//};
//
}

//#define TERMINATE(msg) runtime::terminate(__FILE__, __LINE__, msg);
//
//#define WARNING(msg) runtime::warning(__FILE__, __LINE__, msg);
//
//#define STOP() TERMINATE("terminated by request")

#define DUMP(...)                                                                     \
{                                                                                     \
    char str__[1024];                                                                 \
    /* int x__ = snprintf(str__, 1024, "[%s:%04i] ", __func__, mpi_comm_world().rank()); */ \
    int x__ = snprintf(str__, 1024, "[rank%04i] ", mpi_comm_world().rank()); \
    x__ += snprintf(&str__[x__], 1024, __VA_ARGS__ );                                 \
    printf("%s\n", str__);                                                            \
}

#define PRINT(...)                    \
{                                     \
    if (mpi_comm_world().rank() == 0) \
    {                                 \
        printf(__VA_ARGS__);          \
        printf("\n");                 \
    }                                 \
}

inline void print_memory_usage(const char* file__, int line__)
{
    size_t VmRSS, VmHWM;
    runtime::get_proc_status(&VmHWM, &VmRSS);

    std::vector<char> str(2048);
    int n = snprintf(&str[0], 2048, "[rank%04i at line %i of file %s]", mpi_comm_world().rank(), line__, file__);

    n += snprintf(&str[n], 2048, " VmHWM: %i Mb, VmRSS: %i Mb", static_cast<int>(VmHWM >> 20), static_cast<int>(VmRSS >> 20));

    #ifdef __GPU
    size_t gpu_mem = cuda_get_free_mem();
    n += snprintf(&str[n], 2048, ", GPU free memory: %i Mb", static_cast<int>(gpu_mem >> 20));
    #endif

    printf("%s\n", &str[0]);
}

#define MEMORY_USAGE_INFO() print_memory_usage(__FILE__, __LINE__);

#endif // __RUNTIME_H__
