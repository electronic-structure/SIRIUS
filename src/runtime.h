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
#include "config.h"
#include "communicator.h"

namespace runtime {

    inline void terminate(const char* file_name__, int line_number__, const std::string& message__)
    {
        printf("\n=== Fatal error at line %i of file %s ===\n", line_number__, file_name__);
        printf("%s\n\n", message__.c_str());
        raise(SIGTERM);
        throw std::runtime_error("terminating...");
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

    inline void get_proc_status(size_t* VmHWM__, size_t* VmRSS__)
    {
        *VmHWM__ = 0;
        *VmRSS__ = 0;
    
        std::ifstream ifs("/proc/self/status");
        if (ifs.is_open())
        {
            size_t tmp;
            std::string str; 
            std::string units;
            while (std::getline(ifs, str))
            {
                auto p = str.find("VmHWM:");
                if (p != std::string::npos)
                {
                    std::stringstream s(str.substr(p + 7));
                    s >> tmp;
                    s >> units;
    
                    if (units != "kB")
                    {
                        printf("runtime::get_proc_status(): wrong units");
                    }
                    else
                    {
                        *VmHWM__ = tmp * 1024;
                    }
                }
    
                p = str.find("VmRSS:");
                if (p != std::string::npos)
                {
                    std::stringstream s(str.substr(p + 7));
                    s >> tmp;
                    s >> units;
    
                    if (units != "kB")
                    {
                        printf("runtime::get_proc_status(): wrong units");
                    }
                    else
                    {
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

    /// Simple timer interface.
    class Timer
    {
        private:

            #ifdef __TIMER
            /// String label of the timer.
            std::string label_;
            
            /// Starting time.
            #if defined(__TIMER_TIMEOFDAY)
            timeval starting_time_;
            #elif defined(__TIMER_MPI_WTIME)
            double starting_time_;
            #elif defined(__TIMER_CHRONO)
            std::chrono::high_resolution_clock::time_point starting_time_;
            #endif
    
            Communicator const* comm_;
    
            /// true if timer is running
            bool active_;
            #endif
        
        public:

            struct timer_stats
            {
                int count;
                double min_value;
                double max_value;
                double total_value;
                double average_value;
            };
            
            #ifdef __TIMER
            Timer(std::string const& label__) 
                : label_(label__),
                  comm_(nullptr),
                  active_(false)
            {
                if (timers().count(label_) == 0) timers()[label_] = std::vector<double>();
    
                start();
            }
            #else
            Timer(std::string const& label__)
            {
            }
            #endif
            
            #ifdef __TIMER
            Timer(std::string const& label__, Communicator const& comm__) 
                : label_(label__),
                  comm_(&comm__),
                  active_(false)
            {
                if (timers().count(label_) == 0) timers()[label_] = std::vector<double>();
    
                start();
            }
            #else
            Timer(std::string const& label__, Communicator const& comm__)
            {
            }
            #endif
    
            ~Timer()
            {
                #ifdef __TIMER
                if (active_) stop();
                #endif
            }
    
            void start()
            {
                #ifdef __TIMER
                #ifndef NDEBUG
                if (omp_get_num_threads() != 1)
                {
                    printf("std::map used by Timer is not thread-safe\n");
                    printf("timer name: %s\n", label_.c_str());
                    exit(-1);
                }
                #endif

                if (active_)
                {
                    printf("timer %s is already running\n", label_.c_str());
                    exit(-2);
                }
                if (comm_ != nullptr) comm_->barrier();
                #if defined(__TIMER_TIMEOFDAY)
                gettimeofday(&starting_time_, NULL);
                #elif defined(__TIMER_MPI_WTIME)
                starting_time_ = MPI_Wtime();
                #elif defined (__TIMER_CHRONO)
                starting_time_ = std::chrono::high_resolution_clock::now();
                #endif
                active_ = true;
                #endif
            }
    
            double stop()
            {
                #ifdef __TIMER
                if (!active_)
                {
                    printf("timer %s was not running\n", label_.c_str());
                    exit(-2);
                }
                if (comm_ != nullptr) comm_->barrier();

                #if defined(__TIMER_TIMEOFDAY)
                timeval end;
                gettimeofday(&end, NULL);
                double val = double(end.tv_sec - starting_time_.tv_sec) + 
                             double(end.tv_usec - starting_time_.tv_usec) / 1e6;
                #elif defined(__TIMER_MPI_WTIME)
                double val = MPI_Wtime() - starting_time_;
                #elif defined(__TIMER_CHRONO)
                auto t2 = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> tdiff = std::chrono::duration_cast< std::chrono::duration<double> >(t2 - starting_time_);
                double val = tdiff.count();
                #endif
                
                timers()[label_].push_back(val);

                active_ = false;

                return val;
                #else
                return 0;
                #endif
            }
    
            static void clear()
            {
                #ifdef __TIMER
                timers().clear();
                #endif
            }
            
            #ifdef __TIMER
            static std::map< std::string, std::vector<double> >& timers()
            {
                static std::map< std::string, std::vector<double> > timers_;
                return timers_;
            }
    
            static std::map< std::string, timer_stats> collect_timer_stats()
            {
                std::map< std::string, timer_stats> tstats;

                /* collect local timers */
                for (auto& it: timers())
                {
                    timer_stats ts;

                    ts.count = (int)it.second.size();
                    ts.total_value = 0.0;
                    ts.min_value = 1e100;
                    ts.max_value = 0.0;
                    for (int i = 0; i < ts.count; i++)
                    {
                        ts.total_value += it.second[i];
                        ts.min_value = std::min(ts.min_value, it.second[i]);
                        ts.max_value = std::max(ts.max_value, it.second[i]);
                    }
                    ts.average_value = (ts.count == 0) ? 0.0 : ts.total_value / ts.count;
                    if (ts.count == 0) ts.min_value = 0.0;

                    tstats[it.first] = ts;
                }

                return tstats;
            }

            static double value(std::string const& label__)
            {
                auto values = timers()[label__];
    
                double d = 0;
                for (double v: values) d += v;
                return d;
            }
    
            static void print()
            {
                std::map< std::string, timer_stats> tstats = collect_timer_stats();

                if (mpi_comm_world().rank() == 0)
                {
                    printf("\n");
                    printf("Timers\n");
                    for (int i = 0; i < 115; i++) printf("-");
                    printf("\n");
                    printf("name                                                              count      total        min        max    average\n");
                    for (int i = 0; i < 115; i++) printf("-");
                    printf("\n");

                    std::map<std::string, timer_stats>::iterator it;
                    for (it = tstats.begin(); it != tstats.end(); it++)
                    {
                        auto ts = it->second;
                        printf("%-60s :    %5i %10.4f %10.4f %10.4f %10.4f\n", it->first.c_str(), ts.count, ts.total_value, 
                               ts.min_value, ts.max_value, ts.average_value);
                    }
                    
                    //#ifdef __GPU
                    //print_cuda_timers();
                    //#endif
                }
            }
            #else
            static void print()
            {
            }
            #endif
    };

    class Profiler
    {
        private:
    
            std::string name_;
            std::string file_;
            int line_;
            Timer* timer_;
    
            std::string timestamp()
            {
                timeval t;
                gettimeofday(&t, NULL);
            
                char buf[100]; 
            
                tm* ptm = localtime(&t.tv_sec); 
                //strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", ptm); 
                strftime(buf, sizeof(buf), "%H:%M:%S", ptm); 
                return std::string(buf);
            }
    
            #ifdef __PROFILE_STACK
            static std::vector<std::string>& call_stack()
            {
                static std::vector<std::string> call_stack_;
                return call_stack_;
            }
            #endif
    
            inline void init(char const* name__, char const* file__, int line__)
            {
                #if defined(__PROFILE_STACK) || defined(__PROFILE_FUNC)
                name_ = std::string(name__);
                file_ = std::string(file__);
                line_ = line__;
    
                char str[1024];
                snprintf(str, 1024, "%s at %s:%i", name__, file__, line__);
                #endif
                
                #ifdef __PROFILE_STACK
                call_stack().push_back(std::string(str));
                #endif
    
                #ifdef __PROFILE_FUNC
                int tab = 0;
                #ifdef __PROFILE_STACK
                tab = static_cast<int>(call_stack().size()) - 1;
                #endif
                for (int i = 0; i < tab; i++) printf(" ");
                printf("[rank%04i] + %s\n", mpi_comm_world().rank(), name_.c_str());
                #endif
            }
    
        public:
    
            Profiler(char const* name__, char const* file__, int line__) : timer_(nullptr)
            {
                init(name__, file__, line__);
            }
    
            Profiler(char const* name__, char const* file__, int line__, char const* timer_name_)
            {
                init(name__, file__, line__);
                
                #ifdef __PROFILE_TIME
                timer_ = new Timer(timer_name_);
                #endif
            }
    
            ~Profiler()
            {
                #ifdef __PROFILE_TIME
                if (timer_ != nullptr) delete timer_;
                #endif
    
                #ifdef __PROFILE_FUNC
                int tab = 0;
                #ifdef __PROFILE_STACK
                tab = static_cast<int>(call_stack().size()) - 1;
                #endif
                for (int i = 0; i < tab; i++) printf(" ");
                printf("[rank%04i] - %s\n", mpi_comm_world().rank(), name_.c_str());
                #endif
    
                #ifdef __PROFILE_STACK
                call_stack().pop_back();
                #endif
            }
    
            static void stack_trace()
            {
                #ifdef __PROFILE_STACK
                int t = 0;
                for (auto it = call_stack().rbegin(); it != call_stack().rend(); it++)
                {
                    for (int i = 0; i < t; i++) printf(" ");
                    printf("[%s]\n", it->c_str());
                    t++;
                }
                #endif
            }
    };

};

#define TERMINATE(msg) runtime::terminate(__FILE__, __LINE__, msg);

#define WARNING(msg) runtime::warning(__FILE__, __LINE__, msg);

#define STOP() TERMINATE("terminated by request")

#if (__VERBOSITY > 1)
const bool _enable_dump_ = true;
#else
const bool _enable_dump_ = false;
#endif

#define DUMP(...)                                                                     \
if (_enable_dump_)                                                                    \
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

#define MEMORY_USAGE_INFO()                                                                  \
{                                                                                            \
    size_t VmRSS, VmHWM;                                                                     \
    runtime::get_proc_status(&VmHWM, &VmRSS);                                                \
    printf("[rank%04i at line %i of file %s] VmHWM: %i Mb, VmRSS: %i Mb\n",                  \
           mpi_comm_world().rank(), __LINE__, __FILE__, int(VmHWM >> 20), int(VmRSS >> 20)); \
}

#ifdef __GNUC__
#define __function_name__ __PRETTY_FUNCTION__
#else
#define __function_name__ __func__
#endif

#ifdef __PROFILE
  #define PROFILE() runtime::Profiler profiler__(__function_name__, __FILE__, __LINE__);
  #define PROFILE_WITH_TIMER(name) runtime::Profiler profiler__(__function_name__, __FILE__, __LINE__, name);
#else
  #define PROFILE(...)
  #define PROFILE_WITH_TIMER(name) 
#endif

#define TIMER(name) runtime::Timer timer__(name);

#endif // __RUNTIME_H__
