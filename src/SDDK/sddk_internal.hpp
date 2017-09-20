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

namespace sddk {

struct timer_stats_t
{
    double min_val{1e10};
    double max_val{0};
    double tot_val{0};
    double avg_val{0};
    int count{0};
};

using time_point_t = std::chrono::high_resolution_clock::time_point;

const std::string main_timer_label = "+global_timer";

class timer
{
  private:

    std::string label_;
    time_point_t starting_time_;
    bool active_{false};

    static std::vector<std::string>& stack()
    {
        static std::vector<std::string> stack_;
        return stack_;
    }

    static std::map<std::string, std::map<std::string, double>>& timer_values_ex()
    {
        static std::map<std::string, std::map<std::string, double>> timer_values_ex_;
        return timer_values_ex_;
    }

  public:

    timer(std::string label__)
        : label_(label__)
    {
        starting_time_ = std::chrono::high_resolution_clock::now();
        stack().push_back(label_);
        active_ = true;
    }

    ~timer()
    {
        stop();
    }

    static std::map<std::string, timer_stats_t>& timer_values()
    {
        static std::map<std::string, timer_stats_t> timer_values_;
        return timer_values_;
    }

    double stop()
    {
        if (!active_) {
            return 0;
        }

        stack().pop_back();

        auto t2    = std::chrono::high_resolution_clock::now();
        auto tdiff = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - starting_time_);
        double val = tdiff.count();

        auto& ts = timer_values()[label_];
        ts.min_val = std::min(ts.min_val, val);
        ts.max_val = std::max(ts.max_val, val);
        ts.tot_val += val;
        ts.count++;

        if (stack().size() != 0) {
            auto parent_label = stack().back();
            if (timer_values_ex().count(parent_label) == 0) {
                timer_values_ex()[parent_label] = std::map<std::string, double>();
            }
            if (timer_values_ex()[parent_label].count(label_) == 0) {
                timer_values_ex()[parent_label][label_] = 0;
            }
            timer_values_ex()[parent_label][label_] += val;
        }
        active_ = false;
        return val;
    }

    static void print()
    {
        if (mpi_comm_world().rank()) {
            return;
        }
        for (int i = 0; i < 140; i++) {
            printf("-");
        }
        printf("\n");
        printf("name                                                                 count      total        min        max    average    self (%%)\n");
        for (int i = 0; i < 140; i++) {
            printf("-");
        }
        printf("\n");
        for (auto& it: timer_values()) {

            double te{0};
            if (timer_values_ex().count(it.first)) {
                for (auto& it2: timer_values_ex()[it.first]) {
                    te += it2.second;
                }
            }
            if (it.second.tot_val > 0.01) {
                printf("%-65s : %6i %10.4f %10.4f %10.4f %10.4f     %6.2f\n", it.first.c_str(),
                                                                              it.second.count,
                                                                              it.second.tot_val,
                                                                              it.second.min_val,
                                                                              it.second.max_val,
                                                                              it.second.tot_val / it.second.count,
                                                                              (it.second.tot_val - te) / it.second.tot_val * 100);
            }
            //if (timer_values_ex().count(it.first)) {
            //    for (auto& it2: timer_values_ex()[it.first]) {
            //        printf("|-%s (%6.2f %%) \n", it2.first.c_str(), (it2.second / it.second.tot_val) * 100);
            //    }
            //}

            //for (int i = 0; i < 140; i++) {
            //    printf("-");
            //}
            //printf("\n");
        }
    }

    static void print_tree()
    {
        if (!timer_values().count(main_timer_label)) {
            return;
        }
        if (mpi_comm_world().rank()) {
            return;
        }
        for (int i = 0; i < 140; i++) {
            printf("-");
        }
        printf("\n");

        double ttot = timer_values()[main_timer_label].tot_val;

        for (auto& it: timer_values()) {
            if (timer_values_ex().count(it.first)) {
                /* collect external times */
                double te{0};
                for (auto& it2: timer_values_ex()[it.first]) {
                    te += it2.second;
                }
                double f = it.second.tot_val / ttot;
                if (f > 0.01) {
                    printf("%s (%10.4fs, %.2f %% of self, %.2f %% of total)\n",
                           it.first.c_str(), it.second.tot_val, (it.second.tot_val - te) / it.second.tot_val * 100, f * 100);
                
                    std::vector<std::pair<double, std::string>> tmp;
            
                    for (auto& it2: timer_values_ex()[it.first]) {
                        tmp.push_back(std::pair<double, std::string>(it2.second / it.second.tot_val, it2.first));
                    }
                    std::sort(tmp.rbegin(), tmp.rend());
                    for (auto& e: tmp) {
                        printf("|--%s (%10.4fs, %.2f %%) \n", e.second.c_str(), timer_values_ex()[it.first][e.second], e.first * 100);
                    }
                }
            }
        }
    }

    static std::map<std::string, timer_stats_t>& collect_timer_stats()
    {
        for (auto& it: timer_values()) {
            it.second.avg_val = it.second.tot_val / it.second.count;
        }
        return timer_values();
    }
};

inline static timer& global_timer()
{
    static timer global_timer__(main_timer_label);
    return global_timer__;
}

inline void start_global_timer()
{
    global_timer();
}

inline void stop_global_timer()
{
    global_timer().stop();
}


///// Simple timer interface.
//class timer
//{
//  private:
//    /// String label of the timer.
//    std::string label_;
//
//    /// Starting time.
//    std::chrono::high_resolution_clock::time_point starting_time_;
//
//    /// True if timer is running.
//    bool active_{false};
//
//  public:
//    struct timer_stats
//    {
//        int count;
//        double min_value;
//        double max_value;
//        double average_value;
//        double total_value;
//    };
//
//    timer(std::string const& label__)
//        : label_(label__)
//    {
//        if (omp_get_num_threads() != 1) {
//            std::stringstream s;
//            s << "sddk::timer does not support threads";
//            throw std::runtime_error(s.str());
//        }
//        //#pragma omp critical
//        if (timers().count(label_) == 0) {
//            timers()[label_] = std::vector<double>();
//        }
//        start();
//    }
//
//    ~timer()
//    {
//        if (active_) {
//            stop();
//        }
//    }
//
//    void start()
//    {
//        if (active_) {
//            std::stringstream s;
//            s << "timer " << label_ << " is already running";
//            throw std::runtime_error(s.str());
//        }
//        starting_time_ = std::chrono::high_resolution_clock::now();
//        active_        = true;
//    }
//
//    double stop()
//    {
//        if (!active_) {
//            std::stringstream s;
//            s << "timer " << label_ << " was not running";
//            throw std::runtime_error(s.str());
//        }
//        auto t2    = std::chrono::high_resolution_clock::now();
//        auto tdiff = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - starting_time_);
//        double val = tdiff.count();
//
//        //#pragma omp critical
//        timers()[label_].push_back(val);
//
//        active_ = false;
//
//        return val;
//    }
//
//    // static void clear()
//    //{
//    //    #ifdef __TIMER
//    //    timers().clear();
//    //    #endif
//    //}
//
//    static std::map<std::string, std::vector<double>>& timers()
//    {
//        static std::map<std::string, std::vector<double>> timers_;
//        return timers_;
//    }
//
//    static std::map<std::string, timer_stats> collect_timer_stats()
//    {
//        std::map<std::string, timer_stats> tstats;
//
//        /* collect local timers */
//        for (auto& it : timers()) {
//            timer_stats ts;
//
//            ts.count       = static_cast<int>(it.second.size());
//            ts.total_value = 0.0;
//            ts.min_value   = 1e100;
//            ts.max_value   = 0.0;
//            for (auto& e : it.second) {
//                ts.total_value += e;
//                ts.min_value = std::min(ts.min_value, e);
//                ts.max_value = std::max(ts.max_value, e);
//            }
//            ts.average_value = (ts.count == 0) ? 0.0 : ts.total_value / ts.count;
//            if (ts.count == 0) {
//                ts.min_value = 0.0;
//            }
//
//            tstats[it.first] = ts;
//        }
//
//        return tstats;
//    }
//
//    //==static json serialize()
//    //=={
//    //==    json dict;
//
//    //==    /* collect local timers */
//    //==    for (auto& it: timers()) {
//    //==        timer_stats ts;
//
//    //==        ts.count = static_cast<int>(it.second.size());
//    //==        ts.total_value = 0.0;
//    //==        ts.min_value = 1e100;
//    //==        ts.max_value = 0.0;
//    //==        for (int i = 0; i < ts.count; i++) {
//    //==            ts.total_value += it.second[i];
//    //==            ts.min_value = std::min(ts.min_value, it.second[i]);
//    //==            ts.max_value = std::max(ts.max_value, it.second[i]);
//    //==        }
//    //==        ts.average_value = (ts.count == 0) ? 0.0 : ts.total_value / ts.count;
//    //==        if (ts.count == 0) {
//    //==            ts.min_value = 0.0;
//    //==        }
//
//    //==        dict[it.first] = {ts.total_value, ts.average_value, ts.min_value, ts.max_value};
//    //==    }
//    //==    return std::move(dict);
//    //==}
//
//    static double value(std::string const& label__)
//    {
//        auto values = timers()[label__];
//
//        double d{0};
//        for (double v : values) {
//            d += v;
//        }
//        return d;
//    }
//
//    static void print(double min_total__ = 0.1)
//    {
//        auto tstats = collect_timer_stats();
//
//        if (mpi_comm_world().rank() == 0) {
//            printf("\n");
//            printf("Timers\n");
//            for (int i = 0; i < 120; i++)
//                printf("-");
//            printf("\n");
//            printf("name                                                                   count      total        min        max    average\n");
//            for (int i = 0; i < 120; i++) {
//                printf("-");
//            }
//            printf("\n");
//
//            for (auto it = tstats.begin(); it != tstats.end(); it++) {
//                auto ts = it->second;
//                if (ts.total_value > min_total__) {
//                    printf("%-65s :    %5i %10.4f %10.4f %10.4f %10.4f\n", it->first.c_str(), ts.count, ts.total_value,
//                           ts.min_value, ts.max_value, ts.average_value);
//                }
//            }
//
//            //#ifdef __GPU
//            // print_cuda_timers();
//            //#endif
//        }
//    }
//
//    // static void print_all()
//    //{
//    //    char host_name[1024];
//    //    gethostname(host_name, 1024);
//
//    //    auto tstats = collect_timer_stats();
//
//    //    std::vector<std::string> timer_names;
//
//    //    if (mpi_comm_world().rank() == 0)
//    //    {
//    //        for (auto it = tstats.begin(); it != tstats.end(); it++)
//    //        {
//    //            timer_names.push_back(it->first);
//    //        }
//    //    }
//    //    int nt = static_cast<int>(timer_names.size());
//    //    mpi_comm_world().bcast(&nt, 1, 0);
//    //    if (mpi_comm_world().rank() != 0) timer_names = std::vector<std::string>(nt);
//
//    //    for (int i = 0; i < nt; i++)
//    //        mpi_comm_world().bcast(timer_names[i], 0);
//
//    //    if (mpi_comm_world().rank() == 0)
//    //    {
//    //        printf("\n");
//    //        printf("Timers\n");
//    //        for (int i = 0; i < 115; i++) printf("-");
//    //        printf("\n");
//    //        printf("name                                                              count      total        min
//    //        max    average\n");
//    //        for (int i = 0; i < 115; i++) printf("-");
//    //        printf("\n");
//    //    }
//    //    pstdout pout(mpi_comm_world());
//
//    //    for (int i = 0; i < nt; i++)
//    //    {
//    //        if (mpi_comm_world().rank() == 0)
//    //            pout.printf("---- %s ----\n", timer_names[i].c_str());
//
//    //        if (tstats.count(timer_names[i]) != 0)
//    //        {
//    //            auto ts = tstats[timer_names[i]];
//    //            pout.printf("%-60s :    %5i %10.4f %10.4f %10.4f %10.4f\n", host_name, ts.count, ts.total_value,
//    //                        ts.min_value, ts.max_value, ts.average_value);
//    //        }
//    //        pout.flush();
//    //    }
//
//    //    //std::vector<std::string> host_name(comm__.size());
//    //    //host_name[comm__.rank()] = std::string(buf);
//
//    //    //for (int i = 0; i < comm__.size(); i++)
//    //    //{
//    //    //    int sz;
//    //    //    if (i = comm__.rank())
//    //    //    {
//    //    //        sz = static_cast<int>(host_name[i].size());
//    //    //        std::memcpy(buf, host_name[i].str().c_str(), sz);
//    //    //    }
//    //    //    comm__.bcast(&sz, 1, i);
//    //    //    comm__.bcast(buf, sz, i);
//    //    //    buf[sz] = 0;
//    //    //    host_name[i] = std::string(buf);
//    //    //}
//
//    //    //for (int i = 0; i < comm__.size(); i++)
//    //    //{
//    //    //    if (i = comm__.rank());
//
//    //    //}
//    //}
//};

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

/// Wrapper for data types
template <typename T>
class sddk_type_wrapper;

template <>
class sddk_type_wrapper<double>
{
  public:
    static inline double conjugate(double const& v)
    {
        return v;
    }

    static inline double real(double const& v)
    {
        return v;
    }
};

template <>
class sddk_type_wrapper<double_complex>
{
  public:
    static inline double_complex conj(double_complex const& v)
    {
        return std::conj(v);
    }

    static inline double real(double_complex const& v)
    {
        return v.real();
    }
};

inline void terminate(const char* file_name__, int line_number__, const std::string& message__)
{
    printf("\n=== Fatal error at line %i of file %s ===\n", line_number__, file_name__);
    printf("%s\n\n", message__.c_str());
    // raise(SIGTERM);
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

} // namespace sddk

#define TERMINATE(msg) sddk::terminate(__FILE__, __LINE__, msg);

#define WARNING(msg) sddk::warning(__FILE__, __LINE__, msg);

#define STOP() TERMINATE("terminated by request")

#endif // __SDDK_INTERNAL_HPP__
