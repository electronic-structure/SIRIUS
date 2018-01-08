#ifndef __TIMER_HPP__
#define __TIMER_HPP__

//#define __TIMER_SEQUENCE

#include <omp.h>
#include <string>
#include <sstream>
#include <chrono>
#include <map>
#include <vector>
#include <memory>
#include <complex>
#include <algorithm>
#include "json.hpp"

struct timer_stats_t
{
    double min_val{1e10};
    double max_val{0};
    double tot_val{0};
    double avg_val{0};
    int count{0};
#ifdef __TIMER_SEQUENCE
    std::vector<double> sequence;
#endif
};

using time_point_t = std::chrono::high_resolution_clock::time_point;

const std::string main_timer_label = "+global_timer";

/// A simple timer implementation.
class timer
{
  private:
    /// Unique label of the timer.
    std::string label_;

    /// Starting time.
    time_point_t starting_time_;

    /// True if timer is active.
    bool active_{false};

    /// List of child timers that we called inside another timer.
    static std::vector<std::string>& stack()
    {
        static std::vector<std::string> stack_;
        return stack_;
    }

    /// Mapping between timer label and timer counters.
    static std::map<std::string, timer_stats_t>& timer_values()
    {
        static std::map<std::string, timer_stats_t> timer_values_;
        return timer_values_;
    }

    static timer_stats_t& timer_values(std::string label__)
    {
        return timer_values()[label__];
    }
    
    /// Mapping between parent timer and child timers.
    /** This map is needed to build a call tree of timers with the information about "self" time
        and time spent in calling other timers. */
    static std::map<std::string, std::map<std::string, double>>& timer_values_ex()
    {
        /* the following map is stored:
           
           parent_timer_label1  |--- child_timer_label1, time1a
                                |--- child timer_label2, time2
                                |--- child_timer_label3, time3

           parent_timer_label2  |--- child_timer_label1, time1b
                                |--- child_timer_label4, time4

           etc.
        */
        static std::map<std::string, std::map<std::string, double>> timer_values_ex_;
        return timer_values_ex_;
    }

  public:
    
    /// Constructor.
    timer(std::string label__)
        : label_(label__)
    {
        /* measure the starting time */
        starting_time_ = std::chrono::high_resolution_clock::now();
        /* add timer label to the list of called timers */
        stack().push_back(label_);
        active_ = true;
    }
    
    /// Destructor.
    ~timer()
    {
        stop();
    }
    
    /// Stop the timer and update the statistics.
    double stop()
    {
        if (!active_) {
            return 0;
        }

        /* remove this timer name from the list; now last element contains
           the name of the parent timer */
        stack().pop_back();
        
        /* measure the time difference */
        auto t2    = std::chrono::high_resolution_clock::now();
        auto tdiff = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - starting_time_);
        double val = tdiff.count();

        auto& ts = timer_values(label_);
#ifdef __TIMER_SEQUENCE
        auto abs_time = std::chrono::duration_cast<std::chrono::duration<double>>(starting_time_.time_since_epoch());
        double abs_time_val = abs_time.count();
        ts.sequence.push_back(abs_time_val);
        abs_time = std::chrono::duration_cast<std::chrono::duration<double>>(t2.time_since_epoch());
        abs_time_val = abs_time.count();
        ts.sequence.push_back(abs_time_val);
#endif
        ts.min_val = std::min(ts.min_val, val);
        ts.max_val = std::max(ts.max_val, val);
        ts.tot_val += val;
        ts.count++;

        if (stack().size() != 0) {
            /* last element contains the name of the parent timer */
            auto parent_label = stack().back();
            /* add value to the parent timer */
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
        global_timer().stop();

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
            if (te > it.second.tot_val) {
                printf("wrong timer values: %f %f\n", te, it.second.tot_val);
                throw std::runtime_error("terminating...");

            }
            printf("%-65s : %6i %10.4f %10.4f %10.4f %10.4f     %6.2f\n", it.first.c_str(),
                                                                          it.second.count,
                                                                          it.second.tot_val,
                                                                          it.second.min_val,
                                                                          it.second.max_val,
                                                                          it.second.tot_val / it.second.count,
                                                                          (it.second.tot_val - te) / it.second.tot_val * 100);
        }
    }

    static void print_tree()
    {
        global_timer().stop();

        if (!timer_values().count(main_timer_label)) {
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

    static nlohmann::json serialize_timers()
    {
        global_timer().stop();

        nlohmann::json dict;

#ifdef __TIMER_SEQUENCE
        double start_time = timer_values()[main_timer_label].sequence[0];
#endif
        /* collect local timers */
        for (auto& it: timer::timer_values()) {
            timer_stats_t ts;
            nlohmann::json node;
            node["count"] = it.second.count;
            node["total"] = it.second.tot_val;
            node["min"] = it.second.min_val;
            node["max"] = it.second.max_val;
            node["avg"] = it.second.tot_val / it.second.count;
#ifdef __TIMER_SEQUENCE
            for (size_t i = 0; i < it.second.sequence.size(); i++) {
                it.second.sequence[i] -= start_time;
            }
            node["sequence"] = it.second.sequence;
#endif
            dict[it.first] = node;
        }
        return std::move(dict);
    }

    static nlohmann::json serialize_timers_tree()
    {
        global_timer().stop();

        nlohmann::json dict;

        if (!timer_values().count(main_timer_label)) {
            return {};
        }
        /* total execution time */
        double ttot = timer_values()[main_timer_label].tot_val;

        for (auto& it: timer_values()) {
            if (timer_values_ex().count(it.first)) {
                /* collect external times */
                double te{0};
                for (auto& it2: timer_values_ex()[it.first]) {
                    te += it2.second;
                }
                nlohmann::json node;

                double f = it.second.tot_val / ttot;
                if (f > 0.01) {
                    node["cumulitive_time"]        = it.second.tot_val;
                    node["self_time"]              = it.second.tot_val - te;
                    node["percent_of_global_time"] = f * 100;

                    std::vector<std::pair<double, std::string>> tmp;

                    for (auto& it2: timer_values_ex()[it.first]) {
                        tmp.push_back(std::make_pair(it2.second / it.second.tot_val, it2.first));
                    }
                    std::sort(tmp.rbegin(), tmp.rend());
                    node["call"] = {};
                    for (auto& e: tmp) {
                        nlohmann::json n;
                        n["time"]              = timer_values_ex()[it.first][e.second];
                        n["percent_of_parent"] = e.first * 100;
                        node["call"][e.second] = n;
                    }
                    dict[it.first] = node;
                }
            }
        }

        return std::move(dict);
    }

    inline static timer& global_timer()
    {
        static timer global_timer__(main_timer_label);
        return global_timer__;
    }
};

/* this is needed only to call timer::global_timer() at the beginning */
static timer* global_timer_init__ = &timer::global_timer();

#endif // __SDDK_INTERNAL_HPP__
