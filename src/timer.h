#ifndef __TIMER_H__
#define __TIMER_H__

#include <sys/time.h>
#include <map>
#include <string>
#include <iostream>
#include <vector>
#include "platform.h"
#include "error_handling.h"

namespace sirius 
{

const int _local_timer_ = 0;
const int _global_timer_ = 1;

struct timer_stats
{
    int count;
    double min_value;
    double max_value;
    double total_value;
    double average_value;
    int timer_type;
};

class Timer
{
    private:
        
        /// string label of the timer
        std::string label_;
        
        /// starting time
        timeval starting_time_;

        /// true if timer is running
        bool active_;

        int timer_type_;

        /// mapping between timer name and timer values
        static std::map< std::string, std::vector<double> > timers_;

        static std::map< std::string, std::vector<double> > global_timers_;
    
    public:
        
        Timer(const std::string& label__) 
            : label_(label__), 
              active_(false), 
              timer_type_(_local_timer_)
        {
            if (timers_.count(label_) == 0) timers_[label_] = std::vector<double>();

            start();
        }

        Timer(const std::string& label__, int timer_type__) 
            : label_(label__), 
              active_(false), 
              timer_type_(timer_type__)
        {
            switch (timer_type_)
            {
                case _local_timer_:
                {
                    if (timers_.count(label_) == 0) timers_[label_] = std::vector<double>();
                    break;
                }
                case _global_timer_:
                {
                    if (global_timers_.count(label_) == 0) global_timers_[label_] = std::vector<double>();
                    break;
                }
            }

            start();
        }

        ~Timer()
        {
            if (active_) stop();
        }

        void start();

        void stop();

        double value();

        static void clear()
        {
            timers_.clear();
        }

        static std::map< std::string, std::vector<double> >& timers()
        {
            return timers_;
        }

        static std::map< std::string, timer_stats> collect_timer_stats();

        static void print();

        static void delay(double dsec);
};

};

#endif // __TIMER_H__
