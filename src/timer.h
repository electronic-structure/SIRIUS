#ifndef __TIMER_H__
#define __TIMER_H__

#include <sys/time.h>
#include <map>
#include <string>
#include <vector>
#include "platform.h"
#include "error_handling.h"

namespace sirius 
{

class Timer
{
    private:
        
        /// string label of the timer
        std::string label_;
        
        /// starting time
        timeval starting_time_;

        /// true if timer is running
        bool active_;

        /// mapping between timer name and timer values
        static std::map<std::string, std::vector<double> > timers_;
    
    public:
        
        Timer(const std::string& label__, bool start__ = true) : label_(label__), active_(false)
        {
            if (timers_.count(label_) == 0) timers_[label_] = std::vector<double>();

            if (start__) start();
        }

        ~Timer()
        {
            if (active_) stop();
        }

        static void clear()
        {
            timers_.clear();
        }

        static std::map<std::string, std::vector<double> >& timers()
        {
            return timers_;
        }

        void start();

        void stop();

        static void print();

        static void delay(double dsec);
};

};

#endif // __TIMER_H__
