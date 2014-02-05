#include "timer.h"

namespace sirius
{

std::map<std::string, Timer*> ftimers;

std::map<std::string, std::vector<double> > Timer::timers_;

void Timer::start()
{
    if (active_) error_local(__FILE__, __LINE__, "timer is already running");

    gettimeofday(&starting_time_, NULL);
    active_ = true;
}

void Timer::stop()
{
    if (!active_)
    {
        std::stringstream s;
        s << "Timer " << label_ << " was not running";
        error_local(__FILE__, __LINE__, s);
    }

    timeval end;
    gettimeofday(&end, NULL);

    double val = double(end.tv_sec - starting_time_.tv_sec) + 
                 double(end.tv_usec - starting_time_.tv_usec) / 1e6;

    timers_[label_].push_back(val);

    active_ = false;
}

void Timer::print()
{
    if (Platform::mpi_rank() == 0)
    {
        printf("\n");
        printf("Timers\n");
        for (int i = 0; i < 115; i++) printf("-");
        printf("\n");
        printf("name                                                              count      total        min        max    average\n");
        for (int i = 0; i < 115; i++) printf("-");
        printf("\n");

        std::map<std::string, std::vector<double> >::iterator it;
        for (it = timers_.begin(); it != timers_.end(); it++)
        {
            int count = (int)it->second.size();
            double total = 0.0;
            double minval = 1e100;
            double maxval = 0.0;
            for (int i = 0; i < count; i++)
            {
                total += it->second[i];
                minval = std::min(minval, it->second[i]);
                maxval = std::max(maxval, it->second[i]);
            }
            double average = (count == 0) ? 0.0 : total / count;
            if (count == 0) minval = 0.0;

            printf("%-60s :    %5i %10.4f %10.4f %10.4f %10.4f\n", it->first.c_str(), count, total, minval, maxval, average);
        }
    }
}

void Timer::delay(double dsec)
{
    timeval t1;
    timeval t2;
    double d;

    gettimeofday(&t1, NULL);
    do
    {
        gettimeofday(&t2, NULL);
        d = double(t2.tv_sec - t1.tv_sec) + double(t2.tv_usec - t1.tv_usec) / 1e6;
    } while (d < dsec);
}

};

