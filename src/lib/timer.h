#ifndef __TIMER_H__
#define __TIMER_H__

namespace sirius 
{

class Timer
{
    private:
    
        std::string tname_;

        timeval start_;

        static std::map<std::string, double> timers_;

        static std::map<std::string, int> tcount_;
    
    public:
        
        Timer(std::string tname) : tname_(tname)
        {
            if (timers_.count(tname_) == 0) 
            {
                timers_[tname_] = 0;
                tcount_[tname_] = 0;
            }

            gettimeofday(&start_, NULL);
        }

        ~Timer()
        {
            timeval end;
            gettimeofday(&end, NULL);
            timers_[tname_] += double(end.tv_sec - start_.tv_sec) + double(end.tv_usec - start_.tv_usec) / 1e6;
            tcount_[tname_]++;
        }


        static void print()
        {
            std::map<std::string, double>::iterator it;
            for (it = timers_.begin(); it != timers_.end(); it++)
                printf("%s : %f (total) %f (average)\n", it->first.c_str(), it->second, it->second/tcount_[it->first]);
        }
 
        //static void print(std::string tname);
};

std::map<std::string, double> Timer::timers_;
std::map<std::string, int> Timer::tcount_;

};

#endif // __TIMER_H__
