#ifndef __TIMER_H__
#define __TIMER_H__

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

        /// mapping between timer name and timer's value
        static std::map<std::string, double> timers_;
        
        /// number of measures
        static std::map<std::string, int> tcount_;
    
    public:
        
        Timer(const std::string& label__, bool start__ = true) : label_(label__), active_(false)
        {
            if (timers_.count(label_) == 0) 
            {
                timers_[label_] = 0;
                tcount_[label_] = 0;
            }

            if (start__) start();
        }

        ~Timer()
        {
            if (active_) stop();
        }

        void start()
        {
            if (active_)
                error(__FILE__, __LINE__, "timer is already running", fatal_err);

            gettimeofday(&starting_time_, NULL);
            active_ = true;
        }

        void stop()
        {
            if (!active_)
                error(__FILE__, __LINE__, "timer was not running", fatal_err);

            timeval end;
            gettimeofday(&end, NULL);
            timers_[label_] += double(end.tv_sec - starting_time_.tv_sec) + 
                               double(end.tv_usec - starting_time_.tv_usec) / 1e6;
            tcount_[label_]++;

            active_ = false;
        }

        static void print()
        {
            if (Platform::verbose())
            {
                printf("\n");
                printf("Timers\n");
                for (int i = 0; i < 80; i++) printf("-");
                printf("\n");
 
                std::map<std::string, double>::iterator it;
                for (it = timers_.begin(); it != timers_.end(); it++)
                {
                    double avg = (tcount_[it->first] == 0) ? 0.0 : it->second/tcount_[it->first];
                    printf("%-60s : %10.4f (total)   %10.4f (average)\n", it->first.c_str(), it->second, avg);
                }
            }

            json_write();
        }

        static void json_write()
        {
            if (Platform::mpi_rank() == 0)
            {
                FILE* fout = fopen("timers.json", "w");

                fprintf(fout, "{");

                std::map<std::string, double>::iterator it;
                for (it = timers_.begin(); it != timers_.end(); it++)
                {
                    if (it != timers_.begin()) fprintf(fout, ",");
                    double avg = (tcount_[it->first] == 0) ? 0.0 : it->second/tcount_[it->first];
                    fprintf(fout, "\n    \"%s\" : %10.4f", it->first.c_str(), avg);
                }
                
                fprintf(fout, "\n}\n");
                
                fclose(fout);
            }
        }

        static void delay(double dsec)
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

std::map<std::string,double> Timer::timers_;
std::map<std::string,int> Timer::tcount_;
std::map<std::string,Timer*> ftimers;

};

#endif // __TIMER_H__
