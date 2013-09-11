#ifndef __TIMER_H__
#define __TIMER_H__

namespace sirius 
{

struct timer_descriptor
{
    timer_descriptor() : total(0), last(0), count(0)
    {
    }

    double total;
    
    double last;

    int count;
};

class Timer
{
    private:

        /// descriptor of the current timer
        timer_descriptor* td_; 
        
        /// string label of the timer
        std::string label_;
        
        /// starting time
        timeval starting_time_;

        /// true if timer is running
        bool active_;

        /// mapping between timer name and timer descriptor pointer
        static std::map<std::string, timer_descriptor*> timer_descriptors_;
    
    public:
        
        Timer(const std::string& label__, bool start__ = true) : label_(label__), active_(false)
        {
            if (timer_descriptors_.count(label_) == 0)
            {   
                td_ = new timer_descriptor();
                timer_descriptors_[label_] = td_;
            }
            else 
            {   
                td_ = timer_descriptors_[label_];
            }

            if (start__) start();
        }

        ~Timer()
        {
            if (active_) stop();
        }

        static void clear()
        {
            timer_descriptors_.clear();
        }

        void start()
        {
            if (active_) error_local(__FILE__, __LINE__, "timer is already running");

            gettimeofday(&starting_time_, NULL);
            active_ = true;
        }

        void stop()
        {
            if (!active_)
            {
                std::stringstream s;
                s << "Timer " << label_ << " was not running";
                error_local(__FILE__, __LINE__, s);
            }

            timeval end;
            gettimeofday(&end, NULL);
           
            td_->last = double(end.tv_sec - starting_time_.tv_sec) + 
                        double(end.tv_usec - starting_time_.tv_usec) / 1e6;
            td_->count++;
            td_->total += td_->last;

            active_ = false;
        }

        static void print()
        {
            if (Platform::mpi_rank() == 0)
            {
                printf("\n");
                printf("Timers\n");
                for (int i = 0; i < 80; i++) printf("-");
                printf("\n");
 
                std::map<std::string, timer_descriptor*>::iterator it;
                for (it = timer_descriptors_.begin(); it != timer_descriptors_.end(); it++)
                {
                    double avg = (it->second->count == 0) ? 0.0 : it->second->total / it->second->count;
                    
                    printf("%-60s : %10.4f (total)   %10.4f (average)\n", it->first.c_str(), it->second->total, avg);
                }
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

        static std::map<std::string, timer_descriptor*>& timer_descriptors()
        {
            return timer_descriptors_;
        }
};

std::map<std::string, Timer*> ftimers;

std::map<std::string, timer_descriptor*> Timer::timer_descriptors_;

};

#endif // __TIMER_H__
