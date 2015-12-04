#ifndef __DEBUG_HPP__
#define __DEBUG_HPP__

#include <fstream>
#include <sys/time.h>
#include "platform.h"
#include "communicator.h"

namespace debug
{

class Profiler
{
    private:
        std::string name_;
        std::string file_;
        int line_;

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

        static std::vector<std::string>& call_stack()
        {
            static std::vector<std::string> call_stack_;
            return call_stack_;
        }

    public:

        Profiler(char const* name__, char const* file__, int line__)
        {
            name_ = std::string(name__);
            file_ = std::string(file__);
            line_ = line__;

            char str[1024];
            snprintf(str, 1024, "%s at %s:%i", name__, file__, line__);

            call_stack().push_back(std::string(str));

            #ifdef __LOG_FUNC
            printf("rank%04i + %s\n", mpi_comm_world.rank(), name_.c_str());
            #endif
        }

        ~Profiler()
        {
            #ifdef __LOG_FUNC
            printf("rank%04i - %s\n", mpi_comm_world.rank(), name_.c_str());
            #endif
            call_stack().pop_back();
        }

        static void stack_trace()
        {
            int t = 0;
            for (auto it = call_stack().rbegin(); it != call_stack().rend(); it++)
            {
                for (int i = 0; i < t; i++) printf(" ");
                printf("[%s]\n", it->c_str());
                t++;
            }
        }
};

#ifdef __GNUC__
#define function_name__ __PRETTY_FUNCTION__
#else
#define function_name__ __func__
#endif

#ifdef __PROFILE
  #define PROFILE() debug::Profiler profiler__(function_name__, __FILE__, __LINE__);
#else
  #define PROFILE()
#endif

inline void get_proc_status(size_t* VmHWM, size_t* VmRSS)
{
    *VmHWM = 0;
    *VmRSS = 0;

    std::stringstream fname;
    fname << "/proc/self/status";
    
    std::ifstream ifs(fname.str().c_str());
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
                    printf("Platform::get_proc_status(): wrong units");
                    abort();
                }
                *VmHWM = tmp * 1024;
            }

            p = str.find("VmRSS:");
            if (p != std::string::npos)
            {
                std::stringstream s(str.substr(p + 7));
                s >> tmp;
                s >> units;

                if (units != "kB")
                {
                    printf("Platform::get_proc_status(): wrong units");
                    abort();
                }
                *VmRSS = tmp * 1024;
            }
        } 
    }
}

inline int get_num_threads()
{
    std::stringstream fname;
    fname << "/proc/self/status";

    int num_threds = -1;
    
    std::ifstream ifs(fname.str().c_str());
    if (ifs.is_open())
    {
        std::string str; 
        while (std::getline(ifs, str))
        {
            auto p = str.find("Threads:");
            if (p != std::string::npos)
            {
                std::stringstream s(str.substr(p + 9));
                s >> num_threds;
                break;
            }
        }
    }

    return num_threds;
}

#define MEMORY_USAGE_INFO()                                                                 \
{                                                                                           \
    size_t VmRSS, VmHWM;                                                                    \
    debug::get_proc_status(&VmHWM, &VmRSS);                                                 \
    printf("[rank%04i at line %i of file %s] VmHWM: %i Mb, VmRSS: %i Mb, mdarray: %i Mb\n", \
           Platform::rank(), __LINE__, __FILE__, int(VmHWM >> 20), int(VmRSS >> 20),        \
           int(mdarray_mem_count::allocated() >> 20));                                      \
}

};

#endif
