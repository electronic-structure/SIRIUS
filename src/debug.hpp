#ifndef __DEBUG_HPP__
#define __DEBUG_HPP__

#include <fstream>

namespace debug
{

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

template <typename T>
inline T check_sum(matrix<T> const& mtrx, int irow0, int icol0, int nrow, int ncol)
{
    T sum = 0;

    for (int j = 0; j < ncol; j++)
    {
        for (int i = 0; i < nrow; i++) sum += mtrx(irow0 + i, icol0 + j);
    }

    return sum;
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
