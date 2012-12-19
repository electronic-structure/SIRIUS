#ifndef __UTILS_H__
#define __UTILS_H__

class Utils
{
    public:

        static bool file_exists(const std::string& file_name)
        {
            std::ifstream ifs(file_name.c_str());
            if (ifs.is_open()) return true;
            return false;
        }

        static inline double vector_length(double v[3])
        {
            return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        }


};

#endif

