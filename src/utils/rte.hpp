#ifndef __RTE_HPP__
#define __RTE_HPP__

#include <stdexcept>
#include <sstream>
#include <vector>
#include <iostream>

namespace rte {

inline std::vector<std::string> split(std::string const str__)
{
    std::stringstream iss(str__);
    std::vector<std::string> result;

    while (iss.good()) {
        std::string s;
        std::getline(iss, s, '\n');
        result.push_back(s);
    }
    return result;
}

inline void throw_impl(const char* func__, const char* file__, int line__, std::string const& msg,
        std::string const& pmsg = "")
{
    auto split_msg = ::rte::split(msg);
    std::stringstream s;

    s << pmsg << std::endl << "[" << func__ << "] " << file__ << ":" << line__ << std::endl;
    for (auto e: split_msg) {
        s << "[" << func__ << "] " << e << std::endl;
    }
    throw std::runtime_error(s.str());
}

inline void throw_impl(const char* func__, const char* file__, int line__, std::stringstream const& msg,
        std::string const& pmsg = "")
{
    throw_impl(func__, file__, line__, msg.str(), pmsg);
}

#define RTE_THROW(...) \
{\
    ::rte::throw_impl(__func__, __FILE__, __LINE__, __VA_ARGS__);\
}

}

#endif
