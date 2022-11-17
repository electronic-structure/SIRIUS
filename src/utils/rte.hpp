#ifndef __RTE_HPP__
#define __RTE_HPP__

#include <stdexcept>
#include <sstream>
#include <ostream>
#include <vector>
#include <iostream>

#define FILE_AND_LINE std::string(__FILE__) + ":" + std::to_string(__LINE__)

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

#ifdef NDEBUG
#define RTE_ASSERT(condition__)
#else
#define RTE_ASSERT(condition__)                                  \
{                                                                \
    if (!(condition__)) {                                        \
        std::stringstream _s;                                    \
        _s << "Assertion (" <<  #condition__ << ") failed "      \
           << "at line " << __LINE__ << " of file " << __FILE__; \
        RTE_THROW(_s);                                           \
    }                                                            \
}
#endif

class rte_ostream : public std::ostringstream
{
  private:
    std::ostream& out_;
    std::string prefix_;
  public:
    rte_ostream(std::ostream& out__, std::string prefix__)
        : out_(out__)
        , prefix_(prefix__)
    {
    }
    ~rte_ostream()
    {
        auto strings = rte::split(this->str());
        for (size_t i = 0; i < strings.size(); i++) {
            if (!(i == strings.size() - 1 && strings[i].size() == 0)) {
                out_ << "[" << prefix_ << "] " << strings[i];
            }
            if (i != strings.size() - 1) {
                out_ << std::endl;
            }
        }
    }
};

#define RTE_OUT(_out) rte::rte_ostream(_out, std::string(__func__))

}

#endif
