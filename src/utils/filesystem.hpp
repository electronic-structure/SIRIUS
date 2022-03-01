#ifndef __FILESYSTEM_HPP__
#define __FILESYSTEM_HPP__

#if defined(SIRIUS_BOOST_FILESYSTEM)

#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

#elif defined(SIRIUS_STD_FILESYSTEM_EXPERIMENTAL)

#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;

#else

#include <filesystem>

namespace fs = std::filesystem;

#endif // BOOST_FILESYSTEM__

#endif // __FILESYSTEM_HPP__