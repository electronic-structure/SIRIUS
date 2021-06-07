/** \file linalg.hpp
 *
 *  \brief Define some common type used in SDDK::
 */

#ifndef SIRIUS_LINALG_BASE_H
#define SIRIUS_LINALG_BASE_H

using double_complex = std::complex<double>;

// define type traits for a single template implementation of both real and complex matrix
// general case for real matrix
template <typename T>
struct real_type {using type = T;};

// special case for complex matrix
template <typename T>
struct real_type<std::complex<T>> {using type = T;};


#endif // SIRIUS_LINALG_BASE_H
