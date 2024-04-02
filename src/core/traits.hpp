/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file traits.hpp
 *
 *  \brief Helper functions for type traits.
 */

#ifndef __TRAITS_HPP__
#define __TRAITS_HPP__

namespace sirius {

template <class X>
struct identity
{
    typedef X type;
};

template <class X>
using identity_t = typename identity<X>::type;

} // namespace sirius

#endif /* __TRAITS_HPP__ */
