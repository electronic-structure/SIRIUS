/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file profiler.cpp
 *
 *  \brief A time-based profiler.
 */

#include "profiler.hpp"

namespace sirius {
::rt_graph::Timer global_rtgraph_timer;

#if defined(SIRIUS_TX)
acc::txprofiler::Timer global_tx_timer;
#endif
} // namespace sirius
