#include "dlaf.hpp"

#if defined(SIRIUS_DLAF)

#include <dlaf_c/grid.h>
#include <dlaf_c/init.h>

namespace sirius::la::dlaf {

void
init()
{
    const char* pika_argv[] = {"sirius"};
    const char* dlaf_argv[] = {"sirius"};
    // If DLAF is already initialized this call has no effect
    dlaf_initialize(1, pika_argv, 1, dlaf_argv);
}

void
finalize()
{
    dlaf_free_all_grids();
    dlaf_finalize();
}

} // namespace sirius::la::dlaf

#endif // SIRIUS_DLAF
