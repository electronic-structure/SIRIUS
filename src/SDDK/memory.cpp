#include "memory.hpp"

namespace sddk {

/// Return a memory pool.
/** A memory pool is created when this function called for the first time. */
sddk::memory_pool&
get_memory_pool(sddk::memory_t M__)
{
    static std::map<sddk::memory_t, sddk::memory_pool> memory_pool_;
    if (memory_pool_.count(M__) == 0) {
        memory_pool_.emplace(M__, sddk::memory_pool(M__));
    }
    return memory_pool_.at(M__);
}

} // namespace sddk
