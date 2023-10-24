#include "ostream_tools.hpp"

namespace sirius {

null_stream_t& null_stream()
{
    static null_stream_t null_stream__;
    return null_stream__;
}

}
