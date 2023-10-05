#ifndef __PACKED_INDEX_HPP__
#define __PACKED_INDEX_HPP__

namespace sirius {

/// Pack two indices into one for symmetric matrices.
inline int packed_index(int i__, int j__)
{
    /* suppose we have a symmetric matrix: M_{ij} = M_{ji}
           j
       +-------+
       | + + + |
      i|   + + |   -> idx = j * (j + 1) / 2 + i  for  i <= j
       |     + |
       +-------+

       i, j are row and column indices 
    */

    if (i__ > j__) {
        std::swap(i__, j__);
    }
    return j__ * (j__ + 1) / 2 + i__;
}

}

#endif


