#ifndef __SERIALIZE_GVEC__
#define __SERIALIZE_GVEC__

#include "serializer.hpp"
#include "gvec.hpp"

namespace sddk {

inline void serialize(serializer& s__, Gvec& gv__)
{
    serialize(s__, gv__.vk_);
    serialize(s__, gv__.Gmax_);
    serialize(s__, gv__.lattice_vectors_);
    serialize(s__, gv__.reduce_gvec_);
    serialize(s__, gv__.bare_gvec_);
    serialize(s__, gv__.num_gvec_);
    serialize(s__, gv__.num_gvec_shells_);
    serialize(s__, gv__.gvec_full_index_);
    serialize(s__, gv__.gvec_shell_);
    serialize(s__, gv__.gvec_shell_len_);
    serialize(s__, gv__.gvec_index_by_xy_);
    serialize(s__, gv__.z_columns_);
    serialize(s__, gv__.gvec_distr_);
    serialize(s__, gv__.zcol_distr_);
    serialize(s__, gv__.gvec_base_mapping_);
}

inline void deserialize(serializer& s__, Gvec& gv__)
{
    deserialize(s__, gv__.vk_);
    deserialize(s__, gv__.Gmax_);
    deserialize(s__, gv__.lattice_vectors_);
    deserialize(s__, gv__.reduce_gvec_);
    deserialize(s__, gv__.bare_gvec_);
    deserialize(s__, gv__.num_gvec_);
    deserialize(s__, gv__.num_gvec_shells_);
    deserialize(s__, gv__.gvec_full_index_);
    deserialize(s__, gv__.gvec_shell_);
    deserialize(s__, gv__.gvec_shell_len_);
    deserialize(s__, gv__.gvec_index_by_xy_);
    deserialize(s__, gv__.z_columns_);
    deserialize(s__, gv__.gvec_distr_);
    deserialize(s__, gv__.zcol_distr_);
    deserialize(s__, gv__.gvec_base_mapping_);
}

}

#endif
