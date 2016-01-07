#include "k_point.h"

namespace sirius {

void K_point::build_gklo_basis_descriptors()
{
    PROFILE();

    gklo_basis_descriptors_.clear();

    gklo_basis_descriptor gklo;

    /* G+k basis functions */
    for (int igk = 0; igk < num_gkvec(); igk++)
    {
        gklo.gvec       = gkvec_[igk];
        gklo.gkvec      = gkvec_.gvec_shifted(igk);
        gklo.ig         = igk;
        gklo.ia         = -1;
        gklo.l          = -1;
        gklo.lm         = -1;
        gklo.order      = -1;
        gklo.idxrf      = -1;

        gklo_basis_descriptors_.push_back(gklo);
    }

    if (parameters_.full_potential())
    {
        /* local orbital basis functions */
        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
        {
            auto& atom = unit_cell_.atom(ia);
            auto& type = atom.type();
        
            int lo_index_offset = type.mt_aw_basis_size();
            
            for (int j = 0; j < type.mt_lo_basis_size(); j++) 
            {
                int l           = type.indexb(lo_index_offset + j).l;
                int lm          = type.indexb(lo_index_offset + j).lm;
                int order       = type.indexb(lo_index_offset + j).order;
                int idxrf       = type.indexb(lo_index_offset + j).idxrf;
                gklo.gvec       = vector3d<int>(0, 0, 0);
                gklo.gkvec      = vector3d<double>(0, 0, 0);
                gklo.ig         = -1;
                gklo.ia         = static_cast<uint16_t>(ia);
                gklo.l          = static_cast<uint8_t>(l);
                gklo.lm         = static_cast<uint16_t>(lm);
                gklo.order      = static_cast<uint8_t>(order);
                gklo.idxrf      = static_cast<uint8_t>(idxrf);

                gklo_basis_descriptors_.push_back(gklo);
            }
        }
    
        /* ckeck if we count basis functions correctly */
        if ((int)gklo_basis_descriptors_.size() != (num_gkvec() + unit_cell_.mt_lo_basis_size()))
        {
            std::stringstream s;
            s << "(L)APW+lo basis descriptors array has a wrong size" << std::endl
              << "size of apwlo_basis_descriptors_ : " << gklo_basis_descriptors_.size() << std::endl
              << "num_gkvec : " << num_gkvec() << std::endl 
              << "mt_lo_basis_size : " << unit_cell_.mt_lo_basis_size();
            error_local(__FILE__, __LINE__, s);
        }
    }
}

};
