#include "k_point.h"

namespace sirius {

void K_point::build_gklo_basis_descriptors()
{
    gklo_basis_descriptors_.clear();

    gklo_basis_descriptor gklo;

    int id = 0;

    /* G+k basis functions */
    for (int igk = 0; igk < num_gkvec(); igk++)
    {
        gklo.id         = id++;
        gklo.igk        = igk;
        gklo.gvec       = gkvec_[igk];
        gklo.gkvec      = gkvec<fractional>(igk);
        gklo.gkvec_cart = gkvec<cartesian>(igk);
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
            Atom* atom = unit_cell_.atom(ia);
            Atom_type* type = atom->type();
        
            int lo_index_offset = type->mt_aw_basis_size();
            
            for (int j = 0; j < type->mt_lo_basis_size(); j++) 
            {
                int l           = type->indexb(lo_index_offset + j).l;
                int lm          = type->indexb(lo_index_offset + j).lm;
                int order       = type->indexb(lo_index_offset + j).order;
                int idxrf       = type->indexb(lo_index_offset + j).idxrf;
                gklo.id         = id++;
                gklo.igk        = -1;
                gklo.gvec       = vector3d<int>(0, 0, 0);
                gklo.gkvec      = vector3d<double>(0, 0, 0);
                gklo.gkvec_cart = vector3d<double>(0, 0, 0);
                gklo.ig         = -1;
                gklo.ia         = ia;
                gklo.l          = l;
                gklo.lm         = lm;
                gklo.order      = order;
                gklo.idxrf      = idxrf;

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
