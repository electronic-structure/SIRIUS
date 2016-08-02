#include "density.h"

namespace sirius {

void Density::generate_density_matrix(K_set& ks__)
{
    PROFILE_WITH_TIMER("sirius::Density::generate_density_matrix");

    /* if we have ud and du spin blocks, don't compute one of them (du in this implementation)
       because density matrix is symmetric */
    int ndm = std::max(ctx_.num_mag_dims(), ctx_.num_spins());

    if (ctx_.esm_type() == electronic_structure_method_t::full_potential_lapwlo) {
        /* complex density matrix */
        mdarray<double_complex, 4> mt_complex_density_matrix(unit_cell_.max_mt_basis_size(), 
                                                             unit_cell_.max_mt_basis_size(),
                                                             ndm, unit_cell_.num_atoms());
        mt_complex_density_matrix.zero();

        /* add k-point contribution */
        for (int ikloc = 0; ikloc < ks__.spl_num_kpoints().local_size(); ikloc++) {
            int ik = ks__.spl_num_kpoints(ikloc);
            add_k_point_contribution_mt(ks__[ik], mt_complex_density_matrix);
        }

        for (int j = 0; j < ndm; j++) {
            for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
                int ialoc = unit_cell_.spl_num_atoms().local_index(ia);
                int rank = unit_cell_.spl_num_atoms().local_rank(ia);
                double_complex* dest_ptr = (ctx_.comm().rank() == rank) ? &density_matrix_(0, 0, j, ialoc) : nullptr;
                ctx_.comm().reduce(&mt_complex_density_matrix(0, 0, j, ia), dest_ptr,
                                   unit_cell_.max_mt_basis_size() * unit_cell_.max_mt_basis_size(), rank);
            }
        }
    }

    if (ctx_.esm_type() == ultrasoft_pseudopotential || ctx_.esm_type() == paw_pseudopotential) {
        density_matrix_.zero();
        
        /* add k-point contribution */
        for (int ikloc = 0; ikloc < ks__.spl_num_kpoints().local_size(); ikloc++) {
            int ik = ks__.spl_num_kpoints(ikloc);
            if (ctx_.gamma_point()) {
                add_k_point_contribution<double>(ks__[ik], density_matrix_);
            } else {
                add_k_point_contribution<double_complex>(ks__[ik], density_matrix_);
            }
        }

        ctx_.comm().allreduce(density_matrix_.at<CPU>(), static_cast<int>(density_matrix_.size()));
    }

    #ifdef __PRINT_OBJECT_CHECKSUM
    {
        auto cs = density_matrix_.checksum();
        DUMP("checksum(density_matrix): %20.14f %20.14f", cs.real(), cs.imag());
    }
    #endif
}

};
