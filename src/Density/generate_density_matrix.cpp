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

void Density::symmetrize_density_matrix()
{
    PROFILE_WITH_TIMER("sirius::Density::symmetrize_density_matrix");

    auto& sym = unit_cell_.symmetry();

    int ndm = std::max(ctx_.num_mag_dims(), ctx_.num_spins());

    mdarray<double_complex, 4> dm(unit_cell_.max_mt_basis_size(), unit_cell_.max_mt_basis_size(), 
                                  ndm, unit_cell_.num_atoms());
    dm.zero();

    int lmax = unit_cell_.lmax();
    int lmmax = Utils::lmmax(lmax);

    mdarray<double, 2> rotm(lmmax, lmmax);

    double alpha = 1.0 / double(sym.num_mag_sym());

    for (int i = 0; i < sym.num_mag_sym(); i++) {
        int pr = sym.magnetic_group_symmetry(i).spg_op.proper;
        auto eang = sym.magnetic_group_symmetry(i).spg_op.euler_angles;
        int isym = sym.magnetic_group_symmetry(i).isym;
        SHT::rotation_matrix(lmax, eang, pr, rotm);

        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
            auto& atom_type = unit_cell_.atom(ia).type();
            int ja = sym.sym_table(ia, isym);

            for (int xi1 = 0; xi1 < unit_cell_.atom(ia).mt_basis_size(); xi1++) {
                int l1  = atom_type.indexb(xi1).l;
                int lm1 = atom_type.indexb(xi1).lm;
                int o1  = atom_type.indexb(xi1).order;

                for (int xi2 = 0; xi2 < unit_cell_.atom(ia).mt_basis_size(); xi2++) {
                    int l2  = atom_type.indexb(xi2).l;
                    int lm2 = atom_type.indexb(xi2).lm;
                    int o2  = atom_type.indexb(xi2).order;
                    
                    for (int j = 0; j < ndm; j++) {
                        for (int m3 = -l1; m3 <= l1; m3++) {
                            int lm3 = Utils::lm_by_l_m(l1, m3);
                            int xi3 = atom_type.indexb().index_by_lm_order(lm3, o1);
                            for (int m4 = -l2; m4 <= l2; m4++) {
                                int lm4 = Utils::lm_by_l_m(l2, m4);
                                int xi4 = atom_type.indexb().index_by_lm_order(lm4, o2);
                                dm(xi1, xi2, j, ja) += density_matrix_(xi3, xi4, j, ia) * rotm(lm1, lm3) * rotm(lm2, lm4) * alpha;
                            }
                        }
                    }
                }
            }
        }
    }
    
    ctx_.comm().allreduce(dm.at<CPU>(), static_cast<int>(dm.size()));
    dm >> density_matrix_;
}

};
