#include "density.h"

namespace sirius {

void Density::generate(K_set& ks__)
{
    PROFILE_WITH_TIMER("sirius::Density::generate");

    //if (ctx_.processing_unit() == GPU &&
    //    ctx_.mpi_grid().communicator(1 << _mpi_dim_k_row_ | 1 << _mpi_dim_k_col_).size() == 1 &&
    //    !ctx_.full_potential()) {
    //    generate_valence_new(ks__);
    //} else {
    //    generate_valence(ks__);
    //}

    generate_valence_new(ks__);

    if (ctx_.full_potential()) {
        /* find the core states */
        generate_core_charge_density();
        /* add core contribution */
        for (int ialoc = 0; ialoc < (int)unit_cell_.spl_num_atoms().local_size(); ialoc++) {
            int ia = unit_cell_.spl_num_atoms(ialoc);
            for (int ir = 0; ir < unit_cell_.atom(ia).num_mt_points(); ir++) {
                rho_->f_mt<index_domain_t::local>(0, ir, ialoc) += unit_cell_.atom(ia).symmetry_class().core_charge_density(ir) / y00;
            }
        }
        /* synchronize muffin-tin part */
        rho_->sync_mt();
        for (int j = 0; j < ctx_.num_mag_dims(); j++) {
            magnetization_[j]->sync_mt();
        }
    }
    
    double nel{0};
    if (ctx_.full_potential()) {
        std::vector<double> nel_mt;
        double nel_it;
        nel = rho_->integrate(nel_mt, nel_it);
    } else {
        nel = rho_->f_pw(0).real() * unit_cell_.omega();
    }

    if (std::abs(nel - unit_cell_.num_electrons()) > 1e-5) {
        std::stringstream s;
        s << "wrong charge density after k-point summation" << std::endl
          << "obtained value : " << nel << std::endl 
          << "target value : " << unit_cell_.num_electrons() << std::endl
          << "difference : " << fabs(nel - unit_cell_.num_electrons()) << std::endl;
        if (ctx_.full_potential()) {
            s << "total core leakage : " << core_leakage();
            for (int ic = 0; ic < unit_cell_.num_atom_symmetry_classes(); ic++) 
                s << std::endl << "  atom class : " << ic << ", core leakage : " << core_leakage(ic);
        }
        WARNING(s);
    }

    #ifdef __PRINT_OBJECT_HASH
    DUMP("hash(rhomt): %16llX", rho_->f_mt().hash());
    DUMP("hash(rhoit): %16llX", rho_->f_it().hash());
    #endif

    //if (debug_level > 1) check_density_continuity_at_mt();
}

};
