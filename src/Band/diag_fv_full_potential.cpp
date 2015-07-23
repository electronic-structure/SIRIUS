#include "band.h"

namespace sirius {

void Band::diag_fv_full_potential(K_point* kp, Periodic_function<double>* effective_potential)
{
    log_function_enter(__func__);
    Timer t("sirius::Band::diag_fv_full_potential", kp->comm());

    if (kp->num_ranks() > 1 && !gen_evp_solver()->parallel())
        error_local(__FILE__, __LINE__, "eigen-value solver is not parallel");

    dmatrix<double_complex> h(nullptr, kp->gklo_basis_size(), kp->gklo_basis_size(), kp->blacs_grid(), parameters_.cyclic_block_size(), parameters_.cyclic_block_size());

    dmatrix<double_complex> o(nullptr, kp->gklo_basis_size(), kp->gklo_basis_size(), kp->blacs_grid(), parameters_.cyclic_block_size(), parameters_.cyclic_block_size());
    
    h.allocate(alloc_mode);
    o.allocate(alloc_mode);
    
    /* setup Hamiltonian and overlap */
    switch (parameters_.processing_unit())
    {
        case CPU:
        {
            set_fv_h_o<CPU, full_potential_lapwlo>(kp, effective_potential, h, o);
            break;
        }
        #ifdef __GPU
        case GPU:
        {
            set_fv_h_o<GPU, full_potential_lapwlo>(kp, effective_potential, h, o);
            break;
        }
        #endif
        default:
        {
            TERMINATE("wrong processing unit");
        }
    }

    // TODO: move debug code to a separate function
    if (debug_level > 0 && !gen_evp_solver()->parallel())
    {
        Utils::check_hermitian("h", h.panel());
        Utils::check_hermitian("o", o.panel());
    }

    #ifdef __PRINT_OBJECT_CHECKSUM
    auto z1 = h.panel().checksum();
    auto z2 = o.panel().checksum();
    DUMP("checksum(h): %18.10f %18.10f", std::real(z1), std::imag(z1));
    DUMP("checksum(o): %18.10f %18.10f", std::real(z2), std::imag(z2));
    #endif

    #ifdef __PRINT_OBJECT_HASH
    DUMP("hash(h): %16llX", h.panel().hash());
    DUMP("hash(o): %16llX", o.panel().hash());
    #endif

    assert(kp->gklo_basis_size() > parameters_.num_fv_states());
    
    if (fix_apwlo_linear_dependence)
    {
        //solve_fv_evp_2stage(kp, h, o);
    }
    else
    {
        std::vector<double> eval(parameters_.num_fv_states());
    
        Timer t("sirius::Band::diag_fv_full_potential|genevp");
    
        if (gen_evp_solver()->solve(kp->gklo_basis_size(), kp->gklo_basis_size_row(), kp->gklo_basis_size_col(),
                                    parameters_.num_fv_states(), h.at<CPU>(), h.ld(), o.at<CPU>(), o.ld(), 
                                    &eval[0], kp->fv_eigen_vectors().at<CPU>(), kp->fv_eigen_vectors().ld()))
        {
            TERMINATE("error in generalized eigen-value problem");
        }
        kp->set_fv_eigen_values(&eval[0]);
    }

    h.deallocate();
    o.deallocate();
    
    log_function_exit(__func__);
}

};
