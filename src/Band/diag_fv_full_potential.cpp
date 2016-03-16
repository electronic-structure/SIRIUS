#include "band.h"

namespace sirius {

void Band::diag_fv_full_potential(K_point* kp, Periodic_function<double>* effective_potential) const
{
    PROFILE_WITH_TIMER("sirius::Band::diag_fv_full_potential");

    if (kp->num_ranks() > 1 && !gen_evp_solver()->parallel())
        TERMINATE("eigen-value solver is not parallel");

    int ngklo = kp->gklo_basis_size();
    int bs = ctx_.cyclic_block_size();
    dmatrix<double_complex> h(nullptr, ngklo, ngklo, ctx_.blacs_grid(), bs, bs);
    dmatrix<double_complex> o(nullptr, ngklo, ngklo, ctx_.blacs_grid(), bs, bs);
    
    h.allocate(alloc_mode);
    o.allocate(alloc_mode);
    
    /* setup Hamiltonian and overlap */
    switch (ctx_.processing_unit())
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
    #if (__VERIFICATION > 0)
    if (!gen_evp_solver()->parallel())
    {
        Utils::check_hermitian("h", h.panel());
        Utils::check_hermitian("o", o.panel());
    }
    #endif

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

    assert(kp->gklo_basis_size() > ctx_.num_fv_states());
    
    if (fix_apwlo_linear_dependence)
    {
        //solve_fv_evp_2stage(kp, h, o);
    }
    else
    {
        std::vector<double> eval(ctx_.num_fv_states());
    
        runtime::Timer t("sirius::Band::diag_fv_full_potential|genevp");
    
        if (gen_evp_solver()->solve(kp->gklo_basis_size(), ctx_.num_fv_states(), h.at<CPU>(), h.ld(), o.at<CPU>(), o.ld(), 
                                    &eval[0], kp->fv_eigen_vectors().coeffs().at<CPU>(), kp->fv_eigen_vectors().coeffs().ld(),
                                    kp->gklo_basis_size_row(), kp->gklo_basis_size_col()))

        {
            TERMINATE("error in generalized eigen-value problem");
        }
        kp->set_fv_eigen_values(&eval[0]);
    }

    h.deallocate();
    o.deallocate();
}

};
