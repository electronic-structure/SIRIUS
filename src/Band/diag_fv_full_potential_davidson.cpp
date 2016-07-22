#include "band.h"

namespace sirius {

void Band::diag_fv_full_potential_davidson(K_point* kp, Periodic_function<double>* effective_potential) const
{
    PROFILE_WITH_TIMER("sirius::Band::diag_fv_full_potential_davidson");

    if (kp->num_ranks() > 1 && !gen_evp_solver()->parallel())
        TERMINATE("eigen-value solver is not parallel");

    int ngklo = kp->gklo_basis_size();
    int bs = ctx_.cyclic_block_size();
    dmatrix<double_complex> h(nullptr, ngklo, ngklo, ctx_.blacs_grid(), bs, bs);
    dmatrix<double_complex> o(nullptr, ngklo, ngklo, ctx_.blacs_grid(), bs, bs);
    
    printf("gklo_basis_size = %10i\n", ngklo);
    printf("cyclic_block_size = %10i\n", bs);

    int nsingular = 0;

    printf("number of atom types = %10i\n", unit_cell_.num_atom_types());
    printf("number of atoms = %10i\n", unit_cell_.num_atoms());
  
    double gk_cutoff = ctx_.gk_cutoff();
    double gk_cutoff0 = ctx_.aw_cutoff() / unit_cell_.max_mt_radius();
    double gk_cutoff1 = ctx_.gk_cutoff() / unit_cell_.max_mt_radius();
    printf("gkmax = %16.8f\n", gk_cutoff);
    printf("gkmax0 = %16.8f\n", gk_cutoff0);
    printf("gkmax1 = %16.8f\n", gk_cutoff1);

    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++)
    { 
        auto& atom_type = unit_cell_.atom_type(iat);
        printf("iat = %10i, RMT = %16.8f\n", iat, atom_type.mt_radius());
        nsingular += atom_type.num_atoms() * int( 0.0028 * pow(atom_type.mt_radius() * gk_cutoff0, 4));
        printf("iat = %10i, nsigular= %10i\n", iat, nsingular);
    }
    

    if ( nsingular == 0 )
    {   nsingular = 1 ; } 

    int nblock = 4;

    int num_gkvec = kp->num_gkvec();
    printf("num_gkvec = %10i\n", kp->num_gkvec());


    std::vector<double_complex> sdiag(num_gkvec);
    std::fill (sdiag.begin(), sdiag.end(), ctx_.step_function().theta_pw(0));

    for (int i = 0; i < num_gkvec; i++)
    {
        printf("i = %10i\n", i);
//        std::cout << "sdiag = " <<  sdiag[i] << '\n';

        printf("sdiag: %18.10f %18.10f\n", std::real(sdiag[i]), std::imag(sdiag[i]));
    }   

    std::cout << "ctx_.step_function().theta_pw(0) = " << ctx_.step_function().theta_pw(0) << '\n';

    for (int igk = 0; igk < kp->num_gkvec(); igk++)
    {   for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++)
        {   
        }      
    }



    h.allocate(alloc_mode);
    o.allocate(alloc_mode);
    
    /* setup Hamiltonian and overlap */
    switch (ctx_.processing_unit())
    {
        case CPU:
        {

        /* initialize the start Hamiltonian and oberlap matrix in a subspace */

////            initialize_fv_h_o_davidson<CPU, full_potential_lapwlo>(kp, effective_potential, h, o);
            set_fv_h_o<CPU, full_potential_lapwlo>(kp, effective_potential, h, o);
            break;
        }
//        #ifdef __GPU
//        case GPU:
//        {
//            set_fv_h_o<GPU, full_potential_lapwlo>(kp, effective_potential, h, o);
//            break;
//        }
//        #endif
        default:
        {
            TERMINATE("wrong processing unit");
        }
    }


    /* get diagonal elements for preconditioning */
    // auto h_diag = get_h_diag()
    // auto o_diag = get_o_diag()   


//    // TODO: move debug code to a separate function
//    #if (__VERIFICATION > 0)
//    if (!gen_evp_solver()->parallel())
//    {
//        Utils::check_hermitian("h", h.panel());
//        Utils::check_hermitian("o", o.panel());
//    }
//    #endif
//
//    #ifdef __PRINT_OBJECT_CHECKSUM
//    auto z1 = h.panel().checksum();
//    auto z2 = o.panel().checksum();
//    DUMP("checksum(h): %18.10f %18.10f", std::real(z1), std::imag(z1));
//    DUMP("checksum(o): %18.10f %18.10f", std::real(z2), std::imag(z2));
//    #endif
//
//    #ifdef __PRINT_OBJECT_HASH
//    DUMP("hash(h): %16llX", h.panel().hash());
//    DUMP("hash(o): %16llX", o.panel().hash());
//    #endif

    assert(kp->gklo_basis_size() > ctx_.num_fv_states());
    
    printf("gklo_basis_size = %10i\n", kp->gklo_basis_size());
    printf("num_fv_states = %10i\n", ctx_.num_fv_states());
    printf("fix_apwlo_linear_dependence = %10i\n", fix_apwlo_linear_dependence);

    if (fix_apwlo_linear_dependence)
    {
        //solve_fv_evp_2stage(kp, h, o);
    }
    else
    {
        std::vector<double> eval(ctx_.num_fv_states());
    
        runtime::Timer t("sirius::Band::diag_fv_full_potential_davidson|genevp");
    

        /* Davidson loop here*/

        /* trial basis */

        /* number of newly added basis */

        /* start iterative */

             /* apply Hamiltonian and overlap with new basis */

             /* setup eigen-value problem */

             /* increase size of subspace */

             /* solve generalized eigen-value problem with new larger size */

             /* check convergecy of occupied bands */

             /* get new residuals, hpsi and opsi */

             /* check if we run out of variational space */

             /* check convergecy of eigen-vectors */

             

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
