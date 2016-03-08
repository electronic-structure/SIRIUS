#include "band.h"

namespace sirius {

/** \param [in] phi Input wave-functions [storage: CPU && GPU].
 *  \param [in] hphi Hamiltonian, applied to wave-functions [storage: CPU || GPU].
 *  \param [in] ophi Overlap operator, applied to wave-functions [storage: CPU || GPU].
 */
template <>
void Band::set_h_o<double_complex>(K_point* kp__,
                                   int N__,
                                   int n__,
                                   Wave_functions<false>& phi__,
                                   Wave_functions<false>& hphi__,
                                   Wave_functions<false>& ophi__,
                                   matrix<double_complex>& h__,
                                   matrix<double_complex>& o__,
                                   matrix<double_complex>& h_old__,
                                   matrix<double_complex>& o_old__)
{
    PROFILE_WITH_TIMER("sirius::Band::set_h_o");
    
    assert(n__ != 0);

    /* copy old Hamiltonian and overlap */
    for (int i = 0; i < N__; i++)
    {
        std::memcpy(&h__(0, i), &h_old__(0, i), N__ * sizeof(double_complex));
        std::memcpy(&o__(0, i), &o_old__(0, i), N__ * sizeof(double_complex));
    }

    /* <{phi,res}|H|res> */
    phi__.inner<double_complex>(0, N__ + n__, hphi__, N__, n__, h__, 0, N__);
    /* <{phi,res}|O|res> */
    phi__.inner<double_complex>(0, N__ + n__, ophi__, N__, n__, o__, 0, N__);

    #ifdef __PRINT_OBJECT_CHECKSUM
    double_complex cs1(0, 0);
    double_complex cs2(0, 0);
    for (int i = 0; i < N__ + n__; i++)
    {
        for (int j = 0; j <= i; j++) 
        {
            cs1 += h__(j, i);
            cs2 += o__(j, i);
        }
    }
    DUMP("checksum(h): %18.10f %18.10f", cs1.real(), cs1.imag());
    DUMP("checksum(o): %18.10f %18.10f", cs2.real(), cs2.imag());
    #endif

    for (int i = 0; i < N__ + n__; i++)
    {
        if (h__(i, i).imag() > 1e-12)
        {
            std::stringstream s;
            s << "wrong diagonal of H: " << h__(i, i);
            TERMINATE(s);
        }
        if (o__(i, i).imag() > 1e-12)
        {
            std::stringstream s;
            s << "wrong diagonal of O: " << o__(i, i);
            TERMINATE(s);
        }
        h__(i, i) = h__(i, i).real();
        o__(i, i) = o__(i, i).real();
    }

    #if (__VERIFICATION > 0)
    /* check n__ * n__ block */
    for (int i = N__; i < N__ + n__; i++)
    {
        for (int j = N__; j < N__ + n__; j++)
        {
            if (std::abs(h__(i, j) - std::conj(h__(j, i))) > 1e-10 ||
                std::abs(o__(i, j) - std::conj(o__(j, i))) > 1e-10)
            {
                double_complex z1, z2;
                z1 = h__(i, j);
                z2 = h__(j, i);

                std::cout << "h(" << i << "," << j << ")=" << z1 << " "
                          << "h(" << j << "," << i << ")=" << z2 << ", diff=" << std::abs(z1 - std::conj(z2)) << std::endl;
                
                z1 = o__(i, j);
                z2 = o__(j, i);

                std::cout << "o(" << i << "," << j << ")=" << z1 << " "
                          << "o(" << j << "," << i << ")=" << z2 << ", diff=" << std::abs(z1 - std::conj(z2)) << std::endl;
                
            }
        }
    }
    #endif
    
    int i0 = N__;
    if (gen_evp_solver_->type() == ev_magma || gen_evp_solver_->type() == ev_elpa1 || gen_evp_solver_->type() == ev_elpa2)
    {
        /* restore the lower part */
        #pragma omp parallel for
        for (int i = 0; i < N__; i++)
        {
            for (int j = N__; j < N__ + n__; j++)
            {
                h__(j, i) = std::conj(h__(i, j));
                o__(j, i) = std::conj(o__(i, j));
            }
        }
        i0 = 0;
    }

    /* save Hamiltonian and overlap */
    #pragma omp parallel for
    for (int i = i0; i < N__ + n__; i++)
    {
        std::memcpy(&h_old__(0, i), &h__(0, i), (N__ + n__) * sizeof(double_complex));
        std::memcpy(&o_old__(0, i), &o__(0, i), (N__ + n__) * sizeof(double_complex));
    }
}

template <>
void Band::set_h_o<double>(K_point* kp__,
                           int N__,
                           int n__,
                           Wave_functions<false>& phi__,
                           Wave_functions<false>& hphi__,
                           Wave_functions<false>& ophi__,
                           matrix<double>& h__,
                           matrix<double>& o__,
                           matrix<double>& h_old__,
                           matrix<double>& o_old__)
{
    PROFILE_WITH_TIMER("sirius::Band::set_h_o");
    
    assert(n__ != 0);

    /* copy old Hamiltonian and overlap */
    for (int i = 0; i < N__; i++)
    {
        std::memcpy(&h__(0, i), &h_old__(0, i), N__ * sizeof(double));
        //std::memcpy(&o__(0, i), &o_old__(0, i), N__ * sizeof(double));
    }


    auto inner = [kp__](double_complex* f, double_complex* g)
    {
        double prod = 0;
        for (int igk = 0; igk < kp__->num_gkvec_loc(); igk++)
                prod += 2.0 * (f[igk].real() * g[igk].real() + f[igk].imag() * g[igk].imag());
        prod -= f[0].real() * g[0].real();
        return prod;
    };

    //auto scale = [kp__](double_complex* f, double a)
    //{
    //    for (int igk = 0; igk < kp__->num_gkvec_loc(); igk++) f[igk] *= a;
    //};
    
    runtime::Timer t1("sirius::Band::set_h_o|ortho");
    double prod;
    for (int i = N__; i < N__ + n__; i++)
    {
        for (int j = 0; j < i; j++)
        {
            prod = inner(&phi__(0, j), &ophi__(0, i));
            
            #pragma omp parallel for
            for (int igk = 0; igk < kp__->num_gkvec_loc(); igk++)
            {
                phi__(igk, i) -= phi__(igk, j) * prod;
                ophi__(igk, i) -= ophi__(igk, j) * prod;
                hphi__(igk, i) -= hphi__(igk, j) * prod;
            }
        }
        prod = inner(&phi__(0, i), &ophi__(0, i));
        prod = 1.0 / std::sqrt(prod);
        #pragma omp parallel for
        for (int igk = 0; igk < kp__->num_gkvec_loc(); igk++)
        {
            phi__(igk, i) *= prod;
            hphi__(igk, i) *= prod;
            ophi__(igk, i) *= prod;
        }
    }
    t1.stop();

    /* <{phi,res}|H|res> */
    phi__.inner<double>(0, N__ + n__, hphi__, N__, n__, h__, 0, N__);
    ///* <{phi,res}|O|res> */
    //phi__.inner<double>(0, N__ + n__, ophi__, N__, n__, o__, 0, N__);

    //for (int i = 0; i < N__ + n__; i++)
    //{
    //    for (int j = i; j < N__ + n__; j++)
    //    {
    //        double a = o__(j, i);
    //        if (i == j) a -= 1;

    //        if (std::abs(a) > 1e-10)
    //        {
    //            printf("wrong overlap");
    //            TERMINATE("wrong overlap");
    //        }
    //    }
    //}

    //#ifdef __PRINT_OBJECT_CHECKSUM
    //double_complex cs1(0, 0);
    //double_complex cs2(0, 0);
    //for (int i = 0; i < N__ + n__; i++)
    //{
    //    for (int j = 0; j <= i; j++) 
    //    {
    //        cs1 += h__(j, i);
    //        cs2 += o__(j, i);
    //    }
    //}
    //DUMP("checksum(h): %18.10f %18.10f", cs1.real(), cs1.imag());
    //DUMP("checksum(o): %18.10f %18.10f", cs2.real(), cs2.imag());
    //#endif

    #if (__VERIFICATION > 0)
    /* check n__ * n__ block */
    for (int i = N__; i < N__ + n__; i++)
    {
        for (int j = N__; j < N__ + n__; j++)
        {
            if (std::abs(h__(i, j) - h__(j, i)) > 1e-10 ||
                std::abs(o__(i, j) - o__(j, i)) > 1e-10)
            {
                double z1, z2;
                z1 = h__(i, j);
                z2 = h__(j, i);

                std::cout << "h(" << i << "," << j << ")=" << z1 << " "
                          << "h(" << j << "," << i << ")=" << z2 << ", diff=" << std::abs(z1 - z2) << std::endl;
                
                z1 = o__(i, j);
                z2 = o__(j, i);

                std::cout << "o(" << i << "," << j << ")=" << z1 << " "
                          << "o(" << j << "," << i << ")=" << z2 << ", diff=" << std::abs(z1 - z2) << std::endl;
                
            }
        }
    }
    #endif
    
    int i0 = N__;
    if (gen_evp_solver_->type() == ev_magma || gen_evp_solver_->type() == ev_elpa1 || gen_evp_solver_->type() == ev_elpa2)
    {
        /* restore the lower part */
        #pragma omp parallel for
        for (int i = 0; i < N__; i++)
        {
            for (int j = N__; j < N__ + n__; j++)
            {
                h__(j, i) = h__(i, j);
                //o__(j, i) = o__(i, j);
            }
        }
        i0 = 0;
    }

    /* save Hamiltonian and overlap */
    #pragma omp parallel for
    for (int i = i0; i < N__ + n__; i++)
    {
        std::memcpy(&h_old__(0, i), &h__(0, i), (N__ + n__) * sizeof(double));
        //std::memcpy(&o_old__(0, i), &o__(0, i), (N__ + n__) * sizeof(double));
    }


    //printf("checking <phi|phi> matrix\n");
    //mdarray<double, 2> ovlp_tmp(N__ + n__, N__ + n__);

    //phi__.inner<double>(0, N__ + n__, phi__, 0, N__ + n__, ovlp_tmp, 0, 0);

    //Eigenproblem_lapack evp;
    //mdarray<double, 2> evec_tmp(N__ + n__, N__ + n__);
    //std::vector<double> eval_tmp(N__ + n__);
    //std::vector<double> ovlp_diag(N__ + n__);
    //for (int i = 0; i < N__ + n__; i++) ovlp_diag[i] = ovlp_tmp(i, i);

    //evp.solve(N__ + n__, &ovlp_tmp(0, 0), N__ + n__, &eval_tmp[0], &evec_tmp(0, 0), N__ + n__);
    //if (kp__->comm().rank() == 0)
    //{
    //    for (int i = 0; i < std::min(N__ + n__, 10000000); i++)
    //        printf("eval_<phi|phi>[%i] = %18.10f, diag = %18.10f\n", i, eval_tmp[i], ovlp_diag[i]);
    //}
    


}


};
