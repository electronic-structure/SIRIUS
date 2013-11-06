#include <sirius.h>

using namespace sirius;

//== // N : subspace dimension
//== // n : required number of eigen pairs
//== // bs : block size (number of new basis functions)
//== // phi : basis functions
//== // hphi : H|phi>
//== void subspace_diag(Global& parameters, int N, int n, int bs, mdarray<complex16, 2>& phi, mdarray<complex16, 2>& hphi, 
//==                    mdarray<complex16, 2>& res, complex16 v0, mdarray<complex16, 2>& evec, std::vector<double>& eval)
//== {
//==     //== for (int i = 0; i < N; i++)
//==     //== {
//==     //==     for (int j = 0; j < N; j++)
//==     //==     {
//==     //==         complex16 z(0, 0);
//==     //==         for (int ig = 0; ig < parameters.num_gvec(); ig++)
//==     //==         {
//==     //==             z += conj(phi(ig, i)) * phi(ig, j);
//==     //==         }
//==     //==         if (i == j) z -= 1.0;
//==     //==         if (abs(z) > 1e-12) error_local(__FILE__, __LINE__, "basis is not orthogonal");
//==     //==     }
//==     //== }
//== 
//==     standard_evp* solver = new standard_evp_lapack();
//== 
//==     eval.resize(N);
//== 
//==     mdarray<complex16, 2> hmlt(N, N);
//==     blas<cpu>::gemm(2, 0, N, N, parameters.num_gvec(), &phi(0, 0), phi.ld(), &hphi(0, 0), hphi.ld(), &hmlt(0, 0), hmlt.ld());
//==     //== for (int i = 0; i < N; i++)
//==     //== {
//==     //==     for (int j = 0; j < N; j++)
//==     //==     {
//==     //==         for (int ig = 0; ig < parameters.num_gvec(); ig++) hmlt(i, j) += conj(phi(ig, i)) * hphi(ig, j);
//==     //==     }
//==     //== }
//== 
//==     solver->solve(N, hmlt.get_ptr(), hmlt.ld(), &eval[0], evec.get_ptr(), evec.ld());
//== 
//==     delete solver;
//== 
//==     printf("\n");
//==     printf("Lowest eigen-values : \n");
//==     for (int i = 0; i < std::min(N, 10); i++)
//==     {
//==         printf("i : %i,  eval : %16.10f\n", i, eval[i]);
//==     }
//== 
//==     // compute residuals
//==     res.zero();
//==     for (int j = 0; j < bs; j++)
//==     {
//==         int i = j; //n - bs + j;
//==         for (int mu = 0; mu < N; mu++)
//==         {
//==             for (int ig = 0; ig < parameters.num_gvec(); ig++)
//==             {
//==                 res(ig, j) += (evec(mu, i) * hphi(ig, mu) - eval[i] * evec(mu, i) * phi(ig, mu));
//==             }
//==         }
//==         double norm = 0.0;
//==         for (int ig = 0; ig < parameters.num_gvec(); ig++) norm += real(conj(res(ig, j)) * res(ig, j));
//==         for (int ig = 0; ig < parameters.num_gvec(); ig++) res(ig, j) /= sqrt(norm);
//==     }
//==     
//==     // additional basis vectors
//==     for (int j = 0; j < bs; j++)
//==     {
//==         int i = j; //n - bs + j;
//==         for (int ig = 0; ig < parameters.num_gvec(); ig++)
//==         {
//==             complex16 t = pow(parameters.gvec_len(ig), 2) / 2.0 + v0 - eval[i];
//==             if (abs(t) < 1e-12) error_local(__FILE__, __LINE__, "problematic division");
//==             res(ig, j) /= t;
//==         }
//==     }
//== }
//== 
//== void expand_subspace(Global& parameters, int N, int bs, mdarray<complex16, 2>& phi, mdarray<complex16, 2>& res)
//== {
//==     // overlap between new addisional basis vectors and old basis vectors
//==     mdarray<complex16, 2> ovlp(N, bs);
//==     ovlp.zero();
//==     for (int i = 0; i < N; i++)
//==     {
//==         for (int j = 0; j < bs; j++) 
//==         {
//==             for (int ig = 0; ig < parameters.num_gvec(); ig++) ovlp(i, j) += conj(phi(ig, i)) * res(ig, j);
//==         }
//==     }
//== 
//==     // project out the the old subspace
//==     for (int j = 0; j < bs; j++)
//==     {
//==         for (int i = 0; i < N; i++)
//==         {
//==             for (int ig = 0; ig < parameters.num_gvec(); ig++) res(ig, j) -= ovlp(i, j) * phi(ig, i);
//==         }
//==     }
//== 
//==     // orthogonalize
//==     for (int j = 0; j < bs; j++)
//==     {
//==         std::vector<complex16> v(parameters.num_gvec());
//==         memcpy(&v[0], &res(0, j), parameters.num_gvec() * sizeof(complex16));
//==         for (int j1 = 0; j1 < j; j1++)
//==         {
//==             complex16 z(0, 0);
//==             for (int ig = 0; ig < parameters.num_gvec(); ig++) z += conj(res(ig, j1)) * v[ig];
//==             for (int ig = 0; ig < parameters.num_gvec(); ig++) v[ig] -= z * res(ig, j1);
//==         }
//==         double norm = 0;
//==         for (int ig = 0; ig < parameters.num_gvec(); ig++) norm += real(conj(v[ig]) * v[ig]);
//==         //std::cout << "j=" << j <<" final norm=" << norm << std::endl;
//==         for (int ig = 0; ig < parameters.num_gvec(); ig++) res(ig, j) = v[ig] / sqrt(norm);
//==     }
//== 
//==     for (int j = 0; j < bs; j++)
//==     {
//==         for (int ig = 0; ig < parameters.num_gvec(); ig++) phi(ig, N + j) = res(ig, j);
//==     }
//== }

void apply_h(Global& parameters, K_point& kp, int n, std::vector<complex16>& v_r, complex16* phi__, complex16* hphi__)
{
    mdarray<complex16, 2> phi(phi__, kp.num_gkvec(), n);
    mdarray<complex16, 2> hphi(hphi__, kp.num_gkvec(), n);
    std::vector<complex16> phi_r(parameters.fft().size());

    for (int i = 0; i < n; i++)
    {
        parameters.fft().input(kp.num_gkvec(), kp.fft_index(), &phi(0, i));
        parameters.fft().transform(1);
        parameters.fft().output(&phi_r[0]);

        for (int ir = 0; ir < parameters.fft().size(); ir++) phi_r[ir] *= v_r[ir];

        parameters.fft().input(&phi_r[0]);
        parameters.fft().transform(-1);
        parameters.fft().output(kp.num_gkvec(), kp.fft_index(), &hphi(0, i));

        for (int ig = 0; ig < kp.num_gkvec(); ig++) hphi(ig, i) += phi(ig, i) * pow(kp.gkvec_cart(ig).length(), 2) / 2.0;
    }
}

//==int diag_davidson(Global& parameters, int niter, int bs, int n, std::vector<complex16>& v_pw, mdarray<complex16, 2>& phi, 
//==                  mdarray<complex16, 2>& evec)
//=={
//==    std::vector<complex16> v_r(parameters.fft().size());
//==    parameters.fft().input(parameters.num_gvec(), parameters.fft_index(), &v_pw[0]);
//==    parameters.fft().transform(1);
//==    parameters.fft().output(&v_r[0]);
//==
//==    for (int ir = 0; ir < parameters.fft().size(); ir++)
//==    {
//==        if (fabs(imag(v_r[ir])) > 1e-14) error_local(__FILE__, __LINE__, "potential is complex");
//==    }
//==
//==    mdarray<complex16, 2> hphi(parameters.num_gvec(), phi.size(1));
//==    mdarray<complex16, 2> res(parameters.num_gvec(), bs);
//==    
//==    int N = n;
//==
//==    apply_h(parameters, N, v_r, &phi(0, 0), &hphi(0, 0));
//==
//==    std::vector<double> eval1;
//==    std::vector<double> eval2;
//==
//==    for (int iter = 0; iter < niter; iter++)
//==    {
//==        eval2 = eval1;
//==        subspace_diag(parameters, N, n, bs, phi, hphi, res, v_pw[0], evec, eval1);
//==        expand_subspace(parameters, N, bs, phi, res);
//==        apply_h(parameters, bs, v_r, &res(0, 0), &hphi(0, N));
//==
//==        if (iter)
//==        {
//==            double diff = 0;
//==            for (int i = 0; i < n; i++) diff += fabs(eval1[i] - eval2[i]);
//==            std::cout << "Eigen-value error : " << diff << std::endl;
//==        }
//==        
//==        N += bs;
//==    }
//==    return N - bs;
//==}

void orthonormalize(mdarray<complex16, 2>& f)
{
    std::vector<complex16> v(f.size(0));
    for (int j = 0; j < f.size(1); j++)
    {
        memcpy(&v[0], &f(0, j), f.size(0) * sizeof(complex16));
        for (int j1 = 0; j1 < j; j1++)
        {
            complex16 z(0, 0);
            for (int ig = 0; ig < f.size(0); ig++) z += conj(f(ig, j1)) * v[ig];
            for (int ig = 0; ig < f.size(0); ig++) v[ig] -= z * f(ig, j1);
        }
        double norm = 0;
        for (int ig = 0; ig < f.size(0); ig++) norm += real(conj(v[ig]) * v[ig]);
        for (int ig = 0; ig < f.size(0); ig++) f(ig, j) = v[ig] / sqrt(norm);
    }

    //for (int i = 0; i < f.size(1); i++)
    //{
    //    for (int j = 0; j < f.size(1); j++)
    //    {
    //        complex16 z(0, 0);
    //        for (int ig = 0; ig < f.size(0); ig++)
    //        {
    //            z += conj(f(ig, i)) * f(ig, j);
    //        }
    //        if (i == j) z -= 1.0;
    //        if (abs(z) > 1e-12) error_local(__FILE__, __LINE__, "basis is not orthogonal");
    //    }
    //}
}

void check_orth(mdarray<complex16, 2>& f)
{
    for (int i = 0; i < f.size(1); i++)
    {
        for (int j = 0; j < f.size(1); j++)
        {
            complex16 z(0, 0);
            for (int ig = 0; ig < f.size(0); ig++)
            {
                z += conj(f(ig, i)) * f(ig, j);
            }
            if (i == j) z -= 1.0;
            if (abs(z) > 1e-10)
            {
                std::stringstream s;
                s << "basis is not orthonormal, error : " << abs(z);
                error_local(__FILE__, __LINE__, s);
            }
        }
    }
}

void apply_p(K_point& kp, mdarray<complex16, 2>& r)
{
    for (int i = 0; i < r.size(1); i++)
    {
        // compute kinetic energy of the vector
        double ekin = 0;
        for (int ig = 0; ig < kp.num_gkvec(); ig++) ekin += real(conj(r(ig, i)) * r(ig, i)) * pow(kp.gkvec_cart(ig).length(), 2) / 2.0;

        // apply the preconditioner
        for (int ig = 0; ig < kp.num_gkvec(); ig++)
        {
            double x = pow(kp.gkvec_cart(ig).length(), 2) / 1.5 / ekin;
            r(ig, i) = r(ig, i) * (27 + 18 * x + 12 * x * x + 8 * x * x * x) / (27 + 18 * x + 12 * x * x + 8 * x * x * x + 16 * x * x * x * x);
        }
    }
}

void diag_lobpcg(Global& parameters, K_point& kp, std::vector<complex16>& v_pw, int num_bands)
{
    std::vector<complex16> v_r(parameters.fft().size());
    parameters.fft().input(parameters.num_gvec(), parameters.fft_index(), &v_pw[0]);
    parameters.fft().transform(1);
    parameters.fft().output(&v_r[0]);

    for (int ir = 0; ir < parameters.fft().size(); ir++)
    {
        if (fabs(imag(v_r[ir])) > 1e-10) error_local(__FILE__, __LINE__, "potential is complex");
    }

    // initial basis functions
    mdarray<complex16, 2> phi(kp.num_gkvec(), num_bands);
    phi.zero();
    for (int i = 0; i < num_bands; i++) phi(i, i) = 1.0;

    mdarray<complex16, 2> hphi(kp.num_gkvec(), num_bands);

    apply_h(parameters, kp, num_bands, v_r, &phi(0, 0), &hphi(0, 0));

    mdarray<complex16, 2> ovlp(3 * num_bands, 3 * num_bands);

    mdarray<complex16, 2> hmlt(3 * num_bands, 3 * num_bands);
    blas<cpu>::gemm(2, 0, num_bands, num_bands, kp.num_gkvec(), &phi(0, 0), phi.ld(), &hphi(0, 0), hphi.ld(), &hmlt(0, 0), hmlt.ld());

    std::vector<double> eval(3 * num_bands);
    mdarray<complex16, 2> evec(3 * num_bands, 3 * num_bands);
    
    standard_evp* solver = new standard_evp_lapack();
    solver->solve(num_bands, hmlt.get_ptr(), hmlt.ld(), &eval[0], evec.get_ptr(), evec.ld());
    delete solver;

    mdarray<complex16, 2> zm(kp.num_gkvec(), num_bands);
    blas<cpu>::gemm(0, 0, kp.num_gkvec(), num_bands, num_bands, &phi(0, 0), phi.ld(), &evec(0, 0), evec.ld(), &zm(0, 0), zm.ld());
    zm >> phi;

    mdarray<complex16, 2> res(kp.num_gkvec(), num_bands);
    mdarray<complex16, 2> hres(kp.num_gkvec(), num_bands);

    mdarray<complex16, 2> grad(kp.num_gkvec(), num_bands);
    grad.zero();
    mdarray<complex16, 2> hgrad(kp.num_gkvec(), num_bands);
    
    generalized_evp* gevp = new generalized_evp_lapack(-1.0);

    for (int k = 1; k < 300; k++)
    {
        apply_h(parameters, kp, num_bands, v_r, &phi(0, 0), &hphi(0, 0));
        // res = H|phi> - E|phi>
        for (int i = 0; i < num_bands; i++)
        {
            for (int ig = 0; ig < kp.num_gkvec(); ig++) 
            {
                //complex16 t = pow(kp.gkvec_cart(ig).length(), 2) / 2.0 + v_pw[0] - eval[i];
                res(ig, i) = hphi(ig, i) - eval[i] * phi(ig, i);
                
                //if (abs(t) < 1e-12) error_local(__FILE__, __LINE__, "problematic division");
                //res(ig, i) /= t;
            }
        }

        std::cout << "Iteration : " << k << std::endl;
        for (int i = 0; i< num_bands; i++)
        {
            double r = 0;
            for (int ig = 0; ig < kp.num_gkvec(); ig++) r += real(conj(res(ig, i)) * res(ig, i));
            std::cout << "band : " << i << " residiual : " << r << " eigen-value : " << eval[i] << std::endl;
        }

        apply_p(kp, res);

        //orthonormalize(res);
        apply_h(parameters, kp, num_bands, v_r, &res(0, 0), &hres(0, 0));

        hmlt.zero();
        ovlp.zero();
        for (int i = 0; i < 3 * num_bands; i++) ovlp(i, i) = complex16(1, 0);

        // <phi|H|phi>
        blas<cpu>::gemm(2, 0, num_bands, num_bands, kp.num_gkvec(), &phi(0, 0), phi.ld(), &hphi(0, 0), hphi.ld(), 
                        &hmlt(0, 0), hmlt.ld());
        // <phi|H|res>
        blas<cpu>::gemm(2, 0, num_bands, num_bands, kp.num_gkvec(), &phi(0, 0), phi.ld(), &hres(0, 0), hres.ld(), 
                        &hmlt(0, num_bands), hmlt.ld());
        // <res|H|res>
        blas<cpu>::gemm(2, 0, num_bands, num_bands, kp.num_gkvec(), &res(0, 0), res.ld(), &hres(0, 0), hres.ld(), 
                        &hmlt(num_bands, num_bands), hmlt.ld());

        // <phi|res> 
        blas<cpu>::gemm(2, 0, num_bands, num_bands, kp.num_gkvec(), &phi(0, 0), phi.ld(), &res(0, 0), res.ld(), 
                        &ovlp(0, num_bands), ovlp.ld());
        
        // <res|res> 
        blas<cpu>::gemm(2, 0, num_bands, num_bands, kp.num_gkvec(), &res(0, 0), res.ld(), &res(0, 0), res.ld(), 
                        &ovlp(num_bands, num_bands), ovlp.ld());

        if (k == 1)
        {
            gevp->solve(2 * num_bands, num_bands, hmlt.get_ptr(), hmlt.ld(), ovlp.get_ptr(), ovlp.ld(), 
                        &eval[0], evec.get_ptr(), evec.ld());
        } 
        else
        {
            //orthonormalize(grad);
            apply_h(parameters, kp, num_bands, v_r, &grad(0, 0), &hgrad(0, 0));

            // <phi|H|grad>
            blas<cpu>::gemm(2, 0, num_bands, num_bands, kp.num_gkvec(), &phi(0, 0), phi.ld(), &hgrad(0, 0), hgrad.ld(), 
                            &hmlt(0, 2 * num_bands), hmlt.ld());
            // <res|H|grad>
            blas<cpu>::gemm(2, 0, num_bands, num_bands, kp.num_gkvec(), &res(0, 0), res.ld(), &hgrad(0, 0), hgrad.ld(), 
                            &hmlt(num_bands, 2 * num_bands), hmlt.ld());
            // <grad|H|grad>
            blas<cpu>::gemm(2, 0, num_bands, num_bands, kp.num_gkvec(), &grad(0, 0), grad.ld(), &hgrad(0, 0), hgrad.ld(), 
                            &hmlt(2 * num_bands, 2 * num_bands), hmlt.ld());
            
            // <phi|grad> 
            blas<cpu>::gemm(2, 0, num_bands, num_bands, kp.num_gkvec(), &phi(0, 0), phi.ld(), &grad(0, 0), grad.ld(), 
                            &ovlp(0, 2 * num_bands), ovlp.ld());
            // <res|grad> 
            blas<cpu>::gemm(2, 0, num_bands, num_bands, kp.num_gkvec(), &res(0, 0), res.ld(), &grad(0, 0), grad.ld(), 
                            &ovlp(num_bands, 2 * num_bands), ovlp.ld());
            // <grad|grad> 
            blas<cpu>::gemm(2, 0, num_bands, num_bands, kp.num_gkvec(), &grad(0, 0), grad.ld(), &grad(0, 0), grad.ld(), 
                            &ovlp(2 * num_bands, 2 * num_bands), ovlp.ld());
            
            gevp->solve(3 * num_bands, num_bands, hmlt.get_ptr(), hmlt.ld(), ovlp.get_ptr(), ovlp.ld(), 
                        &eval[0], evec.get_ptr(), evec.ld());
            
        }
        
        grad >> zm;
        // P^{k+1} = P^{k} * Z_{grad} + res^{k} * Z_{res}
        blas<cpu>::gemm(0, 0, kp.num_gkvec(), num_bands, num_bands, &res(0, 0), res.ld(), 
                        &evec(num_bands, 0), evec.ld(), &grad(0, 0), grad.ld());
        if (k > 1) 
        {
            blas<cpu>::gemm(0, 0, kp.num_gkvec(), num_bands, num_bands, complex16(1, 0), &zm(0, 0), zm.ld(), 
                            &evec(2 * num_bands, 0), evec.ld(), complex16(1, 0), &grad(0, 0), grad.ld());
        }

        // phi^{k+1} = phi^{k} * Z_{phi} + P^{k+1}
        phi >> zm;
        grad >> phi;
        blas<cpu>::gemm(0, 0, kp.num_gkvec(), num_bands, num_bands, complex16(1, 0), &zm(0, 0), zm.ld(), 
                        &evec(0, 0), evec.ld(), complex16(1, 0), &phi(0, 0), phi.ld());

        //check_orth(phi);    
    }
    

    delete gevp;

}

void test_lobpcg()
{
    Global parameters;

    double a0[] = {12.975 * 1.889726125, 0, 0};
    double a1[] = {0, 12.975 * 1.889726125, 0};
    double a2[] = {0, 0, 12.975 * 1.889726125};

    double Ekin = 3.0; // 40 Ry = 20 Ha

    parameters.set_lattice_vectors(a0, a1, a2);
    parameters.set_pw_cutoff(2 * sqrt(2 * Ekin) + 0.5);
    parameters.initialize();
    parameters.print_info();
    
    double vk[] = {0, 0, 0};
    K_point kp(parameters, vk, 1.0);
    kp.generate_gkvec(sqrt(2 * Ekin));

    std::cout << "num_gkvec = " << kp.num_gkvec() << std::endl;

    // generate some potential in plane-wave domain
    std::vector<complex16> v_pw(parameters.num_gvec());
    for (int ig = 0; ig < parameters.num_gvec(); ig++) v_pw[ig] = complex16(1.0 / (parameters.gvec_len(ig) + 1.0), 0.0);

    //== // cook the Hamiltonian
    //== mdarray<complex16, 2> hmlt(kp.num_gkvec(), kp.num_gkvec());
    //== hmlt.zero();
    //== for (int ig1 = 0; ig1 < kp.num_gkvec(); ig1++)
    //== {
    //==     for (int ig2 = 0; ig2 < kp.num_gkvec(); ig2++)
    //==     {
    //==         int ig = parameters.index_g12(kp.gvec_index(ig2), kp.gvec_index(ig1));
    //==         hmlt(ig2, ig1) = v_pw[ig];
    //==         if (ig1 == ig2) hmlt(ig2, ig1) += pow(kp.gkvec_cart(ig1).length(), 2) / 2.0;
    //==     }
    //== }

    //== standard_evp* solver = new standard_evp_lapack();

    //== std::vector<double> eval(kp.num_gkvec());
    //== mdarray<complex16, 2> evec(kp.num_gkvec(), kp.num_gkvec());

    //== solver->solve(kp.num_gkvec(), hmlt.get_ptr(), hmlt.ld(), &eval[0], evec.get_ptr(), evec.ld());

    //== delete solver;
    
    int num_bands = 10;

    //== printf("\n");
    //== printf("Lowest eigen-values (exact): \n");
    //== for (int i = 0; i < num_bands; i++)
    //== {
    //==     printf("i : %i,  eval : %16.10f\n", i, eval[i]);
    //== }


    diag_lobpcg(parameters, kp, v_pw, num_bands);

    
//    mdarray<complex16, 2> evec(Nmax, Nmax);
//    mdarray<complex16, 2> psi(parameters.num_gvec(), n);
//
//    // initial basis functions
//    mdarray<complex16, 2> phi(parameters.num_gvec(), Nmax);
//    phi.zero();
//    for (int i = 0; i < n; i++) phi(i, i) = 1.0;
//
//    for (int k = 0; k < 2; k++)
//    {
//        int nphi = diag_davidson(parameters, niter, bs, n, v_pw, phi, evec);
//
//        psi.zero();
//        for (int j = 0; j < n; j++)
//        {
//            for (int mu = 0; mu < nphi; mu++)
//            {
//                for (int ig = 0; ig < parameters.num_gvec(); ig++) psi(ig, j) += evec(mu, j) * phi(ig, mu);
//            }
//        }
//        for (int j = 0; j < n; j++) memcpy(&phi(0, j), &psi(0, j), parameters.num_gvec() * sizeof(complex16));
//    }
//





    
     

    parameters.clear();
}

int main(int argn, char** argv)
{
    Platform::initialize(true);

    test_lobpcg();

    Platform::finalize();
}
