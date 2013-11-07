#include <sirius.h>

using namespace sirius;

void apply_h(Global& parameters, K_point& kp, int n, std::vector<complex16>& v_r, complex16* phi__, complex16* hphi__)
{
    Timer t("apply_h");

    mdarray<complex16, 2> phi(phi__, kp.num_gkvec(), n);
    mdarray<complex16, 2> hphi(hphi__, kp.num_gkvec(), n);
    
    int num_fft_threads = Platform::num_fft_threads();
    #pragma omp parallel default(shared) num_threads(num_fft_threads)
    {        
        int thread_id = omp_get_thread_num();
        std::vector<complex16> phi_r(parameters.fft().size());
        
        #pragma omp for
        for (int i = 0; i < n; i++)
        {
            parameters.fft().input(kp.num_gkvec(), kp.fft_index(), &phi(0, i), thread_id);
            parameters.fft().transform(1, thread_id);
            parameters.fft().output(&phi_r[0], thread_id);

            for (int ir = 0; ir < parameters.fft().size(); ir++) phi_r[ir] *= v_r[ir];

            parameters.fft().input(&phi_r[0], thread_id);
            parameters.fft().transform(-1, thread_id);
            parameters.fft().output(kp.num_gkvec(), kp.fft_index(), &hphi(0, i), thread_id);

            for (int ig = 0; ig < kp.num_gkvec(); ig++) hphi(ig, i) += phi(ig, i) * pow(kp.gkvec_cart(ig).length(), 2) / 2.0;
        }
    }
}

void orthonormalize(mdarray<complex16, 2>& f)
{
    Timer t("orthonormalize");

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
}

void check_orth(mdarray<complex16, 2>& f, int num_f)
{
    for (int i = 0; i < num_f; i++)
    {
        for (int j = 0; j < num_f; j++)
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
            double x = pow(kp.gkvec_cart(ig).length(), 2) / 2 / 1.5 / ekin;
            r(ig, i) = r(ig, i) * (27 + 18 * x + 12 * x * x + 8 * x * x * x) / (27 + 18 * x + 12 * x * x + 8 * x * x * x + 16 * x * x * x * x);
        }
    }
}

void expand_subspace(K_point& kp, int N, int num_bands, mdarray<complex16, 2>& phi, mdarray<complex16, 2>& res)
{
    Timer t("expand_subspace");

    // overlap between new addisional basis vectors and old basis vectors
    mdarray<complex16, 2> ovlp(N, num_bands);
    ovlp.zero();
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < num_bands; j++) 
        {
            for (int ig = 0; ig < kp.num_gkvec(); ig++) ovlp(i, j) += conj(phi(ig, i)) * res(ig, j);
        }
    }

    // project out the the old subspace
    for (int j = 0; j < num_bands; j++)
    {
        for (int i = 0; i < N; i++)
        {
            for (int ig = 0; ig < kp.num_gkvec(); ig++) res(ig, j) -= ovlp(i, j) * phi(ig, i);
        }
    }
    orthonormalize(res);

    for (int j = 0; j < num_bands; j++)
    {
        for (int ig = 0; ig < kp.num_gkvec(); ig++) phi(ig, N + j) = res(ig, j);
    }
}

void expand_subspace_v2(K_point& kp, int N, int num_bands, mdarray<complex16, 2>& phi, mdarray<complex16, 2>& res)
{
    Timer t("expand_subspace");

    assert(phi.size(0) == res.size(0));
    assert(phi.size(0) == kp.num_gkvec());
    memcpy(&phi(0, N), &res(0, 0), num_bands * kp.num_gkvec() * sizeof(complex16));
}

void diag_davidson(Global& parameters, K_point& kp, std::vector<complex16>& v_pw, int num_bands)
{
    Timer t("diag_davidson");

    std::vector<complex16> v_r(parameters.fft().size());
    parameters.fft().input(parameters.num_gvec(), parameters.fft_index(), &v_pw[0]);
    parameters.fft().transform(1);
    parameters.fft().output(&v_r[0]);

    for (int ir = 0; ir < parameters.fft().size(); ir++)
    {
        if (fabs(imag(v_r[ir])) > 1e-10) error_local(__FILE__, __LINE__, "potential is complex");
    }

    int num_phi = num_bands * 5;

    int num_iter = 100;

    mdarray<complex16, 2> phi(kp.num_gkvec(), num_phi);
    mdarray<complex16, 2> hphi(kp.num_gkvec(), num_phi);
    
    // initial basis functions
    phi.zero();
    for (int i = 0; i < num_bands; i++) phi(i, i) = 1.0;
    apply_h(parameters, kp, num_bands, v_r, &phi(0, 0), &hphi(0, 0));


    mdarray<complex16, 2> hmlt(num_phi, num_phi);
    mdarray<complex16, 2> evec(num_phi, num_phi);
    std::vector<double> eval(num_phi);
    
    mdarray<complex16, 2> res(kp.num_gkvec(), num_bands);

    mdarray<complex16, 2> res_active(kp.num_gkvec(), num_bands);

    mdarray<complex16, 2> zm(kp.num_gkvec(), num_bands); // temporary storage

    standard_evp* solver = new standard_evp_lapack();

    int N = num_bands; // intial eigen-value problem size

    for (int k = 0; k < num_iter; k++)
    {
        std::cout << "Iteration : " << k << ", subspace size : " << N << std::endl;

        // compute the Hamiltonian matrix: <phi|H|phi>
        blas<cpu>::gemm(2, 0, N, N, kp.num_gkvec(), &phi(0, 0), phi.ld(), &hphi(0, 0), hphi.ld(), &hmlt(0, 0), hmlt.ld());

        solver->solve(N, hmlt.get_ptr(), hmlt.ld(), &eval[0], evec.get_ptr(), evec.ld());
        
        // compute residuals
        // 1. \Psi_{i} = \phi_{mu} * Z_{mu, i}
        blas<cpu>::gemm(0, 0, kp.num_gkvec(), num_bands, N, &phi(0, 0), phi.ld(), &evec(0, 0), evec.ld(), &res(0, 0), res.ld());
        // 2. multiply \Psi_{i} with energy
        for (int i = 0; i < num_bands; i++)
        {
            for (int ig = 0; ig < kp.num_gkvec(); ig++) res(ig, i) *= eval[i];
        }
        // 3. r_{i} = H\Psi_{i} - E_{i}\Psi_{i}
        blas<cpu>::gemm(0, 0, kp.num_gkvec(), num_bands, N, complex16(1, 0), &hphi(0, 0), hphi.ld(), &evec(0, 0), evec.ld(), 
                        complex16(-1, 0), &res(0, 0), res.ld());

        // check which residuals are converged
        int n = 0;
        for (int i = 0; i < num_bands; i++)
        {
            double r = 0;
            for (int ig = 0; ig < kp.num_gkvec(); ig++) r += real(conj(res(ig, i)) * res(ig, i));
            if (r > 1e-5) 
            {
                memcpy(&res_active(0, n), &res(0, i), kp.num_gkvec() * sizeof(complex16));
                
                // apply preconditioner
                for (int ig = 0; ig < kp.num_gkvec(); ig++)
                {
                    complex16 t = pow(kp.gkvec_cart(ig).length(), 2) / 2.0 + v_pw[0] - eval[i];
                    if (abs(t) < 1e-12) error_local(__FILE__, __LINE__, "problematic division");
                    res_active(ig, n) /= t;
                }

                n++;
            }
            std::cout << "band : " << i << " residiual : " << r << " eigen-value : " << eval[i] << std::endl;
        }
        std::cout << "number of non-converged eigen-vectors : " << n << std::endl;
        if (n == 0) break;
        
        // check if we run out of variational space
        if (N + n > num_phi)
        {   
            // use current espansion of \Psi as a new starting basis
            blas<cpu>::gemm(0, 0, kp.num_gkvec(), num_bands, N, &phi(0, 0), phi.ld(), &evec(0, 0), evec.ld(), 
                            &zm(0, 0), zm.ld());
            memcpy(phi.get_ptr(), zm.get_ptr(), num_bands * kp.num_gkvec() * sizeof(complex16));
            apply_h(parameters, kp, num_bands, v_r, &phi(0, 0), &hphi(0, 0));
            N = num_bands;
        }
        
        //apply_p(kp, res_active);
        
        // expand variational subspace with new basis vectors obtatined from residuals
        expand_subspace(kp, N, n, phi, res_active);

        // apply Hamiltonian to the new basis functions
        apply_h(parameters, kp, n, v_r, &phi(0, N), &hphi(0, N));
        
        // increase the size of the variation space
        N += n;
    }

    delete solver;
}

void diag_davidson_v2(Global& parameters, K_point& kp, std::vector<complex16>& v_pw, int num_bands)
{
    Timer t("diag_davidson");

    std::vector<complex16> v_r(parameters.fft().size());
    parameters.fft().input(parameters.num_gvec(), parameters.fft_index(), &v_pw[0]);
    parameters.fft().transform(1);
    parameters.fft().output(&v_r[0]);

    for (int ir = 0; ir < parameters.fft().size(); ir++)
    {
        if (fabs(imag(v_r[ir])) > 1e-10) error_local(__FILE__, __LINE__, "potential is complex");
    }

    int num_phi = num_bands * 5;

    int num_iter = 20;

    mdarray<complex16, 2> phi(kp.num_gkvec(), num_phi);
    mdarray<complex16, 2> hphi(kp.num_gkvec(), num_phi);
    
    // initial basis functions
    phi.zero();
    for (int i = 0; i < num_bands; i++) phi(i, i) = 1.0;
    // apply Hamiltonian to intial states
    apply_h(parameters, kp, num_bands, v_r, &phi(0, 0), &hphi(0, 0));

    mdarray<complex16, 2> hmlt(num_phi, num_phi);
    mdarray<complex16, 2> ovlp(num_phi, num_phi);
    mdarray<complex16, 2> evec(num_phi, num_phi);
    std::vector<double> eval(num_phi);
    
    mdarray<complex16, 2> res(kp.num_gkvec(), num_bands);

    mdarray<complex16, 2> res_active(kp.num_gkvec(), num_bands);

    mdarray<complex16, 2> zm(kp.num_gkvec(), num_bands); // temporary storage

    std::vector<double> res_norm(num_bands); // norm of residuals

    standard_evp* solver = new standard_evp_lapack();
    
    generalized_evp* gevp = new generalized_evp_lapack(-1.0);

    int N = num_bands; // intial eigen-value problem size

    for (int k = 0; k < num_iter; k++)
    {
        std::cout << "Iteration : " << k << ", subspace size : " << N << std::endl;

        // compute the Hamiltonian matrix: <phi|H|phi>
        blas<cpu>::gemm(2, 0, N, N, kp.num_gkvec(), &phi(0, 0), phi.ld(), &hphi(0, 0), hphi.ld(), &hmlt(0, 0), hmlt.ld());
       
        // compute overlap matrix <phi|phi>
        blas<cpu>::gemm(2, 0, N, N, kp.num_gkvec(), &phi(0, 0), phi.ld(), &phi(0, 0), phi.ld(), &ovlp(0, 0), ovlp.ld());
        
        //solver->solve(N, hmlt.get_ptr(), hmlt.ld(), &eval[0], evec.get_ptr(), evec.ld());
        gevp->solve(N, num_bands, hmlt.get_ptr(), hmlt.ld(), ovlp.get_ptr(), ovlp.ld(), &eval[0], 
                    evec.get_ptr(), evec.ld());
        
        // compute residuals
        // 1. \Psi_{i} = \phi_{mu} * Z_{mu, i}
        blas<cpu>::gemm(0, 0, kp.num_gkvec(), num_bands, N, &phi(0, 0), phi.ld(), &evec(0, 0), evec.ld(), &res(0, 0), res.ld());
        // 2. multiply \Psi_{i} with energy
        for (int i = 0; i < num_bands; i++)
        {
            for (int ig = 0; ig < kp.num_gkvec(); ig++) res(ig, i) *= eval[i];
        }
        // 3. r_{i} = H\Psi_{i} - E_{i}\Psi_{i}
        blas<cpu>::gemm(0, 0, kp.num_gkvec(), num_bands, N, complex16(1, 0), &hphi(0, 0), hphi.ld(), &evec(0, 0), evec.ld(), 
                        complex16(-1, 0), &res(0, 0), res.ld());
        
        // compute norm and apply preconditioner
        #pragma omp parallel for
        for (int i = 0; i < num_bands; i++)
        {
            double r = 0;
            for (int ig = 0; ig < kp.num_gkvec(); ig++) r += real(conj(res(ig, i)) * res(ig, i));
            res_norm[i] = r;
            
            // apply preconditioner
            for (int ig = 0; ig < kp.num_gkvec(); ig++)
            {
                complex16 t = pow(kp.gkvec_cart(ig).length(), 2) / 2.0 + v_pw[0] - eval[i];
                if (abs(t) < 1e-12) error_local(__FILE__, __LINE__, "problematic division");
                res(ig, i) /= t;
            }
        }

        // check which residuals are converged
        int n = 0;
        for (int i = 0; i < num_bands; i++)
        {
            // take the residual if it's norm is above the threshold
            if (res_norm[i] > 1e-5) 
            {
                memcpy(&res_active(0, n), &res(0, i), kp.num_gkvec() * sizeof(complex16));
                n++;
            }
            std::cout << "band : " << i << " residiual : " << res_norm[i] << " eigen-value : " << eval[i] << std::endl;
        }
        std::cout << "number of non-converged eigen-vectors : " << n << std::endl;
        if (n == 0) break;
        
        // check if we run out of variational space
        if (N + n > num_phi)
        {   
            // use current espansion of \Psi as a new starting basis
            blas<cpu>::gemm(0, 0, kp.num_gkvec(), num_bands, N, &phi(0, 0), phi.ld(), &evec(0, 0), evec.ld(), 
                            &zm(0, 0), zm.ld());
            memcpy(phi.get_ptr(), zm.get_ptr(), num_bands * kp.num_gkvec() * sizeof(complex16));
            apply_h(parameters, kp, num_bands, v_r, &phi(0, 0), &hphi(0, 0));
            N = num_bands;
        }
        
        //apply_p(kp, res_active);
        
        // expand variational subspace with new basis vectors obtatined from residuals
        expand_subspace_v2(kp, N, n, phi, res_active);

        // apply Hamiltonian to the new basis functions
        apply_h(parameters, kp, n, v_r, &phi(0, N), &hphi(0, N));
        
        // increase the size of the variation space
        N += n;
    }

    delete solver;
    delete gevp;
}

void test_davidson()
{
    Global parameters;

    double a0[] = {12.975 * 1.889726125, 0, 0};
    double a1[] = {0, 12.975 * 1.889726125, 0};
    double a2[] = {0, 0, 12.975 * 1.889726125};

    double Ekin = 20.0; // 40 Ry in QE = 20 Ha here

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
    for (int ig = 0; ig < parameters.num_gvec(); ig++) v_pw[ig] = complex16(1.0 / pow(parameters.gvec_len(ig) + 1.0, 1), 0.0);

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
    
    int num_bands = 20;

    //== printf("\n");
    //== printf("Lowest eigen-values (exact): \n");
    //== for (int i = 0; i < num_bands; i++)
    //== {
    //==     printf("i : %i,  eval : %16.10f\n", i, eval[i]);
    //== }


    //diag_davidson(parameters, kp, v_pw, num_bands);
    diag_davidson_v2(parameters, kp, v_pw, num_bands);

    parameters.clear();
}

int main(int argn, char** argv)
{
    Platform::initialize(true);

    test_davidson();

    Timer::print();

    Platform::finalize();
}
