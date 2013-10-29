#include <sirius.h>

using namespace sirius;

// N : subspace dimension
// n : required number of eigen pairs
// bs : block size (number of new basis functions)
// phi : basis functions
// hphi : H|phi>
void subspace_diag(Global& parameters, int N, int n, int bs, mdarray<complex16, 2>& phi, mdarray<complex16, 2>& hphi, 
                   mdarray<complex16, 2>& res, complex16 v0, mdarray<complex16, 2>& evec, std::vector<double>& eval)
{
    //== for (int i = 0; i < N; i++)
    //== {
    //==     for (int j = 0; j < N; j++)
    //==     {
    //==         complex16 z(0, 0);
    //==         for (int ig = 0; ig < parameters.num_gvec(); ig++)
    //==         {
    //==             z += conj(phi(ig, i)) * phi(ig, j);
    //==         }
    //==         if (i == j) z -= 1.0;
    //==         if (abs(z) > 1e-12) error_local(__FILE__, __LINE__, "basis is not orthogonal");
    //==     }
    //== }

    standard_evp* solver = new standard_evp_lapack();

    eval.resize(N);

    mdarray<complex16, 2> hmlt(N, N);
    blas<cpu>::gemm(2, 0, N, N, parameters.num_gvec(), &phi(0, 0), phi.ld(), &hphi(0, 0), hphi.ld(), &hmlt(0, 0), hmlt.ld());
    //== for (int i = 0; i < N; i++)
    //== {
    //==     for (int j = 0; j < N; j++)
    //==     {
    //==         for (int ig = 0; ig < parameters.num_gvec(); ig++) hmlt(i, j) += conj(phi(ig, i)) * hphi(ig, j);
    //==     }
    //== }

    solver->solve(N, hmlt.get_ptr(), hmlt.ld(), &eval[0], evec.get_ptr(), evec.ld());

    delete solver;

    printf("\n");
    printf("Lowest eigen-values : \n");
    for (int i = 0; i < std::min(N, 10); i++)
    {
        printf("i : %i,  eval : %16.10f\n", i, eval[i]);
    }

    // compute residuals
    res.zero();
    for (int j = 0; j < bs; j++)
    {
        int i = j; //n - bs + j;
        for (int mu = 0; mu < N; mu++)
        {
            for (int ig = 0; ig < parameters.num_gvec(); ig++)
            {
                res(ig, j) += (evec(mu, i) * hphi(ig, mu) - eval[i] * evec(mu, i) * phi(ig, mu));
            }
        }
        double norm = 0.0;
        for (int ig = 0; ig < parameters.num_gvec(); ig++) norm += real(conj(res(ig, j)) * res(ig, j));
        for (int ig = 0; ig < parameters.num_gvec(); ig++) res(ig, j) /= sqrt(norm);
    }
    
    // additional basis vectors
    for (int j = 0; j < bs; j++)
    {
        int i = j; //n - bs + j;
        for (int ig = 0; ig < parameters.num_gvec(); ig++)
        {
            complex16 t = pow(parameters.gvec_len(ig), 2) / 2.0 + v0 - eval[i];
            if (abs(t) < 1e-12) error_local(__FILE__, __LINE__, "problematic division");
            res(ig, j) /= t;
        }
    }
}

void expand_subspace(Global& parameters, int N, int bs, mdarray<complex16, 2>& phi, mdarray<complex16, 2>& res)
{
    // overlap between new addisional basis vectors and old basis vectors
    mdarray<complex16, 2> ovlp(N, bs);
    ovlp.zero();
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < bs; j++) 
        {
            for (int ig = 0; ig < parameters.num_gvec(); ig++) ovlp(i, j) += conj(phi(ig, i)) * res(ig, j);
        }
    }

    // project out the the old subspace
    for (int j = 0; j < bs; j++)
    {
        for (int i = 0; i < N; i++)
        {
            for (int ig = 0; ig < parameters.num_gvec(); ig++) res(ig, j) -= ovlp(i, j) * phi(ig, i);
        }
    }

    // orthogonalize
    for (int j = 0; j < bs; j++)
    {
        std::vector<complex16> v(parameters.num_gvec());
        memcpy(&v[0], &res(0, j), parameters.num_gvec() * sizeof(complex16));
        for (int j1 = 0; j1 < j; j1++)
        {
            complex16 z(0, 0);
            for (int ig = 0; ig < parameters.num_gvec(); ig++) z += conj(res(ig, j1)) * v[ig];
            for (int ig = 0; ig < parameters.num_gvec(); ig++) v[ig] -= z * res(ig, j1);
        }
        double norm = 0;
        for (int ig = 0; ig < parameters.num_gvec(); ig++) norm += real(conj(v[ig]) * v[ig]);
        //std::cout << "j=" << j <<" final norm=" << norm << std::endl;
        for (int ig = 0; ig < parameters.num_gvec(); ig++) res(ig, j) = v[ig] / sqrt(norm);
    }

    for (int j = 0; j < bs; j++)
    {
        for (int ig = 0; ig < parameters.num_gvec(); ig++) phi(ig, N + j) = res(ig, j);
    }
}

void apply_h(Global& parameters, int n, std::vector<complex16>& v_r, complex16* phi__, complex16* hphi__)
{
    mdarray<complex16, 2> phi(phi__, parameters.num_gvec(), n);
    mdarray<complex16, 2> hphi(hphi__, parameters.num_gvec(), n);
    std::vector<complex16> phi_r(parameters.fft().size());

    for (int i = 0; i < n; i++)
    {
        parameters.fft().input(parameters.num_gvec(), parameters.fft_index(), &phi(0, i));
        parameters.fft().transform(1);
        parameters.fft().output(&phi_r[0]);

        for (int ir = 0; ir < parameters.fft().size(); ir++) phi_r[ir] *= v_r[ir];

        parameters.fft().input(&phi_r[0]);
        parameters.fft().transform(-1);
        parameters.fft().output(parameters.num_gvec(), parameters.fft_index(), &hphi(0, i));

        for (int ig = 0; ig < parameters.num_gvec(); ig++) hphi(ig, i) += phi(ig, i) * pow(parameters.gvec_len(ig), 2) / 2.0;
    }
}

int diag_davidson(Global& parameters, int niter, int bs, int n, std::vector<complex16>& v_pw, mdarray<complex16, 2>& phi, 
                  mdarray<complex16, 2>& evec)
{
    std::vector<complex16> v_r(parameters.fft().size());
    parameters.fft().input(parameters.num_gvec(), parameters.fft_index(), &v_pw[0]);
    parameters.fft().transform(1);
    parameters.fft().output(&v_r[0]);

    for (int ir = 0; ir < parameters.fft().size(); ir++)
    {
        if (fabs(imag(v_r[ir])) > 1e-14) error_local(__FILE__, __LINE__, "potential is complex");
    }

    mdarray<complex16, 2> hphi(parameters.num_gvec(), phi.size(1));
    mdarray<complex16, 2> res(parameters.num_gvec(), bs);
    
    int N = n;

    apply_h(parameters, N, v_r, &phi(0, 0), &hphi(0, 0));

    std::vector<double> eval1;
    std::vector<double> eval2;

    for (int iter = 0; iter < niter; iter++)
    {
        eval2 = eval1;
        subspace_diag(parameters, N, n, bs, phi, hphi, res, v_pw[0], evec, eval1);
        expand_subspace(parameters, N, bs, phi, res);
        apply_h(parameters, bs, v_r, &res(0, 0), &hphi(0, N));

        if (iter)
        {
            double diff = 0;
            for (int i = 0; i < n; i++) diff += fabs(eval1[i] - eval2[i]);
            std::cout << "Eigen-value error : " << diff << std::endl;
        }
        
        N += bs;
    }
    return N - bs;
}


void test_davidson()
{
    Global parameters;

    double a0[] = {4, 0, 0};
    double a1[] = {0, 4, 0};
    double a2[] = {0, 0, 4};

    parameters.set_lattice_vectors(a0, a1, a2);
    parameters.set_pw_cutoff(10.0);
    parameters.initialize();
    parameters.print_info();

    // generate some potential in plane-wave domain
    std::vector<complex16> v_pw(parameters.num_gvec());
    int nnz = 0;
    for (int ig = 0; ig < parameters.num_gvec(); ig++)
    {   
        if (parameters.gvec_len(ig) <= 4)
        {
            v_pw[ig] = complex16(1.0 / (parameters.gvec_len(ig) + 1.0), 0.0);
            nnz++;
        }
        else
        {
            v_pw[ig] = 0.0;
        }

    }
    std::cout << "number of non-zero harmonics in the potential : " << nnz << std::endl;

    //=== // cook the Hamiltonian
    //=== mdarray<complex16, 2> hmlt(parameters.num_gvec(), parameters.num_gvec());
    //=== hmlt.zero();
    //=== for (int ig1 = 0; ig1 < parameters.num_gvec(); ig1++)
    //=== {
    //===     for (int ig2 = 0; ig2 < parameters.num_gvec(); ig2++)
    //===     {
    //===         int ig = parameters.index_g12_safe(ig2, ig1);
    //===         if (ig >= 0 && ig < parameters.num_gvec()) hmlt(ig2, ig1) = v_pw[ig];
    //===         if (ig1 == ig2) hmlt(ig2, ig1) += pow(parameters.gvec_len(ig1), 2) / 2.0;
    //===     }
    //=== }

    //=== standard_evp* solver = new standard_evp_lapack();

    //=== std::vector<double> eval(parameters.num_gvec());
    //=== mdarray<complex16, 2> evec(parameters.num_gvec(), parameters.num_gvec());

    //=== solver->solve(parameters.num_gvec(), hmlt.get_ptr(), hmlt.ld(), &eval[0], evec.get_ptr(), evec.ld());

    //=== delete solver;

    //=== printf("\n");
    //=== printf("Lowest eigen-values (exact): \n");
    //=== for (int i = 0; i < 10; i++)
    //=== {
    //===     printf("i : %i,  eval : %16.10f\n", i, eval[i]);
    //=== }


    
    int bs = 10; // increment of basis function
    int niter = 8; // number of diagonalization iterations
    int n = 10; // number of required eigen pairs
    int Nmax = n + bs * niter; // maximum capacity of the basis functions array


    mdarray<complex16, 2> evec(Nmax, Nmax);
    mdarray<complex16, 2> psi(parameters.num_gvec(), n);

    // initial basis functions
    mdarray<complex16, 2> phi(parameters.num_gvec(), Nmax);
    phi.zero();
    for (int i = 0; i < n; i++) phi(i, i) = 1.0;

    for (int k = 0; k < 2; k++)
    {
        int nphi = diag_davidson(parameters, niter, bs, n, v_pw, phi, evec);

        psi.zero();
        for (int j = 0; j < n; j++)
        {
            for (int mu = 0; mu < nphi; mu++)
            {
                for (int ig = 0; ig < parameters.num_gvec(); ig++) psi(ig, j) += evec(mu, j) * phi(ig, mu);
            }
        }
        for (int j = 0; j < n; j++) memcpy(&phi(0, j), &psi(0, j), parameters.num_gvec() * sizeof(complex16));
    }






    
     

    parameters.clear();
}

int main(int argn, char** argv)
{
    Platform::initialize(true);

    test_davidson();

    Platform::finalize();
}
