#include <sirius.hpp>
#include <testing.hpp>

using namespace sirius;

using f_type = double;

int
test_vector_calculus(cmd_args const& args__)
{
    /* matrix of reciprocal vectors */
    r3::matrix<double> M({{1, 0, 0}, {0, 1, 0}, {0, 0, 1}});
    M = M * twopi;

    double pw_cutoff = args__.value<double>("pw_cutoff", 10.0);

    /* list of G-vectors distributed over MPI ranks */
    fft::Gvec gvec(M, pw_cutoff, mpi::Communicator::world(), true);

    /* this is how G-vetors need to be arranged for FFT transforms */
    auto gvp = std::make_shared<fft::Gvec_fft>(gvec, mpi::Communicator::world(), mpi::Communicator::self());

    /* get some estimation of the FFT grid */
    auto fft_grid = fft::get_min_grid(pw_cutoff, M);

    std::cout << "fft_grid: " << fft_grid[0] << " " << fft_grid[1] << " " << fft_grid[2] << std::endl;

    /* for SpFFT: we need to provide local size of z-dimension in real space */
    auto spl_z = fft::split_z_dimension(fft_grid[2], mpi::Communicator::world());

    /* select the device */
    auto spfft_pu = SPFFT_PU_HOST; //(pu__ == device_t::CPU) ? SPFFT_PU_HOST : SPFFT_PU_GPU;

    /* create SpFFT grid object */
    int const maxNumThreads{-1};
    spfft::Grid spfft_grid(fft_grid[0], fft_grid[1], fft_grid[2], gvp->zcol_count(), spl_z.local_size(), spfft_pu,
                           maxNumThreads, mpi::Communicator::world().native(), SPFFT_EXCH_DEFAULT);

    // fft::spfft_grid_type<f_type> spfft_grid(fft_grid[0], fft_grid[1], fft_grid[2], gvp->zcol_count(),
    //         spl_z.local_size(), spfft_pu, maxNumThreads, mpi::Communicator::world().native(), SPFFT_EXCH_DEFAULT);

    /* transform type: complex to real */
    const auto fft_type = SPFFT_TRANS_R2C;

    /* G-vector triplets in the FFT storage format */
    auto const& gv = gvp->gvec_array();
    /* create the FFT transform object */
    auto spfft =
            spfft_grid.create_transform(spfft_pu, fft_type, fft_grid[0], fft_grid[1], fft_grid[2], spl_z.local_size(),
                                        gvp->count(), SPFFT_INDEX_TRIPLETS, gv.at(memory_t::host));
    // fft::spfft_transform_type<f_type> spfft(spfft_grid.create_transform(spfft_pu, fft_type, fft_grid[0], fft_grid[1],
    //                                                                fft_grid[2], spl_z.local_size(), gvp->count(),
    //                                                                SPFFT_INDEX_TRIPLETS, gv.at(memory_t::host)));

    int num_points = spfft.local_slice_size();

    // mdarray<std::complex<double>, 1> fpw({gvp->count()});
    // mdarray<std::complex<double>, 1> frg({num_points});
    // mdarray<std::complex<double>, 1> gpw({gvp->count()});
    // mdarray<std::complex<double>, 1> grg({num_points});

    // double* fft_buf = spfft.space_domain_data(SPFFT_PU_HOST);

    // for (int iv = 0; iv < 10; iv++) {
    //     fpw.zero();
    //     fpw(iv) = std::complex<double>(1, 0);
    //     spfft.backward(reinterpret_cast<double const*>(fpw.at(memory_t::host)), SPFFT_PU_HOST);
    //     fft::spfft_output(spfft, frg.at(memory_t::host));

    //    double* ptr = reinterpret_cast<double*>(frg.at(memory_t::host));
    //    for (int i = 0; i < 2 * num_points; i++) {
    //        if (ptr[i] != fft_buf[i]) {
    //            std::cout << "wrong value" << std::endl;
    //        }
    //    }

    //    copy(frg, grg);

    //    fft::spfft_input(spfft, grg.at(memory_t::host));
    //    double* ptr1 = reinterpret_cast<double*>(grg.at(memory_t::host));
    //    //for (int i = 0; i < 2 * num_points; i++) {
    //    //    if (ptr1[i] != fft_buf[i]) {
    //    //        std::cout << "wrong value after spfft_input" << std::endl;
    //    //    }
    //    //}
    //    spfft.forward(SPFFT_PU_HOST, reinterpret_cast<double*>(gpw.at(memory_t::host)), SPFFT_FULL_SCALING);

    //    double d1{0};
    //    for (int ig = 0; ig < gvec.count(); ig++) {
    //        d1 += std::abs(gpw[ig] - fpw[ig]);
    //    }
    //    std::cout << "iv: " << iv << "  diff: " << d1 << std::endl;

    //    for (int ig = 0; ig < 20; ig++) {
    //        std::cout << "ig: " << ig << "  fpw: " << fpw[ig] << "  gpw: " << gpw[ig] << std::endl;
    //    }

    //}
    //
    if (true) {
        Smooth_periodic_function<f_type> f(spfft, gvp);
        Smooth_periodic_function<f_type> g(spfft, gvp);

        auto f1d = [](double t) -> double { return std::exp(std::cos(2 * twopi * t) + sin(twopi * t)); };

        for (int ix = 0; ix < fft_grid[0]; ix++) {
            double x = static_cast<double>(ix) / fft_grid[0];
            for (int iy = 0; iy < fft_grid[1]; iy++) {
                double y = static_cast<double>(iy) / fft_grid[1];
                for (int iz = 0; iz < fft_grid[2]; iz++) {
                    double z     = static_cast<double>(iz) / fft_grid[2];
                    int idx      = fft_grid.index_by_coord(ix, iy, iz);
                    f.value(idx) = f1d(x) * f1d(y) * f1d(z);
                    g.value(idx) = std::pow(f.value(idx), 1.0 / 3);
                }
            }
        }
        f.fft_transform(-1);
        g.fft_transform(-1);

        auto grad_g = gradient(g);
        /* transform to real space */
        for (int x : {0, 1, 2}) {
            grad_g[x].fft_transform(1);
        }

        Smooth_periodic_vector_function<f_type> f_grad_g(spfft, gvp);
        for (int x : {0, 1, 2}) {
            for (int i = 0; i < num_points; i++) {
                f_grad_g[x].value(i) = grad_g[x].value(i) * f.value(i);
            }
            /* transform to reciprocal space */
            f_grad_g[x].fft_transform(-1);
        }
        auto div_f_grad_g = divergence(f_grad_g);
        /* transform to real space */
        div_f_grad_g.fft_transform(1);

        auto grad_f = gradient(f);
        for (int x : {0, 1, 2}) {
            /* transform to real space */
            grad_f[x].fft_transform(1);
        }

        auto grad_f_grad_g = dot(grad_f, grad_g);
        auto lapl_g        = laplacian(g);
        lapl_g.fft_transform(1);

        double abs_diff{0};
        for (int i = 0; i < num_points; i++) {
            auto v1 = div_f_grad_g.value(i);
            auto v2 = grad_f_grad_g.value(i) + f.value(i) * lapl_g.value(i);
            abs_diff += std::abs(v1 - v2);
        }
        std::cout << " difference: " << abs_diff << std::endl;

        std::cout << "values along z" << std::endl;
        for (int z = 0; z < fft_grid[2]; z++) {
            int idx = fft_grid.index_by_coord(0, 0, z);
            std::cout << "z: " << static_cast<double>(z) / fft_grid[2] << std::setprecision(12)
                      << "  ∇(f * ∇g) = " << div_f_grad_g.value(idx)
                      << "  ∇f * ∇g + f ∆g = " << grad_f_grad_g.value(idx) + f.value(idx) * lapl_g.value(idx)
                      << std::endl;
        }
    }

    for (int iv = 0; iv < 10; iv++) { // gvec.count(); iv++) {
        std::cout << "Gvec: lattice: " << gvec.gvec<index_domain_t::local>(iv)
                  << " Cartesian: " << gvec.gvec_cart<index_domain_t::local>(iv) << std::endl;

        Smooth_periodic_function<f_type> f(spfft, gvp);
        Smooth_periodic_function<f_type> g(spfft, gvp);
        f.zero();
        f.f_pw_local(iv) = std::complex<double>(1, 0);
        // for (int ig = 0; ig < 10; ig++) {
        //     f.f_pw_local(ig) = random<std::complex<double>>() / std::pow(gvec.gvec_len<index_domain_t::local>(ig) +
        //     1, 2);
        // }
        f.fft_transform(1);
        if (true) {
            std::cout << " testing ∇(∇f) == ∆f identity;";
            auto lapl_f = laplacian(f);
            auto grad_f = gradient(f);
            lapl_f.fft_transform(1);
            auto div_grad_f = divergence(grad_f);
            div_grad_f.fft_transform(1);
            double diff{0};
            for (int i = 0; i < num_points; i++) {
                diff += std::abs(lapl_f.value(i) - div_grad_f.value(i));
            }
            std::cout << " difference: " << diff << std::endl;
        }

        g.zero();
        for (int i = 0; i < num_points; i++) {
            g.value(i) = f.value(i);
        }
        /* transform to reciprocal space */
        g.fft_transform(-1);
        if (true) {
            std::cout << " testing backward transformation;";
            double d1{0};
            for (int ig = 0; ig < gvec.count(); ig++) {
                d1 += std::abs(f.f_pw_local(ig) - g.f_pw_local(ig));
            }
            std::cout << " difference: " << d1 << std::endl;
        }

        std::cout << " testing ∇(f * ∇g) == ∇f * ∇g + f ∆g identity;";
        auto grad_g = gradient(g);
        /* transform to real space */
        for (int x : {0, 1, 2}) {
            grad_g[x].fft_transform(1);
        }

        Smooth_periodic_vector_function<f_type> f_grad_g(spfft, gvp);
        for (int x : {0, 1, 2}) {
            for (int i = 0; i < num_points; i++) {
                f_grad_g[x].value(i) = grad_g[x].value(i) * f.value(i);
            }
            /* transform to reciprocal space */
            f_grad_g[x].fft_transform(-1);
        }
        auto div_f_grad_g = divergence(f_grad_g);
        /* transform to real space */
        div_f_grad_g.fft_transform(1);

        auto grad_f = gradient(f);
        for (int x : {0, 1, 2}) {
            /* transform to real space */
            grad_f[x].fft_transform(1);
        }

        auto grad_f_grad_g = dot(grad_f, grad_g);
        auto lapl_g        = laplacian(g);
        lapl_g.fft_transform(1);

        double abs_diff{0};
        double s1{0};
        double s2{0};
        for (int i = 0; i < num_points; i++) {
            auto v1 = div_f_grad_g.value(i);
            auto v2 = grad_f_grad_g.value(i) + f.value(i) * lapl_g.value(i);
            s1 += std::abs(v1);
            s2 += std::abs(v2);
            abs_diff += std::abs(v1 - v2);
        }
        std::cout << " difference: " << abs_diff << std::endl;

        std::cout << "values along z" << std::endl;
        for (int z = 0; z < fft_grid[2]; z++) {
            int idx = fft_grid.index_by_coord(0, 0, z);
            std::cout << "z: " << static_cast<double>(z) / fft_grid[2] << "  ∇(f * ∇g) = " << div_f_grad_g.value(idx)
                      << "  ∇f * ∇g + f ∆g = " << grad_f_grad_g.value(idx) + f.value(idx) * lapl_g.value(idx)
                      << std::endl;
        }
        // if (abs_diff > 1e-6) {
        //     //std::cout << "pw=" << iv <<"   ∇(f * ∇g) = " << v1 << "    ∇f * ∇g + f ∆g = " << v2 << "    diff=" <<
        //     abs_diff << std::endl; std::cout << "pw=" << iv <<"  diff=" << abs_diff << std::endl;
        // }
    }

    return 0;
}

int
main(int argn, char** argv)
{
    cmd_args args(argn, argv,
                  {
                          {"pw_cutoff=", "(double) plane-wave cutoff"},
                  });

    sirius::initialize(1);

    call_test("test_vector_calculus", test_vector_calculus, args);

    sirius::finalize(1);
}
