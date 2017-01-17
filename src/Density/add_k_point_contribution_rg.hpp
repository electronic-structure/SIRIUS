inline void Density::add_k_point_contribution_rg(K_point* kp__)
{
    PROFILE("sirius::Density::add_k_point_contribution_rg");

    int nfv = ctx_.num_fv_states();
    double omega = unit_cell_.omega();

    auto& fft = ctx_.fft_coarse();
    
    /* get preallocated memory */
    double* ptr = static_cast<double*>(ctx_.memory_buffer(fft.local_size() * (ctx_.num_mag_dims() + 1) * sizeof(double)));

    mdarray<double, 2> density_rg(ptr, fft.local_size(), ctx_.num_mag_dims() + 1, "density_rg");
    density_rg.zero();

    #ifdef __GPU
    if (fft.hybrid()) {
        density_rg.allocate(memory_t::device);
        density_rg.zero_on_device();
    }
    #endif

    fft.prepare(kp__->gkvec().partition());

    /* non-magnetic or collinear case */
    if (ctx_.num_mag_dims() != 3) {
        /* loop over pure spinor components */
        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            /* trivial case */
            if (!kp__->spinor_wave_functions(ispn).pw_coeffs().spl_num_col().global_index_size()) {
                continue;
            }

            #pragma omp for schedule(dynamic, 1)
            for (int i = 0; i < kp__->spinor_wave_functions(ispn).pw_coeffs().spl_num_col().local_size(); i++) {
                int j = kp__->spinor_wave_functions(ispn).pw_coeffs().spl_num_col()[i];
                double w = kp__->band_occupancy(j + ispn * nfv) * kp__->weight() / omega;

                /* transform to real space; in case of GPU wave-function stays in GPU memory */
                if (fft.gpu_only()) {
                    fft.transform<1>(kp__->gkvec().partition(),
                                     kp__->spinor_wave_functions(ispn).pw_coeffs().extra().template at<GPU>(0, i));
                } else {
                    fft.transform<1>(kp__->gkvec().partition(),
                                     kp__->spinor_wave_functions(ispn).pw_coeffs().extra().template at<CPU>(0, i));
                }

                if (fft.hybrid()) {
                    #ifdef __GPU
                    update_density_rg_1_gpu(fft.local_size(), fft.buffer<GPU>(), w, density_rg.at<GPU>(0, ispn));
                    #else
                    TERMINATE_NO_GPU
                    #endif
                } else {
                    #pragma omp parallel for
                    for (int ir = 0; ir < fft.local_size(); ir++) {
                        auto z = fft.buffer(ir);
                        density_rg(ir, ispn) += w * (std::pow(z.real(), 2) + std::pow(z.imag(), 2));
                    }
                }
            }
        }
    } else { /* non-collinear case */
        assert(kp__->spinor_wave_functions(0).pw_coeffs().spl_num_col().local_size() ==
               kp__->spinor_wave_functions(1).pw_coeffs().spl_num_col().local_size());

        std::vector<double_complex> psi_r(fft.local_size());

        #pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < kp__->spinor_wave_functions(0).pw_coeffs().spl_num_col().local_size(); i++) {
            int j = kp__->spinor_wave_functions(0).pw_coeffs().spl_num_col()[i];
            double w = kp__->band_occupancy(j) * kp__->weight() / omega;

            /* transform up- component of spinor function to real space; in case of GPU wave-function stays in GPU memory */
            fft.transform<1>(kp__->gkvec().partition(), kp__->spinor_wave_functions(0).pw_coeffs().extra().template at<CPU>(0, i));
            /* save in auxiliary buffer */
            fft.output(&psi_r[0]);
            /* transform dn- component of spinor wave function */
            fft.transform<1>(kp__->gkvec().partition(), kp__->spinor_wave_functions(1).pw_coeffs().extra().template at<CPU>(0, i));

            if (fft.hybrid()) {
                STOP();
                //#ifdef __GPU
                //update_it_density_matrix_1_gpu(ctx_.fft(thread_id)->local_size(), ispn, ctx_.fft(thread_id)->buffer<GPU>(), w,
                //                               it_density_matrix_gpu.at<GPU>());
                //#else
                //TERMINATE_NO_GPU
                //#endif
            } else {
                #pragma omp parallel for
                for (int ir = 0; ir < fft.local_size(); ir++) {
                    auto r0 = (std::pow(psi_r[ir].real(), 2) + std::pow(psi_r[ir].imag(), 2)) * w;
                    auto r1 = (std::pow(fft.buffer(ir).real(), 2) + std::pow(fft.buffer(ir).imag(), 2)) * w;

                    auto z2 = psi_r[ir] * std::conj(fft.buffer(ir)) * w;

                    density_rg(ir, 0) += r0;
                    density_rg(ir, 1) += r1;
                    density_rg(ir, 2) += 2.0 * std::real(z2);
                    density_rg(ir, 3) -= 2.0 * std::imag(z2);
                }
            }
        }
    }

    #ifdef __GPU
    if (fft.hybrid()) {
        density_rg.copy_to_host();
    }
    #endif
    
    /* switch from real density matrix to density and magnetization */
    switch (ctx_.num_mag_dims()) {
        case 3: {
            #pragma omp parallel for
            for (int ir = 0; ir < fft.local_size(); ir++) {
                rho_mag_coarse_[2]->f_rg(ir) += density_rg(ir, 2); // Mx
                rho_mag_coarse_[3]->f_rg(ir) += density_rg(ir, 3); // My
            }
        }
        case 1: {
            #pragma omp parallel for
            for (int ir = 0; ir < fft.local_size(); ir++) {
                rho_mag_coarse_[0]->f_rg(ir) += (density_rg(ir, 0) + density_rg(ir, 1)); // rho
                rho_mag_coarse_[1]->f_rg(ir) += (density_rg(ir, 0) - density_rg(ir, 1)); // Mz
            }
            break;
        }
        case 0: {
            #pragma omp parallel for
            for (int ir = 0; ir < fft.local_size(); ir++) {
                rho_mag_coarse_[0]->f_rg(ir) += density_rg(ir, 0); // rho
            }
        }
    }

    fft.dismiss();
}

