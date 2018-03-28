//TODO: in case of Gamma-point transfrom two wave functions at once
//TODO: use GPU pointer

inline void Density::add_k_point_contribution_rg(K_point* kp__)
{
    PROFILE("sirius::Density::add_k_point_contribution_rg");

    double omega = unit_cell_.omega();

    auto& fft = ctx_.fft_coarse();
    
    /* get preallocated memory */
    double* ptr = static_cast<double*>(ctx_.memory_buffer(fft.local_size() * (ctx_.num_mag_dims() + 1) * sizeof(double)));

    mdarray<double, 2> density_rg(ptr, fft.local_size(), ctx_.num_mag_dims() + 1, "density_rg");
    density_rg.zero();

    if (fft.pu() == GPU) {
        density_rg.allocate(memory_t::device);
        density_rg.zero<memory_t::device>();
    }

    fft.prepare(kp__->gkvec_partition());

    /* non-magnetic or collinear case */
    if (ctx_.num_mag_dims() != 3) {
        /* loop over pure spinor components */
        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            /* trivial case */
            if (!kp__->spinor_wave_functions().pw_coeffs(ispn).spl_num_col().global_index_size()) {
                continue;
            }

            for (int i = 0; i < kp__->spinor_wave_functions().pw_coeffs(ispn).spl_num_col().local_size(); i++) {
                int j = kp__->spinor_wave_functions().pw_coeffs(ispn).spl_num_col()[i];
                double w = kp__->band_occupancy(j, ispn) * kp__->weight() / omega;

                ///* transform to real space; in case of GPU wave-function stays in GPU memory */
                fft.transform<1>(kp__->spinor_wave_functions().pw_coeffs(ispn).extra().template at<CPU>(0, i));
                //switch (fft.pu()) {
                //    case CPU: {
                //        fft.transform<1>(kp__->gkvec().partition(),
                //                         kp__->spinor_wave_functions(ispn).pw_coeffs().extra().template at<CPU>(0, i));
                //        break;
                //    }
                //    case GPU: {
                //        fft.transform<1, GPU>(kp__->gkvec().partition(),
                //                              kp__->spinor_wave_functions(ispn).pw_coeffs().extra().template at<GPU>(0, i));
                //        break;
                //    }
                //}
                
                /* add to density */
                switch (fft.pu()) {
                    case CPU: {
                        #pragma omp parallel for schedule(static)
                        for (int ir = 0; ir < fft.local_size(); ir++) {
                            auto z = fft.buffer(ir);
                            density_rg(ir, ispn) += w * (std::pow(z.real(), 2) + std::pow(z.imag(), 2));
                        }
                        break;
                    }
                    case GPU: {
#ifdef __GPU
                        update_density_rg_1_gpu(fft.local_size(), fft.buffer().at<GPU>(), w, density_rg.at<GPU>(0, ispn));
#else
                        TERMINATE_NO_GPU
#endif
                        break;
                    }
                }
            }
        }
    } else { /* non-collinear case */
        assert(kp__->spinor_wave_functions().pw_coeffs(0).spl_num_col().local_size() ==
               kp__->spinor_wave_functions().pw_coeffs(1).spl_num_col().local_size());
        
        /* allocate on CPU or GPU */
        mdarray<double_complex, 1> psi_r(fft.local_size(), ctx_.main_memory_t());

        for (int i = 0; i < kp__->spinor_wave_functions().pw_coeffs(0).spl_num_col().local_size(); i++) {
            int j    = kp__->spinor_wave_functions().pw_coeffs(0).spl_num_col()[i];
            double w = kp__->band_occupancy(j, 0) * kp__->weight() / omega;

            /* transform up- component of spinor function to real space; in case of GPU wave-function stays in GPU memory */
            fft.transform<1>(kp__->spinor_wave_functions().pw_coeffs(0).extra().template at<CPU>(0, i));
            /* save in auxiliary buffer */
            switch (fft.pu()) {
                case CPU: {
                    fft.output(&psi_r[0]);
                    break;
                }
                case GPU: {
#ifdef __GPU
                    acc::copyout(psi_r.at<GPU>(), fft.buffer().at<GPU>(), fft.local_size());
#endif
                    break;
                }
            }

            /* transform dn- component of spinor wave function */
            fft.transform<1>(kp__->spinor_wave_functions().pw_coeffs(1).extra().template at<CPU>(0, i));

            switch (fft.pu()) {
                case CPU: {
                    #pragma omp parallel for schedule(static)
                    for (int ir = 0; ir < fft.local_size(); ir++) {
                        auto r0 = (std::pow(psi_r[ir].real(), 2) + std::pow(psi_r[ir].imag(), 2)) * w;
                        auto r1 = (std::pow(fft.buffer(ir).real(), 2) + std::pow(fft.buffer(ir).imag(), 2)) * w;

                        auto z2 = psi_r[ir] * std::conj(fft.buffer(ir)) * w;

                        density_rg(ir, 0) += r0;
                        density_rg(ir, 1) += r1;
                        density_rg(ir, 2) += 2.0 * std::real(z2);
                        density_rg(ir, 3) -= 2.0 * std::imag(z2);
                    }
                    break;
                }
                case GPU: {
#ifdef __GPU
                    /* add up-up contribution */
                    update_density_rg_1_gpu(fft.local_size(), psi_r.at<GPU>(), w, density_rg.at<GPU>(0, 0));
                    /* add dn-dn contribution */
                    update_density_rg_1_gpu(fft.local_size(), fft.buffer().at<GPU>(), w, density_rg.at<GPU>(0, 1));
                    /* add off-diagonal contribution */
                    update_density_rg_2_gpu(fft.local_size(), psi_r.at<GPU>(), fft.buffer().at<GPU>(), w,
                                            density_rg.at<GPU>(0, 2), density_rg.at<GPU>(0, 3));
#endif
                    break;
                }
            }
        }
    }

    if (fft.pu() == GPU) {
        density_rg.copy<memory_t::device, memory_t::host>();
    }
    
    /* switch from real density matrix to density and magnetization */
    switch (ctx_.num_mag_dims()) {
        case 3: {
            #pragma omp parallel for schedule(static)
            for (int ir = 0; ir < fft.local_size(); ir++) {
                rho_mag_coarse_[2]->f_rg(ir) += density_rg(ir, 2); // Mx
                rho_mag_coarse_[3]->f_rg(ir) += density_rg(ir, 3); // My
            }
        }
        case 1: {
            #pragma omp parallel for schedule(static)
            for (int ir = 0; ir < fft.local_size(); ir++) {
                rho_mag_coarse_[0]->f_rg(ir) += (density_rg(ir, 0) + density_rg(ir, 1)); // rho
                rho_mag_coarse_[1]->f_rg(ir) += (density_rg(ir, 0) - density_rg(ir, 1)); // Mz
            }
            break;
        }
        case 0: {
            #pragma omp parallel for schedule(static)
            for (int ir = 0; ir < fft.local_size(); ir++) {
                rho_mag_coarse_[0]->f_rg(ir) += density_rg(ir, 0); // rho
            }
        }
    }

    fft.dismiss();
}

