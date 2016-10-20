template <device_t pu>
inline void Density::generate_rho_aug(std::vector<Periodic_function<double>*> rho__,
                                      mdarray<double_complex, 2>& rho_aug__)
{
    PROFILE_WITH_TIMER("sirius::Density::generate_rho_aug");

    if (pu == CPU) {
        rho_aug__.zero();
    }

    #ifdef __GPU
    if (pu == GPU) {
        rho_aug__.zero_on_device();
    }
    #endif
    
    ctx_.augmentation_op(0).prepare(0);

    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
        auto& atom_type = unit_cell_.atom_type(iat);
        if (!atom_type.pp_desc().augment) {
            continue;
        }

        int nbf = atom_type.mt_basis_size();
        
        /* convert to real matrix */
        mdarray<double, 3> dm(nbf * (nbf + 1) / 2, atom_type.num_atoms(), ctx_.num_mag_dims() + 1);
        #pragma omp parallel for
        for (int i = 0; i < atom_type.num_atoms(); i++) {
            int ia = atom_type.atom_id(i);

            for (int xi2 = 0; xi2 < nbf; xi2++) {
                for (int xi1 = 0; xi1 <= xi2; xi1++) {
                    int idx12 = xi2 * (xi2 + 1) / 2 + xi1;
                    switch (ctx_.num_mag_dims()) {
                        case 0: {
                            dm(idx12, i, 0) = density_matrix_(xi2, xi1, 0, ia).real();
                            break;
                        }
                        case 1: {
                            dm(idx12, i, 0) = std::real(density_matrix_(xi2, xi1, 0, ia) + density_matrix_(xi2, xi1, 1, ia));
                            dm(idx12, i, 1) = std::real(density_matrix_(xi2, xi1, 0, ia) - density_matrix_(xi2, xi1, 1, ia));
                            break;
                        }
                    }
                }
            }
        }

        if (pu == CPU) {
            runtime::Timer t2("sirius::Density::generate_rho_aug|phase_fac");
            /* treat phase factors as real array with x2 size */
            mdarray<double, 2> phase_factors(atom_type.num_atoms(), ctx_.gvec_count() * 2);

            #pragma omp parallel for
            for (int igloc = 0; igloc < ctx_.gvec_count(); igloc++) {
                int ig = ctx_.gvec_offset() + igloc;
                for (int i = 0; i < atom_type.num_atoms(); i++) {
                    int ia = atom_type.atom_id(i);
                    double_complex z = std::conj(ctx_.gvec_phase_factor(ig, ia));
                    phase_factors(i, 2 * igloc)     = z.real();
                    phase_factors(i, 2 * igloc + 1) = z.imag();
                }
            }
            t2.stop();
            
            /* treat auxiliary array as double with x2 size */
            mdarray<double, 2> dm_pw(nbf * (nbf + 1) / 2, ctx_.gvec_count() * 2);

            for (int iv = 0; iv < ctx_.num_mag_dims() + 1; iv++) {
                runtime::Timer t3("sirius::Density::generate_rho_aug|gemm");
                linalg<CPU>::gemm(0, 0, nbf * (nbf + 1) / 2, 2 * ctx_.gvec_count(), atom_type.num_atoms(), 
                                  &dm(0, 0, iv), dm.ld(),
                                  &phase_factors(0, 0), phase_factors.ld(), 
                                  &dm_pw(0, 0), dm_pw.ld());
                t3.stop();

                #ifdef __PRINT_OBJECT_CHECKSUM
                {
                    auto cs = dm_pw.checksum();
                    ctx_.comm().allreduce(&cs, 1);
                    DUMP("checksum(dm_pw) : %18.10f", cs);
                }
                #endif

                runtime::Timer t4("sirius::Density::generate_rho_aug|sum");
                #pragma omp parallel for
                for (int igloc = 0; igloc < ctx_.gvec_count(); igloc++) {
                    double_complex zsum(0, 0);
                    /* get contribution from non-diagonal terms */
                    for (int i = 0; i < nbf * (nbf + 1) / 2; i++) {
                        double_complex z1 = double_complex(ctx_.augmentation_op(iat).q_pw(i, 2 * igloc),
                                                           ctx_.augmentation_op(iat).q_pw(i, 2 * igloc + 1));
                        double_complex z2(dm_pw(i, 2 * igloc), dm_pw(i, 2 * igloc + 1));

                        zsum += z1 * z2 * ctx_.augmentation_op(iat).sym_weight(i);
                    }
                    rho_aug__(igloc, iv) += zsum;
                }
                t4.stop();
            }
        }

        #ifdef __GPU
        if (pu == GPU) {
            dm.allocate(memory_t::device);
            dm.copy_to_device();

            /* treat auxiliary array as double with x2 size */
            mdarray<double, 2> dm_pw(nullptr, nbf * (nbf + 1) / 2, ctx_.gvec_count() * 2);
            dm_pw.allocate(memory_t::device);

            mdarray<double, 1> phase_factors(nullptr, atom_type.num_atoms() * ctx_.gvec_count() * 2);
            phase_factors.allocate(memory_t::device);

            acc::sync_stream(0);
            if (iat + 1 != unit_cell_.num_atom_types()) {
                ctx_.augmentation_op(iat + 1).prepare(0);
            }

            for (int iv = 0; iv < ctx_.num_mag_dims() + 1; iv++) {
                generate_dm_pw_gpu(atom_type.num_atoms(),
                                   ctx_.gvec_count(),
                                   nbf,
                                   ctx_.atom_coord(iat).at<GPU>(),
                                   ctx_.gvec_coord().at<GPU>(),
                                   phase_factors.at<GPU>(),
                                   dm.at<GPU>(0, 0, iv),
                                   dm_pw.at<GPU>(),
                                   1);
                sum_q_pw_dm_pw_gpu(ctx_.gvec_count(), 
                                   nbf,
                                   ctx_.augmentation_op(iat).q_pw().at<GPU>(),
                                   dm_pw.at<GPU>(),
                                   ctx_.augmentation_op(iat).sym_weight().at<GPU>(),
                                   rho_aug__.at<GPU>(0, iv),
                                   1);
            }
            acc::sync_stream(1);
            ctx_.augmentation_op(iat).dismiss();
        }
        #endif
    }

    #ifdef __GPU
    if (pu == GPU) {
        rho_aug__.copy_to_host();
    }
    #endif
    
    #ifdef __PRINT_OBJECT_CHECKSUM
    {
         auto cs = rho_aug__.checksum();
         DUMP("checksum(rho_aug): %20.14f %20.14f", cs.real(), cs.imag());
    }
    #endif
}

