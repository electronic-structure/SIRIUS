
//void Density::check_density_continuity_at_mt()
//{
//    // generate plane-wave coefficients of the potential in the interstitial region
//    ctx_.fft().input(&rho_->f_it<global>(0));
//    ctx_.fft().transform(-1);
//    ctx_.fft().output(ctx_.num_gvec(), ctx_.fft_index(), &rho_->f_pw(0));
//    
//    SHT sht(ctx_.lmax_rho());
//
//    double diff = 0.0;
//    for (int ia = 0; ia < ctx_.num_atoms(); ia++)
//    {
//        for (int itp = 0; itp < sht.num_points(); itp++)
//        {
//            double vc[3];
//            for (int x = 0; x < 3; x++) vc[x] = sht.coord(x, itp) * ctx_.atom(ia)->mt_radius();
//
//            double val_it = 0.0;
//            for (int ig = 0; ig < ctx_.num_gvec(); ig++) 
//            {
//                double vgc[3];
//                ctx_.get_coordinates<cartesian, reciprocal>(ctx_.gvec(ig), vgc);
//                val_it += real(rho_->f_pw(ig) * exp(double_complex(0.0, Utils::scalar_product(vc, vgc))));
//            }
//
//            double val_mt = 0.0;
//            for (int lm = 0; lm < ctx_.lmmax_rho(); lm++)
//                val_mt += rho_->f_rlm(lm, ctx_.atom(ia)->num_mt_points() - 1, ia) * sht.rlm_backward(lm, itp);
//
//            diff += fabs(val_it - val_mt);
//        }
//    }
//    printf("Total and average charge difference at MT boundary : %.12f %.12f\n", diff, diff / ctx_.num_atoms() / sht.num_points());
//}

