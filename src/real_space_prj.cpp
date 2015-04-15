#include "real_space_prj.h"

namespace sirius {

Real_space_prj::Real_space_prj(Unit_cell* unit_cell__,
                               Communicator const& comm__,
                               double R_mask_scale__,
                               double mask_alpha__,
                               double pw_cutoff__,
                               int num_fft_threads__,
                               int num_fft_workers__)
    : unit_cell_(unit_cell__),
      comm_(comm__),
      R_mask_scale_(R_mask_scale__),
      mask_alpha_(mask_alpha__)
{
    Timer t("sirius::Real_space_prj::Real_space_prj");

    std::cout << "pw_cutoff__" << pw_cutoff__ << std::endl;

    //== for (int iat = 0; iat < unit_cell_->num_atom_types(); iat++)
    //== {
    //==     std::stringstream s;
    //==     s << "beta_rf_" << iat << ".dat";
    //==     FILE* fout = fopen(s.str().c_str(), "w");
    //==     for (int idxrf = 0; idxrf < unit_cell_->atom_type(iat)->uspp().num_beta_radial_functions; idxrf++)
    //==     {
    //==         for (int ir = 0; ir < unit_cell_->atom_type(iat)->uspp().num_beta_radial_points[idxrf]; ir++)
    //==         {
    //==             fprintf(fout, "%18.12f %18.12f\n", unit_cell_->atom_type(iat)->radial_grid(ir), 
    //==                                                unit_cell_->atom_type(iat)->uspp().beta_radial_functions(ir, idxrf));
    //==         }
    //==         fprintf(fout, "\n");
    //==     }
    //==     fclose(fout);
    //== }


    std::vector<double> lww_mask = {
        0.10000000E+01, 0.10000000E+01, 0.99948662E+00, 0.99863154E+00,
        0.99743557E+00, 0.99589985E+00, 0.99402586E+00, 0.99181538E+00,
        0.98927052E+00, 0.98639370E+00, 0.98318766E+00, 0.97965544E+00,
        0.97580040E+00, 0.97162618E+00, 0.96713671E+00, 0.96233623E+00,
        0.95722924E+00, 0.95182053E+00, 0.94611516E+00, 0.94011842E+00,
        0.93383589E+00, 0.92727338E+00, 0.92043693E+00, 0.91333282E+00,
        0.90596753E+00, 0.89834777E+00, 0.89048044E+00, 0.88237263E+00,
        0.87403161E+00, 0.86546483E+00, 0.85667987E+00, 0.84768450E+00,
        0.83848659E+00, 0.82909416E+00, 0.81951535E+00, 0.80975838E+00,
        0.79983160E+00, 0.78974340E+00, 0.77950227E+00, 0.76911677E+00,
        0.75859548E+00, 0.74794703E+00, 0.73718009E+00, 0.72630334E+00,
        0.71532544E+00, 0.70425508E+00, 0.69310092E+00, 0.68187158E+00,
        0.67057566E+00, 0.65922170E+00, 0.64781819E+00, 0.63637355E+00,
        0.62489612E+00, 0.61339415E+00, 0.60187581E+00, 0.59034914E+00,
        0.57882208E+00, 0.56730245E+00, 0.55579794E+00, 0.54431609E+00,
        0.53286431E+00, 0.52144984E+00, 0.51007978E+00, 0.49876105E+00,
        0.48750040E+00, 0.47630440E+00, 0.46517945E+00, 0.45413176E+00,
        0.44316732E+00, 0.43229196E+00, 0.42151128E+00, 0.41083069E+00,
        0.40025539E+00, 0.38979038E+00, 0.37944042E+00, 0.36921008E+00,
        0.35910371E+00, 0.34912542E+00, 0.33927912E+00, 0.32956851E+00,
        0.31999705E+00, 0.31056799E+00, 0.30128436E+00, 0.29214897E+00,
        0.28316441E+00, 0.27433307E+00, 0.26565709E+00, 0.25713844E+00,
        0.24877886E+00, 0.24057988E+00, 0.23254283E+00, 0.22466884E+00,
        0.21695884E+00, 0.20941357E+00, 0.20203357E+00, 0.19481920E+00,
        0.18777065E+00, 0.18088790E+00, 0.17417080E+00, 0.16761900E+00,
        0.16123200E+00, 0.15500913E+00, 0.14894959E+00, 0.14305240E+00,
        0.13731647E+00, 0.13174055E+00, 0.12632327E+00, 0.12106315E+00,
        0.11595855E+00, 0.11100775E+00, 0.10620891E+00, 0.10156010E+00,
        0.97059268E-01, 0.92704295E-01, 0.88492966E-01, 0.84422989E-01,
        0.80492001E-01, 0.76697569E-01, 0.73037197E-01, 0.69508335E-01,
        0.66108380E-01, 0.62834685E-01, 0.59684561E-01, 0.56655284E-01,
        0.53744102E-01, 0.50948236E-01, 0.48264886E-01, 0.45691239E-01,
        0.43224469E-01, 0.40861744E-01, 0.38600231E-01, 0.36437098E-01,
        0.34369520E-01, 0.32394681E-01, 0.30509780E-01, 0.28712032E-01,
        0.26998673E-01, 0.25366964E-01, 0.23814193E-01, 0.22337676E-01,
        0.20934765E-01, 0.19602844E-01, 0.18339338E-01, 0.17141711E-01,
        0.16007467E-01, 0.14934157E-01, 0.13919377E-01, 0.12960772E-01,
        0.12056034E-01, 0.11202905E-01, 0.10399183E-01, 0.96427132E-02,
        0.89313983E-02, 0.82631938E-02, 0.76361106E-02, 0.70482151E-02,
        0.64976294E-02, 0.59825322E-02, 0.55011581E-02, 0.50517982E-02,
        0.46327998E-02, 0.42425662E-02, 0.38795566E-02, 0.35422853E-02,
        0.32293218E-02, 0.29392897E-02, 0.26708663E-02, 0.24227820E-02,
        0.21938194E-02, 0.19828122E-02, 0.17886449E-02, 0.16102512E-02,
        0.14466132E-02, 0.12967606E-02, 0.11597692E-02, 0.10347601E-02,
        0.92089812E-03, 0.81739110E-03, 0.72348823E-03, 0.63847906E-03,
        0.56169212E-03, 0.49249371E-03, 0.43028657E-03, 0.37450862E-03,
        0.32463165E-03, 0.28016004E-03, 0.24062948E-03, 0.20560566E-03,
        0.17468305E-03, 0.14748362E-03, 0.12365560E-03, 0.10287226E-03,
        0.84830727E-04, 0.69250769E-04, 0.55873673E-04, 0.44461100E-04,
        0.34793983E-04, 0.26671449E-04, 0.19909778E-04, 0.14341381E-04,
        0.98138215E-05
    };

    std::vector<double> x(201);
    for (int i = 0; i < 201; i++) x[i] = i * 0.005;
    Radial_grid r(x);
    mask_spline_ = Spline<double>(r, lww_mask);
    mask_spline_.interpolate();

    if (mask(1, 1) > 1e-13) TERMINATE("wrong mask function");

    fft_ = new FFT3D<CPU>(Utils::find_translation_limits(pw_cutoff__, unit_cell_->reciprocal_lattice_vectors()),
                          num_fft_threads__, num_fft_workers__);

    fft_->init_gvec(pw_cutoff__, unit_cell_->reciprocal_lattice_vectors());

    //double rmin = 15 / fft_->gvec_shell_len(fft_->num_gvec_shells_inner() - 1) / R_mask_scale_;

    spl_num_gvec_ = splindex<block>(fft_->size(), comm_.size(), comm_.rank());

    get_beta_R();
    get_beta_grid();

    
    /* radial beta functions */
    mdarray<Spline<double>, 2> beta_rf(unit_cell_->max_mt_radial_basis_size(), unit_cell_->num_atom_types());

    for (int iat = 0; iat < unit_cell_->num_atom_types(); iat++)
    {
        auto atom_type = unit_cell_->atom_type(iat);

        /* create radial grid for all beta-projectors */
        auto beta_radial_grid = unit_cell_->atom_type(iat)->radial_grid().segment(nr_beta_[iat]);

        double Rmask = R_beta_[iat] * R_mask_scale_;

        for (int idxrf = 0; idxrf < atom_type->mt_radial_basis_size(); idxrf++)
        {
            beta_rf(idxrf, iat) = Spline<double>(beta_radial_grid);
            for (int ir = 0; ir < nr_beta_[iat]; ir++) 
            {
                double x = beta_radial_grid[ir];
                beta_rf(idxrf, iat)[ir] = atom_type->uspp().beta_radial_functions(ir, idxrf) / mask(x, Rmask);
            }
            beta_rf(idxrf, iat).interpolate();
        }
    }







    auto beta_radial_integrals = generate_beta_radial_integrals(beta_rf, 1);

    //filter_radial_functions(mask_alpha_ * pw_cutoff__);

    filter_radial_functions_v2(pw_cutoff__);
    
    auto beta_radial_integrals_filtered = generate_beta_radial_integrals(beta_rf_filtered_, 2);
    
    //== double err = 0;
    //== for (int igsh = 0; igsh < fft_->num_gvec_shells_total(); igsh++)
    //== {
    //==     for (int iat = 0; iat < unit_cell_->num_atom_types(); iat++)
    //==     {
    //==         auto atom_type = unit_cell_->atom_type(iat);

    //==         for (int idxrf = 0; idxrf < atom_type->mt_radial_basis_size(); idxrf++)
    //==             err += std::abs(beta_radial_integrals(idxrf, iat, igsh) - beta_radial_integrals_filtered(idxrf, iat, igsh));
    //==     }
    //== }
    //== printf("Total error in radial integrals: %f\n", err);



    auto beta_pw_t = generate_beta_pw_t(unit_cell_, beta_radial_integrals_filtered);







    std::vector<double_complex> beta_pw(fft_->size());
    for (int ia = 0; ia < unit_cell_->num_atoms(); ia++)
    {
        int iat = unit_cell_->atom(ia)->type_id();
        auto atom_type = unit_cell_->atom_type(iat);
        //double Rmask = R_beta_[iat] * R_mask_scale_;
        
        beta_projectors_[ia].beta_ = mdarray<double, 2>(beta_projectors_[ia].num_points_, atom_type->mt_basis_size());

        for (int xi = 0; xi < atom_type->mt_basis_size(); xi++)
        {
            for (int ig_loc = 0; ig_loc < (int)spl_num_gvec_.local_size(); ig_loc++)
            {
                int ig = (int)spl_num_gvec_[ig_loc];
                double_complex phase_factor = std::exp(double_complex(0.0, twopi * (fft_->gvec(ig) * unit_cell_->atom(ia)->position())));

                beta_pw[ig] = beta_pw_t(ig_loc, atom_type->offset_lo() + xi) * conj(phase_factor);
            }
            comm_.allgather(&beta_pw[0], (int)spl_num_gvec_.global_offset(), (int)spl_num_gvec_.local_size());
            
            memset(&fft_->buffer(0), 0, fft_->size() * sizeof(double_complex));
            for (int ig = 0; ig < fft_->size(); ig++)
            {
                auto gvec = fft_->gvec(ig);
                double_complex z = beta_pw[ig];
                //auto gvec_cart = fft_->gvec_cart(ig);
                //if (gvec_cart.length() > pw_cutoff__) z *= std::exp(-mask_alpha_ * std::pow(gvec_cart.length() / pw_cutoff__ - 1, 2));
                fft_->buffer(fft_->index(gvec[0], gvec[1], gvec[2])) = z;
            }
            //fft_->input(fft_->num_gvec(), fft_->index_map(), &beta_pw[0]);
            fft_->transform(1);

            //== double p = 0;
            //== for (int i = 0; i < fft_->size(); i++)
            //==     p += std::pow(std::abs(fft_->buffer(i)), 2);
            //== std::cout << "ia,xi="<<ia<<", "<<xi<<"  prod="<<p * unit_cell_->omega() / fft_->size() << std::endl;
            
            
            for (int i = 0; i < beta_projectors_[ia].num_points_; i++)
            {
                int ir = beta_projectors_[ia].ir_[i];
                //double dist = beta_projectors_[ia].dist_[i];
                double b = real(fft_->buffer(ir));// * mask(dist, Rmask));
                beta_projectors_[ia].beta_(i, xi) = b;
            }
            //int lm = atom_type->indexb(xi).lm;
            //int idxrf = atom_type->indexb(xi).idxrf;
            //int lmax = unit_cell_->lmax_beta();
            //double rlm[Utils::lmmax(lmax)];

            //for (int i = 0; i < beta_projectors_[ia].num_points_; i++)
            //{
            //    auto rtp = beta_projectors_[ia].rtp_[i];

            //    SHT::spherical_harmonics(lmax, rtp[1], rtp[2], &rlm[0]);
            //    beta_projectors_[ia].beta_(i, xi) = beta_rf_filtered_(idxrf, iat)(rtp[0]) * rlm[lm];
            //}
        }
    }

    if (false)
    {
        for (int iat = 0; iat < unit_cell_->num_atom_types(); iat++)
        {
            auto atom_type = unit_cell_->atom_type(iat);
            for (int idxrf = 0; idxrf < atom_type->mt_radial_basis_size(); idxrf++)
            {
                for (int ir = 0; ir < nr_beta_[iat]; ir++) 
                    beta_rf(idxrf, iat)[ir] = atom_type->uspp().beta_radial_functions(ir, idxrf);
                beta_rf(idxrf, iat).interpolate();
            }
        }
        beta_radial_integrals = generate_beta_radial_integrals(beta_rf, 1);
        beta_pw_t = generate_beta_pw_t(unit_cell_, beta_radial_integrals);

        for (int ia = 0; ia < unit_cell_->num_atoms(); ia++)
        {
            int iat = unit_cell_->atom(ia)->type_id();
            auto atom_type = unit_cell_->atom_type(iat);

            double err = 0;
            for (int xi = 0; xi < atom_type->mt_basis_size(); xi++)
            {
                for (int ig_loc = 0; ig_loc < (int)spl_num_gvec_.local_size(); ig_loc++)
                {
                    int ig = (int)spl_num_gvec_[ig_loc];
                    double_complex phase_factor = std::exp(double_complex(0.0, twopi * (fft_->gvec(ig) * unit_cell_->atom(ia)->position())));

                    beta_pw[ig] = beta_pw_t(ig_loc, atom_type->offset_lo() + xi) * conj(phase_factor);
                }
                comm_.allgather(&beta_pw[0], (int)spl_num_gvec_.global_offset(), (int)spl_num_gvec_.local_size());

                for (int ig = 0; ig < fft_->num_gvec(); ig++)
                {
                    double_complex z(0, 0);
                    for (int i = 0; i < beta_projectors_[ia].num_points_; i++)
                    {
                        double phase = twopi * (beta_projectors_[ia].r_[i] * fft_->gvec(ig));
                        z += std::exp(double_complex(0, -phase)) * beta_projectors_[ia].beta_(i, xi);
                    }
                    err += std::abs(beta_pw[ig] - z / std::sqrt(unit_cell_->omega()));
                }
            }
            std::cout << "err=" << err << std::endl;
            STOP();
                

            //for (int xi1 = 0; xi1 < atom_type->mt_basis_size(); xi1++)
            //{
            //    int lm1 = atom_type->indexb(xi1).lm;
            //    int idxrf1 = atom_type->indexb(xi1).idxrf;
            //    for (int xi2 = 0; xi2 < atom_type->mt_basis_size(); xi2++)
            //    {
            //        int lm2 = atom_type->indexb(xi2).lm;
            //        int idxrf2 = atom_type->indexb(xi2).idxrf;
            //        double prod_rs = 0;
            //        for (int i = 0; i < beta_projectors_[ia].num_points_; i++)
            //        {
            //            prod_rs += beta_projectors_[ia].beta_(i, xi1) * beta_projectors_[ia].beta_(i, xi2);
            //        }
            //        prod_rs *= (unit_cell_->omega() / fft_->size());
            //        
            //        double prod_pw = 0;
            //        for (int ig = 0; ig < fft_->num_gvec(); ig++)
            //        { 
            //            prod_pw += real(conj(beta_pw_t(ig, atom_type->offset_lo() + xi1)) * beta_pw_t(ig, atom_type->offset_lo() + xi2));
            //        }
            //        prod_pw *= unit_cell_->omega();

            //        double prod_exact = 0;
            //        if (lm1 == lm2)
            //        {
            //            for (int ir = 0; ir < nr_beta_[iat]; ir++) 
            //            {
            //                s[ir] = atom_type->uspp().beta_radial_functions(ir, idxrf1) * atom_type->uspp().beta_radial_functions(ir, idxrf2);
            //            }
            //            prod_exact = s.interpolate().integrate(0);
            //        }
            //        err += std::abs(prod_rs - prod_pw);

            //        printf("xi1,xi2=%2i,%2i,  prod(rs, pw, exact): %18.12f %18.12f %18.12f\n", xi1, xi2, prod_rs, prod_pw, prod_exact);
            //    }
            //}
            //if (comm_.rank() == 0)
            //{
            //    printf("atom: %i, projector errror: %12.6f\n", ia, err);
            //}
        }
    }

}

mdarray<double, 3> Real_space_prj::generate_beta_radial_integrals(mdarray<Spline<double>, 2>& beta_rf__, int m__)
{
    Timer t("sirius::Real_space_prj::generate_beta_radial_integrals");

    mdarray<double, 3> beta_radial_integrals(unit_cell_->max_mt_radial_basis_size(), unit_cell_->num_atom_types(), fft_->num_gvec_shells_total());

    splindex<block> spl_gsh(fft_->num_gvec_shells_total(), comm_.size(), comm_.rank());
    #pragma omp parallel
    {
        mdarray<Spline<double>, 2> jl(unit_cell_->lmax_beta() + 1, unit_cell_->num_atom_types());
        for (int iat = 0; iat < unit_cell_->num_atom_types(); iat++)
        {
            for (int l = 0; l <= unit_cell_->lmax_beta(); l++) jl(l, iat) = Spline<double>(beta_rf__(0, iat).radial_grid());
        }

        #pragma omp for
        for (int igsh_loc = 0; igsh_loc < (int)spl_gsh.local_size(); igsh_loc++)
        {
            int igsh = (int)spl_gsh[igsh_loc];

            /* get spherical Bessel functions */
            double G = fft_->gvec_shell_len(igsh);
            std::vector<double> v(unit_cell_->lmax_beta() + 1);
            for (int iat = 0; iat < unit_cell_->num_atom_types(); iat++)
            {
                for (int ir = 0; ir < beta_rf__(0, iat).num_points(); ir++)
                {
                    double x = beta_rf__(0, iat).x(ir) * G;
                    gsl_sf_bessel_jl_array(unit_cell_->lmax_beta(), x, &v[0]);
                    for (int l = 0; l <= unit_cell_->lmax_beta(); l++) jl(l, iat)[ir] = v[l];
                }
                for (int l = 0; l <= unit_cell_->lmax_beta(); l++) jl(l, iat).interpolate();
            }
            
            /* compute radial integrals */
            for (int iat = 0; iat < unit_cell_->num_atom_types(); iat++)
            {
                auto atom_type = unit_cell_->atom_type(iat);
                for (int idxrf = 0; idxrf < atom_type->mt_radial_basis_size(); idxrf++)
                {
                    int l = atom_type->indexr(idxrf).l;
                    int nr = beta_rf__(idxrf, iat).num_points();
                    beta_radial_integrals(idxrf, iat, igsh) = Spline<double>::integrate(&jl(l, iat), &beta_rf__(idxrf, iat), m__, nr);
                }
            }
        }
    }
    
    int ld = unit_cell_->max_mt_radial_basis_size() * unit_cell_->num_atom_types();
    comm_.allgather(beta_radial_integrals.at<CPU>(), static_cast<int>(ld * spl_gsh.global_offset()), 
                    static_cast<int>(ld * spl_gsh.local_size()));
    
    return beta_radial_integrals;
}

mdarray<double_complex, 2> Real_space_prj::generate_beta_pw_t(Unit_cell* uc__,
                                                              mdarray<double, 3>& beta_radial_integrals__)
{
    Timer t("sirius::Real_space_prj::generate_beta_pw_t");

    mdarray<double_complex, 2> beta_pw_t(spl_num_gvec_.local_size(), uc__->num_beta_t());
    
    #pragma omp parallel
    {
        std::vector<double> gvec_rlm(Utils::lmmax(uc__->lmax_beta()));
        #pragma omp for
        for (int ig_loc = 0; ig_loc < (int)spl_num_gvec_.local_size(); ig_loc++)
        {
            int ig = (int)spl_num_gvec_[ig_loc];

            auto rtp = SHT::spherical_coordinates(fft_->gvec_cart(ig));
            SHT::spherical_harmonics(uc__->lmax_beta(), rtp[1], rtp[2], &gvec_rlm[0]);

            int igsh = fft_->gvec_shell(ig);
            for (int iat = 0; iat < uc__->num_atom_types(); iat++)
            {
                auto atom_type = uc__->atom_type(iat);
            
                for (int xi = 0; xi < atom_type->mt_basis_size(); xi++)
                {
                    int l = atom_type->indexb(xi).l;
                    int lm = atom_type->indexb(xi).lm;
                    int idxrf = atom_type->indexb(xi).idxrf;

                    double_complex z = std::pow(double_complex(0, -1), l) * fourpi / uc__->omega();
                    beta_pw_t(ig_loc, atom_type->offset_lo() + xi) = z * gvec_rlm[lm] * beta_radial_integrals__(idxrf, iat, igsh);
                }
            }
        }
    }

    return beta_pw_t;
}

void Real_space_prj::filter_radial_functions(double pw_cutoff__)
{
    int nq = 200;
    
    //double alpha = mask_alpha_;
    //double qcut = pw_cutoff__ / alpha;
    //double b = 5.0 * std::log(10.0) / std::pow(alpha - 1, 2);


    mdarray<double, 3> beta_radial_integrals(unit_cell_->max_mt_radial_basis_size(), unit_cell_->num_atom_types(), nq);

    /* interpolate beta radial functions divided by a mask function */
    std::vector<Radial_grid> beta_radial_grid(unit_cell_->num_atom_types());

    mdarray<Spline<double>, 2> beta_rf(unit_cell_->max_mt_radial_basis_size(), unit_cell_->num_atom_types());
    for (int iat = 0; iat < unit_cell_->num_atom_types(); iat++)
    {
        auto atom_type = unit_cell_->atom_type(iat);

        /* create radial grid for all beta-projectors */
        beta_radial_grid[iat] = unit_cell_->atom_type(iat)->radial_grid().segment(nr_beta_[iat]);

        double Rmask = R_beta_[iat] * R_mask_scale_;

        for (int idxrf = 0; idxrf < atom_type->mt_radial_basis_size(); idxrf++)
        {
            beta_rf(idxrf, iat) = Spline<double>(beta_radial_grid[iat]);
            for (int ir = 0; ir < nr_beta_[iat]; ir++) 
            {
                double x = beta_radial_grid[iat][ir];
                beta_rf(idxrf, iat)[ir] = atom_type->uspp().beta_radial_functions(ir, idxrf) / mask(x, Rmask);
            }
            beta_rf(idxrf, iat).interpolate();
        }
    }
    
    splindex<block> spl_nq(nq, comm_.size(), comm_.rank());
    #pragma omp parallel
    {
        mdarray<Spline<double>, 2> jl(unit_cell_->lmax_beta() + 1, unit_cell_->num_atom_types());
        for (int iat = 0; iat < unit_cell_->num_atom_types(); iat++)
        {
            for (int l = 0; l <= unit_cell_->lmax_beta(); l++) jl(l, iat) = Spline<double>(beta_radial_grid[iat]);
        }

        #pragma omp for
        for (int iq_loc = 0; iq_loc < (int)spl_nq.local_size(); iq_loc++)
        {
            int iq = (int)spl_nq[iq_loc];
            double q = pw_cutoff__ * iq / (nq - 1);

            /* get spherical Bessel functions */
            std::vector<double> v(unit_cell_->lmax_beta() + 1);
            for (int iat = 0; iat < unit_cell_->num_atom_types(); iat++)
            {
                for (int ir = 0; ir < nr_beta_[iat]; ir++)
                {
                    double qx = beta_radial_grid[iat][ir] * q;
                    gsl_sf_bessel_jl_array(unit_cell_->lmax_beta(), qx, &v[0]);
                    for (int l = 0; l <= unit_cell_->lmax_beta(); l++) jl(l, iat)[ir] = v[l];
                }
                for (int l = 0; l <= unit_cell_->lmax_beta(); l++) jl(l, iat).interpolate();
            }
            
            /* compute radial integrals */
            for (int iat = 0; iat < unit_cell_->num_atom_types(); iat++)
            {
                auto atom_type = unit_cell_->atom_type(iat);
                for (int idxrf = 0; idxrf < atom_type->mt_radial_basis_size(); idxrf++)
                {
                    int l = atom_type->indexr(idxrf).l;
                    int nr = nr_beta_[iat];
                    beta_radial_integrals(idxrf, iat, iq) = Spline<double>::integrate(&jl(l, iat), &beta_rf(idxrf, iat), 1, nr);
                }
            }
        }
    }
    
    int ld = unit_cell_->max_mt_radial_basis_size() * unit_cell_->num_atom_types();
    comm_.allgather(beta_radial_integrals.at<CPU>(), static_cast<int>(ld * spl_nq.global_offset()), 
                    static_cast<int>(ld * spl_nq.local_size()));
    

    
   
   int N = 3000;
    
    
    beta_rf_filtered_ = mdarray<Spline<double>, 2>(unit_cell_->max_mt_radial_basis_size(), unit_cell_->num_atom_types());
    for (int iat = 0; iat < unit_cell_->num_atom_types(); iat++)
    {
        auto atom_type = unit_cell_->atom_type(iat);

        double Rmask = R_beta_[iat] * R_mask_scale_;

        beta_radial_grid[iat] = Radial_grid(pow2_grid, N, 0, Rmask);

        for (int idxrf = 0; idxrf < atom_type->mt_radial_basis_size(); idxrf++)
        {
            beta_rf_filtered_(idxrf, iat) = Spline<double>(beta_radial_grid[iat]);
        }
    }
    
    mdarray<Spline<double>, 2> jl(unit_cell_->lmax_beta() + 1, unit_cell_->num_atom_types());
    for (int iat = 0; iat < unit_cell_->num_atom_types(); iat++)
    {
        for (int l = 0; l <= unit_cell_->lmax_beta(); l++) jl(l, iat) = Spline<double>(beta_radial_grid[iat]);
    }

    for (int iq_loc = 0; iq_loc < (int)spl_nq.local_size(); iq_loc++)
    {
        int iq = (int)spl_nq[iq_loc];
        double q = pw_cutoff__ * iq / (nq - 1);
        
        double w = 1; //(q < qcut) ? 1 : std::exp(-b * std::pow(q / qcut - 1, 2));

        /* get spherical Bessel functions */
        std::vector<double> v(unit_cell_->lmax_beta() + 1);
        for (int iat = 0; iat < unit_cell_->num_atom_types(); iat++)
        {
            for (int ir = 0; ir < N; ir++)
            {
                double qx = beta_radial_grid[iat][ir] * q;
                gsl_sf_bessel_jl_array(unit_cell_->lmax_beta(), qx, &v[0]);
                for (int l = 0; l <= unit_cell_->lmax_beta(); l++) jl(l, iat)[ir] = v[l];
            }
        }
        
        for (int iat = 0; iat < unit_cell_->num_atom_types(); iat++)
        {
            //double Rmask = R_beta_[iat] * R_mask_scale_;
            auto atom_type = unit_cell_->atom_type(iat);
            #pragma omp parallel for
            for (int idxrf = 0; idxrf < atom_type->mt_radial_basis_size(); idxrf++)
            {
                int l = atom_type->indexr(idxrf).l;
                double p = (2.0 / pi) * q * q * beta_radial_integrals(idxrf, iat, iq) * pw_cutoff__ / (nq - 1) * w;
                for (int ir = 0; ir < N; ir++)
                {
                    //double x = beta_radial_grid[iat][ir];
                    beta_rf_filtered_(idxrf, iat)[ir] += p * jl(l, iat)[ir];// * mask(x, Rmask);
                }
            }
        }
    }

    FILE* fout = fopen("beta_q.dat", "w");
    for (int iq = 0; iq < nq; iq++)
    {
        //double q = pw_cutoff__ * iq / (nq - 1);
        double w = 1; //(q < qcut) ? 1 : std::exp(-b * std::pow(q / qcut - 1, 2));
        fprintf(fout, "%i %18.12f\n", iq, beta_radial_integrals(0, 0, iq) * w);
    }
    fclose(fout);
    
    for (int iat = 0; iat < unit_cell_->num_atom_types(); iat++)
    {
        auto atom_type = unit_cell_->atom_type(iat);
        for (int idxrf = 0; idxrf < atom_type->mt_radial_basis_size(); idxrf++)
            beta_rf_filtered_(idxrf, iat).interpolate();
    }

    for (int iat = 0; iat < unit_cell_->num_atom_types(); iat++)
    {
        std::stringstream s;
        s << "beta_rf_" << iat << ".dat";
        FILE* fout = fopen(s.str().c_str(), "w");
        for (int idxrf = 0; idxrf < unit_cell_->atom_type(iat)->uspp().num_beta_radial_functions; idxrf++)
        {
            for (int ir = 0; ir < N; ir++)
            {
                fprintf(fout, "%18.12f %18.12f\n", beta_radial_grid[iat][ir], beta_rf_filtered_(idxrf, iat)[ir]);
            }
            fprintf(fout, "\n");
        }
        fclose(fout);
    }

    //STOP();

}

void Real_space_prj::filter_radial_functions_v2(double pw_cutoff__)
{
    int nq0 = 40;
    double dq = pw_cutoff__ / (nq0 - 1);

    int nqmax = 4 * nq0;
    //double qmax = dq * (nqmax - 1);

    int nq1 = nqmax - nq0;
    
    mdarray<double, 3> beta_radial_integrals(unit_cell_->max_mt_radial_basis_size(), unit_cell_->num_atom_types(), nq0);

    mdarray<double, 3> beta_radial_integrals_optimized(unit_cell_->max_mt_radial_basis_size(), unit_cell_->num_atom_types(), nqmax);

    /* interpolate beta radial functions */
    mdarray<Spline<double>, 2> beta_rf(unit_cell_->max_mt_radial_basis_size(), unit_cell_->num_atom_types());
    for (int iat = 0; iat < unit_cell_->num_atom_types(); iat++)
    {
        auto atom_type = unit_cell_->atom_type(iat);

        /* create radial grid for all beta-projectors */
        auto beta_radial_grid = unit_cell_->atom_type(iat)->radial_grid().segment(nr_beta_[iat]);

        for (int idxrf = 0; idxrf < atom_type->mt_radial_basis_size(); idxrf++)
        {
            beta_rf(idxrf, iat) = Spline<double>(beta_radial_grid);
            for (int ir = 0; ir < nr_beta_[iat]; ir++) 
            {
                beta_rf(idxrf, iat)[ir] = atom_type->uspp().beta_radial_functions(ir, idxrf);
            }
            beta_rf(idxrf, iat).interpolate();
        }
    }
    
    splindex<block> spl_nq(nq0, comm_.size(), comm_.rank());
    #pragma omp parallel
    {
        mdarray<Spline<double>, 2> jl(unit_cell_->lmax_beta() + 1, unit_cell_->num_atom_types());
        for (int iat = 0; iat < unit_cell_->num_atom_types(); iat++)
        {
            for (int l = 0; l <= unit_cell_->lmax_beta(); l++) jl(l, iat) = Spline<double>(beta_rf(0, iat).radial_grid());
        }

        #pragma omp for
        for (int iq_loc = 0; iq_loc < (int)spl_nq.local_size(); iq_loc++)
        {
            int iq = (int)spl_nq[iq_loc];
            double q = iq * dq;

            /* get spherical Bessel functions */
            std::vector<double> v(unit_cell_->lmax_beta() + 1);
            for (int iat = 0; iat < unit_cell_->num_atom_types(); iat++)
            {
                for (int ir = 0; ir < nr_beta_[iat]; ir++)
                {
                    double qx = beta_rf(0, iat).x(ir) * q;
                    gsl_sf_bessel_jl_array(unit_cell_->lmax_beta(), qx, &v[0]);
                    for (int l = 0; l <= unit_cell_->lmax_beta(); l++) jl(l, iat)[ir] = v[l];
                }
                for (int l = 0; l <= unit_cell_->lmax_beta(); l++) jl(l, iat).interpolate();
            }
            
            /* compute radial integrals */
            for (int iat = 0; iat < unit_cell_->num_atom_types(); iat++)
            {
                auto atom_type = unit_cell_->atom_type(iat);
                for (int idxrf = 0; idxrf < atom_type->mt_radial_basis_size(); idxrf++)
                {
                    int l = atom_type->indexr(idxrf).l;
                    int nr = nr_beta_[iat];
                    beta_radial_integrals(idxrf, iat, iq) = Spline<double>::integrate(&jl(l, iat), &beta_rf(idxrf, iat), 1, nr);
                }
            }
        }
    }
    
    int ld = unit_cell_->max_mt_radial_basis_size() * unit_cell_->num_atom_types();
    comm_.allgather(beta_radial_integrals.at<CPU>(), static_cast<int>(ld * spl_nq.global_offset()), 
                    static_cast<int>(ld * spl_nq.local_size()));

    

    int N = 3000;

    mdarray<Spline<double>, 2> jl(unit_cell_->lmax_beta() + 1, nqmax);
    auto jl_radial_grid = Radial_grid(linear_grid, N, 0, 100);
    std::vector<double> v(unit_cell_->lmax_beta() + 1);
    for (int iq = 0; iq < nqmax; iq++)
    {
        for (int l = 0; l <= unit_cell_->lmax_beta(); l++) jl(l, iq) = Spline<double>(jl_radial_grid);
        for (int ir = 0; ir < N; ir++)
        {
            double qx = jl_radial_grid[ir] * iq * dq;
            gsl_sf_bessel_jl_array(unit_cell_->lmax_beta(), qx, &v[0]);
            for (int l = 0; l <= unit_cell_->lmax_beta(); l++) jl(l, iq)[ir] = v[l];
        }
        for (int l = 0; l <= unit_cell_->lmax_beta(); l++) jl(l, iq).interpolate();
    }




    for (int iat = 0; iat < unit_cell_->num_atom_types(); iat++)
    {
        auto atom_type = unit_cell_->atom_type(iat);

        double Rmask = R_beta_[iat] * R_mask_scale_;
        int nr = -1;
        for (int ir = 0; ir < N; ir++)
        {
            if (jl_radial_grid[ir] >= Rmask) 
            {
                nr = ir;
                break;
            }
        }


        mdarray<double, 3> M01(nq0, nq1, unit_cell_->lmax_beta() + 1);
        mdarray<double, 3> M11(nq1, nq1, unit_cell_->lmax_beta() + 1);
        for (int l = 0; l <= unit_cell_->lmax_beta(); l++)
        {
            for (int jq = 0; jq < nq1; jq++)
            {
                for (int iq = 0; iq < nq0; iq++)
                {
                    M01(iq, jq, l) = (Spline<double>::integrate(&jl(l, iq), &jl(l, nq0 + jq), 0) -
                                      Spline<double>::integrate(&jl(l, iq), &jl(l, nq0 + jq), 0, nr)) * 
                                      std::pow(dq * iq, 2) * std::pow(dq * (jq + nq0), 2) * dq * dq * std::pow(2.0 / pi, 2);
                }
                for (int iq = 0; iq < nq1; iq++)
                {
                    M11(iq, jq, l) = (Spline<double>::integrate(&jl(l, nq0 + iq), &jl(l, nq0 + jq), 0) -
                                      Spline<double>::integrate(&jl(l, nq0 + iq), &jl(l, nq0 + jq), 0, nr)) * 
                                      std::pow(dq * (iq + nq0), 2) * std::pow(dq * (jq + nq0), 2) * dq * dq * std::pow(2.0 / pi, 2);
                    std::cout << "M11=" << M11(iq,jq,l) << std::endl;
                }
            }
            matrix<double> tmp(&M11(0, 0, l), nq1, nq1);
            linalg<CPU>::syinv(nq1, tmp);
            /* restore lower triangular part */
            for (int j1 = 0; j1 < nq1; j1++)
            {
                for (int j2 = 0; j2 < j1; j2++)
                {
                    M11(j1, j2, l) = M11(j2, j1);
                    std::cout << "M11_inv=" << M11(j1,j2,l) << std::endl;
                }
            }
        }

        for (int idxrf = 0; idxrf < atom_type->mt_radial_basis_size(); idxrf++)
        {
            int l = atom_type->indexr(idxrf).l;
            std::vector<double> v1(nq1, 0);
            std::vector<double> f1(nq1, 0);
            for (int iq = 0; iq < nq1; iq++)
            {
                for (int jq = 0; jq < nq0; jq++) v1[iq] += M01(jq, iq, l) * beta_radial_integrals(idxrf, iat, jq);
                std::cout << "v1[jq]="<<v1[iq] << std::endl;
            }
            for (int iq = 0; iq < nq1; iq++)
            {
                for (int jq = 0; jq < nq1; jq++) f1[iq] += M11(iq, jq, l) * v1[jq];
            }
            for (int iq = 0; iq < nq0; iq++) beta_radial_integrals_optimized(idxrf, iat, iq) = beta_radial_integrals(idxrf, iat, iq);
            for (int iq = 0; iq < nq1; iq++) beta_radial_integrals_optimized(idxrf, iat, nq0 + iq) = -f1[iq];
        }
    }


    

    
   
    
    
    beta_rf_filtered_ = mdarray<Spline<double>, 2>(unit_cell_->max_mt_radial_basis_size(), unit_cell_->num_atom_types());
    for (int iat = 0; iat < unit_cell_->num_atom_types(); iat++)
    {
        auto atom_type = unit_cell_->atom_type(iat);

        double Rmask = R_beta_[iat] * R_mask_scale_;

        auto beta_radial_grid = Radial_grid(pow2_grid, N, 0, Rmask);

        for (int idxrf = 0; idxrf < atom_type->mt_radial_basis_size(); idxrf++)
        {
            beta_rf_filtered_(idxrf, iat) = Spline<double>(beta_radial_grid);
        }
    }
    
    mdarray<Spline<double>, 2> sf_bessel_jl(unit_cell_->lmax_beta() + 1, unit_cell_->num_atom_types());
    for (int iat = 0; iat < unit_cell_->num_atom_types(); iat++)
    {
        for (int l = 0; l <= unit_cell_->lmax_beta(); l++) sf_bessel_jl(l, iat) = Spline<double>(beta_rf_filtered_(0, iat).radial_grid());
    }
    
    spl_nq = splindex<block>(nqmax, comm_.size(), comm_.rank());

    for (int iq_loc = 0; iq_loc < (int)spl_nq.local_size(); iq_loc++)
    {
        int iq = (int)spl_nq[iq_loc];
        double q = iq * dq;
        
        /* get spherical Bessel functions */
        std::vector<double> v(unit_cell_->lmax_beta() + 1);
        for (int iat = 0; iat < unit_cell_->num_atom_types(); iat++)
        {
            for (int ir = 0; ir < N; ir++)
            {
                double qx = beta_rf_filtered_(0, iat).x(ir) * q;
                gsl_sf_bessel_jl_array(unit_cell_->lmax_beta(), qx, &v[0]);
                for (int l = 0; l <= unit_cell_->lmax_beta(); l++) sf_bessel_jl(l, iat)[ir] = v[l];
            }
        }
        
        for (int iat = 0; iat < unit_cell_->num_atom_types(); iat++)
        {
            auto atom_type = unit_cell_->atom_type(iat);
            #pragma omp parallel for
            for (int idxrf = 0; idxrf < atom_type->mt_radial_basis_size(); idxrf++)
            {
                int l = atom_type->indexr(idxrf).l;
                double p = (2.0 / pi) * q * q * beta_radial_integrals_optimized(idxrf, iat, iq) * dq;
                for (int ir = 0; ir < N; ir++)
                {
                    beta_rf_filtered_(idxrf, iat)[ir] += p * sf_bessel_jl(l, iat)[ir];
                }
            }
        }
    }

    for (int iat = 0; iat < unit_cell_->num_atom_types(); iat++)
    {
        std::stringstream s;
        s << "beta_q_" << iat << ".dat";
        FILE* fout = fopen(s.str().c_str(), "w");
        for (int idxrf = 0; idxrf < unit_cell_->atom_type(iat)->uspp().num_beta_radial_functions; idxrf++)
        {
            for (int iq = 0; iq < nqmax; iq++)
            {
                fprintf(fout, "%i %18.12f\n", iq, beta_radial_integrals_optimized(idxrf, iat, iq));
            }
            fprintf(fout, "\n");
        }
        fclose(fout);
    }
    
    for (int iat = 0; iat < unit_cell_->num_atom_types(); iat++)
    {
        auto atom_type = unit_cell_->atom_type(iat);
        for (int idxrf = 0; idxrf < atom_type->mt_radial_basis_size(); idxrf++)
            beta_rf_filtered_(idxrf, iat).interpolate();
    }

    for (int iat = 0; iat < unit_cell_->num_atom_types(); iat++)
    {
        std::stringstream s;
        s << "beta_rf_" << iat << ".dat";
        FILE* fout = fopen(s.str().c_str(), "w");
        for (int idxrf = 0; idxrf < unit_cell_->atom_type(iat)->uspp().num_beta_radial_functions; idxrf++)
        {
            for (int ir = 0; ir < N; ir++)
            {
                fprintf(fout, "%18.12f %18.12f\n", beta_rf_filtered_(idxrf, iat).x(ir), beta_rf_filtered_(idxrf, iat)[ir]);
            }
            fprintf(fout, "\n");
        }
        fclose(fout);
    }


}

void Real_space_prj::get_beta_R()
{
    R_beta_ = std::vector<double>(unit_cell_->num_atom_types(), 0.0);
    nr_beta_ = std::vector<int>(unit_cell_->num_atom_types(), 0);
    /* get the list of max beta radii */
    for (int iat = 0; iat < unit_cell_->num_atom_types(); iat++)
    {
        for (int idxrf = 0; idxrf < unit_cell_->atom_type(iat)->uspp().num_beta_radial_functions; idxrf++)
        {
            int nr = 0;
            for (int ir = unit_cell_->atom_type(iat)->uspp().num_beta_radial_points[idxrf] - 1; ir >= 0; ir--)
            {
                 if (std::abs(unit_cell_->atom_type(iat)->uspp().beta_radial_functions(ir, idxrf)) > 1e-10)
                 {
                    nr = ir + 1;
                    break;
                 }
            }

            R_beta_[iat] = std::max(R_beta_[iat], unit_cell_->atom_type(iat)->radial_grid(nr - 1));
            //R_beta_[iat] = std::max(R_beta_[iat], rmin);
            nr_beta_[iat] = std::max(nr_beta_[iat], nr);

            if (comm_.rank() == 0)
            {
                std::cout << "iat, idxrf = " << iat << ", " << idxrf 
                          << "   R_beta, N_beta = " << R_beta_[iat] << ", " << nr_beta_[iat] << std::endl;
            }
        }
    }
}

void Real_space_prj::get_beta_grid()
{
    beta_projectors_ = std::vector<beta_real_space_prj_descriptor>(unit_cell_->num_atoms());
    max_num_points_ = 0;
    num_points_ = 0;
    for (int ia = 0; ia < unit_cell_->num_atoms(); ia++)
    {
        int iat = unit_cell_->atom(ia)->type_id();
        double Rmask = R_beta_[iat] * R_mask_scale_;

        auto v0 = unit_cell_->lattice_vector(0);
        auto v1 = unit_cell_->lattice_vector(1);
        auto v2 = unit_cell_->lattice_vector(2);

        if (2 * Rmask > v0.length() || 2 * Rmask > v1.length() || 2 * Rmask > v2.length())
        {
            TERMINATE("unit cell is too smal for real-space projection (beta-sphere overlaps with itself)");
        }

        beta_projectors_[ia].offset_ = num_points_;

        /* loop over 3D array (real space) */
        for (int j0 = 0; j0 < fft_->size(0); j0++)
        {
            for (int j1 = 0; j1 < fft_->size(1); j1++)
            {
                for (int j2 = 0; j2 < fft_->size(2); j2++)
                {
                    /* get real space fractional coordinate */
                    vector3d<double> v0(double(j0) / fft_->size(0), double(j1) / fft_->size(1), double(j2) / fft_->size(2));
                    /* index of real space point */
                    int ir = static_cast<int>(j0 + j1 * fft_->size(0) + j2 * fft_->size(0) * fft_->size(1));

                    bool found = false;

                    for (int t0 = -1; t0 <= 1; t0++)
                    {
                        for (int t1 = -1; t1 <= 1; t1++)
                        {
                            for (int t2 = -1; t2 <= 1; t2++)
                            {
                                vector3d<double> v1 = v0 - (unit_cell_->atom(ia)->position() + vector3d<double>(t0, t1, t2));
                                auto r = unit_cell_->get_cartesian_coordinates(vector3d<double>(v1));
                                if (r.length() <= Rmask)
                                {
                                    if (found) TERMINATE("point was already found");
                                    beta_projectors_[ia].num_points_++;
                                    beta_projectors_[ia].ir_.push_back(ir);
                                    beta_projectors_[ia].r_.push_back(v0);
                                    beta_projectors_[ia].T_.push_back(vector3d<int>(t0, t1, t2));
                                    beta_projectors_[ia].dist_.push_back(r.length());
                                    beta_projectors_[ia].rtp_.push_back(SHT::spherical_coordinates(r));
                                    found = true;
                                }
                            }
                        }
                    }
                }
            }
        }
        num_points_ += beta_projectors_[ia].num_points_;
        max_num_points_ = std::max(max_num_points_, beta_projectors_[ia].num_points_);
    }
    if (comm_.rank() == 0)
    {
        for (int ia = 0; ia < unit_cell_->num_atoms(); ia++)
        {
            int iat = unit_cell_->atom(ia)->type_id();
            printf("atom: %3i,  R_beta: %8.4f, num_points: %5i, estimated num_points: %5i\n", ia, R_beta_[iat],
                   beta_projectors_[ia].num_points_,
                   static_cast<int>(fft_->size() * fourpi * std::pow(R_mask_scale_ * R_beta_[iat], 3) / 3.0 / unit_cell_->omega()));
        }
        printf("sum(num_points): %i\n", num_points_);
    }
}

};
