namespace sirius
{

class kpoint
{
    private:

        mdarray<double,2> gkvec_;

        std::vector<int> idxg_;

    public:

        kpoint(double* vk)
        {
            if (global.aw_cutoff() > double(global.lmax_apw()))
                error(__FILE__, __LINE__, "aw cutoff is too large for a given lmax");

            double gk_cutoff = global.aw_cutoff() / global.min_mt_radius();

            std::vector< std::pair<double, int> > gkmap;

            // find G-vectors for which |G+k| < cutoff
            for (int ig = 0; ig < global.num_gvec(); ig++)
            {
                double vgk[3];
                for (int x = 0; x < 3; x++)
                    vgk[x] = global.gvec(ig)[x] + vk[x];

                double v[3];
                global.get_coordinates<cartesian,reciprocal>(vgk, v);
                double gklen = vector_length(v);

                if (gklen < gk_cutoff) gkmap.push_back(std::pair<double,int>(gklen, ig));
            }

            std::sort(gkmap.begin(), gkmap.end());

            gkvec_.set_dimensions(3, gkmap.size());
            gkvec_.allocate();

            idxg_.resize(gkmap.size());

            for (int ig = 0; ig < (int)gkmap.size(); ig++)
            {
                idxg_[ig] = gkmap[ig].second;
                for (int x = 0; x < 3; x++)
                    gkvec_(x, ig) = global.gvec(gkmap[ig].second)[x] + vk[x];
            }
        }

        void generate_matching_coefficients()
        {
            Timer t("sirius::kpoint::generate_matching_coefficients");

            std::vector<complex16> ylmgk(global.lmmax_apw());
            
            mdarray<double,2> jl(global.lmax_apw() + 2, 2);
            
            std::vector<complex16> zil(global.lmax_apw() + 1);
            for (int l = 0; l <= global.lmax_apw(); l++)
                zil[l] = pow(zi, l);
       
            // TODO: check if spherical harmonic generation is slowing things: then it can be cached
            // TODO: number of Bessel functions can be considerably decreased (G+k shells, atom types)
            // TOTO[?]: G+k shells
            for (int ig = 0; ig < num_gkvec(); ig++)
            {
                double v[3];
                double vs[3];
                global.get_coordinates<cartesian,reciprocal>(gkvec(ig), v);
                SHT::spherical_coordinates(v, vs);
                SHT::spherical_harmonics(global.lmax_apw(), vs[1], vs[2], &ylmgk[0]);

                for (int ia = 0; ia < global.num_atoms(); ia++)
                {
                    complex16 phase_factor = exp(complex16(0.0, twopi * scalar_product(gkvec(ig), global.atom(ia)->position())));

                    assert(global.atom(ia)->type()->max_aw_order() <= 2);

                    double R = global.atom(ia)->type()->mt_radius();
                    double gkR = vs[0] * R; // |G+k|*R

                    // generate spherical Bessel functions
                    gsl_sf_bessel_jl_array(global.lmax_apw() + 1, gkR, &jl(0, 0));
                    // Bessel function derivative: f_{{n}}^{{\prime}}(z)=-f_{{n+1}}(z)+(n/z)f_{{n}}(z)
                    for (int l = 0; l <= global.lmax_apw(); l++)
                        jl(l, 1) = -jl(l + 1, 0) + (l / R) * jl(l, 0);

                    for (int l = 0; l <= global.lmax_apw(); l++)
                    {
                        int num_aw = global.atom(ia)->type()->aw_descriptor(l).size();

                        complex16 a[2][2];
                        mdarray<complex16,2> b(2, 2 * global.lmax_apw() + 1);

                        for (int order = 0; order < num_aw; order++)
                            for (int order1 = 0; order1 < num_aw; order1++)
                                a[order][order1] = complex16(global.atom(ia)->symmetry_class()->aw_surface_dm(l, order, order1), 0.0);
                        
                        for (int m = -l; m <= l; m++)
                            for (int order = 0; order < num_aw; order++)
                                b(order, m + l) = (fourpi / sqrt(global.omega())) * zil[l] * jl(l, order) * phase_factor * 
                                    conj(ylmgk[lm_by_l_m(l, m)]) * pow(gkR, order); 
                        
                        int info = gesv<complex16>(num_aw, 2 * l + 1, &a[0][0], 2, &b(0, 0), 2);

                        if (info)
                        {
                            std::stringstream s;
                            s << "gtsv returned " << info;
                            error(__FILE__, __LINE__, s);
                        }

                        /*for (int order = 0; order < num_aw; order++)
                            for (int m = -l; m <= l; m++)
                                matching_coefficients(ig, global.atom(ia)->offset_aw() + global.atom(ia)->type().indexb().index_by_l_m_order(l, m, order)) = 
                                    b(order, l + m);*/
                    }
                }
            }
        }

        inline int num_gkvec()
        {
            assert(gkvec_.size(1) == (int)idxg_.size());

            return gkvec_.size(1);
        }

        inline double* gkvec(int ig)
        {
            assert(ig >= 0 && ig < gkvec_.size(1));

            return &gkvec_(0, ig);
        }

};

};

