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

            std::vector< std::pair<double, int> > gklen;

            // find G-vectors for which |G+k| < cutoff
            for (int ig = 0; ig < global.num_gvec(); ig++)
            {
                double vgk[3];
                for (int x = 0; x < 3; x++)
                    vgk[x] = global.gvec(ig)[x] + vk[x];

                double v[3];
                global.get_coordinates<cartesian,reciprocal>(vgk, v);
                double len = vector_length(v);

                if (len < gk_cutoff) gklen.push_back(std::pair<double,int>(len, ig));
            }

            std::sort(gklen.begin(), gklen.end());

            gkvec_.set_dimensions(3, gklen.size());
            gkvec_.allocate();

            idxg_.resize(gklen.size());

            for (int ig = 0; ig < (int)gklen.size(); ig++)
            {
                idxg_[ig] = gklen[ig].second;
                for (int x = 0; x < 3; x++)
                    gkvec_(x, ig) = global.gvec(gklen[ig].second)[x] + vk[x];
            }
        }

        void generate_matching_coefficients()
        {
            Timer t("sirius::kpoint::generate_matching_coefficients");

            std::vector<double> gklen(num_gkvec());
            
            mdarray<complex16,2> ylmgk(NULL, global.lmmax_apw(), num_gkvec());
            ylmgk.allocate();
            
            mdarray<double,3> jl(NULL, global.lmax_apw() + 2, 2, num_gkvec());
            jl.allocate();
            
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
                SHT::spherical_harmonics(global.lmax_apw(), vs[1], vs[2], &ylmgk(0, ig));
                gklen[ig] = vs[0];

                for (int ia = 0; ia < global.num_atoms(); ia++)
                {
                    complex16 phase_factor = exp(complex16(0.0, twopi * scalar_product(gkvec(ig), global.atom(ia)->position())));

                    assert(global.atom(ia)->type()->max_aw_order() <= 2);

                    double R = global.atom(ia)->type()->mt_radius();
                    double t = R * vs[0]; // |G+k|*R

                    // generate spherical Bessel functions
                    gsl_sf_bessel_jl_array(global.lmax_apw() + 1, t, &jl(0, 0, ig));
                    // Bessel function derivative: f_{{n}}^{{\prime}}(z)=-f_{{n+1}}(z)+(n/z)f_{{n}}(z)
                    for (int l = 0; l <= global.lmax_apw(); l++)
                        jl(l, 1, ig) = -jl(l + 1, 0, ig) + (l / R) * jl(l, 0, ig);

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
                                b(order, m + l) = (fourpi / sqrt(global.omega())) * zil[l] * jl(l, order, ig) * phase_factor * 
                                    conj(ylmgk(lm_by_l_m(l, m), ig)) * pow(t, order); 
                        
                        int info = gesv<complex16>(num_aw, 2 * l + 1, &a[0][0], 2, &b(0, 0), 2);

                        if (info)
                        {
                            std::stringstream s;
                            s << "gtsv returned " << info;
                            error(__FILE__, __LINE__, s);
                        }
                    }
                }
            }
        }

        inline int num_gkvec()
        {
            return idxg_.size();
        }

        inline double* gkvec(int ig)
        {
            return &gkvec_(0, ig);
        }

};

};

