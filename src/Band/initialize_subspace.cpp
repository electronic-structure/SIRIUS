#include "band.h"

namespace sirius {

void Band::initialize_subspace(K_point* kp__)
{
    int nq = 20;
    int lmax = 4;
    Radial_grid qgrid(linear_grid, nq, 0, ctx_.gk_cutoff());

    /* interpolate <jl(q*x) | wf_n(x) > with splines */
    std::vector< std::vector< Spline<double> > > rad_int(ctx_.unit_cell().num_atom_types());
    
    mdarray<Spherical_Bessel_functions, 2> jl(nq, unit_cell_.num_atom_types());

    for (int iat = 0; iat < ctx_.unit_cell().num_atom_types(); iat++)
    {
        for (int iq = 0; iq < nq; iq++)
            jl(iq, iat) = Spherical_Bessel_functions(lmax, ctx_.unit_cell().atom_type(iat).radial_grid(), qgrid[iq]);

        rad_int[iat].resize(ctx_.unit_cell().atom_type(iat).uspp().l_wf_pseudo_.size());
        for (size_t i = 0; i < ctx_.unit_cell().atom_type(iat).uspp().l_wf_pseudo_.size(); i++)
        {
            rad_int[iat][i] = Spline<double>(qgrid);

            Spline<double> wf(ctx_.unit_cell().atom_type(iat).radial_grid());
            for (int ir = 0; ir < ctx_.unit_cell().atom_type(iat).num_mt_points(); ir++)
                wf[ir] = ctx_.unit_cell().atom_type(iat).uspp().wf_pseudo_(ir, i);
            wf.interpolate();
            
            int l = ctx_.unit_cell().atom_type(iat).uspp().l_wf_pseudo_[i];
            for (int iq = 0; iq < nq; iq++)
            {
                rad_int[iat][i][iq] = sirius::inner(jl(iq, iat)[l], wf, 1);
            }
            rad_int[iat][i].interpolate();
        }
    }

    /* get the total number of atomic-centered orbitals */
    int N = 0;
    for (int iat = 0; iat < ctx_.unit_cell().num_atom_types(); iat++)
    {
        int n = 0;
        for (auto l: ctx_.unit_cell().atom_type(iat).uspp().l_wf_pseudo_)
        {
            n += (2 * l + 1);
        }
        N += ctx_.unit_cell().atom_type(iat).num_atoms() * n;
    }
    printf("number of atomic orbitals: %i\n", N);

    std::vector<double> gkvec_rlm(Utils::lmmax(lmax));

    Wave_functions<false> phi(N, kp__->gkvec(), ctx_.mpi_grid_fft(), CPU);

    for (int igk_loc = 0; igk_loc < kp__->num_gkvec_loc(); igk_loc++)
    {
        int igk = kp__->gkvec().offset_gvec(kp__->comm().rank()) + igk_loc;
        /* vs = {r, theta, phi} */
        auto vs = SHT::spherical_coordinates(kp__->gkvec().cart_shifted(igk));

        /* compute real spherical harmonics for G+k vector */
        SHT::spherical_harmonics(lmax, vs[1], vs[2], &gkvec_rlm[0]);

        int n = 0;
        for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++)
        {
            double phase = twopi * (kp__->gkvec().gvec_shifted(igk) * ctx_.unit_cell().atom(ia).position());
            double_complex phase_factor =  std::exp(double_complex(0.0, -phase));

            auto& atom_type = ctx_.unit_cell().atom(ia).type();
            for (size_t i = 0; i < atom_type.uspp().l_wf_pseudo_.size(); i++)
            {
                int l = atom_type.uspp().l_wf_pseudo_[i];
                for (int m = -l; m <= l; m++)
                {
                    int lm = Utils::lm_by_l_m(l, m);
                    double_complex z = std::pow(double_complex(0, -1), l) * fourpi / std::sqrt(ctx_.unit_cell().omega());
                    phi(igk_loc, n++) = z * phase_factor * gkvec_rlm[lm] * rad_int[atom_type.id()][i](vs[0]);
                }
            }
        }
    }

    for (int ispn = 0; ispn < ctx_.num_spins(); ispn++)
    {
        kp__->spinor_wave_functions<false>(ispn).copy_from(phi, 0, std::min(N, ctx_.num_fv_states()), 0);
    }

    //if (kp__->gkvec().reduced())
    //{
    //    matrix<double> ovlp(N, N);
    //    phi.inner<double>(0, N, phi, 0, N, ovlp, 0, 0);

    //    Utils::write_matrix("ovlp_real.txt", true, ovlp);
    //}
    //else
    //{
    //    matrix<double_complex> ovlp(N, N);
    //    phi.inner<double_complex>(0, N, phi, 0, N, ovlp, 0, 0);
    //    Utils::write_matrix("ovlp_cmplx.txt", true, ovlp);
    //}
}

};
