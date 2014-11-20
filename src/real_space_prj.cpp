#include "real_space_prj.h"

namespace sirius {

Real_space_prj::Real_space_prj(Reciprocal_lattice* reciprocal_lattice__,
                               double gk_cutoff__,
                               Communicator const& comm__)
    : reciprocal_lattice_(reciprocal_lattice__),
      comm_(comm__)
{
    Timer t("sirius::Real_space_prj::Real_space_prj");

    auto uc = reciprocal_lattice_->unit_cell();

    /* take coarse FFT grid */
    fft_ = reciprocal_lattice_->fft_coarse();

    matrix3d<double> M; // TODO: get directly form from reciprocal_lattice
    for (int l = 0; l < 3; l++)
    {
        for (int x = 0; x < 3; x++) M(x, l) = uc->reciprocal_lattice_vectors(l, x);
    }
  
    fft_->init_gvec(2 * gk_cutoff__, M);
    spl_num_gvec_ = splindex<block>(fft_->num_gvec(), comm_.size(), comm_.rank());

    std::vector<double> R_beta(uc->num_atom_types(), 0.0);
    std::vector<int> nmt_beta(uc->num_atom_types(), 0);
    /* get the list of max beta radii */
    for (int iat = 0; iat < uc->num_atom_types(); iat++)
    {
        for (int idxrf = 0; idxrf < uc->atom_type(iat)->uspp().num_beta_radial_functions; idxrf++)
        {
            int nr = uc->atom_type(iat)->uspp().num_beta_radial_points[idxrf];
            R_beta[iat] = std::max(R_beta[iat], uc->atom_type(iat)->radial_grid(nr - 1));
            nmt_beta[iat] = std::max(nmt_beta[iat], nr);
        }
    }

    auto beta_radial_integrals = generate_beta_radial_integrals(uc, nmt_beta, R_beta);

    auto beta_pw_t = generate_beta_pw_t(uc, beta_radial_integrals);

    beta_projectors_ = std::vector<beta_real_space_prj_descriptor>(uc->num_atoms());
    max_num_points_ = 0;
    for (int ia = 0; ia < uc->num_atoms(); ia++)
    {
        int iat = uc->atom(ia)->type_id();
        //auto atom_type = uc->atom_type(iat);
        double Rmask = R_beta[iat] * 1.2;

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

                    for (int t0 = -1; t0 <= 1; t0++)
                    {
                        for (int t1 = -1; t1 <= 1; t1++)
                        {
                            for (int t2 = -1; t2 <= 1; t2++)
                            {
                                vector3d<double> v1 = v0 - (uc->atom(ia)->position() + vector3d<double>(t0, t1, t2));
                                auto r = uc->get_cartesian_coordinates(vector3d<double>(v1));
                                if (r.length() <= Rmask)
                                {
                                    beta_projectors_[ia].num_points_++;
                                    beta_projectors_[ia].ir_.push_back(ir);
                                    beta_projectors_[ia].r_.push_back(v0);
                                    beta_projectors_[ia].T_.push_back(vector3d<int>(t0, t1, t2));
                                    beta_projectors_[ia].dist_.push_back(r.length());
                                }
                            }
                        }
                    }
                }
            }
        }
        max_num_points_ = std::max(max_num_points_, beta_projectors_[ia].num_points_);
    }

    std::vector<double_complex> beta_pw(fft_->num_gvec());
    for (int ia = 0; ia < uc->num_atoms(); ia++)
    {
        int iat = uc->atom(ia)->type_id();
        auto atom_type = uc->atom_type(iat);
        double Rmask = R_beta[iat] * 1.2;
        
        beta_projectors_[ia].beta_ = mdarray<double_complex, 2>(beta_projectors_[ia].num_points_, atom_type->mt_basis_size());

        for (int xi = 0; xi < atom_type->mt_basis_size(); xi++)
        {
            for (int ig_loc = 0; ig_loc < (int)spl_num_gvec_.local_size(); ig_loc++)
            {
                int ig = (int)spl_num_gvec_[ig_loc];
                double_complex phase_factor = std::exp(double_complex(0.0, twopi * Utils::scalar_product(fft_->gvec(ig), uc->atom(ia)->position())));

                beta_pw[ig] = beta_pw_t(ig_loc, atom_type->offset_lo() + xi) * conj(phase_factor);
            }
            comm_.allgather(&beta_pw[0], (int)spl_num_gvec_.global_offset(), (int)spl_num_gvec_.local_size());

            fft_->input(fft_->num_gvec(), fft_->map(), &beta_pw[0]);
            fft_->transform(1);

            for (int i = 0; i < beta_projectors_[ia].num_points_; i++)
            {
                int ir = beta_projectors_[ia].ir_[i];
                double dist = beta_projectors_[ia].dist_[i];
                beta_projectors_[ia].beta_(i, xi) = fft_->buffer(ir) * mask(dist, Rmask);
            }
        }
    }
}

mdarray<double, 3> Real_space_prj::generate_beta_radial_integrals(Unit_cell* uc__,
                                                                  std::vector<int>& nmt_beta__,
                                                                  std::vector<double>& R_beta__)
{
    Timer t("sirius::Real_space_prj::generate_beta_radial_integrals");

    mdarray<double, 3> beta_radial_integrals(uc__->max_mt_radial_basis_size(), uc__->num_atom_types(), fft_->num_gvec_shells_inner());

    /* interpolate beta radial functions divided by a mask function */
    std::vector<Radial_grid> beta_radial_grid(uc__->num_atom_types());
    mdarray<Spline<double>, 2> beta_rf(uc__->max_mt_radial_basis_size(), uc__->num_atom_types());
    for (int iat = 0; iat < uc__->num_atom_types(); iat++)
    {
        auto atom_type = uc__->atom_type(iat);

        /* create radial grid for all beta-projectors */
        std::vector<double> radial_grid_points(nmt_beta__[iat]);
        for (int ir = 0; ir < nmt_beta__[iat]; ir++) radial_grid_points[ir] = uc__->atom_type(iat)->radial_grid(ir);
        beta_radial_grid[iat] = Radial_grid(nmt_beta__[iat], &radial_grid_points[0]);
        double Rmask = R_beta__[iat] * 1.5;

        for (int idxrf = 0; idxrf < atom_type->mt_radial_basis_size(); idxrf++)
        {
            beta_rf(idxrf, iat) = Spline<double>(beta_radial_grid[iat]);
            for (int ir = 0; ir < nmt_beta__[iat]; ir++) 
            {
                double x = beta_radial_grid[iat][ir];
                beta_rf(idxrf, iat)[ir] = atom_type->uspp().beta_radial_functions(ir, idxrf) / mask(x, Rmask);
            }
            beta_rf(idxrf, iat).interpolate();
        }
    }
    
    splindex<block> spl_gsh(fft_->num_gvec_shells_inner(), comm_.size(), comm_.rank());
    #pragma omp parallel
    {
        mdarray<Spline<double>, 2> jl(uc__->lmax_beta() + 1, uc__->num_atom_types());
        for (int iat = 0; iat < uc__->num_atom_types(); iat++)
        {
            for (int l = 0; l <= uc__->lmax_beta(); l++) jl(l, iat) = Spline<double>(beta_radial_grid[iat]);
        }

        #pragma omp for
        for (int igsh_loc = 0; igsh_loc < (int)spl_gsh.local_size(); igsh_loc++)
        {
            int igsh = (int)spl_gsh[igsh_loc];

            /* get spherical Bessel functions */
            double G = fft_->gvec_shell_len(igsh);
            std::vector<double> v(uc__->lmax_beta() + 1);
            for (int iat = 0; iat < uc__->num_atom_types(); iat++)
            {
                for (int ir = 0; ir < nmt_beta__[iat]; ir++)
                {
                    double x = beta_radial_grid[iat][ir] * G;
                    gsl_sf_bessel_jl_array(uc__->lmax_beta(), x, &v[0]);
                    for (int l = 0; l <= uc__->lmax_beta(); l++) jl(l, iat)[ir] = v[l];
                }
                for (int l = 0; l <= uc__->lmax_beta(); l++) jl(l, iat).interpolate();
            }
            
            /* compute radial integrals */
            for (int iat = 0; iat < uc__->num_atom_types(); iat++)
            {
                auto atom_type = uc__->atom_type(iat);
                for (int idxrf = 0; idxrf < atom_type->mt_radial_basis_size(); idxrf++)
                {
                    int l = atom_type->indexr(idxrf).l;
                    int nr = nmt_beta__[iat];
                    beta_radial_integrals(idxrf, iat, igsh) = Spline<double>::integrate(&jl(l, iat), &beta_rf(idxrf, iat), 1, nr);
                }
            }
        }
    }

    int ld = uc__->max_mt_radial_basis_size() * uc__->num_atom_types();
    comm_.allgather(beta_radial_integrals.ptr(), static_cast<int>(ld * spl_gsh.global_offset()), 
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
        std::vector<double_complex> gvec_ylm(Utils::lmmax(uc__->lmax_beta()));
        #pragma omp for
        for (int ig_loc = 0; ig_loc < (int)spl_num_gvec_.local_size(); ig_loc++)
        {
            int ig = (int)spl_num_gvec_[ig_loc];

            auto rtp = SHT::spherical_coordinates(fft_->gvec_cart(ig));
            SHT::spherical_harmonics(uc__->lmax_beta(), rtp[1], rtp[2], &gvec_ylm[0]);

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
                    beta_pw_t(ig_loc, atom_type->offset_lo() + xi) = z * gvec_ylm[lm] * beta_radial_integrals__(idxrf, iat, igsh);
                }
            }
        }
    }

    return beta_pw_t;
}

};
