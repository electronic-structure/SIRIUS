#include "k_point.h"

namespace sirius {

void K_point::generate_beta_gk_t()
{
    Timer t("sirius::K_point::generate_beta_gk_t");
    
    /* find shells of G+k vectors */
    std::vector<std::pair<double, std::vector<int> > > gkvec_shells_;
    
    for (int igk_loc = 0; igk_loc < num_gkvec_loc(); igk_loc++)
    {
        int igk = gklo_basis_descriptors_row_[igk_loc].igk;
        double gk_len = gkvec<cartesian>(igk).length();

        if (gkvec_shells_.empty() || std::abs(gkvec_shells_.back().first - gk_len) > 1e-10) 
            gkvec_shells_.push_back(std::pair<double, std::vector<int> >(gk_len, std::vector<int>()));
        gkvec_shells_.back().second.push_back(igk_loc);
    }
    
    /* allocate array */
    beta_gk_t_ = matrix<double_complex>(num_gkvec_loc(), unit_cell_.num_beta_t()); 
    
    /* interpolate beta radial functions */
    mdarray<Spline<double>, 2> beta_rf(unit_cell_.max_mt_radial_basis_size(), unit_cell_.num_atom_types());
    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++)
    {
        auto atom_type = unit_cell_.atom_type(iat);
        for (int idxrf = 0; idxrf < atom_type->mt_radial_basis_size(); idxrf++)
        {
            int nr = atom_type->uspp().num_beta_radial_points[idxrf];
            beta_rf(idxrf, iat) = Spline<double>(atom_type->radial_grid());
            for (int ir = 0; ir < nr; ir++) 
                beta_rf(idxrf, iat)[ir] = atom_type->uspp().beta_radial_functions(ir, idxrf);
            beta_rf(idxrf, iat).interpolate();
        }
    }
    
    /* compute <G+k|beta> */
    #pragma omp parallel
    {
        std::vector<double> gkvec_rlm(Utils::lmmax(parameters_.lmax_beta()));
        std::vector<double> beta_radial_integrals_(unit_cell_.max_mt_radial_basis_size());
        sbessel_pw<double> jl(unit_cell_, parameters_.lmax_beta());
        #pragma omp for
        for (int ish = 0; ish < (int)gkvec_shells_.size(); ish++)
        {
            /* find spherical bessel function for |G+k|r argument */
            jl.interpolate(gkvec_shells_[ish].first);
            for (int i = 0; i < (int)gkvec_shells_[ish].second.size(); i++)
            {
                int igk_loc = gkvec_shells_[ish].second[i];
                int igk = gklo_basis_descriptors_row_[igk_loc].igk;
                /* vs = {r, theta, phi} */
                auto vs = SHT::spherical_coordinates(gkvec<cartesian>(igk));
                /* compute real spherical harmonics for G+k vector */
                SHT::spherical_harmonics(parameters_.lmax_beta(), vs[1], vs[2], &gkvec_rlm[0]);

                for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++)
                {
                    auto atom_type = unit_cell_.atom_type(iat);
                    for (int idxrf = 0; idxrf < atom_type->mt_radial_basis_size(); idxrf++)
                    {
                        int l = atom_type->indexr(idxrf).l;
                        int nr = atom_type->uspp().num_beta_radial_points[idxrf];
                        /* compute \int j_l(|G+k|r) beta_l(r) r dr */
                        /* remeber that beta(r) are defined as miltiplied by r */
                        beta_radial_integrals_[idxrf] = inner(*jl(l, iat), beta_rf(idxrf, iat), 1, nr);
                    }

                    for (int xi = 0; xi < atom_type->mt_basis_size(); xi++)
                    {
                        int l = atom_type->indexb(xi).l;
                        int lm = atom_type->indexb(xi).lm;
                        int idxrf = atom_type->indexb(xi).idxrf;

                        double_complex z = pow(double_complex(0, -1), l) * fourpi / std::sqrt(unit_cell_.omega());
                        beta_gk_t_(igk_loc, atom_type->offset_lo() + xi) = z * gkvec_rlm[lm] * beta_radial_integrals_[idxrf];
                    }
                }
            }
        }
    }
}

};
