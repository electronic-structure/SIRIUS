#ifndef __BETA_PROJECTORS_H__
#define __BETA_PROJECTORS_H__

namespace sirius {

class Beta_projectors
{
    private:

        Communicator const& comm_;

        Unit_cell const& unit_cell_;

        Gvec const& gkvec_;

        int lmax_beta_;

        int max_num_phi_;

        int num_gkvec_loc_;

        /// Total number of beta-projectors among atom types.
        int num_beta_t_;

        /// Phase-factor independent plane-wave coefficients of |beta> functions for atom types.
        matrix<double_complex> beta_gk_t_;

        /// Plane-wave coefficients of |beta> functions for atoms.
        matrix<double_complex> beta_gk_;

        matrix<double_complex> beta_phi_;

        struct beta_chunk
        {
            int num_beta_;
            int num_atoms_;
            mdarray<int, 2> desc_;
            mdarray<double, 2> atom_pos_;
        };

        std::vector<beta_chunk> beta_chunks_;

        /// Generate plane-wave coefficients for beta-projectors of atom types.
        void generate_beta_gk_t()
        {
            /* find shells of G+k vectors */
            std::map<size_t, std::vector<int> > gksh;
            for (int igk_loc = 0; igk_loc < gkvec_.num_gvec(comm_.rank()); igk_loc++)
            {
                int igk = gkvec_.offset_gvec(comm_.rank()) + igk_loc;
                size_t gk_len = static_cast<size_t>(gkvec_.cart_shifted(igk).length() * 1e10);
                if (!gksh.count(gk_len)) gksh[gk_len] = std::vector<int>();
                gksh[gk_len].push_back(igk_loc);
            }
        
            std::vector<std::pair<double, std::vector<int> > > gkvec_shells;
        
            for (auto it = gksh.begin(); it != gksh.end(); it++)
            {
                gkvec_shells.push_back(std::pair<double, std::vector<int> >(static_cast<double>(it->first) * 1e-10, it->second));
            }
            
            /* allocate array */
            beta_gk_t_ = matrix<double_complex>(gkvec_.num_gvec(comm_.rank()), num_beta_t_); 
            
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
                std::vector<double> gkvec_rlm(Utils::lmmax(lmax_beta_));
                std::vector<double> beta_radial_integrals_(unit_cell_.max_mt_radial_basis_size());
                sbessel_pw<double> jl(unit_cell_, lmax_beta_);
                #pragma omp for
                for (size_t ish = 0; ish < gkvec_shells.size(); ish++)
                {
                    /* find spherical bessel function for |G+k|r argument */
                    jl.interpolate(gkvec_shells[ish].first);
                    for (size_t i = 0; i < gkvec_shells[ish].second.size(); i++)
                    {
                        int igk_loc = gkvec_shells[ish].second[i];
                        int igk = gkvec_.offset_gvec(comm_.rank()) + igk_loc;
                        /* vs = {r, theta, phi} */
                        auto vs = SHT::spherical_coordinates(gkvec_.cart_shifted(igk));
                        /* compute real spherical harmonics for G+k vector */
                        SHT::spherical_harmonics(lmax_beta_, vs[1], vs[2], &gkvec_rlm[0]);
        
                        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++)
                        {
                            auto atom_type = unit_cell_.atom_type(iat);
                            for (int idxrf = 0; idxrf < atom_type->mt_radial_basis_size(); idxrf++)
                            {
                                int l = atom_type->indexr(idxrf).l;
                                int nr = atom_type->uspp().num_beta_radial_points[idxrf];
                                /* compute \int j_l(|G+k|r) beta_l(r) r dr */
                                /* remeber that beta(r) are defined as miltiplied by r */
                                beta_radial_integrals_[idxrf] = sirius::inner(jl(l, iat), beta_rf(idxrf, iat), 1, nr);
                            }
        
                            for (int xi = 0; xi < atom_type->mt_basis_size(); xi++)
                            {
                                int l = atom_type->indexb(xi).l;
                                int lm = atom_type->indexb(xi).lm;
                                int idxrf = atom_type->indexb(xi).idxrf;
        
                                double_complex z = std::pow(double_complex(0, -1), l) * fourpi / std::sqrt(unit_cell_.omega());
                                beta_gk_t_(igk_loc, atom_type->offset_lo() + xi) = z * gkvec_rlm[lm] * beta_radial_integrals_[idxrf];
                            }
                        }
                    }
                }
            }
        
            #ifdef __PRINT_OBJECT_CHECKSUM
            auto c1 = beta_gk_t_.checksum();
            DUMP("checksum(beta_gk_t) : %18.10f %18.10f", std::real(c1), std::imag(c1))
            #endif
        }

        void split_in_chunks()
        {
            /* split beta-projectors into chunks */
            int num_atoms_in_chunk = (comm_.size() == 1) ? unit_cell_.num_atoms() : std::min(unit_cell_.num_atoms(), 256);
            int num_beta_chunks = unit_cell_.num_atoms() / num_atoms_in_chunk + std::min(1, unit_cell_.num_atoms() % num_atoms_in_chunk);
            splindex<block> spl_beta_chunks(unit_cell_.num_atoms(), num_beta_chunks, 0);
            beta_chunks_.resize(num_beta_chunks);
            
            for (int ib = 0; ib < num_beta_chunks; ib++)
            {
                /* number of atoms in chunk */
                int na = spl_beta_chunks.local_size(ib);
                beta_chunks_[ib].num_atoms_ = na;
                beta_chunks_[ib].desc_ = mdarray<int, 2>(4, na);
                beta_chunks_[ib].atom_pos_ = mdarray<double, 2>(3, na);

                int num_beta = 0;
    
                for (int i = 0; i < na; i++)
                {
                    int ia = spl_beta_chunks.global_index(i, ib);
                    auto type = unit_cell_.atom(ia)->type();
                    /* atom fractional coordinates */
                    for (int x = 0; x < 3; x++) beta_chunks_[ib].atom_pos_(x, i) = unit_cell_.atom(ia)->position(x);
                    /* number of beta functions for atom */
                    beta_chunks_[ib].desc_(0, i) = type->mt_basis_size();
                    /* offset in beta_gk*/
                    beta_chunks_[ib].desc_(1, i) = num_beta;
                    /* offset in beta_gk_t */
                    beta_chunks_[ib].desc_(2, i) = type->offset_lo();
                    beta_chunks_[ib].desc_(3, i) = ia;
    
                    num_beta += type->mt_basis_size();
                }
                beta_chunks_[ib].num_beta_ = num_beta;

                //if (parameters_.processing_unit() == GPU)
                //{
                //    STOP();
                //    #ifdef __GPU
                //    //beta_chunks_[ib].desc_.allocate_on_device();
                //    //beta_chunks_[ib].desc_.copy_to_device();

                //    //beta_chunks_[ib].atom_pos_.allocate_on_device();
                //    //beta_chunks_[ib].atom_pos_.copy_to_device();
                //    #endif
                //}
            }

            num_beta_t_ = 0;
            for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) num_beta_t_ += unit_cell_.atom_type(iat)->mt_lo_basis_size();
        }


    public:

        Beta_projectors(Communicator const& comm__,
                        Unit_cell const& unit_cell__,
                        Gvec const& gkvec__,
                        int lmax_beta__)
            : comm_(comm__),
              unit_cell_(unit_cell__),
              gkvec_(gkvec__),
              lmax_beta_(lmax_beta__),
              max_num_phi_(0)
        {
            num_gkvec_loc_ = gkvec_.num_gvec(comm_.rank());

            split_in_chunks();

            generate_beta_gk_t();

            beta_gk_ = matrix<double_complex>(num_gkvec_loc_, unit_cell_.mt_basis_size());

            for (int i = 0; i < unit_cell_.mt_basis_size(); i++)
            {
                int ia = unit_cell_.mt_lo_basis_descriptor(i).ia;
                int xi = unit_cell_.mt_lo_basis_descriptor(i).xi;

                auto atom_type = unit_cell_.atom(ia)->type();

                for (int igk_loc = 0; igk_loc < num_gkvec_loc_; igk_loc++)
                {
                    int igk = gkvec_.offset_gvec(comm_.rank()) + igk_loc;

                    double phase = twopi * (gkvec_.gvec_shifted(igk) * unit_cell_.atom(ia)->position());

                    beta_gk_(igk_loc, i) = beta_gk_t_(igk_loc, atom_type->offset_lo() + xi) *
                                           std::exp(double_complex(0.0, -phase));
                }
            }

            #ifdef __PRINT_OBJECT_CHECKSUM
            auto c1 = beta_gk_.checksum();
            DUMP("checksum(beta_gk) : %18.10f %18.10f", c1.real(), c1.imag());
            #endif
        }

        matrix<double_complex>& beta_gk_t()
        {
            return beta_gk_t_;
        }

        matrix<double_complex> const& beta_gk() const
        {
            return beta_gk_;
        }

        matrix<double_complex> const& beta_phi() const
        {
            return beta_phi_;
        }

        Unit_cell const& unit_cell() const
        {
            return unit_cell_;
        }

        inline int num_beta_chunks() const
        {
            return static_cast<int>(beta_chunks_.size());
        }

        inline beta_chunk const& beta_chunk(int idx__) const
        {
            return beta_chunks_[idx__];
        }

        inline int num_gkvec_loc() const
        {
            return num_gkvec_loc_;
        }

        void inner(Wave_functions& phi__, int idx0__, int n__)
        {
            Timer t("sirius::Beta_projectors::inner");
            
            if (n__ > max_num_phi_)
            {
                max_num_phi_ = n__;
                beta_phi_ = matrix<double_complex>(unit_cell_.mt_basis_size(), max_num_phi_);
            }

            int nbeta__ = unit_cell_.mt_basis_size();
  
            /* compute <beta|phi> */
            linalg<CPU>::gemm(2, 0, nbeta__, n__, num_gkvec_loc_, beta_gk_.at<CPU>(), num_gkvec_loc_, 
                              &phi__(0, idx0__), num_gkvec_loc_, beta_phi_.at<CPU>(), beta_phi_.ld());

            comm_.allreduce(beta_phi_.at<CPU>(), (int)beta_phi_.size());

            #ifdef __PRINT_OBJECT_CHECKSUM
            auto c1 = beta_phi_.checksum();
            DUMP("checksum(beta_phi) : %18.10f %18.10f", c1.real(), c1.imag());
            #endif
        }
};

};

#endif
