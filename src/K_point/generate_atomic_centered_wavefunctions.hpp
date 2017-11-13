inline void K_point::generate_atomic_centered_wavefunctions(const int num_ao__, Wave_functions &phi)
{
  int lmax{0};
  for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
    auto& atom_type = unit_cell_.atom_type(iat);
    for (auto& wf: atom_type.pp_desc().atomic_pseudo_wfs_) {
      lmax = std::max(lmax, wf.first);
    }
  }
  lmax = std::max(lmax, unit_cell_.lmax());

  if (num_ao__ > 0) {
    mdarray<double, 2> rlm_gk(this->num_gkvec_loc(), Utils::lmmax(lmax));
    mdarray<std::pair<int, double>, 1> idx_gk(this->num_gkvec_loc());
#pragma omp parallel for schedule(static)
    for (int igk_loc = 0; igk_loc < this->num_gkvec_loc(); igk_loc++) {
      int igk = this->idxgk(igk_loc);
      /* vs = {r, theta, phi} */
      auto vs = SHT::spherical_coordinates(this->gkvec().gkvec_cart(igk));
      /* compute real spherical harmonics for G+k vector */
      std::vector<double> rlm(Utils::lmmax(lmax));
      SHT::spherical_harmonics(lmax, vs[1], vs[2], &rlm[0]);
      for (int lm = 0; lm < Utils::lmmax(lmax); lm++) {
        rlm_gk(igk_loc, lm) = rlm[lm];
      }
      int i = static_cast<int>((vs[0] / ctx_.gk_cutoff()) * (ctx_.centered_atm_wfc().orbital(0, 0).num_points() - 1));
      double dgk = vs[0] - ctx_.centered_atm_wfc().orbital(0, 0).radial_grid()[i];
      idx_gk(igk_loc) = std::pair<int, double>(i, dgk);
    }

    /* starting index of atomic orbital block for each atom */
    std::vector<int> idxao;
    int n{0};
    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
      auto& atom_type = unit_cell_.atom(ia).type();
      idxao.push_back(n);
      /* increment index of atomic orbitals */
      for (size_t i = 0; i < atom_type.pp_desc().atomic_pseudo_wfs_.size(); i++) {
        int l = atom_type.pp_desc().atomic_pseudo_wfs_[i].first;
        n += (2 * l + 1);
      }
    }

    //mdarray<double, 3> ri(this->num_gkvec_loc(), unit_cell_.lmax() + 1, unit_cell_.num_atom_types());
    //for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
    //    for (size_t i = 0; i < atom_type.pp_desc().atomic_pseudo_wfs_.size(); i++) {
    //    for (int l = 0; l <= unit_cell_.atom_type(iat).indexr().lmax(); l++) {
    //        #pragma omp parallel for
    //        for (int igk_loc = 0; igk_loc < this->num_gkvec_loc(); igk_loc++) {
    //            ri(igk_loc, l, iat) = rad_int__[iat][l](idx_gk[igk_loc].first, idx_gk[igk_loc].second);
    //        }
    //    }
    //}

#pragma omp parallel for schedule(static)
    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
      double phase = twopi * geometry3d::dot(this->gkvec().vk() , unit_cell_.atom(ia).position());
      double_complex phase_k = std::exp(double_complex(0.0, phase));

      std::vector<double_complex> phase_gk(this->num_gkvec_loc());
      for (int igk_loc = 0; igk_loc < this->num_gkvec_loc(); igk_loc++) {
        int igk = this->idxgk(igk_loc);
        auto G = this->gkvec().gvec(igk);
        phase_gk[igk_loc] = std::conj(ctx_.gvec_phase_factor(G, ia) * phase_k);
      }
      auto& atom_type = unit_cell_.atom(ia).type();
      int n{0};
      for (size_t i = 0; i < atom_type.pp_desc().atomic_pseudo_wfs_.size(); i++) {
        int l = atom_type.pp_desc().atomic_pseudo_wfs_[i].first;
        double_complex z = std::pow(double_complex(0, -1), l) * fourpi / std::sqrt(unit_cell_.omega());
        for (int m = -l; m <= l; m++) {
          int lm = Utils::lm_by_l_m(l, m);
          for (int igk_loc = 0; igk_loc < this->num_gkvec_loc(); igk_loc++) {
            phi.component(0).pw_coeffs().prime(igk_loc, idxao[ia] + n) =
              z * phase_gk[igk_loc] * rlm_gk(igk_loc, lm) * ctx_.centered_atm_wfc().orbital(i, atom_type.id())(idx_gk[igk_loc].first, idx_gk[igk_loc].second);
            //phi.component(0).pw_coeffs().prime(igk_loc, idxao[ia] + n) =
            //    z * phase_gk[igk_loc] * rlm_gk(igk_loc, lm) * ri(igk_loc, l, atom_type.id());
          }
          n++;
        }
      }
    }
  }
}
