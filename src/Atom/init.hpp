inline void Atom::init(int offset_aw__, int offset_lo__, int offset_mt_coeffs__)
{
    assert(offset_aw__ >= 0);

    offset_aw_        = offset_aw__;
    offset_lo_        = offset_lo__;
    offset_mt_coeffs_ = offset_mt_coeffs__;

    lmax_pot_ = type().parameters().lmax_pot();

    if (type().parameters().full_potential()) {
        int lmmax = Utils::lmmax(lmax_pot_);
        int nrf   = type().indexr().size();

        h_radial_integrals_ = mdarray<double, 3>(lmmax, nrf, nrf);
        h_radial_integrals_.zero();

        b_radial_integrals_ = mdarray<double, 4>(lmmax, nrf, nrf, type().parameters().num_mag_dims());
        b_radial_integrals_.zero();

        occupation_matrix_ = mdarray<double_complex, 4>(16, 16, 2, 2);

        uj_correction_matrix_ = mdarray<double_complex, 4>(16, 16, 2, 2);
    }

    if (!type().parameters().full_potential()) {
        int nbf = type().mt_lo_basis_size();
        d_mtrx_ = mdarray<double, 3>(nbf, nbf, type().parameters().num_mag_dims() + 1);
        d_mtrx_.zero();

        for (int xi2 = 0; xi2 < nbf; xi2++) {
            int lm2    = type().indexb(xi2).lm;
            int idxrf2 = type().indexb(xi2).idxrf;
            for (int xi1 = 0; xi1 < nbf; xi1++) {
                int lm1    = type().indexb(xi1).lm;
                int idxrf1 = type().indexb(xi1).idxrf;
                if (lm1 == lm2) {
                    d_mtrx_(xi1, xi2, 0) = type().pp_desc().d_mtrx_ion(idxrf1, idxrf2);
                }
            }
        }
    }
}
