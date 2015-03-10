#include "band.h"

namespace sirius {

void Band::apply_h_o_fast_parallel_rs(K_point* kp__,
                                      std::vector<double> const& effective_potential__,
                                      std::vector<double> const& pw_ekin__,
                                      int N__,
                                      int n__,
                                      matrix<double_complex>& phi_slice__,
                                      matrix<double_complex>& hphi_slice__,
                                      matrix<double_complex>& ophi_slice__,
                                      matrix<double_complex>& phi_slab__,
                                      matrix<double_complex>& hphi_slab__,
                                      matrix<double_complex>& ophi_slab__,
                                      mdarray<int, 1>& packed_mtrx_offset__,
                                      mdarray<double_complex, 1>& d_mtrx_packed__,
                                      mdarray<double_complex, 1>& q_mtrx_packed__,
                                      mdarray<double_complex, 1>& kappa__)
{
    LOG_FUNC_BEGIN();

    Timer t("sirius::Band::apply_h_o_fast_parallel_rs", kp__->comm());

    splindex<block> spl_phi(n__, kp__->comm().size(), kp__->comm().rank());

    kp__->collect_all_gkvec(spl_phi, &phi_slab__(0, N__), &phi_slice__(0, 0));
    
    if (spl_phi.local_size())
    {
        apply_h_local_slice(kp__, effective_potential__, pw_ekin__, (int)spl_phi.local_size(), phi_slice__, hphi_slice__);
        
        memcpy(&ophi_slice__(0, 0), &phi_slice__(0, 0), spl_phi.local_size() * kp__->num_gkvec() * sizeof(double_complex));

        add_nl_h_o_rs(kp__, (int)spl_phi.local_size(), phi_slice__, hphi_slice__, ophi_slice__, packed_mtrx_offset__,
                      d_mtrx_packed__, q_mtrx_packed__, kappa__);
    }

    kp__->collect_all_bands(spl_phi, &hphi_slice__(0, 0),  &hphi_slab__(0, N__));
    kp__->collect_all_bands(spl_phi, &ophi_slice__(0, 0),  &ophi_slab__(0, N__));

    LOG_FUNC_END();
}

};
