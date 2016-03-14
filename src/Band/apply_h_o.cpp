#include "band.h"

namespace sirius {

/** \param [in] phi Input wave-functions [storage: CPU && GPU].
 *  \param [out] hphi Hamiltonian, applied to wave-functions [storage: CPU || GPU].
 *  \param [out] ophi Overlap operator, applied to wave-functions [storage: CPU || GPU].
 */
template <typename T>
void Band::apply_h_o(K_point* kp__, 
                     int ispn__,
                     int N__,
                     int n__,
                     Wave_functions<false>& phi__,
                     Wave_functions<false>& hphi__,
                     Wave_functions<false>& ophi__,
                     Hloc_operator& h_op,
                     D_operator<T>& d_op,
                     Q_operator<T>& q_op)
{
    PROFILE_WITH_TIMER("sirius::Band::apply_h_o");

    /* set initial hphi */
    hphi__.copy_from(phi__, N__, n__);

    #ifdef __GPU
    if (ctx_.processing_unit() == GPU) hphi__.copy_to_host(N__, n__);
    #endif
    /* apply local part of Hamiltonian */
    h_op.apply(ispn__, hphi__, N__, n__);
    #ifdef __GPU
    if (ctx_.processing_unit() == GPU) hphi__.copy_to_device(N__, n__);
    #endif

    #ifdef __PRINT_OBJECT_CHECKSUM
    auto cs1 = mdarray<double_complex, 1>(&phi__(0, N__), kp__->num_gkvec_loc() * n__).checksum();
    auto cs2 = mdarray<double_complex, 1>(&hphi__(0, N__), kp__->num_gkvec_loc() * n__).checksum();
    kp__->comm().allreduce(&cs1, 1);
    kp__->comm().allreduce(&cs2, 1);
    DUMP("checksum(phi): %18.10f %18.10f", cs1.real(), cs1.imag());
    DUMP("checksum(hloc_phi): %18.10f %18.10f", cs2.real(), cs2.imag());
    #endif

    /* set intial ophi */
    ophi__.copy_from(phi__, N__, n__);

    for (int i = 0; i < kp__->beta_projectors().num_beta_chunks(); i++)
    {
        kp__->beta_projectors().generate(i);

        kp__->beta_projectors().inner<T>(i, phi__, N__, n__);

        if (!ctx_.iterative_solver_input_section().real_space_prj_)
        {
            d_op.apply(i, ispn__, hphi__, N__, n__);
            q_op.apply(i, 0, ophi__, N__, n__);
        }
        else
        {
            STOP();
            //add_nl_h_o_rs(kp__, n__, phi, hphi, ophi, packed_mtrx_offset__, d_mtrx_packed__, q_mtrx_packed__, kappa__);
        }
    }
}

template void Band::apply_h_o<double_complex>(K_point* kp__, 
                                              int ispn__,
                                              int N__,
                                              int n__,
                                              Wave_functions<false>& phi__,
                                              Wave_functions<false>& hphi__,
                                              Wave_functions<false>& ophi__,
                                              Hloc_operator& h_op,
                                              D_operator<double_complex>& d_op,
                                              Q_operator<double_complex>& q_op);

template void Band::apply_h_o<double>(K_point* kp__, 
                                      int ispn__,
                                      int N__,
                                      int n__,
                                      Wave_functions<false>& phi__,
                                      Wave_functions<false>& hphi__,
                                      Wave_functions<false>& ophi__,
                                      Hloc_operator& h_op,
                                      D_operator<double>& d_op,
                                      Q_operator<double>& q_op);
};
