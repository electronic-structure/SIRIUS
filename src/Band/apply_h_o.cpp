#include "band.h"

namespace sirius {

/** \param [in] phi Input wave-functions [storage: CPU && GPU].
 *  \param [out] hphi Hamiltonian, applied to wave-functions [storage: CPU || GPU].
 *  \param [out] ophi Overlap operator, applied to wave-functions [storage: CPU || GPU].
 */
void Band::apply_h_o(K_point* kp__, 
                     int N__,
                     int n__,
                     Wave_functions& phi__,
                     Wave_functions& hphi__,
                     Wave_functions& ophi__,
                     mdarray<double_complex, 1>& kappa__,
                     Hloc_operator& h_op,
                     D_operator& d_op,
                     Q_operator& q_op)
{
    PROFILE();

    Timer t("sirius::Band::apply_h_o");

    /* set initial hphi */
    hphi__.copy_from(phi__, N__, n__);
    /* set intial ophi */
    ophi__.copy_from(phi__, N__, n__);
    /* apply local part of Hamiltonian */
    h_op.apply(hphi__, N__, n__);

    #ifdef __GPU
    if (parameters_.processing_unit() == GPU)
    {
        hphi__.copy_to_device(N__, n__);
        ophi__.copy_to_device(N__, n__);
    }
    #endif

    kp__->beta_projectors().inner(phi__, N__, n__);

    if (!kp__->iterative_solver_input_section_.real_space_prj_)
    {
        d_op.apply(hphi__, N__, n__);
        q_op.apply(ophi__, N__, n__);
    }
    else
    {
        STOP();
        //add_nl_h_o_rs(kp__, n__, phi, hphi, ophi, packed_mtrx_offset__, d_mtrx_packed__, q_mtrx_packed__, kappa__);
    }
}

};
