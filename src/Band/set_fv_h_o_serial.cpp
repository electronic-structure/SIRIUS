#include "band.h"

namespace sirius {

/** \param [in] phi Input wave-functions [storage: CPU && GPU].
 *  \param [in] hphi Hamiltonian, applied to wave-functions [storage: CPU || GPU].
 *  \param [in] ophi Overlap operator, applied to wave-functions [storage: CPU || GPU].
 *
 *  Temporary array kappa is expected to have the size num_gkvec * (num_phi + num_bands)
 */
void Band::set_fv_h_o_serial(K_point* kp__,
                             int N__,
                             int n__,
                             Wave_functions& phi__,
                             Wave_functions& hphi__,
                             Wave_functions& ophi__,
                             matrix<double_complex>& h__,
                             matrix<double_complex>& o__,
                             matrix<double_complex>& h_old__,
                             matrix<double_complex>& o_old__,
                             mdarray<double_complex, 1>& kappa__)
{
    PROFILE();

    Timer t("sirius::Band::set_fv_h_o_serial");
    
    assert(n__ != 0);

    /* copy old Hamiltonian and overlap */
    for (int i = 0; i < N__; i++)
    {
        std::memcpy(&h__(0, i), &h_old__(0, i), N__ * sizeof(double_complex));
        std::memcpy(&o__(0, i), &o_old__(0, i), N__ * sizeof(double_complex));
    }

    if (parameters_.processing_unit() == CPU)
    {
        /* <{phi,res}|H|res> */
        phi__.inner(0, N__ + n__, hphi__, N__, n__, &h__(0, N__), h__.ld());
        /* <{phi,res}|O|res> */
        phi__.inner(0, N__ + n__, ophi__, N__, n__, &o__(0, N__), o__.ld());
    }

    if (parameters_.processing_unit() == GPU)
    {
        #ifdef __GPU
        bool economize_gpu_memory = (kappa__.size() != 0);
        if (!economize_gpu_memory)
        {
            linalg<GPU>::gemm(2, 0, N__ + n__, n__, kp__->num_gkvec(), phi__.at<GPU>(0, 0), phi__.ld(),
                              hphi__.at<GPU>(0, N__), hphi__.ld(), h__.at<GPU>(0, N__), h__.ld());
            
            linalg<GPU>::gemm(2, 0, N__ + n__, n__, kp__->num_gkvec(), phi__.at<GPU>(0, 0), phi__.ld(),
                              ophi__.at<GPU>(0, N__), ophi__.ld(), o__.at<GPU>(0, N__), o__.ld());
        }
        else
        {
            /* copy phi to device */
            matrix<double_complex> phi(phi__.at<CPU>(), kappa__.at<GPU>(), kp__->num_gkvec(), N__ + n__);
            phi.copy_to_device();

            /* copy hphi to device */
            matrix<double_complex> hphi(hphi__.at<CPU>(0, N__), kappa__.at<GPU>(phi.size()), kp__->num_gkvec(), n__);
            hphi.copy_to_device();

            linalg<GPU>::gemm(2, 0, N__ + n__, n__, kp__->num_gkvec(), phi.at<GPU>(), phi.ld(),
                              hphi.at<GPU>(), hphi.ld(), h__.at<GPU>(0, N__), h__.ld());
            
            /* copy ophi to device */
            matrix<double_complex> ophi(ophi__.at<CPU>(0, N__), kappa__.at<GPU>(phi.size()), kp__->num_gkvec(), n__);
            ophi.copy_to_device();

            linalg<GPU>::gemm(2, 0, N__ + n__, n__, kp__->num_gkvec(), phi.at<GPU>(), phi.ld(),
                              ophi.at<GPU>(), ophi.ld(), o__.at<GPU>(0, N__), o__.ld());

        }
        cublas_get_matrix(N__ + n__, n__, sizeof(double_complex), h__.at<GPU>(0, N__), h__.ld(), h__.at<CPU>(0, N__), h__.ld());
        cublas_get_matrix(N__ + n__, n__, sizeof(double_complex), o__.at<GPU>(0, N__), o__.ld(), o__.at<CPU>(0, N__), o__.ld());
        #else
        TERMINATE_NO_GPU
        #endif
    }
        
    /* save Hamiltonian and overlap */
    for (int i = N__; i < N__ + n__; i++)
    {
        std::memcpy(&h_old__(0, i), &h__(0, i), (N__ + n__) * sizeof(double_complex));
        std::memcpy(&o_old__(0, i), &o__(0, i), (N__ + n__) * sizeof(double_complex));
    }

    for (int i = 0; i < N__ + n__; i++)
    {
        if (h__(i, i).imag() > 1e-12)
        {
            TERMINATE("wrong diagonal of H");
        }
        if (o__(i, i).imag() > 1e-12)
        {
            std::stringstream s;
            s << "wrong diagonal of O: " << o__(i, i);
            TERMINATE(s);
        }
        h__(i, i) = h__(i, i).real();
        o__(i, i) = o__(i, i).real();
    }
}

};
