#include "band.h"

namespace sirius {

void Band::apply_h_local_parallel(K_point* kp__,
                                  std::vector<double> const& effective_potential__,
                                  std::vector<double> const& pw_ekin__,
                                  int nst__,
                                  dmatrix<double_complex>& phi__,
                                  dmatrix<double_complex>& hphi__)
{
    PROFILE();

    Timer t("sirius::Band::apply_h_local_parallel");

    STOP();

//    #ifdef __GPU
//    if (parameters_.processing_unit() == GPU)
//    {
//         ctx_.fft_coarse(0)->allocate_on_device();
//    }
//    #endif
//
//    splindex<block> spl_gkvec(kp__->num_gkvec(), kp__->num_ranks_row(), kp__->rank_row());
//
//    assert(phi__.num_rows_local() == (int)spl_gkvec.local_size());
//    assert(hphi__.num_rows_local() == (int)spl_gkvec.local_size());
//
//    auto a2a = kp__->comm_row().map_alltoall(spl_gkvec.counts(), kp__->gkvec_coarse().counts());
//    std::vector<double_complex> buf(kp__->gkvec_coarse().num_gvec_loc());
//    std::vector<double_complex> htmp(hphi__.num_rows_local());
//    
//    for (int i = 0; i < nst__; i++)
//    {
//        Timer t1("fft|a2a_external");
//        /* redistribute plane-wave coefficients between slabs of FFT buffer */
//        kp__->comm_row().alltoall(&phi__(0, i), &a2a.sendcounts[0], &a2a.sdispls[0], &buf[0],
//                                  &a2a.recvcounts[0], &a2a.rdispls[0]);
//        t1.stop();
//        /* load local part of coefficients into local part of FFT buffer */
//        ctx_.fft_coarse(0)->input(kp__->gkvec_coarse().num_gvec_loc(), kp__->gkvec_coarse().index_map(), &buf[0]);
//        /* transform to real space */
//        ctx_.fft_coarse(0)->transform(1, kp__->gkvec_coarse().z_sticks_coord());
//        /* multiply by effective potential */
//        for (int ir = 0; ir < ctx_.fft_coarse(0)->local_size(); ir++) ctx_.fft_coarse(0)->buffer(ir) *= effective_potential__[ir];
//        /* transform back to reciprocal space */
//        ctx_.fft_coarse(0)->transform(-1, kp__->gkvec_coarse().z_sticks_coord());
//        /* gather pw coefficients in the temporary buffer */
//        ctx_.fft_coarse(0)->output(kp__->gkvec_coarse().num_gvec_loc(), kp__->gkvec_coarse().index_map(), &buf[0]);
//        t1.start();
//        /* redistribute uniformly local sets of coefficients */
//        kp__->comm_row().alltoall(&buf[0], &a2a.recvcounts[0], &a2a.rdispls[0], &htmp[0], &a2a.sendcounts[0], &a2a.sdispls[0]);
//        t1.stop();
//        /* add kinetic energy */
//        for (int igk = 0; igk < (int)spl_gkvec.local_size(); igk++) hphi__(igk, i) = htmp[igk] + phi__(igk, i) * pw_ekin__[spl_gkvec[igk]];
//    }
//
//    #ifdef __GPU
//    if (parameters_.processing_unit() == GPU)
//    {
//         ctx_.fft_coarse(0)->deallocate_on_device();
//    }
//    #endif
}

};
