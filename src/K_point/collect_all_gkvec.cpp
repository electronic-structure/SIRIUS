#include "k_point.h"

namespace sirius {

void K_point::collect_all_gkvec(splindex<block>& spl_phi__, double_complex const* phi_slab__, double_complex* phi_slice__)
{
    LOG_FUNC_BEGIN();
    Timer t("sirius::K_point::collect_all_gkvec");

    std::vector<int> sendcounts(comm_.size());
    std::vector<int> sdispls(comm_.size());
    std::vector<int> recvcounts(comm_.size());
    std::vector<int> rdispls(comm_.size());

    sdispls[0] = 0;
    rdispls[0] = 0;
    for (int rank = 0; rank < comm_.size(); rank++)
    {
        sendcounts[rank] = int(spl_gkvec_.local_size() * spl_phi__.local_size(rank));
        if (rank) sdispls[rank] = sdispls[rank - 1] + sendcounts[rank - 1];

        recvcounts[rank] = int(spl_gkvec_.local_size(rank) * spl_phi__.local_size());
        if (rank) rdispls[rank] = rdispls[rank - 1] + recvcounts[rank - 1];
    }

    std::vector<double_complex> recvbuf(num_gkvec() * spl_phi__.local_size()); // TODO: preallocate buffer

    comm_.alltoall(phi_slab__, &sendcounts[0], &sdispls[0], &recvbuf[0], &recvcounts[0], &rdispls[0]);
    
    for (int rank = 0; rank < comm_.size(); rank++)
    {
        matrix<double_complex> tmp(&recvbuf[rdispls[rank]], spl_gkvec_.local_size(rank), spl_phi__.local_size());
        for (int i = 0; i < (int)spl_phi__.local_size(); i++)
        {
            memcpy(&phi_slice__[i * num_gkvec() + spl_gkvec_.global_offset(rank)], &tmp(0, i),
                   spl_gkvec_.local_size(rank) * sizeof(double_complex));
       }
    }
    LOG_FUNC_END();
}

};
