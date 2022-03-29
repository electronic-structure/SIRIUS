#ifndef __INVERSE_SQRT_HPP__
#define __INVERSE_SQRT_HPP__

#include "SDDK/dmatrix.hpp"
#include "linalg/eigensolver.hpp"
#include "linalg/linalg.hpp"
#include "utils/rte.hpp"

namespace sirius {

template <typename T>
inline dmatrix<T>
inverse_sqrt(dmatrix<T>& A__, int N__)
{
    auto solver = (A__.blacs_grid().comm().size() == 1) ? Eigensolver_factory("lapack", nullptr) :
                                                          Eigensolver_factory("scalapack", nullptr);

    dmatrix<T> evec(A__.num_rows(), A__.num_cols(), A__.blacs_grid(), A__.bs_row(), A__.bs_col());
    std::vector<real_type<T>> eval(N__);

    if (solver->solve(N__, N__, A__, &eval[0], evec)) {
        RTE_THROW("error in diagonalization");
    }
    for (int i = 0; i < N__; i++) {
        eval[i] = 1.0 / std::sqrt(eval[i]);
    }
    for (int i = 0; i < evec.num_cols_local(); i++) {
        int icol = evec.icol(i);
        for (int j = 0; j < evec.num_rows_local(); j++) {
            A__(j, i) = evec(j, i) * eval[icol];
        }
    }

    dmatrix<T> B(A__.num_rows(), A__.num_cols(), A__.blacs_grid(), A__.bs_row(), A__.bs_col());

    if (A__.blacs_grid().comm().size() == 1) {
        linalg(linalg_t::blas).gemm('N', 'C', N__, N__, N__, &linalg_const<T>::one(),
            &A__(0, 0), A__.ld(), &evec(0, 0), evec.ld(), &linalg_const<T>::zero(), &B(0, 0), B.ld());
    } else {
        linalg(linalg_t::scalapack).gemm('N', 'C', N__, N__, N__, &linalg_const<T>::one(),
            A__, 0, 0, evec, 0, 0, &linalg_const<T>::zero(), B, 0, 0);
    }

    return B;
}

}

#endif
