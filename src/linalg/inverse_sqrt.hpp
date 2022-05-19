#ifndef __INVERSE_SQRT_HPP__
#define __INVERSE_SQRT_HPP__

#include "SDDK/dmatrix.hpp"
#include "linalg/eigensolver.hpp"
#include "linalg/linalg.hpp"
#include "utils/rte.hpp"

namespace sirius {

/// Compute inverse square root of the matrix.
/** As by-product, return the eigen-vectors and the eigen-values of the matrix. */
template <typename T>
inline auto
inverse_sqrt(dmatrix<T>& A__, int N__)
{
    auto solver = (A__.comm().size() == 1) ? Eigensolver_factory("lapack", nullptr) :
                                             Eigensolver_factory("scalapack", nullptr);

    dmatrix<T> Z;
    dmatrix<T> B;
    if (A__.comm().size() == 1) {
        Z = dmatrix<T>(A__.num_rows(), A__.num_cols());
        B = dmatrix<T>(A__.num_rows(), A__.num_cols());
    } else {
        Z = dmatrix<T>(A__.num_rows(), A__.num_cols(), A__.blacs_grid(), A__.bs_row(), A__.bs_col());
        B = dmatrix<T>(A__.num_rows(), A__.num_cols(), A__.blacs_grid(), A__.bs_row(), A__.bs_col());
    }
    std::vector<real_type<T>> eval(N__);

    if (solver->solve(N__, N__, A__, &eval[0], Z)) {
        RTE_THROW("error in diagonalization");
    }
    for (int i = 0; i < Z.num_cols_local(); i++) {
        int icol = Z.icol(i);
        RTE_ASSERT(eval[icol] > 0);
        auto f = 1.0 / std::sqrt(eval[icol]);
        for (int j = 0; j < Z.num_rows_local(); j++) {
            A__(j, i) = Z(j, i) * f;
        }
    }

    if (A__.comm().size() == 1) {
        linalg(linalg_t::blas).gemm('N', 'C', N__, N__, N__, &linalg_const<T>::one(),
            &A__(0, 0), A__.ld(), &Z(0, 0), Z.ld(), &linalg_const<T>::zero(), &B(0, 0), B.ld());
    } else {
        linalg(linalg_t::scalapack).gemm('N', 'C', N__, N__, N__, &linalg_const<T>::one(),
            A__, 0, 0, Z, 0, 0, &linalg_const<T>::zero(), B, 0, 0);
    }

    return std::make_tuple(std::move(B), std::move(Z), std::move(eval));
}

}

#endif
