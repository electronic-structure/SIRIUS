/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "multi_cg/multi_cg.hpp"

#include <Eigen/Core>

#include <iostream>
#include <iterator>
#include <iomanip>

using namespace Eigen;
using namespace sirius;

struct BlockVector;

struct BlockVector
{
    MatrixXd vec;

    typedef double value_type;

    void
    block_axpy(std::vector<double> alphas, BlockVector const& X, size_t num)
    {
        DiagonalMatrix<double, Dynamic, Dynamic> D = Map<VectorXd>(alphas.data(), num).asDiagonal();
        vec.leftCols(num) += X.vec.leftCols(num) * D;
    }

    void
    block_axpy_scatter(std::vector<double> alphas, BlockVector const& X, std::vector<int> ids, size_t num)
    {
        for (size_t i = 0; i < num; ++i) {
            vec.col(ids[i]) += alphas[i] * X.vec.col(i);
        }
    }

    // rhos[i] = dot(X[i], Y[i])
    void
    block_dot(BlockVector const& Y, std::vector<double>& rhos, size_t num)
    {
        VectorXd result                           = (vec.leftCols(num).transpose() * Y.vec.leftCols(num)).diagonal();
        VectorXd::Map(rhos.data(), result.size()) = result;
    }

    // X[:, i] = Z[:, i] + alpha[i] * X[:, i] for i < num_unconverged
    void
    block_xpby(BlockVector const& Z, std::vector<double> alphas, size_t num)
    {
        DiagonalMatrix<double, Dynamic, Dynamic> D = Map<VectorXd>(alphas.data(), num).asDiagonal();
        vec.leftCols(num)                          = Z.vec.leftCols(num) + vec.leftCols(num) * D;
    }

    void
    copy(BlockVector const& X, size_t num)
    {
        vec.leftCols(num) = X.vec.leftCols(num);
    }

    void
    zero()
    {
        vec.fill(0);
    }

    auto
    cols()
    {
        return vec.cols();
    }

    void
    repack(std::vector<int> const& ids)
    {
        for (int i = 0; i < static_cast<int>(ids.size()); ++i) {
            auto j = ids[i];
            if (j != i) {
                vec.col(i) = vec.col(j);
            }
        }
    }
};

// This is a linear but special operator A(X)
// producing AX + XD where D_ii = shifts[i] is a diagonal matrix.
// So column-wise it performs (A + shift[i])X[:, i]
// the multiply function basically does a gemv on every column with a different shift
// so alpha * A(X) + beta * Y.
struct PosDefMatrixShifted
{
    DiagonalMatrix<double, Dynamic, Dynamic> A;
    VectorXd shifts;

    void
    multiply(double alpha, BlockVector const& u, double beta, BlockVector& v, size_t num)
    {
        v.vec.leftCols(num) = alpha * A * u.vec.leftCols(num) +
                              alpha * u.vec.leftCols(num) * shifts.head(num).asDiagonal() + beta * v.vec.leftCols(num);
    }

    void
    repack(std::vector<int> const& ids)
    {
        for (int i = 0; i < static_cast<int>(ids.size()); ++i) {
            auto j = ids[i];
            if (j != i) {
                shifts[i] = shifts[j];
            }
        }
    }
};

struct IdentityPreconditioner
{
    memory_t mem;
    mdarray<double, 1> eigvals;
    void
    apply(BlockVector& C, BlockVector const& B)
    {
        C = B;
    }
    void
    repack(std::vector<int> const& ids)
    {
        // nothing to do;
    }
};

int
main()
{
    size_t m = 40;
    size_t n = 10;

    // A is just diagonal(1, 2, ..., m) on the diag
    auto A_diag = VectorXd::LinSpaced(m, 1, m);

    // We solve systems (A + shifts[i]) = rhs[i] for i = 1 .. n.
    // To properly test repacking, make sure both large and small shifts
    // are somewhere in the middle.
    VectorXd A_shifts(n);
    for (size_t i = 0; i < n; ++i)
        A_shifts(i) = (7 * i) % n;

    auto A = PosDefMatrixShifted{A_diag.asDiagonal(), A_shifts};

    auto P = IdentityPreconditioner{};

    auto U = BlockVector{MatrixXd::Zero(m, n)};
    auto C = BlockVector{MatrixXd::Zero(m, n)};
    auto X = BlockVector{MatrixXd::Constant(m, n, 0.5)};
    auto B = BlockVector{MatrixXd::Constant(m, n, 1.0)};
    auto R = B;

    auto tol = 1e-10;

    auto resnorms = sirius::cg::multi_cg(A, P, X, R, U, C, 100, tol, false);

    // check the residual norms according to the algorithm
    for (size_t i = 0; i < resnorms.residual_history.size(); ++i) {
        std::cout << "shift " << i << " needed " << resnorms.residual_history[i].size() << " iterations "
                  << resnorms.residual_history[i].back() << "\n";

        if (resnorms.residual_history[i].back() > tol) {
            return 1;
        }
    }

    // True residual norms might be different! because of rounding errors.
    VectorXd true_resnorms = (A_diag.asDiagonal() * X.vec + X.vec * A_shifts.asDiagonal() - B.vec).colwise().norm();

    for (Eigen::Index i = 0; i < true_resnorms.size(); ++i) {
        std::cout << "true resnorm " << i << ": " << true_resnorms[i] << '\n';
        if (true_resnorms[i] > tol * 100) {
            return 2;
        }
    }

    std::cout << "Convergence history:\n";
    for (size_t i = 0; i < resnorms.residual_history.size(); ++i) {
        std::cout << "resnorms[" << i << "] = ";
        std::copy(resnorms.residual_history[i].begin(), resnorms.residual_history[i].end(),
                  std::ostream_iterator<double>(std::cout << std::setprecision(16) << std::scientific, " "));
        std::cout << '\n';
    }
}
