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

using namespace Eigen;
using namespace sirius;

template <typename T>
struct BlockVector
{
    Matrix<T, Dynamic, Dynamic> vec;

    typedef T value_type;

    typedef Matrix<T, Dynamic, 1> VectorT;

    BlockVector(Matrix<T, Dynamic, Dynamic>&& X)
        : vec(std::move(X))
    {
    }

    // Make it easy to switch between f32 and f64.
    template <typename U>
    BlockVector(BlockVector<U> const& X)
        : vec(X.vec.template cast<T>())
    {
    }

    template <typename U>
    void
    block_add(BlockVector<U> const& X, size_t num)
    {
        vec.leftCols(num) += X.vec.leftCols(num).template cast<T>();
    }

    void
    block_axpy(std::vector<T> alphas, BlockVector const& X, size_t num)
    {
        DiagonalMatrix<T, Dynamic, Dynamic> D = Map<VectorT>(alphas.data(), num).asDiagonal();
        vec.leftCols(num) += X.vec.leftCols(num) * D;
    }

    void
    block_axpy_scatter(std::vector<T> alphas, BlockVector const& X, std::vector<int> ids, size_t num)
    {
        for (size_t i = 0; i < num; ++i) {
            vec.col(ids[i]) += alphas[i] * X.vec.col(i);
        }
    }

    // rhos[i] = dot(X[i], Y[i])
    void
    block_dot(BlockVector const& Y, std::vector<T>& rhos, size_t num)
    {
        VectorT result                           = (vec.leftCols(num).transpose() * Y.vec.leftCols(num)).diagonal();
        VectorT::Map(rhos.data(), result.size()) = result;
    }

    // X[:, i] = Z[:, i] + alpha[i] * X[:, i] for i < num_unconverged
    void
    block_xpby(BlockVector const& Z, std::vector<T> alphas, size_t num)
    {
        DiagonalMatrix<T, Dynamic, Dynamic> D = Map<VectorT>(alphas.data(), num).asDiagonal();
        vec.leftCols(num)                     = Z.vec.leftCols(num) + vec.leftCols(num) * D;
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
template <typename T>
struct PosDefMatrixShifted
{
    DiagonalMatrix<T, Dynamic, Dynamic> A;
    Matrix<T, Dynamic, 1> shifts;

    PosDefMatrixShifted(DiagonalMatrix<T, Dynamic, Dynamic>&& A, Matrix<T, Dynamic, 1>&& shifts)
        : A(std::move(A))
        , shifts(std::move(shifts))
    {
    }

    // Make it easy to switch between f32 and f64.
    template <typename U>
    PosDefMatrixShifted(PosDefMatrixShifted<U> const& mat)
        : A(mat.A.diagonal().template cast<T>().asDiagonal())
        , shifts(mat.shifts.template cast<T>())
    {
    }

    void
    multiply(T alpha, BlockVector<T> const& u, T beta, BlockVector<T>& v, size_t num)
    {
        v.vec.leftCols(num) = alpha * A * u.vec.leftCols(num) +
                              alpha * u.vec.leftCols(num) * shifts.head(num).asDiagonal() + beta * v.vec.leftCols(num);
    }

    void
    repack(std::vector<int> const& ids)
    {
        for (int i = 0; i < static_cast<int>(ids.size()); ++i) {
            auto j = ids[i];

            if (j != i)
                shifts[i] = shifts[j];
        }
    }
};

struct IdentityPreconditioner
{
    memory_t mem;
    mdarray<double, 1> eigvals;
    template <typename T>
    void
    apply(BlockVector<T>& C, BlockVector<T> const& B)
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
main(int argc, char** argv)
{
    // The general idea is to solve Ax = b in mixed precision
    // Define the residual r_k = b - Ax_k
    // and the error e_k := x - x_k
    // which satisfy Ae_k = Ax - Ax_k = b - Ax_k = r_k.
    // So we are gonna solve Ae_k = r_k approximately for e_k, and by definition
    // x = e_k + x_k.
    // So we compute r_k = b - Ax_k in f64, and then solve Ae_k = r_k
    // approximately in f32.
    auto m          = argc >= 2 ? std::stoul(argv[1]) : 1000;
    auto n          = argc >= 3 ? std::stoul(argv[2]) : 20;
    auto outer_iter = argc >= 4 ? std::stoul(argv[3]) : 100;
    auto inner_iter = argc >= 5 ? std::stoul(argv[4]) : 40;

    if (n > m)
        throw std::runtime_error("matrix order should be >= block size");

    // Let's stick to this no-op preconditioner.
    auto P = IdentityPreconditioner{};

    // Setup f64 matrices and vecs
    auto A_hi = PosDefMatrixShifted<double>{VectorXd::LinSpaced(m, 1, m).asDiagonal(), VectorXd::LinSpaced(n, 1, n)};
    auto X_hi = BlockVector<double>{MatrixXd::Zero(m, n)};
    auto B_hi = BlockVector<double>{MatrixXd::Random(m, n)};
    auto R_hi = BlockVector<double>{MatrixXd::Zero(m, n)};
    auto U_hi = BlockVector<double>{MatrixXd::Zero(m, n)};
    auto C_hi = BlockVector<double>{MatrixXd::Zero(m, n)};

    // Setup f32 stuff.
    auto U_lo = BlockVector<float>{MatrixXf::Zero(m, n)};
    auto C_lo = BlockVector<float>{MatrixXf::Zero(m, n)};
    auto E_lo = BlockVector<float>(MatrixXf::Zero(m, n));
    auto R_lo = BlockVector<float>(MatrixXf::Zero(m, n));

    auto tol = 1e-10;

    std::vector<std::vector<float>> all_resnorms(n);

    for (size_t outer = 0; outer < outer_iter; ++outer) {
        // A_lo is mutated during multi_cg, so let's just reinitialize.
        auto A_lo = PosDefMatrixShifted<float>(A_hi);
        R_hi      = B_hi;
        A_hi.multiply(-1.0, X_hi, 1.0, R_hi, n);

        E_lo.zero();
        R_lo               = R_hi;
        auto iter_resnorms = sirius::cg::multi_cg(A_lo, P, E_lo, R_lo, U_lo, C_lo, inner_iter, tol, true);
        X_hi.block_add(E_lo, n);

        // Save all the resnorms
        bool done = true;
        for (size_t i = 0; i < n; ++i) {
            done &= iter_resnorms.residual_history[i].back() <= tol;
            all_resnorms[i].insert(all_resnorms[i].end(), iter_resnorms.residual_history[i].begin(),
                                   iter_resnorms.residual_history[i].end());
        }
        if (done)
            break;
    }

    for (auto r : all_resnorms[0])
        std::cout << r << '\n';
    std::cout << '\n';

    // Compare to a f64-only run.
    X_hi.zero();
    R_hi             = B_hi;
    auto resnorms_64 = sirius::cg::multi_cg(A_hi, P, X_hi, R_hi, U_hi, C_hi, m * n, tol, true);

    for (auto r : resnorms_64.residual_history[0])
        std::cout << r << '\n';
    std::cout << '\n';
}
