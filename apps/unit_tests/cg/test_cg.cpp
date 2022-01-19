#include "multi_cg/multi_cg.hpp"

#include <Eigen/Core>

#include <iostream>

using namespace Eigen;

struct BlockVector;

struct BlockVector {
    MatrixXd vec;

    typedef double value_type;

    void block_axpy(std::vector<double> alphas, BlockVector const &X, size_t num) {
        DiagonalMatrix<double,Dynamic,Dynamic> D = Map<VectorXd>(alphas.data(), num).asDiagonal();
        vec.leftCols(num) += X.vec.leftCols(num) * D;
    }

    void block_axpy_scatter(std::vector<double> alphas, BlockVector const &X, std::vector<size_t> ids) {
        for (size_t i = 0; i < ids.size(); ++i) {
            vec.col(ids[i]) += alphas[i] * X.vec.col(i);
        }
    }

    // rhos[i] = dot(X[i], Y[i])
    void block_dot(BlockVector const &Y, std::vector<double> &rhos, size_t num) {
        VectorXd result = (vec.leftCols(num).transpose() * Y.vec.leftCols(num)).diagonal();
        VectorXd::Map(rhos.data(), result.size()) = result;
    }

    // X[:, i] = Z[:, i] + alpha[i] * X[:, i] for i < num_unconverged
    void block_xpby(BlockVector const &Z, std::vector<double> alphas, size_t num) {
        DiagonalMatrix<double,Dynamic,Dynamic> D = Map<VectorXd>(alphas.data(), num).asDiagonal();
        vec.leftCols(num) = Z.vec.leftCols(num) + vec.leftCols(num) * D;
    }

    void copy(BlockVector const &X, size_t num) {
        vec.leftCols(num) = X.vec.leftCols(num);
    }

    void fill(double val) {
        vec.fill(val);
    }

    auto cols() {
        return vec.cols();
    }

    void repack(std::vector<size_t> const &ids) {
        for (size_t i = 0; i < ids.size(); ++i) {
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
struct PosDefMatrixShifted {
    DiagonalMatrix<double, Dynamic, Dynamic> A;
    VectorXd shifts;

    void multiply(double alpha, BlockVector const &u, double beta, BlockVector &v, size_t num) {
        v.vec.leftCols(num) = alpha * A * u.vec.leftCols(num) + alpha * u.vec.leftCols(num) * shifts.head(num).asDiagonal() + beta * v.vec.leftCols(num);
    }

    void repack(std::vector<size_t> const &ids) {
        for (size_t i = 0; i < ids.size(); ++i) {
            auto j = ids[i];
            if (j != i) {
                shifts[i] = shifts[j];
            }
        }
    }
};

struct IdentityPreconditioner {
    void apply(BlockVector &C, BlockVector const &B) {
        C = B;
    }
    void repack(std::vector<size_t> const &ids) {
        // nothing to do;
    }
};

int main() {
    size_t m = 40;
    size_t n = 10;

    auto A_shifts = VectorXd::LinSpaced(n, 1, n);
    auto A_diag = VectorXd::LinSpaced(m, 1, m);

    auto A = PosDefMatrixShifted{
        A_diag.asDiagonal(),
        A_shifts
    };

    auto P = IdentityPreconditioner{};

    auto U = BlockVector{MatrixXd::Zero(m, n)};
    auto C = BlockVector{MatrixXd::Zero(m, n)};
    auto X = BlockVector{MatrixXd::Zero(m, n)};
    auto B = BlockVector{MatrixXd::Random(m, n)};
    auto R = B;

    auto tol = 1e-10;

    auto resnorms = sirius::cg::block_cg(
        A, P,
        X, R, U, C,
        100, tol
    );

    // check the residual norms according to the algorithm
    for (size_t i = 0; i < resnorms.size(); ++i) {
        std::cout << "shift " << i << " needed " << resnorms[i].size() << " iterations " << resnorms[i].back() << "\n";

        if (resnorms[i].back() > tol) {
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
}