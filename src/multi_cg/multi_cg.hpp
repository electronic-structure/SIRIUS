#ifndef MULTICG_
#define MULTICG_

#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

#include <iostream>
#include <iterator>

namespace sirius {
namespace cg {

template <class T>
void repack(std::vector<T> &data, std::vector<size_t> const&ids) {
    for (size_t i = 0; i < ids.size(); ++i) {
        data[i] = data[ids[i]];
    }
}

template<class Matrix, class Prec, class StateVec>
std::vector<std::vector<typename StateVec::value_type>> block_cg(
    Matrix &A, Prec &P, StateVec &X, StateVec &B, StateVec &U, StateVec &C, 
    size_t maxiters = 10, double tol = 1e-3
) {
    auto n = X.cols();

    U.fill(0);

    // Use B effectively as the residual block-vector
    // B = B - A * X
    A.multiply(-1.0, X, 1.0, B, n);

    auto rhos = std::vector<typename StateVec::value_type>(n);
    auto rhos_old = rhos;
    auto sigmas = rhos;
    auto alphas = rhos;

    // When vectors converge we move them to the front, but we can't really do
    // that with X, so we have to keep track of where is what.
    auto ids = std::vector<size_t>(n);
    std::iota(ids.begin(), ids.end(), 0);

    size_t num_unconverged = n;

    auto residual_history = std::vector<std::vector<typename StateVec::value_type>>(n);

    for (size_t iter = 0; iter < maxiters; ++iter) {
        // Check the residual norms

        // active = OneTo(num_unconverged)
        P.apply(C, B);

        rhos_old = rhos;

        C.block_dot(B, rhos, num_unconverged);

        for (size_t i = 0; i < num_unconverged; ++i) {
            residual_history[ids[i]].push_back(std::sqrt(rhos[i]));
        }

        auto not_converged = std::vector<size_t>{};
        for (size_t i = 0; i < num_unconverged; ++i) {
            if (rhos[i] > tol * tol) {
                not_converged.push_back(i);
            }
        }

        num_unconverged = not_converged.size();
        
        if (not_converged.empty()) {
            break;
        }

        // Move everything contiguously to the front,
        // except for X, since that's updated in-place.
        repack(ids, not_converged);
        repack(rhos, not_converged);
        repack(rhos_old, not_converged);

        U.repack(not_converged);
        C.repack(not_converged);
        B.repack(not_converged);

        A.repack(not_converged);
        P.repack(not_converged);

        // # In the first iteration we have U == 0, so no need for an axpy.
        if (iter == 0) {
            U.copy(C, num_unconverged);
        } else {
            for (size_t i = 0; i < num_unconverged; ++i) {
                alphas[i] = rhos[i] / rhos_old[i];
            }

            // U[:, i] = C[:, i] + alpha[i] * U[:, i] for i < num_unconverged
            U.block_xpby(C, alphas, num_unconverged);
        }

        // do the next matrix-vector product
        A.multiply(1.0, U, 0.0, C, num_unconverged);

        // compute the optimal distance for the search direction
        U.block_dot(C, sigmas, num_unconverged);

        // Update the solution and the residual
        for (size_t i = 0; i < num_unconverged; ++i) {
            alphas[i] = rhos[i] / sigmas[i];
        }

        X.block_axpy_scatter(alphas, U, ids);
        
        for (size_t i = 0; i < num_unconverged; ++i) {
            alphas[i] *= -1;
        }

        B.block_axpy(alphas, C, num_unconverged);
    }

    return residual_history;
}
}

}
#endif