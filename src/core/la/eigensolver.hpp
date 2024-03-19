/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file eigensolver.hpp
 *
 *  \brief Contains definition of eigensolver factory.
 */

#ifndef __EIGENSOLVER_HPP__
#define __EIGENSOLVER_HPP__

#include "core/la/dmatrix.hpp"

namespace sirius {

namespace la {

/// Type of eigen-value solver.
enum class ev_solver_t
{
    /// LAPACK
    lapack,

    /// ScaLAPACK
    scalapack,

    /// ELPA solver
    elpa,

    /// DLA-Future solver
    dlaf,

    /// MAGMA with CPU pointers
    magma,

    /// MAGMA with GPU pointers
    magma_gpu,

    /// CUDA eigen-solver
    cusolver
};

/// Get type of an eigen solver by name (provided as a string).
inline ev_solver_t
get_ev_solver_t(std::string name__)
{
    std::transform(name__.begin(), name__.end(), name__.begin(), ::tolower);

    static const std::map<std::string, ev_solver_t> map_to_type = {
            {"lapack", ev_solver_t::lapack},       {"scalapack", ev_solver_t::scalapack}, {"elpa1", ev_solver_t::elpa},
            {"elpa2", ev_solver_t::elpa},          {"dlaf", ev_solver_t::dlaf},           {"magma", ev_solver_t::magma},
            {"magma_gpu", ev_solver_t::magma_gpu}, {"cusolver", ev_solver_t::cusolver}};

    if (map_to_type.count(name__) == 0) {
        std::stringstream s;
        s << "wrong label of eigen-solver : " << name__;
        RTE_THROW(s);
    }

    return map_to_type.at(name__);
}

/// Interface to different eigen-solvers.
class Eigensolver
{
  protected:
    /// Type of the eigen-value solver.
    ev_solver_t ev_solver_type_;
    /// Common error message.
    const std::string error_msg_not_implemented = "solver is not implemented";
    /// True if solver is MPI parallel.
    bool is_parallel_{false};
    /// Type of host memory needed for the solver.
    /** Some solvers, for example MAGMA, require host pilnned memory. */
    memory_t host_memory_t_{memory_t::none};
    /// Type of input data memory.
    /** CPU solvers start from host memory, MAGMA can start from host or device memory, cuSolver starts from
     *  device memory. */
    memory_t data_memory_t_{memory_t::none};

  public:
    /// Constructor.
    Eigensolver(ev_solver_t type__, bool is_parallel__, memory_t host_memory_t__, memory_t data_memory_t__)
        : ev_solver_type_(type__)
        , is_parallel_(is_parallel__)
        , host_memory_t_(host_memory_t__)
        , data_memory_t_(data_memory_t__)
    {
    }

    /// Destructor.
    virtual ~Eigensolver()
    {
    }

    /// Solve a standard eigen-value problem for all eigen-pairs.
    virtual int
    solve(ftn_int matrix_size__, dmatrix<double>& A__, double* eval__, dmatrix<double>& Z__)
    {
        RTE_THROW(error_msg_not_implemented);
        return -1;
    }

    /// Solve a standard eigen-value problem for all eigen-pairs.
    virtual int
    solve(ftn_int matrix_size__, dmatrix<std::complex<double>>& A__, double* eval__, dmatrix<std::complex<double>>& Z__)
    {
        RTE_THROW(error_msg_not_implemented);
        return -1;
    }

    /// Solve a standard eigen-value problem for all eigen-pairs.
    virtual int
    solve(ftn_int matrix_size__, dmatrix<float>& A__, float* eval__, dmatrix<float>& Z__)
    {
        RTE_THROW(error_msg_not_implemented);
        return -1;
    }

    /// Solve a standard eigen-value problem for all eigen-pairs.
    virtual int
    solve(ftn_int matrix_size__, dmatrix<std::complex<float>>& A__, float* eval__, dmatrix<std::complex<float>>& Z__)
    {
        RTE_THROW(error_msg_not_implemented);
        return -1;
    }

    /// Solve a standard eigen-value problem of a sub-matrix for N lowest eigen-pairs
    virtual int
    solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<double>& A__, double* eval__, dmatrix<double>& Z__)
    {
        RTE_THROW(error_msg_not_implemented);
        return -1;
    }

    /// Solve a standard eigen-value problem of a sub-matrix for N lowest eigen-pairs.
    virtual int
    solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<std::complex<double>>& A__, double* eval__,
          dmatrix<std::complex<double>>& Z__)
    {
        RTE_THROW(error_msg_not_implemented);
        return -1;
    }

    /// Solve a standard eigen-value problem of a sub-matrix for N lowest eigen-pairs
    virtual int
    solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<float>& A__, float* eval__, dmatrix<float>& Z__)
    {
        RTE_THROW(error_msg_not_implemented);
        return -1;
    }

    /// Solve a standard eigen-value problem of a sub-matrix for N lowest eigen-pairs.
    virtual int
    solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<std::complex<float>>& A__, float* eval__,
          dmatrix<std::complex<float>>& Z__)
    {
        RTE_THROW(error_msg_not_implemented);
        return -1;
    }

    /// Solve a generalized eigen-value problem for all eigen-pairs.
    virtual int
    solve(ftn_int matrix_size__, dmatrix<double>& A__, dmatrix<double>& B__, double* eval__, dmatrix<double>& Z__)
    {
        RTE_THROW(error_msg_not_implemented);
        return -1;
    }

    /// Solve a generalized eigen-value problem for all eigen-pairs.
    virtual int
    solve(ftn_int matrix_size__, dmatrix<std::complex<double>>& A__, dmatrix<std::complex<double>>& B__, double* eval__,
          dmatrix<std::complex<double>>& Z__)
    {
        RTE_THROW(error_msg_not_implemented);
        return -1;
    }

    /// Solve a generalized eigen-value problem for all eigen-pairs.
    virtual int
    solve(ftn_int matrix_size__, dmatrix<float>& A__, dmatrix<float>& B__, float* eval__, dmatrix<float>& Z__)
    {
        RTE_THROW(error_msg_not_implemented);
        return -1;
    }

    /// Solve a generalized eigen-value problem for all eigen-pairs.
    virtual int
    solve(ftn_int matrix_size__, dmatrix<std::complex<float>>& A__, dmatrix<std::complex<float>>& B__, float* eval__,
          dmatrix<std::complex<float>>& Z__)
    {
        RTE_THROW(error_msg_not_implemented);
        return -1;
    }

    /// Solve a generalized eigen-value problem for N lowest eigen-pairs.
    virtual int
    solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<double>& A__, dmatrix<double>& B__, double* eval__,
          dmatrix<double>& Z__)
    {
        RTE_THROW(error_msg_not_implemented);
        return -1;
    }

    /// Solve a generalized eigen-value problem for N lowest eigen-pairs.
    virtual int
    solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<std::complex<double>>& A__, dmatrix<std::complex<double>>& B__,
          double* eval__, dmatrix<std::complex<double>>& Z__)
    {
        RTE_THROW(error_msg_not_implemented);
        return -1;
    }

    /// Solve a generalized eigen-value problem for N lowest eigen-pairs.
    virtual int
    solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<float>& A__, dmatrix<float>& B__, float* eval__,
          dmatrix<float>& Z__)
    {
        RTE_THROW(error_msg_not_implemented);
        return -1;
    }

    /// Solve a generalized eigen-value problem for N lowest eigen-pairs.
    virtual int
    solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<std::complex<float>>& A__, dmatrix<std::complex<float>>& B__,
          float* eval__, dmatrix<std::complex<float>>& Z__)
    {
        RTE_THROW(error_msg_not_implemented);
        return -1;
    }

    /// Parallel or sequential solver.
    bool
    is_parallel() const
    {
        return is_parallel_;
    }

    /// Type of host memory, required by the solver.
    inline auto
    host_memory_t() const
    {
        return host_memory_t_;
    }

    /// Type of input memory for the solver.
    inline auto
    data_memory_t() const
    {
        return data_memory_t_;
    }

    /// Type of eigen-solver.
    inline auto
    type() const
    {
        return ev_solver_type_;
    }
};

std::unique_ptr<Eigensolver>
Eigensolver_factory(std::string name__);

} // namespace la

} // namespace sirius

#endif
