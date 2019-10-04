// Copyright (c) 2013-2018 Anton Kozhevnikov, Thomas Schulthess
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that
// the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
//    following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
//    and the following disclaimer in the documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/** \file mixer.hpp
 *
 *   \brief Contains definition and implementation of sirius::Mixer base class.
 */

#ifndef __MIXER_HPP__
#define __MIXER_HPP__

#include <tuple>
#include <functional>
#include <utility>
#include <vector>
#include <limits>
#include <memory>
#include <exception>
#include <cmath>
#include <numeric>

#include "SDDK/communicator.hpp"

namespace sirius {
namespace mixer {

// Describes a function type used for mixing.
template <typename FUNC>
struct FunctionProperties
{
    using type = FUNC;

    FunctionProperties(bool is_local_, std::function<std::size_t(const FUNC&)> local_size_,
                       std::function<double(const FUNC&, const FUNC&)> inner_, std::function<void(double, FUNC&)> scal_,
                       std::function<void(const FUNC&, FUNC&)> copy_,
                       std::function<void(double, const FUNC&, FUNC&)> axpy_)
        : is_local(is_local_)
        , local_size(local_size_)
        , inner(inner_)
        , scal(scal_)
        , copy(copy_)
        , axpy(axpy_)
    {
    }

    // The function is stored fully locally on each MPI rank.
    bool is_local;

    // Size proportional to the local contribution of the inner product.
    std::function<std::size_t(const FUNC&)> local_size;

    // Inner product function. Determines contribution to mixing.
    std::function<double(const FUNC&, const FUNC&)> inner;

    // scaling. x = alpha * x
    std::function<void(double, FUNC&)> scal;

    // copy function. y = x
    std::function<void(const FUNC&, FUNC&)> copy;

    // axpy function. y = alpha * x + y
    std::function<void(double, const FUNC&, FUNC&)> axpy;
};

// Implemenation of templated recursive calls through tuples
namespace mixer_impl {
template <std::size_t FUNC_REVERSE_INDEX, typename... FUNCS>
struct LocalSize
{
    static double apply(bool local, const std::tuple<FunctionProperties<FUNCS>...>& function_prop,
                        const std::tuple<std::unique_ptr<FUNCS>...>& x)
    {
        std::size_t size = 0;
        if (std::get<FUNC_REVERSE_INDEX>(function_prop).is_local == local && std::get<FUNC_REVERSE_INDEX>(x)) {
            size += std::get<FUNC_REVERSE_INDEX>(function_prop).local_size(*std::get<FUNC_REVERSE_INDEX>(x));
        }
        return size + LocalSize<FUNC_REVERSE_INDEX - 1, FUNCS...>::apply(local, function_prop, x);
    }
};

template <typename... FUNCS>
struct LocalSize<0, FUNCS...>
{
    static double apply(bool local, const std::tuple<FunctionProperties<FUNCS>...>& function_prop,
                        const std::tuple<std::unique_ptr<FUNCS>...>& x)
    {
        std::size_t size = 0;
        if (std::get<0>(function_prop).is_local == local && std::get<0>(x)) {
            size += std::get<0>(function_prop).local_size(*std::get<0>(x));
        }
        return size;
    }
};

template <std::size_t FUNC_REVERSE_INDEX, typename... FUNCS>
struct InnerProduct
{
    static double apply(bool local, const std::tuple<FunctionProperties<FUNCS>...>& function_prop,
                        const std::tuple<std::unique_ptr<FUNCS>...>& x, const std::tuple<std::unique_ptr<FUNCS>...>& y)
    {
        double result = 0.0;
        if (std::get<FUNC_REVERSE_INDEX>(function_prop).is_local == local && std::get<FUNC_REVERSE_INDEX>(x) &&
            std::get<FUNC_REVERSE_INDEX>(y)) {
            result += std::get<FUNC_REVERSE_INDEX>(function_prop)
                          .inner(*std::get<FUNC_REVERSE_INDEX>(x), *std::get<FUNC_REVERSE_INDEX>(y));
        }
        return result + InnerProduct<FUNC_REVERSE_INDEX - 1, FUNCS...>::apply(local, function_prop, x, y);
    }
};

template <typename... FUNCS>
struct InnerProduct<0, FUNCS...>
{
    static double apply(bool local, const std::tuple<FunctionProperties<FUNCS>...>& function_prop,
                        const std::tuple<std::unique_ptr<FUNCS>...>& x, const std::tuple<std::unique_ptr<FUNCS>...>& y)
    {
        double result = 0.0;
        if (std::get<0>(function_prop).is_local == local && std::get<0>(x) && std::get<0>(y)) {
            result += std::get<0>(function_prop).inner(*std::get<0>(x), *std::get<0>(y));
        }
        return result;
    }
};

template <std::size_t FUNC_REVERSE_INDEX, typename... FUNCS>
struct Scaling
{
    static void apply(const std::tuple<FunctionProperties<FUNCS>...>& function_prop, double alpha,
                      std::tuple<std::unique_ptr<FUNCS>...>& x)
    {
        if (std::get<FUNC_REVERSE_INDEX>(x)) {
            std::get<FUNC_REVERSE_INDEX>(function_prop).scal(alpha, *std::get<FUNC_REVERSE_INDEX>(x));
        }
        Scaling<FUNC_REVERSE_INDEX - 1, FUNCS...>::apply(function_prop, alpha, x);
    }
};

template <typename... FUNCS>
struct Scaling<0, FUNCS...>
{
    static void apply(const std::tuple<FunctionProperties<FUNCS>...>& function_prop, double alpha,
                      std::tuple<std::unique_ptr<FUNCS>...>& x)
    {
        if (std::get<0>(x)) {
            std::get<0>(function_prop).scal(alpha, *std::get<0>(x));
        }
    }
};

template <std::size_t FUNC_REVERSE_INDEX, typename... FUNCS>
struct Copy
{
    static void apply(const std::tuple<FunctionProperties<FUNCS>...>& function_prop,
                      const std::tuple<std::unique_ptr<FUNCS>...>& x, std::tuple<std::unique_ptr<FUNCS>...>& y)
    {
        if (std::get<FUNC_REVERSE_INDEX>(x) && std::get<FUNC_REVERSE_INDEX>(y)) {
            std::get<FUNC_REVERSE_INDEX>(function_prop)
                .copy(*std::get<FUNC_REVERSE_INDEX>(x), *std::get<FUNC_REVERSE_INDEX>(y));
        }
        Copy<FUNC_REVERSE_INDEX - 1, FUNCS...>::apply(function_prop, x, y);
    }
};

template <typename... FUNCS>
struct Copy<0, FUNCS...>
{
    static void apply(const std::tuple<FunctionProperties<FUNCS>...>& function_prop,
                      const std::tuple<std::unique_ptr<FUNCS>...>& x, std::tuple<std::unique_ptr<FUNCS>...>& y)
    {
        if (std::get<0>(x) && std::get<0>(y)) {
            std::get<0>(function_prop).copy(*std::get<0>(x), *std::get<0>(y));
        }
    }
};

template <std::size_t FUNC_REVERSE_INDEX, typename... FUNCS>
struct Axpy
{
    static void apply(const std::tuple<FunctionProperties<FUNCS>...>& function_prop, double alpha,
                      const std::tuple<std::unique_ptr<FUNCS>...>& x, std::tuple<std::unique_ptr<FUNCS>...>& y)
    {
        if (std::get<FUNC_REVERSE_INDEX>(x) && std::get<FUNC_REVERSE_INDEX>(y)) {
            std::get<FUNC_REVERSE_INDEX>(function_prop)
                .axpy(alpha, *std::get<FUNC_REVERSE_INDEX>(x), *std::get<FUNC_REVERSE_INDEX>(y));
        }
        Axpy<FUNC_REVERSE_INDEX - 1, FUNCS...>::apply(function_prop, alpha, x, y);
    }
};

template <typename... FUNCS>
struct Axpy<0, FUNCS...>
{
    static void apply(const std::tuple<FunctionProperties<FUNCS>...>& function_prop, double alpha,
                      const std::tuple<std::unique_ptr<FUNCS>...>& x, std::tuple<std::unique_ptr<FUNCS>...>& y)
    {
        if (std::get<0>(x) && std::get<0>(y)) {
            std::get<0>(function_prop).axpy(alpha, *std::get<0>(x), *std::get<0>(y));
        }
    }
};

} // namespace mixer_impl

/// Abstract mixer for variadic number of Function objects, which are described by FunctionProperties.
template <typename... FUNCS>
class Mixer
{
  public:
    static_assert(sizeof...(FUNCS) > 0, "At least one function type must be provided");

    static constexpr std::size_t number_of_functions = sizeof...(FUNCS);

    Mixer(std::size_t max_history, sddk::Communicator const& comm, const FunctionProperties<FUNCS>&... function_prop)
        : step_(0)
        , max_history_(max_history)
        , comm_(comm)
        , rmse_history_(max_history)
        , functions_(function_prop...)
        , output_history_(max_history)
        , residual_history_(max_history)
    {
    }

    virtual ~Mixer() = default;

    // Initialize function at given index with given value. A new function object is created with "args" passed to the
    // constructor. Only initialized functions are mixed.
    template <std::size_t FUNC_INDEX, typename... ARGS>
    void initialize_function(const typename std::tuple_element<FUNC_INDEX, std::tuple<FUNCS...>>::type& init_value,
                             ARGS&&... args)
    {
        if (step_ > 0) {
            throw std::runtime_error("Initializing function_prop after mixing not allowed!");
        }
        // NOTE: don't use std::forward for args, because we need them multiple times (don't forward
        // r-value references)
        std::get<FUNC_INDEX>(input_).reset(
            new typename std::tuple_element<FUNC_INDEX, std::tuple<FUNCS...>>::type(args...));
        std::get<FUNC_INDEX>(tmp1_).reset(new
                                          typename std::tuple_element<FUNC_INDEX, std::tuple<FUNCS...>>::type(args...));
        std::get<FUNC_INDEX>(tmp2_).reset(new
                                          typename std::tuple_element<FUNC_INDEX, std::tuple<FUNCS...>>::type(args...));

        for (std::size_t i = 0; i < max_history_; ++i) {
            std::get<FUNC_INDEX>(output_history_[i])
                .reset(new typename std::tuple_element<FUNC_INDEX, std::tuple<FUNCS...>>::type(args...));
            std::get<FUNC_INDEX>(residual_history_[i])
                .reset(new typename std::tuple_element<FUNC_INDEX, std::tuple<FUNCS...>>::type(args...));
        }
        std::get<FUNC_INDEX>(functions_).copy(init_value, *std::get<FUNC_INDEX>(output_history_[0]));
        std::get<FUNC_INDEX>(functions_).copy(init_value, *std::get<FUNC_INDEX>(input_));
    }

    // Set input for next mixing step
    template <std::size_t FUNC_INDEX>
    void set_input(const typename std::tuple_element<FUNC_INDEX, std::tuple<FUNCS...>>::type& input)
    {
        if (std::get<FUNC_INDEX>(input_)) {
            std::get<FUNC_INDEX>(functions_).copy(input, *std::get<FUNC_INDEX>(input_));
        } else {
            throw std::runtime_error("Mixer function not initialized!");
        }
    }

    // Access last generated output. Mixing must have been performed at least once.
    template <std::size_t FUNC_INDEX>
    void get_output(typename std::tuple_element<FUNC_INDEX, std::tuple<FUNCS...>>::type& output)
    {
        const auto idx = idx_hist(step_);
        if (!std::get<FUNC_INDEX>(output_history_[idx])) {
            throw std::runtime_error("Mixer function not initialized!");
        }
        std::get<FUNC_INDEX>(functions_).copy(*std::get<FUNC_INDEX>(output_history_[idx]), output);
    }

    // mixing step. If the mse is below mse_min, no mixing is performed. Returns the root mean square error computed by
    // inner products of residuals.
    double mix(double mse_min)
    {
        this->update_residual();
        this->update_rms();
        double rmse = rmse_history_[idx_hist(step_)];
        if (rmse * rmse < mse_min) {
            return rmse;
        }

        // call mixing implementation
        this->mix_impl();

        ++step_;
        return rmse;
    }

  protected:
    // Mixing implementation
    virtual void mix_impl() = 0;

    // update residual histroy for current step
    void update_residual()
    {
        this->copy(input_, residual_history_[idx_hist(step_)]);
        this->axpy(-1.0, output_history_[idx_hist(step_)], residual_history_[idx_hist(step_)]);
    }

    // update rmse histroy for current step. Residuals must have been updated before.
    void update_rms()
    {
        const auto idx    = idx_hist(step_);
        double rmse       = inner_product(false, residual_history_[idx], residual_history_[idx]);
        double rmse_local = inner_product(true, residual_history_[idx], residual_history_[idx]);

        comm_.allreduce(&rmse, 1);
        rmse += rmse_local;

        auto size                  = this->local_size(false, residual_history_[idx]);
        const auto size_local_only = this->local_size(true, residual_history_[idx]);
        this->comm_.allreduce(&size, 1);
        size += size_local_only;

        rmse = std::sqrt(rmse / size);

        rmse_history_[idx_hist(step_)] = rmse;
    }

    // Storage index of given step
    std::size_t idx_hist(std::size_t step) const
    {
        return step % max_history_;
    }

    double local_size(bool local, const std::tuple<std::unique_ptr<FUNCS>...>& x)
    {
        return mixer_impl::LocalSize<sizeof...(FUNCS) - 1, FUNCS...>::apply(local, functions_, x);
    }

    double inner_product(bool local, const std::tuple<std::unique_ptr<FUNCS>...>& x,
                         const std::tuple<std::unique_ptr<FUNCS>...>& y)
    {
        return mixer_impl::InnerProduct<sizeof...(FUNCS) - 1, FUNCS...>::apply(local, functions_, x, y);
    }

    void scale(double alpha, std::tuple<std::unique_ptr<FUNCS>...>& x)
    {
        mixer_impl::Scaling<sizeof...(FUNCS) - 1, FUNCS...>::apply(functions_, alpha, x);
    }

    void copy(const std::tuple<std::unique_ptr<FUNCS>...>& x, std::tuple<std::unique_ptr<FUNCS>...>& y)
    {
        mixer_impl::Copy<sizeof...(FUNCS) - 1, FUNCS...>::apply(functions_, x, y);
    }

    void axpy(double alpha, const std::tuple<std::unique_ptr<FUNCS>...>& x, std::tuple<std::unique_ptr<FUNCS>...>& y)
    {
        mixer_impl::Axpy<sizeof...(FUNCS) - 1, FUNCS...>::apply(functions_, alpha, x, y);
    }

    // Strictly increasing counter, indicating the number of mixing steps
    std::size_t step_;

    // The maximum history size kept for each function
    std::size_t max_history_;

    /// Base communicator.
    sddk::Communicator const& comm_;

    // Properties, describing the each function type
    std::vector<double> rmse_history_;

    // Properties, describing the each function type
    std::tuple<FunctionProperties<FUNCS>...> functions_;

    // Input storage for next mixing step
    std::tuple<std::unique_ptr<FUNCS>...> input_;

    // The history of generated mixer outputs. The last generated output is at step_.
    std::vector<std::tuple<std::unique_ptr<FUNCS>...>> output_history_;

    // The residual history between input and output
    std::vector<std::tuple<std::unique_ptr<FUNCS>...>> residual_history_;

    // Tempory storage for compuations
    std::tuple<std::unique_ptr<FUNCS>...> tmp1_;
    std::tuple<std::unique_ptr<FUNCS>...> tmp2_;
};
} // namespace mixer
} // namespace sirius

#endif // __MIXER_HPP__
