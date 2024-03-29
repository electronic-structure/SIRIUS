/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file mixer.hpp
 *
 *  \brief Contains definition and implementation of sirius::Mixer base class.
 */

#ifndef __MIXER_HPP__
#define __MIXER_HPP__

#include <tuple>
#include <functional>
#include <utility>
#include <vector>
#include <limits>
#include <memory>
#include <stdexcept>
#include <cmath>
#include <numeric>

namespace sirius {

/// Mixer functions and objects.
namespace mixer {

/// Describes operations on a function type used for mixing.
/** The properties contain functions, which determine the behaviour of a given type during mixing. The inner product
 * function result is used for calculating mixing parameters. If a function should not contribute to generation of
 * mixing parameters, the inner product function should always return 0.
 */
template <typename FUNC>
struct FunctionProperties
{
    using type = FUNC;

    ///
    /**
     *  \param [in]  size_         Function, which returns a measure of size of the (global) function.
     *  \param [in]  inner_        Function, which computes the (global) inner product. This determines the contribution
     * to mixing parameters rmse. \param [in]  scal_         Function, which scales the input (x = alpha * x). \param
     * [in]  copy_         Function, which copies from one object to the other (y = x). \param [in]  axpy_ Function,
     * which scales and adds one object to the other (y = alpha * x + y).
     */
    FunctionProperties(std::function<double(const FUNC&)> size_, std::function<double(const FUNC&, const FUNC&)> inner_,
                       std::function<void(double, FUNC&)> scal_, std::function<void(const FUNC&, FUNC&)> copy_,
                       std::function<void(double, const FUNC&, FUNC&)> axpy_,
                       std::function<void(double, double, FUNC&, FUNC&)> rotate_)
        : size(size_)
        , inner(inner_)
        , scal(scal_)
        , copy(copy_)
        , axpy(axpy_)
        , rotate(rotate_)
    {
    }

    FunctionProperties()
        : size([](const FUNC&) -> double { return 0; })
        , inner([](const FUNC&, const FUNC&) -> double { return 0.0; })
        , scal([](double, FUNC&) -> void {})
        , copy([](const FUNC&, FUNC&) -> void {})
        , axpy([](double, const FUNC&, FUNC&) -> void {})
        , rotate([](double, double, FUNC&, FUNC&) -> void {})
    {
    }

    // Size proportional to the local contribution of the inner product.
    std::function<double(const FUNC&)> size; // TODO: this sounds more like a normalization factor.

    // Inner product function. Determines contribution to mixing.
    std::function<double(const FUNC&, const FUNC&)> inner;

    // scaling. x = alpha * x
    std::function<void(double, FUNC&)> scal;

    // copy function. y = x
    std::function<void(const FUNC&, FUNC&)> copy;

    // axpy function. y = alpha * x + y
    std::function<void(double, const FUNC&, FUNC&)> axpy;

    // rotate function [x y] * [c -s; s c]
    std::function<void(double, double, FUNC&, FUNC&)> rotate;
};

// Implementation of templated recursive calls through tuples
namespace mixer_impl {

/// Compute inner product <x|y> between pairs of functions in tuples and accumulate in the result.
/** This function is used in Broyden mixers to compute inner products of residuals. */
template <std::size_t FUNC_REVERSE_INDEX, bool normalize, typename... FUNCS>
struct InnerProduct
{
    static double
    apply(const std::tuple<FunctionProperties<FUNCS>...>& function_prop, const std::tuple<std::unique_ptr<FUNCS>...>& x,
          const std::tuple<std::unique_ptr<FUNCS>...>& y)
    {
        double result = 0.0;
        if (std::get<FUNC_REVERSE_INDEX>(x) && std::get<FUNC_REVERSE_INDEX>(y)) {
            /* compute inner product */
            auto v = std::get<FUNC_REVERSE_INDEX>(function_prop)
                             .inner(*std::get<FUNC_REVERSE_INDEX>(x), *std::get<FUNC_REVERSE_INDEX>(y));
            /* normalize if necessary */
            if (normalize) {
                auto sx = std::get<FUNC_REVERSE_INDEX>(function_prop).size(*std::get<FUNC_REVERSE_INDEX>(x));
                auto sy = std::get<FUNC_REVERSE_INDEX>(function_prop).size(*std::get<FUNC_REVERSE_INDEX>(y));
                if (sx != sy) {
                    throw std::runtime_error("[sirius::mixer::InnerProduct] sizes of two functions don't match");
                }
                if (sx) {
                    v /= sx;
                } else {
                    v = 0;
                }
            }

            result += v;
        }
        return result + InnerProduct<FUNC_REVERSE_INDEX - 1, normalize, FUNCS...>::apply(function_prop, x, y);
    }
};

template <bool normalize, typename... FUNCS>
struct InnerProduct<0, normalize, FUNCS...>
{
    static double
    apply(const std::tuple<FunctionProperties<FUNCS>...>& function_prop, const std::tuple<std::unique_ptr<FUNCS>...>& x,
          const std::tuple<std::unique_ptr<FUNCS>...>& y)
    {
        if (std::get<0>(x) && std::get<0>(y)) {
            auto v = std::get<0>(function_prop).inner(*std::get<0>(x), *std::get<0>(y));
            if (normalize) {
                auto sx = std::get<0>(function_prop).size(*std::get<0>(x));
                auto sy = std::get<0>(function_prop).size(*std::get<0>(y));
                if (sx != sy) {
                    throw std::runtime_error("[sirius::mixer::InnerProduct] sizes of two functions don't match");
                }
                if (sx) {
                    v /= sx;
                } else {
                    v = 0;
                }
            }
            return v;
        } else {
            return 0;
        }
    }
};

template <std::size_t FUNC_REVERSE_INDEX, typename... FUNCS>
struct Scaling
{
    static void
    apply(const std::tuple<FunctionProperties<FUNCS>...>& function_prop, double alpha,
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
    static void
    apply(const std::tuple<FunctionProperties<FUNCS>...>& function_prop, double alpha,
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
    static void
    apply(const std::tuple<FunctionProperties<FUNCS>...>& function_prop, const std::tuple<std::unique_ptr<FUNCS>...>& x,
          std::tuple<std::unique_ptr<FUNCS>...>& y)
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
    static void
    apply(const std::tuple<FunctionProperties<FUNCS>...>& function_prop, const std::tuple<std::unique_ptr<FUNCS>...>& x,
          std::tuple<std::unique_ptr<FUNCS>...>& y)
    {
        if (std::get<0>(x) && std::get<0>(y)) {
            std::get<0>(function_prop).copy(*std::get<0>(x), *std::get<0>(y));
        }
    }
};

template <std::size_t FUNC_REVERSE_INDEX, typename... FUNCS>
struct Axpy
{
    static void
    apply(const std::tuple<FunctionProperties<FUNCS>...>& function_prop, double alpha,
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
    static void
    apply(const std::tuple<FunctionProperties<FUNCS>...>& function_prop, double alpha,
          const std::tuple<std::unique_ptr<FUNCS>...>& x, std::tuple<std::unique_ptr<FUNCS>...>& y)
    {
        if (std::get<0>(x) && std::get<0>(y)) {
            std::get<0>(function_prop).axpy(alpha, *std::get<0>(x), *std::get<0>(y));
        }
    }
};

template <std::size_t FUNC_REVERSE_INDEX, typename... FUNCS>
struct Rotate
{
    static void
    apply(const std::tuple<FunctionProperties<FUNCS>...>& function_prop, double c, double s,
          std::tuple<std::unique_ptr<FUNCS>...>& x, std::tuple<std::unique_ptr<FUNCS>...>& y)
    {
        if (std::get<FUNC_REVERSE_INDEX>(x) && std::get<FUNC_REVERSE_INDEX>(y)) {
            std::get<FUNC_REVERSE_INDEX>(function_prop)
                    .rotate(c, s, *std::get<FUNC_REVERSE_INDEX>(x), *std::get<FUNC_REVERSE_INDEX>(y));
        }
        Rotate<FUNC_REVERSE_INDEX - 1, FUNCS...>::apply(function_prop, c, s, x, y);
    }
};

template <typename... FUNCS>
struct Rotate<0, FUNCS...>
{
    static void
    apply(const std::tuple<FunctionProperties<FUNCS>...>& function_prop, double c, double s,
          std::tuple<std::unique_ptr<FUNCS>...>& x, std::tuple<std::unique_ptr<FUNCS>...>& y)
    {
        if (std::get<0>(x) && std::get<0>(y)) {
            std::get<0>(function_prop).rotate(c, s, *std::get<0>(x), *std::get<0>(y));
        }
    }
};

} // namespace mixer_impl

/// Abstract mixer for variadic number of Function objects, which are described by FunctionProperties.
/** Can mix variadic number of functions objects, for which operations are defined in FunctionProperties. Only
 *  functions, which are explicitly initialized, are mixed.
 */
template <typename... FUNCS>
class Mixer
{
  public:
    static_assert(sizeof...(FUNCS) > 0, "At least one function type must be provided");

    static constexpr std::size_t number_of_functions = sizeof...(FUNCS);

    /// Construct a mixer. Functions have to initialized individually.
    /** \param [in]  max_history   Maximum number of steps stored, which contribute to the mixing.
     *  \param [in]  commm         Communicator used for exchaning mixing contributions.
     */
    Mixer(std::size_t max_history)
        : step_(0)
        , max_history_(max_history)
        , rmse_history_(max_history)
        , output_history_(max_history)
        , residual_history_(max_history)
    {
    }

    virtual ~Mixer() = default;

    /// Initialize function at given index with given value. A new function object is created with "args" passed to the
    /// constructor. Only initialized functions are mixed.
    /** \param [in]  function_prop   Function properties, which describe operations.
     *  \param [in]  init_value      Initial function value for input / output.
     *  \param [in]  args            Arguments, which are passed to the constructor of function placeholder objects.
     */
    template <std::size_t FUNC_INDEX, typename... ARGS>
    void
    initialize_function(const FunctionProperties<typename std::tuple_element<FUNC_INDEX, std::tuple<FUNCS...>>::type>&
                                function_prop,
                        const typename std::tuple_element<FUNC_INDEX, std::tuple<FUNCS...>>::type& init_value,
                        ARGS&&... args)
    {
        if (step_ > 0) {
            throw std::runtime_error("Initializing function_prop after mixing not allowed!");
        }

        std::get<FUNC_INDEX>(functions_) = function_prop;

        // NOTE: don't use std::forward for args, because we need them multiple times (don't forward
        // r-value references)

        // create function object placeholders with arguments provided
        std::get<FUNC_INDEX>(input_).reset(
                new typename std::tuple_element<FUNC_INDEX, std::tuple<FUNCS...>>::type(args...));

        for (std::size_t i = 0; i < max_history_; ++i) {
            std::get<FUNC_INDEX>(output_history_[i])
                    .reset(new typename std::tuple_element<FUNC_INDEX, std::tuple<FUNCS...>>::type(args...));
            std::get<FUNC_INDEX>(residual_history_[i])
                    .reset(new typename std::tuple_element<FUNC_INDEX, std::tuple<FUNCS...>>::type(args...));
        }

        // initialize output and input with given initial value
        std::get<FUNC_INDEX>(functions_).copy(init_value, *std::get<FUNC_INDEX>(output_history_[0]));
        std::get<FUNC_INDEX>(functions_).copy(init_value, *std::get<FUNC_INDEX>(input_));
    }

    /// Set input for next mixing step
    /** \param [in]  input   Input functions, for which a copy operation is invoked.
     */
    template <std::size_t FUNC_INDEX>
    void
    set_input(const typename std::tuple_element<FUNC_INDEX, std::tuple<FUNCS...>>::type& input)
    {
        if (std::get<FUNC_INDEX>(input_)) {
            std::get<FUNC_INDEX>(functions_).copy(input, *std::get<FUNC_INDEX>(input_));
        } else {
            throw std::runtime_error("Mixer function not initialized!");
        }
    }

    /// Access last generated output. Mixing must have been performed at least once.
    /** \param [out]  output  Output function, into which the mixer output is copied.
     */
    template <std::size_t FUNC_INDEX>
    void
    get_output(typename std::tuple_element<FUNC_INDEX, std::tuple<FUNCS...>>::type& output)
    {
        const auto idx = idx_hist(step_);
        if (!std::get<FUNC_INDEX>(output_history_[idx])) {
            throw std::runtime_error("Mixer function not initialized!");
        }
        std::get<FUNC_INDEX>(functions_).copy(*std::get<FUNC_INDEX>(output_history_[idx]), output);
    }

    /// Mix input and stored history. Returns the root mean square error computed by inner products of residuals.
    /** \param [in]  rms_min  Minimum root mean square error. Mixing is only performed, if current RMS is above this
     *                        threshold.
     */
    double
    mix(double rms_min__)
    {
        this->update_residual();
        this->update_rms();
        double rmse = rmse_history_[idx_hist(step_)];
        if (rmse < rms_min__) {
            return rmse;
        }

        /* call mixing implementation */
        this->mix_impl();

        ++step_;
        return rmse;
    }

  protected:
    // Mixing implementation
    virtual void
    mix_impl() = 0;

    // update residual history for current step
    void
    update_residual()
    {
        this->copy(input_, residual_history_[idx_hist(step_)]);
        this->axpy(-1.0, output_history_[idx_hist(step_)], residual_history_[idx_hist(step_)]);
    }

    // update rmse histroy for current step. Residuals must have been updated before.
    void
    update_rms()
    {
        const auto idx = idx_hist(step_);

        /* compute sum of inner products; each inner product is normalized */
        double rmse = inner_product<true>(residual_history_[idx], residual_history_[idx]);

        rmse_history_[idx_hist(step_)] = std::sqrt(rmse);
    }

    // Storage index of given step
    std::size_t
    idx_hist(std::size_t step) const
    {
        return step % max_history_;
    }

    template <bool normalize>
    double
    inner_product(const std::tuple<std::unique_ptr<FUNCS>...>& x, const std::tuple<std::unique_ptr<FUNCS>...>& y)
    {
        return mixer_impl::InnerProduct<sizeof...(FUNCS) - 1, normalize, FUNCS...>::apply(functions_, x, y);
    }

    void
    scale(double alpha, std::tuple<std::unique_ptr<FUNCS>...>& x)
    {
        mixer_impl::Scaling<sizeof...(FUNCS) - 1, FUNCS...>::apply(functions_, alpha, x);
    }

    void
    copy(const std::tuple<std::unique_ptr<FUNCS>...>& x, std::tuple<std::unique_ptr<FUNCS>...>& y)
    {
        mixer_impl::Copy<sizeof...(FUNCS) - 1, FUNCS...>::apply(functions_, x, y);
    }

    void
    axpy(double alpha, const std::tuple<std::unique_ptr<FUNCS>...>& x, std::tuple<std::unique_ptr<FUNCS>...>& y)
    {
        mixer_impl::Axpy<sizeof...(FUNCS) - 1, FUNCS...>::apply(functions_, alpha, x, y);
    }

    void
    rotate(double c, double s, std::tuple<std::unique_ptr<FUNCS>...>& x, std::tuple<std::unique_ptr<FUNCS>...>& y)
    {
        mixer_impl::Rotate<sizeof...(FUNCS) - 1, FUNCS...>::apply(functions_, c, s, x, y);
    }

    // Strictly increasing counter, indicating the number of mixing steps
    std::size_t step_;

    // The maximum history size kept for each function
    std::size_t max_history_;

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
};
} // namespace mixer
} // namespace sirius

#endif // __MIXER_HPP__
