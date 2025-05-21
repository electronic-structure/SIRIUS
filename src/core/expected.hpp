#pragma once

// clang-format off

// Taken from https://github.com/eth-cscs/arbor, https://arbor-sim.org/
/*
Copyright 2016-2020 Eidgenössische Technische Hochschule Zürich and
Forschungszentrum Jülich GmbH.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// C++14 version of the proposed std::expected class
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/p0323r9.html
//
// Difference from proposal:
//
// * Left out lots of constexpr.
// * Lazy about explicitness of some conversions.

#include <initializer_list>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>

namespace util {

template <typename T> struct remove_cvref {
  typedef std::remove_cv_t<std::remove_reference_t<T>> type;
};

template <typename T> using remove_cvref_t = typename remove_cvref<T>::type;

namespace detail {
// True if T can constructed from or converted from [const, ref] X.
template <typename T, typename X>
struct conversion_hazard: std::integral_constant<bool,
    std::is_constructible_v<T, X> ||
    std::is_constructible_v<T, const X> ||
    std::is_constructible_v<T, X&> ||
    std::is_constructible_v<T, const X&> ||
    std::is_convertible_v<X, T> ||
    std::is_convertible_v<const X, T> ||
    std::is_convertible_v<X&, T> ||
    std::is_convertible_v<const X&, T>> {};

template <typename T, typename X>
inline constexpr bool conversion_hazard_v = conversion_hazard<T, X>::value;
} // namespace detail

struct unexpect_t {};
inline constexpr unexpect_t unexpect{};

template <typename E = void>
struct bad_expected_access;

template <>
struct bad_expected_access<void>: std::exception {
    bad_expected_access() = default;
    virtual const char* what() const noexcept override { return "bad expected access"; }
};

template <typename E>
struct bad_expected_access: public bad_expected_access<void> {
    explicit bad_expected_access(E error): error_(error) {}

    E& error() & { return error_; }
    const E& error() const& { return error_; }
    E&& error() && { return std::move(error_); }
    const E&& error() const && { return std::move(error_); }

private:
    E error_;
};

// The unexpected<E> wrapper is mainly boiler-plate for a box that wraps
// a value of type E, with corresponding conversion and assignment semantics.

template <typename E>
struct unexpected {
    template <typename F>
    friend struct unexpected;

    unexpected() = default;
    unexpected(const unexpected&) = default;
    unexpected(unexpected&&) = default;

    // Emplace-style ctors.

    template <typename... Args>
    explicit unexpected(std::in_place_t, Args&&... args):
        value_(std::forward<Args>(args)...) {}

    template <typename X, typename... Args>
    explicit unexpected(std::in_place_t, std::initializer_list<X> il, Args&&... args):
        value_(il, std::forward<Args>(args)...) {}

    // Converting ctors.

    template <typename F,
        typename = std::enable_if_t<std::is_constructible_v<E, F&&>>,
        typename = std::enable_if_t<!std::is_same_v<std::in_place_t, remove_cvref_t<F>>>,
        typename = std::enable_if_t<!std::is_same_v<unexpected, remove_cvref_t<F>>>
    >
    explicit unexpected(F&& f): value_(std::forward<F>(f)) {}

    template <
        typename F,
        typename = std::enable_if_t<!detail::conversion_hazard_v<E, unexpected<F>>>
    >
    unexpected(unexpected<F>&& u): value_(std::move(u.value_)) {}

    template <
        typename F,
        typename = std::enable_if_t<!detail::conversion_hazard_v<E, unexpected<F>>>
    >
    unexpected(const unexpected<F>& u): value_(u.value_) {}

    // Assignment operators.

    unexpected& operator=(const unexpected& u) { return value_ = u.value_, *this; }

    unexpected& operator=(unexpected&& u) { return value_ = std::move(u.value_), *this; }

    template <typename F>
    unexpected& operator=(const unexpected<F>& u) { return value_ = u.value_, *this; }

    template <typename F>
    unexpected& operator=(unexpected<F>&& u) { return value_ = std::move(u.value_), *this; }

    // Access.

    E& value() & { return value_; }
    const E& value() const & { return value_; }
    E&& value() && { return std::move(value_); }
    const E&& value() const && { return std::move(value_); }

    // Equality.

    template <typename F>
    bool operator==(const unexpected<F>& other) const { return value()==other.value(); }

    template <typename F>
    bool operator!=(const unexpected<F>& other) const { return value()!=other.value(); }

    // Swap.

    void swap(unexpected& other) noexcept(std::is_nothrow_swappable_v<E>) {
        using std::swap;
        swap(value_, other.value_);
    }

    friend void swap(unexpected& a, unexpected& b) noexcept(noexcept(a.swap(b))) { a.swap(b); }

private:
    E value_;
};

template <typename E>
unexpected(E) -> unexpected<E>;

template <typename T, typename E>
struct expected {
    using value_type = T;
    using error_type = E;
    using unexpected_type = unexpected<E>;
    using data_type = std::variant<T, unexpected_type>;

    expected() requires std::is_default_constructible_v<T>: data_(T()) {};
    expected(const expected&) = default;
    expected(expected&&) = default;

    // Emplace-style ctors.

    template <typename... Args>
    explicit expected(std::in_place_t, Args&&... args):
        data_(std::in_place_index<0>, std::forward<Args>(args)...) {}

    template <typename X, typename... Args>
    explicit expected(std::in_place_t, std::initializer_list<X> il, Args&&... args):
        data_(std::in_place_index<0>, il, std::forward<Args>(args)...) {}

    // (Proposal says to forward args to unexpected<E>, but this is not compatible
    // with the requirement that E is constructible from args; so here we're forwarding
    // to unexpected<E> with an additional 'in_place' argument.)
    template <typename... Args>
    explicit expected(unexpect_t, Args&&... args):
        data_(std::in_place_index<1>, std::in_place_t{}, std::forward<Args>(args)...) {}

    template <typename X, typename... Args>
    explicit expected(unexpect_t, std::initializer_list<X> il, Args&&... args):
        data_(std::in_place_index<1>, std::in_place_t{}, il, std::forward<Args>(args)...) {}

    // Converting ctors.

    template <
        typename S,
        typename F,
        typename = std::enable_if_t<!std::is_same_v<expected, expected<S, F>>>,
        typename = std::enable_if_t<!detail::conversion_hazard_v<T, expected<S, F>>>,
        typename = std::enable_if_t<!detail::conversion_hazard_v<unexpected<E>, expected<S, F>>>
    >
    expected(const expected<S, F>& other):
        data_(other? data_type(std::in_place_index<0>, *other): data_type(std::in_place_index<1>, other.error()))
    {}

    template <
        typename S,
        typename F,
        typename = std::enable_if_t<!std::is_same_v<expected, expected<S, F>>>,
        typename = std::enable_if_t<!detail::conversion_hazard_v<T, expected<S, F>>>,
        typename = std::enable_if_t<!detail::conversion_hazard_v<unexpected<E>, expected<S, F>>>
    >
    expected(expected<S, F>&& other):
        data_(other? data_type(std::in_place_index<0>, *std::move(other)): data_type(std::in_place_index<1>, std::move(other).error()))
    {}

    template <
        typename S,
        typename = std::enable_if_t<!std::is_same_v<std::in_place_t, remove_cvref_t<S>>>,
        typename = std::enable_if_t<!std::is_same_v<expected, remove_cvref_t<S>>>,
        typename = std::enable_if_t<!std::is_same_v<unexpected<E>, remove_cvref_t<S>>>,
        typename = std::enable_if_t<std::is_constructible_v<T, S&&>>
    >
    expected(S&& x): data_(std::in_place_index<0>, std::forward<S>(x)) {}

    template <typename F>
    expected(const unexpected<F>& u): data_(std::in_place_index<1>, u) {}

    template <typename F>
    expected(unexpected<F>&& u): data_(std::in_place_index<1>, std::move(u)) {}

    // Assignment ops.

    expected& operator=(const expected& other) noexcept(std::is_nothrow_copy_assignable_v<data_type>) {
        data_ = other.data_;
        return *this;
    }

    expected& operator=(expected&& other) noexcept(std::is_nothrow_move_assignable_v<data_type>) {
        data_ = std::move(other.data_);
        return *this;
    }

    template <
        typename S,
        typename = std::enable_if_t<!std::is_same_v<expected, remove_cvref_t<S>>>,
        typename = std::enable_if_t<std::is_constructible_v<T, S>>,
        typename = std::enable_if_t<std::is_assignable_v<T&, S>>
    >
    expected& operator=(S&& v) {
        data_ = data_type(std::in_place_index<0>, std::forward<S>(v));
        return *this;
    }

    template <typename F>
    expected& operator=(const unexpected<F>& u) {
        data_ = data_type(std::in_place_index<1>, u);
        return *this;
    }

    template <typename F>
    expected& operator=(unexpected<F>&& u) {
        data_ = data_type(std::in_place_index<1>, std::move(u));
        return *this;
    }

    // Emplace ops.

    template <typename... Args>
    T& emplace(Args&&... args) {
        data_ = data_type(std::in_place_index<0>, std::forward<Args>(args)...);
        return value();
    }

    template <typename U, typename... Args>
    T& emplace(std::initializer_list<U> il, Args&&... args) {
        data_ = data_type(std::in_place_index<0>, il, std::forward<Args>(args)...);
        return value();
    }

    // Swap ops.

    void swap(expected& other) noexcept(std::is_nothrow_swappable_v<data_type>) {
        data_.swap(other.data_);
    }

    friend void swap(expected& a, expected& b) { a.swap(b); }

    // Accessors.

    bool has_value() const noexcept { return data_.index()==0; }
    explicit operator bool() const noexcept { return has_value(); }

    T& value() & {
        if (*this) return std::get<0>(data_);
        throw bad_expected_access(error());
    }
    const T& value() const& {
        if (*this) return std::get<0>(data_);
        throw bad_expected_access(error());
    }
    T&& value() && {
        if (*this) return std::get<0>(std::move(data_));
        throw bad_expected_access(error());
    }
    const T&& value() const&& {
        if (*this) return std::get<0>(std::move(data_));
        throw bad_expected_access(error());
    }

    T& unwrap() & {
        if (*this) return std::get<0>(data_);
        throw error();
    }
    const T& unwrap() const& {
        if (*this) return std::get<0>(data_);
        throw error();
    }
    T&& unwrap() && {
        if (*this) return std::get<0>(std::move(data_));
        throw error();
    }
    const T&& unwrap() const&& {
        if (*this) return std::get<0>(std::move(data_));
        throw error();
    }

    const E& error() const& { return std::get<1>(data_).value(); }
    E& error() & { return std::get<1>(data_).value(); }

    const E&& error() const&& { return std::get<1>(std::move(data_)).value(); }
    E&& error() && { return std::get<1>(std::move(data_)).value(); }

    const T& operator*() const& { return std::get<0>(data_); }
    T& operator*() & { return std::get<0>(data_); }

    const T&& operator*() const&& { return std::get<0>(std::move(data_)); }
    T&& operator*() && { return std::get<0>(std::move(data_)); }

    const T* operator->() const { return std::get_if<0>(&data_); }
    T* operator->() { return std::get_if<0>(&data_); }

    template <typename S>
    T value_or(S&& s) const& { return has_value()? value(): std::forward<S>(s); }

    template <typename S>
    T value_or(S&& s) && { return has_value()? value(): std::forward<S>(s); }

private:
    data_type data_;
};

// Equality comparisons for non-void expected.

template <typename T1, typename E1, typename T2, typename E2,
          typename = std::enable_if_t<!std::is_void_v<T1>>,
          typename = std::enable_if_t<!std::is_void_v<T2>>>
inline bool operator==(const expected<T1, E1>& a, const expected<T2, E2>& b) {
    return a? b && a.value()==b.value(): !b && a.error()==b.error();
}

template <typename T1, typename E1, typename T2, typename E2,
          typename = std::enable_if_t<!std::is_void_v<T1>>,
          typename = std::enable_if_t<!std::is_void_v<T2>>>
inline bool operator!=(const expected<T1, E1>& a, const expected<T2, E2>& b) {
    return a? !b || a.value()!=b.value(): b || a.error()!=b.error();
}

template <typename T1, typename E1, typename T2,
          typename = std::enable_if_t<!std::is_void_v<T1>>,
          typename = decltype(static_cast<bool>(std::declval<const expected<T1, E1>>().value()==std::declval<T2>()))>
inline bool operator==(const expected<T1, E1>& a, const T2& v) {
    return a && a.value()==v;
}

template <typename T1, typename E1, typename T2,
          typename = std::enable_if_t<!std::is_void_v<T1>>,
          typename = decltype(static_cast<bool>(std::declval<const expected<T1, E1>>().value()==std::declval<T2>()))>
inline bool operator==(const T2& v, const expected<T1, E1>& a) {
    return a==v;
}

template <typename T1, typename E1, typename T2,
          typename = std::enable_if_t<!std::is_void_v<T1>>,
          typename = decltype(static_cast<bool>(std::declval<const expected<T1, E1>>().value()!=std::declval<T2>()))>
inline bool operator!=(const expected<T1, E1>& a, const T2& v) {
    return !a || a.value()!=v;
}

template <typename T1, typename E1, typename T2,
          typename = std::enable_if_t<!std::is_void_v<T1>>,
          typename = decltype(static_cast<bool>(std::declval<const expected<T1, E1>>().value()!=std::declval<T2>()))>
inline bool operator!=(const T2& v, const expected<T1, E1>& a) {
    return a!=v;
}

// Equality comparisons against unexpected.

template <typename T1, typename E1, typename E2,
          typename = decltype(static_cast<bool>(unexpected(std::declval<const expected<T1, E1>>().error())
                                                ==std::declval<const unexpected<E2>>()))>
inline bool operator==(const expected<T1, E1>& a, const unexpected<E2>& u) {
    return !a && unexpected(a.error())==u;
}

template <typename T1, typename E1, typename E2,
          typename = decltype(static_cast<bool>(unexpected(std::declval<const expected<T1, E1>>().error())
                                                ==std::declval<const unexpected<E2>>()))>
inline bool operator==(const unexpected<E2>& u, const expected<T1, E1>& a) {
    return a==u;
}

template <typename T1, typename E1, typename E2,
          typename = decltype(static_cast<bool>(unexpected(std::declval<const expected<T1, E1>>().error())
                                                !=std::declval<const unexpected<E2>>()))>
inline bool operator!=(const expected<T1, E1>& a, const unexpected<E2>& u) {
    return a || unexpected(a.error())!=u;
}

template <typename T1, typename E1, typename E2,
          typename = decltype(static_cast<bool>(unexpected(std::declval<const expected<T1, E1>>().error())
                                                !=std::declval<const unexpected<E2>>()))>
inline bool operator!=(const unexpected<E2>& u, const expected<T1, E1>& a) {
    return a!=u;
}


template <typename E>
struct expected<void, E> {
    using value_type = void;
    using error_type = E;
    using unexpected_type = unexpected<E>;
    using data_type = std::optional<unexpected_type>;

    expected() = default;
    expected(const expected&) = default;
    expected(expected&&) = default;

    // Emplace-style ctors.

    explicit expected(std::in_place_t) {}

    template <typename... Args>
    explicit expected(unexpect_t, Args&&... args):
        data_(std::in_place_t{}, std::in_place_t{}, std::forward<Args>(args)...) {}

    template <typename X, typename... Args>
    explicit expected(unexpect_t, std::initializer_list<X> il, Args&&... args):
        data_(std::in_place_t{}, std::in_place_t{}, il, std::forward<Args>(args)...) {}

    // Converting ctors.

    template <
        typename F,
        typename = std::enable_if_t<!detail::conversion_hazard_v<unexpected<E>, expected<void, F>>>
    >
    expected(const expected<void, F>& other): data_(other.data_) {}

    template <
        typename F,
        typename = std::enable_if_t<!detail::conversion_hazard_v<unexpected<E>, expected<void, F>>>
    >
    expected(expected<void, F>&& other): data_(std::move(other.data_)) {}

    template <typename F>
    expected(const unexpected<F>& u): data_(u) {}

    template <typename F>
    expected(unexpected<F>&& u): data_(std::move(u)) {}

    // Assignment ops.

    expected& operator=(const expected& other) noexcept(std::is_nothrow_copy_assignable_v<data_type>) {
        data_ = other.data_;
        return *this;
    }

    expected& operator=(expected&& other) noexcept(std::is_nothrow_move_assignable_v<data_type>) {
        data_ = std::move(other.data_);
        return *this;
    }

    template <typename F>
    expected& operator=(const unexpected<F>& u) {
        data_ = u;
        return *this;
    }

    template <typename F>
    expected& operator=(unexpected<F>&& u) {
        data_ = std::move(u);
        return *this;
    }

    // No emplace ops.

    // Swap ops.

    void swap(expected& other) {
        data_.swap(other.data_);
    }

    // Accessors.

    bool has_value() const noexcept { return !data_; }
    explicit operator bool() const noexcept { return has_value(); }

    void value() const {
        if (!has_value()) throw bad_expected_access<E>(error());
    }

    const E& error() const& { return data_->value(); }
    E& error() & { return data_->value(); }

    const E&& error() const&& { return std::move(data_->value()); }
    E&& error() && { return std::move(data_->value()); }

private:
    data_type data_;
};

// Equality comparisons for void expected.

template <typename T1, typename E1, typename T2, typename E2,
          typename = std::enable_if_t<std::is_void_v<T1> || std::is_void_v<T2>>>
inline bool operator==(const expected<T1, E1>& a, const expected<T2, E2>& b) {
    return a? b && std::is_void_v<T1> && std::is_void_v<T2>: !b && a.error()==b.error();
}

template <typename T1, typename E1, typename T2, typename E2,
          typename = std::enable_if_t<std::is_void_v<T1> || std::is_void_v<T2>>>
inline bool operator!=(const expected<T1, E1>& a, const expected<T2, E2>& b) {
    return a? !b || !std::is_void_v<T2> || !std::is_void_v<T1>: b || a.error()!=b.error();
}
} // namespace util
