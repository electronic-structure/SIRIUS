#ifndef __TRAITS_HPP__
#define __TRAITS_HPP__

namespace sirius {

template <class X>
struct identity
{
    typedef X type;
};

template <class X>
using identity_t = typename identity<X>::type;

} // namespace sirius

#endif /* __TRAITS_HPP__ */
