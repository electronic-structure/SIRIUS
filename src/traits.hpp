#ifndef TRAITS_H
#define TRAITS_H

namespace sirius {

template <class X>
struct identity
{
    typedef X type;
};

template <class X>
using identity_t = typename identity<X>::type;

} // namespace sirius

#endif /* TRAITS_H */
