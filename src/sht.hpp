// Copyright (c) 2013-2014 Anton Kozhevnikov, Thomas Schulthess
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

/** \file sht.hpp
 *   
 *  \brief Contains templated implementation of sirius::SHT class.
 */

template <int direction, typename T>
Spheric_function<T> SHT::transform(Spheric_function<T>& f)
{
    Spheric_function<T> g;

    switch (direction)
    {
        /* forward transform, f(t, p) -> g(l, m) */
        case 1:
        {
            g = Spheric_function<T>(lmmax(), f.radial_grid());
            break;
        }
        /* backward transform, f(l, m) -> g(t, p) */
        case -1:
        {
            g = Spheric_function<T>(num_points(), f.radial_grid());
            break;
        }
        default:
        {
            error_local(__FILE__, __LINE__, "Wrong direction of transformation");
        }
    }
    transform<direction, T>(f, g);
    return g;
}

template <int direction, typename T>
void SHT::transform(Spheric_function<T>& f, Spheric_function<T>& g)
{
    assert(f.radial_grid().hash() == g.radial_grid().hash());

    switch (direction)
    {
        /* forward transform, f(t, p) -> g(l, m) */
        case 1:
        {
            forward_transform(&f(0, 0), lmmax(), f.radial_grid().num_points(), &g(0, 0));
            break;
        }
        /* backward transform, f(l, m) -> g(t, p) */
        case -1:
        {
            backward_transform(&f(0, 0), f.angular_domain_size(), f.radial_grid().num_points(), &g(0, 0));
            break;
        }
        default:
        {
            error_local(__FILE__, __LINE__, "Wrong direction of transformation");
        }
    }
}

