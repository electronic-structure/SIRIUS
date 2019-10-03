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

/** \file fft3d_grid.hpp
 *
 *  \brief Contains declaration and implementation of sddk::FFT3D_grid class.
 */

#include <array>
#include <cassert>

#ifndef __FFT3D_GRID_HPP__
#define __FFT3D_GRID_HPP__

namespace sddk {

/// Handling of FFT grids.
class FFT3D_grid : public std::array<int, 3>
{
  private:
    /// Reciprocal space range.
    std::array<std::pair<int, int>, 3> grid_limits_;

    /// Find smallest optimal grid size starting from n.
    int find_grid_size(int n)
    {
        while (true) {
            int m = n;
            for (int k = 2; k <= 5; k++) {
                while (m % k == 0) {
                    m /= k;
                }
            }
            if (m == 1) {
                return n;
            } else {
                n++;
            }
        }
    }

    /// Find grid sizes and limits for all three dimensions.
    void find_grid_size(std::array<int, 3> initial_dims__)
    {
        for (int i = 0; i < 3; i++) {
            (*this)[i] = find_grid_size(initial_dims__[i]);

            grid_limits_[i].second = (*this)[i] / 2;
            grid_limits_[i].first  = grid_limits_[i].second - (*this)[i] + 1;
        }

        for (int x = 0; x < (*this)[0]; x++) {
            if (coord_by_freq<0>(freq_by_coord<0>(x)) != x) {
                throw std::runtime_error("FFT3D_grid::find_grid_size(): wrong mapping of x-coordinates");
            }
        }
        for (int x = 0; x < (*this)[1]; x++) {
            if (coord_by_freq<1>(freq_by_coord<1>(x)) != x) {
                throw std::runtime_error("FFT3D_grid::find_grid_size(): wrong mapping of y-coordinates");
            }
        }
        for (int x = 0; x < (*this)[2]; x++) {
            if (coord_by_freq<2>(freq_by_coord<2>(x)) != x) {
                throw std::runtime_error("FFT3D_grid::find_grid_size(): wrong mapping of z-coordinates");
            }
        }
    }

  public:

    /// Default constructor.
    FFT3D_grid()
    {
    }

    /// Create FFT grid with initial dimensions.
    FFT3D_grid(std::array<int, 3> initial_dims__)
    {
        find_grid_size(initial_dims__);
    }

    /// Limits of a given dimension.
    inline const std::pair<int, int>& limits(int idim__) const
    {
        assert(idim__ >= 0 && idim__ < 3);
        return grid_limits_[idim__];
    }

    /// Total size of the FFT grid.
    inline int num_points() const
    {
        return (*this)[0] * (*this)[1] * (*this)[2];
    }

    /// Get coordinate in range [0, N_d) by the frequency index.
    template <int d>
    inline int coord_by_freq(int i__) const
    {
        if (i__ < 0) {
            i__ += (*this)[d];
        }
        return i__;
    }

    /// Return {x, y, z} coordinates by frequency indices.
    inline std::array<int, 3> coord_by_freq(int i0__, int i1__, int i2__) const
    {
        return {coord_by_freq<0>(i0__), coord_by_freq<1>(i1__), coord_by_freq<2>(i2__)};
    }

    /// Get frequency by coordinate.
    template <int d>
    inline int freq_by_coord(int x__) const
    {
        if (x__ > grid_limits_[d].second) {
            x__ -= (*this)[d];
        }
        return x__;
    }

    /// Return 3d vector of frequencies corresponding to {x, y, z} position in the FFT buffer.
    inline std::array<int, 3> freq_by_coord(int x__, int y__, int z__) const
    {
        return {freq_by_coord<0>(x__), freq_by_coord<1>(y__), freq_by_coord<2>(z__)};
    }

    /// Linear index inside FFT buffer by grid coordinates.
    inline int index_by_coord(int x__, int y__, int z__) const
    {
        return (x__ + (*this)[0] * (y__  + z__ * (*this)[1]));
    }

    /// Return linear index of a plane-wave harmonic with fractional coordinates (i0, i1, i2) inside FFT buffer.
    inline int index_by_freq(int i0__, int i1__, int i2__) const
    {
        auto coord = coord_by_freq(i0__, i1__, i2__);
        return index_by_coord(coord[0], coord[1], coord[2]);
    }
};

} // namespace sddk

#endif // __FFT3D_GRID_HPP__
