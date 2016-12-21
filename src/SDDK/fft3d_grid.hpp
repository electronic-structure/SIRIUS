// Copyright (c) 2013-2016 Anton Kozhevnikov, Thomas Schulthess
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
 *  \brief Contains declaration and implementation of FFT3D_grid class.
 */

#ifndef __FFT3D_GRID_HPP__
#define __FFT3D_GRID_HPP__

namespace sddk {

/// Handling of FFT grids.
class FFT3D_grid
{
  private:
    /// Size of each dimension.
    int grid_size_[3];

    /// Reciprocal space range.
    std::pair<int, int> grid_limits_[3];

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

    void find_grid_size(std::array<int, 3> initial_dims__)
    {
        for (int i = 0; i < 3; i++) {
            grid_size_[i] = find_grid_size(initial_dims__[i]);

            grid_limits_[i].second = grid_size_[i] / 2;
            grid_limits_[i].first  = grid_limits_[i].second - grid_size_[i] + 1;
        }

        for (int i = 0; i < 3; i++) {
            for (int x = 0; x < size(i); x++) {
                if (coord_by_gvec(gvec_by_coord(x, i), i) != x) {
                    throw std::runtime_error("find_grid_size: wrong mapping of coordinates");
                }
            }
        }
    }

  public:
    FFT3D_grid()
    {
    }

    FFT3D_grid(std::array<int, 3> initial_dims__)
    {
        find_grid_size(initial_dims__);
    }

    //FFT3D_grid(double cutoff__, matrix3d<double> M__)
    //{
    //    auto initial_dims = Utils::find_translations(cutoff__, M__);
    //    find_grid_size(initial_dims);
    //}

    /// Limits of a given dimension.
    inline const std::pair<int, int>& limits(int idim__) const
    {
        assert(idim__ >= 0 && idim__ < 3);
        return grid_limits_[idim__];
    }

    /// Size of a given dimension.
    inline int size(int idim__) const
    {
        assert(idim__ >= 0 && idim__ < 3);
        return grid_size_[idim__];
    }

    /// Total size of the FFT grid.
    inline int size() const
    {
        return grid_size_[0] * grid_size_[1] * grid_size_[2];
    }

    inline std::array<int, 3> coord_by_gvec(int i0__, int i1__, int i2__) const
    {
        if (i0__ < 0) {
            i0__ += grid_size_[0];
        }
        if (i1__ < 0) {
            i1__ += grid_size_[1];
        }
        if (i2__ < 0) {
            i2__ += grid_size_[2];
        }

        return {i0__, i1__, i2__};
    }

    inline int coord_by_gvec(int i__, int idim__) const
    {
        if (i__ < 0) {
            i__ += grid_size_[idim__];
        }
        return i__;
    }

    inline std::array<int, 3> gvec_by_coord(int x__, int y__, int z__) const
    {
        if (x__ > grid_limits_[0].second) {
            x__ -= grid_size_[0];
        }
        if (y__ > grid_limits_[1].second) {
            y__ -= grid_size_[1];
        }
        if (z__ > grid_limits_[2].second) {
            z__ -= grid_size_[2];
        }

        return {x__, y__, z__};
    }

    inline int gvec_by_coord(int x__, int idim__) const
    {
        if (x__ > limits(idim__).second) {
            x__ -= size(idim__);
        }
        return x__;
    }

    /// Linear index inside FFT buffer by grid coordinates.
    inline int index_by_coord(int x__, int y__, int z__) const
    {
        return (x__ + y__ * grid_size_[0] + z__ * grid_size_[0] * grid_size_[1]);
    }

    /// Return linear index of a plane-wave harmonic with fractional coordinates (i0, i1, i2) inside FFT buffer.
    inline int index_by_gvec(int i0__, int i1__, int i2__) const
    {
        auto coord = coord_by_gvec(i0__, i1__, i2__);
        return index_by_coord(coord[0], coord[1], coord[2]);
    }
};

} // namespace sddk

#endif // __FFT3D_GRID_HPP__
