#ifndef __FFT_GRID_H__
#define __FFT_GRID_H__

class FFT_grid
{
    private:

        /// Size of each dimension.
        int grid_size_[3];
        
        /// Reciprocal space range.
        std::pair<int, int> grid_limits_[3];

        /// Find smallest optimal grid size starting from n.
        int find_grid_size(int n)
        {
            while (true)
            {
                int m = n;
                for (int k = 2; k <= 5; k++)
                {
                    while (m % k == 0) m /= k;
                }
                if (m == 1) 
                {
                    return n;
                }
                else 
                {
                    n++;
                }
            }
        }

    public:

        FFT_grid()
        {
        }

        FFT_grid(vector3d<int> initial_dims__)
        {
            for (int i = 0; i < 3; i++)
            {
                grid_size_[i] = find_grid_size(initial_dims__[i]);
                
                grid_limits_[i].second = grid_size_[i] / 2;
                grid_limits_[i].first = grid_limits_[i].second - grid_size_[i] + 1;
            }
        }

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

        inline vector3d<int> coord_by_gvec(int i0__, int i1__, int i2__) const
        {
            if (i0__ < 0) i0__ += grid_size_[0];
            if (i1__ < 0) i1__ += grid_size_[1];
            if (i2__ < 0) i2__ += grid_size_[2];

            return vector3d<int>(i0__, i1__, i2__);
        }

        inline vector3d<int> gvec_by_coord(int x__, int y__, int z__) const
        {
            if (x__ > grid_limits_[0].second) x__ -= grid_size_[0];
            if (y__ > grid_limits_[1].second) y__ -= grid_size_[1];
            if (z__ > grid_limits_[2].second) z__ -= grid_size_[2];

            return vector3d<int>(x__, y__, z__);
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

#endif
