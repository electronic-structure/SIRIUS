#ifndef __FFT3D_BASE_H__
#define __FFT3D_BASE_H__

namespace sirius
{

class FFT3D_base
{    
    protected:
        
        /// size of each dimension
        int grid_size_[3];

        /// reciprocal space range
        int grid_limits_[3][2];

        /// linear index in FFT buffer
        mdarray<int,3> index_;

        /*!
            \brief Find smallest optimal grid size starting from n.
        */
        int find_grid_size(int n)
        {
            while (true)
            {
                int m = n;
                for (int k = 2; k <= 7; k++)
                    while (m % k == 0)
                        m /= k;
                if (m == 1) return n;
                else n++;
            }
        } 
        
        /*!
            \brief Determine the optimal FFT grid size and set grid limits.
        */
        void set_grid_size(int* n)
        {
            for (int i = 0; i < 3; i++)
            {
                grid_size_[i] = find_grid_size(n[i]);
                
                grid_limits_[i][1] = grid_size_[i] / 2;
                grid_limits_[i][0] = grid_limits_[i][1] - grid_size_[i] + 1;
            }
        }

        /*!
            \brief Build a mapping between reciprocal domain and linear index in FFT buffer 
        */ 
        void build_fft_index()
        {
            index_.set_dimensions(dimension(grid_limits_[0][0], grid_limits_[0][1]),
                                  dimension(grid_limits_[1][0], grid_limits_[1][1]),
                                  dimension(grid_limits_[2][0], grid_limits_[2][1]));
            index_.allocate();
            for (int i = grid_limits_[0][0]; i <= grid_limits_[0][1]; i++)
                for (int j = grid_limits_[1][0]; j <= grid_limits_[1][1]; j++)
                    for (int k = grid_limits_[2][0]; k <= grid_limits_[2][1]; k++)
                    {
                        int i0 = i;
                        int i1 = j;
                        int i2 = k;

                        if (i0 < 0) i0 += grid_size_[0];
                        if (i1 < 0) i1 += grid_size_[1];
                        if (i2 < 0) i2 += grid_size_[2];

                        index_(i, j, k) = i0 + i1 * grid_size_[0] + i2 * grid_size_[0] * grid_size_[1];
                    }
        }

 
    public:

        void init(int* n)
        {
            set_grid_size(n);
            build_fft_index();
        }

        inline int grid_limits(int d, int i)
        {
            return grid_limits_[d][i];
        }

        inline int size()
        {
            return grid_size_[0] * grid_size_[1] * grid_size_[2]; 
        }
        
        inline int size(int i)
        {
            return grid_size_[i]; 
        }

        inline int index(int i0, int i1, int i2)
        {
            return index_(i0, i1, i2);
        }
};

};

#endif // __FFT3D_BASE_H__
