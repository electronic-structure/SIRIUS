
namespace sirius
{

class FFT3D_base
{    
    protected:
    
        int grid_size_[3];
        int grid_limits_[3][2];

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
       
    public:

        inline int grid_limits(int d, int i)
        {
            return grid_limits_[d][i];
        }

        void set_grid_size(int* n)
        {
            for (int i = 0; i < 3; i++)
            {
                grid_size_[i] = find_grid_size(n[i]);
                
                grid_limits_[i][1] = grid_size_[i] / 2;
                grid_limits_[i][0] = grid_limits_[i][1] - grid_size_[i] + 1;
            }
            
            for (int i = 0; i < 3; i++)
                printf("grid size : %i   limits : %i %i\n", grid_size_[i], grid_limits_[i][0], grid_limits_[i][1]);
        }
        
        inline int size()
        {
            return grid_size_[0] * grid_size_[1] * grid_size_[2]; 
        }
        
        inline int size(int i)
        {
            return grid_size_[i]; 
        }
};

};


