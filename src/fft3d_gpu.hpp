template<> 
class FFT3D<gpu>
{
    private:

        vector3d<int> grid_size_;

    public:

        FFT3D(vector3d<int> grid_size__) : grid_size_(grid_size__)
        {
            cufft_create_plan_handle();
        }

        int num_fft_max(size_t free_mem)
        {
            int nfft = 0;
            while (cufft_get_size(grid_size_[0], grid_size_[1], grid_size_[2], nfft + 1) < free_mem) nfft++;
            return nfft;
        }

        inline size_t work_area_size(int nfft)
        {
            return cufft_get_size(grid_size_[0], grid_size_[1], grid_size_[2], nfft);
        }

        inline void initialize(int num_fft, void* work_area)
        {
            cufft_create_batch_plan(grid_size_[0], grid_size_[1], grid_size_[2], num_fft);
            cufft_set_work_area(work_area);
        }

        inline void finalize()
        {
            cufft_destroy_batch_plan();
        }

        inline void batch_load(int num_elements, int* map, void* data, void* fft_buffer)
        {
            cufft_batch_load_gpu(num_elements, map, data, fft_buffer);
        }

        inline void batch_unload(int num_elements, int* map, void* fft_buffer, void* data)
        {
            cufft_batch_unload_gpu(num_elements, map, fft_buffer, data);
        }

        inline void transform(int direction, void* fft_buffer)
        {
            switch (direction)
            {
                case 1:
                {
                    cufft_backward_transform(fft_buffer);
                    break;
                }
                case -1:
                {
                    cufft_forward_transform(fft_buffer);
                    break;
                }
                default:
                {
                    error_local(__FILE__, __LINE__, "wrong FFT direction");
                }
            }
        }
        
        /// Total size of the FFT grid.
        inline int size()
        {
            return grid_size_[0] * grid_size_[1] * grid_size_[2]; 
        }

        /// Size of a given dimension.
        inline int size(int d)
        {
            return grid_size_[d]; 
        }
};
