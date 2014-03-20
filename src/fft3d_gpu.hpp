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

        //== int num_fft_max()
        //== {
        //==     /* From cuFFT documentation:

        //==        In the worst case, the CUFFT Library allocates space for 8*batch*n[0]*..*n[rank-1] cufftComplex or 
        //==        cufftDoubleComplex elements (where batch denotes the number of transforms that will be executed in 
        //==        parallel, rank is the number of dimensions of the input data (see Multidimensional transforms) and n[] 
        //==        is the array of transform dimensions) for single and double- precision transforms respectively. */
        //==     int n = 8;
        //==     int num_fft = (int)(cuda_get_free_mem() / size() / n / sizeof(double_complex));
        //==     if (num_fft == 0)
        //==     {
        //==         std::stringstream s;
        //==         s << "Not enough memory for cuFFT" << std::endl 
        //==           << "  available GPU memory : " << cuda_get_free_mem() << std::endl
        //==           << "  size of FFT buffer : " << size() * sizeof(double_complex);
        //==         error_local(__FILE__, __LINE__, s);
        //==     }
        //==     return num_fft;
        //== }

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
