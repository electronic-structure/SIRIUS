template<> 
class FFT3D<gpu>
{
    private:

        vector3d<int> grid_size_;

        void* fft_buffer_ptr_device_;

        mdarray<double_complex, 2> fft_buffer_;

        int num_fft_;

    public:

        FFT3D(vector3d<int> grid_size__) : grid_size_(grid_size__), fft_buffer_ptr_device_(NULL), num_fft_(-1)
        {
        }

        int num_fft_max(int num_fft_min = 1 << 20)
        {
            /* From cuFFT documentation:

               In the worst case, the CUFFT Library allocates space for 8*batch*n[0]*..*n[rank-1] cufftComplex or 
               cufftDoubleComplex elements (where batch denotes the number of transforms that will be executed in 
               parallel, rank is the number of dimensions of the input data (see Multidimensional transforms) and n[] 
               is the array of transform dimensions) for single and double- precision transforms respectively. */
            int num_fft = (int)(cuda_get_free_mem() / size() / 8 / sizeof(double_complex));
            if (num_fft == 0)
            {
                std::stringstream s;
                s << "Not enough memory for cuFFT" << std::endl 
                  << "  available GPU memory : " << cuda_get_free_mem() << std::endl
                  << "  size of FFT buffer : " << size() * sizeof(double_complex);
                error_local(__FILE__, __LINE__, s);
            }
            return std::min(num_fft, num_fft_min);
        }

        inline void initialize(int num_fft__, void* fft_buffer_ptr_device__)
        {
            num_fft_ = num_fft__;
            fft_buffer_.set_dimensions(size(), num_fft_);
            if (fft_buffer_ptr_device__) 
            {
                fft_buffer_ptr_device_ = fft_buffer_ptr_device__;
            }
            else
            {
                fft_buffer_.allocate();
                fft_buffer_.allocate_on_device();
                fft_buffer_ptr_device_ = fft_buffer_.get_ptr_device();
            }
            cufft_create_batch_plan(grid_size_[0], grid_size_[1], grid_size_[2], num_fft_, fft_buffer_ptr_device_);

        }

        inline void finalize()
        {
            fft_buffer_.deallocate();
            fft_buffer_.deallocate_on_device();
            cufft_destroy_batch_plan();
        }
        
        inline void copy_to_device()
        {
            fft_buffer_.copy_to_device();
        }

        inline void copy_to_host()
        {
            fft_buffer_.copy_to_host();
        }
        
        inline void input(int n, int* map, double_complex* data, int id)
        {
            memset(&fft_buffer_(0, id), 0, size() * sizeof(double_complex));
            
            for (int i = 0; i < n; i++) fft_buffer_(map[i], id) = data[i];
        }
        
        inline void input(double_complex* data, int id)
        {
            memcpy(&fft_buffer_(0, id), data, size() * sizeof(double_complex));
        }

        inline void transform(int direction)
        {
            switch (direction)
            {
                case 1:
                {
                    cufft_backward_transform();
                    break;
                }
                case -1:
                {
                    cufft_forward_transform();
                    break;
                }
                default:
                {
                    error_local(__FILE__, __LINE__, "wrong FFT direction");
                }
            }
        }
        
        inline void output(int n, int* map, double_complex* data, int id)
        {
            for (int i = 0; i < n; i++) data[i] = fft_buffer_(map[i], id);
        }

        inline void output(double_complex* data, int id)
        {
            memcpy(data, &fft_buffer_(0, id), size() * sizeof(double_complex));
        }

        //void allocate_batch_fft_buffer(int nfft_max)
        //{
        //    cuda_malloc(&fft_buffer_device_ptr, size() * nfft_max * sizeof(double_complex));
        //}

        //void deallocate_batch_fft_buffer()
        //{
        //    cuda_free(fft_buffer_device_ptr);
        //}

        //void create_batch_plan(int nfft)
        //{
        //     cufft_create_batch_plan(grid_size_[0], grid_size_[1], grid_size_[2], nfft);
        //}

        //void destroy_batch_plan()
        //{
        //    cufft_destroy_batch_plan();
        //}

        //void batch_apply_v(int num_gkvec, int num_phi, int* map, double_complex* v_r, double_complex* phi)
        //{
        //    cufft_batch_apply_v(size(), num_gkvec, num_phi, fft_buffer_device_ptr, map, v_r, phi);
        //}

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

        inline mdarray<double_complex, 2>& fft_buffer()
        {
            return fft_buffer_;
        }
};
