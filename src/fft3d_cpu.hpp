template<> 
class FFT3D<cpu>
{
    private:

        /// size of each dimension
        int grid_size_[3];

        /// reciprocal space range
        std::pair<int, int> grid_limits_[3];
        
        /// backward transformation plan for each thread
        std::vector<fftw_plan> plan_backward_;
        
        /// forward transformation plan for each thread
        std::vector<fftw_plan> plan_forward_;
    
        /// inout buffer for each thread
        mdarray<double_complex, 2> fftw_input_buffer_;
        
        /// output buffer for each thread
        mdarray<double_complex, 2> fftw_output_buffer_;

        /// split index of FFT buffer
        splindex<block> spl_fft_size_;

        /// Execute backward transformation.
        inline void backward(int thread_id = 0)
        {    
            fftw_execute(plan_backward_[thread_id]);
        }
        
        /// Execute forward transformation.
        inline void forward(int thread_id = 0)
        {    
            fftw_execute(plan_forward_[thread_id]);
            double norm = 1.0 / size();
            for (int i = 0; i < size(); i++) fftw_output_buffer_(i, thread_id) *= norm;
        }

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

        FFT3D<cpu>(vector3d<int> dims)
        {
            Timer t("sirius::FFT3D<cpu>::FFT3D<cpu>");
            for (int i = 0; i < 3; i++)
            {
                grid_size_[i] = find_grid_size(dims[i]);
                
                grid_limits_[i].second = grid_size_[i] / 2;
                grid_limits_[i].first = grid_limits_[i].second - grid_size_[i] + 1;
            }

            fftw_input_buffer_.set_dimensions(size(), Platform::num_fft_threads());
            fftw_input_buffer_.allocate();

            fftw_output_buffer_.set_dimensions(size(), Platform::num_fft_threads());
            fftw_output_buffer_.allocate();
 
            plan_backward_.resize(Platform::num_fft_threads());
            plan_forward_.resize(Platform::num_fft_threads());

            for (int i = 0; i < Platform::num_fft_threads(); i++)
            {
                plan_backward_[i] = fftw_plan_dft_3d(size(2), size(1), size(0), 
                                                     (fftw_complex*)&fftw_input_buffer_(0, i), 
                                                     (fftw_complex*)&fftw_output_buffer_(0, i), 1, FFTW_MEASURE);
                plan_forward_[i] = fftw_plan_dft_3d(size(2), size(1), size(0), 
                                                    (fftw_complex*)&fftw_input_buffer_(0, i), 
                                                    (fftw_complex*)&fftw_output_buffer_(0, i), -1, FFTW_MEASURE);
            }

            spl_fft_size_.split(size(), Platform::num_mpi_ranks(), Platform::mpi_rank());
        }

        ~FFT3D<cpu>()
        {
            for (int i = 0; i < Platform::num_fft_threads(); i++)
            {
                fftw_destroy_plan(plan_backward_[i]);
                fftw_destroy_plan(plan_forward_[i]);
            }
            fftw_cleanup();
        }

        /// Zero the input buffer for a given thread.
        inline void zero(int thread_id = 0)
        {
            assert(thread_id < Platform::num_fft_threads());

            memset(&fftw_input_buffer_(0, thread_id), 0, size() * sizeof(double_complex));
        }

        template<typename T>
        inline void input(int n, int* map, T* data, int thread_id = 0)
        {
            assert(thread_id < Platform::num_fft_threads());
            
            zero(thread_id);

            for (int i = 0; i < n; i++) fftw_input_buffer_(map[i], thread_id) = data[i];
        }

        inline void input(double* data, int thread_id = 0)
        {
            assert(thread_id < Platform::num_fft_threads());
            
            for (int i = 0; i < size(); i++) fftw_input_buffer_(i, thread_id) = data[i];
        }
        
        inline void input(double_complex* data, int thread_id = 0)
        {
            assert(thread_id < Platform::num_fft_threads());
            
            memcpy(&fftw_input_buffer_(0, thread_id), data, size() * sizeof(double_complex));
        }
        
        /// Execute the transformation for a given thread.
        inline void transform(int direction, int thread_id = 0)
        {
            assert(thread_id < Platform::num_fft_threads());

            switch(direction)
            {
                case 1:
                {
                    backward(thread_id);
                    break;
                }
                case -1:
                {
                    forward(thread_id);
                    break;
                }
                default:
                {
                    error_local(__FILE__, __LINE__, "wrong FFT direction");
                }
            }
        }

        inline void output(double* data, int thread_id = 0)
        {
            assert(thread_id < Platform::num_fft_threads());

            for (int i = 0; i < size(); i++) data[i] = real(fftw_output_buffer_(i, thread_id));
        }
        
        inline void output(double_complex* data, int thread_id = 0)
        {
            assert(thread_id < Platform::num_fft_threads());

            memcpy(data, &fftw_output_buffer_(0, thread_id), size() * sizeof(double_complex));
        }
        
        inline void output(int n, int* map, double_complex* data, int thread_id = 0)
        {
            assert(thread_id < Platform::num_fft_threads());

            for (int i = 0; i < n; i++) data[i] = fftw_output_buffer_(map[i], thread_id);
        }
        
        inline const std::pair<int, int>& grid_limits(int idim)
        {
            return grid_limits_[idim];
        }

        /// Total size of the FFT grid.
        inline int size()
        {
            return grid_size_[0] * grid_size_[1] * grid_size_[2]; 
        }

        /// Size of a given dimension.
        inline int size(int d)
        {
            assert(d >= 0 && d < 3);
            return grid_size_[d]; 
        }

        inline int index(int i0, int i1, int i2)
        {
            if (i0 < 0) i0 += grid_size_[0];
            if (i1 < 0) i1 += grid_size_[1];
            if (i2 < 0) i2 += grid_size_[2];

            return (i0 + i1 * grid_size_[0] + i2 * grid_size_[0] * grid_size_[1]);
        }

        inline int local_size()
        {
            return spl_fft_size_.local_size();
        }

        inline int global_index(int irloc)
        {
            return spl_fft_size_[irloc];
        }

        inline int global_offset()
        {
            return spl_fft_size_.global_offset();
        }
        
        /// Direct access to the output buffer
        inline double_complex& output_buffer(int i, int thread_id = 0)
        {
            return fftw_output_buffer_(i, thread_id);
        }
        
        /// Direct access to the input buffer
        inline double_complex& input_buffer(int i, int thread_id = 0)
        {
            return fftw_input_buffer_(i, thread_id);
        }

        vector3d<int> grid_size()
        {
            return vector3d<int>(grid_size_);
        }
};
