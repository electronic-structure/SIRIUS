class Platform
{
    private:

        static bool verbose_;

        static int64_t heap_allocated_;

        static int num_fft_threads_;
    
    public:

        static void initialize(bool call_mpi_init__)
        {
            if (call_mpi_init__) MPI_Init(NULL, NULL);
            //MPI_Comm_size(MPI_COMM_WORLD, &size_);
            //MPI_Comm_rank(MPI_COMM_WORLD, &rank_);

            // rank 0 prints by default
            verbose_ = false;
            if (mpi_rank() == 0) verbose_ = true;

            cuda_init();
            cublas_init();
        }

        static int mpi_rank(MPI_Comm comm = MPI_COMM_WORLD)
        {
            int rank;
            MPI_Comm_rank(comm, &rank);
            return rank;
        }

        static int num_mpi_ranks(MPI_Comm comm = MPI_COMM_WORLD)
        {
            int size;
            MPI_Comm_size(comm, &size);
            return size;
        }

        static void abort()
        {
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        static int num_threads()
        {
            return omp_get_max_threads();
        }

        static int thread_id()
        {
            return omp_get_thread_num();
        }
        
        static bool verbose()
        {
            return verbose_;
        }

        static void set_verbose(bool verbose__)
        {
            verbose_ = verbose__;
        }

        static int num_fft_threads()
        {
            return num_fft_threads_;
        }

        static void set_num_fft_threads(int num_fft_threads__)
        {
            num_fft_threads_ = num_fft_threads__;
        }

        static void barrier(MPI_Comm comm = MPI_COMM_WORLD)
        {
            MPI_Barrier(comm);
        }

        /// Broadcast array 
        template <typename T>
        static void bcast(T* buffer, int count, const MPI_Comm& comm, int root)
        {
            MPI_Bcast(buffer, count, primitive_type_wrapper<T>::mpi_type_id(), root, comm); 
        }
        
        /// Broadcast array 
        template <typename T>
        static void bcast(T* buffer, int count, int root)
        {
            bcast(buffer, count, MPI_COMM_WORLD, root);
        }
       
        /// Perform the in-place (the output buffer is used as the input buffer) all-to-one reduction 
        template<typename T>
        static void reduce(T* buffer, int count, const MPI_Comm& comm, int root)
        {
            MPI_Reduce(MPI_IN_PLACE, buffer, count, primitive_type_wrapper<T>::mpi_type_id(), MPI_SUM, root, comm);
        }

        /// Perform the in-place (the output buffer is used as the input buffer) all-to-all reduction 
        template<typename T>
        static void allreduce(T* buffer, int count, const MPI_Comm& comm)
        {
            if (comm != MPI_COMM_NULL)
                MPI_Allreduce(MPI_IN_PLACE, buffer, count, primitive_type_wrapper<T>::mpi_type_id(), MPI_SUM, comm);
        }

        /// Perform the in-place (the output buffer is used as the input buffer) all-to-all reduction 
        template<mpi_op_t op, typename T>
        static void allreduce(T* buffer, int count, const MPI_Comm& comm)
        {
            if (comm != MPI_COMM_NULL)
            {
                switch(op)
                {
                    case op_sum:
                        MPI_Allreduce(MPI_IN_PLACE, buffer, count, primitive_type_wrapper<T>::mpi_type_id(), 
                                      MPI_SUM, comm);
                        break;

                    case op_max:
                        MPI_Allreduce(MPI_IN_PLACE, buffer, count, primitive_type_wrapper<T>::mpi_type_id(), 
                                      MPI_MAX, comm);
                        break;
                }
            }
        }
        
        /// Perform the in-place (the output buffer is used as the input buffer) all-to-all reduction 
        template<typename T>
        static void allreduce(T* buffer, int count)
        {
            allreduce(buffer, count, MPI_COMM_WORLD);
        }
        
        /// Perform the in-place (the output buffer is used as the input buffer) all-to-all reduction 
        template<mpi_op_t op, typename T>
        static void allreduce(T* buffer, int count)
        {
            allreduce<op>(buffer, count, MPI_COMM_WORLD);
        }

        static void adjust_heap_allocated(int64_t size)
        {
            #pragma omp critical
            heap_allocated_ += size;
        }

        static int64_t heap_allocated()
        {
            return heap_allocated_;
        }
};

bool Platform::verbose_ = false;
int64_t Platform::heap_allocated_ = 0;
int Platform::num_fft_threads_ = -1;

