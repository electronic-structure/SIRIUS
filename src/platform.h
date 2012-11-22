class Platform
{
    private:

        static bool verbose_;
    
    public:

        static void initialize(bool call_mpi_init__)
        {
            if (call_mpi_init__) MPI_Init(NULL, NULL);
            //MPI_Comm_size(MPI_COMM_WORLD, &size_);
            //MPI_Comm_rank(MPI_COMM_WORLD, &rank_);

            // rank 0 prints by default
            verbose_ = false;
            if (mpi_rank() == 0) verbose_ = true;
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
        template<typename T>
        static void allreduce(T* buffer, int count)
        {
            allreduce(buffer, count, MPI_COMM_WORLD);
        }
};

bool Platform::verbose_ = false;

