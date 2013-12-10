class Platform
{
    private:

        static int num_fft_threads_;
    
    public:

        static void initialize(bool call_mpi_init, bool call_cublas_init = true)
        {
            if (call_mpi_init) MPI_Init(NULL, NULL);

            #ifdef _GPU_
            if (call_cublas_init) cublas_init();
            if (mpi_rank() == 0) cuda_device_info();
            #endif
            #ifdef _MAGMA_
            magma_init_wrapper();
            #endif

            assert(sizeof(int) == 4);
            assert(sizeof(double) == 8);
        }

        static void finalize()
        {
            MPI_Finalize();
            #ifdef _MAGMA_
            magma_finalize_wrapper();
            #endif
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
            if (num_mpi_ranks() == 1)
            {
                raise(SIGTERM);
            }
            else
            {   
                MPI_Abort(MPI_COMM_WORLD, -13);
            }
            exit(-13);
        }

        static int num_threads()
        {
            return omp_get_max_threads();
        }

        static int thread_id()
        {
            return omp_get_thread_num();
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
       
        /// Perform all-to-one in-place reduction
        template<typename T>
        static void reduce(T* buf, int count, const MPI_Comm& comm, int root)
        {
            T* buf_tmp = (T*)malloc(count * sizeof(T));
            MPI_Reduce(buf, buf_tmp, count, primitive_type_wrapper<T>::mpi_type_id(), MPI_SUM, root, comm);
            memcpy(buf, buf_tmp, count * sizeof(T));
            free(buf_tmp);
        }

        /// Perform all-to-one out-of-place reduction 
        template<typename T>
        static void reduce(T* sendbuf, T* recvbuf, int count, const MPI_Comm& comm, int root)
        {
            MPI_Reduce(sendbuf, recvbuf, count, primitive_type_wrapper<T>::mpi_type_id(), MPI_SUM, root, comm);
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
                    {
                        MPI_Allreduce(MPI_IN_PLACE, buffer, count, primitive_type_wrapper<T>::mpi_type_id(), 
                                      MPI_SUM, comm);
                        break;
                    }
                    case op_max:
                    {
                        MPI_Allreduce(MPI_IN_PLACE, buffer, count, primitive_type_wrapper<T>::mpi_type_id(), 
                                      MPI_MAX, comm);
                        break;
                    }
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

        template<typename T>
        static void allgather(T* sendbuf, T* recvbuf, int offset, int count)
        {
            std::vector<int> counts(num_mpi_ranks());
            counts[mpi_rank()] = count;
            MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &counts[0], 1, primitive_type_wrapper<int>::mpi_type_id(), 
                          MPI_COMM_WORLD);
            
            std::vector<int> offsets(num_mpi_ranks());
            offsets[mpi_rank()] = offset;
            MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &offsets[0], 1, primitive_type_wrapper<int>::mpi_type_id(), 
                          MPI_COMM_WORLD);

            MPI_Allgatherv(sendbuf, count, primitive_type_wrapper<T>::mpi_type_id(), recvbuf, &counts[0], &offsets[0],
                           primitive_type_wrapper<T>::mpi_type_id(), MPI_COMM_WORLD);
        }
        
        template<typename T>
        static void allgather(T* buf, int offset, int count, MPI_Comm comm = MPI_COMM_WORLD)
        {

            std::vector<int> v(num_mpi_ranks(comm) * 2);
            v[2 * mpi_rank(comm)] = count;
            v[2 * mpi_rank(comm) + 1] = offset;

            MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &v[0], 2, primitive_type_wrapper<int>::mpi_type_id(), comm);

            std::vector<int> counts(num_mpi_ranks(comm));
            std::vector<int> offsets(num_mpi_ranks(comm));

            for (int i = 0; i < num_mpi_ranks(comm); i++)
            {
                counts[i] = v[2 * i];
                offsets[i] = v[2 * i + 1];
            }

            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, buf, &counts[0], &offsets[0],
                           primitive_type_wrapper<T>::mpi_type_id(), comm);
        }
};

int Platform::num_fft_threads_ = -1;

