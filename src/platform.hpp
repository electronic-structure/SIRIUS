template <typename T>
void Platform::bcast(T* buffer, int count, const MPI_Comm& comm, int root)
{
    MPI_Bcast(buffer, count, type_wrapper<T>::mpi_type_id(), root, comm); 
}
        
template <typename T>
void Platform::bcast(T* buffer, int count, int root)
{
    bcast(buffer, count, MPI_COMM_WORLD, root);
}
       
template<typename T>
void Platform::reduce(T* buf, int count, const MPI_Comm& comm, int root)
{
    T* buf_tmp = (T*)malloc(count * sizeof(T));
    MPI_Reduce(buf, buf_tmp, count, type_wrapper<T>::mpi_type_id(), MPI_SUM, root, comm);
    memcpy(buf, buf_tmp, count * sizeof(T));
    free(buf_tmp);
}

template<typename T>
void Platform::reduce(T* sendbuf, T* recvbuf, int count, const MPI_Comm& comm, int root)
{
    MPI_Reduce(sendbuf, recvbuf, count, type_wrapper<T>::mpi_type_id(), MPI_SUM, root, comm);
}
        
template<typename T>
void Platform::allreduce(T* buffer, int count, const MPI_Comm& comm)
{
    if (comm != MPI_COMM_NULL)
        MPI_Allreduce(MPI_IN_PLACE, buffer, count, type_wrapper<T>::mpi_type_id(), MPI_SUM, comm);
}

template<mpi_op_t op, typename T>
void Platform::allreduce(T* buffer, int count, const MPI_Comm& comm)
{
    if (comm != MPI_COMM_NULL)
    {
        switch(op)
        {
            case op_sum:
            {
                MPI_Allreduce(MPI_IN_PLACE, buffer, count, type_wrapper<T>::mpi_type_id(), MPI_SUM, comm);
                break;
            }
            case op_max:
            {
                MPI_Allreduce(MPI_IN_PLACE, buffer, count, type_wrapper<T>::mpi_type_id(), MPI_MAX, comm);
                break;
            }
        }
    }
}
        
template<typename T>
void Platform::allreduce(T* buffer, int count)
{
    allreduce(buffer, count, MPI_COMM_WORLD);
}
        
template<mpi_op_t op, typename T>
void Platform::allreduce(T* buffer, int count)
{
    allreduce<op>(buffer, count, MPI_COMM_WORLD);
}

template<typename T>
void Platform::allgather(T* sendbuf, T* recvbuf, int offset, int count)
{
    std::vector<int> counts(num_mpi_ranks());
    counts[mpi_rank()] = count;
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &counts[0], 1, type_wrapper<int>::mpi_type_id(), MPI_COMM_WORLD);
    
    std::vector<int> offsets(num_mpi_ranks());
    offsets[mpi_rank()] = offset;
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &offsets[0], 1, type_wrapper<int>::mpi_type_id(), MPI_COMM_WORLD);

    MPI_Allgatherv(sendbuf, count, type_wrapper<T>::mpi_type_id(), recvbuf, &counts[0], &offsets[0],
                   type_wrapper<T>::mpi_type_id(), MPI_COMM_WORLD);
}
        
template<typename T>
void Platform::allgather(T* buf, int offset, int count, MPI_Comm comm)
{

    std::vector<int> v(num_mpi_ranks(comm) * 2);
    v[2 * mpi_rank(comm)] = count;
    v[2 * mpi_rank(comm) + 1] = offset;

    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &v[0], 2, type_wrapper<int>::mpi_type_id(), comm);

    std::vector<int> counts(num_mpi_ranks(comm));
    std::vector<int> offsets(num_mpi_ranks(comm));

    for (int i = 0; i < num_mpi_ranks(comm); i++)
    {
        counts[i] = v[2 * i];
        offsets[i] = v[2 * i + 1];
    }

    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, buf, &counts[0], &offsets[0],
                   type_wrapper<T>::mpi_type_id(), comm);
}
