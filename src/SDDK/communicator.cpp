#include "SDDK/communicator.hpp"

namespace sddk {
    Communicator sddk::Communicator::cart_create(int ndims__, int const *dims__, int const *periods__) const {
        Communicator new_comm;
        new_comm.mpi_comm_ = std::unique_ptr<MPI_Comm, mpi_comm_deleter>(new MPI_Comm);
        CALL_MPI(MPI_Cart_create, (mpi_comm(), ndims__, dims__, periods__, 0, new_comm.mpi_comm_.get()));
        new_comm.mpi_comm_raw_ = *new_comm.mpi_comm_;
        return new_comm;
    }

    Communicator sddk::Communicator::cart_sub(int const *remain_dims__) const {
        Communicator new_comm;
        new_comm.mpi_comm_ = std::unique_ptr<MPI_Comm, mpi_comm_deleter>(new MPI_Comm);
        CALL_MPI(MPI_Cart_sub, (mpi_comm(), remain_dims__, new_comm.mpi_comm_.get()));
        new_comm.mpi_comm_raw_ = *new_comm.mpi_comm_;
        return new_comm;
    }

    Communicator sddk::Communicator::split(int color__) const {
        Communicator new_comm;
        new_comm.mpi_comm_ = std::unique_ptr<MPI_Comm, mpi_comm_deleter>(new MPI_Comm);
        CALL_MPI(MPI_Comm_split, (mpi_comm(), color__, rank(), new_comm.mpi_comm_.get()));
        new_comm.mpi_comm_raw_ = *new_comm.mpi_comm_;
        return new_comm;
    }

    Communicator sddk::Communicator::duplicate() const {
        Communicator new_comm;
        new_comm.mpi_comm_ = std::unique_ptr<MPI_Comm, mpi_comm_deleter>(new MPI_Comm);
        CALL_MPI(MPI_Comm_dup, (mpi_comm(), new_comm.mpi_comm_.get()));
        new_comm.mpi_comm_raw_ = *new_comm.mpi_comm_;
        return new_comm;
    }

    const Communicator &sddk::Communicator::map_fcomm(int fcomm__) {
        //static std::map<int, std::unique_ptr<Communicator>> fcomm_map;
        static std::map<int, Communicator> fcomm_map;
        if (!fcomm_map.count(fcomm__)) {
            //fcomm_map[fcomm__] = std::unique_ptr<Communicator>(new Communicator(MPI_Comm_f2c(fcomm__)));
            fcomm_map[fcomm__] = Communicator(MPI_Comm_f2c(fcomm__));
        }

        auto &comm = fcomm_map[fcomm__];
        return comm;
    }


    int num_ranks_per_node() {
        static int num_ranks{-1};
        if (num_ranks == -1) {
            char name[MPI_MAX_PROCESSOR_NAME];
            int len;
            CALL_MPI(MPI_Get_processor_name, (name, &len));
            std::vector <size_t> hash(Communicator::world().size());
            hash[Communicator::world().rank()] = std::hash < std::string > {}(std::string(name, len));
            Communicator::world().allgather(hash.data(), Communicator::world().rank(), 1);
            std::sort(hash.begin(), hash.end());

            int n{1};
            for (int i = 1; i < (int) hash.size(); i++) {
                if (hash[i] == hash.front()) {
                    n++;
                } else {
                    break;
                }
            }
            int m{1};
            for (int i = (int) hash.size() - 2; i >= 0; i--) {
                if (hash[i] == hash.back()) {
                    m++;
                } else {
                    break;
                }
            }
            num_ranks = std::max(n, m);
        }

        return num_ranks;
    }

    int get_device_id(int num_devices__) {
        static int id{-1};
        if (num_devices__ == 0) {
            return id;
        }
        if (id == -1) {
#pragma omp single
            {
                int r = Communicator::world().rank();
                char name[MPI_MAX_PROCESSOR_NAME];
                int len;
                CALL_MPI(MPI_Get_processor_name, (name, &len));
                std::vector <size_t> hash(Communicator::world().size());
                hash[r] = std::hash < std::string > {}(std::string(name, len));
                Communicator::world().allgather(hash.data(), r, 1);
                std::map <size_t, std::vector<int>> rank_map;
                for (int i = 0; i < Communicator::world().size(); i++) {
                    rank_map[hash[i]].push_back(i);
                }
                for (int i = 0; i < (int) rank_map[hash[r]].size(); i++) {
                    if (rank_map[hash[r]][i] == r) {
                        id = i % num_devices__;
                        break;
                    }
                }
            }
            assert(id >= 0);
        }
        return id;
    }

    void sddk::pstdout::printf(const char *fmt, ...) {
        std::vector<char> str(1024); // assume that one printf will not output more than this

        std::va_list arg;

        int n = vsnprintf(&str[0], str.size(), fmt, arg);
        va_end(arg);

        n = std::min(n, (int) str.size());

        if ((int) buffer_.size() - count_ < n) {
            buffer_.resize(buffer_.size() + str.size());
        }
        std::memcpy(&buffer_[count_], &str[0], n);
        count_ += n;
    }

    void sddk::pstdout::flush() {
        std::vector<int> counts(comm_.size());
        comm_.allgather(&count_, counts.data(), comm_.rank(), 1);

        int offset{0};
        for (int i = 0; i < comm_.rank(); i++) {
            offset += counts[i];
        }

        /* total size of the output buffer */
        int sz = count_;
        comm_.allreduce(&sz, 1);

        if (sz != 0) {
            std::vector<char> outb(sz + 1);
            comm_.allgather(&buffer_[0], &outb[0], offset, count_);
            outb[sz] = 0;

            if (comm_.rank() == 0) {
                std::printf("%s", &outb[0]);
            }
        }
        count_ = 0;
    }
}
