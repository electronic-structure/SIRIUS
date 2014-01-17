#ifndef __SIRIUS_IO_H__
#define __SIRIUS_IO_H__

namespace sirius
{

class pstdout
{
    private:
        
        std::vector<char> buffer_;

        int offset_;

    public:

        pstdout()
        {
            buffer_.resize(8129);
            offset_ = 0;
        }

        void printf(const char* fmt, ...)
        {
            std::vector<char> str(1024); // assume that one printf will not output more than this

            std::va_list arg;
            va_start(arg, fmt);
            int n = vsnprintf(&str[0], str.size(), fmt, arg);
            va_end(arg);

            n = std::min(n, (int)str.size());
            
            if ((int)buffer_.size() - offset_ < n) buffer_.resize(buffer_.size() + str.size());
            memcpy(&buffer_[offset_], &str[0], n);
            offset_ += n;
        }

        void flush(int rank)
        {
            mdarray<int, 2> offsets(Platform::num_mpi_ranks(), 2);
            offsets.zero();
            Platform::allgather(&offset_, &offsets(0, 0), Platform::mpi_rank(), 1); 
            
            for (int i = 1; i < Platform::num_mpi_ranks(); i++) offsets(i, 1) = offsets(i - 1, 1) + offsets(i - 1, 0);
            
            // total size of the output buffer
            int sz = 0;
            for (int i = 0; i < Platform::num_mpi_ranks(); i++) sz += offsets(i, 0);

            std::vector<char> outb(sz + 1);
            Platform::allgather(&buffer_[0], &outb[0], offsets(Platform::mpi_rank(), 1), offset_);
            outb[sz] = 0;

            if (Platform::mpi_rank() == rank) std::printf("%s", &outb[0]);

            offset_ = 0;
        }

        //== static void printf(const char* fmt, ...)
        //== {
        //==     if (Platform::mpi_rank() == 0)
        //==     {
        //==         std::va_list arg;
        //==         va_start(arg, fmt);
        //==         printf(fmt, arg);
        //==         va_end(arg);
        //==     }
        //== }
};
            
class sirius_io
{
    public:

        static void hdf5_write_matrix(const std::string& fname, mdarray<complex16, 2>& matrix)
        {
            static int icount = 0;

            icount++;
            std::stringstream s;
            s << icount;
            std::string full_name = s.str() + "_" + fname;
            
            HDF5_tree fout(full_name, true);
            int size0 = matrix.size(0);
            int size1 = matrix.size(1);
            fout.write("nrow", &size0); 
            fout.write("ncol", &size1);
            fout.write_mdarray("matrix", matrix);
        }


};

}

#endif
