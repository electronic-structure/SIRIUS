#ifndef __SIRIUS_IO_H__
#define __SIRIUS_IO_H__

namespace sirius
{

class pstdout
{
    private:
        
        mdarray<char, 1> buffer_;

        int offset_;

    public:

        pstdout(int size = 8192)
        {
            buffer_.set_dimensions(size);
            buffer_.allocate();
            buffer_.zero();
            offset_ = 0;
        }

        void printf(const char* fmt, ...)
        {
            std::va_list arg;
            va_start(arg, fmt);
            offset_ += vsnprintf(&buffer_(offset_), buffer_.size() - offset_, fmt, arg);
            va_end(arg);
        }

        void flush(int rank)
        {
            mdarray<int, 2> offsets(Platform::num_mpi_ranks(), 2);
            offsets.zero();
            Platform::allgather(&offset_, &offsets(0, 0), Platform::mpi_rank(), 1); 
            
            for (int i = 1; i < Platform::num_mpi_ranks(); i++) offsets(i, 1) = offsets(i - 1, 1) + offsets(i - 1, 0);

            mdarray<char, 1> outb((int)buffer_.size() * Platform::num_mpi_ranks());
            outb.zero();

            Platform::allgather(&buffer_(0), &outb(0), offsets(Platform::mpi_rank(), 1), offset_);  

            if (Platform::mpi_rank() == rank) std::printf("%s", outb.get_ptr());

            offset_ = 0;
            buffer_.zero();
        }
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
