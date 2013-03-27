#ifndef __SIRIUS_IO_H__
#define __SIRIUS_IO_H__

namespace sirius
{

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
            
            hdf5_tree fout(full_name, true);
            int size0 = matrix.size(0);
            int size1 = matrix.size(1);
            fout.write("nrow", &size0); 
            fout.write("ncol", &size1);
            fout.write("matrix", matrix);
        }


};

}

#endif
