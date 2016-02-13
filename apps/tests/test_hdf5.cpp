#include <sirius.h>

void test1()
{
    sirius::HDF5_tree f("f.h5", true);
    
    mdarray<double, 2> dat(2, 4);
    dat.zero();
    
    dat(0, 0) = 1.1;
    dat(0, 1) = 2.2;
    dat(0, 2) = 3.3;
    dat(1, 0) = 4.4;
    dat(1, 1) = 5.5;
    dat(1, 2) = 6.6;
    
    std::cout << "hash  = " << dat.hash() << std::endl;
    
    f.create_node("aaa");
    f["aaa"].write("dat_name", dat);
    dat.zero();
    f["aaa"].read_mdarray("dat_name", dat);
    std::cout << "hash  = " << dat.hash() << std::endl;
    
    f.write("dat_name", dat);
    
}


int main(int argn, char** argv)
{
    test1();
    test1();

}
