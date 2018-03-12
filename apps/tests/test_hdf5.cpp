#include <sirius.h>

void test1()
{
    sddk::HDF5_tree f("f.h5", hdf5_access_t::truncate);
    
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
    f["aaa"].read("dat_name", dat);
    std::cout << "hash  = " << dat.hash() << std::endl;
    
    f.write("dat_name", dat);
    
}

void test2()
{
    sddk::HDF5_tree f("f.h5", hdf5_access_t::truncate);
    f.create_node("node1");

    mdarray<double, 2> md1(2, 4);
    md1.zero();
    f["node1"].write("md1", md1);
    f["node1"].write(0, md1);

    mdarray<double_complex, 2> md2(2, 4);
    md2.zero();
    f["node1"].write("md2", md2);
    f["node1"].write(1, md2);

    mdarray<int, 2> md3(2, 4);
    md3.zero();
    f["node1"].write("md3", md3);
    f["node1"].write(2, md3);
}

void test3()
{
    sddk::HDF5_tree f("f.h5", hdf5_access_t::read_only);

    mdarray<double, 2> md1(2, 4);
    f["node1"].read("md1", md1);
    f["node1"].read(0, md1);

    mdarray<double_complex, 2> md2(2, 4);
    md2.zero();
    f["node1"].read("md2", md2);
    f["node1"].read(1, md2);

    mdarray<int, 2> md3(2, 4);
    md3.zero();
    f["node1"].read("md3", md3);
    f["node1"].read(2, md3);
}


int main(int argn, char** argv)
{
    test1();
    test1();
    test2();
    test3();
}
