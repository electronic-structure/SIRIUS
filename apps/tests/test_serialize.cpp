#include <sirius.h>

using namespace sirius;

void test1()
{
    int a1{100};
    double b1{200.88};
    
    std::vector<std::complex<float>> c1;
    c1.push_back(std::complex<float>(1, 2));
    c1.push_back(std::complex<float>(3, 4));

    std::complex<double> d1(4,10);

    mdarray<double, 2> m1(4, 5);
    m1.zero();
    m1(0, 0) = 1.1;
    m1(3, 4) = 100.1;

    serializer s;

    serialize(s, a1);
    serialize(s, b1);
    serialize(s, c1);
    serialize(s, d1);
    serialize(s, m1);

    int a2;
    double b2;
    std::vector<std::complex<float>> c2;
    std::complex<double> d2;
    mdarray<double, 2> m2;

    deserialize(s, a2);
    std::cout << a2 << "\n";
    deserialize(s, b2);
    std::cout << b2 << "\n";
    deserialize(s, c2);
    std::cout << c2[0] << "\n";
    std::cout << c2[1] << "\n";
    deserialize(s, d2);
    deserialize(s, m2);
    std::cout << m1(0, 0) << " " << m1(3, 4) << "\n";

};

void test2()
{
    vector3d<double> vk{1.1, 2.2, 3.3};
    serializer s;
    serialize(s, vk);

    vector3d<double> vk1;
    deserialize(s, vk1);
    std::cout << vk << " " << vk1 << "\n";

    std::cout << std::is_pod<vector3d<double>>::value << "\n";
    std::cout << sizeof(vk) << "\n";

    std::vector<double> v3({1, 2, 3, 4, 5, 6, 7, 8});
    std::cout << sizeof(v3) << "\n";
}


int main(int argn, char** argv)
{
    cmd_args args;

    args.parse_args(argn, argv);
    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }

    sirius::initialize(1);
    test1();
    test2();
    sirius::finalize();
}
