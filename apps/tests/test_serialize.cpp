/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.h>

using namespace sirius;

int
test1()
{
    int a1{100};
    double b1{200.88};

    std::vector<std::complex<float>> c1;
    c1.push_back(std::complex<float>(1.2f, 2.3f));
    c1.push_back(std::complex<float>(3.5f, 4.6f));
    c1.push_back(std::complex<float>(-3.14f, -6.99f));

    std::complex<double> d1(4.464663636, 10.37374992921);

    mdarray<double, 2> m1(4, 5);
    m1 = [](uint64_t i1, uint64_t i2) { return type_wrapper<double>::random(); };

    vector3d<double> v1{1.1, 2.2, 3.3};
    matrix3d<double> u1{{4.4, 5.5, 6.6}, {1.2, 2.43334, 4.56666}, {400.333, 1e14, 2.33e20}};

    serializer s;

    serialize(s, a1);
    serialize(s, b1);
    serialize(s, c1);
    serialize(s, d1);
    serialize(s, m1);
    serialize(s, v1);
    serialize(s, u1);

    int a2;
    double b2;
    std::vector<std::complex<float>> c2;
    std::complex<double> d2;
    mdarray<double, 2> m2;
    vector3d<double> v2;
    matrix3d<double> u2;

    deserialize(s, a2);
    deserialize(s, b2);
    deserialize(s, c2);
    deserialize(s, d2);
    deserialize(s, m2);
    deserialize(s, v2);
    deserialize(s, u2);

    if (a1 != a2) {
        return 1;
    }
    if (b1 != b2) {
        return 1;
    }
    for (int i = 0; i < 3; i++) {
        if (c1[i] != c2[i]) {
            return 1;
        }
    }
    if (d1 != d2) {
        return 1;
    }
    for (int i = 0; i < 3; i++) {
        if (v1[i] != v2[i]) {
            return 1;
        }
    }
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (u1(i, j) != u2(i, j)) {
                return 1;
            }
        }
    }

    return 0;
};

int
main(int argn, char** argv)
{
    cmd_args args;

    args.parse_args(argn, argv);
    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }

    sirius::initialize(1);
    printf("%-30s", "testing serialization: ");
    int result = test1();
    if (result) {
        printf("\x1b[31m"
               "Failed"
               "\x1b[0m"
               "\n");
    } else {
        printf("\x1b[32m"
               "OK"
               "\x1b[0m"
               "\n");
    }
    sirius::finalize();

    return result;
}
