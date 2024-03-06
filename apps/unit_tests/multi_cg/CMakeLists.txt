find_package (Eigen3 3.3 REQUIRED NO_MODULE)
 
add_executable (test_multi_cg test_multi_cg.cpp)
target_link_libraries (test_multi_cg Eigen3::Eigen sirius_cxx)
add_test(NAME test_multi_cg COMMAND test_multi_cg)

add_executable (test_multi_cg_multiprecision test_multi_cg_multiprecision.cpp)
target_link_libraries (test_multi_cg_multiprecision Eigen3::Eigen sirius_cxx)
add_test(NAME test_multi_cg_multiprecision COMMAND test_multi_cg_multiprecision)

add_executable (test_multi_cg_complex test_multi_cg_complex.cpp)
target_link_libraries (test_multi_cg_complex Eigen3::Eigen sirius_cxx)
add_test(NAME test_multi_cg_complex COMMAND test_multi_cg_complex)
