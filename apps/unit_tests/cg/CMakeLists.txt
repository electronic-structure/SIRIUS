find_package (Eigen3 3.3 REQUIRED NO_MODULE)
 
add_executable (test_cg test_cg.cpp)
target_link_libraries (test_cg Eigen3::Eigen sirius)
add_test(NAME test_cg COMMAND test_cg)