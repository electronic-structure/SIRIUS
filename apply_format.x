#!/bin/bash

find ./src -name \*.cpp -o -name \*.hpp ! -name nlohmann_json.hpp ! -name lebedev_grids.hpp ! -name config.hpp | python3 check_format.py -f
find ./apps -name \*.cpp -o -name \*.hpp | python3 check_format.py -f
find ./python_module -maxdepth 1 -name \*.cpp -o -name \*.hpp | python3 check_format.py -f
