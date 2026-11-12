# Basic information
All input parameters that SIRIUS understands are defined in the `input_schema.json` file. If you need to introduce
new parameters, edit the schema and then generate new `config.hpp` and `config.cpp` by running
```
python3 ../../apps/utils/gen_config.py input_schema.json
clang-format -i config.cpp
rm DOC.md
```
`config.hpp` contains only the declarations of the accessors and needs only the forward declarations of
`nlohmann::json`; the definitions are in `config.cpp`.

Input schema is converted into const json dictionary inside SIRIUS with the following cmake command:
```
# generate schema
file(READ "${PROJECT_SOURCE_DIR}/src/context/input_schema.json" SIRIUS_INPUT_SCHEMA NEWLINE_CONSUME)
configure_file("${PROJECT_SOURCE_DIR}/src/context/input_schema.hpp.in"
               "${PROJECT_BINARY_DIR}/src/context/input_schema.hpp"
               @ONLY)
```
If you change input schema, please make sure that you delete the previously generated file in the build directory:
```
rm ./build/src/context/input_schema.hpp
```
