# Basic information
All input parameters that SIRIUS understands are defined in the `input_schema.json` file. If you need to introduce
new parameters, edit the schema and then generate a new `config.hpp` by running
```
python3 ../../apps/utils/gen_input_struct.py input_schema.json &> config.hpp
```

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
