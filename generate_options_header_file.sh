#!/bin/bash

echo "const json all_options_dictionary_ = \"" > tmp.json && sed -e 's/\"/\\"/g' $1 >> tmp.json && echo "\"_json;" >> tmp.json
cat tmp.json | tr '\n' ' '
