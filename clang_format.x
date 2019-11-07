#!/bin/sh

rm -rf $1.fmt

sed 's/#pragma omp/\/\/#pragma omp/g' $1 | clang-format-mp-7.0 -style=file  | sed 's/\/\/ *#pragma omp/#pragma omp/g' >> $1.fmt

