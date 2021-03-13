#!/bin/bash

if [ -z "$SIRIUS_BINARIES" ];
then
    export SIRIUS_BINARIES=$(pwd)/../build/apps/dft_loop
fi

exe=${SIRIUS_BINARIES}/sirius.scf
# check if path is correct
type -f ${exe} || exit 1

for f in ./test*; do
    if [ -d "$f" ]; then
        echo "running '${f}'"
        cd ${f}
        ${exe} --test_against=output_ref.json --control.processing_unit=gpu
        err=$?

        if [ ${err} == 0 ]; then
            echo "OK"
        else
            echo "'${f}' failed"
            exit ${err}
        fi
        cd ../
    fi
done

echo "All tests were passed correctly!"
