#!/bin/bash

if [ -z "$SIRIUS_BINARIES" ];
then
    export SIRIUS_BINARIES=$(pwd)/../build/apps/dft_loop
fi

if [[ $(type -f srun 2> /dev/null) ]]; then
    SRUN_CMD="srun -u"
else
    SRUN_CMD="mpirun -np 2"
fi


exe=${SIRIUS_BINARIES}/sirius.scf
# check if path is correct
type -f ${exe} || exit 1

for f in ./*; do
    if [ -d "$f" ]; then
        echo "running '${f}'"
        (
            cd ${f}
            ${SRUN_CMD} ${exe} --test_against=output_ref.json
            err=$?

            if [ ${err} == 0 ]; then
                echo "OK"
            else
                echo "'${f}' failed"
                exit ${err}
            fi
        )
    fi
done

echo "All tests were passed correctly!"
