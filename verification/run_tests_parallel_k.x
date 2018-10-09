#!/bin/bash

if [ -z "$SIRIUS_BINARIES" ];
then
    export SIRIUS_BINARIES=../../apps/dft_loop
fi

if [[ $HOST == nid* ]]; then
    SRUN_CMD=srun
else
    SRUN_CMD="mpi -np 2"
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
