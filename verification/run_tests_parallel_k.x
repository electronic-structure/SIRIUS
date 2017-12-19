#!/bin/bash

for f in ./*; do
  if [ -d "$f" ]; then
    echo "running '${f}'"
    cd ${f}
    mpirun -np 2 ../../apps/dft_loop/sirius.scf --test_against=output_ref.json
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
