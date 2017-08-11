#!/bin/bash

for f in ./*; do
  if [ -d "$f" ]; then
    echo "running '${f}'"
    cd ${f}
    mpirun -np 4 ../../apps/dft_loop/sirius.scf --test_against=output_ref.json --std_evp_solver_name=scalapack --gen_evp_solver_name=scalapack --processing_unit=gpu --mpi_grid="2 2"
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
