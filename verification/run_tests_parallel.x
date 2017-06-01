#!/bin/bash

tests='test1 test2 test3 test4 test5 test6'

for test in $tests; do
  echo "running '${test}'"
  cd ${test}
  mpirun -np 4 ../../apps/dft_loop/sirius.scf --test_against=output_ref.json --std_evp_solver_name=scalapack --gen_evp_solver_name=scalapack --mpi_grid="2 2"
  err=$?

  if [ ${err} == 0 ]; then
    echo "OK"
  else
    echo "'${test}' failed"
    exit ${err}
  fi
  cd ../
done

echo "All tests were passed correctly!"
