#!/bin/bash

tests='test1 test2 test3 test4 test5 test6 test7'

for test in $tests; do
  echo "running '${test}'"
  cd ${test}
  ../../apps/dft_loop/sirius.scf --test_against=output_ref.json
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
