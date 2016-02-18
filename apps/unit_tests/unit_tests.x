#!/bin/bash

tests='test_init test_sht test_fft_correctness test_fft_real test_spline test_rot_ylm test_linalg'

for test in $tests; do
  echo "running '${test}'"
  ./${test}
  err=$?

  if [ ${err} == 0 ]; then
    echo "OK"
  else
    echo "'${test}' failed"
    exit ${err}
  fi
done

echo "All tests were passed correctly!"
