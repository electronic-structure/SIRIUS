#!/bin/bash

tests='test_init test_sht'

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
