#!/bin/bash

export PATH=$HOME/local/bin:$HOME/reframe/bin:$PATH
git clone -b v3.1 --depth=1 https://github.com/eth-cscs/reframe.git $HOME/reframe
pushd $HOME/reframe
./bootstrap.sh
popd
reframe -C ./reframe/config.py --system=linux:cpu -c ./reframe/checks -R -r --tag serial --exec-policy=serial --skip-prgenv-check

