#!/bin/bash

export PATH=$HOME/local/bin:$HOME/reframe/bin:$PATH
git clone https://github.com/eth-cscs/reframe.git $HOME/reframe
reframe -C ./reframe/config.py --system=linux:cpu -c ./reframe/checks -R -r --tag serial --exec-policy=serial --skip-prgenv-check

