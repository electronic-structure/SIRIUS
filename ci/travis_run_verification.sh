#!/bin/bash

export PATH=/home/travis/local/bin:/home/travis/reframe/bin:$PATH
git clone https://github.com/eth-cscs/reframe.git /home/travis/reframe
reframe -C ./reframe/config.py --system=linux:cpu -c ./reframe/checks -R -r --tag serial --exec-policy=serial

