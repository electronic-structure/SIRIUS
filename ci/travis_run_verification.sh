#!/bin/bash

export PATH=/home/travis/local/bin:/home/travis/reframe/bin:$PATH
git clone https://github.com/eth-cscs/reframe.git /home/travis/reframe
reframe -C ./reframe/config.py  --system=linux -c ./reframe/checks -R -r

