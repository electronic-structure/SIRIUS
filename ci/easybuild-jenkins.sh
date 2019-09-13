#!/bin/bash -l

# compile easybuild modules required by jenkins to build SIRIUS
# required easyblocks which can be found in the following git repo
# git clone https://github.com/simonpintarelli/production.git
# clone it into $HOME/production

module purge
export EB_CUSTOM_PREFIX=${HOME}/production/easybuild
export EASYBUILD_PREFIX=${HOME}/jenkins/daint-haswell

module load modules
module load craype
module load PrgEnv-gnu
module load daint-gpu
module load git
module unload cray-libsci
module load EasyBuild-custom/cscs

eb libxc-4.3.4-CrayGNU-18.08.eb -r
eb GSL-2.6-CrayGNU-18.08.eb -r
eb spglib-1.14.1-CrayGNU-18.08.eb -r
eb magma-2.4.0-CrayGNU-18.08-cuda-9.1.eb -r
eb SpFFT-0.9.3-CrayGNU-18.08-cuda-9.1.eb -r

# make newly installed modules world readable
find ${HOME}/jenkins -type d -exec chmod ao+rx {} \;
find ${HOME}/jenkins -type f -exec chmod ao+r {} \;
