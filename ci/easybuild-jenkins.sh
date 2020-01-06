#!/bin/bash -l

# compile easybuild modules required by jenkins to build SIRIUS
# required easyblocks which can be found in the following git repo
# git clone https://github.com/simonpintarelli/production.git
# clone it into $HOME/production

export EASYBUILD_PREFIX=/apps/daint/SSL/sirius-jenkins/daint-haswell

mkdir -p ${EASYBUILD_PREFIX}

module load modules
module load daint-gpu
module load EasyBuild-custom/cscs

(
    set -e
    cd easybuild
    eb libxc-4.3.4-CrayGNU-19.10.eb -r
    eb GSL-2.5-CrayGNU-19.10.eb -r
    eb spglib-1.14.1-CrayGNU-19.10.eb -r
    eb magma-2.5.1-CrayGNU-19.10-cuda-10.1.eb -r
    eb SpFFT-0.9.8-CrayGNU-19.10-cuda-10.1.eb -r
    eb mpi4py-3.0.2-CrayGNU-19.10-python3-cuda10.1.eb -r
    eb ELPA-2019.05.001-CrayGNU-19.10.eb -r
)

chmod ao+rx ${EASYBUILD_PREFIX}
# make newly installed modules world readable
find ${EASYBUILD_PREFIX} -type d -exec chmod ao+rx {} \;
find ${EASYBUILD_PREFIX} -type f -exec chmod ao+r {} \;
