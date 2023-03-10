ARG SPEC
ARG BASE_IMAGE
FROM $BASE_IMAGE

RUN spack spec $SPEC

COPY . /sirius-src

RUN spack --color always -e sirius-env dev-build --source-path /sirius-src $SPEC

RUN cd $(spack -e sirius-env location -b $SPEC) && ls

#cd /sirius-src && mkdir build && cd build && spack -e sirius-env build-env cmake .. && make
