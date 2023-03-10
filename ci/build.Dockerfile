ARG BASE_IMAGE
FROM $BASE_IMAGE

ARG SPEC

RUN spack spec $SPEC

COPY . /sirius-src

RUN spack --color always -e sirius-env dev-build --source-path /sirius-src $SPEC

RUN cd /sirius-src && ln -s $(find . -name "spack-build-*" -type d) spack-build && ls

#cd /sirius-src && mkdir build && cd build && spack -e sirius-env build-env cmake .. && make
