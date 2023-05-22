ARG BASE_IMAGE
FROM $BASE_IMAGE

ARG SPECDEV

# show the spack's spec
RUN spack spec -I $SPECDEV

RUN spack env create --with-view /opt/sirius sirius-env
RUN spack -e sirius-env add $SPECDEV

# copy source files of the pull request into container
COPY . /sirius-src

# build SIRIUS
RUN spack --color always -e sirius-env dev-build --source-path /sirius-src $SPECDEV

# we need a fixed name for the build directory
# here is a hacky workaround to link ./spack-build-{hash} to ./spack-build
RUN cd /sirius-src && ln -s $(find . -name "spack-build-*" -type d) spack-build
