ARG BASE_IMAGE
FROM $BASE_IMAGE

ARG ENVPATH

# copy source files of the pull request into container
COPY . /sirius-src

# build SIRIUS
RUN spack -e $ENVPATH install

# # show the spack's spec
RUN spack -e $ENVPATH find -lcdv sirius

# we need a fixed name for the build directory
# here is a hacky workaround to link ./spack-build-{hash} to ./spack-build
RUN cd /sirius-src && ln -s $(spack -e $ENVPATH location -b sirius) spack-build
