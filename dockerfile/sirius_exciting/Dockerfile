FROM sirius_base as builder

# build latest develop version of SIRIUS
RUN spack -e sirius-env install

# needed for Exciting
RUN apt-get install -y xsltproc

# copy source code and build scripts
COPY exciting /exciting
COPY build_with_sirius_mkl.x /build_with_sirius_mkl.x
COPY make.inc /exciting/build/make.inc
RUN /build_with_sirius_mkl.x

RUN apt-get install libtree
RUN mkdir stage-copy
# use libtree utility to find all shared libraries of a binary (libtree -p ...)
# extract the full path of the *so files (grep -o ...)
# copy (cp --parent ...) with the directory structure into /stage-copy directory
RUN libtree -p /exciting/build/mpiandsmp/exciting | grep -o '/[-a-zA-Z0-9_/\.]*\.so[\.0-9]*' | xargs -I{} cp --parent {} stage-copy

# Bare OS image to run the installed executables
#FROM sirius_base
FROM ubuntu:22.04
RUN apt-get -y update
# add helpful utilities to the slim image
RUN apt-get install -y apt-utils libtree

# copy main executable
COPY --from=builder /exciting/build/mpiandsmp/exciting /exciting
#COPY --from=builder /opt/local /opt/local
COPY --from=builder /stage-copy/opt /opt
COPY --from=builder /stage-copy/lib /usr/lib

# For the sarus mpi hook
COPY --from=builder /etc/ld.so.conf.d/mpich.conf /etc/ld.so.conf.d/mpich.conf
RUN ldconfig

