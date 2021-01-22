# The build image should have dependencies installed
ARG BUILD_BASE=ubuntu:18.04

# This is stripped down deployment image
ARG DEPLOY_BASE=ubuntu:18.04

FROM $BUILD_BASE as builder

# Spack setup: $SPEC is the version of SIRIUS we want to build
#              $ENVIRONMENT is the path to the environment file
ARG SPEC
ARG ENVIRONMENT

COPY . /sources

SHELL ["/bin/bash", "-c"]

# Setup spack and install SIRIUS
RUN spack --color always repo rm --scope site user && \
    spack --color always repo add /sources/spack && \
    spack --color always env create --without-view ci_run "/sources/$ENVIRONMENT" && \
    spack --color always -e ci_run spec $SPEC && \
    spack --color always -e ci_run dev-build --source-path /sources $SPEC

# Bundling: this is a bit more complicated than it should unfortunately :(
# We want to run `ctest`, but ctest does not work well after running `make install`
# So instead we don't install, but prune the build folder as much as possible.
# We move the libs to `/root/sirius.bundle/lib` and copy `ctest` to `/root/sirius.bundle/bin`.
RUN cd /sources/spack-build && \
    export TEST_BINARIES=`spack -e ci build-env $SPEC -- ctest --show-only=json-v1 | jq '.tests | map(.command[0]) | .[]' | tr -d \" | uniq` && \
    export CTEST=`spack -e ci build-env $SPEC -- which ctest` && \
    libtree -d /root/sirius.bundle ${TEST_BINARIES} && \
    rm -rf /root/sirius.bundle/bin && \
    libtree -d /root/sirius.bundle ${CTEST} && \
    mkdir /sources/spack-build-tmp && \
    echo "$TEST_BINARIES" | xargs -I{file} find -samefile {file} -exec cp --parents '{}' /sources/spack-build-tmp ';' && \
    find -name CTestTestfile.cmake -exec cp --parent '{}' /sources/spack-build-tmp ';' && \
    rm -rf /sources/spack-build && \
    mv /sources/spack-build-tmp /sources/spack-build

# Create a stripped down image
FROM $DEPLOY_BASE

# Make nvidia happy
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=7.5"

# Make some things available in the path
ENV PATH="/root/sirius.bundle/usr/bin:$PATH"

COPY --from=builder /root/sirius.bundle /root/sirius.bundle
COPY --from=builder /sources/verification /sources/verification
COPY --from=builder /sources/spack-build /sources/spack-build

# Make sarus happy
RUN echo "/root/sirius.bundle/usr/lib/" > /etc/ld.so.conf.d/sirius.conf && ldconfig
