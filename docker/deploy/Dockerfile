ARG BUILD_IMAGE

ARG BUILD_TAG

FROM $BUILD_IMAGE:$BUILD_TAG as builder

FROM ubuntu:18.04

# We do multistage builds, but want to save the
# previous container as well since it serves as
# a nice build cache
COPY --from=builder /root/SIRIUS.AppDir /root/SIRIUS.AppDir

ENV PATH="/root/SIRIUS.AppDir/usr/bin:$PATH"

WORKDIR /root/SIRIUS.AppDir/usr/bin
