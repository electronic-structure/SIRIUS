ARG BUILD_IMAGE

ARG BUILD_TAG

FROM $BUILD_IMAGE:$BUILD_TAG as builder

# Add bundle tooling
RUN wget https://github.com/linuxdeploy/linuxdeploy/releases/download/continuous/linuxdeploy-x86_64.AppImage

# Working with appimages requires fuse, which inside docker requires it to be mounted and --cap-add SYS_ADMIN
# so instead just extract the appimage.
RUN chmod +x linuxdeploy-x86_64.AppImage && ./linuxdeploy-x86_64.AppImage --appimage-extract

# And run it like this.
RUN /root/squashfs-root/AppRun --appdir SIRIUS.AppDir/ -e SIRIUS.AppDir/usr/bin/sirius.scf -i /build-sirius/deploy/SIRIUS.png -d /build-sirius/deploy/SIRIUS.desktop

# Remove the cuda libs, they are in the base image already
RUN rm -rf SIRIUS.AppDir/usr/lib/libcu*

FROM nvidia/cuda:10.1-devel-ubuntu18.04

# We do multistage builds, but want to save the
# previous container as well since it serves as
# a nice build cache
COPY --from=builder /root/SIRIUS.AppDir /root/SIRIUS.AppDir

# NVIDIA messed up the LD_LIBRARY_PATH; it's set to /usr/local/nvidia/lib:/usr/local/nvidia/lib64 which don't exist :/.
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64"

# Make it easy to call our binaries.
ENV PATH="/root/SIRIUS.AppDir/usr/bin:$PATH"

WORKDIR /root/SIRIUS.AppDir/usr/bin
