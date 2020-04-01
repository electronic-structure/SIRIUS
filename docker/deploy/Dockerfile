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

FROM ubuntu:18.04

# This is the only thing necessary really from nvidia/cuda's ubuntu18.04 runtime image
ENV NVIDIA_VISIBLE_DEVICES all

ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

ENV NVIDIA_REQUIRE_CUDA "cuda>=10.1 brand=tesla,driver>=384,driver<385 brand=tesla,driver>=396,driver<397 brand=tesla,driver>=410,driver<411"

COPY --from=builder /root/SIRIUS.AppDir /root/SIRIUS.AppDir

# Make it easy to call our binaries.
ENV PATH="/root/SIRIUS.AppDir/usr/bin:$PATH"

WORKDIR /root/SIRIUS.AppDir/usr/bin
