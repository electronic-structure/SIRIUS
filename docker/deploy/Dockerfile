ARG BUILD_IMAGE

ARG BUILD_TAG

FROM $BUILD_IMAGE:$BUILD_TAG as builder

COPY ./deploy /root/deploy

# Add bundle tooling
RUN wget -q https://github.com/linuxdeploy/linuxdeploy/releases/download/continuous/linuxdeploy-x86_64.AppImage

# Working with appimages requires fuse, which inside docker requires it to be mounted and --cap-add SYS_ADMIN
# so instead just extract the appimage.
RUN chmod +x linuxdeploy-x86_64.AppImage && ./linuxdeploy-x86_64.AppImage --appimage-extract

# Run linuxdeploy, and add a bunch of libs that are dlopen'ed by mkl
RUN /root/squashfs-root/AppRun --appdir SIRIUS.AppDir/ \
        -e SIRIUS.AppDir/usr/bin/sirius.scf \
        -i /root/deploy/SIRIUS.png \
        -d /root/deploy/SIRIUS.desktop \
        -l /opt/intel/compilers_and_libraries/linux/mkl/lib/intel64/libmkl_tbb_thread.so \
        -l /opt/intel/compilers_and_libraries/linux/mkl/lib/intel64/libmkl_intel_thread.so \
        -l /opt/intel/compilers_and_libraries/linux/mkl/lib/intel64/libmkl_sequential.so \
        -l /opt/intel/compilers_and_libraries/linux/mkl/lib/intel64/libmkl_def.so \
        -l /opt/intel/compilers_and_libraries/linux/mkl/lib/intel64/libmkl_vml_def.so \
        -l /opt/intel/compilers_and_libraries/linux/mkl/lib/intel64/libmkl_vml_cmpt.so \
        -l /opt/intel/compilers_and_libraries/linux/mkl/lib/intel64/libmkl_mc.so \
        -l /opt/intel/compilers_and_libraries/linux/mkl/lib/intel64/libmkl_vml_mc.so \
        -l /opt/intel/compilers_and_libraries/linux/mkl/lib/intel64/libmkl_mc3.so \
        -l /opt/intel/compilers_and_libraries/linux/mkl/lib/intel64/libmkl_vml_mc3.so \
        -l /opt/intel/compilers_and_libraries/linux/mkl/lib/intel64/libmkl_avx.so \
        -l /opt/intel/compilers_and_libraries/linux/mkl/lib/intel64/libmkl_vml_avx.so \
        -l /opt/intel/compilers_and_libraries/linux/mkl/lib/intel64/libmkl_avx2.so \
        -l /opt/intel/compilers_and_libraries/linux/mkl/lib/intel64/libmkl_vml_avx2.so \
        -l /opt/intel/compilers_and_libraries/linux/mkl/lib/intel64/libmkl_avx512_mic.so \
        -l /opt/intel/compilers_and_libraries/linux/mkl/lib/intel64/libmkl_vml_avx512_mic.so \
        -l /opt/intel/compilers_and_libraries/linux/mkl/lib/intel64/libmkl_avx512.so \
        -l /opt/intel/compilers_and_libraries/linux/mkl/lib/intel64/libmkl_vml_avx512.so \
        -l /opt/intel/compilers_and_libraries/linux/mkl/lib/intel64/libmkl_core.so 

FROM ubuntu:18.04

# This is the only thing necessary really from nvidia/cuda's ubuntu18.04 runtime image
ENV NVIDIA_VISIBLE_DEVICES all

ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

ENV NVIDIA_REQUIRE_CUDA "cuda>=10.1 brand=tesla,driver>=384,driver<385 brand=tesla,driver>=396,driver<397 brand=tesla,driver>=410,driver<411"

COPY --from=builder /root/SIRIUS.AppDir /root/SIRIUS.AppDir

# Make it easy to call our binaries.
ENV PATH="/root/SIRIUS.AppDir/usr/bin:$PATH"

WORKDIR /root/SIRIUS.AppDir/usr/bin
