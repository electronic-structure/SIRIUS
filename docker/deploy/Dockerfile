ARG BUILD_ENV_IMAGE

ARG BUILD_ENV_TAG

FROM $BUILD_ENV_IMAGE:$BUILD_ENV_TAG as builder

# Add bundle tooling
RUN wget -q https://github.com/haampie/bundler/releases/download/v0.1.3/bundler_x86_64.tar.gz && \
    tar -xzf bundler_x86_64.tar.gz && \
    rm bundler_x86_64.tar.gz

# Build SIRIUS
COPY . /build-sirius

RUN mkdir /build-sirius/build && cd /build-sirius/build && \
    CC=mpicc CXX=mpicxx FC=mpif90 FCCPP=cpp python3 ../prerequisite.py /usr/local xc spg && \
    LIBXCROOT=/usr/local LIBSPGROOT=/usr/local cmake .. \
      -DSpFFT_DIR=/usr/local/lib/cmake/SpFFT \
      -DUSE_SCALAPACK=1 \
      -DUSE_MKL=1 \
      -DBUILD_TESTS=1 \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr \
      -DUSE_CUDA=On \
      -DGPU_MODEL='P100' && \
      make -j$(nproc) DESTDIR=/root/SIRIUS-build install && \
      rm -rf /build-sirius

# Run the bundler and make sure to include MKL's libs
RUN /root/bundler/bundler -d SIRIUS.bundle/ \
        -e SIRIUS-build/usr/bin/sirius.scf \
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

COPY --from=builder /root/SIRIUS.bundle /root/SIRIUS.bundle

# Make it easy to call our binaries.
ENV PATH="/root/SIRIUS.bundle/usr/bin:$PATH"

# Make sarus happy in locating libmpi.so (we have to make this path persistent because nvidia also calls ldconfig prior to running)
# ENV LD_LIBRARY_PATH="/root/SIRIUS.bundle/usr/lib:$LD_LIBRARY_PATH"
RUN echo "/root/SIRIUS.bundle/usr/lib/" > /etc/ld.so.conf.d/sirius.conf && ldconfig

WORKDIR /root/SIRIUS.bundle/usr/bin
