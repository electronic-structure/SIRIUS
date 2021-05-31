FROM stabbles/sirius-cuda-11

ENV SIRIUS_SPEC sirius@develop target=x86_64 +fortran +scalapack +cuda cuda_arch=60 ^mpich ^intel-mkl

RUN spack install $SIRIUS_SPEC

RUN echo "export SIRIUS_ROOT=$(spack location -i sirius@develop)" >> /etc/profile.d/sirius.sh

WORKDIR /root

COPY test.f90 /root

ENTRYPOINT ["bash", "-l"]
