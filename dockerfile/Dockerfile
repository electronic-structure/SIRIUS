FROM stabbles/sirius-cuda-11

RUN spack install sirius@develop target=x86_64 +fortran +scalapack ^mpich ^intel-mkl

RUN echo "export SIRIUS_ROOT=$(spack location -i sirius@develop)" >> /etc/profile.d/sirius.sh
RUN echo "export SIRIUS_SPEC='sirius@develop target=x86_64 +fortran +scalapack ^mpich ^intel-mkl'" >>  /etc/profile.d/sirius.sh

WORKDIR /root

COPY test.f90 /root

ENTRYPOINT ["bash", "-l"]

