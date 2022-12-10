FROM sirius_base as builder

RUN spack -e sirius-env install

RUN apt-get install -y xsltproc

COPY exciting /exciting
COPY build_with_sirius_mkl.x /build_with_sirius_mkl.x
COPY make.inc /exciting/build/make.inc
RUN /build_with_sirius_mkl.x

# Bare OS image to run the installed executables
FROM sirius_base
COPY --from=builder /exciting/build/mpiandsmp/exciting /exciting
COPY --from=builder /opt/local /opt/local

# For the sarus mpi hook
COPY --from=builder /etc/ld.so.conf.d/mpich.conf /etc/ld.so.conf.d/mpich.conf
RUN ldconfig

