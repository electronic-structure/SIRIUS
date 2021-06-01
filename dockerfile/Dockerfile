FROM stabbles/sirius-cuda-11

WORKDIR /root

COPY spack.yaml test.f90 /root/

# Concretize the spack environment and install missing dependencies and
# sirius itself. Make sure we do not rebuild cuda by marking it external.
RUN spack -e . external find --not-buildable cuda && \
    spack -e . install -v
