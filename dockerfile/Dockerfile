FROM stabbles/sirius-cuda-11

WORKDIR /root

COPY spack.yaml /root/spack.yaml

# Concretize the spack environment and install missing dependencies and
# sirius itself. Make sure we do not rebuild cuda by marking it external.
RUN spack -e . external find --not-buildable cuda && \
    spack -e . install -v

RUN spack env activate --sh -d . >> /etc/profile.d/spack_environment.sh

ENTRYPOINT ["/bin/bash", "--rcfile", "/etc/profile", "-l"]
