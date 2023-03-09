ARG BASE_IMAGE
FROM $BASE_IMAGE

ARG SPEC
ARG CI_PROJECT_DIR

RUN spack spec $SPEC

COPY . /sirius-src

RUN cd /sirius-src && ls
