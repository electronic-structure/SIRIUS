A simple container with CUDA and MKL backends for SIRIUS.

To test the environment, try to run the following command inside the container:

```
spack build-env $SIRIUS_SPEC -- mpif90 test.f90 -I$SIRIUS_ROOT/include/sirius -L$SIRIUS_ROOT/lib -lsirius && ./a.out
```

The output should contain the line
```
SIRIUS 7.2.5, git hash: d63f6defc093b64a828ec18fd88079fa76700ec6
```
