A simple container with CUDA and MKL backends for SIRIUS.

To build and run, use the following:

```
$ docker build -t sirius .
$ docker run -it --rm sirius
# mpifort test.f90 -Iview/include/sirius -Wl,-rpath=view/lib -Lview/lib -lsirius
# ./a.out
```

The output should contain the line
```
SIRIUS x.y.z, git hash: ...
```

To make devices visible to the container, use the `--gpus` flag:

```
docker run --gpus 'all,"capabilities=compute,utility"' ...
```
