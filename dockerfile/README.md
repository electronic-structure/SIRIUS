A simple container with CUDA and MKL backends for SIRIUS.

To build the particular configuration of SIRIUS, run:

```
$ docker build -t sirius .
```

To build your application against SIRIUS, mount or copy your sources, and use
the Spack environment view for the headers and libraries. For example:

```
$ docker run -v $PWD/src:/root/src -it --rm sirius
~ # mpifort src/test.f90 -Iview/include/sirius -Wl,-rpath=view/lib -Lview/lib -lsirius
~ # ./a.out
```

The output should contain the line
```
SIRIUS x.y.z, git hash: ...
```

To make devices visible to the container, use the `--gpus` flag:

```
docker run --gpus 'all,"capabilities=compute,utility"' ...
```
