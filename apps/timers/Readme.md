## Flame graph

`timers.json` can be turned into a flame graph with the help of [FlameGraph](https://github.com/brendangregg/FlameGraph).

Run `sirius.scf` or SIRIUS-enabled `pw.x` with `export SIRIUS_PRINT_TIMINGS=1`.


1. Transform `timers.json` into a FlameGraph compatible format:
```bash
./collapse.py timers.json > timers.out
```

2. Use `flamegraph.pl` from [FlameGraph](https://github.com/brendangregg/FlameGraph):
```bash
./FlameGraph/flamegraph.pl timers.out > timers.svg
```
