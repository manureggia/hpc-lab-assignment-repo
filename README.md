# First Assignment

## The algorithm 

In linear algebra, the ADI (Alternating Direction Implicit) algorithm is an iterative method used to solve Sylvester matrix equations. It's particularly useful for solving large, sparse linear systems that arise from discretizing partial differential equations (PDEs), such as the heat equation or the wave equation.

## How to run
On Host:
```bash
source setup_clang.sh
make EXT_CFLAGS="-DPOLYBENCH_TIME -DOPT_TYPE=HOST" clean all run 
```
On Device (with offloading):
```bash
source setup_clang.sh
make EXT_CFLAGS="-DPOLYBENCH_TIME -DOPT_TYPE=DEVICE" clean all run
```

## How to profile
with pref:
```bash
module load perf/1.0
perf stat -e cycles,instructions,cache-references,cache-misses ./adi_acc 
```


# Second Assignment

## How to run

```bash
make EXERCISE=adi-v4.cu clean all run
```

For profiling:

```bash
make EXERCISE=adi-v4.cu clean all profile
```

