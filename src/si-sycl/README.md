# An Evaluation of Large Set Intersection Techniques on a GPU

## Abstract
We focus on large set intersection, which is a pivotal operation in information retrieval, graph analytics and database systems.  We aim to experimentally detect under which conditions, using a graphics processing unit (GPU) is beneficial over CPU techniques and which exact techniques are capable of yielding improvements.
We cover and adapt techniques initially proposed for graph analytics, while we investigate new hybrids for completeness.
We experiment when both a single pair of two large sets are processed and all pairs in a dataset are examined.


## Compile (Intel DevCloud)

```
mkdir release && cd release
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=dpcpp ..
make -j
```
