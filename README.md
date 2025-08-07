# ml_from_scratch

A minimalist machine learning library implemented from scratch in **NumPy** and **C++** (independently) for conceptual clarity and performance benchmarking.

---

##Build Instructions (C++)

To compile the C++ backend with full CPU optimizations:

```bash
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
