# CUDA Parallel Programming 🚀

This repository contains basic CUDA programs implemented to understand 
GPU parallelism, thread mapping, and memory management.

The goal of this repo is to build a strong foundation in GPU computing 
before moving to advanced topics like AI/ML acceleration.

---

## 🔹 Concepts Covered

- Thread & Block indexing
- Grid configuration
- Kernel launching
- Device vs Host memory
- Synchronization (cudaDeviceSynchronize)
- Parallel matrix operations

---

## 🔹 Implemented Programs

- ✅ Vector Addition
- ✅ Matrix Addition
- ✅ Matrix Multiplication (Naive)
- ✅ Count Equal Elements
- ✅ Image Inversion Kernel

---

## 🔹 Sample Kernel Structure

```cpp
__global__ void kernel(...) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
}
