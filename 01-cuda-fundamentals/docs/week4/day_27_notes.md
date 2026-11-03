# 📅 Day 27: Custom CUDA Kernels for LLMs & FlashAttention-Style Tiling

Aaj ke day ka main target yeh samajhna hai ki Large Language Models ko optimize karne ke liye custom CUDA kernels kaise likhe jate hain, GPU execution model (Threads, Thread Blocks, Warps, Shared Memory) kaise work karta hai, aur **FlashAttention** implementation ka kernel-level tiling logic code level par kaise structure hota hai.

---

## 1. GPU Execution Hierarchy & Memory Architecture

Custom CUDA kernel likhne se pehle GPU ke hierarchy ko samajhna zaroori hai:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            GPU Execution Hierarchy                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  Grid                                                                       │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Thread Block (Shared Memory / SRAM - Fast, Low Capacity ~100KB/SM)     │  │
│  │ ┌───────────────────────────────────────────────────────────────────┐ │  │
│  │ │ Warp (32 SIMT Threads executing instructions in lockstep)          │ │  │
│  │ │ [T0] [T1] [T2] ... [T31]                                          │ │  │
│  │ └───────────────────────────────────────────────────────────────────┘ │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ▲
                                      │ High Bandwidth Memory (HBM)
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ High Bandwidth Memory (HBM / VRAM - Slower, High Capacity e.g. 80GB A100)    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Memory Speed Spectrum:
* **Registers:** Direct ALU access (0 cycles latency).
* **Shared Memory (SRAM):** On-chip memory per Thread Block (~20-30 cycles latency, ~19 TB/s bandwidth).
* **Global Memory (HBM):** Off-chip memory shared by all blocks (~400-800 cycles latency, ~2 TB/s bandwidth).

---

## 2. Why Custom Kernels for LLMs? (Kernel Fusion)

PyTorch me jab aap standard operations karte hain (e.g., `GeLU(LayerNorm(X))`):

1. **Unfused Execution:**
   * Step 1: Read $X$ from HBM $\rightarrow$ Compute LayerNorm $\rightarrow$ Write result to HBM.
   * Step 2: Read intermediate from HBM $\rightarrow$ Compute GeLU $\rightarrow$ Write output to HBM.
   * **Bottleneck:** High HBM Memory Read/Write Overhead (Memory-bound).

2. **Fused CUDA Kernel Execution:**
   * Read $X$ from HBM **once** into Shared Memory/Registers.
   * Compute LayerNorm + GeLU **in-place** inside SRAM.
   * Write final output to HBM **once**.

---

## 3. FlashAttention Tiling Algorithm Mechanics

Standard Attention memory $O(N^2)$ ko evade karne ke liye FlashAttention $Q, K, V$ matrices ko small blocks (tiles) me baant kar SRAM me process karta hai:

```
Global Memory (HBM)                          Thread Block SRAM
┌──────────────────┐                         ┌──────────────┐
│  Q Matrix (N x d)│ ──── Block Loading ────►│ Q_tile (Br x d)│
└──────────────────┘                         └──────────────┘
┌──────────────────┐                         ┌──────────────┐
│  K Matrix (N x d)│ ──── Block Loading ────►│ K_tile (Bc x d)│
└──────────────────┘                         └──────────────┘
                                                    │
                                                    ▼
                                           S_tile = Q_tile x K_tile^T
                                           (Online Softmax Computation)
```

### Online Softmax Trick (Without Full $N \times N$ Matrix)
Standard Softmax ke liye pure row ki values pata honi chahiye:
$$\text{Softmax}(x_i) = \frac{e^{x_i - m}}{\sum e^{x_j - m}} \quad \text{where } m = \max(x)$$

FlashAttention block-by-block update karta hai using running max $m_{new}$ and running sum $l_{new}$:
$$m_{new} = \max(m_{old}, m_{block})$$
$$l_{new} = l_{old} \cdot e^{m_{old} - m_{new}} + l_{block} \cdot e^{m_{block} - m_{new}}$$

---

## 4. Simplified CUDA C++ Tiled Kernel Concept

Below is a conceptual CUDA C++ snippet showing how Shared Memory (`__shared__`) tiling is set up for LLM operations:

```cpp
#include <cuda_runtime.h>
#include <torch/extension.h>

#define TILE_SIZE 32

// CUDA Kernel executing on GPU
__global__ void tiled_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N) 
{
    // Allocate Shared Memory on SRAM for Thread Block
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float acc = 0.0f;

    // Loop over tiles along the inner dimension
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Collaborative loading of HBM data into SRAM
        if (row < N && (t * TILE_SIZE + threadIdx.x) < N)
            s_A[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        else
            s_A[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && (t * TILE_SIZE + threadIdx.y) < N)
            s_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            s_B[threadIdx.y][threadIdx.x] = 0.0f;

        // Synchronize all threads in block until SRAM is loaded
        __syncthreads();

        // Multiply tile from SRAM (Zero HBM latency during compute)
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            acc += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }

        // Sync before loading next tile
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = acc;
    }
}
```

---

## 5. Summary Checkpoints for Day 27

1. **SRAM vs HBM:** High Bandwidth Memory (HBM) bandwidth bottleneck se bachne ke liye data ko SRAM (`__shared__`) me tile-by-tile laya jata hai.
2. **Kernel Fusion:** Multiple operations (like LayerNorm + Activation) ko single CUDA kernel me merge karke HBM read/write reduce kiya jata hai.
3. **Online Softmax:** FlashAttention intermediate $N \times N$ attention matrix build kiye bina, block-wise running statistics ($m, l$) se SRAM inside exact attention compute karta hai.