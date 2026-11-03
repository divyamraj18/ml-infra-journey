# 📅 Day 15: Data Parallelism (DP) Mechanics & GPU Profiling (Nsight Systems / Compute)

Today’s focus is understanding how **Data Parallelism (DP)** operates under the hood, comparing **Synchronous vs. Asynchronous SGD**, evaluating **Parameter Server vs. AllReduce** topologies, diagnosing **Compute-bound vs. Memory-bound** CUDA kernels, and using NVIDIA **Nsight Systems (`nsys`)** and **Nsight Compute (`ncu`)** for performance profiling.

---

## 1. Data Parallelism (DP) Architecture

Data Parallelism is the most common distributed strategy when a model fits within a single GPU’s memory, but training throughput needs to scale by distributing batch processing across $N$ GPUs.

### High-Level Data Parallel Execution Flow

```
                                  [ Global Batch Size: B ]
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
            [ Micro-Batch: B/2 ]                       [ Micro-Batch: B/2 ]
                   GPU 0                                      GPU 1
         ┌───────────────────────┐                  ┌───────────────────────┐
         │  Model Copy (Rank 0)  │                  │  Model Copy (Rank 1)  │
         └───────────┬───────────┘                  └───────────┬───────────┘
                     │ (Forward Pass)                           │ (Forward Pass)
                     ▼                                          ▼
                [ Loss L0 ]                                [ Loss L1 ]
                     │ (Backward Pass)                          │ (Backward Pass)
                     ▼                                          ▼
            [ Local Gradients G0 ]                     [ Local Gradients G1 ]
                     │                                          │
                     └───────────────────┬──────────────────────┘
                                         ▼
                             [ NCCL AllReduce (Sum) ]
                                         │
                         ┌───────────────┴───────────────┐
                         ▼                               ▼
                 [ Synced Gradients ]            [ Synced Gradients ]
                      (G0 + G1)                       (G0 + G1)
                         │                               │
                         ▼                               ▼
               [ Optimizer Step (W0) ]         [ Optimizer Step (W1) ]
```

### Steps in Data Parallel Training:
1. **Model Replication**: A complete copy of the model parameters $W$ resides on every GPU (Rank).
2. **Data Sharding**: The global mini-batch of size $B$ is split equally across $N$ GPUs into micro-batches of size $B/N$.
3. **Independent Forward & Backward Passes**: Each GPU processes its local micro-batch to compute local loss $L_i$ and local gradients $\nabla L_i$.
4. **Gradient Synchronization**: All GPUs perform an **AllReduce** operation to sum gradients $\sum \nabla L_i$ across all ranks.
5. **Optimizer Step**: Each GPU updates its local model weights independently using the synchronized gradients. Because all GPUs start with identical initial weights $W_0$ and receive identical reduced gradients, the weights remain perfectly synchronized ($W_{t+1}$) across all ranks.

---

## 2. Synchronous vs. Asynchronous SGD

| Metric / Aspect | Synchronous SGD | Asynchronous SGD |
| :--- | :--- | :--- |
| **Execution Flow** | All GPUs wait for every rank to complete backward pass before updating weights. | GPUs update global model weights as soon as their own backward pass completes, without waiting for others. |
| **Gradient Accuracy** | **Exact**. Mathematically equivalent to running a single GPU with global batch size $B$. | **Stale Gradients**. Faster workers update weights $W_t \rightarrow W_{t+k}$, so slower workers compute gradients on outdated weights. |
| **Convergence** | Highly stable; predictable loss decay curves. | Can suffer from severe instability, gradient explosion, or poor final accuracy. |
| **Straggler Sensitivity**| High. The slowest GPU (straggler) bottlenecking the entire cluster iteration time. | Low. Fast GPUs process more steps without waiting for slow GPUs. |
| **Modern LLM Standard**| **100% Dominant**. All modern LLM frameworks (PyTorch DDP, DeepSpeed, Megatron) use **Synchronous SGD**. | Seldom used for LLM training due to loss degradation at scale. |

---

## 3. Parameter Server Architecture vs. AllReduce Architecture

### A. Parameter Server (PS) Architecture (Legacy / CPU-based)
Used historically in large-scale ML systems (e.g., TF v1, Parameter Server pattern).

```
         +-------------------------------------------------------+
         |               Parameter Server Node(s)                |
         |         Holds Master Weights & Optimizer States       |
         +───────────────────────────┬───────────────────────────+
                                     │
           ┌─────────────────────────┼─────────────────────────┐
           │ (Push Gradients)        │                         │ (Pull Updated Params)
           ▼                         ▼                         ▼
   +---------------+         +---------------+         +---------------+
   | Worker GPU 0  |         | Worker GPU 1  |         | Worker GPU 2  |
   +---------------+         +---------------+         +---------------+
```

* **Bottleneck**: Network bandwidth on the Parameter Server node scales linearly with workers. When $N$ workers push gradients to a single PS node, the network interface card (NIC) on the PS node chokes (Incast Problem).

### B. AllReduce Architecture (Modern GPU Standard)
Replaces centralized Parameter Servers with decentralized peer-to-peer ring/tree communication across GPUs.

```
          [ GPU 0 ] <====== (NVLink / RDMA) ======> [ GPU 1 ]
             ||                                        ||
          (NVLink)                                  (NVLink)
             ||                                        ||
          [ GPU 3 ] <====== (NVLink / RDMA) ======> [ GPU 2 ]
```

* **Bandwidth Efficiency**: Ring-AllReduce transfers data in $2 \times \frac{N-1}{N} \times S$ total payload per GPU. Communication scale stays invariant with GPU count when interconnect bandwidth is homogeneous.

---

## 4. Compute-Bound vs. Memory-Bound CUDA Kernels

Performance optimization starts by identifying whether a CUDA kernel is constrained by compute throughput (FLOP/s) or memory bandwidth (GB/s).

### Arithmetic Intensity
The metric used to classify a kernel is **Arithmetic Intensity**:

$$\text{Arithmetic Intensity} = \frac{\text{Floating Point Operations (FLOPs)}}{\text{Memory Accesses (Bytes Transferred)}}$$

```
                           Operational Intensity (FLOPs / Byte)
       Low (Memory-Bound)                                      High (Compute-Bound)
◄──────────────────────────────┼──────────────────────────────────────────►
  • Element-wise Ops (Add, Mul)  • Ridge Point                     • Matrix Multiplications (GEMM)
  • LayerNorm, RMSNorm           (Hardware Peak FLOPs /            • Convolution Operations
  • Softmax                      Hardware Peak Bandwidth)          • Attention Projections (QKV)
  • Activations (GELU, ReLU)
```

### A. Memory-Bound Kernels
* **Characteristic**: Execution time is dominated by waiting for data to travel between HBM (Global Memory) and SRAM (Register File / L1 Cache).
* **Examples**: Element-wise addition, Activation functions, LayerNorm, Dropout, Softmax.
* **Optimization Techniques**:
  1. **Kernel Fusion**: Combine multiple element-wise operations into a single CUDA kernel to retain data in registers without writing intermediate values back to Global Memory (e.g., PyTorch `torch.jit.script`, FlashAttention).
  2. **Coalesced Reads**: Ensure adjacent threads read adjacent memory addresses.

### B. Compute-Bound Kernels
* **Characteristic**: Execution time is dominated by the mathematical throughput of Arithmetic Logic Units (ALUs) / Tensor Cores. Memory data transfer is fully hidden behind calculations.
* **Examples**: Dense Matrix Multiplication (GEMM), Convolutions.
* **Optimization Techniques**:
  1. Maximize **Tensor Core** utilization (e.g., using `FP16` / `BF16` / `FP8` precision with dimensions aligned to multiples of 8 or 16).
  2. Increase **Shared Memory Tiling** to maximize data reuse per load.

---

## 5. Profiling Tools: NVIDIA Nsight Systems (`nsys`) vs. Nsight Compute (`ncu`)

NVIDIA provides two distinct profiling tools targeted at different levels of abstraction:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          NVIDIA Profiling Suite                             │
├──────────────────────────────────────┬──────────────────────────────────────┤
│    Nsight Systems (nsys)             │     Nsight Compute (ncu)             │
├──────────────────────────────────────┼──────────────────────────────────────┤
│ • System-wide application trace.     │ • Single CUDA kernel deep dive.      │
│ • Identifies CPU-GPU gaps, CUDA API  │ • Measures Warp Occupancy, Roofline  │
│   launch overhead, & NCCL waits.     │   Model, & Memory Stall reasons.     │
│ • Low overhead trace (< 5%).         │ • High overhead analysis (10x-100x). │
└──────────────────────────────────────┴──────────────────────────────────────┘
```

### A. Nsight Systems (`nsys`) Workflow
Use `nsys` first to get a birds-eye view of your distributed execution timeline.

```bash
# Profile PyTorch script with Nsight Systems
nsys profile \
  --trace=cuda,nvtx,osrt,cublas,nccl \
  --output=ddp_trace_report \
  --export=sqlite \
  python train_ddp.py
```

#### Key Things to Inspect in `nsys` Visual Timeline:
1. **CPU/GPU Gap**: Gaps on the CUDA stream where the GPU sits idle while waiting for the CPU to launch the next kernel (indicates CPU-bound Python dispatch bottlenecks).
2. **NCCL Overlap**: Check if NCCL AllReduce streams overlap with backward pass compute streams. If AllReduce starts only *after* backward pass completes, backward-communication overlap is unoptimized.

```
Expected Overlapped Timeline:
GPU Stream 0 (Compute):  [ Layer 3 Backprop ] [ Layer 2 Backprop ] [ Layer 1 Backprop ]
GPU Stream 1 (NCCL):           [ AllReduce L3 ]      [ AllReduce L2 ]      [ AllReduce L1 ]
```

---

### B. Nsight Compute (`ncu`) Workflow
Use `ncu` after identifying a specific slow CUDA kernel in `nsys` (e.g., custom Softmax or GEMM kernel).

```bash
# Profile a specific CUDA kernel using Nsight Compute
ncu \
  --kernel-name tiledMatrixMul \
  --metrics sm__throughput.of_peak_in_percent,gpu__time_duration.sum,dram__throughput.of_peak_in_percent \
  ./matrix_mul_bin
```

#### Core Metrics Analyzed in `ncu`:
1. **Warp Occupancy** (`sm__warps_active.avg.pct_of_peak`): Percentage of active warps relative to the theoretical maximum per Streaming Multiprocessor (SM). Low occupancy indicates high register usage per thread or excessive shared memory allocation per block.
2. **Warp Stall Reasons**:
   * `Stall Memory Throttle`: Threads waiting for global memory requests to clear.
   * `Stall Execution Dependency`: Threads waiting for input operands from prior instructions.
   * `Stall Sync`: Threads blocked at a `__syncthreads()` barrier.

---

## 6. Summary Checklist for Day 15

1. **Data Parallelism (DP)** replicates model parameters across GPUs while sharding the dataset batch.
2. **Synchronous SGD + Ring-AllReduce** is the standard for LLMs, avoiding the stale-gradient issues of Asynchronous SGD and the network bottlenecks of Parameter Servers.
3. **Kernel Classification**: Classify operations via Arithmetic Intensity ($\text{FLOPs/Byte}$) to determine whether to fuse kernels (Memory-Bound) or optimize Tensor Cores (Compute-Bound).
4. **Profiling Pipeline**: Use **`nsys`** to optimize system-level timeline/communication overlap, then zoom in with **`ncu`** to eliminate hardware stalls in individual kernels.