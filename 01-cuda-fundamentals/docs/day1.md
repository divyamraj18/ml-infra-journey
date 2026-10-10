# GPU Architecture and Execution Mechanics

## CPU vs GPU

- **The CPU** — *Latency optimized*. It is built with a small number of powerful cores and is designed to handle complex, sequential, and unpredictable tasks with a large cache hierarchy.
- **The GPU** — *Throughput optimized*. Designed for massive throughput; built with thousands of smaller, simpler cores to execute the exact same mathematical instruction across a massive dataset in parallel.

## Hardware

1. **Streaming Multiprocessors (SMs) = Factory floor**  
   What? A GPU is split into several independent sections called SMs.

2. **CUDA cores = Factory workers**  
   The basic math engine sitting inside these SMs that performs simple calculations like multiplication or addition.

3. **Register = The worker's toolkit**  
   Ultra-fast private storage dedicated directly to each individual execution thread.

4. **Shared Memory = Workbench on each floor**  
   A small, fast pool of memory shared by all threads running inside the same thread block on an SM.

5. **Global Memory (VRAM/HBM) = Massive warehouse outside the factory**  
   The large off-chip memory where massive datasets and model weights are stored.

## Software Abstraction

1. **Grid** — The whole work itself (e.g. processing 10 million elements).

2. **Thread Blocks** — The grid is broken down into multiple smaller chunks called blocks. Each block is assigned to a single SM. All threads inside a block can share data using shared memory.

3. **Threads** — Inside each block there are individual threads (the actual CUDA cores). Every thread handles a specific piece of data.

## How Software and Hardware Communicate

1. **Warp** — The hardware groups your threads into fixed packs of 32 threads called a warp.

2. **SIMT (Single Instruction, Multiple Threads)** — The warp scheduler tells all the 32 threads in the warp to execute the same instruction at the exact same clock cycle, but each thread works on its own data index (using its private registers).

> **The Trap (Warp Divergence):** If your code has a condition check like an if-else where thread 1 wants to go left and thread 2 wants to go right, the hardware can't do both at once. It has to pause one group while the other runs, cutting your speed in half and leaving the GPU underutilized.
